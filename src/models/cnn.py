"""CNN configurable: cada decisión de diseño es un parámetro.

Las arquitecturas concretas (cnn_v1, cnn_v2...) NO se programan aquí: se
declaran en `src/models/versiones.py` como una lista de valores para estos
parámetros. Así cada versión es reproducible y comparable con las demás.

Estructura:

    entrada (B, 3, 256, 256)       PRE, EARLY, LATE (instantes, no colores)
       │
       ▼
    [REDUCCIÓN] si reduccion = 2: media de cada cuadrado de 2×2 píxeles → (B, 3, 128, 128).
       La red trabaja a la mitad de resolución, pero recibe lo mismo de siempre (D35)
       │
       ▼
    [NORMALIZACIÓN] si normalizacion = "pre": las tres fases se dividen entre la intensidad
       media del PRE en el tejido de ESE corte → intensidad relativa a la propia paciente,
       sin la escala de cada escáner (D28)
       │
       ▼
    [REALCE] si entrada = "fases+realce": se añaden dos canales calculados DENTRO
       de la red, EARLY − PRE (realce) y LATE − EARLY (lavado) → (B, 5, 256, 256)
       si entrada = "fases+cinetica": se añaden los mapas cinéticos de la radiología,
       realce relativo al PRE y SER (lavado), también calculados dentro (D35)
       │
       ▼
    EXTRACTOR: un bloque por elemento de `filtros`
       cada bloque = [conv kxk -> (BatchNorm) -> ReLU] x convs_por_bloque -> pooling 2x2
       (el pooling divide el lado entre 2: 256 -> 128 -> 64 -> ...)
       │
       ▼
    RESUMEN: "global" (media de cada canal), "flatten" (todos los valores), "atencion"
       (media PONDERADA de las posiciones del mapa, con pesos aprendidos: la red decide
       dónde mirar; Ilse et al., 2018) (D28) o "media+max" (media y máximo de cada canal, D34)
       │
       ▼
    CABEZA: [Linear -> ReLU] (si hay capa oculta) -> [Dropout] -> Linear -> 1 logit

La reducción, el realce, los mapas cinéticos y la normalización se calculan dentro de la red a propósito: así la app y
el entrenamiento reciben exactamente lo mismo (las 3 fases en [0, 1]) y no puede haber
una normalización distinta entre los dos.
"""

from __future__ import annotations

import torch
from torch import nn

LADO_ENTRADA = 256
CANALES_ENTRADA = 3            # lo que RECIBE siempre la red: PRE, EARLY, LATE
CANALES_POR_ENTRADA = {"fases": 3, "fases+realce": 5, "fases+cinetica": 5}
NORMALIZACIONES = (None, "pre")
REDUCCIONES = (1, 2, 4)        # 1 = la resolución de siempre; 2 = 128×128 dentro de la red (D35)
RESUMENES = ("global", "flatten", "atencion", "media+max")
UMBRAL_TEJIDO = 0.1            # PRE > 0,1 es tejido (lo demás, aire); igual que la auditoría
REFERENCIA_MINIMA = 0.1        # si un corte casi no tiene tejido, no se divide entre ~0
OCULTA_ATENCION = 32           # tamaño de la capa interna del pooling por atención
# Mapas cinéticos (D35). Convenciones de la radiología de mama (I-SPY: volumen funcional del tumor
# con realce ≥ 70 %; curva persistente si SER < 0,9, meseta 0,9–1,1, lavado si SER > 1,1).
PE_UMBRAL = 0.7                # realce relativo a partir del cual un píxel «realza»
PE_MAX = 3.0                   # se recorta para que un PRE casi negro no dé valores enormes
SER_MAX = 2.0
SER_DENOMINADOR_MINIMO = 0.02


def mapas_cineticos(x: torch.Tensor) -> torch.Tensor:
    """Realce relativo y SER por píxel: (B, 3, H, W) → (B, 2, H, W), los dos en [0, 1].

    - Realce relativo (PE) = (EARLY − PRE) / PRE: cuánto sube cada punto respecto a sí mismo. Solo en
      tejido (PRE > 0,1); el aire vale 0. Se recorta a [0, 3] y se divide entre 3.
    - SER = (EARLY − PRE) / (LATE − PRE): > 1 si al final baja (lavado, típico de tumores agresivos),
      < 1 si sigue subiendo. Solo donde el píxel realza (PE ≥ 0,7); en el resto, 0. Se recorta a
      [0, 2] y se divide entre 2.
    Son divisiones, que una red de convoluciones y ReLU no sabe hacer por sí sola: por eso se le dan.
    """
    pre, early, late = x[:, 0:1], x[:, 1:2], x[:, 2:3]
    tejido = (pre > UMBRAL_TEJIDO).to(x.dtype)
    pe = ((early - pre) / pre.clamp(min=UMBRAL_TEJIDO)).clamp(0.0, PE_MAX) * tejido
    realza = (pe >= PE_UMBRAL).to(x.dtype)
    ser = ((early - pre) / (late - pre).clamp(min=SER_DENOMINADOR_MINIMO)).clamp(0.0, SER_MAX) * realza
    return torch.cat([pe / PE_MAX, ser / SER_MAX], dim=1)


def normalizar_por_pre(x: torch.Tensor) -> torch.Tensor:
    """Divide las 3 fases de cada corte entre la intensidad media del PRE en su tejido.

    Tras esto el tejido en PRE vale ~1 y EARLY ~1,4 significa "realza un 40 %": es una medida
    relativa a la propia paciente, que no depende de la escala de cada escáner u hospital.
    Las tres fases se dividen entre el MISMO número, así que el realce relativo se conserva.
    """
    pre = x[:, 0:1]
    tejido = (pre > UMBRAL_TEJIDO).to(x.dtype)
    referencia = (pre * tejido).sum(dim=(2, 3), keepdim=True) / tejido.sum(dim=(2, 3), keepdim=True).clamp(min=1)
    return x / referencia.clamp(min=REFERENCIA_MINIMA)


class PoolingAtencion(nn.Module):
    """Media ponderada de las posiciones del mapa final, con pesos que aprende la red.

    Ilse, Tomczak y Welling (2018): para cada posición k, con vector de canales h_k,
        a_k = softmax_k( w · tanh(V · h_k) )        resumen = Σ_k a_k · h_k
    Los pesos suman 1. La media global da a todas las posiciones el mismo peso (1/256 en un
    mapa de 16×16) y diluye un tumor pequeño entre el fondo; aquí la red puede dar más peso
    a donde está lo que importa. Los últimos pesos se guardan para dibujarlos (mapa de
    atención): enseñan dónde ha mirado la red.
    """

    def __init__(self, canales: int, oculta: int = OCULTA_ATENCION) -> None:
        super().__init__()
        self.V = nn.Linear(canales, oculta)
        self.w = nn.Linear(oculta, 1)
        self.ultimos_pesos: torch.Tensor | None = None

    def forward(self, mapas: torch.Tensor) -> torch.Tensor:
        b, _, alto, ancho = mapas.shape
        h = mapas.flatten(2).transpose(1, 2)                     # (B, posiciones, canales)
        pesos = torch.softmax(self.w(torch.tanh(self.V(h))), dim=1)  # (B, posiciones, 1)
        self.ultimos_pesos = pesos.detach().view(b, alto, ancho)
        return (pesos * h).sum(dim=1)                            # (B, canales)


class MediaYMaximo(nn.Module):
    """Une la media global y el máximo global de cada canal: (B, C, H, W) → (B, 2C).

    La media resume todo el corte (contexto, tejido de alrededor); el máximo se queda con la
    respuesta más fuerte de cada filtro, que en un tumor pequeño suele ser el propio tumor y que
    la media diluye entre el fondo (D34). Sin parámetros: solo dobla el tamaño del vector.
    """

    def forward(self, mapas: torch.Tensor) -> torch.Tensor:
        return torch.cat([mapas.mean(dim=(2, 3)), mapas.amax(dim=(2, 3))], dim=1)


class CNN(nn.Module):
    def __init__(
        self,
        filtros: tuple[int, ...],
        kernel: int = 3,
        convs_por_bloque: int = 1,
        pooling: str = "max",
        batchnorm: bool = True,
        resumen: str = "global",
        cabeza_oculta: int | None = None,
        dropout: float = 0.0,
        entrada: str = "fases",
        normalizacion: str | None = None,
        reduccion: int = 1,
    ) -> None:
        """
        filtros           canales de salida de cada bloque; su longitud = número de bloques.
        kernel            tamaño del kernel (impar). El padding se calcula para conservar el
                          tamaño del mapa: solo el pooling lo reduce.
        convs_por_bloque  convoluciones seguidas antes de cada pooling.
        pooling           "max" (¿aparece el patrón?) o "avg" (media de la zona).
        batchnorm         BatchNorm2d después de cada convolución.
        resumen           "global" (AdaptiveAvgPool: un valor por canal), "flatten", "atencion"
                          (media ponderada con pesos aprendidos, PoolingAtencion) o "media+max"
                          (media y máximo de cada canal, juntos: MediaYMaximo).
        cabeza_oculta     None = Linear directa a 1 salida; un número = capa intermedia.
        dropout           probabilidad de dropout en la cabeza (0 = sin dropout).
        entrada           "fases" (PRE, EARLY, LATE), "fases+realce" (además EARLY − PRE y
                          LATE − EARLY) o "fases+cinetica" (además realce relativo y SER,
                          mapas_cineticos); todo calculado dentro de la red.
        normalizacion     None (las fases tal cual, en [0, 1]) o "pre" (divididas entre la
                          intensidad media del PRE en el tejido del corte; normalizar_por_pre).
        reduccion         1 = la red trabaja a 256×256; 2 = a 128×128 (media de 2×2 píxeles al
                          entrar). La entrada sigue siendo (B, 3, 256, 256).
        """
        super().__init__()
        if kernel % 2 == 0:
            raise ValueError("kernel debe ser impar para conservar el tamaño con padding")
        if pooling not in {"max", "avg"}:
            raise ValueError("pooling debe ser 'max' o 'avg'")
        if resumen not in RESUMENES:
            raise ValueError(f"resumen debe ser uno de {RESUMENES}")
        if entrada not in CANALES_POR_ENTRADA:
            raise ValueError(f"entrada debe ser una de {sorted(CANALES_POR_ENTRADA)}")
        if normalizacion not in NORMALIZACIONES:
            raise ValueError(f"normalizacion debe ser una de {NORMALIZACIONES}")
        if reduccion not in REDUCCIONES:
            raise ValueError(f"reduccion debe ser una de {REDUCCIONES}")
        self.entrada = entrada
        self.normalizacion = normalizacion
        self.reduccion = reduccion

        # ---- Extractor ------------------------------------------------------ #
        capas: list[nn.Module] = []
        canales = CANALES_POR_ENTRADA[entrada]
        for salida in filtros:
            for _ in range(convs_por_bloque):
                # Con BatchNorm detrás, el sesgo de la convolución sobra (BN ya lo aporta).
                capas.append(nn.Conv2d(canales, salida, kernel, padding=kernel // 2, bias=not batchnorm))
                if batchnorm:
                    capas.append(nn.BatchNorm2d(salida))
                capas.append(nn.ReLU(inplace=True))
                canales = salida
            capas.append(nn.MaxPool2d(2) if pooling == "max" else nn.AvgPool2d(2))
        self.extractor = nn.Sequential(*capas)

        # ---- Resumen: de mapas a vector ------------------------------------ #
        lado_final = LADO_ENTRADA // reduccion // 2 ** len(filtros)
        if resumen == "global":
            self.resumen = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten())
            n_vector = canales
        elif resumen == "atencion":
            self.resumen = PoolingAtencion(canales)
            n_vector = canales
        elif resumen == "media+max":
            self.resumen = MediaYMaximo()
            n_vector = 2 * canales
        else:
            self.resumen = nn.Flatten()
            n_vector = canales * lado_final * lado_final

        # ---- Cabeza: de vector a 1 logit (sin sigmoide) --------------------- #
        cabeza: list[nn.Module] = []
        if cabeza_oculta:
            cabeza += [nn.Linear(n_vector, cabeza_oculta), nn.ReLU(inplace=True)]
            n_vector = cabeza_oculta
        if dropout > 0:
            cabeza.append(nn.Dropout(dropout))
        cabeza.append(nn.Linear(n_vector, 1))
        self.cabeza = nn.Sequential(*cabeza)

    def preparar_entrada(self, x: torch.Tensor) -> torch.Tensor:
        """(B, 3, H, W) → lo que ve el primer bloque: reducida y normalizada si toca, y con los
        canales calculados que pida `entrada` (realce o mapas cinéticos)."""
        if self.reduccion > 1:
            x = nn.functional.avg_pool2d(x, self.reduccion)
        if self.normalizacion == "pre":
            x = normalizar_por_pre(x)
        if self.entrada == "fases+realce":
            pre, early, late = x[:, 0:1], x[:, 1:2], x[:, 2:3]
            x = torch.cat([x, early - pre, late - early], dim=1)
        elif self.entrada == "fases+cinetica":
            x = torch.cat([x, mapas_cineticos(x)], dim=1)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cabeza(self.resumen(self.extractor(self.preparar_entrada(x))))
