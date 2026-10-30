"""CNN configurable: cada decisión de diseño es un parámetro.

Las arquitecturas concretas (cnn_v3, cnn_v4...) NO se programan aquí: se
declaran en `src/models/versiones.py` como una lista de valores para estos
parámetros. Así cada versión es reproducible y comparable con las demás.

Estructura:

    entrada (B, 3, 256, 256)       PRE, EARLY, LATE (instantes, no colores)
       │
       ▼
    [NORMALIZACIÓN] si normalizacion = "pre": las tres fases se dividen entre la intensidad
       media del PRE en el tejido de ESE corte → intensidad relativa a la propia paciente,
       sin la escala de cada escáner (D28)
       │
       ▼
    [REALCE] si entrada = "fases+realce": se añaden dos canales calculados DENTRO
       de la red, EARLY − PRE (realce) y LATE − EARLY (lavado) → (B, 5, 256, 256)
       │
       ▼
    EXTRACTOR: un bloque por elemento de `filtros`
       cada bloque = [conv kxk -> (BatchNorm) -> ReLU] x convs_por_bloque -> pooling 2x2
       (el pooling divide el lado entre 2: 256 -> 128 -> 64 -> ...)
       │
       ▼
    RESUMEN: "global" (media de cada canal), "flatten" (todos los valores), "atencion"
       (media PONDERADA de las posiciones del mapa, con pesos aprendidos: la red decide
       dónde mirar; Ilse et al., 2018) (D28)
       │
       ▼
    CABEZA: [Linear -> ReLU] (si hay capa oculta) -> [Dropout] -> Linear -> 1 logit

El realce y la normalización se calculan dentro de la red a propósito: así la app y el
entrenamiento reciben exactamente lo mismo (las 3 fases en [0, 1]) y no puede haber una
normalización distinta entre los dos.
"""

from __future__ import annotations

import torch
from torch import nn

LADO_ENTRADA = 256
CANALES_ENTRADA = 3            # lo que RECIBE siempre la red: PRE, EARLY, LATE
CANALES_POR_ENTRADA = {"fases": 3, "fases+realce": 5}
NORMALIZACIONES = (None, "pre")
RESUMENES = ("global", "flatten", "atencion")
UMBRAL_TEJIDO = 0.1            # PRE > 0,1 es tejido (lo demás, aire); igual que la auditoría
REFERENCIA_MINIMA = 0.1        # si un corte casi no tiene tejido, no se divide entre ~0
OCULTA_ATENCION = 32           # tamaño de la capa interna del pooling por atención


def normalizar_por_pre(x: torch.Tensor) -> torch.Tensor:
    """Divide las 3 fases de cada corte entre la intensidad media del PRE en su tejido.

    Tras esto el tejido en PRE vale ~1 y EARLY ~1,4 significa "realza un 40 %": es una medida
    relativa a la propia paciente, que no depende de la escala de cada escáner.
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
    ) -> None:
        """
        filtros           canales de salida de cada bloque; su longitud = número de bloques.
        kernel            tamaño del kernel (impar). El padding se calcula para conservar el
                          tamaño del mapa: solo el pooling lo reduce.
        convs_por_bloque  convoluciones seguidas antes de cada pooling.
        pooling           "max" (¿aparece el patrón?) o "avg" (media de la zona).
        batchnorm         BatchNorm2d después de cada convolución.
        resumen           "global" (AdaptiveAvgPool: un valor por canal), "flatten" (todos los
                          valores) o "atencion" (media ponderada con pesos aprendidos,
                          PoolingAtencion).
        cabeza_oculta     None = Linear directa a 1 salida; un número = capa intermedia.
        dropout           probabilidad de dropout en la cabeza (0 = sin dropout).
        entrada           "fases" (PRE, EARLY, LATE) o "fases+realce" (además EARLY − PRE y
                          LATE − EARLY, calculados dentro de la red).
        normalizacion     None (las fases tal cual, en [0, 1]) o "pre" (divididas entre la
                          intensidad media del PRE en el tejido del corte; normalizar_por_pre).
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
        self.entrada = entrada
        self.normalizacion = normalizacion

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
        lado_final = LADO_ENTRADA // 2 ** len(filtros)
        if resumen == "global":
            self.resumen = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten())
            n_vector = canales
        elif resumen == "atencion":
            self.resumen = PoolingAtencion(canales)
            n_vector = canales
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
        """(B, 3, H, W) → lo que ve el primer bloque: normalizada si toca, y con los canales de
        realce si los pide `entrada`."""
        if self.normalizacion == "pre":
            x = normalizar_por_pre(x)
        if self.entrada == "fases+realce":
            pre, early, late = x[:, 0:1], x[:, 1:2], x[:, 2:3]
            x = torch.cat([x, early - pre, late - early], dim=1)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cabeza(self.resumen(self.extractor(self.preparar_entrada(x))))
