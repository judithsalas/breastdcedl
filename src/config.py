"""Configuración de los experimentos, leída de `config.json` en la raíz del repo (D31).

`config.json` es el único sitio con los parámetros de ejecución:
- `modos`: los dos tipos de ejecución. El modo indica QUÉ TIPO DE EJECUCIÓN es, no en qué
  ordenador estamos (el dispositivo se detecta solo, ver `src/utils/dispositivo.py`):
  - `rapido`: comprobar que el código funciona de principio a fin con pocos datos y pocas
    épocas. Sus métricas NO cuentan para nada.
  - `completo`: entrenamiento real. Sus métricas son las que se comparan y van al informe. El
    batch es fijo para que los entrenamientos sean comparables entre sí. La época se elige con la
    validación del propio fold (parada temprana).
  - `fijo` (D37): como `completo`, pero con un número de épocas fijado de antemano y quedándose con
    la última. Así la época no se elige mirando la validación, y su AUC no lleva ese sesgo (R10).
- `evaluacion` y `comparacion`: agregación por paciente, umbrales y límite de sobreajuste.
- `experimento`: la plantilla de lo que lanza `python -m src.training.experimento`.

Lo que define una versión de la red (capas, aumento, optimizador, weight decay) NO está en
config.json, sino en `src/models/versiones.py`: cambiarlo es crear otra versión.

Uso:
    from src.config import obtener_config
    cfg = obtener_config("rapido")                   # valores del modo
    cfg = obtener_config("completo", batch_size=64)  # sobrescribir alguno
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path

RUTA_CONFIG = Path(__file__).resolve().parents[1] / "config.json"
# Los mismos que METODOS_AGREGACION de utils_caso (un test lo comprueba). No se importa utils_caso
# aquí para que leer la configuración no cargue torch.
AGREGACIONES = ("mean", "max", "median", "voto")
ELECCIONES_EPOCA = ("mejor", "ultima")


@dataclass(frozen=True)
class Config:
    modo: str
    # Datos
    n_pacientes: int | None  # None = todas las pacientes de train
    fold_val: int            # pliegue de validación interna (0-4)
    # Entrenamiento
    batch_size: int
    epocas_max: int
    paciencia: int           # épocas sin mejorar (AUC por paciente en validación) antes de parar
    learning_rate: float
    # Sobreajuste: el AUC por paciente en entrenamiento se mide cada época, en modo
    # evaluación, sobre una muestra fija y estratificada de pacientes de entrenamiento
    # (0 = todas). 220 ≈ el tamaño de un fold, para que el AUC de entrenamiento y el de
    # validación tengan un ruido parecido y su diferencia sea comparable.
    pacientes_auc_train: int
    # Reproducibilidad y carga
    semilla: int
    num_workers: int
    # Qué época se guarda (D37): "mejor" = la de mayor AUC de validación, con parada temprana;
    # "ultima" = la última de epocas_max, sin mirar la validación para decidir nada.
    eleccion_epoca: str = "mejor"

    def como_dict(self) -> dict:
        return asdict(self)


CAMPOS_MODO = tuple(f.name for f in fields(Config) if f.name != "modo")


def leer_config(ruta: Path | str = RUTA_CONFIG) -> dict:
    """Lee config.json y comprueba que no falta ni sobra nada, con un mensaje claro si algo está mal."""
    ruta = Path(ruta)
    try:
        datos = json.loads(ruta.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"{ruta.name} no es un JSON válido (línea {error.lineno}): {error.msg}") from None
    for seccion in ("modos", "evaluacion", "comparacion"):
        if seccion not in datos:
            raise ValueError(f"A {ruta.name} le falta la sección {seccion!r}")
    for modo, valores in datos["modos"].items():
        faltan, sobran = set(CAMPOS_MODO) - set(valores), set(valores) - set(CAMPOS_MODO)
        if faltan or sobran:
            raise ValueError(f"Modo {modo!r} de {ruta.name}: faltan {sorted(faltan)}, sobran {sorted(sobran)}")
    for modo, valores in datos["modos"].items():
        if valores["eleccion_epoca"] not in ELECCIONES_EPOCA:
            raise ValueError(f"Modo {modo!r}: eleccion_epoca debe ser una de {ELECCIONES_EPOCA}")
    if datos["evaluacion"].get("agregacion") not in AGREGACIONES:
        raise ValueError(f"evaluacion.agregacion debe ser una de {AGREGACIONES}")
    return datos


CONFIG = leer_config()
MODOS: dict[str, Config] = {nombre: Config(modo=nombre, **valores) for nombre, valores in CONFIG["modos"].items()}
EVALUACION: dict = CONFIG["evaluacion"]
COMPARACION: dict = CONFIG["comparacion"]


def obtener_config(modo: str, **cambios) -> Config:
    """Devuelve la configuración del modo, con los cambios indicados aplicados."""
    if modo not in MODOS:
        raise ValueError(f"Modo desconocido: {modo!r}. Opciones: {list(MODOS)}")
    desconocidos = set(cambios) - set(Config.__dataclass_fields__)
    if desconocidos:
        raise ValueError(f"Parámetros desconocidos: {sorted(desconocidos)}")
    return replace(MODOS[modo], **{k: v for k, v in cambios.items() if v is not None})


def es_estandar(modo: str, batch: float | None, lr: float | None, semilla: float | None,
                lr_version: float | None = None) -> bool:
    """¿Una ejecución usó el batch, el learning rate y la semilla de su modo? Solo esas cuentan para
    comparar versiones (D31): una prueba con otro learning rate no debe pasar por la versión.
    Si la versión fija su propio learning rate en su receta (`lr_version`, D34), se compara con ese."""
    if modo not in MODOS:
        return False
    estandar = MODOS[modo]
    lr_esperado = estandar.learning_rate if lr_version is None else lr_version
    for valor, referencia in ((batch, estandar.batch_size), (lr, lr_esperado),
                              (semilla, estandar.semilla)):
        if valor is None or math.isnan(float(valor)):       # ejecuciones antiguas sin ese dato
            continue
        if not math.isclose(float(valor), float(referencia), rel_tol=1e-9):
            return False
    return True
