"""Configuración de los experimentos: modos `rapido` y `completo`.

El modo indica QUÉ TIPO DE EJECUCIÓN es, no en qué ordenador estamos
(el dispositivo se detecta solo, ver `src/utils/dispositivo.py`):

- `rapido`:   comprobar que el código funciona de principio a fin con pocos
              datos y pocas épocas. Sus métricas NO cuentan para nada.
- `completo`: entrenamiento real. Sus métricas son las que se comparan y van
              al informe. El batch es fijo para que los entrenamientos sean
              comparables entre sí.

Los valores son de partida y se ajustarán al medir tiempos reales; cada cambio
se registra en docs/REGISTRO.md.

Uso:
    from src.config import obtener_config
    cfg = obtener_config("rapido")                 # valores del modo
    cfg = obtener_config("completo", batch_size=32)  # sobrescribir alguno
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace


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

    def como_dict(self) -> dict:
        return asdict(self)


MODOS: dict[str, Config] = {
    "rapido": Config(
        modo="rapido",
        n_pacientes=40,
        fold_val=0,
        batch_size=16,
        epocas_max=2,
        paciencia=2,
        learning_rate=1e-3,
        pacientes_auc_train=0,
        semilla=42,
        num_workers=0,
    ),
    "completo": Config(
        modo="completo",
        n_pacientes=None,
        fold_val=0,
        batch_size=32,
        epocas_max=60,
        paciencia=10,
        learning_rate=1e-3,
        pacientes_auc_train=220,
        semilla=42,
        num_workers=4,
    ),
}


def obtener_config(modo: str, **cambios) -> Config:
    """Devuelve la configuración del modo, con los cambios indicados aplicados."""
    if modo not in MODOS:
        raise ValueError(f"Modo desconocido: {modo!r}. Opciones: {list(MODOS)}")
    desconocidos = set(cambios) - set(Config.__dataclass_fields__)
    if desconocidos:
        raise ValueError(f"Parámetros desconocidos: {sorted(desconocidos)}")
    return replace(MODOS[modo], **{k: v for k, v in cambios.items() if v is not None})
