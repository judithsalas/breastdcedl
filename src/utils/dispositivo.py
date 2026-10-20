"""Detección del dispositivo de cálculo (CPU, GPU AMD con ROCm o GPU NVIDIA con CUDA).

En PyTorch, tanto ROCm como CUDA se usan a través de `torch.cuda`, así que el
mismo código funciona en el PC de la universidad (AMD, ROCm) y en cualquier
máquina sin GPU (CPU). No hay que tocar nada al cambiar de ordenador.
"""

from __future__ import annotations

import platform

import torch


def elegir_dispositivo() -> torch.device:
    """GPU si PyTorch la ve; si no, CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def describir_dispositivo(dispositivo: torch.device | None = None) -> dict:
    """Información de la máquina para guardar junto a cada entrenamiento.

    El enunciado pide indicar en el informe el dispositivo y la versión del entorno.
    """
    dispositivo = dispositivo or elegir_dispositivo()
    info = {
        "dispositivo": dispositivo.type,
        "sistema": f"{platform.system()} {platform.release()}",
        "maquina": platform.node(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "backend": "cpu",
        "gpu": None,
        "vram_gb": None,
    }
    if dispositivo.type == "cuda":
        propiedades = torch.cuda.get_device_properties(dispositivo)
        info["backend"] = f"rocm {torch.version.hip}" if torch.version.hip else f"cuda {torch.version.cuda}"
        info["gpu"] = propiedades.name
        info["vram_gb"] = round(propiedades.total_memory / 1024**3, 1)
    return info


if __name__ == "__main__":
    # python -m src.utils.dispositivo  -> comprobación rápida de la instalación
    for clave, valor in describir_dispositivo().items():
        print(f"{clave:>12}: {valor}")
