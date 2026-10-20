"""Semilla única para que los entrenamientos sean reproducibles."""

from __future__ import annotations

import random

import numpy as np
import torch


def fijar_semilla(semilla: int) -> None:
    """Fija la semilla de Python, NumPy y PyTorch (CPU y GPU)."""
    random.seed(semilla)
    np.random.seed(semilla)
    torch.manual_seed(semilla)
    torch.cuda.manual_seed_all(semilla)
