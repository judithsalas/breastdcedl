"""Muestreo equilibrado por paciente (cnn_v15, D34).

Por qué: con el muestreo de siempre, cada época pasan los ~8.800 cortes de entrenamiento, y cada
paciente aparece 10 veces con cortes casi iguales. Eso facilita que la red memorice pacientes: su
puntuación casi no cambia entre los cortes de una misma paciente (desviación 0,019 frente a 0,096
entre pacientes, R5). Aquí, en cada época se eligen al azar `k` cortes de cada paciente. Todas las
pacientes pesan lo mismo, aunque tengan menos cortes, y la red ve en cada época otros cortes de la
misma paciente.

No cambia lo que el modelo necesita para predecir: sigue siendo un corte 2D (lo que recibe la
app). Solo cambia qué cortes se usan para entrenar en cada época. Las épocas son más cortas
(unos 1.750 cortes con k = 2), así que la parada temprana llega antes en número de cortes vistos.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from torch.utils.data import Sampler


class MuestreoPorPaciente(Sampler[int]):
    """Cada época, `k` cortes al azar de cada paciente, en orden aleatorio.

    Los índices son posiciones en `filas` tras `reset_index`, igual que en el Dataset.
    """

    def __init__(self, filas: pd.DataFrame, k: int = 2, semilla: int = 42) -> None:
        if k < 1:
            raise ValueError("k debe ser al menos 1")
        posiciones = filas.reset_index(drop=True).groupby("patient_id", sort=True).indices
        self.grupos = [np.asarray(v) for v in posiciones.values()]
        self.k = k
        self.generador = np.random.default_rng(semilla)   # se mantiene entre épocas: cada una, otros cortes

    def __len__(self) -> int:
        return sum(min(self.k, len(g)) for g in self.grupos)

    def __iter__(self):
        elegidos = np.concatenate([self.generador.choice(g, min(self.k, len(g)), replace=False)
                                   for g in self.grupos])
        self.generador.shuffle(elegidos)
        return iter(elegidos.tolist())
