"""Copia unos cortes de ejemplo a app/ejemplos/ y guarda la probabilidad que da el pipeline.

    python -m app.preparar_ejemplos

Se lanza una vez, con el modelo final ya cerrado y las imágenes descargadas (PC de la uni).
Elige el corte central de una paciente con pCR y otra sin pCR de cada cohorte, siempre de
TRAIN (los ejemplos son una demostración; no se usa test ni, por supuesto, la validación privada).

Para cada ejemplo guarda en app/ejemplos/esperado.json la probabilidad que da el pipeline,
cargando las imágenes con utils_caso (el camino del entrenamiento). La app la compara con la suya,
que lee los mismos ficheros por su propio camino (src/data/entrada.py): si coinciden, la
normalización es idéntica en entrenamiento e inferencia (lo que se verifica en la defensa).
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import torch

from src.data.caso import RAIZ_DATOS, uc
from src.models.final import cargar_modelo_final
from src.training.entrenar import cohortes, pacientes_con_imagenes

DESTINO = Path(__file__).resolve().parent / "ejemplos"
LEEME = """# Ejemplos de la app

Cortes de **train** del dataset docente derivado de BreastDCEDL (Fridman et al., *Scientific Data*,
2026, doi:10.1038/s41597-026-06589-6), incluidos solo como demostración, con su licencia
(CC BY-NC 4.0, sin uso comercial) y esta atribución. No contienen datos de test ni de la
validación privada del profesor. Generados con `python -m app.preparar_ejemplos`.
"""


def main(semilla: int = 42) -> None:
    modelo, manifiesto = cargar_modelo_final()
    samples = uc.cargar_samples()
    train = pacientes_con_imagenes(samples[samples.split == "train"])    # solo con las 3 fases descargadas
    mapa = cohortes()
    rng = np.random.default_rng(semilla)
    if DESTINO.exists():
        shutil.rmtree(DESTINO)
    DESTINO.mkdir(parents=True)
    esperado = {}
    for cohorte in ("spy1", "spy2", "duke"):
        for clase in (0, 1):
            ids = sorted(pid for pid in train[train.pCR == clase].patient_id.unique() if mapa.get(pid) == cohorte)
            if not ids:
                continue
            cortes = train[train.patient_id == ids[rng.integers(len(ids))]].sort_values("slice_index")
            fila = cortes.iloc[len(cortes) // 2]
            carpeta = DESTINO / fila.sample_id
            carpeta.mkdir()
            for col in ("path_pre", "path_early", "path_late"):
                shutil.copy2(RAIZ_DATOS / fila[col], carpeta / Path(fila[col]).name)
            x = torch.from_numpy(uc.cargar_imagen(fila)).unsqueeze(0)
            esperado[fila.sample_id] = {"probabilidad_pipeline": float(modelo.probabilidad(x)[0]),
                                        "cohorte": cohorte}
            print(f"{fila.sample_id}: P(pCR) del pipeline = {esperado[fila.sample_id]['probabilidad_pipeline']:.6f}")
    (DESTINO / "esperado.json").write_text(json.dumps({"modelo": manifiesto["huella_modelo"], **esperado},
                                                      indent=2), encoding="utf-8")
    (DESTINO / "LEEME.md").write_text(LEEME, encoding="utf-8")
    print(f"{len(esperado)} ejemplos en {DESTINO}")


if __name__ == "__main__":
    main()
