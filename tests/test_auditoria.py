"""Auditoría de datos: metadatos (reales, están en el repo) y medidas de imagen con PNG sintéticos."""

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from src.data import auditoria
from src.data.caso import uc


@pytest.fixture(scope="module")
def metadatos():
    return uc.cargar_samples(), uc.cargar_patients()


def test_recuentos(metadatos):
    r = auditoria.recuentos(*metadatos)
    assert r["por_split"]["train"]["pacientes"] == 1097 and r["por_split"]["test"]["pacientes"] == 176
    assert sum(v["pacientes"] for v in r["por_fold"].values()) == 1097
    assert r["cortes_por_paciente"]["min"] == 5


def test_referencias_sin_imagen(metadatos):
    ref = auditoria.referencias_sin_imagen(*metadatos)
    assert ref["cohorte"]["media"] == pytest.approx(0.549, abs=0.01)     # R0 del registro
    assert ref["subtipo"]["media"] == pytest.approx(0.657, abs=0.01)
    assert len(ref["subtipo"]["auc_por_fold"]) == 5


def test_medir_cortes_con_png_sinteticos(tmp_path):
    filas = []
    for p in range(4):
        base = f"dataset/train/P{p}/P{p}_z010"
        for fase, valor in (("PRE", 0.3), ("EARLY", 0.5 + 0.1 * (p % 2)), ("LATE", 0.45)):
            ruta = tmp_path / f"{base}_{fase}.png"
            ruta.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(np.full((256, 256), int(valor * 255), np.uint8), mode="L").save(ruta)
        filas.append({"sample_id": f"P{p}_z010", "patient_id": f"P{p}", "pCR": p % 2, "path_pre": f"{base}_PRE.png",
                      "path_early": f"{base}_EARLY.png", "path_late": f"{base}_LATE.png"})
    cortes, problemas = auditoria.medir_cortes(pd.DataFrame(filas), tmp_path)
    assert not problemas and len(cortes) == 4
    assert (cortes.PRE < cortes.EARLY).all()
    assert cortes.loc[cortes.pCR == 1, "realce"].mean() > cortes.loc[cortes.pCR == 0, "realce"].mean()


def test_informe_sin_imagenes(metadatos):
    datos = {"fecha": "hoy", "recuentos": auditoria.recuentos(*metadatos), "problemas": [], "muestra": False,
             "referencias_sin_imagen": auditoria.referencias_sin_imagen(*metadatos)}
    texto = auditoria.informe(datos)
    assert "No había imágenes descargadas" in texto and "## 2. Referencias sin imagen" in texto
