"""Recorte alrededor del tumor (D33): coordenadas de la caja, ventana, dataset y comprobación."""

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from src.data import recorte


def test_la_caja_pasa_a_pixeles_del_png():
    patients = pd.DataFrame({"pid": ["A", "B"], "n_xy": [256, 384], "sraw": [100, 150], "eraw": [140, 210],
                             "scol": [60, 30], "ecol": [100, 90]})
    cajas = recorte.cajas_en_png(patients)
    assert cajas["A"] == (100, 140, 60, 100)
    assert cajas["B"] == pytest.approx((100, 140, 20, 60))           # × 256 / 384


def test_la_ventana_se_centra_en_la_caja_y_no_se_sale():
    assert recorte.ventana((100, 140, 60, 100), lado=128) == (56, 16)
    assert recorte.ventana((0, 20, 240, 256), lado=128) == (0, 128)       # pegada a una esquina
    x = torch.arange(3 * 256 * 256, dtype=torch.float32).view(3, 256, 256)
    r = recorte.recortar(x, (100, 140, 60, 100))
    assert r.shape == (3, 128, 128) and torch.equal(r[1], x[1, 56:184, 16:144])   # misma ventana en las 3 fases


def png(ruta, matriz):
    ruta.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(matriz.astype(np.uint8), mode="L").save(ruta)


@pytest.fixture
def datos(tmp_path):
    """Dos pacientes; el «tumor» realza en la caja tal cual (filas 40-80, columnas 150-200)."""
    filas, patients = [], []
    for p in range(2):
        base = f"dataset/train/P{p}/P{p}_z010"
        pre = np.full((256, 256), 60)
        early = pre + 10
        early[40:80, 150:200] = 220                                   # realza mucho dentro de la caja
        for fase, m in (("PRE", pre), ("EARLY", early), ("LATE", early)):
            png(tmp_path / f"{base}_{fase}.png", m)
        filas.append({"sample_id": f"P{p}_z010", "patient_id": f"P{p}", "split": "train", "slice_index": 10,
                      "pCR": p, "path_pre": f"{base}_PRE.png", "path_early": f"{base}_EARLY.png",
                      "path_late": f"{base}_LATE.png"})
        patients.append({"pid": f"P{p}", "n_xy": 256, "sraw": 40, "eraw": 80, "scol": 150, "ecol": 200,
                         "dataset": "spy2"})
    return tmp_path, pd.DataFrame(filas), pd.DataFrame(patients)


def test_dataset_recortado(datos):
    raiz, filas, patients = datos
    ds = recorte.DatasetRecortado(filas, recorte.cajas_en_png(patients), raiz=raiz)
    x, y = ds[1]
    assert x.shape == (3, 128, 128) and float(y) == 1.0
    assert float(x[1].max()) == pytest.approx(220 / 255)              # el tumor está dentro del recorte
    with pytest.raises(ValueError, match="sin caja"):
        recorte.DatasetRecortado(filas, {}, raiz=raiz)
    with pytest.raises(ValueError):
        recorte.crear_dataset(filas, "circulo", raiz=raiz)


def test_la_comprobacion_distingue_la_caja_buena_de_la_traspuesta(datos, tmp_path):
    raiz, filas, patients = datos
    r = recorte.comprobar(filas, patients, n=2, raiz=raiz)
    assert r["tal cual"]["mediana"] > 3 > r["traspuesta"]["mediana"]
    ruta = recorte.figura(filas, patients, tmp_path / "cajas.png", por_cohorte=2, raiz=raiz)
    assert ruta.exists()


def test_las_versiones_con_caja_no_son_desplegables():
    from src.models.historial import desplegable, elegir_mejor
    from tests.test_plan import fila

    assert desplegable("cnn_v3") and not desplegable("cnn_v11")
    h = pd.DataFrame([fila("cnn_v3", 0.58, 0.03, 0.08, n=1), fila("cnn_v11", 0.75, 0.02, 0.05, n=2)])
    assert elegir_mejor(h)["version"] == "cnn_v3"                    # aunque cnn_v11 saque más
