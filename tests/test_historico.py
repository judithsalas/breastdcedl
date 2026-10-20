"""Histórico de ejecuciones y gráficas a partir de carpetas de runs/ falsas (sin imágenes)."""

import json

import pandas as pd
import pytest

from src.models.construir import huella
from src.models.versiones import VERSIONES
from src.training import historico
from src.training.graficas import dibujar_graficas, epoca_elegida


def epocas(n=4, con_auc_tr=True):
    filas = []
    for e in range(1, n + 1):
        fila = {"epoca": e, "perdida_tr": 0.6 - 0.02 * e, "perdida_va": 0.6 + 0.01 * e,
                "auc": 0.55 + (0.02 if e == 2 else 0), "sensibilidad": 0.1, "especificidad": 0.9,
                "accuracy": 0.7, "VP": 6, "VN": 140, "FP": 15, "FN": 58, "segundos": 27.0,
                "mejor": e in (1, 2)}
        if con_auc_tr:
            fila["auc_tr"] = 0.6 + 0.03 * e
        filas.append(fila)
    return filas


def ejecucion_falsa(raiz, nombre, nueva=True):
    """Una carpeta de runs/: 'nueva' con mejor_epoca; si no, como R1/R2 (sin AUC de entrenamiento)."""
    carpeta = raiz / nombre
    carpeta.mkdir(parents=True)
    ep = epocas(con_auc_tr=nueva)
    r = {"modelo": "cnn_v1", "parametros": 23761, "perdida": "normal", "umbral": 0.5, "agregacion": "mean",
         "nota": "", "config": {"modo": "completo", "fold_val": 0, "batch_size": 32, "learning_rate": 1e-3,
                                "semilla": 42},
         "maquina": {"backend": "rocm 7.14", "gpu": "AMD Radeon RX 6700 XT", "torch": "2.14.0+rocm7.14"},
         "epocas": ep, "tiempo_total_s": 120}
    if nueva:
        r["arquitectura"] = VERSIONES["cnn_v1"]
        r["mejor_epoca"] = {"epoca": 2, "auc_val": 0.57, "auc_train": 0.66, "diferencia_auc": 0.09,
                            "sensibilidad": 0.1, "especificidad": 0.9, "f1_pcr": 0.16, "f1_no_pcr": 0.8,
                            "f1_macro": 0.48, "matriz_confusion": {"VP": 6, "VN": 140, "FP": 15, "FN": 58}}
    (carpeta / "resultados.json").write_text(json.dumps(r), encoding="utf-8")
    pd.DataFrame(ep).to_csv(carpeta / "historial.csv", index=False)
    return carpeta


def test_filas_de_ejecuciones_nuevas_y_antiguas(tmp_path):
    ejecucion_falsa(tmp_path, "20261007-211625_cnn_v1_completo", nueva=False)    # como R1
    ejecucion_falsa(tmp_path, "20261015-100000_cnn_v2_completo", nueva=True)
    tabla = pd.read_csv(historico.actualizar(tmp_path))
    assert list(tabla.ejecucion) == ["20261007-211625_cnn_v1_completo", "20261015-100000_cnn_v2_completo"]
    antigua, nueva = tabla.iloc[0], tabla.iloc[1]
    assert antigua.epoca_elegida == 2 and antigua.auc_val == pytest.approx(0.57)
    assert pd.isna(antigua.auc_train)                                   # R1 no medía el AUC de entrenamiento
    assert antigua.f1_no_pcr == pytest.approx(2 * 140 / (2 * 140 + 58 + 15), abs=1e-4)
    assert nueva.diferencia_auc == pytest.approx(0.09)
    assert antigua.fecha == "2026-10-07 21:16" and antigua.dispositivo == "rocm 7.14 AMD Radeon RX 6700 XT"
    assert antigua.torch == "2.14.0+rocm7.14"
    assert pd.isna(antigua.huella) and nueva.huella == huella("cnn_v1")


def test_conserva_las_filas_de_otra_maquina(tmp_path):
    ejecucion_falsa(tmp_path, "20261015-100000_cnn_v2_completo")
    ruta = historico.actualizar(tmp_path)
    # Llega por git una fila entrenada en otra máquina, cuya carpeta aquí no existe.
    tabla = pd.read_csv(ruta)
    otra = tabla.iloc[0].copy()
    otra["ejecucion"], otra["fecha"] = "20261016-090000_cnn_v3_completo", "2026-10-16 09:00"
    pd.concat([tabla, otra.to_frame().T]).to_csv(ruta, index=False)

    ejecucion_falsa(tmp_path, "20261017-120000_cnn_v3_rapido")
    tabla = pd.read_csv(historico.actualizar(tmp_path))
    assert len(tabla) == 3
    assert "20261016-090000_cnn_v3_completo" in set(tabla.ejecucion)
    assert tabla.ejecucion.is_unique


def test_graficas_de_una_ejecucion_antigua(tmp_path):
    carpeta = ejecucion_falsa(tmp_path, "20261007-211625_cnn_v1_completo", nueva=False)
    rutas = dibujar_graficas(carpeta)
    assert {r.name for r in rutas} == {"curvas.png", "matriz_confusion.png"}
    assert epoca_elegida(pd.read_csv(carpeta / "historial.csv")) == 2
