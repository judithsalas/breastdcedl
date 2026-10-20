"""Métricas por clase y matriz de confusión (sin imágenes)."""

import math

import numpy as np
import pytest

from src.evaluation.metricas import dibujar_matriz_confusion, metricas_por_clase, referencias_triviales


def test_metricas_por_clase_a_mano():
    # 10 pCR (VP 6, FN 4) y 20 no pCR (VN 16, FP 4)
    m = metricas_por_clase({"VP": 6, "FN": 4, "VN": 16, "FP": 4})
    assert m["precision_pcr"] == pytest.approx(6 / 10)
    assert m["recall_pcr"] == pytest.approx(6 / 10)          # sensibilidad
    assert m["recall_no_pcr"] == pytest.approx(16 / 20)      # especificidad
    assert m["precision_no_pcr"] == pytest.approx(16 / 20)   # VPN
    assert m["f1_pcr"] == pytest.approx(0.6)
    assert m["f1_no_pcr"] == pytest.approx(0.8)
    assert m["f1_macro"] == pytest.approx(0.7)
    assert m["accuracy"] == pytest.approx(22 / 30)


def test_referencias_triviales_con_la_prevalencia_de_test():
    # 53 pCR y 123 no pCR, como el conjunto test
    r = referencias_triviales(53, 123)
    no = r["siempre_no_pcr"]
    assert no["f1_pcr"] == 0 and no["recall_pcr"] == 0
    assert no["f1_no_pcr"] == pytest.approx(0.823, abs=1e-3)
    assert no["f1_macro"] == pytest.approx(0.411, abs=1e-3)
    assert r["siempre_pcr"]["f1_pcr"] == pytest.approx(0.463, abs=1e-3)


def test_sin_predicciones_positivas():
    # Nunca predice pCR: la precision no existe (0/0), pero la F1 de pCR es 0, no NaN.
    m = metricas_por_clase({"VP": 0, "FN": 5, "VN": 10, "FP": 0})
    assert math.isnan(m["precision_pcr"])
    assert m["f1_pcr"] == 0


def test_dibujar_matriz(tmp_path):
    ruta = dibujar_matriz_confusion({"VP": 6, "FN": 4, "VN": 16, "FP": 4}, tmp_path / "m.png", "prueba", "sub")
    assert ruta.exists() and ruta.stat().st_size > 0


def test_auc_por_rangos_es_el_de_sklearn():
    from sklearn.metrics import roc_auc_score

    from src.evaluation.metricas import auc_por_rangos

    rng = np.random.default_rng(5)
    y = rng.integers(0, 2, 200)
    p = np.round(rng.random(200), 1)                    # muchos empates a propósito
    assert auc_por_rangos(y, p) == pytest.approx(roc_auc_score(y, p), abs=1e-12)
