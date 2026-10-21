"""Resumen de la validación cruzada a partir de folds ya entrenados (sin imágenes)."""

import json

import numpy as np
import pandas as pd
import pytest

from src.training.validacion_cruzada import resumir


def fold_falso(carpeta, k, auc_val, n_pacientes=10):
    """Lo que deja una ejecución de entrenar.py en fold_k: predicciones, historial y resultados."""
    rng = np.random.default_rng(k)
    ids = [f"F{k}P{p}" for p in range(n_pacientes)]
    pred = pd.DataFrame({
        "sample_id": [f"{i}_z{c:03d}" for i in ids for c in range(3)],
        "patient_id": [i for i in ids for _ in range(3)],
        "fold": k,
        "pCR": [int(p % 3 == 0) for p in range(n_pacientes) for _ in range(3)],
        "prob": rng.random(3 * n_pacientes),
    })
    sub = carpeta / f"fold_{k}"
    sub.mkdir(parents=True)
    pred.to_csv(sub / "predicciones_val.csv", index=False)
    pd.DataFrame({"epoca": range(1, 7), "perdida_tr": np.linspace(0.7, 0.5, 6),
                  "perdida_va": np.linspace(0.7, 0.6, 6), "auc_tr": np.linspace(0.55, 0.8, 6),
                  "auc": np.linspace(0.5, auc_val, 6), "mejor": [True] * 4 + [False] * 2}).to_csv(
        sub / "historial.csv", index=False)
    mejor = {"epoca": 4, "auc_val": auc_val, "auc_train": auc_val + 0.1, "diferencia_auc": 0.1,
             "sensibilidad": 0.5, "especificidad": 0.8, "precision_pcr": 0.5, "f1_pcr": 0.5,
             "f1_no_pcr": 0.8, "f1_macro": 0.65, "accuracy": 0.7,
             "matriz_confusion": {"VP": 2, "FN": 2, "FP": 1, "VN": 5}}
    return {"mejor_epoca": mejor, "epocas": [{"segundos": 30.0}] * 6}


def test_resumen_media_desviacion_y_fuera_de_fold(tmp_path):
    por_fold = {k: fold_falso(tmp_path, k, auc) for k, auc in [(0, 0.60), (1, 0.70), (2, 0.65)]}
    resumen = resumir(tmp_path, por_fold, {"modelo": "prueba", "perdida": "normal"})

    md = resumen["media_desviacion"]["auc_val"]
    assert md["media"] == pytest.approx(0.65)
    assert md["desviacion"] == pytest.approx(0.05)          # desviación muestral (ddof=1)

    oof = resumen["fuera_de_fold"]
    assert oof["pacientes"] == 30
    assert sum(oof["matriz_confusion"].values()) == 30
    assert "siempre_no_pcr" in oof["referencias_triviales"]

    for fichero in ("resumen.json", "resumen.csv", "predicciones_oof.csv", "predicciones_oof_pacientes.csv",
                    "graficas/matriz_confusion_oof.png", "graficas/curvas_folds.png"):
        assert (tmp_path / fichero).exists(), fichero
    assert len(pd.read_csv(tmp_path / "predicciones_oof.csv")) == 90
    tabla = pd.read_csv(tmp_path / "resumen.csv")
    assert list(tabla.fold.astype(str)) == ["0", "1", "2", "media", "desviacion"]
    assert json.loads((tmp_path / "resumen.json").read_text(encoding="utf-8"))["folds_terminados"] == [0, 1, 2]


def test_senal_de_hospital():
    import pandas as pd

    from src.training.validacion_cruzada import senal_de_hospital

    pacientes = pd.DataFrame({"cohorte": ["spy2"] * 4 + ["duke"] * 4,
                              "prob_media": [0.9, 0.8, 0.85, 0.7, 0.1, 0.2, 0.15, 0.3]})
    s = senal_de_hospital(pacientes)
    assert s["spy2"] == 1.0 and s["duke"] == 0.0          # la puntuación delata el hospital del todo
    mezcla = pacientes.assign(prob_media=[0.1, 0.9, 0.2, 0.8, 0.1, 0.9, 0.2, 0.8])
    assert senal_de_hospital(mezcla)["spy2"] == 0.5        # no lo delata
