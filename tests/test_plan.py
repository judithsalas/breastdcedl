"""Criterio de comparación entre versiones (D16) y estado del plan de experimentos (D17)."""

import pandas as pd

from src.models import historial
from src.models.construir import huella


def fila(modelo, auc, desv, dif, perdida="normal", folds="0,1,2,3,4", modo="completo", buena=True, n=0):
    return {"fecha": "", "ejecucion": f"2026101{n}-000000_cv_{modelo}_{modo}", "tipo": "cv", "modo": modo,
            "modelo": modelo, "huella": huella(modelo) if buena else "00000000", "perdida": perdida,
            "folds": folds, "epoca_elegida": 10, "auc_val": auc, "auc_val_desv": desv, "auc_train": auc + dif,
            "diferencia_auc": dif, "f1_macro": 0.5}


def test_gana_la_mejor_si_nadie_empata():
    h = pd.DataFrame([fila("cnn_v2", 0.60, 0.02, 0.10, n=1), fila("cnn_v3", 0.68, 0.02, 0.08, n=2)])
    assert historial.elegir_mejor(h)["version"] == "cnn_v3"


def test_empate_gana_la_que_no_sobreajusta_y_luego_la_mas_simple():
    # Las tres empatan (a menos de una desviación de cnn_v4); cnn_v2 sobreajusta.
    h = pd.DataFrame([fila("cnn_v2", 0.630, 0.03, 0.25, n=1), fila("cnn_v3", 0.640, 0.03, 0.08, n=2),
                      fila("cnn_v4", 0.645, 0.03, 0.07, n=3)])
    mejor = historial.elegir_mejor(h)
    assert mejor["referencia"] == "cnn_v4"
    assert mejor["empatadas"] == ["cnn_v2", "cnn_v3", "cnn_v4"]
    assert mejor["version"] == "cnn_v3"          # no sobreajusta y tiene menos parámetros que cnn_v4


def test_solo_cuentan_las_validaciones_validas():
    h = pd.DataFrame([
        fila("cnn_v4", 0.90, 0.01, 0.0, folds="0,1", n=1),            # no son los 5 folds
        fila("cnn_v4", 0.90, 0.01, 0.0, perdida="ponderada", n=2),    # no es pérdida normal
        fila("cnn_v4", 0.90, 0.01, 0.0, buena=False, n=3),            # la versión cambió después
        fila("cnn_v2", 0.61, 0.02, 0.05, n=4),
    ])
    assert historial.elegir_mejor(h)["version"] == "cnn_v2"
    assert historial.elegir_mejor(pd.DataFrame()) is None


def test_estado_del_plan(tmp_path, monkeypatch):
    from src.training import plan
    ruta = tmp_path / "historico.csv"
    pd.DataFrame([fila("cnn_v2", 0.62, 0.02, 0.05, n=1), fila("cnn_v3", 0.66, 0.02, 0.05, n=2)]).to_csv(
        ruta, index=False)
    monkeypatch.setattr(plan, "HISTORICO", ruta)
    estado = {f["plan"]: f for f in plan.estado("completo")}
    assert estado["cnn_v2"]["hecho"] and estado["cnn_v3"]["hecho"] and not estado["cnn_v4"]["hecho"]
    assert estado["mejor"]["version"] == "cnn_v3" and not estado["mejor"]["hecho"]
