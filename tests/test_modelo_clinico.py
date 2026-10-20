"""Modelo clínico de comparación (D25): variables permitidas, sin fugas, reproducible y conectado a test."""

import json
import sys

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from src.evaluation import evaluar_test
from src.evaluation import modelo_clinico as mc
from src.evaluation.metricas import diferencia_auc


def tabla_falsa(n=400, semilla=0):
    """Pacientes de train inventadas: HER2+ y HR− con más pCR, como en el dataset."""
    rng = np.random.default_rng(semilla)
    t = pd.DataFrame({
        "fold": np.arange(n) % 5,
        "dataset": rng.choice(["spy1", "spy2", "duke"], n),
        "age": rng.normal(49, 10, n).round(1),
        "HR": rng.integers(0, 2, n).astype(float),
        "HER2": rng.integers(0, 2, n).astype(float),
        "tum_vol": rng.lognormal(2.5, 1, n),
    }, index=[f"P{i:04d}" for i in range(n)])
    t["menopause"] = (t.age + rng.normal(0, 4, n) > 51).astype(float)
    riesgo = -1.0 + 1.3 * t.HER2 - 1.2 * t.HR - 0.02 * (t.age - 49)
    t["pCR"] = (rng.random(n) < 1 / (1 + np.exp(-riesgo))).astype(int)
    t.loc[t.dataset == "spy1", "menopause"] = np.nan          # como en I-SPY1
    t.loc[t.index[:7], "HER2"] = np.nan
    t["subtipo"] = np.where(t.HER2 == 1, "HER2pos", np.where(t.HR == 1, "HRposHER2neg", "TripleNeg"))
    return t


def test_solo_variables_que_sabe_el_medico():
    prohibidas = {"race_white", "race_black", "dataset", "tum_vol", "n_xy", "n_z", "slice_thick", "mask_start"}
    assert not prohibidas & set(mc.VARIABLES_MEDICO)
    assert set(mc.VARIABLES_COMBINADO) == {mc.CNN, *mc.VARIABLES_MEDICO}
    assert "tum_vol" in mc.VARIANTE_VOLUMEN and "tum_vol" in mc.NO_SE_USAN


def test_lo_que_falta_se_completa_con_lo_aprendido_en_train():
    t = tabla_falsa()
    entrena, valida = t[t.fold != 0], t[t.fold == 0].copy()
    modelo = mc.ModeloLogistico.ajustar(entrena)
    valida["HR"] = np.where(valida.HR.isna(), np.nan, 0.0)
    valida.loc[valida.index[0], "HR"] = np.nan
    assert modelo.medianas["HR"] == entrena.HR.median()          # aprendido solo en train
    x = modelo.completar(valida)
    assert not x.isna().any().any()
    assert x.loc[valida.index[0], "HR"] == entrena.HR.median()
    # Menopausia que falta: probabilidad según la edad, entre 0 y 1 y creciente con la edad
    sin = valida[valida.menopause.isna()].sort_values("age")
    completada = x.loc[sin.index, "menopause"]
    assert ((completada > 0) & (completada < 1)).all()
    assert completada.is_monotonic_increasing


def test_cada_paciente_la_predice_el_modelo_sin_su_fold():
    t = tabla_falsa()
    r = mc.validacion_cruzada(t, mc.VARIABLES_MEDICO)
    a, b = t[t.fold != 2], t[t.fold == 2]
    esperado = mc.ModeloLogistico.ajustar(a).probabilidad(b)
    np.testing.assert_allclose(r["prob"].loc[b.index].to_numpy(), esperado)
    assert len(r["por_fold"]) == 5 and r["auc_val"] > 0.6


def test_el_json_reproduce_las_probabilidades():
    t = tabla_falsa()
    modelo = mc.ModeloLogistico.ajustar(t)
    copia = mc.ModeloLogistico.desde_dict(json.loads(json.dumps(modelo.a_dict())))
    np.testing.assert_array_equal(copia.probabilidad(t), modelo.probabilidad(t))
    razones = modelo.razones_de_odds()
    assert razones["HER2"] > 1 > razones["HR"]                     # aprende la señal que se puso


def test_diferencia_auc_pareada():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 300)
    bueno = y + rng.normal(0, 0.7, 300)
    malo = y + rng.normal(0, 3, 300)
    igual = diferencia_auc(y, bueno, bueno, n=200)
    assert igual["diferencia"] == 0 and igual["ic95"] == [0, 0]
    d = diferencia_auc(y, bueno, malo, n=200)
    assert d["ic95"][0] <= d["diferencia"] <= d["ic95"][1]
    assert d["diferencia"] == pytest.approx(roc_auc_score(y, bueno) - roc_auc_score(y, malo))
    assert d["ic95"][0] > 0


def test_generar_sin_cnn_y_con_cnn(tmp_path):
    t = tabla_falsa()
    rutas = {"destino": tmp_path / "modelo.json", "informe_md": tmp_path / "CLINICO.md",
             "carpeta_figuras": tmp_path / "fig", "n_bootstrap": 20}
    analisis, guardado = mc.generar(t, None, None, "sin CNN", **rutas)
    assert guardado["combinado"] is None and guardado["huella_cnn"] is None
    assert "Pendiente: sin CNN" in rutas["informe_md"].read_text(encoding="utf-8")

    rng = np.random.default_rng(2)
    cnn = pd.Series(np.clip(0.3 + 0.2 * t.pCR + rng.normal(0, 0.15, len(t)), 0.01, 0.99), index=t.index)
    analisis, guardado = mc.generar(t, cnn, "abcd1234", **rutas)
    assert guardado["huella_cnn"] == "abcd1234" and guardado["combinado"]["variables"][0] == mc.CNN
    assert {"cnn", "combinado"} <= set(analisis["modelos"])
    c = analisis["comparacion"]
    assert c["combinado_menos_medico"]["ic95"][0] <= c["combinado_menos_medico"]["diferencia"]
    texto = rutas["informe_md"].read_text(encoding="utf-8")
    assert "CNN frente al médico" in texto and "lo que **añade la imagen**" in texto
    assert (tmp_path / "fig" / "roc_fuera_de_fold.png").exists() and (tmp_path / "fig" / "razones_de_odds.png").exists()

    # evaluar_test usa el guardado: misma predicción que el modelo en memoria
    cargado = mc.cargar(rutas["destino"])
    pred = mc.predecir(t, cnn, cargado)
    np.testing.assert_allclose(pred.prob_medico, analisis["finales"]["medico"].probabilidad(t))
    assert pred.prob_combinado.between(0, 1).all()


def test_comparacion_en_test(tmp_path):
    t = tabla_falsa()
    rng = np.random.default_rng(4)
    cnn = pd.Series(np.clip(0.3 + 0.2 * t.pCR + rng.normal(0, 0.15, len(t)), 0.01, 0.99), index=t.index)
    mc.generar(t, cnn, "abcd1234", destino=tmp_path / "m.json", informe_md=tmp_path / "c.md",
               carpeta_figuras=tmp_path / "fig", n_bootstrap=10)
    guardado = mc.cargar(tmp_path / "m.json")
    pacientes = pd.DataFrame({"pCR": t.pCR, "prob": cnn, "cohorte": t.dataset})
    comparacion, pred = evaluar_test.comparar_con_clinico(pacientes, t, guardado)
    assert set(comparacion["modelos"]) == {"medico", "combinado"}
    assert set(comparacion["diferencias"]) == {"cnn_menos_medico", "combinado_menos_medico"}
    assert comparacion["modelos"]["medico"]["umbral"] == guardado["medico"].umbral
    texto = "\n".join(evaluar_test.seccion_clinica({"auc": 0.6, "ic95": [0.5, 0.7], "sensibilidad": 0.5,
                                                     "especificidad": 0.8, "precision_pcr": 0.5, "f1_macro": 0.6,
                                                     "brier": 0.2, "comparacion_clinica": comparacion}))
    assert "Médico (sin imagen)" in texto and "CNN + médico" in texto
    assert evaluar_test.figura_roc_comparada(pacientes, pred, tmp_path / "roc.png").exists()


def test_test_exige_el_modelo_clinico_de_esta_cnn(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluar_test, "CARPETA_TEST", tmp_path / "test")
    monkeypatch.setattr(evaluar_test, "cargar_modelo_final", lambda _: (None, {"huella_modelo": "cnn_buena"}))
    monkeypatch.setattr(mc, "MODELO_JSON", tmp_path / "no_existe.json")
    monkeypatch.setattr(sys, "argv", ["evaluar_test"])
    with pytest.raises(SystemExit, match="modelo clínico"):
        evaluar_test.main()

    t = tabla_falsa(100)
    mc.generar(t, None, None, destino=tmp_path / "m.json", informe_md=tmp_path / "c.md",
               carpeta_figuras=tmp_path / "fig", n_bootstrap=5)
    monkeypatch.setattr(mc, "MODELO_JSON", tmp_path / "m.json")
    with pytest.raises(SystemExit, match="Vuelve a ejecutar"):          # hecho sin CNN / con otra CNN
        evaluar_test.main()
    assert not (tmp_path / "test").exists()                              # no ha llegado a mirar test


def test_cnn_cerrada_comprueba_que_es_el_modelo_final(tmp_path):
    manifiesto, oof = tmp_path / "manifiesto.json", tmp_path / "cierre" / "oof_pacientes.csv"
    assert mc.cnn_cerrada(oof, manifiesto)[0] is None
    manifiesto.write_text(json.dumps({"huella_modelo": "aaaa"}), encoding="utf-8")
    oof.parent.mkdir()
    pd.DataFrame({"patient_id": ["P1", "P2"], "pCR": [0, 1], "prob": [0.2, 0.7]}).to_csv(oof, index=False)
    (oof.parent / "cierre.json").write_text(json.dumps({"huella_modelo": "bbbb"}), encoding="utf-8")
    prob, huella, motivo = mc.cnn_cerrada(oof, manifiesto)
    assert prob is None and "no corresponde" in motivo
    (oof.parent / "cierre.json").write_text(json.dumps({"huella_modelo": "aaaa"}), encoding="utf-8")
    prob, huella, motivo = mc.cnn_cerrada(oof, manifiesto)
    assert huella == "aaaa" and motivo == "ok" and prob.loc["P2"] == 0.7
