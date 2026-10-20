"""Cierre del modelo (F4), modelo final y evaluación de test (F5), sin imágenes reales."""

import json
import sys

import numpy as np
import pandas as pd
import pytest
import torch

from src.evaluation import cerrar_modelo, evaluar_test, umbral
from src.evaluation.metricas import auc_con_ic, metricas_con_umbral
from src.models.cnn import CNN
from src.models.construir import crear_modelo
from src.models.final import ModeloFinal, cargar_modelo_final
from src.models.versiones import VERSIONES


def cv_falsa(raiz, pacientes_por_fold=12, semilla=0):
    """Una validación cruzada como la que deja validacion_cruzada.py, con modelos sin entrenar."""
    rng = np.random.default_rng(semilla)
    carpeta = raiz / "20261020-000000_cv_cnn_v1_completo"
    filas = []
    for k in range(5):
        sub = carpeta / f"fold_{k}"
        sub.mkdir(parents=True)
        torch.manual_seed(k)
        torch.save({"estado": crear_modelo("cnn_v1").state_dict(), "epoca": 3 + k, "auc_val": 0.6,
                    "arquitectura": VERSIONES["cnn_v1"]}, sub / "mejor.pt")
        (sub / "resultados.json").write_text(json.dumps({
            "arquitectura": VERSIONES["cnn_v1"], "receta": {"aumento": None}, "parametros": 23761,
            "config": {"modo": "completo", "batch_size": 32}, "maquina": {"backend": "cpu"},
            "mejor_epoca": {"auc_val": 0.6 + 0.01 * k}}, default=list), encoding="utf-8")
        for p in range(pacientes_por_fold):
            y = int(p % 3 == 0)
            base = 0.25 + 0.3 * y
            for c in range(3):
                filas.append({"sample_id": f"F{k}P{p}_z{c:03d}", "patient_id": f"F{k}P{p}", "fold": k, "pCR": y,
                              "prob": float(np.clip(base + rng.normal(0, 0.15), 0.01, 0.99))})
        pd.DataFrame([f for f in filas if f["fold"] == k]).to_csv(sub / "predicciones_val.csv", index=False)
    pd.DataFrame(filas).to_csv(carpeta / "predicciones_oof.csv", index=False)
    md = {m: {"media": 0.62, "desviacion": 0.03} for m in ("auc_val", "auc_train", "diferencia_auc")}
    (carpeta / "resumen.json").write_text(json.dumps({"modelo": "cnn_v1", "perdida": "normal",
                                                       "media_desviacion": md}), encoding="utf-8")
    return carpeta


@pytest.fixture
def cerrado(tmp_path):
    carpeta = cv_falsa(tmp_path)
    eleccion = {"version": "cnn_v1", "motivo_version": "prueba", "perdida": "normal", "motivo_perdida": "prueba"}
    manifiesto = cerrar_modelo.cerrar(carpeta, eleccion, destino=tmp_path / "final",
                                      informe_md=tmp_path / "CIERRE.md", carpeta_figuras=tmp_path / "cierre",
                                      comprobar=False)
    return tmp_path, manifiesto


def test_umbral_cumple_la_especificidad_minima():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 300)
    p = np.clip(0.3 + 0.25 * y + rng.normal(0, 0.2, 300), 0, 1)
    elegido = umbral.umbral_por_especificidad(y, p)
    assert elegido["especificidad"] >= umbral.ESPECIFICIDAD_MINIMA
    # Ningún umbral con especificidad suficiente tiene más sensibilidad.
    for t in umbral.candidatos(p):
        m = metricas_con_umbral(y, p, t)
        if m["especificidad"] >= umbral.ESPECIFICIDAD_MINIMA:
            assert m["sensibilidad"] <= elegido["sensibilidad"]


def test_el_umbral_queda_en_el_centro_del_intervalo():
    # Cualquier umbral en (0,2; 0,8] separa perfectamente: se elige el centro, 0,5, no el borde 0,8.
    y, p = np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9])
    elegido = umbral.umbral_por_especificidad(y, p)
    assert elegido["umbral"] == pytest.approx(0.5)
    assert elegido["sensibilidad"] == 1 and elegido["especificidad"] == 1


def test_auc_con_ic_contiene_el_auc():
    rng = np.random.default_rng(2)
    y = rng.integers(0, 2, 200)
    p = 0.4 * y + rng.random(200)
    r = auc_con_ic(y, p, n=300)
    assert r["ic95"][0] <= r["auc"] <= r["ic95"][1]


def test_modelo_final_promedia_logits_y_calibra():
    torch.manual_seed(0)
    redes = [CNN(filtros=(4, 8)) for _ in range(2)]
    final = ModeloFinal({"filtros": [4, 8]}, [r.state_dict() for r in redes], platt_a=0.5, platt_b=-0.2)
    x = torch.rand(3, 3, 256, 256)
    with torch.no_grad():
        esperado = 0.5 * (redes[0].eval()(x) + redes[1].eval()(x)).squeeze(1) / 2 - 0.2
    assert torch.allclose(final(x), esperado, atol=1e-6)
    assert final.probabilidad(x).shape == (3,)


def test_cierre_exporta_y_se_puede_recargar(cerrado):
    raiz, manifiesto = cerrado
    assert len(manifiesto["pesos"]) == 5 and len(manifiesto["huella_modelo"]) == 12
    assert manifiesto["metricas_internas"]["fuera_de_fold"]["especificidad"] >= umbral.ESPECIFICIDAD_MINIMA
    assert (raiz / "CIERRE.md").exists() and (raiz / "cierre" / "cierre.png").exists()
    modelo, leido = cargar_modelo_final(raiz / "final")
    assert leido["huella_modelo"] == manifiesto["huella_modelo"]
    p = modelo.probabilidad(torch.rand(2, 3, 256, 256))
    assert p.shape == (2,) and bool(((p >= 0) & (p <= 1)).all())


def test_cargar_detecta_pesos_modificados(cerrado):
    raiz, _ = cerrado
    estado = torch.load(raiz / "final" / "fold_0.pt", weights_only=True)
    primera = next(iter(estado))
    estado[primera] = estado[primera] + 1
    torch.save(estado, raiz / "final" / "fold_0.pt")
    with pytest.raises(ValueError, match="sha256"):
        cargar_modelo_final(raiz / "final")


def test_evaluar_y_casos_fallados():
    rng = np.random.default_rng(3)
    pacientes = pd.DataFrame({"pCR": rng.integers(0, 2, 80), "cohorte": rng.choice(["spy2", "duke"], 80)},
                             index=[f"P{i}" for i in range(80)])
    pacientes["prob"] = np.clip(0.3 + 0.3 * pacientes.pCR + rng.normal(0, 0.2, 80), 0, 1)
    r = evaluar_test.evaluar(pacientes, umbral=0.5)
    assert r["ic95"][0] <= r["auc"] <= r["ic95"][1]
    assert sum(r["matriz_confusion"].values()) == 80 and set(r["por_cohorte"]) == {"spy2", "duke"}
    casos = evaluar_test.casos_fallados(pacientes, 0.5)
    assert set(casos.error) <= {"FN", "FP"}
    assert (casos[casos.error == "FN"].pCR == 1).all() and (casos[casos.error == "FP"].pCR == 0).all()


def test_test_se_evalua_una_sola_vez(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluar_test, "CARPETA_TEST", tmp_path)
    (tmp_path / "resultado_test.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["evaluar_test"])
    with pytest.raises(SystemExit, match="una sola vez"):
        evaluar_test.main()
