"""Los datos de la página «Resultados» de la app web (D39): `web/datos/resultados.json`.

Lo llama `src.models.exportar_web`. Todo sale de train, con predicciones fuera de fold; test no se usa.
Compara los tres modelos de la comparación clínica (D25):
- **cnn**: el modelo final (solo imagen), con sus predicciones fuera de fold del cierre;
- **medico**: edad, menopausia, HR y HER2, sin imagen;
- **combinado**: el médico más la puntuación de la CNN.

Qué se calcula, y por qué (el conjunto mínimo que recomiendan Van Calster et al., STRATOS, para evaluar
modelos de predicción clínicos, y TRIPOD+AI 2024):
- **discriminación**: curva ROC y AUC con su intervalo, también por fold, por subtipo y por hospital;
- **calibración**: probabilidad predicha frente a la frecuencia observada, por deciles, con intervalo;
- **utilidad clínica**: curva de decisión (beneficio neto frente a tratar a todas o a ninguna);
- **distribución del riesgo** de cada modelo en las pacientes con y sin pCR;
- **umbral**: las probabilidades y etiquetas fuera de fold (sin identificadores), para que la página calcule
  la matriz de confusión con cualquier umbral.
Además: las razones de odds del médico y del combinado, las curvas de entrenamiento de cada versión del
conjunto y el sesgo de la época elegida (R10, D37).

Los datos clínicos (`patients.csv`) solo se usan aquí para COMPARAR y para entender a la CNN (fases B y C,
D34); no entran en la CNN ni en lo que predice la app.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

from src.evaluation import modelo_clinico as mc
from src.evaluation.metricas import auc_con_ic, brier, metricas_con_umbral
from src.models.final import miembros

MODELOS = ("cnn", "medico", "combinado")
NOMBRES = {"cnn": "CNN (solo imagen)", "medico": "Médico: edad, menopausia, HR y HER2",
           "combinado": "CNN + médico"}
COHORTES = {"duke": "Duke", "spy1": "I-SPY1", "spy2": "I-SPY2"}
SUBTIPOS = {"HER2pos": "HER2+", "TripleNeg": "Triple negativo", "HRposHER2neg": "HR+/HER2−"}
VARIABLES = {"age": "Edad (10 años más)", "menopause": "Posmenopausia", "HR": "Receptores hormonales (HR+)",
             "HER2": "HER2+", "cnn": "CNN (una desviación más)"}
BIBLIOGRAFIA = [
    {"estudio": "Fridman et al. 2025, BreastDCEDL", "que": "ViT preentrenado (no permitido aquí), mismo test",
     "auc": 0.72, "conjunto": "test"},
    {"estudio": "Mejores marcadores de RM previa al tratamiento", "que": "volumen tumoral funcional y similares",
     "auc_min": 0.61, "auc": 0.62, "conjunto": "revisión (COMO_SE_HIZO §3.13)"},
    {"estudio": "Joo et al. 2021", "que": "solo clínica", "auc": 0.83, "conjunto": "un centro"},
    {"estudio": "Joo et al. 2021", "que": "imagen + clínica (deep learning)", "auc": 0.89, "conjunto": "un centro"},
    {"estudio": "Correo del profesor", "que": "«buen trabajo» con cortes 2D y tres cohortes, en test", "auc_min": 0.60,
     "auc": 0.70, "conjunto": "test"},
]


def _r(x, d: int = 6):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), d)


def curva_roc(y: np.ndarray, p: np.ndarray, maximo: int = 240) -> list[list[float]]:
    """[[1 − especificidad, sensibilidad, umbral], ...], aligerada a unos `maximo` puntos sin perder la forma."""
    fpr, tpr, umbrales = roc_curve(y, p)
    umbrales = np.clip(umbrales, 0, 1)
    if len(fpr) > maximo:
        idx = np.unique(np.concatenate([[0, len(fpr) - 1], np.linspace(0, len(fpr) - 1, maximo).round().astype(int)]))
        fpr, tpr, umbrales = fpr[idx], tpr[idx], umbrales[idx]
    return [[_r(a), _r(b), _r(c)] for a, b, c in zip(fpr, tpr, umbrales)]


def _wilson(k: int, n: int, z: float = 1.96) -> list[float]:
    if n == 0:
        return [None, None]
    f = k / n
    centro = (f + z * z / (2 * n)) / (1 + z * z / n)
    margen = z * np.sqrt(f * (1 - f) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [_r(centro - margen), _r(centro + margen)]


def calibracion(y: np.ndarray, p: np.ndarray, grupos: int = 10) -> list[dict]:
    """Por deciles de la probabilidad: media predicha, frecuencia observada (con IC de Wilson) y pacientes."""
    tabla = pd.DataFrame({"y": y, "p": p})
    tabla["grupo"] = pd.qcut(tabla.p.rank(method="first"), grupos, labels=False)
    salida = []
    for _, g in tabla.groupby("grupo"):
        k, n = int(g.y.sum()), len(g)
        salida.append({"predicha": _r(g.p.mean()), "observada": _r(k / n), "ic95": _wilson(k, n), "pacientes": n})
    return salida


def curva_decision(y: np.ndarray, p: np.ndarray, umbrales: np.ndarray) -> list[float]:
    """Beneficio neto con cada umbral: VP/n − FP/n · u/(1 − u) (Vickers y Elkin, 2006)."""
    n = len(y)
    salida = []
    for u in umbrales:
        positivo = p >= u
        vp, fp = int((positivo & (y == 1)).sum()), int((positivo & (y == 0)).sum())
        salida.append(_r(vp / n - fp / n * u / (1 - u), 5))
    return salida


def histograma(y: np.ndarray, p: np.ndarray, bordes: np.ndarray) -> dict:
    return {"pcr": np.histogram(p[y == 1], bordes)[0].tolist(), "no_pcr": np.histogram(p[y == 0], bordes)[0].tolist()}


def auc_grupos(tabla: pd.DataFrame, columna: str, nombres: dict, probs: dict[str, pd.Series], n: int) -> list[dict]:
    """AUC de cada modelo, con IC 95 % por bootstrap, dentro de cada grupo (subtipo o hospital)."""
    salida = []
    for clave, nombre in nombres.items():
        g = tabla[tabla[columna] == clave]
        if g.pCR.nunique() < 2:
            continue
        fila = {"grupo": nombre, "pacientes": len(g), "pcr": int(g.pCR.sum())}
        for m, p in probs.items():
            r = auc_con_ic(g.pCR.to_numpy(), p.loc[g.index].to_numpy(), n=n)
            fila[m] = {"auc": _r(r["auc"]), "ic95": [_r(v) for v in r["ic95"]]}
        salida.append(fila)
    return salida


def curvas_entrenamiento(manifiesto: dict, runs: Path) -> list[dict]:
    """Media y desviación, época a época y entre los 5 folds, de AUC y pérdida de cada versión del conjunto."""
    salida = []
    for m in miembros(manifiesto):
        carpeta = Path(runs) / str(m.get("validacion_cruzada", ""))
        historiales = [pd.read_csv(f) for f in sorted(carpeta.glob("fold_*/historial.csv"))]
        if not historiales:
            continue
        todo = pd.concat(historiales)
        por_epoca = todo.groupby("epoca")
        fila = {"version": m["version"], "folds": len(historiales), "epoca": [int(e) for e in por_epoca.groups]}
        for col, nombre in (("auc_tr", "auc_train"), ("auc", "auc_val"), ("perdida_tr", "perdida_train"),
                            ("perdida_va", "perdida_val")):
            if col in todo:
                fila[nombre] = {"media": [_r(v) for v in por_epoca[col].mean()],
                                "desviacion": [_r(v) for v in por_epoca[col].std(ddof=1).fillna(0)]}
        salida.append(fila)
    return salida


def sesgo_epoca(manifiesto: dict, historico: Path) -> list[dict]:
    """Cada versión del conjunto: AUC de validación con la época elegida (modo completo) y con 6 fijas."""
    if not Path(historico).exists():
        return []
    tabla = pd.read_csv(historico)
    tabla = tabla[(tabla.tipo == "cv") & (tabla.perdida == "normal") & (tabla.folds.astype(str) == "0,1,2,3,4")]
    salida = []
    for m in miembros(manifiesto):
        filas = {modo: tabla[(tabla.modelo == m["version"]) & (tabla.modo == modo)] for modo in ("completo", "fijo")}
        if any(f.empty for f in filas.values()):
            continue
        salida.append({"version": m["version"],
                       "elegida": _r(filas["completo"].iloc[-1].auc_val), "fija": _r(filas["fijo"].iloc[-1].auc_val)})
    return salida


def resultados(manifiesto: dict, carpeta_resultados: Path, runs: Path, n_bootstrap: int = 500) -> dict:
    """Todo lo de la página «Resultados». Si la CNN no está cerrada con este modelo, solo lo que no depende de ella."""
    oof = Path(carpeta_resultados) / "2_cierre" / "oof_pacientes.csv"
    cnn, _, motivo = _cnn_de(oof, manifiesto)
    salida = {"huella_modelo": manifiesto["huella_modelo"], "modelos": {k: NOMBRES[k] for k in MODELOS},
              "bibliografia": BIBLIOGRAFIA, "entrenamiento": curvas_entrenamiento(manifiesto, runs),
              "sesgo_epoca": sesgo_epoca(manifiesto, Path(runs) / "historico.csv")}
    if cnn is None:
        salida["aviso"] = motivo
        return salida

    tabla = mc.tabla_train()
    analisis = mc.analizar(tabla, cnn, n_bootstrap=min(n_bootstrap, 200))
    con = tabla.join(cnn.rename("prob_cnn"), how="inner")
    probs = {"cnn": con.prob_cnn, "medico": analisis["modelos"]["medico"]["prob"].loc[con.index],
             "combinado": analisis["modelos"]["combinado"]["prob"].loc[con.index]}
    y = con.pCR.to_numpy()
    umbrales_dca = np.round(np.arange(0.05, 0.71, 0.01), 2)
    bordes = np.linspace(0, 1, 41)
    modelos = {}
    for m, p in probs.items():
        r = analisis["modelos"][m]
        pp = p.to_numpy()
        umbral = (r.get("umbral") or {}).get("umbral")
        orden = np.argsort(pp)
        modelos[m] = {
            "nombre": NOMBRES[m],
            "auc_val": _r(r["auc_val"]), "auc_val_desviacion": _r(r["auc_val_desv"]),
            "auc_val_ic95": [_r(v) for v in r["auc_val_ic95"]],
            "auc_por_fold": [_r(f["auc_val"]) for f in r["por_fold"]],
            "brier": _r(brier(y, pp)),
            "auc_oof": _r(r["fuera_de_fold"]["auc"]), "auc_oof_ic95": [_r(v) for v in r["fuera_de_fold"]["ic95"]],
            "umbral": _r(umbral), "con_umbral": {k: _r(v) for k, v in metricas_con_umbral(y, pp, umbral).items()
                                                 if isinstance(v, (int, float))} if umbral is not None else None,
            "roc": curva_roc(y, pp),
            "calibracion": calibracion(y, pp),
            "decision": curva_decision(y, pp, umbrales_dca),
            "histograma": histograma(y, pp, bordes),
            "puntos": [[_r(pp[i]), int(y[i])] for i in orden],
        }
    c = analisis["comparacion"]

    def _dif(v: dict) -> dict:
        return {"diferencia": _r(v["diferencia"]), "ic95": [_r(x) for x in v["ic95"]],
                "por_fold": [_r(x) for x in v.get("por_fold", [])]}

    # La principal, con la misma cifra que se da de cada modelo (media de folds, D40); todas juntas, aparte
    pares = ("cnn_menos_medico", "combinado_menos_medico", "combinado_menos_cnn")
    comparacion = {k: _dif(c[k]) for k in pares}
    comparacion["todas_juntas"] = {k: _dif(c["todas_juntas"][k]) for k in pares}
    comparacion["estimador"] = c["estimador"]
    comparacion["correlacion_spearman"] = _r(c["correlacion_spearman"], 3)
    # La puntuación de la CNN entra como logit, que apenas varía (σ ≈ 0,1): su razón de odds «por 1 de logit»
    # sería engañosa. Se da por una desviación típica, como se hace con variables continuas de escala arbitraria.
    sd_cnn = float(np.std(mc.logit(con.prob_cnn.to_numpy()), ddof=1))
    escala = {"cnn": sd_cnn}
    odds = {m: [{"variable": VARIABLES.get(v, v), "clave": v, "or": _r(d["or"] ** escala.get(v, 1), 3),
                 "ic95": [_r(x ** escala.get(v, 1), 3) for x in d["ic95"]]}
                for v, d in analisis["razones_odds"][m].items()] for m in ("medico", "combinado")}
    salida.update({
        "pacientes": len(con), "pacientes_pcr": int(y.sum()), "tasa_pcr": _r(y.mean()),
        "brier_constante": _r(brier(y, np.full(len(y), y.mean()))),
        "validacion_interna": _validacion_interna(manifiesto, carpeta_resultados),
        "modelos_datos": modelos, "comparacion": comparacion, "razones_odds": odds,
        "decision_umbrales": umbrales_dca.tolist(),
        "decision_todas": curva_decision(y, np.ones_like(y, dtype=float), umbrales_dca),
        "histograma_bordes": [_r(b, 3) for b in bordes],
        "por_subtipo": auc_grupos(con, "subtipo", SUBTIPOS, probs, n_bootstrap),
        "por_hospital": auc_grupos(con, "dataset", COHORTES, probs, n_bootstrap),
        "auc_solo_hospital": _r(solo_hospital(con)),
    })
    return salida


def _validacion_interna(manifiesto: dict, carpeta_resultados: Path) -> dict | None:
    """Platt y umbral elegidos sin el fold que se mide (D40), si ya se calcularon para este modelo."""
    from src.evaluation.validacion_interna import leer

    v = leer(manifiesto, Path(carpeta_resultados) / "2_cierre" / "validacion_interna.json")
    if v is None:
        return None
    resumen = {"brier_platt_cruzado": _r(v["brier"]["platt_cruzado"]), "brier_platt_mismas": _r(v["brier"]["platt_mismas"]),
               "cnn_mismas": {k: _r(v["umbral"]["mismas"][k]) for k in ("sensibilidad", "especificidad", "precision_pcr")},
               "cnn_cruzado": {k: _r(v["umbral"]["cruzado"][k]) for k in ("sensibilidad", "especificidad", "precision_pcr")}}
    if v.get("medico"):
        resumen["medico_cruzado"] = {k: _r(v["medico"]["umbral"]["cruzado"][k])
                                     for k in ("sensibilidad", "especificidad", "precision_pcr")}
    return resumen


def solo_hospital(tabla: pd.DataFrame) -> float:
    """AUC (media de los 5 folds) de saber solo el hospital: la tasa de pCR de su hospital en los otros folds (R0)."""
    aucs = []
    for k, g in tabla.groupby("fold"):
        tasa = tabla[tabla.fold != k].groupby("dataset").pCR.mean()
        aucs.append(roc_auc_score(g.pCR, g.dataset.map(tasa)))
    return float(np.mean(aucs))


def _cnn_de(oof: Path, manifiesto: dict):
    """Las probabilidades fuera de fold de la CNN, si el cierre es de ESTE modelo."""
    cierre = oof.parent / "cierre.json"
    if not oof.exists() or not cierre.exists():
        return None, None, "faltan las predicciones fuera de fold del cierre (cerrar_modelo)"
    if json.loads(cierre.read_text(encoding="utf-8")).get("huella_modelo") != manifiesto["huella_modelo"]:
        return None, None, "el cierre de docs/resultados es de otro modelo: vuelve a ejecutar cerrar_modelo"
    return pd.read_csv(oof, index_col="patient_id").prob, manifiesto["huella_modelo"], "ok"


def escribir_resultados(manifiesto: dict, ruta: Path, carpeta_resultados: Path, runs: Path,
                        n_bootstrap: int = 500) -> dict:
    datos = resultados(manifiesto, carpeta_resultados, runs, n_bootstrap)
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text(json.dumps(datos, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    return datos
