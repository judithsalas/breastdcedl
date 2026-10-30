"""Modelo clínico de comparación (D25): lo que sabría un médico sin mirar la resonancia.

    python -m src.evaluation.modelo_clinico

Responde a una pregunta del profesor: si es diagnóstico por imagen, ¿qué peso tiene la imagen?
Es decir, ¿la CNN aporta algo que el médico no sepa ya antes de la quimioterapia? Las variables
de patients.csv sirven "de comparación, no de modelo principal" (correo del profesor): este modelo
**no entra en la CNN ni en la app**, es un listón aparte.

Qué sabe el "médico": lo que se conoce antes del tratamiento por la historia clínica y la biopsia.
- edad, menopausia, receptores hormonales (HR) y HER2.
- El dataset NO trae análisis de sangre, peso, IMC, tabaco ni alcohol (guía, D4).

Qué NO se usa y por qué: tabla NO_SE_USAN (raza, cohorte, adquisición, máscara y volumen).
El volumen tumoral sale de la segmentación de la propia imagen y en Duke tiene otra escala:
solo se prueba como variante aparte ("médico + volumen"), sin llevarla a test.

Cómo:
- Regresión logística con las variables estandarizadas. Lo que falta se completa con lo aprendido
  en los folds de entrenamiento: la mediana; y la menopausia, que falta entera en I-SPY1, con la
  probabilidad de estar en la menopausia según la edad. Sin indicador de "falta el dato": en I-SPY1
  delataría la cohorte.
- Los mismos 5 folds que la CNN: predicciones fuera de fold, AUC medio ± desviación.
- Con la CNN ya cerrada (cerrar_modelo): comparación con ella, con la diferencia de AUC y su
  intervalo por bootstrap PAREADO de pacientes, y un modelo combinado CNN + médico (¿suma algo la
  imagen a lo que ya sabe el médico?).
- Test no se toca aquí: se evalúa una sola vez, junto con la CNN, en evaluar_test.

Salida: experimentos/modelo_clinico/modelo_clinico.json (parámetros en JSON, sin pickle; lo usa evaluar_test) y
docs/resultados/3_clinico/ (MODELO_CLINICO.md con sus figuras al lado).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve

from src.data.caso import NOMBRE_COHORTE, RAIZ_REPO, RESULTADOS, uc
from src.evaluation.metricas import auc_con_ic, auc_media_folds, coma, diferencia_auc, diferencia_media_folds
from src.evaluation.umbral import ESPECIFICIDAD_MINIMA, umbral_por_especificidad

# --------------------------------------------------------------------------- #
# Qué variables
# --------------------------------------------------------------------------- #

VARIABLES_MEDICO = ("age", "menopause", "HR", "HER2")
VARIANTE_VOLUMEN = (*VARIABLES_MEDICO, "tum_vol")
CNN = "cnn"                                   # logit de la probabilidad de la CNN por paciente
VARIABLES_COMBINADO = (CNN, *VARIABLES_MEDICO)

NO_SE_USAN = {
    "race_white, race_black": "raza: tratarla como causa es un error metodológico (guía, D4)",
    "dataset": "cohorte: es del dataset, no de la paciente (R0 mide cuánto acierta sola)",
    "n_xy, n_z, n_times, pre, post_early, post_late, slice_thick, xy_spacing":
        "adquisición de la resonancia: no es información clínica",
    "mask_start, mask_end, sraw, eraw, scol, ecol": "localización de la máscara: sale de la imagen",
    "tum_vol": ("volumen tumoral: sale de la segmentación de la imagen y en Duke tiene otra escala; "
                "solo como variante aparte"),
    "HR_HER2_STATUS, TripleNeg, HER2pos, HRposHER2neg": "son combinaciones de HR y HER2, que ya están",
}

NOMBRE_VARIABLE = {"age": "Edad", "menopause": "Posmenopausia", "HR": "Receptores hormonales (HR+)",
                   "HER2": "HER2+", "tum_vol": "Volumen tumoral", CNN: "Puntuación de la CNN"}
# Cuánto cambia la variable en la razón de odds que se muestra.
PASO = {"age": (10.0, "10 años más"), "menopause": (1.0, "sí frente a no"), "HR": (1.0, "positivo frente a negativo"),
        "HER2": (1.0, "positivo frente a negativo"), "tum_vol": (10.0, "10 unidades más"),
        CNN: (1.0, "1 punto más de logit")}
NOMBRE_MODELO = {"cohorte": "Solo la cohorte (R0)", "subtipo": "Solo el subtipo HR/HER2 (R0)",
                 "medico": "Médico: edad, menopausia, HR, HER2", "medico_volumen": "Médico + volumen tumoral",
                 "cnn": "CNN (solo imagen)", "combinado": "CNN + médico"}

C_REGULARIZACION = 1.0        # L2 suave; con ~1.000 pacientes y 4 variables casi no influye
EPS = 1e-6

CARPETA_MODELO = RAIZ_REPO / "experimentos" / "modelo_clinico"
MODELO_JSON = CARPETA_MODELO / "modelo_clinico.json"
FIGURAS = RESULTADOS / "3_clinico"
INFORME = FIGURAS / "MODELO_CLINICO.md"
OOF_CNN = RESULTADOS / "2_cierre" / "oof_pacientes.csv"
MANIFIESTO_CNN = RAIZ_REPO / "experimentos" / "modelo_final" / "manifiesto.json"
AZUL, NARANJA, VERDE, GRIS = "#2a78d6", "#eb6834", "#2f9e6b", "#8a8984"
TINTA, TINTA_SUAVE, REJILLA = "#0b0b0b", "#52514e", "#e4e3df"


def sigmoide(z):
    return 1 / (1 + np.exp(-np.asarray(z, float)))


def logit(p):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


# --------------------------------------------------------------------------- #
# El modelo: regresión logística que se guarda en JSON
# --------------------------------------------------------------------------- #

@dataclass
class ModeloLogistico:
    """Imputación + estandarización + regresión logística, con todo aprendido en train.

    Se guarda como números (a_dict) y predice con numpy: no hace falta pickle para usarlo.
    """

    variables: list[str]
    medianas: dict[str, float]
    menopausia_por_edad: list[float] | None   # [a, b]: P(posmenopausia) = sigmoide(a · edad + b)
    medias: dict[str, float]
    desviaciones: dict[str, float]
    coeficientes: dict[str, float]
    intercepto: float
    umbral: float | None = None

    @classmethod
    def ajustar(cls, tabla: pd.DataFrame, variables=VARIABLES_MEDICO, c: float = C_REGULARIZACION) -> ModeloLogistico:
        variables = list(variables)
        if "menopause" in variables and "age" not in variables:
            raise ValueError("La menopausia se completa con la edad: hace falta la edad entre las variables.")
        medianas = {v: float(tabla[v].median()) for v in variables}
        menopausia = None
        if "menopause" in variables:
            ambas = tabla[["age", "menopause"]].dropna()
            media, desv = float(ambas.age.mean()), float(ambas.age.std(ddof=0)) or 1.0
            reg = LogisticRegression(C=1e6, max_iter=1000).fit(((ambas[["age"]] - media) / desv).to_numpy(),
                                                                ambas.menopause.astype(int))
            a = float(reg.coef_[0, 0]) / desv                      # de vuelta a años
            menopausia = [a, float(reg.intercept_[0]) - a * media]
        provisional = cls(variables, medianas, menopausia, {}, {}, {}, 0.0)
        x = provisional.completar(tabla)
        medias = {v: float(x[v].mean()) for v in variables}
        desviaciones = {v: float(x[v].std(ddof=0)) or 1.0 for v in variables}
        z = (x - pd.Series(medias)) / pd.Series(desviaciones)
        reg = LogisticRegression(C=c, max_iter=1000).fit(z[variables].to_numpy(), tabla.pCR.astype(int))
        return cls(variables, medianas, menopausia, medias, desviaciones,
                   {v: float(b) for v, b in zip(variables, reg.coef_[0], strict=True)}, float(reg.intercept_[0]))

    def completar(self, tabla: pd.DataFrame) -> pd.DataFrame:
        """Las variables con los datos que faltan completados (con lo aprendido al ajustar)."""
        x = tabla[self.variables].astype(float).copy()
        if "age" in x:
            x["age"] = x.age.fillna(self.medianas["age"])
        if "menopause" in x and self.menopausia_por_edad is not None:
            a, b = self.menopausia_por_edad
            x["menopause"] = x.menopause.fillna(pd.Series(sigmoide(a * x.age + b), index=x.index))
        return x.fillna(self.medianas)

    def logit(self, tabla: pd.DataFrame) -> np.ndarray:
        x = self.completar(tabla)
        z = (x - pd.Series(self.medias)) / pd.Series(self.desviaciones)
        return z[self.variables].to_numpy() @ np.array([self.coeficientes[v] for v in self.variables]) + self.intercepto

    def probabilidad(self, tabla: pd.DataFrame) -> np.ndarray:
        return sigmoide(self.logit(tabla))

    def razones_de_odds(self) -> dict[str, float]:
        """Por cuánto se multiplica la odds de pCR al cambiar cada variable en su PASO."""
        return {v: float(np.exp(self.coeficientes[v] / self.desviaciones[v] * PASO[v][0])) for v in self.variables}

    def a_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def desde_dict(cls, datos: dict) -> ModeloLogistico:
        return cls(**datos)


def huella(datos) -> str:
    return hashlib.sha1(json.dumps(datos, sort_keys=True).encode()).hexdigest()[:8]


# --------------------------------------------------------------------------- #
# Datos
# --------------------------------------------------------------------------- #

def tabla_train(samples: pd.DataFrame | None = None, patients: pd.DataFrame | None = None) -> pd.DataFrame:
    """Una fila por paciente de train con su fold, cohorte, subtipo, etiqueta y variables clínicas."""
    from src.data.auditoria import pacientes_train_con_fold

    samples = uc.cargar_samples() if samples is None else samples
    patients = uc.cargar_patients() if patients is None else patients
    return pacientes_train_con_fold(samples, patients)


def ausentes_por_cohorte(tabla: pd.DataFrame, variables=VARIANTE_VOLUMEN) -> dict:
    """Porcentaje de datos que faltan, por cohorte y variable."""
    return {str(c): {v: float(100 * g[v].isna().mean()) for v in variables} for c, g in tabla.groupby("dataset")}


def cnn_cerrada(oof: Path = OOF_CNN, manifiesto: Path = MANIFIESTO_CNN) -> tuple[pd.Series | None, str | None, str]:
    """Probabilidad fuera de fold de la CNN por paciente (calibrada, media de cortes), si ya está cerrada.

    Devuelve (probabilidades, huella del modelo final, motivo). La tabla la deja cerrar_modelo en
    docs/resultados/2_cierre/oof_pacientes.csv; se comprueba que corresponde al modelo de experimentos/modelo_final/.
    """
    if not manifiesto.exists():
        return None, None, "la CNN aún no está cerrada (falta experimentos/modelo_final/manifiesto.json)"
    huella_cnn = json.loads(manifiesto.read_text(encoding="utf-8"))["huella_modelo"]
    cierre = oof.parent / "cierre.json"
    if not oof.exists() or not cierre.exists():
        return None, huella_cnn, f"faltan {oof.name} o cierre.json en docs/resultados/2_cierre: vuelve a ejecutar cerrar_modelo"
    if json.loads(cierre.read_text(encoding="utf-8")).get("huella_modelo") != huella_cnn:
        return None, huella_cnn, "docs/resultados/2_cierre no corresponde al modelo de experimentos/modelo_final: vuelve a ejecutar cerrar_modelo"
    return pd.read_csv(oof, index_col="patient_id").prob, huella_cnn, "ok"


# --------------------------------------------------------------------------- #
# Validación cruzada y comparación (solo train)
# --------------------------------------------------------------------------- #

def resumen(tabla: pd.DataFrame, prob: pd.Series, folds: list[dict]) -> dict:
    """AUC medio ± desviación de los folds (la cifra principal, con su IC), AUC con todas las pacientes juntas
    (con IC) y umbral con el criterio de la CNN."""
    y, p = tabla.pCR.to_numpy(), prob.loc[tabla.index].to_numpy()
    val = [f["auc_val"] for f in folds]
    train = [f["auc_train"] for f in folds if f.get("auc_train") is not None]
    salida = {"prob": prob.loc[tabla.index], "por_fold": folds, "auc_val": float(np.mean(val)),
              "auc_val_desv": float(np.std(val, ddof=1)),
              "auc_val_ic95": auc_media_folds(y, p, tabla.fold.to_numpy())["ic95"],
              "auc_train": float(np.mean(train)) if train else None,
              "fuera_de_fold": auc_con_ic(y, p), "umbral": umbral_por_especificidad(y, p),
              "pacientes": len(y)}
    salida["diferencia"] = None if salida["auc_train"] is None else salida["auc_train"] - salida["auc_val"]
    return salida


def validacion_cruzada(tabla: pd.DataFrame, variables) -> dict:
    """Cada paciente, predicha por el modelo ajustado SIN su fold (imputación incluida)."""
    oof = pd.Series(np.nan, index=tabla.index)
    folds = []
    for k in sorted(tabla.fold.unique()):
        entrena, valida = tabla[tabla.fold != k], tabla[tabla.fold == k]
        modelo = ModeloLogistico.ajustar(entrena, variables)
        oof.loc[valida.index] = modelo.probabilidad(valida)
        folds.append({"fold": int(k), "auc_val": float(roc_auc_score(valida.pCR, oof.loc[valida.index])),
                      "auc_train": float(roc_auc_score(entrena.pCR, modelo.probabilidad(entrena)))})
    return resumen(tabla, oof, folds)


def por_folds(tabla: pd.DataFrame, prob: pd.Series) -> dict:
    """Resumen de unas predicciones ya hechas fuera de fold (las de la CNN)."""
    folds = [{"fold": int(k), "auc_val": float(roc_auc_score(g.pCR, prob.loc[g.index]))}
             for k, g in tabla.groupby("fold")]
    return resumen(tabla, prob, folds)


def razones_de_odds(tabla: pd.DataFrame, variables, n: int = 500, semilla: int = 42) -> dict:
    """Razones de odds del modelo ajustado con todo train, con IC 95 % por bootstrap de pacientes."""
    final = ModeloLogistico.ajustar(tabla, variables)
    rng = np.random.default_rng(semilla)
    muestras = []
    for _ in range(n):
        i = rng.integers(0, len(tabla), len(tabla))
        muestras.append(ModeloLogistico.ajustar(tabla.iloc[i].reset_index(drop=True), variables).razones_de_odds())
    return {v: {"or": final.razones_de_odds()[v], "ic95": [float(np.quantile([m[v] for m in muestras], q))
                                                         for q in (0.025, 0.975)], "paso": PASO[v][1]}
            for v in final.variables}


def auc_por_grupo(tabla: pd.DataFrame, columna: str, probs: dict[str, pd.Series]) -> dict:
    """AUC de cada modelo dentro de cada grupo (subtipo o cohorte)."""
    salida = {}
    for grupo, g in tabla.groupby(columna):
        fila = {"pacientes": len(g), "pcr": int(g.pCR.sum())}
        for nombre, p in probs.items():
            fila[nombre] = float(roc_auc_score(g.pCR, p.loc[g.index])) if g.pCR.nunique() == 2 else float("nan")
        salida[str(grupo)] = fila
    return salida


def analizar(tabla: pd.DataFrame, cnn: pd.Series | None = None, referencias: dict | None = None,
             n_bootstrap: int = 500) -> dict:
    """Todo el análisis con train: modelos clínicos y, si está la CNN, comparación y combinado."""
    modelos = {"medico": validacion_cruzada(tabla, VARIABLES_MEDICO),
               "medico_volumen": validacion_cruzada(tabla, VARIANTE_VOLUMEN)}
    medico = ModeloLogistico.ajustar(tabla, VARIABLES_MEDICO)
    medico.umbral = modelos["medico"]["umbral"]["umbral"]
    analisis = {"pacientes": len(tabla), "pacientes_pcr": int(tabla.pCR.sum()), "ausentes": ausentes_por_cohorte(tabla),
                "referencias": referencias or {}, "modelos": modelos, "finales": {"medico": medico},
                "razones_odds": {"medico": razones_de_odds(tabla, VARIABLES_MEDICO, n_bootstrap)},
                "por_cohorte": auc_por_grupo(tabla, "dataset", {"medico": modelos["medico"]["prob"]})}
    if cnn is None:
        return analisis

    con = tabla.join(cnn.rename("prob_cnn"), how="inner")
    if len(con) < len(tabla):
        print(f"Aviso: {len(tabla) - len(con)} pacientes de train sin predicción de la CNN; la comparación usa {len(con)}.")
    con[CNN] = logit(con.prob_cnn)
    modelos["cnn"] = por_folds(con, con.prob_cnn)
    modelos["combinado"] = validacion_cruzada(con, VARIABLES_COMBINADO)
    combinado = ModeloLogistico.ajustar(con, VARIABLES_COMBINADO)
    combinado.umbral = modelos["combinado"]["umbral"]["umbral"]
    analisis["finales"]["combinado"] = combinado
    analisis["razones_odds"]["combinado"] = razones_de_odds(con, VARIABLES_COMBINADO, n_bootstrap)

    y = con.pCR.to_numpy()
    p_medico = modelos["medico"]["prob"].loc[con.index]
    p_cnn, p_comb = con.prob_cnn, modelos["combinado"]["prob"]
    # La comparación usa la misma cifra que se da de cada modelo, la media de los folds (D40): la diferencia
    # cuadra con las cifras de la tabla. Con todas las pacientes juntas se da aparte, como comprobación.
    folds = con.fold.to_numpy()
    pares = {"cnn_menos_medico": (p_cnn, p_medico), "combinado_menos_medico": (p_comb, p_medico),
             "combinado_menos_cnn": (p_comb, p_cnn)}
    analisis["comparacion"] = {
        "pacientes": len(con),
        "estimador": "media de los 5 folds, bootstrap pareado de pacientes dentro de cada fold",
        **{k: diferencia_media_folds(y, a.to_numpy(), b.to_numpy(), folds) for k, (a, b) in pares.items()},
        "todas_juntas": {k: diferencia_auc(y, a.to_numpy(), b.to_numpy()) for k, (a, b) in pares.items()},
        "correlacion_spearman": float(spearmanr(p_cnn, p_medico).statistic),
    }
    probs = {"cnn": p_cnn, "medico": p_medico, "combinado": p_comb}
    analisis["por_cohorte"] = auc_por_grupo(con, "dataset", probs)
    analisis["por_subtipo"] = auc_por_grupo(con[con.subtipo != "desconocido"], "subtipo", {"cnn": p_cnn})
    return analisis


# --------------------------------------------------------------------------- #
# Exportar y cargar (lo usa evaluar_test)
# --------------------------------------------------------------------------- #

def exportar(analisis: dict, huella_cnn: str | None, destino: Path = MODELO_JSON) -> dict:
    finales = analisis["finales"]
    datos = {
        "fecha": datetime.now().astimezone().isoformat(timespec="seconds"),
        "uso": "solo comparación (D25): no entra en la CNN ni en la app",
        "criterio_umbral": f"mayor sensibilidad con especificidad ≥ {ESPECIFICIDAD_MINIMA:.2f}, fuera de fold",
        "medico": finales["medico"].a_dict(),
        "combinado": finales["combinado"].a_dict() if "combinado" in finales else None,
        "huella_cnn": huella_cnn if "combinado" in finales else None,
        "validacion": {nombre: {"auc_val": m["auc_val"], "auc_val_desv": m["auc_val_desv"],
                                "auc_train": m["auc_train"], "pacientes": m["pacientes"]}
                       for nombre, m in analisis["modelos"].items()},
    }
    datos["huella"] = huella({k: datos[k] for k in ("medico", "combinado", "huella_cnn")})
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(json.dumps(datos, indent=2, ensure_ascii=False), encoding="utf-8")
    return datos


def cargar(ruta: Path = MODELO_JSON) -> dict:
    """El modelo guardado, con los modelos ya reconstruidos. FileNotFoundError si no existe."""
    datos = json.loads(Path(ruta).read_text(encoding="utf-8"))
    datos["medico"] = ModeloLogistico.desde_dict(datos["medico"])
    if datos.get("combinado"):
        datos["combinado"] = ModeloLogistico.desde_dict(datos["combinado"])
    return datos


def predecir(clinicos: pd.DataFrame, prob_cnn: pd.Series, guardado: dict) -> pd.DataFrame:
    """Probabilidad del médico y del combinado para las pacientes de `prob_cnn` (índice = pid)."""
    datos = clinicos.loc[prob_cnn.index]
    salida = pd.DataFrame({"prob_medico": guardado["medico"].probabilidad(datos)}, index=prob_cnn.index)
    if guardado.get("combinado"):
        salida["prob_combinado"] = guardado["combinado"].probabilidad(datos.assign(**{CNN: logit(prob_cnn)}))
    return salida


# --------------------------------------------------------------------------- #
# Figuras e informe
# --------------------------------------------------------------------------- #

def _estilo(ax, titulo, xlabel, ylabel):
    ax.set_title(titulo, fontsize=10.5, fontweight="bold", color=TINTA, loc="left")
    ax.set_xlabel(xlabel, color=TINTA_SUAVE)
    ax.set_ylabel(ylabel, color=TINTA_SUAVE)
    ax.grid(True, color=REJILLA, linewidth=0.8)
    ax.set_axisbelow(True)
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    for lado in ("left", "bottom"):
        ax.spines[lado].set_color(REJILLA)
    ax.tick_params(colors=TINTA_SUAVE, length=0)


def figura_roc(curvas: list[tuple[str, np.ndarray, np.ndarray, str, str]], titulo: str, ruta: Path) -> Path:
    """curvas: (nombre, y, p, color, estilo de línea)."""
    fig, ax = plt.subplots(figsize=(5.4, 4.8))
    for nombre, y, p, color, linea in curvas:
        fpr, tpr, _ = roc_curve(y, p)
        ax.plot(fpr, tpr, color=color, lw=2, ls=linea, label=f"{nombre} · AUC {coma(roc_auc_score(y, p), 3)}")
    ax.plot([0, 1], [0, 1], color=TINTA_SUAVE, lw=0.8, ls="--")
    ax.axvline(1 - ESPECIFICIDAD_MINIMA, color=GRIS, lw=0.8, ls=":")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    _estilo(ax, titulo, "1 − especificidad", "Sensibilidad")
    formato = matplotlib.ticker.FuncFormatter(lambda v, _: coma(v, 1))
    ax.xaxis.set_major_formatter(formato)
    ax.yaxis.set_major_formatter(formato)
    fig.savefig(ruta, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return ruta


def figura_odds(razones: dict, ruta: Path) -> Path:
    paneles = [(n, razones[n]) for n in ("medico", "combinado") if n in razones]
    fig, ejes = plt.subplots(1, len(paneles), figsize=(5.6 * len(paneles), 3.2), squeeze=False)
    for ax, (nombre, r) in zip(ejes[0], paneles, strict=True):
        variables = list(r)[::-1]
        for i, v in enumerate(variables):
            lo, hi = r[v]["ic95"]
            ax.plot([lo, hi], [i, i], color=AZUL, lw=2)
            ax.scatter([r[v]["or"]], [i], color=AZUL, s=36, zorder=3)
        ax.axvline(1, color=TINTA_SUAVE, lw=0.8, ls="--")
        ax.set_xscale("log")
        extremos = [x for v in variables for x in r[v]["ic95"]]
        marcas = [m for m in (0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16, 32)
                  if min(extremos) / 1.6 <= m <= max(extremos) * 1.6 or m == 1]
        ax.set_xlim(min(*extremos, 1) / 1.3, max(*extremos, 1) * 1.3)
        ax.xaxis.set_major_locator(matplotlib.ticker.FixedLocator(marcas))
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: coma(v, 2).rstrip("0").rstrip(",")))
        ax.set_yticks(range(len(variables)), [f"{NOMBRE_VARIABLE[v]}\n({r[v]['paso']})" for v in variables], fontsize=8)
        _estilo(ax, f"Razón de odds de pCR · {NOMBRE_MODELO[nombre]}", "Razón de odds (escala log; 1 = no influye)", "")
        ax.grid(False, axis="y")
    fig.tight_layout()
    fig.savefig(ruta, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return ruta


def figuras(analisis: dict, tabla: pd.DataFrame, carpeta: Path = FIGURAS) -> list[Path]:
    carpeta.mkdir(parents=True, exist_ok=True)
    m = analisis["modelos"]
    y = tabla.pCR
    curvas = [("Médico", y.to_numpy(), m["medico"]["prob"].to_numpy(), AZUL, "-"),
              ("Médico + volumen", y.to_numpy(), m["medico_volumen"]["prob"].to_numpy(), GRIS, ":")]
    if "cnn" in m:
        indice = m["cnn"]["prob"].index
        curvas += [("CNN", y.loc[indice].to_numpy(), m["cnn"]["prob"].to_numpy(), NARANJA, "-"),
                   ("CNN + médico", y.loc[indice].to_numpy(), m["combinado"]["prob"].to_numpy(), VERDE, "-")]
    return [figura_roc(curvas, "ROC por paciente, fuera de fold (train)", carpeta / "roc_fuera_de_fold.png"),
            figura_odds(analisis["razones_odds"], carpeta / "razones_de_odds.png")]


def _ic(v: dict, d: int = 3) -> str:
    return f"{coma(v['auc'], d)} ({coma(v['ic95'][0], 2)}–{coma(v['ic95'][1], 2)})"


def _dif(v: dict) -> str:
    return f"{coma(v['diferencia'], 3)} (IC 95 % {coma(v['ic95'][0], 3)} a {coma(v['ic95'][1], 3)})"


def informe(analisis: dict, guardado: dict, motivo_cnn: str) -> str:
    m, aus = analisis["modelos"], analisis["ausentes"]
    lineas = [
        "# Modelo clínico de comparación (sin imagen)", "",
        (f"Generado por `python -m src.evaluation.modelo_clinico` el {guardado['fecha'][:16].replace('T', ' ')}. "
         "Solo train, con"),
        ("los mismos 5 folds que la CNN; test no se ha usado (se evalúa una sola vez, junto con la CNN, en "
         "`evaluar_test`)."),
        "",
        ("**Para qué sirve.** Es lo que sabría un médico antes de la quimioterapia sin mirar la resonancia. "
         "Si la CNN no lo"),
        ("supera, la imagen (tal como la usamos) aporta menos que la biopsia. **No entra en la CNN ni en la app**: "
         "el profesor"), "pide estas variables como comparación, no como modelo principal.", "",
        "## 1. Qué sabe el médico", "",
        "| Variable | Para qué | % que falta en train: Duke · I-SPY1 · I-SPY2 |", "|---|---|---|"]
    para = {"age": "edad de la paciente", "menopause": "estado menopáusico (1 = posmenopausia)",
            "HR": "receptores hormonales en la biopsia", "HER2": "HER2 en la biopsia",
            "tum_vol": "solo en la variante (sale de la imagen)"}
    for v in VARIANTE_VOLUMEN:
        lineas.append(f"| `{v}` · {NOMBRE_VARIABLE[v]} | {para[v]} | "
                      + " · ".join(f"{coma(aus.get(c, {}).get(v, float('nan')), 1)} %" for c in ("duke", "spy1", "spy2"))
                      + " |")
    lineas += ["", ("Lo que falta se completa con lo aprendido en los folds de entrenamiento: la mediana; la "
                    "menopausia, con la"),
               ("probabilidad de posmenopausia según la edad (falta entera en I-SPY1). Sin indicador de «falta el "
                "dato», que en"), "I-SPY1 delataría la cohorte.", "",
               "**El dataset no trae análisis de sangre, peso, IMC, tabaco ni alcohol** (guía). Lo que no se usa:", "",
               "| Columnas | Por qué no |", "|---|---|"]
    lineas += [f"| `{c}` | {motivo} |" for c, motivo in NO_SE_USAN.items()]

    lineas += ["", (f"## 2. Resultados en validación cruzada ({analisis['pacientes']} pacientes, "
                    f"{analisis['pacientes_pcr']} con pCR)"), "",
               ("| Modelo | AUC validación (media de 5 folds) | IC 95 % | AUC entrenamiento | Diferencia | "
                "Todas juntas (IC 95 %) | Sensibilidad | Especificidad |"),
               "|---|---|---|---|---|---|---|---|"]
    for clave in ("cohorte", "subtipo"):
        if clave in analisis["referencias"]:
            r = analisis["referencias"][clave]
            lineas.append(f"| {NOMBRE_MODELO[clave]} | {coma(r['media'], 3)} ± {coma(r['desviacion'], 3)} | — | — | — | — | — "
                          "| — |")
    for clave in ("medico", "medico_volumen", "cnn", "combinado"):
        if clave not in m:
            continue
        r, u = m[clave], m[clave]["umbral"]
        tr = "—" if r["auc_train"] is None else coma(r["auc_train"], 3)
        di = "—" if r["diferencia"] is None else coma(r["diferencia"], 3)
        lineas.append(f"| {'**' + NOMBRE_MODELO[clave] + '**' if clave == 'medico' else NOMBRE_MODELO[clave]} | "
                      f"{coma(r['auc_val'], 3)} ± {coma(r['auc_val_desv'], 3)} | "
                      f"{coma(r['auc_val_ic95'][0], 2)}–{coma(r['auc_val_ic95'][1], 2)} | {tr} | {di} | "
                      f"{_ic(r['fuera_de_fold'])} | {coma(u['sensibilidad'], 3)} | {coma(u['especificidad'], 3)} |")
    lineas += ["", ("La cifra principal es la media de los 5 folds (rúbrica, punto 2); su IC 95 % sale de remuestrear "
                    "pacientes dentro de cada fold."),
               ("«Todas juntas» es el AUC de las predicciones fuera de fold de todas las pacientes a la vez: es otro "
                "estimador y se da solo como comprobación."),
               ("Sensibilidad y especificidad con el umbral de cada modelo elegido con el mismo criterio que la "
                f"CNN (mayor sensibilidad con especificidad ≥ {coma(ESPECIFICIDAD_MINIMA, 2)}), y medidas en las "
                "mismas pacientes con las que se eligió: son optimistas (validación interna en "
                "`docs/resultados/2_cierre/VALIDACION_INTERNA.md`)."),
               ("El volumen se prueba solo como variante: sale de la segmentación de la imagen y en Duke tiene "
                "otra escala, así que no es «lo que sabe el médico» y no se lleva a test."), "",
               "![ROC fuera de fold](roc_fuera_de_fold.png)", "",
               "## 3. Qué pesa cada variable (modelo ajustado con todo train)", "",
               "| Variable | Cambio | Razón de odds de pCR (IC 95 %, bootstrap) |", "|---|---|---|"]
    for v, r in analisis["razones_odds"]["medico"].items():
        lineas.append(f"| {NOMBRE_VARIABLE[v]} | {r['paso']} | {coma(r['or'], 2)} ({coma(r['ic95'][0], 2)}–"
                      f"{coma(r['ic95'][1], 2)}) |")
    lineas += ["", ("Razón de odds > 1: más probable la pCR; < 1: menos; si el intervalo cruza el 1, no se puede "
                    "afirmar que influya. Son asociaciones en este dataset, no causas."), "",
               "![Razones de odds](razones_de_odds.png)", ""]

    if "comparacion" in analisis:
        c = analisis["comparacion"]
        j = c.get("todas_juntas", {})
        lineas += [f"## 4. CNN frente al médico ({c['pacientes']} pacientes, fuera de fold)", "",
                   ("| Comparación | Diferencia de AUC, media de folds (IC 95 %, bootstrap pareado) | Por fold | "
                    "Todas juntas (comprobación) |"), "|---|---|---|---|"]
        for clave, etiqueta in (("cnn_menos_medico", "CNN − médico"),
                                ("combinado_menos_medico", "(CNN + médico) − médico: lo que **añade la imagen**"),
                                ("combinado_menos_cnn", "(CNN + médico) − CNN: lo que añade la clínica")):
            fold = " · ".join(coma(v, 3) for v in c[clave].get("por_fold", []))
            lineas.append(f"| {etiqueta} | {_dif(c[clave])} | {fold} | {_dif(j[clave]) if clave in j else '—'} |")
        lineas += ["", ("La diferencia es la resta de las cifras principales de la tabla anterior (media de folds), así "
                        "que cuadra con ellas."), "",
                   (f"Correlación de Spearman entre la puntuación de la CNN y la del médico: "
                    f"{coma(c['correlacion_spearman'], 2)}. Si fuera alta, la CNN estaría"),
                   "repitiendo sobre todo el subtipo; si es baja, mira otra cosa.", "",
                   "**AUC de la CNN dentro de cada subtipo.** Dentro de un subtipo el médico apenas distingue (solo le",
                   "quedan edad y menopausia). Si la CNN separa ahí, la imagen aporta algo más que el subtipo.", "",
                   "| Subtipo | Pacientes | Con pCR | AUC de la CNN |", "|---|---|---|---|"]
        for s, v in analisis["por_subtipo"].items():
            lineas.append(f"| {s} | {v['pacientes']} | {v['pcr']} | {coma(v['cnn'], 3)} |")
        lineas += ["", "**Por cohorte (AUC fuera de fold).**", "",
                   "| Cohorte | Pacientes | Con pCR | CNN | Médico | CNN + médico |", "|---|---|---|---|---|---|"]
        for co, v in analisis["por_cohorte"].items():
            lineas.append(f"| {NOMBRE_COHORTE.get(co, co)} | {v['pacientes']} | {v['pcr']} | {coma(v['cnn'], 3)} | "
                          f"{coma(v['medico'], 3)} | {coma(v['combinado'], 3)} |")
        lineas += ["", ("El combinado usa la puntuación fuera de fold de la CNN (cada paciente, predicha por el modelo "
                        "que no la vio):"),
                   ("es apilado honesto. Fuera de fold, a cada paciente la predicen las 6 redes de su fold; en test, "
                    "las 30 del conjunto."),
                   ("La media de 30 varía menos entre pacientes, así que en test el peso de la CNN en el combinado podría "
                    "quedarse corto: `evaluar_test` mide esa dispersión (D40)."), ""]
    else:
        lineas += ["## 4. CNN frente al médico", "",
                   f"Pendiente: {motivo_cnn}. Se completa ejecutando de nuevo este script después de `cerrar_modelo`.",
                   ""]
        lineas += ["**Por cohorte (AUC fuera de fold del médico).**", "", "| Cohorte | Pacientes | Con pCR | Médico |",
                   "|---|---|---|---|"]
        for co, v in analisis["por_cohorte"].items():
            lineas.append(f"| {NOMBRE_COHORTE.get(co, co)} | {v['pacientes']} | {v['pcr']} | {coma(v['medico'], 3)} |")
        lineas.append("")

    lineas += ["## 5. Cómo leerlo", "",
               "- Es el listón que la CNN tiene que superar para decir que la imagen aporta algo que no sabe ya el médico.",
               ("- Las diferencias de AUC se juzgan por su intervalo: si cruza el 0, no hay pruebas de que un modelo sea "
                "mejor que otro."),
               ("- En la bibliografía pasa lo mismo: en Joo et al. 2021 la clínica sola daba 0,83 y la imagen la subía "
                "a 0,89."),
               (f"- Modelo guardado en `experimentos/modelo_clinico/modelo_clinico.json` (huella `{guardado['huella']}`"
                + (f", hecho con la CNN `{guardado['huella_cnn']}`" if guardado.get("huella_cnn") else "")
                + "): lo usa `evaluar_test`."), ""]
    return "\n".join(lineas)


# --------------------------------------------------------------------------- #
# Todo junto
# --------------------------------------------------------------------------- #

def generar(tabla: pd.DataFrame, cnn: pd.Series | None, huella_cnn: str | None, motivo_cnn: str = "ok",
            referencias: dict | None = None, destino: Path = MODELO_JSON, informe_md: Path = INFORME,
            carpeta_figuras: Path = FIGURAS, n_bootstrap: int = 500) -> tuple[dict, dict]:
    analisis = analizar(tabla, cnn, referencias, n_bootstrap)
    guardado = exportar(analisis, huella_cnn, destino)
    figuras(analisis, tabla, carpeta_figuras)
    informe_md.parent.mkdir(parents=True, exist_ok=True)
    informe_md.write_text(informe(analisis, guardado, motivo_cnn), encoding="utf-8")
    resumen_json = {k: v for k, v in analisis.items() if k not in ("modelos", "finales")}
    resumen_json["modelos"] = {n: {k: v for k, v in r.items() if k != "prob"} for n, r in analisis["modelos"].items()}
    (carpeta_figuras / "modelo_clinico.json").write_text(json.dumps(resumen_json, indent=2, ensure_ascii=False,
                                                                    default=float), encoding="utf-8")
    return analisis, guardado


def main() -> None:
    argparse.ArgumentParser(description="Modelo clínico de comparación (sin imagen), con los 5 folds de train.").parse_args()
    from src.data.auditoria import referencias_sin_imagen

    samples, patients = uc.cargar_samples(), uc.cargar_patients()
    tabla = tabla_train(samples, patients)
    cnn, huella_cnn, motivo = cnn_cerrada()
    print(f"{len(tabla)} pacientes de train · CNN: {'sí (' + huella_cnn + ')' if cnn is not None else motivo}")
    analisis, guardado = generar(tabla, cnn, huella_cnn, motivo, referencias_sin_imagen(samples, patients))
    for clave, r in analisis["modelos"].items():
        print(f"  {NOMBRE_MODELO[clave]:<40} AUC {r['auc_val']:.3f} ± {r['auc_val_desv']:.3f}")
    if "comparacion" in analisis:
        d = analisis["comparacion"]["combinado_menos_medico"]
        print(f"  Lo que añade la imagen al médico: {d['diferencia']:+.3f} (IC {d['ic95'][0]:+.3f} a {d['ic95'][1]:+.3f})")
    print(f"Informe en {INFORME.relative_to(RAIZ_REPO)} · modelo en {MODELO_JSON.relative_to(RAIZ_REPO)} "
          f"(huella {guardado['huella']})")
    if cnn is None:
        print("Cuando la CNN esté cerrada (cerrar_modelo), vuelve a ejecutar este script ANTES de evaluar_test.")


if __name__ == "__main__":
    main()
