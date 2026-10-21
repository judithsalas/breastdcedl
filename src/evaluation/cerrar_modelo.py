"""Cierre del modelo (F4, D21): de la validación cruzada ganadora al modelo final en models/final/.

    python -m src.evaluation.cerrar_modelo                          # lo elige todo con los criterios
    python -m src.evaluation.cerrar_modelo --cv runs/<carpeta_cv>   # forzar una validación cruzada
    python -m src.evaluation.cerrar_modelo --modo rapido            # prueba de humo

Todo se decide con las predicciones FUERA DE FOLD de train: cada paciente de train está predicha
por el modelo que no la vio al entrenar. Test no se toca aquí (rúbrica: test una sola vez).

Pasos:
1. **Qué validación cruzada.** La versión ganadora según el criterio de comparación (D16). De sus
   dos pérdidas, la ponderada solo gana si su AUC medio supera al de la normal en más de una
   desviación; si no, la normal, que es la más simple.
2. **Calibración de Platt.** Se ajusta logit_calibrado = a · logit + b con regresión logística
   sobre los cortes. No cambia el orden de las pacientes (el AUC apenas se mueve), pero hace que
   la probabilidad mostrada se parezca a la frecuencia real. Es imprescindible con la pérdida
   ponderada, que infla las probabilidades.
3. **Agregación por paciente:** la media de los cortes (lo que indica el profesor). Las demás se
   calculan solo para enseñar que la elección no cambia mucho las cosas.
4. **Umbral:** la mayor sensibilidad con especificidad ≥ 0,80 (src/evaluation/umbral.py).
5. **Exportar** a models/final/ los 5 modelos (uno por fold) y manifiesto.json con todo lo
   necesario: arquitectura, sha256 de cada fichero, calibración, umbral y métricas internas.
6. **Informe** en docs/resultados/2_cierre/CIERRE_MODELO.md, con sus figuras al lado. Ahí queda
   también oof_pacientes.csv (probabilidad fuera de fold por paciente), que usa el modelo clínico
   (D25).
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve

from src.data.caso import NOMBRE_COHORTE, RAIZ_REPO, RESULTADOS
from src.evaluation.metricas import auc_con_ic, brier, coma, metricas_con_umbral, tabla_calibracion
from src.evaluation.umbral import ESPECIFICIDAD_MINIMA, comparar_umbrales
from src.models.final import CARPETA_FINAL, cargar_modelo_final, huella_modelo, sha256
from src.models.historial import HISTORICO, desplegable, elegir_mejor, leer_historico, validaciones_validas

CARPETA_RUNS = RAIZ_REPO / "runs"
FIGURAS = RESULTADOS / "2_cierre"
INFORME = FIGURAS / "CIERRE_MODELO.md"
AZUL, NARANJA, TINTA, TINTA_SUAVE, REJILLA = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df"
EPS = 1e-6


# --------------------------------------------------------------------------- #
# 1. Qué validación cruzada
# --------------------------------------------------------------------------- #

def elegir_validacion(modo: str = "completo", historico: Path = HISTORICO) -> tuple[Path, dict]:
    """Carpeta de la validación cruzada ganadora y el porqué."""
    tabla = leer_historico(historico, modo)
    mejor = elegir_mejor(tabla)
    if mejor is None:
        raise SystemExit("No hay validaciones cruzadas completas para elegir. Lanza antes: python -m src.training.plan")
    normal = validaciones_validas(tabla, "normal").set_index("modelo").loc[mejor["version"]]
    ponderadas = validaciones_validas(tabla, "ponderada")
    eleccion = {"version": mejor["version"], "motivo_version": mejor["motivo"], "perdida": "normal",
                "auc_normal": float(normal.auc_val), "auc_normal_desv": float(normal.auc_val_desv)}
    fila = normal
    if mejor["version"] in set(ponderadas.modelo):
        pond = ponderadas.set_index("modelo").loc[mejor["version"]]
        eleccion["auc_ponderada"] = float(pond.auc_val)
        if pond.auc_val > normal.auc_val + (0 if pd.isna(normal.auc_val_desv) else normal.auc_val_desv):
            eleccion["perdida"], fila = "ponderada", pond
            eleccion["motivo_perdida"] = "la ponderada supera a la normal en más de una desviación"
        else:
            eleccion["motivo_perdida"] = "la ponderada no supera a la normal en más de una desviación: se queda la normal"
    else:
        eleccion["motivo_perdida"] = "no hay validación cruzada con pérdida ponderada de esta versión"
    return CARPETA_RUNS / fila.ejecucion, eleccion


# --------------------------------------------------------------------------- #
# 2-4. Calibración, agregación y umbral (fuera de fold)
# --------------------------------------------------------------------------- #

def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def ajustar_platt(logits: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Regresión logística de una variable (casi sin regularizar): devuelve a y b."""
    reg = LogisticRegression(C=1e6, max_iter=1000).fit(np.asarray(logits).reshape(-1, 1), np.asarray(y).astype(int))
    return float(reg.coef_[0, 0]), float(reg.intercept_[0])


def sigmoide(z: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-z))


def analizar(oof: pd.DataFrame, cohortes: dict | None = None) -> dict:
    """Calibración, agregaciones y umbral con las predicciones fuera de fold por corte."""
    oof = oof.copy()
    a, b = ajustar_platt(logit(oof.prob), oof.pCR)
    oof["prob_cal"] = sigmoide(a * logit(oof.prob) + b)

    g = oof.groupby("patient_id")
    pacientes = pd.DataFrame({"pCR": g.pCR.first(), "prob_sin_calibrar": g.prob.mean(), "prob": g.prob_cal.mean(),
                              "mediana": g.prob_cal.median(), "maximo": g.prob_cal.max()})
    if cohortes is not None:
        pacientes["cohorte"] = pacientes.index.map(cohortes)
    y = pacientes.pCR.to_numpy()

    umbrales = comparar_umbrales(y, pacientes.prob.to_numpy())
    t = umbrales["elegido"]["umbral"]
    pacientes["voto"] = (oof.assign(sobre=oof.prob_cal >= t).groupby("patient_id").sobre.mean())
    agregaciones = {nombre: auc_con_ic(y, pacientes[col].to_numpy()) for nombre, col in
                    (("media", "prob"), ("mediana", "mediana"), ("máximo", "maximo"), ("voto", "voto"))}

    por_cohorte = {}
    if "cohorte" in pacientes:
        for c, grupo in pacientes.groupby("cohorte"):
            por_cohorte[str(c)] = {"pacientes": len(grupo), "pcr": int(grupo.pCR.sum()),
                                   **auc_con_ic(grupo.pCR.to_numpy(), grupo.prob.to_numpy())}
    return {
        "calibracion": {"a": a, "b": b, "metodo": "Platt (regresión logística sobre el logit de cada corte)",
                        "brier_sin_calibrar": brier(y, pacientes.prob_sin_calibrar),
                        "brier_calibrado": brier(y, pacientes.prob),
                        "tabla_sin_calibrar": tabla_calibracion(y, pacientes.prob_sin_calibrar),
                        "tabla_calibrada": tabla_calibracion(y, pacientes.prob)},
        "agregaciones": agregaciones, "umbrales": umbrales, "por_cohorte": por_cohorte,
        "pacientes": len(pacientes), "pacientes_pcr": int(y.sum()),
        "_tabla": pacientes,
    }


# --------------------------------------------------------------------------- #
# 5. Exportar
# --------------------------------------------------------------------------- #

def exportar(carpeta_cv: Path, eleccion: dict, analisis: dict, destino: Path = CARPETA_FINAL) -> dict:
    """Copia los 5 modelos (solo los pesos) y escribe manifiesto.json."""
    folds = sorted(carpeta_cv.glob("fold_*"))
    if len(folds) != 5:
        raise SystemExit(f"{carpeta_cv.name} tiene {len(folds)} folds; el modelo final necesita los 5.")
    destino.mkdir(parents=True, exist_ok=True)
    for viejo in [*destino.glob("*.pt"), destino / "manifiesto.json"]:
        viejo.unlink(missing_ok=True)

    resultados = [json.loads((f / "resultados.json").read_text(encoding="utf-8")) for f in folds]
    pesos = []
    for f, r in zip(folds, resultados, strict=True):
        punto = torch.load(f / "mejor.pt", map_location="cpu", weights_only=True)
        ruta = destino / f"{f.name}.pt"
        torch.save(punto["estado"], ruta)
        pesos.append({"fichero": ruta.name, "sha256": sha256(ruta), "epoca": punto["epoca"],
                      "auc_val_fold": r["mejor_epoca"]["auc_val"]})

    definicion = resultados[0]["arquitectura"]
    resumen = json.loads((carpeta_cv / "resumen.json").read_text(encoding="utf-8"))
    elegido = analisis["umbrales"]["elegido"]
    manifiesto = {
        "fecha": datetime.now().astimezone().isoformat(timespec="seconds"),
        "version": eleccion["version"], "eleccion": eleccion, "validacion_cruzada": carpeta_cv.name,
        "definicion": definicion,
        "arquitectura": {k: v for k, v in definicion.items() if k != "entrenamiento"},
        "receta": resultados[0].get("receta"),
        "entrada": "3 PNG en gris de 8 bits (PRE, EARLY, LATE), 256×256, divididos entre 255 → [0, 1]",
        "conjunto": "media de los logits de los 5 modelos (uno por fold)",
        "calibracion": {k: analisis["calibracion"][k] for k in ("a", "b", "metodo")},
        "agregacion": "media de las probabilidades de los cortes de una paciente",
        "umbral": float(elegido["umbral"]),
        "criterio_umbral": elegido["criterio"],
        "pesos": pesos,
        "entrenamiento": {"perdida": eleccion["perdida"], "config": resultados[0]["config"],
                          "parametros": resultados[0]["parametros"],
                          "maquina": resultados[0]["maquina"]},
        "metricas_internas": {
            "auc_val_media": resumen["media_desviacion"]["auc_val"]["media"],
            "auc_val_desviacion": resumen["media_desviacion"]["auc_val"]["desviacion"],
            "auc_train_media": resumen["media_desviacion"]["auc_train"]["media"],
            "diferencia_media": resumen["media_desviacion"]["diferencia_auc"]["media"],
            "fuera_de_fold": {"auc": analisis["agregaciones"]["media"],
                              **{k: elegido[k] for k in ("sensibilidad", "especificidad", "precision_pcr",
                                                         "precision_no_pcr", "f1_pcr", "f1_no_pcr", "f1_macro",
                                                         "accuracy", "matriz_confusion")}},
            "brier_calibrado": analisis["calibracion"]["brier_calibrado"],
        },
    }
    manifiesto["huella_modelo"] = huella_modelo(manifiesto)
    (destino / "manifiesto.json").write_text(json.dumps(manifiesto, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifiesto


def comprobar_exportacion(destino: Path, carpeta_cv: Path) -> float | None:
    """Recarga el modelo final y comprueba que reproduce el pipeline en unos cortes.

    Cada modelo de fold, solo, debe dar en sus cortes de validación lo mismo que guardó el
    entrenamiento; el conjunto, lo mismo que la media de logits de los 5. Devuelve la mayor
    diferencia encontrada en un modelo individual (debería ser ~1e-6), o None si no hay imágenes.
    """
    from src.data.caso import RAIZ_DATOS, uc

    modelo, _ = cargar_modelo_final(destino)
    samples = uc.cargar_samples().set_index("sample_id")
    diferencia = 0.0
    for k, red in enumerate(modelo.redes):
        pred = pd.read_csv(carpeta_cv / f"fold_{k}" / "predicciones_val.csv").head(8)
        if not pred.sample_id.isin(samples.index).all():
            return None
        filas = samples.loc[pred.sample_id].reset_index()
        if not all((RAIZ_DATOS / r).exists() for r in filas.path_pre):
            return None                      # sin imágenes (por ejemplo, en los tests): no se comprueba
        x = torch.stack([torch.from_numpy(uc.cargar_imagen(f)) for f in filas.itertuples()])
        with torch.no_grad():
            p = torch.sigmoid(red(x).squeeze(1)).numpy()
        diferencia = max(diferencia, float(np.abs(p - pred.prob.to_numpy()).max()))
    return diferencia


# --------------------------------------------------------------------------- #
# 6. Informe
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
    formato = matplotlib.ticker.FuncFormatter(lambda v, _: coma(v, 1))
    ax.xaxis.set_major_formatter(formato)
    ax.yaxis.set_major_formatter(formato)


def figuras(analisis: dict, carpeta: Path) -> list[Path]:
    carpeta.mkdir(parents=True, exist_ok=True)
    tabla = analisis["_tabla"]
    y, p = tabla.pCR.to_numpy(), tabla.prob.to_numpy()
    elegido = analisis["umbrales"]["elegido"]
    rutas = []

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 4.3))
    fpr, tpr, _ = roc_curve(y, p)
    ax1.plot(fpr, tpr, color=AZUL, lw=2, label=f"AUC {coma(roc_auc_score(y, p), 3)}")
    ax1.plot([0, 1], [0, 1], color=TINTA_SUAVE, lw=0.8, ls="--")
    ax1.scatter([1 - elegido["especificidad"]], [elegido["sensibilidad"]], s=60, color=NARANJA, zorder=5,
                edgecolor="white", linewidth=1.5, label=f"umbral {coma(elegido['umbral'], 3)}")
    ax1.axvline(1 - ESPECIFICIDAD_MINIMA, color=NARANJA, lw=0.8, ls=":")
    ax1.legend(frameon=False, fontsize=8.5, loc="lower right")
    _estilo(ax1, "ROC fuera de fold", "1 − especificidad", "Sensibilidad")

    ts = np.linspace(0, 1, 201)
    sens = [metricas_con_umbral(y, p, t)["sensibilidad"] for t in ts]
    espe = [metricas_con_umbral(y, p, t)["especificidad"] for t in ts]
    ax2.plot(ts, sens, color=AZUL, lw=2, label="sensibilidad")
    ax2.plot(ts, espe, color=NARANJA, lw=2, label="especificidad")
    ax2.axhline(ESPECIFICIDAD_MINIMA, color=TINTA_SUAVE, lw=0.8, ls="--")
    ax2.axvline(elegido["umbral"], color=TINTA_SUAVE, lw=0.8, ls=":")
    ax2.legend(frameon=False, fontsize=8.5)
    _estilo(ax2, "Sensibilidad y especificidad según el umbral", "Umbral", "")

    for clave, color, nombre in (("tabla_sin_calibrar", NARANJA, "sin calibrar"), ("tabla_calibrada", AZUL, "calibrada")):
        t = analisis["calibracion"][clave]
        ax3.plot([g["prob_media"] for g in t], [g["proporcion_real"] for g in t], marker="o", ms=5, lw=1.6,
                 color=color, label=nombre)
    ax3.plot([0, 1], [0, 1], color=TINTA_SUAVE, lw=0.8, ls="--")
    ax3.legend(frameon=False, fontsize=8.5)
    _estilo(ax3, "Calibración por deciles", "Probabilidad media predicha", "Proporción real de pCR")
    fig.tight_layout()
    ruta = carpeta / "cierre.png"
    fig.savefig(ruta, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    rutas.append(ruta)
    return rutas


def informe(eleccion: dict, analisis: dict, manifiesto: dict, diferencia: float | None) -> str:
    u, cal = analisis["umbrales"], analisis["calibracion"]
    mi = manifiesto["metricas_internas"]
    lineas = [
        "# Cierre del modelo", "",
        f"Generado por `python -m src.evaluation.cerrar_modelo` el {manifiesto['fecha'][:16].replace('T', ' ')}.",
        "Todo con predicciones **fuera de fold** de train; test no se ha usado. Explicación de cada",
        "decisión: [`COMO_SE_HIZO.md`](../../COMO_SE_HIZO.md).", "",
        "## 1. Qué modelo", "",
        f"- **Versión:** {eleccion['version']}. Motivo: {eleccion['motivo_version']}.",
        f"- **Pérdida:** {eleccion['perdida']}. Motivo: {eleccion['motivo_perdida']}.",
        f"- **Validación cruzada usada:** `{manifiesto['validacion_cruzada']}`.",
        (f"- **AUC en validación (5 folds):** {coma(mi['auc_val_media'], 3)} ± "
         f"{coma(mi['auc_val_desviacion'], 3)}; en entrenamiento {coma(mi['auc_train_media'], 3)}; "
         f"diferencia media {coma(mi['diferencia_media'], 3)}."),
        ("- **Modelo final:** media de los logits de los 5 modelos de esa validación cruzada "
         f"(huella del modelo `{manifiesto['huella_modelo']}`)."), "",
        "## 2. Calibración (Platt)", "",
        (f"logit calibrado = {coma(cal['a'], 3)} · logit + ({coma(cal['b'], 3)}). Error de Brier por "
         f"paciente: {coma(cal['brier_sin_calibrar'], 4)} sin calibrar → {coma(cal['brier_calibrado'], 4)} "
         "calibrado (menos es mejor; 0,25 sería decir siempre 0,5)."), "",
        f"## 3. Agregación por paciente ({analisis['pacientes']} pacientes, {analisis['pacientes_pcr']} con pCR)", "",
        "| Agregación | AUC | IC 95 % (bootstrap) |", "|---|---|---|"]
    for nombre, v in analisis["agregaciones"].items():
        marca = " **(elegida)**" if nombre == "media" else ""
        lineas.append(f"| {nombre}{marca} | {coma(v['auc'], 3)} | {coma(v['ic95'][0], 3)}–{coma(v['ic95'][1], 3)} |")
    lineas += ["", "## 4. Umbral", "",
               "| Criterio | Umbral | Sensibilidad | Especificidad | Precision pCR | F1 pCR | F1 no pCR | F1 macro |",
               "|---|---|---|---|---|---|---|---|"]
    for clave, v in u.items():
        marca = " **(elegido)**" if clave == "elegido" else ""
        lineas.append(f"| {v['criterio']}{marca} | {coma(v['umbral'], 3)} | {coma(v['sensibilidad'], 3)} | "
                      f"{coma(v['especificidad'], 3)} | "
                      f"{'—' if np.isnan(v['precision_pcr']) else coma(v['precision_pcr'], 3)} | "
                      f"{coma(v['f1_pcr'], 3)} | {coma(v['f1_no_pcr'], 3)} | {coma(v['f1_macro'], 3)} |")
    m = u["elegido"]["matriz_confusion"]
    lineas += ["", f"Con el umbral elegido, fuera de fold: VP {m['VP']}, FN {m['FN']}, FP {m['FP']}, VN {m['VN']}.", ""]
    if analisis["por_cohorte"]:
        lineas += ["## 5. Por cohorte (fuera de fold)", "", "| Cohorte | Pacientes | Con pCR | AUC | IC 95 % |",
                   "|---|---|---|---|---|"]
        for c, v in analisis["por_cohorte"].items():
            lineas.append(f"| {NOMBRE_COHORTE.get(c, c)} | {v['pacientes']} | {v['pcr']} | {coma(v['auc'], 3)} | "
                          f"{coma(v['ic95'][0], 3)}–{coma(v['ic95'][1], 3)} |")
        lineas.append("")
    lineas += ["## 6. Exportación", "",
               "| Fichero | Época | AUC del fold | sha256 |", "|---|---|---|---|"]
    for f in manifiesto["pesos"]:
        lineas.append(f"| `models/final/{f['fichero']}` | {f['epoca']} | {coma(f['auc_val_fold'], 3)} | "
                      f"`{f['sha256'][:16]}…` |")
    if diferencia is not None:
        lineas += ["", ("Comprobación: cada modelo recargado desde `models/final/` reproduce las predicciones "
                        f"del entrenamiento con una diferencia máxima de {diferencia:.1e}.")]
    lineas += ["", "![Cierre del modelo](cierre.png)", ""]
    return "\n".join(lineas)


def cerrar(carpeta_cv: Path, eleccion: dict, destino: Path = CARPETA_FINAL, informe_md: Path = INFORME,
           carpeta_figuras: Path = FIGURAS, comprobar: bool = True) -> dict:
    """Hace todo el cierre a partir de una validación cruzada. Devuelve el manifiesto."""
    from src.training.entrenar import cohortes

    oof = pd.read_csv(carpeta_cv / "predicciones_oof.csv")
    analisis = analizar(oof, cohortes())
    manifiesto = exportar(carpeta_cv, eleccion, analisis, destino)
    diferencia = comprobar_exportacion(destino, carpeta_cv) if comprobar else None
    figuras(analisis, carpeta_figuras)
    informe_md.parent.mkdir(parents=True, exist_ok=True)
    informe_md.write_text(informe(eleccion, analisis, manifiesto, diferencia), encoding="utf-8")
    (carpeta_figuras / "cierre.json").write_text(json.dumps(
        {"huella_modelo": manifiesto["huella_modelo"], **{k: v for k, v in analisis.items() if k != "_tabla"}},
        indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    # Probabilidad fuera de fold de cada paciente de train: la usa el modelo clínico (D25) para
    # comparar y combinar. Se versiona, así no hace falta la carpeta runs/ en otra máquina.
    analisis["_tabla"].to_csv(carpeta_figuras / "oof_pacientes.csv", index_label="patient_id")
    return manifiesto


def main() -> None:
    p = argparse.ArgumentParser(description="Cierra el modelo: calibración, umbral y exportación a models/final/.")
    p.add_argument("--cv", help="carpeta de una validación cruzada concreta (por defecto, la ganadora)")
    p.add_argument("--modo", choices=["rapido", "completo"], default="completo")
    args = p.parse_args()
    if args.cv:
        carpeta = Path(args.cv).resolve()
        resumen = json.loads((carpeta / "resumen.json").read_text(encoding="utf-8"))
        eleccion = {"version": resumen["modelo"], "motivo_version": "elegida a mano con --cv",
                    "perdida": resumen["perdida"], "motivo_perdida": "elegida a mano con --cv"}
    else:
        carpeta, eleccion = elegir_validacion(args.modo)
    if not desplegable(eleccion["version"]):
        raise SystemExit(f"{eleccion['version']} se entrenó recortando con la caja del tumor, que la app y la "
                         "validación privada no tienen: no puede ser el modelo final (D33).")
    manifiesto = cerrar(carpeta, eleccion)
    print(f"Modelo final: {manifiesto['version']} ({eleccion['perdida']}), umbral {manifiesto['umbral']:.3f}, "
          f"huella {manifiesto['huella_modelo']}")
    print(f"Pesos y manifiesto en {CARPETA_FINAL.relative_to(RAIZ_REPO)}/ · informe en {INFORME.relative_to(RAIZ_REPO)}")
    print("Siguiente paso: python -m src.evaluation.modelo_clinico (comparación con el médico, D25)")
    print("Después, UNA sola vez: python -m src.evaluation.evaluar_test")


if __name__ == "__main__":
    main()
