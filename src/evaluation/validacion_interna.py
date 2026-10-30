"""Validación interna de la calibración y del umbral (D40): ¿cuánto de optimistas son las cifras internas?

    python -m src.evaluation.validacion_interna

En el cierre (cerrar_modelo), la calibración de Platt y el umbral se eligieron con las predicciones fuera de
fold de TODAS las pacientes de train, y las métricas internas (Brier, sensibilidad, especificidad, precisión) se
midieron con esas mismas pacientes. Eso es correcto para fijar el modelo, pero las cifras salen algo
optimistas. Aquí se repite el procedimiento entero SIN el fold que se mide:

- para cada fold k, Platt se ajusta con los cortes de los otros cuatro;
- el umbral se elige con las pacientes de los otros cuatro (ya calibradas con ese Platt), con el mismo criterio;
- los dos se aplican al fold k, que no ha intervenido en nada.

Así, cada paciente recibe una probabilidad y una clase como las recibirá una paciente nueva. Se compara además
con la referencia que cualquier probabilidad tiene que batir: decir a todas la tasa de pCR (Brier de la
probabilidad constante). Con el modelo del médico se hace lo mismo con el umbral, para comparar igual.

No cambia el modelo final (experimentos/modelo_final queda como está) ni toca test. Escribe
docs/resultados/2_cierre/VALIDACION_INTERNA.md y validacion_interna.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.caso import RAIZ_REPO, RESULTADOS
from src.evaluation.cerrar_modelo import CARPETA_RUNS, ajustar_platt, combinar_oof, logit, sigmoide
from src.evaluation.metricas import auc_por_rangos, brier, coma
from src.evaluation.umbral import ESPECIFICIDAD_MINIMA, umbral_por_especificidad
from src.models.final import CARPETA_FINAL

CARPETA = RESULTADOS / "2_cierre"
INFORME = CARPETA / "VALIDACION_INTERNA.md"
JSON = CARPETA / "validacion_interna.json"


def carpetas_del_conjunto(manifiesto: dict, runs: Path = CARPETA_RUNS) -> list[Path]:
    """Las validaciones cruzadas de las versiones del modelo final, en runs/."""
    return [Path(runs) / m["validacion_cruzada"] for m in manifiesto["miembros"]]


def _por_paciente(cortes: pd.DataFrame, columna: str) -> pd.Series:
    return cortes.groupby("patient_id")[columna].mean()


def _division(a: float, b: float) -> float:
    return a / b if b else float("nan")


def _clase(y: np.ndarray, p: np.ndarray, umbral: float | np.ndarray) -> dict:
    """Matriz de confusión y métricas con un umbral (o un umbral por paciente)."""
    pred = np.asarray(p) >= np.asarray(umbral)
    y = np.asarray(y).astype(bool)
    vp, fp = int((pred & y).sum()), int((pred & ~y).sum())
    fn, vn = int((~pred & y).sum()), int((~pred & ~y).sum())
    return {"sensibilidad": _division(vp, vp + fn), "especificidad": _division(vn, vn + fp),
            "precision_pcr": _division(vp, vp + fp), "precision_no_pcr": _division(vn, vn + fn), "marcadas_pcr": vp + fp,
            "matriz_confusion": {"VP": vp, "FP": fp, "FN": fn, "VN": vn}}


def umbral_cruzado(y: np.ndarray, p: np.ndarray, folds: np.ndarray) -> dict:
    """El umbral de cada fold se elige con los otros cuatro y se aplica a él. Para probabilidades que ya son
    fuera de fold y no necesitan calibrarse (el médico)."""
    y, p, folds = np.asarray(y), np.asarray(p, float), np.asarray(folds)
    umbrales = np.zeros(len(y))
    por_fold = []
    for k in np.unique(folds):
        t = umbral_por_especificidad(y[folds != k], p[folds != k])["umbral"]
        umbrales[folds == k] = t
        por_fold.append(float(t))
    return {"umbrales_por_fold": por_fold, **_clase(y, p, umbrales)}


def validar(cortes: pd.DataFrame) -> dict:
    """Todo el análisis con las predicciones fuera de fold POR CORTE del conjunto (prob, pCR, fold, patient_id)."""
    z = logit(cortes.prob.to_numpy())
    yc = cortes.pCR.to_numpy().astype(int)
    fc = cortes.fold.to_numpy()

    # Como en el cierre: Platt y umbral con todas las pacientes
    a, b = ajustar_platt(z, yc)
    cortes = cortes.assign(mismas=sigmoide(a * z + b), cruzada=np.nan)
    pacientes = pd.DataFrame({"pCR": cortes.groupby("patient_id").pCR.first(),
                              "fold": cortes.groupby("patient_id").fold.first(),
                              "sin_calibrar": _por_paciente(cortes, "prob"),
                              "mismas": _por_paciente(cortes, "mismas")})
    y, f = pacientes.pCR.to_numpy().astype(int), pacientes.fold.to_numpy()
    t_mismas = umbral_por_especificidad(y, pacientes.mismas.to_numpy())["umbral"]

    # Sin el fold que se mide: Platt con los cortes de los otros cuatro, umbral con sus pacientes
    platt, umbrales = [], pd.Series(np.nan, index=pacientes.index)
    for k in np.unique(fc):
        ak, bk = ajustar_platt(z[fc != k], yc[fc != k])
        platt.append({"fold": int(k), "a": ak, "b": bk})
        cal = sigmoide(ak * z + bk)
        cortes.loc[fc == k, "cruzada"] = cal[fc == k]
        otras = pd.Series(cal[fc != k], index=cortes.index[fc != k]).groupby(cortes.patient_id[fc != k]).mean()
        tk = umbral_por_especificidad(pacientes.pCR.loc[otras.index].to_numpy(), otras.to_numpy())["umbral"]
        umbrales[pacientes.fold == k] = tk
        platt[-1]["umbral"] = float(tk)
    pacientes["cruzada"] = _por_paciente(cortes, "cruzada")

    constante_cruzada = np.array([y[f != k].mean() for k in f])
    auc_folds = [auc_por_rangos(y[f == k], pacientes.mismas.to_numpy()[f == k]) for k in np.unique(f)]
    return {
        "pacientes": len(y), "pacientes_pcr": int(y.sum()), "prevalencia": float(y.mean()),
        "platt": {"todas": {"a": a, "b": b}, "por_fold": platt},
        "brier": {"constante": brier(y, np.full(len(y), y.mean())), "constante_cruzada": brier(y, constante_cruzada),
                  "sin_calibrar": brier(y, pacientes.sin_calibrar), "platt_mismas": brier(y, pacientes.mismas),
                  "platt_cruzado": brier(y, pacientes.cruzada)},
        "umbral": {"mismas": {"umbral": float(t_mismas), **_clase(y, pacientes.mismas.to_numpy(), t_mismas)},
                   "cruzado": {"umbrales_por_fold": [p["umbral"] for p in platt],
                               **_clase(y, pacientes.cruzada.to_numpy(), umbrales.to_numpy())}},
        # El AUC de cada fold no depende de la calibración (Platt no cambia el orden dentro de un fold). El de
        # todas juntas sí: depende de cómo encajan entre sí las probabilidades de modelos distintos.
        "auc": {"media_folds": float(np.mean(auc_folds)),
                "todas_juntas_platt_global": auc_por_rangos(y, pacientes.mismas.to_numpy()),
                "todas_juntas_platt_por_fold": auc_por_rangos(y, pacientes.cruzada.to_numpy())},
        "dispersion": {"desviacion_logit_paciente": float(np.std(logit(pacientes.mismas.to_numpy()), ddof=1)),
                       "percentil_5": float(np.quantile(pacientes.mismas, 0.05)),
                       "percentil_95": float(np.quantile(pacientes.mismas, 0.95)),
                       "redes_por_paciente": "las de su fold (una por versión)"},
        "_pacientes": pacientes,
    }


def medico(tabla: pd.DataFrame | None = None) -> dict | None:
    """El umbral del médico, elegido con las mismas pacientes y sin el fold que se mide, para comparar igual."""
    from src.evaluation import modelo_clinico as mc

    try:
        tabla = mc.tabla_train() if tabla is None else tabla
    except FileNotFoundError:
        return None
    r = mc.validacion_cruzada(tabla, mc.VARIABLES_MEDICO)
    y, p, f = tabla.pCR.to_numpy(), r["prob"].to_numpy(), tabla.fold.to_numpy()
    u = r["umbral"]
    return {"brier": brier(y, p),
            "umbral": {"mismas": {"umbral": float(u["umbral"]), **_clase(y, p, u["umbral"])},
                       "cruzado": umbral_cruzado(y, p, f)}}


def _fila(nombre: str, m: dict) -> str:
    c = m["matriz_confusion"]
    return (f"| {nombre} | {coma(m['sensibilidad'], 3)} | {coma(m['especificidad'], 3)} | "
            f"{coma(m['precision_pcr'], 3)} | {c['VP']} · {c['FP']} · {c['FN']} · {c['VN']} |")


def informe(r: dict, huella: str) -> str:
    b, u, auc = r["brier"], r["umbral"], r["auc"]
    lineas = [
        "# Validación interna de la calibración y del umbral", "",
        (f"Generado por `python -m src.evaluation.validacion_interna` para el modelo `{huella}`. Solo train; test no "
         "se toca."), "",
        ("En el cierre, Platt y el umbral se eligieron con todas las pacientes de train (fuera de fold) y se midieron "
         "con esas mismas."), "Aquí, cada fold recibe un Platt y un umbral elegidos sin él, como una paciente nueva.", "",
        f"## 1. ¿Sirve la probabilidad? Brier ({r['pacientes']} pacientes, {r['pacientes_pcr']} con pCR)", "",
        ("El Brier es el error cuadrático medio entre la probabilidad y lo que pasó (0 es perfecto). La referencia es "
         f"decir a todas la tasa de pCR ({coma(100 * r['prevalencia'], 1)} %)."), "",
        "| Probabilidad | Brier |", "|---|---|",
        f"| Constante: la tasa de pCR a todas | {coma(b['constante'], 4)} |",
        f"| Constante, con la tasa de los otros folds | {coma(b['constante_cruzada'], 4)} |",
        f"| CNN sin calibrar | {coma(b['sin_calibrar'], 4)} |",
        f"| CNN con Platt ajustado con las mismas pacientes (cifra del cierre) | {coma(b['platt_mismas'], 4)} |",
        f"| **CNN con Platt ajustado sin el fold** | **{coma(b['platt_cruzado'], 4)}** |"]
    if r.get("medico"):
        lineas.append(f"| Médico (fuera de fold) | {coma(r['medico']['brier'], 4)} |")
    mejor = b["platt_cruzado"] < b["constante_cruzada"]
    lineas += ["", ("**Lectura:** la probabilidad de la CNN, calibrada sin el fold, " +
                    ("bate por poco a la probabilidad constante." if mejor else
                     "no bate a la probabilidad constante: como probabilidad, dice tanto como la tasa de pCR.")
                    + " Lo que ordena (el AUC) es otra cosa, y se mide aparte."), "",
               "| Fold | a | b | Umbral |", "|---|---|---|---|"]
    lineas += [f"| {p['fold']} | {coma(p['a'], 3)} | {coma(p['b'], 3)} | {coma(p['umbral'], 3)} |" for p in r["platt"]["por_fold"]]
    lineas += [(f"| todos (modelo final) | {coma(r['platt']['todas']['a'], 3)} | {coma(r['platt']['todas']['b'], 3)} | "
                f"{coma(u['mismas']['umbral'], 3)} |"), "",
               f"## 2. El umbral (mayor sensibilidad con especificidad ≥ {coma(ESPECIFICIDAD_MINIMA, 2)})", "",
               "| Cómo se eligió | Sensibilidad | Especificidad | Precisión pCR | VP · FP · FN · VN |", "|---|---|---|---|---|",
               _fila("CNN, con las mismas pacientes (cifra del cierre)", u["mismas"]),
               _fila("**CNN, sin el fold que se mide**", u["cruzado"])]
    if r.get("medico"):
        lineas += [_fila("Médico, con las mismas pacientes", r["medico"]["umbral"]["mismas"]),
                   _fila("Médico, sin el fold que se mide", r["medico"]["umbral"]["cruzado"])]
    lineas += ["", ("El criterio de especificidad ≥ 0,80 es una elección ilustrativa del trabajo (limitar los «pCR» "
                    "equivocados), no un requisito clínico validado."), "",
               "## 3. Por qué la cifra principal es la media de los folds", "",
               "| AUC | Valor |", "|---|---|",
               f"| Media de los 5 folds (no depende de la calibración) | {coma(auc['media_folds'], 3)} |",
               f"| Todas juntas, con un Platt para todos | {coma(auc['todas_juntas_platt_global'], 3)} |",
               f"| Todas juntas, con un Platt por fold | {coma(auc['todas_juntas_platt_por_fold'], 3)} |", "",
               ("Con todas las pacientes juntas se comparan probabilidades de modelos distintos, y el resultado cambia "
                "según cómo se calibre cada uno."),
               "La media de los folds solo compara pacientes predichas por el mismo modelo.", "",
               "## 4. Dispersión de la probabilidad", "",
               (f"Fuera de fold, cada paciente la predicen las redes de su fold ({r.get('versiones', '?')}, una por "
                f"versión): la desviación del "
                f"logit entre pacientes es {coma(r['dispersion']['desviacion_logit_paciente'], 3)}, y el 90 % de las "
                f"probabilidades está entre {coma(r['dispersion']['percentil_5'], 3)} y "
                f"{coma(r['dispersion']['percentil_95'], 3)}."),
               (f"En la app y en test predicen todas ({5 * r['versiones'] if 'versiones' in r else '?'}). La media de más redes varía menos, así que en test la "
                "probabilidad puede salir más concentrada y con menos pacientes por encima del umbral. `evaluar_test` "
                "lo mide."), ""]
    return "\n".join(lineas)


def leer(manifiesto: dict, ruta: Path = JSON) -> dict | None:
    """El resultado guardado, si existe y es de este modelo (lo usan la app web y evaluar_test)."""
    ruta = Path(ruta)
    if not ruta.exists():
        return None
    datos = json.loads(ruta.read_text(encoding="utf-8"))
    return datos if datos.get("huella_modelo") == manifiesto["huella_modelo"] else None


def main() -> None:
    argparse.ArgumentParser(description="Validación interna de Platt y del umbral, sin el fold que se mide.").parse_args()
    manifiesto = json.loads((CARPETA_FINAL / "manifiesto.json").read_text(encoding="utf-8"))
    carpetas = carpetas_del_conjunto(manifiesto)
    faltan = [c.name for c in carpetas if not (c / "predicciones_oof.csv").exists()]
    if faltan:
        raise SystemExit(f"Faltan las predicciones fuera de fold de: {', '.join(faltan)} (en runs/).")
    r = validar(combinar_oof(carpetas))
    r["medico"] = medico()
    r["huella_modelo"], r["versiones"] = manifiesto["huella_modelo"], len(carpetas)
    CARPETA.mkdir(parents=True, exist_ok=True)
    JSON.write_text(json.dumps({k: v for k, v in r.items() if not k.startswith("_")}, indent=2, ensure_ascii=False,
                               default=float), encoding="utf-8")
    INFORME.write_text(informe(r, manifiesto["huella_modelo"]), encoding="utf-8")
    b, c = r["brier"], r["umbral"]["cruzado"]
    print(f"Brier: constante {b['constante']:.4f} · CNN con Platt sin el fold {b['platt_cruzado']:.4f} "
          f"(con las mismas pacientes {b['platt_mismas']:.4f})")
    print(f"Umbral sin el fold: sensibilidad {c['sensibilidad']:.3f} · especificidad {c['especificidad']:.3f} · "
          f"precisión pCR {c['precision_pcr']:.3f}")
    print(f"Informe en {INFORME.relative_to(RAIZ_REPO)}")


if __name__ == "__main__":
    main()
