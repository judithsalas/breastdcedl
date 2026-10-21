"""Validación cruzada por paciente: la misma arquitectura entrenada en los 5 folds.

    python -m src.training.validacion_cruzada --modo completo --modelo cnn_v2
    python -m src.training.validacion_cruzada --modo completo --modelo cnn_v2 --ponderada
    python -m src.training.validacion_cruzada --modo rapido   --modelo cnn_v1 --folds 0 1

Cada fold es un entrenamiento normal de src/training/entrenar.py, con un fold distinto
reservado para validar y la misma configuración y semilla en todos. Todo queda en:

    runs/<fecha>_cv_<modelo>_<modo>[_ponderada]/
        fold_0/ ... fold_4/              lo mismo que una ejecución normal
        resumen.json                     época elegida y métricas de cada fold, media ± desviación,
                                         y métricas fuera de fold con las pacientes de todos los folds
        resumen.csv                      tabla por fold, con media y desviación al final
        predicciones_oof.csv             probabilidad de cada corte de train, dada por el modelo del
                                         fold en el que esa paciente estaba en validación
        predicciones_oof_pacientes.csv   lo mismo por paciente: base para elegir agregación y umbral
                                         (F4), estudiar la calibración y mirar resultados por cohorte
        graficas/                        curvas_folds.png (curvas de todos los folds superpuestas) y
                                         matriz_confusion_oof.png (todas las pacientes, fuera de fold)
y una sola fila en runs/historico.csv con la media de los folds (los folds no van sueltos), que
pasa también a las tablas automáticas de docs/ARQUITECTURAS.md.

Cómo leerlo:
- La cifra que se reporta es la media ± desviación de los 5 folds (lo que pide la rúbrica).
- La matriz y el AUC "fuera de fold" juntan las predicciones de 5 modelos distintos: sirven
  para elegir umbral y agregación con todas las pacientes, no como sexto resultado.
- El AUC de cada fold es el de su mejor época, elegida mirando ese mismo fold: es algo
  optimista. Por eso test se evalúa aparte y una sola vez.
- Los resúmenes se reescriben al acabar cada fold: si se corta, lo hecho no se pierde.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import obtener_config
from src.evaluation.metricas import auc_por_cohorte, referencias_triviales, senal_de_hospital
from src.training import historico
from src.training.entrenar import (
    CARPETA_RUNS,
    METODO_AGREGACION,
    UMBRAL,
    anadir_argumentos,
    cambios_de_config,
    ejecutar,
    evaluar,
    registrar_ejecucion,
    tabla_por_paciente,
)
from src.training.graficas import dibujar_graficas

# Métricas de la época elegida que se resumen con media ± desviación.
METRICAS = ["epoca", "auc_val", "auc_train", "diferencia_auc", "sensibilidad", "especificidad",
            "precision_pcr", "f1_pcr", "f1_no_pcr", "f1_macro", "accuracy"]


def fila_de_fold(fold: int, resultados: dict) -> dict:
    """Métricas de la época elegida en un fold, más su coste."""
    e = resultados.get("mejor_epoca", {})
    epocas = resultados.get("epocas", [])
    return {
        "fold": fold,
        **{m: e.get(m, float("nan")) for m in METRICAS},
        **e.get("matriz_confusion", {}),
        "epocas_entrenadas": len(epocas),
        "segundos_por_epoca": float(np.mean([x["segundos"] for x in epocas])) if epocas else float("nan"),
    }


def media_y_desviacion(filas: list[dict]) -> dict:
    """Media y desviación típica (muestral, ddof=1) de cada métrica entre folds."""
    tabla = pd.DataFrame(filas)
    resumen = {}
    for m in [*METRICAS, "segundos_por_epoca"]:
        valores = tabla[m].astype(float).dropna()
        resumen[m] = {
            "media": float(valores.mean()) if len(valores) else float("nan"),
            "desviacion": float(valores.std(ddof=1)) if len(valores) > 1 else float("nan"),
        }
    return resumen


def resumir(carpeta: Path, por_fold: dict[int, dict], cabecera: dict) -> dict:
    """Junta los folds terminados: tabla por fold, media ± desviación y métricas fuera de fold."""
    filas = [fila_de_fold(k, r) for k, r in sorted(por_fold.items())]
    resumen = {**cabecera, "folds_terminados": sorted(por_fold), "por_fold": filas,
               "media_desviacion": media_y_desviacion(filas)}

    # Predicciones fuera de fold: cada paciente, predicha por el modelo que no la vio.
    partes = [pd.read_csv(carpeta / f"fold_{k}" / "predicciones_val.csv") for k in sorted(por_fold)
              if (carpeta / f"fold_{k}" / "predicciones_val.csv").exists()]
    if partes:
        oof = pd.concat(partes, ignore_index=True)
        pacientes = tabla_por_paciente(oof)
        oof.to_csv(carpeta / "predicciones_oof.csv", index=False)
        pacientes.to_csv(carpeta / "predicciones_oof_pacientes.csv", index=False)

        m = evaluar(oof.prob.values, oof)
        n_pcr = int(pacientes.pCR.sum())
        resumen["fuera_de_fold"] = {
            "pacientes": m["pacientes"], "agregacion": METODO_AGREGACION, "umbral": UMBRAL,
            "auc": m["auc"], "matriz_confusion": m["matriz_confusion"],
            **{k: m[k] for k in ("sensibilidad", "especificidad", "precision_pcr", "recall_pcr", "f1_pcr",
                                 "precision_no_pcr", "recall_no_pcr", "f1_no_pcr", "f1_macro", "accuracy")},
            "por_cohorte": auc_por_cohorte(pacientes),
            "senal_de_hospital": senal_de_hospital(pacientes),
            "referencias_triviales": referencias_triviales(n_pcr, len(pacientes) - n_pcr),
        }

    (carpeta / "resumen.json").write_text(json.dumps(resumen, indent=2, ensure_ascii=False), encoding="utf-8")
    tabla = pd.DataFrame(filas)
    md = resumen["media_desviacion"]
    tabla = pd.concat([tabla, pd.DataFrame([
        {"fold": "media", **{k: v["media"] for k, v in md.items()}},
        {"fold": "desviacion", **{k: v["desviacion"] for k, v in md.items()}},
    ])], ignore_index=True)
    tabla.to_csv(carpeta / "resumen.csv", index=False)
    dibujar_graficas(carpeta)   # graficas/: curvas de todos los folds y matriz fuera de fold
    return resumen


def imprimir(resumen: dict) -> None:
    md = resumen["media_desviacion"]

    def fmt(m: str) -> str:
        media, desv = md[m]["media"], md[m]["desviacion"]
        return f"{media:.3f}" + ("" if math.isnan(desv) else f" ± {desv:.3f}")

    print("\n" + "=" * 72)
    print(f"Validación cruzada · {resumen['modelo']} · pérdida {resumen['perdida']} · "
          f"folds {resumen['folds_terminados']}")
    print(f"  AUC validación   {fmt('auc_val')}")
    print(f"  AUC entrenam.    {fmt('auc_train')}")
    print(f"  diferencia       {fmt('diferencia_auc')}")
    print(f"  sensibilidad     {fmt('sensibilidad')}   especificidad {fmt('especificidad')}")
    print(f"  F1 pCR {fmt('f1_pcr')} · F1 no pCR {fmt('f1_no_pcr')} · F1 macro {fmt('f1_macro')}")
    print(f"  época elegida    {fmt('epoca')}")
    oof = resumen.get("fuera_de_fold")
    if oof:
        mc = oof["matriz_confusion"]
        print(f"Fuera de fold ({oof['pacientes']} pacientes): AUC {oof['auc']:.3f} | "
              f"VP {mc['VP']} FN {mc['FN']} FP {mc['FP']} VN {mc['VN']}")
        print("  por cohorte: " + " · ".join(f"{c} {v['auc']:.3f} (n={v['pacientes']})"
                                             for c, v in oof["por_cohorte"].items()))
        if oof.get("senal_de_hospital"):
            print("  ¿delata el hospital? (0,5 = no): " + " · ".join(
                f"{c} {v:.3f}" for c, v in oof["senal_de_hospital"].items()))
    print("=" * 72)


def validar(modo: str, modelo: str, ponderada: bool = False, nota: str = "", cambios: dict | None = None,
            folds: list[int] | None = None) -> Path:
    """Entrena `modelo` en los folds indicados (por defecto los 5) y devuelve la carpeta de la validación."""
    cambios = dict(cambios or {})
    folds = list(folds) if folds is not None else [0, 1, 2, 3, 4]
    if any(k not in range(5) for k in folds):
        raise SystemExit("Los folds van de 0 a 4.")
    sufijo = "_ponderada" if ponderada else ""
    carpeta = CARPETA_RUNS / f"{datetime.now().astimezone():%Y%m%d-%H%M%S}_cv_{modelo}_{modo}{sufijo}"
    carpeta.mkdir(parents=True, exist_ok=True)

    cabecera = {"ejecucion": carpeta.name, "modelo": modelo,
                "perdida": "ponderada" if ponderada else "normal", "nota": nota,
                "config": obtener_config(modo, **cambios).como_dict()}
    inicio, por_fold = time.perf_counter(), {}
    for k in folds:
        print(f"\n{'#' * 72}\n# Fold {k} de validación ({folds.index(k) + 1}/{len(folds)})\n{'#' * 72}")
        cfg = obtener_config(modo, **{**cambios, "fold_val": k})
        por_fold[k] = ejecutar(cfg, modelo, ponderada, nota, carpeta=carpeta / f"fold_{k}", registrar=False)
        cabecera["maquina"] = por_fold[k]["maquina"]
        cabecera["tiempo_total_s"] = time.perf_counter() - inicio
        resumen = resumir(carpeta, por_fold, cabecera)
        historico.actualizar(CARPETA_RUNS)

    imprimir(resumen)
    print(f"Resumen en {carpeta / 'resumen.json'} · gráficas en {carpeta / 'graficas'}")
    registrar_ejecucion()
    return carpeta


def main() -> None:
    p = anadir_argumentos(argparse.ArgumentParser(
        description="Entrena una CNN en los 5 folds (validación cruzada por paciente)."), con_fold=False)
    p.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4],
                   help="folds a usar como validación (por defecto los 5)")
    args = p.parse_args()
    validar(args.modo, args.modelo, args.ponderada, args.nota, cambios_de_config(args), args.folds)


if __name__ == "__main__":
    main()
