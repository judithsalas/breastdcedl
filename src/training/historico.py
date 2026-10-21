"""Histórico de ejecuciones: runs/historico.csv, una fila por ejecución.

    python -m src.training.historico          # rehace el histórico mirando runs/ y lo muestra

Se actualiza solo al terminar cada entrenamiento y después de cada fold de una
validación cruzada. Cada fila resume la época elegida de una ejecución (o la media de
los folds, en una validación cruzada); el detalle sigue en la carpeta de la ejecución.

Las filas se identifican por el nombre de la carpeta (`ejecucion`). Al actualizar se
recalculan las de las carpetas que hay en ESTA máquina y se conservan las demás. Así el
histórico se puede versionar (es la única excepción dentro de runs/) y junta lo
entrenado en el portátil y en la universidad sin perder filas.

También recoge ejecuciones anteriores a este fichero (R1, R2): su época elegida se saca
del historial por épocas, y el AUC de entrenamiento queda vacío porque no se medía.

La columna `huella` identifica la definición exacta de la arquitectura entrenada (ver
`src.models.construir.huella`): si no coincide con la actual, la versión se cambió después de entrenarla.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.caso import RAIZ_REPO
from src.evaluation.metricas import metricas_por_clase
from src.models.construir import huella

CARPETA_RUNS = RAIZ_REPO / "runs"
RUTA = CARPETA_RUNS / "historico.csv"

COLUMNAS = [
    "fecha", "ejecucion", "tipo", "modo", "modelo", "huella", "parametros", "perdida", "folds",
    "epoca_elegida", "epocas_entrenadas", "auc_val", "auc_val_desv", "auc_train", "diferencia_auc",
    "sensibilidad", "especificidad", "f1_pcr", "f1_no_pcr", "f1_macro", "umbral", "agregacion",
    "batch", "lr", "semilla", "segundos_por_epoca", "tiempo_total_min", "dispositivo", "torch", "nota",
]


def _leer(ruta: Path) -> dict:
    return json.loads(ruta.read_text(encoding="utf-8"))


def _fecha(nombre: str) -> str:
    """'20261007-211625_cnn_v1...' -> '2026-10-07 21:16'."""
    m = re.match(r"(\d{4})(\d{2})(\d{2})-(\d{2})(\d{2})", nombre)
    return f"{m[1]}-{m[2]}-{m[3]} {m[4]}:{m[5]}" if m else ""


def _dispositivo(maquina: dict) -> str:
    return f"{maquina.get('backend', '')} {maquina.get('gpu') or ''}".strip()


def _comunes(nombre: str, cfg: dict, maquina: dict, nota: str) -> dict:
    return {"fecha": _fecha(nombre), "ejecucion": nombre, "modo": cfg.get("modo"),
            "batch": cfg.get("batch_size"), "lr": cfg.get("learning_rate"), "semilla": cfg.get("semilla"),
            "dispositivo": _dispositivo(maquina), "torch": maquina.get("torch"), "nota": nota}


def _mejor_epoca(r: dict) -> dict | None:
    """Época elegida de una ejecución; en las antiguas se reconstruye desde las épocas."""
    if r.get("mejor_epoca"):
        return r["mejor_epoca"]
    mejores = [e for e in r.get("epocas", []) if e.get("mejor")]
    if not mejores:
        return None
    e = mejores[-1]
    matriz = {k: e[k] for k in ("VP", "VN", "FP", "FN") if k in e}
    return {"epoca": e["epoca"], "auc_val": e["auc"], "auc_train": e.get("auc_tr", float("nan")),
            "diferencia_auc": e.get("diferencia_auc", float("nan")), "sensibilidad": e["sensibilidad"],
            "especificidad": e["especificidad"], **(metricas_por_clase(matriz) if len(matriz) == 4 else {})}


def fila_ejecucion(carpeta: Path) -> dict | None:
    """Fila de una ejecución normal (resultados.json)."""
    r = _leer(carpeta / "resultados.json")
    e = _mejor_epoca(r)
    if e is None:
        return None
    epocas = r.get("epocas", [])
    cfg = r.get("config", {})
    return {
        **_comunes(carpeta.name, cfg, r.get("maquina", {}), r.get("nota", "")),
        "tipo": "simple", "modelo": r.get("modelo"), "huella": huella(r.get("arquitectura")),
        "parametros": r.get("parametros"),
        "perdida": r.get("perdida"), "folds": str(cfg.get("fold_val", "")),
        "epoca_elegida": e.get("epoca"), "epocas_entrenadas": len(epocas),
        "auc_val": e.get("auc_val"), "auc_train": e.get("auc_train"), "diferencia_auc": e.get("diferencia_auc"),
        **{k: e.get(k) for k in ("sensibilidad", "especificidad", "f1_pcr", "f1_no_pcr", "f1_macro")},
        "umbral": r.get("umbral"), "agregacion": r.get("agregacion"),
        "segundos_por_epoca": float(np.mean([x["segundos"] for x in epocas])) if epocas else None,
        "tiempo_total_min": r.get("tiempo_total_s", 0) / 60,
    }


def fila_validacion_cruzada(carpeta: Path) -> dict | None:
    """Fila de una validación cruzada (resumen.json): media de los folds, con su desviación del AUC."""
    s = _leer(carpeta / "resumen.json")
    md = s.get("media_desviacion", {})
    if not md:
        return None
    media = {k: v["media"] for k, v in md.items()}
    folds = sorted(carpeta.glob("fold_*/resultados.json"))
    primero = _leer(folds[0]) if folds else {}
    oof = s.get("fuera_de_fold", {})
    return {
        **_comunes(carpeta.name, s.get("config", {}), s.get("maquina", {}), s.get("nota", "")),
        "tipo": "cv", "modelo": s.get("modelo"), "huella": huella(primero.get("arquitectura")),
        "parametros": primero.get("parametros"), "perdida": s.get("perdida"),
        "folds": ",".join(str(k) for k in s.get("folds_terminados", [])),
        "epoca_elegida": media.get("epoca"), "epocas_entrenadas": None,
        "auc_val": media.get("auc_val"), "auc_val_desv": md.get("auc_val", {}).get("desviacion"),
        "auc_train": media.get("auc_train"), "diferencia_auc": media.get("diferencia_auc"),
        **{k: media.get(k) for k in ("sensibilidad", "especificidad", "f1_pcr", "f1_no_pcr", "f1_macro")},
        "umbral": oof.get("umbral"), "agregacion": oof.get("agregacion"),
        "segundos_por_epoca": media.get("segundos_por_epoca"),
        "tiempo_total_min": s.get("tiempo_total_s", 0) / 60,
    }


def filas_locales(raiz: Path = CARPETA_RUNS) -> list[dict]:
    """Una fila por carpeta de primer nivel de runs/ (los folds de una validación cruzada no)."""
    raiz = Path(raiz)
    if not raiz.exists():
        return []
    filas = []
    for carpeta in sorted(p for p in raiz.iterdir() if p.is_dir()):
        try:
            if (carpeta / "resumen.json").exists():
                fila = fila_validacion_cruzada(carpeta)
            elif (carpeta / "resultados.json").exists():
                fila = fila_ejecucion(carpeta)
            else:
                continue
        except (json.JSONDecodeError, KeyError, TypeError) as error:
            print(f"Aviso: no se pudo leer {carpeta.name} para el histórico ({error})")
            continue
        if fila:
            filas.append(fila)
    return filas


def actualizar(raiz: Path = CARPETA_RUNS) -> Path:
    """Recalcula las filas de las ejecuciones de esta máquina y conserva las de otras."""
    raiz = Path(raiz)
    ruta = raiz / "historico.csv"
    locales = pd.DataFrame(filas_locales(raiz), columns=COLUMNAS)
    if ruta.exists():
        anterior = pd.read_csv(ruta)
        anterior = anterior[~anterior.ejecucion.isin(locales.ejecucion)]
        tabla = pd.concat([anterior, locales], ignore_index=True) if len(anterior) else locales
    else:
        tabla = locales
    tabla = tabla.reindex(columns=[*COLUMNAS, *[c for c in tabla.columns if c not in COLUMNAS]])
    for columna in ("parametros", "epocas_entrenadas", "batch", "semilla"):   # enteros, aunque haya vacíos
        tabla[columna] = pd.to_numeric(tabla[columna], errors="coerce").round().astype("Int64")
    raiz.mkdir(parents=True, exist_ok=True)
    # 6 cifras: con 4, un AUC de 0,57449 se guardaba como 0,5745 y la tabla lo redondeaba a 0,575.
    tabla.sort_values("ejecucion").to_csv(ruta, index=False, float_format="%.6g")
    return ruta


def main() -> None:
    from src.models.historial import actualizar_documento  # tablas de docs/ARQUITECTURAS.md

    ruta = actualizar()
    if actualizar_documento():
        print("Tablas de docs/ARQUITECTURAS.md actualizadas.")
    tabla = pd.read_csv(ruta)
    print(f"{len(tabla)} ejecuciones en {ruta.relative_to(RAIZ_REPO)}\n")
    if len(tabla):
        columnas = ["fecha", "tipo", "modo", "modelo", "perdida", "folds", "epoca_elegida",
                    "auc_val", "auc_val_desv", "auc_train", "diferencia_auc", "f1_macro"]
        with pd.option_context("display.width", 160, "display.max_columns", 20):
            print(tabla[columnas].tail(20).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
