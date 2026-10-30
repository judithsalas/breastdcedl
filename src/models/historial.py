"""El histórico de ejecuciones (runs/historico.csv) y las reglas que eligen con él.

- `validaciones_validas`: la última validación cruzada de 5 folds de cada versión que cuenta.
- `elegir_mejor`: el criterio de comparación (D16), para quedarse con una versión.
- `elegir_conjunto`: la regla del conjunto (D36), con la que entraron las 6 versiones del modelo
  final.

Los números salen de runs/historico.csv (D13), que se queda en local; el modo rápido no cuenta.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import COMPARACION, es_estandar
from src.data.caso import RAIZ_REPO
from src.evaluation.metricas import coma
from src.models.construir import contar_parametros, crear_modelo, huella, receta
from src.models.versiones import VERSIONES

HISTORICO = RAIZ_REPO / "runs" / "historico.csv"


def leer_historico(ruta: Path = HISTORICO, modo: str = "completo") -> pd.DataFrame:
    """Ejecuciones de un modo del histórico (por defecto, las completas; vacío si aún no hay)."""
    if not Path(ruta).exists():
        return pd.DataFrame()
    tabla = pd.read_csv(ruta, dtype={"huella": str, "folds": str})
    return tabla[tabla.modo == modo].sort_values("ejecucion")


TODOS_LOS_FOLDS = "0,1,2,3,4"
# Diferencia de AUC train − val a partir de la cual se considera que sobreajusta (config.json, D16).
SOBREAJUSTE = COMPARACION["sobreajuste_maximo"]


def validaciones_validas(historico: pd.DataFrame, perdida: str = "normal") -> pd.DataFrame:
    """Última validación cruzada de 5 folds de cada versión, con huella ✓, la pérdida pedida y el
    batch, el learning rate y la semilla estándar de su modo (config.json, D31)."""
    if historico.empty:
        return historico
    filas = historico[(historico.tipo == "cv") & (historico.perdida == perdida)
                      & (historico.folds.astype(str) == TODOS_LOS_FOLDS)
                      & historico.modelo.isin(list(VERSIONES))]
    filas = filas[filas.apply(lambda f: _huella_ok(f) is True, axis=1)] if len(filas) else filas
    filas = filas[filas.apply(lambda f: es_estandar(f.modo, f.get("batch"), f.get("lr"), f.get("semilla"),
                                                    receta(f.modelo).get("learning_rate")),
                              axis=1)] if len(filas) else filas
    return filas.groupby("modelo").tail(1)


def desplegable(modelo: str) -> bool:
    """¿Puede ser el modelo final? Solo si es una versión definida: todas usan solo las 3 fases, lo
    mismo que tiene la app."""
    return modelo in VERSIONES


def elegir_mejor(historico: pd.DataFrame) -> dict | None:
    """Aplica el criterio de comparación (D16).

    1. Validaciones cruzadas completas de 5 folds, pérdida normal y huella ✓.
    2. La de mayor AUC medio en validación es la referencia.
    3. Empatan con ella las que quedan a menos de una desviación (la de la referencia).
    4. Entre las empatadas, primero las que no sobreajustan (diferencia train − val ≤ 0,15);
       después, la de menos parámetros; a igualdad, la más antigua.
    """
    candidatas = validaciones_validas(historico, "normal")
    if not candidatas.empty:
        candidatas = candidatas[candidatas.modelo.map(desplegable)]
    if candidatas.empty:
        return None
    referencia = candidatas.loc[candidatas.auc_val.idxmax()]
    margen = 0.0 if pd.isna(referencia.auc_val_desv) else float(referencia.auc_val_desv)
    empatadas = candidatas[candidatas.auc_val >= referencia.auc_val - margen]
    orden = list(VERSIONES)
    def sobreajusta(modelo: str) -> bool:
        dif = empatadas.loc[empatadas.modelo == modelo, "diferencia_auc"].iloc[0]
        return bool(not pd.isna(dif) and dif > SOBREAJUSTE)

    ganadora = min(empatadas.modelo,
                   key=lambda m: (sobreajusta(m), contar_parametros(crear_modelo(m)), orden.index(m)))
    fila = candidatas[candidatas.modelo == ganadora].iloc[0]
    if ganadora == referencia.modelo:
        motivo = f"mayor AUC medio ({coma(referencia.auc_val, 3)} ± {coma(margen, 3)}, {referencia.modelo})"
    else:
        if sobreajusta(referencia.modelo) and not sobreajusta(ganadora):
            desempate = (f"{referencia.modelo} sobreajusta (diferencia {coma(referencia.diferencia_auc, 3)} > "
                         f"{coma(SOBREAJUSTE, 2)}) y {ganadora} no ({coma(fila.diferencia_auc, 3)})")
        elif contar_parametros(crear_modelo(ganadora)) < contar_parametros(crear_modelo(referencia.modelo)):
            desempate = f"{ganadora} tiene menos parámetros"
        else:
            desempate = f"{ganadora} es más antigua (cambia menos cosas)"
        motivo = (f"empata con {referencia.modelo} ({coma(referencia.auc_val, 3)} ± {coma(margen, 3)}); "
                  f"{desempate}")
    return {"version": ganadora, "auc_val": float(fila.auc_val), "auc_val_desv": fila.auc_val_desv,
            "empatadas": sorted(empatadas.modelo, key=orden.index), "referencia": referencia.modelo,
            "motivo": motivo, "ejecucion": fila.ejecucion}


def elegir_conjunto(historico: pd.DataFrame) -> dict | None:
    """Versiones que entran en el modelo final como conjunto (D36).

    Regla, la misma del criterio de comparación (D16) pero sin quedarse con una sola: entran todas
    las versiones con validación cruzada completa de 5 folds, pérdida normal, huella ✓, valores
    estándar del modo y desplegables, **que no sobreajustan** (diferencia train − val ≤ 0,15). No se
    mira cuánto sube el conjunto con unas u otras: elegirlas por eso sería ajustar con validación
    hasta encontrar la mezcla que más da.
    """
    candidatas = validaciones_validas(historico, "normal")
    if not candidatas.empty:
        candidatas = candidatas[candidatas.modelo.map(desplegable)]
    if candidatas.empty:
        return None
    orden = list(VERSIONES)
    candidatas = candidatas.assign(_orden=candidatas.modelo.map(orden.index)).sort_values("_orden")
    sobreajustan = candidatas.diferencia_auc.notna() & (candidatas.diferencia_auc > SOBREAJUSTE)
    entran, fuera = candidatas[~sobreajustan], candidatas[sobreajustan]
    if entran.empty:
        return None
    excluidas = {f.modelo: f"sobreajusta (diferencia {coma(f.diferencia_auc, 3)} > {coma(SOBREAJUSTE, 2)})"
                 for f in fuera.itertuples()}
    return {"versiones": list(entran.modelo), "ejecuciones": list(entran.ejecucion),
            "auc_val": dict(zip(entran.modelo, entran.auc_val.astype(float), strict=True)),
            "diferencia_auc": dict(zip(entran.modelo, entran.diferencia_auc.astype(float), strict=True)),
            "excluidas": excluidas,
            "motivo": (f"todas las versiones con validación cruzada completa que no sobreajustan "
                       f"(diferencia ≤ {coma(SOBREAJUSTE, 2)}): {', '.join(entran.modelo)}")}


def _huella_ok(fila: pd.Series) -> bool | None:
    """True si la ejecución usó la definición actual, False si no, None si no se sabe."""
    if fila.modelo not in VERSIONES or pd.isna(fila.get("huella")):
        return None
    return fila.huella == huella(fila.modelo)
