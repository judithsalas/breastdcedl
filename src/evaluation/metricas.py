"""Métricas por paciente a partir de la matriz de confusión, y su dibujo.

`uc.evaluar_por_paciente` (material del profesor, no se modifica) agrega los cortes
por paciente y devuelve la matriz de confusión, la sensibilidad, la especificidad y
el AUC. La rúbrica pide además precision, recall y F1 de CADA clase, la F1
equilibrada (macro) y la comparación con decir siempre «no pCR». Todo eso sale de
la misma matriz, así que se calcula aquí sin repetir la agregación.

    m = uc.evaluar_por_paciente(probs, filas, umbral=0.5, metodo="mean")
    m.update(metricas_por_clase(m["matriz_confusion"]))
    dibujar_matriz_confusion(m["matriz_confusion"], "matriz.png", "cnn_v3 · fold 0")

Convención: la clase positiva es pCR = 1. Para la clase pCR = 0 los papeles se
invierten: su recall es la especificidad y su precision es el valor predictivo
negativo (VPN).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

if TYPE_CHECKING:
    import pandas as pd


def _division(a: float, b: float) -> float:
    return a / b if b else float("nan")


def coma(valor: float, decimales: int = 2) -> str:
    """Número con coma decimal, para los textos de las figuras."""
    return f"{valor:.{decimales}f}".replace(".", ",")


def _f1(aciertos: int, falsos_pos: int, falsos_neg: int) -> float:
    """F1 = 2·VP / (2·VP + FP + FN). Vale 0 (no NaN) si el modelo nunca predice esa clase."""
    return _division(2 * aciertos, 2 * aciertos + falsos_pos + falsos_neg)


def metricas_por_clase(matriz: dict) -> dict:
    """Precision, recall y F1 de cada clase, F1 macro y exactitud, desde {VP, VN, FP, FN}."""
    vp, vn, fp, fn = (matriz[k] for k in ("VP", "VN", "FP", "FN"))
    p1, r1 = _division(vp, vp + fp), _division(vp, vp + fn)   # clase pCR = 1
    p0, r0 = _division(vn, vn + fn), _division(vn, vn + fp)   # clase pCR = 0
    f1_1, f1_0 = _f1(vp, fp, fn), _f1(vn, fn, fp)
    return {
        "precision_pcr": p1, "recall_pcr": r1, "f1_pcr": f1_1,
        "precision_no_pcr": p0, "recall_no_pcr": r0, "f1_no_pcr": f1_0,
        "f1_macro": float(np.mean([f1_1, f1_0])),
        "accuracy": _division(vp + vn, vp + vn + fp + fn),
    }


def referencias_triviales(n_pcr: int, n_no_pcr: int) -> dict:
    """Lo que sacarían dos modelos que no miran la imagen: decir siempre una clase.

    Un modelo propio tiene que superar claramente a estos dos en F1 macro, y su AUC
    tiene que estar por encima de 0,5 (el AUC de ambos).
    """
    siempre_no = {"VP": 0, "VN": n_no_pcr, "FP": 0, "FN": n_pcr}
    siempre_si = {"VP": n_pcr, "VN": 0, "FP": n_no_pcr, "FN": 0}
    return {
        "siempre_no_pcr": {"matriz_confusion": siempre_no, **metricas_por_clase(siempre_no), "auc": 0.5},
        "siempre_pcr": {"matriz_confusion": siempre_si, **metricas_por_clase(siempre_si), "auc": 0.5},
    }


# --------------------------------------------------------------------------- #
# Dibujo
# --------------------------------------------------------------------------- #

# Rampa secuencial de un solo tono (azul), de claro a oscuro: más pacientes = más oscuro.
_RAMPA = LinearSegmentedColormap.from_list("azul", ["#f4f8fd", "#cde2fb", "#86b6ef", "#2a78d6", "#104281"])
_TINTA, _TINTA_SUAVE = "#0b0b0b", "#52514e"


def dibujar_matriz_confusion(matriz: dict, ruta: Path | str, titulo: str, subtitulo: str = "") -> Path:
    """Matriz 2x2 por paciente: filas = clase real, columnas = clase predicha.

    El color de cada celda es el porcentaje de su fila (qué fracción de las pacientes
    de esa clase real acaba en cada predicción); el texto da el número de pacientes y
    ese porcentaje. Así la diagonal se lee directamente como especificidad (arriba) y
    sensibilidad (abajo).
    """
    vp, vn, fp, fn = (matriz[k] for k in ("VP", "VN", "FP", "FN"))
    cuentas = np.array([[vn, fp], [fn, vp]], dtype=float)
    filas = cuentas.sum(axis=1, keepdims=True)
    fraccion = np.divide(cuentas, filas, out=np.zeros_like(cuentas), where=filas > 0)
    nombres = [["VN", "FP"], ["FN", "VP"]]

    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    ax.imshow(fraccion, cmap=_RAMPA, vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            oscuro = fraccion[i, j] > 0.55
            color = "white" if oscuro else _TINTA
            ax.text(j, i - 0.12, f"{int(cuentas[i, j])}", ha="center", va="center", fontsize=20,
                    fontweight="bold", color=color)
            ax.text(j, i + 0.2, f"{nombres[i][j]} · {fraccion[i, j]:.0%} de la fila", ha="center",
                    va="center", fontsize=8.5, color=color if oscuro else _TINTA_SUAVE)
    ax.set_xticks([0, 1], ["no pCR", "pCR"])
    ax.set_yticks([0, 1], ["no pCR", "pCR"])
    ax.set_xlabel("Predicción", color=_TINTA_SUAVE)
    ax.set_ylabel("Real", color=_TINTA_SUAVE)
    ax.tick_params(length=0, labelcolor=_TINTA)
    for borde in ax.spines.values():
        borde.set_visible(False)
    # Separación de 2 px entre celdas, del color del fondo.
    ax.axhline(0.5, color="white", linewidth=2)
    ax.axvline(0.5, color="white", linewidth=2)
    ax.set_title(titulo, fontsize=11, fontweight="bold", color=_TINTA, pad=22)
    if subtitulo:
        ax.text(0.5, 1.03, subtitulo, transform=ax.transAxes, ha="center", va="bottom", fontsize=8.5,
                color=_TINTA_SUAVE)

    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ruta, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return ruta


# --------------------------------------------------------------------------- #
# Métricas a partir de probabilidades por paciente (cierre del modelo y test)
# --------------------------------------------------------------------------- #

def matriz_con_umbral(y: np.ndarray, p: np.ndarray, umbral: float) -> dict:
    """Matriz de confusión {VP, VN, FP, FN} prediciendo pCR si p ≥ umbral."""
    y, pred = np.asarray(y).astype(int), (np.asarray(p) >= umbral).astype(int)
    return {"VP": int(((pred == 1) & (y == 1)).sum()), "VN": int(((pred == 0) & (y == 0)).sum()),
            "FP": int(((pred == 1) & (y == 0)).sum()), "FN": int(((pred == 0) & (y == 1)).sum())}


def metricas_con_umbral(y: np.ndarray, p: np.ndarray, umbral: float) -> dict:
    """Matriz, sensibilidad, especificidad y métricas por clase con un umbral dado."""
    m = matriz_con_umbral(y, p, umbral)
    clase = metricas_por_clase(m)
    return {"umbral": float(umbral), "matriz_confusion": m, "sensibilidad": clase["recall_pcr"],
            "especificidad": clase["recall_no_pcr"], **clase}


def ic_bootstrap(y: np.ndarray, p: np.ndarray, funcion, n: int = 2000, semilla: int = 42,
                 confianza: float = 0.95) -> tuple[float, float]:
    """Intervalo de confianza por bootstrap: se remuestrean PACIENTES con reemplazo.

    `funcion(y, p)` es la métrica. Las muestras con una sola clase se saltan (el AUC no existe).
    """
    y, p = np.asarray(y), np.asarray(p)
    rng = np.random.default_rng(semilla)
    valores = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) < 2:
            continue
        valores.append(funcion(y[i], p[i]))
    alfa = (1 - confianza) / 2
    return float(np.quantile(valores, alfa)), float(np.quantile(valores, 1 - alfa))


def auc_por_rangos(y: np.ndarray, p: np.ndarray) -> float:
    """El mismo AUC que sklearn (Mann-Whitney, empates a la mitad), mucho más rápido: para el bootstrap.

    AUC = probabilidad de que una paciente con pCR tenga más puntuación que una sin pCR.
    """
    from scipy.stats import rankdata

    y = np.asarray(y).astype(bool)
    rangos = rankdata(np.asarray(p, float))
    n1 = int(y.sum())
    n0 = len(y) - n1
    return float((rangos[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def auc_con_ic(y: np.ndarray, p: np.ndarray, n: int = 2000, semilla: int = 42) -> dict:
    from sklearn.metrics import roc_auc_score

    if len(np.unique(y)) < 2:
        return {"auc": float("nan"), "ic95": [float("nan"), float("nan")]}
    inferior, superior = ic_bootstrap(y, p, auc_por_rangos, n, semilla)
    return {"auc": float(roc_auc_score(y, p)), "ic95": [inferior, superior]}


def diferencia_auc(y: np.ndarray, p1: np.ndarray, p2: np.ndarray, n: int = 2000, semilla: int = 42) -> dict:
    """AUC(p1) − AUC(p2) en las MISMAS pacientes, con IC 95 % por bootstrap pareado.

    En cada remuestreo se toman las mismas pacientes para los dos modelos: así el intervalo mide
    la diferencia entre modelos y no la variación de la muestra, que comparten.
    """
    pares = np.column_stack([np.asarray(p1, float), np.asarray(p2, float)])

    def diferencia(yy, pp):
        return auc_por_rangos(yy, pp[:, 0]) - auc_por_rangos(yy, pp[:, 1])

    inferior, superior = ic_bootstrap(y, pares, diferencia, n, semilla)
    return {"diferencia": float(diferencia(np.asarray(y), pares)), "ic95": [inferior, superior]}


def _auc_media_folds(y: np.ndarray, columnas: np.ndarray, grupos: list[np.ndarray]) -> np.ndarray:
    """Media, sobre los folds, del AUC de cada columna de `columnas` (pacientes × modelos)."""
    return np.mean([[auc_por_rangos(y[g], columnas[g, j]) for j in range(columnas.shape[1])] for g in grupos], axis=0)


def _bootstrap_por_fold(y: np.ndarray, columnas: np.ndarray, folds: np.ndarray, n: int, semilla: int) -> np.ndarray:
    """Remuestrea pacientes DENTRO de cada fold (con reemplazo) y devuelve el AUC medio de cada columna."""
    rng = np.random.default_rng(semilla)
    indices = [np.flatnonzero(folds == k) for k in np.unique(folds)]
    salida = []
    for _ in range(n):
        grupos = [rng.choice(g, len(g)) for g in indices]
        if any(len(np.unique(y[g])) < 2 for g in grupos):
            continue
        salida.append(_auc_media_folds(y, columnas, grupos))
    return np.array(salida)


def auc_media_folds(y: np.ndarray, p: np.ndarray, folds: np.ndarray, n: int = 2000, semilla: int = 42) -> dict:
    """La cifra principal (rúbrica, punto 2): media del AUC de cada fold, con su desviación y su IC 95 %.

    El intervalo sale de remuestrear pacientes dentro de cada fold y volver a hacer la media. Mide la
    incertidumbre por las pacientes que tocan; no la de volver a entrenar las redes.
    """
    y, folds = np.asarray(y), np.asarray(folds)
    columnas = np.asarray(p, float).reshape(-1, 1)
    grupos = [np.flatnonzero(folds == k) for k in np.unique(folds)]
    por_fold = [auc_por_rangos(y[g], columnas[g, 0]) for g in grupos]
    muestras = _bootstrap_por_fold(y, columnas, folds, n, semilla)[:, 0]
    return {"auc": float(np.mean(por_fold)), "desviacion": float(np.std(por_fold, ddof=1)) if len(por_fold) > 1 else 0.0,
            "por_fold": [float(v) for v in por_fold],
            "ic95": [float(np.quantile(muestras, 0.025)), float(np.quantile(muestras, 0.975))]}


def diferencia_media_folds(y: np.ndarray, p1: np.ndarray, p2: np.ndarray, folds: np.ndarray, n: int = 2000,
                           semilla: int = 42) -> dict:
    """AUC medio de folds de p1 menos el de p2, en las MISMAS pacientes, con IC 95 % por bootstrap pareado.

    Es la diferencia entre las cifras principales de los dos modelos (media de folds), así que cuadra con
    ellas. En cada remuestreo, dentro de cada fold, se toman las mismas pacientes para los dos modelos.
    """
    y, folds = np.asarray(y), np.asarray(folds)
    columnas = np.column_stack([np.asarray(p1, float), np.asarray(p2, float)])
    grupos = [np.flatnonzero(folds == k) for k in np.unique(folds)]
    por_fold = [auc_por_rangos(y[g], columnas[g, 0]) - auc_por_rangos(y[g], columnas[g, 1]) for g in grupos]
    muestras = _bootstrap_por_fold(y, columnas, folds, n, semilla)
    diferencias = muestras[:, 0] - muestras[:, 1]
    return {"diferencia": float(np.mean(por_fold)), "por_fold": [float(v) for v in por_fold],
            "ic95": [float(np.quantile(diferencias, 0.025)), float(np.quantile(diferencias, 0.975))]}


def brier(y: np.ndarray, p: np.ndarray) -> float:
    """Error cuadrático medio entre probabilidad y resultado (0 perfecto; 0,25 = decir siempre 0,5)."""
    return float(np.mean((np.asarray(p, float) - np.asarray(y, float)) ** 2))


def tabla_calibracion(y: np.ndarray, p: np.ndarray, grupos: int = 10) -> list[dict]:
    """Agrupa por deciles de probabilidad: probabilidad media predicha frente a proporción real de pCR."""
    import pandas as pd

    datos = pd.DataFrame({"y": np.asarray(y, float), "p": np.asarray(p, float)})
    datos["grupo"] = pd.qcut(datos.p.rank(method="first"), q=min(grupos, len(datos)), labels=False)
    return [{"prob_media": float(g.p.mean()), "proporcion_real": float(g.y.mean()), "pacientes": len(g)}
            for _, g in datos.groupby("grupo")]


def auc_por_cohorte(pacientes: pd.DataFrame) -> dict:
    """AUC por paciente dentro de cada cohorte (NaN si en una cohorte solo hay una clase).

    `pacientes` necesita las columnas cohorte, pCR y prob_media (como predicciones_val_pacientes.csv).
    """
    from sklearn.metrics import roc_auc_score

    salida = {}
    for cohorte, grupo in pacientes.groupby("cohorte"):
        auc = (float(roc_auc_score(grupo.pCR, grupo.prob_media)) if grupo.pCR.nunique() == 2
               else float("nan"))
        salida[str(cohorte)] = {"pacientes": len(grupo), "pcr": int(grupo.pCR.sum()), "auc": auc}
    return salida


def senal_de_hospital(pacientes: pd.DataFrame) -> dict:
    """¿La puntuación de la red delata el hospital? (D28)

    AUC de la probabilidad para distinguir cada cohorte del resto. 0,5 = mirando la puntuación
    no se sabe de qué hospital es la paciente; cuanto más lejos de 0,5 (por arriba o por abajo),
    más «sabe» la red el hospital. Con cnn_v3 salió 0,66 para I-SPY2 y 0,28 para I-SPY1; si la
    puntuación solo reflejara la pCR, saldría 0,49–0,51 (R5).
    """
    from sklearn.metrics import roc_auc_score

    salida = {}
    for cohorte in sorted(pacientes.cohorte.dropna().unique()):
        es = pacientes.cohorte == cohorte
        if 0 < es.sum() < len(pacientes):
            salida[str(cohorte)] = float(roc_auc_score(es, pacientes.prob_media))
    return salida
