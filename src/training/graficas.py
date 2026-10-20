"""Gráficas de una ejecución, guardadas aparte en <ejecución>/graficas/.

    python -m src.training.graficas runs/20261007-211625_cnn_v1_completo
    python -m src.training.graficas runs/<carpeta_de_validacion_cruzada>
    python -m src.training.graficas runs/*            # todas las ejecuciones

Se generan solas durante el entrenamiento (las curvas se redibujan al acabar cada
época, así que se puede seguir un entrenamiento largo abriendo el PNG). Este script
sirve para rehacerlas o para dibujar ejecuciones anteriores como R1 y R2, que no
medían el AUC de entrenamiento: en ellas solo aparece la curva de validación.

Ejecución normal:
    graficas/curvas.png                pérdida y AUC por paciente por época, entrenamiento y
                                       validación juntos, con la época elegida marcada
    graficas/matriz_confusion.png      matriz de confusión por paciente de la época elegida
Validación cruzada:
    graficas/curvas_folds.png          las curvas de todos los folds superpuestas
    graficas/matriz_confusion_oof.png  matriz fuera de fold (todas las pacientes de train)

Las gráficas que vayan al informe se copian a docs/figuras/ (runs/ no se versiona).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter, MaxNLocator

from src.evaluation.metricas import coma, dibujar_matriz_confusion

# Dos series: entrenamiento (azul) y validación (naranja), como en las figuras del profesor.
COLOR_TR, COLOR_VA = "#2a78d6", "#eb6834"
TINTA, TINTA_SUAVE, REJILLA = "#0b0b0b", "#52514e", "#e4e3df"


def _leer_json(ruta: Path) -> dict:
    return json.loads(ruta.read_text(encoding="utf-8")) if ruta.exists() else {}


def leer_historial(carpeta: Path) -> pd.DataFrame | None:
    ruta = carpeta / "historial.csv"
    if not ruta.exists():
        return None
    historial = pd.read_csv(ruta)
    if "mejor" in historial:
        historial["mejor"] = historial["mejor"].astype(str).str.lower() == "true"
    return historial


def epoca_elegida(historial: pd.DataFrame) -> int:
    """La última época que mejoró el AUC de validación (la que guarda mejor.pt)."""
    if "mejor" in historial and historial["mejor"].any():
        return int(historial.loc[historial["mejor"], "epoca"].iloc[-1])
    return int(historial.loc[historial["auc"].idxmax(), "epoca"])


def _estilo(ax, titulo: str, ylabel: str) -> None:
    ax.set_title(titulo, fontsize=10.5, fontweight="bold", color=TINTA, loc="left")
    ax.set_xlabel("Época", color=TINTA_SUAVE)
    ax.set_ylabel(ylabel, color=TINTA_SUAVE)
    ax.grid(True, color=REJILLA, linewidth=0.8)
    ax.set_axisbelow(True)
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    for lado in ("left", "bottom"):
        ax.spines[lado].set_color(REJILLA)
    ax.tick_params(colors=TINTA_SUAVE, length=0)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: coma(v, 2)))


def _curvas(historiales: list[tuple[pd.DataFrame, int]], ruta: Path, titulo: str, subtitulo: str) -> Path:
    """Dos paneles (pérdida y AUC por paciente) con entrenamiento y validación juntos."""
    varios = len(historiales) > 1
    grosor, alfa = (1.3, 0.85) if varios else (2.0, 1.0)
    fig, (ax_p, ax_a) = plt.subplots(1, 2, figsize=(11, 4.3))
    hay_auc_tr = any("auc_tr" in h and h["auc_tr"].notna().any() for h, _ in historiales)

    for i, (h, elegida) in enumerate(historiales):
        etiqueta = i == 0
        ax_p.plot(h.epoca, h.perdida_tr, color=COLOR_TR, lw=grosor, alpha=alfa,
                  label="entrenamiento" if etiqueta else None)
        ax_p.plot(h.epoca, h.perdida_va, color=COLOR_VA, lw=grosor, alpha=alfa,
                  label="validación" if etiqueta else None)
        if hay_auc_tr and "auc_tr" in h:
            ax_a.plot(h.epoca, h.auc_tr, color=COLOR_TR, lw=grosor, alpha=alfa,
                      label="entrenamiento" if etiqueta else None)
        ax_a.plot(h.epoca, h.auc, color=COLOR_VA, lw=grosor, alpha=alfa,
                  label="validación" if etiqueta else None)
        # Época elegida: punto en la curva de validación (con anillo del color del fondo).
        fila = h[h.epoca == elegida].iloc[0]
        for ax, y in ((ax_p, fila.perdida_va), (ax_a, fila.auc)):
            ax.scatter([elegida], [y], s=46, color=COLOR_VA, edgecolor="white", linewidth=1.5, zorder=5,
                       label="época elegida" if etiqueta and ax is ax_a else None)
        if not varios:
            for ax in (ax_p, ax_a):
                ax.axvline(elegida, color=TINTA_SUAVE, lw=0.8, ls=":", zorder=1)

    ax_a.axhline(0.5, color=TINTA_SUAVE, lw=0.8, ls="--", zorder=1)
    ax_a.text(ax_a.get_xlim()[1], 0.5, " azar", va="center", ha="left", fontsize=8, color=TINTA_SUAVE)
    valores = np.concatenate([np.r_[h.auc.values, h["auc_tr"].values if "auc_tr" in h else []]
                              for h, _ in historiales]).astype(float)
    ax_a.set_ylim(min(0.4, np.nanmin(valores) - 0.03), 1.0)

    # Diferencia de AUC entrenamiento − validación en la época elegida (rúbrica, punto 5).
    if hay_auc_tr:
        difs = [float(h.loc[h.epoca == e, "auc_tr"].iloc[0] - h.loc[h.epoca == e, "auc"].iloc[0])
                for h, e in historiales]
        difs = [d for d in difs if not np.isnan(d)]
        if varios and len(difs) > 1:
            texto = f"diferencia de AUC en la época elegida: {coma(np.mean(difs), 3)} ± {coma(np.std(difs, ddof=1), 3)}"
        elif difs:
            h, e = historiales[0]
            f = h[h.epoca == e].iloc[0]
            texto = (f"época {e}: AUC entrenamiento {coma(f.auc_tr, 3)} · validación {coma(f.auc, 3)}"
                     f" · diferencia {coma(difs[0], 3)}")
        else:
            texto = ""
    else:
        texto = "AUC de entrenamiento no medido en esta ejecución"

    _estilo(ax_p, "Pérdida", "BCE")
    _estilo(ax_a, "ROC-AUC por paciente", "AUC")
    for ax in (ax_p, ax_a):
        ax.legend(frameon=False, fontsize=8.5, labelcolor=TINTA, loc="best")
    fig.suptitle(titulo, x=0.01, y=0.99, ha="left", fontsize=12, fontweight="bold", color=TINTA)
    fig.text(0.01, 0.915, subtitulo, ha="left", fontsize=8.5, color=TINTA_SUAVE)
    if texto:
        fig.text(0.01, 0.87, texto, ha="left", fontsize=9, color=TINTA)
    fig.tight_layout(rect=(0, 0, 1, 0.86))

    ruta.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ruta, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return ruta


def _matriz_de_epoca(resultados: dict, historial: pd.DataFrame, elegida: int) -> dict | None:
    """Matriz de la época elegida: de resultados.json o, en ejecuciones antiguas, del historial."""
    if resultados.get("mejor_epoca", {}).get("matriz_confusion"):
        return resultados["mejor_epoca"]["matriz_confusion"]
    fila = historial[historial.epoca == elegida]
    if fila.empty or not {"VP", "VN", "FP", "FN"} <= set(fila.columns):
        return None
    return {k: int(fila[k].iloc[0]) for k in ("VP", "VN", "FP", "FN")}


def graficas_ejecucion(carpeta: Path) -> list[Path]:
    """Curvas y matriz de confusión de una ejecución normal."""
    historial = leer_historial(carpeta)
    if historial is None or historial.empty:
        return []
    r = _leer_json(carpeta / "resultados.json")
    elegida = epoca_elegida(historial)
    fold = r.get("config", {}).get("fold_val", "?")
    perdida = r.get("perdida", "normal")
    umbral = coma(r.get("umbral", 0.5), 2)
    titulo = f"{r.get('modelo', carpeta.name)} · pérdida {perdida} · fold {fold}"
    rutas = [_curvas([(historial, elegida)], carpeta / "graficas" / "curvas.png", titulo,
                     f"{carpeta.name} · época elegida {elegida} de {len(historial)}")]
    matriz = _matriz_de_epoca(r, historial, elegida)
    if matriz:
        pacientes = sum(matriz.values())
        rutas.append(dibujar_matriz_confusion(
            matriz, carpeta / "graficas" / "matriz_confusion.png",
            f"{r.get('modelo', '')} · época {elegida}",
            f"validación fold {fold} · {pacientes} pacientes · media de cortes, umbral {umbral} · pérdida {perdida}"))
    return rutas


def graficas_validacion_cruzada(carpeta: Path) -> list[Path]:
    """Curvas de todos los folds superpuestas y matriz fuera de fold."""
    resumen = _leer_json(carpeta / "resumen.json")
    historiales = []
    for sub in sorted(carpeta.glob("fold_*")):
        h = leer_historial(sub)
        if h is not None and not h.empty:
            historiales.append((h, epoca_elegida(h)))
    rutas = []
    modelo, perdida = resumen.get("modelo", carpeta.name), resumen.get("perdida", "normal")
    if historiales:
        rutas.append(_curvas(historiales, carpeta / "graficas" / "curvas_folds.png",
                             f"{modelo} · pérdida {perdida} · {len(historiales)} folds",
                             f"{carpeta.name} · cada línea es un fold; el punto, su época elegida"))
    oof = resumen.get("fuera_de_fold")
    if oof:
        rutas.append(dibujar_matriz_confusion(
            oof["matriz_confusion"], carpeta / "graficas" / "matriz_confusion_oof.png",
            f"{modelo} · validación cruzada, fuera de fold",
            f"{oof['pacientes']} pacientes de {len(resumen.get('folds_terminados', []))} folds · "
            f"media de cortes, umbral {coma(oof.get('umbral', 0.5), 2)} · pérdida {perdida}"))
    return rutas


def dibujar_graficas(carpeta: Path | str) -> list[Path]:
    """Detecta si la carpeta es una ejecución normal o una validación cruzada y la dibuja."""
    carpeta = Path(carpeta)
    if (carpeta / "resumen.json").exists() or any(carpeta.glob("fold_*")):
        return graficas_validacion_cruzada(carpeta)
    return graficas_ejecucion(carpeta)


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit("Uso: python -m src.training.graficas runs/<carpeta> [más carpetas]")
    for nombre in sys.argv[1:]:
        carpeta = Path(nombre)
        if not carpeta.is_dir():
            continue
        for ruta in dibujar_graficas(carpeta):
            print(f"Guardado {ruta}")


if __name__ == "__main__":
    main()
