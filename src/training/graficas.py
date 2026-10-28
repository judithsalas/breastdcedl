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
    graficas/curvas_folds.png          media de los folds ± 1 desviación, y la época elegida de cada uno
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
# Son los dos primeros colores de la paleta validada para daltonismo.
COLOR_TR, COLOR_VA = "#2a78d6", "#eb6834"
TINTA, TINTA_SUAVE, REJILLA, REFERENCIA, FONDO = "#0b0b0b", "#52514e", "#e4e3df", "#9a9893", "#ffffff"
SOMBRA = 0.13          # opacidad de la banda de ± 1 desviación entre folds
MIN_FOLDS = 3          # la media se dibuja mientras siguen entrenando al menos 3 folds


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
    """Ejes discretos: rejilla fina, sin marco arriba ni a la derecha, texto en gris."""
    ax.set_title(titulo, fontsize=11.5, fontweight="bold", color=TINTA, loc="left", pad=10)
    ax.set_xlabel("Época", color=TINTA_SUAVE, fontsize=10)
    ax.set_ylabel(ylabel, color=TINTA_SUAVE, fontsize=10)
    ax.grid(True, color=REJILLA, linewidth=0.8)
    ax.set_axisbelow(True)
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    for lado in ("left", "bottom"):
        ax.spines[lado].set_color(REJILLA)
    ax.tick_params(colors=TINTA_SUAVE, length=0, labelsize=9.5)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: coma(v, 2)))


def _hay(h: pd.DataFrame, columna: str) -> bool:
    return columna in h and h[columna].notna().any()


def _media_por_epoca(historiales: list[tuple[pd.DataFrame, int]], columna: str):
    """Media y desviación entre folds en cada época, mientras sigan entrenando al menos MIN_FOLDS."""
    series = [h.set_index("epoca")[columna].rename(i) for i, (h, _) in enumerate(historiales) if _hay(h, columna)]
    if not series:
        return None
    tabla = pd.concat(series, axis=1)
    tabla = tabla[tabla.notna().sum(axis=1) >= min(MIN_FOLDS, len(series))]
    return tabla.index.to_numpy(), tabla.mean(axis=1).to_numpy(), tabla.std(axis=1, ddof=1).fillna(0).to_numpy()


def _etiqueta_final(ax, x, y, texto: str) -> None:
    """Nombre de la serie al final de su línea (en gris: el color lo lleva la línea)."""
    ax.annotate(texto, (x, y), xytext=(6, 0), textcoords="offset points", va="center", ha="left",
                fontsize=9, color=TINTA_SUAVE)


def _numeros_epoca_elegida(historiales: list[tuple[pd.DataFrame, int]]) -> str:
    """AUC de validación, de entrenamiento y su diferencia en la época elegida (rúbrica, punto 5)."""
    filas = [h[h.epoca == e].iloc[0] for h, e in historiales]
    val = np.array([f.auc for f in filas], float)
    tr = np.array([f.auc_tr if "auc_tr" in f else np.nan for f in filas], float)
    if len(filas) == 1:
        texto = f"Época elegida {historiales[0][1]}: AUC de validación {coma(val[0], 3)}"
        if not np.isnan(tr[0]):
            texto += f" · entrenamiento {coma(tr[0], 3)} · diferencia {coma(tr[0] - val[0], 3)}"
        return texto
    texto = (f"En la época elegida de cada fold: AUC de validación {coma(val.mean(), 3)} ± "
             f"{coma(val.std(ddof=1), 3)}")
    if not np.isnan(tr).all():
        texto += (f" · entrenamiento {coma(np.nanmean(tr), 3)} · diferencia "
                  f"{coma(np.nanmean(tr - val), 3)}")
    return texto


def _curvas(historiales: list[tuple[pd.DataFrame, int]], ruta: Path, titulo: str, pie: str) -> Path:
    """AUC por paciente (izquierda, la medida que importa) y pérdida (derecha), entrenamiento y validación.

    Con varios folds se dibuja la media de los folds y una banda de ± 1 desviación, en vez de diez
    líneas cruzadas: se ve de un vistazo cuánto sube el entrenamiento y dónde se queda la validación.
    Los puntos son la época elegida de cada fold, en su valor real.
    """
    varios = len(historiales) > 1
    fig, (ax_a, ax_p) = plt.subplots(1, 2, figsize=(12.5, 5.6))
    fig.patch.set_facecolor(FONDO)
    hay_auc_tr = any(_hay(h, "auc_tr") for h, _ in historiales)
    paneles = ((ax_a, "auc_tr", "auc"), (ax_p, "perdida_tr", "perdida_va"))

    ultima = max(e for _, e in historiales)        # el eje llega hasta lo dibujado y las épocas elegidas
    for ax, col_tr, col_va in paneles:
        for columna, color, nombre in ((col_tr, COLOR_TR, "entrenamiento"), (col_va, COLOR_VA, "validación")):
            if varios:
                media = _media_por_epoca(historiales, columna)
                if media is None:
                    continue
                x, m, s = media
                ax.fill_between(x, m - s, m + s, color=color, alpha=SOMBRA, lw=0, zorder=2)
            else:
                h = historiales[0][0]
                if not _hay(h, columna):
                    continue
                x, m = h.epoca.to_numpy(), h[columna].to_numpy(float)
            ax.plot(x, m, color=color, lw=2.2, solid_capstyle="round", solid_joinstyle="round", zorder=3,
                    label=nombre if ax is ax_a or not hay_auc_tr else None)
            _etiqueta_final(ax, x[-1], m[-1], nombre)
            ultima = max(ultima, int(x[-1]))
        # Época elegida: un punto por fold en su valor real de validación, con anillo blanco.
        puntos = [(e, float(h.loc[h.epoca == e, col_va].iloc[0])) for h, e in historiales]
        ax.scatter([e for e, _ in puntos], [v for _, v in puntos], s=58, color=COLOR_VA, edgecolor=FONDO,
                   linewidth=2, zorder=5, label=("época elegida de cada fold" if varios else "época elegida")
                   if ax is ax_a else None)
        if not varios:
            ax.axvline(historiales[0][1], color=REJILLA, lw=1.2, zorder=1)

    # AUC: la referencia del azar y una escala fija, para comparar figuras entre versiones.
    ax_a.axhline(0.5, color=REFERENCIA, lw=1, zorder=1)
    ax_a.annotate("azar (0,5)", (0, 0.5), xycoords=("axes fraction", "data"), xytext=(4, -11),
                  textcoords="offset points", fontsize=8.5, color=TINTA_SUAVE)
    valores = np.concatenate([h[c].to_numpy(float) for h, _ in historiales for c in ("auc", "auc_tr") if _hay(h, c)])
    ax_a.set_ylim(min(0.4, np.nanmin(valores) - 0.03), 1.0)
    _estilo(ax_a, "ROC-AUC por paciente", "AUC")
    _estilo(ax_p, "Pérdida (entropía cruzada)", "BCE")
    for ax in (ax_a, ax_p):
        ax.margins(x=0.02)
        ax.set_xlim(0.5, ultima + 0.5 + (ultima - 0.5) * 0.17)          # sitio para las etiquetas finales
        ax.set_xticks([x for x in ax.get_xticks() if 1 <= x <= ultima])

    # Cabecera: título, números clave, cómo leerla y leyenda, cada uno en su línea.
    fig.text(0.012, 0.975, titulo, ha="left", va="top", fontsize=15, fontweight="bold", color=TINTA)
    fig.text(0.012, 0.895, _numeros_epoca_elegida(historiales) if hay_auc_tr or not varios else
             "Esta ejecución no medía el AUC de entrenamiento", ha="left", va="top", fontsize=10.5, color=TINTA)
    if varios:
        fig.text(0.012, 0.845, (f"Línea: media de los {len(historiales)} folds (mientras siguen al menos {MIN_FOLDS}) · "
                                "sombra: ± 1 desviación · puntos: la mejor época de cada fold, por eso quedan por "
                                "encima de la línea"),
                 ha="left", va="top", fontsize=9, color=TINTA_SUAVE)
    manejadores, etiquetas = ax_a.get_legend_handles_labels()
    fig.legend(manejadores, etiquetas, loc="upper left", bbox_to_anchor=(0.006, 0.80 if varios else 0.85),
               ncol=len(etiquetas), frameon=False, fontsize=9.5, labelcolor=TINTA, handlelength=1.8,
               columnspacing=1.8)
    fig.text(0.995, 0.005, pie, ha="right", va="bottom", fontsize=7.5, color=REFERENCIA)
    fig.subplots_adjust(left=0.06, right=0.985, bottom=0.12, top=0.67 if varios else 0.71, wspace=0.2)

    ruta.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ruta, dpi=200, bbox_inches="tight", facecolor=FONDO)
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
                     f"{carpeta.name} · {len(historial)} épocas")]
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
                             f"{modelo} · validación cruzada de {len(historiales)} folds · pérdida {perdida}",
                             carpeta.name))
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
