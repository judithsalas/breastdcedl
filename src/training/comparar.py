"""Compara dos versiones en el mismo fold, época a época: una línea por versión.

    python -m src.training.comparar cnn_v3 cnn_v10              # fold 0, últimas validaciones cruzadas
    python -m src.training.comparar cnn_v3 cnn_v10 --fold 2
    python -m src.training.comparar runs/<carpeta_a> runs/<carpeta_b>

Para qué: no esperar a los 5 folds (~45 min) para ver si una versión nueva va bien. El fold 0
es el primero que se entrena (~10 min) y su historial se guarda al acabar cada época, así que
se puede comparar con la referencia mientras entrena. Este script no carga torch ni toca la GPU.

Guarda <carpeta de la nueva>/graficas/comparacion_fold<k>_con_<referencia>.png, con cuatro paneles
(AUC y pérdida, en validación y en entrenamiento) y en cada uno una línea por versión. Imprime además
la época elegida de cada una y, si el fold ha terminado alguna época buena, el AUC dentro de cada
hospital y si la puntuación delata el hospital, en las pacientes de ese fold.

Cuidado al leerlo. Un fold tiene ~220 pacientes y su AUC baila unas ±0,04 de una ejecución a otra.
Solo dice algo una diferencia grande y que se mantiene varias épocas (≥ 0,05). Para decidir
cuentan los 5 folds (D16). Sirve para la regla de trabajo: una versión se descarta si en 5–10
épocas no mejora el AUC por paciente en validación.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FuncFormatter

from src.config import MODOS
from src.evaluation.metricas import auc_por_cohorte, coma, senal_de_hospital
from src.training.graficas import TINTA_SUAVE, _estilo, epoca_elegida, leer_historial

# Sin importar src.data.caso: este carga utils_caso y, con él, torch. Así el script es ligero y
# se puede usar mientras se entrena.
CARPETA_RUNS = Path(__file__).resolve().parents[2] / "runs"
COLOR_REFERENCIA, COLOR_NUEVA = "#8a8984", "#eb6834"


def buscar_ejecucion(nombre: str, modo: str = "completo", raiz: Path = CARPETA_RUNS) -> Path:
    """La carpeta de una ejecución: la ruta tal cual, o la última validación cruzada de esa versión
    (pérdida normal) en `raiz`."""
    ruta = Path(nombre)
    if ruta.is_dir():
        return ruta
    candidatas = sorted(p for p in raiz.glob(f"*_cv_{nombre}_{modo}") if p.is_dir())
    if not candidatas:
        raise SystemExit(f"No hay ninguna validación cruzada de {nombre} en modo {modo} en {raiz}.")
    return candidatas[-1]


def carpeta_del_fold(ejecucion: Path, fold: int) -> Path:
    """En una validación cruzada, la carpeta fold_<k>; en una ejecución suelta, ella misma."""
    sub = ejecucion / f"fold_{fold}"
    return sub if sub.is_dir() else ejecucion


def resumen_fold(carpeta: Path) -> dict | None:
    """Épocas hechas, época elegida y sus AUC; y, si hay predicciones, los AUC por hospital."""
    h = leer_historial(carpeta)
    if h is None or h.empty:
        return None
    elegida = epoca_elegida(h)
    fila = h[h.epoca == elegida].iloc[0]
    salida = {"historial": h, "epocas": int(h.epoca.max()), "elegida": elegida, "auc_val": float(fila.auc),
              "auc_train": float(fila.auc_tr) if "auc_tr" in h else float("nan")}
    salida["diferencia"] = salida["auc_train"] - salida["auc_val"]
    pacientes = carpeta / "predicciones_val_pacientes.csv"
    if pacientes.exists():
        tabla = pd.read_csv(pacientes)
        if {"cohorte", "pCR", "prob_media"} <= set(tabla.columns):
            salida["por_cohorte"] = {c: v["auc"] for c, v in auc_por_cohorte(tabla).items()}
            salida["senal_de_hospital"] = senal_de_hospital(tabla)
    return salida


def dibujar(resumenes: dict[str, dict], fold: int, ruta: Path) -> Path:
    """Cuatro paneles (AUC y pérdida, en validación y en entrenamiento), con una línea por versión."""
    fig, ejes = plt.subplots(2, 2, figsize=(11, 7.8))
    (auc_va, auc_tr), (per_va, per_tr) = ejes
    auc_tr.sharey(auc_va)       # el AUC, en la misma escala; la pérdida no: un pico de validación
                                # aplastaría la curva de entrenamiento
    for (nombre, r), color in zip(resumenes.items(), (COLOR_REFERENCIA, COLOR_NUEVA)):
        h = r["historial"]
        fila = h[h.epoca == r["elegida"]].iloc[0]
        for ax, columna in ((auc_va, "auc"), (auc_tr, "auc_tr"), (per_va, "perdida_va"), (per_tr, "perdida_tr")):
            if columna in h and h[columna].notna().any():
                ax.plot(h.epoca, h[columna], color=color, lw=2, label=nombre)
        for ax, columna in ((auc_va, "auc"), (per_va, "perdida_va")):
            ax.scatter([r["elegida"]], [fila[columna]], s=46, color=color, edgecolor="white", linewidth=1.5,
                       zorder=5)
    for ax in (auc_va, auc_tr):
        ax.axhline(0.5, color=TINTA_SUAVE, lw=0.8, ls="--", zorder=1)
    _estilo(auc_va, f"AUC por paciente · validación (fold {fold})", "AUC")
    _estilo(auc_tr, "AUC por paciente · entrenamiento", "AUC")
    _estilo(per_va, "Pérdida · validación", "BCE")
    _estilo(per_tr, "Pérdida · entrenamiento", "BCE")
    # La pérdida de entrenamiento se mueve en centésimas: con dos decimales se repetirían las etiquetas.
    for ax in (per_va, per_tr):
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: coma(v, 3)))
    for ax in ejes.flat:
        ax.legend(frameon=False, fontsize=8.5, loc="best")
    nombres = list(resumenes)
    fig.suptitle(f"{nombres[0]} frente a {nombres[1]}, fold {fold}", x=0.01, y=0.995, ha="left", fontsize=12,
                 fontweight="bold")
    fig.text(0.01, 0.95, "El punto marca la época elegida. Sobreajuste: la pérdida de validación sube "
             "mientras la de entrenamiento baja. Un solo fold baila ±0,04 de AUC: para decidir, los 5 folds.",
             fontsize=8.5, color=TINTA_SUAVE)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    ruta.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ruta, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return ruta


def texto_resumen(nombre: str, r: dict) -> str:
    lineas = [(f"{nombre}: {r['epocas']} épocas · elegida la {r['elegida']} · AUC val {coma(r['auc_val'], 3)} · "
               f"train {coma(r['auc_train'], 3)} · diferencia {coma(r['diferencia'], 3)}")]
    if r.get("por_cohorte"):
        lineas.append("    AUC dentro de cada hospital: " + " · ".join(
            f"{c} {coma(v, 2)}" for c, v in r["por_cohorte"].items()))
    if r.get("senal_de_hospital"):
        lineas.append("    ¿delata el hospital? (0,5 = no): " + " · ".join(
            f"{c} {coma(v, 2)}" for c, v in r["senal_de_hospital"].items()))
    return "\n".join(lineas)


def comparar(referencia: str, nueva: str, fold: int = 0, modo: str = "completo",
             raiz: Path = CARPETA_RUNS) -> Path:
    carpetas = {referencia: buscar_ejecucion(referencia, modo, raiz), nueva: buscar_ejecucion(nueva, modo, raiz)}
    resumenes = {}
    for nombre, carpeta in carpetas.items():
        r = resumen_fold(carpeta_del_fold(carpeta, fold))
        if r is None:
            raise SystemExit(f"{carpeta.name} todavía no tiene ninguna época del fold {fold}.")
        resumenes[Path(nombre).name] = r
    ruta = (carpetas[nueva] / "graficas"
            / f"comparacion_fold{fold}_con_{Path(referencia).name}.png")
    dibujar(resumenes, fold, ruta)
    for nombre, r in resumenes.items():
        print(texto_resumen(nombre, r))
    print(f"Gráfica en {ruta}")
    return ruta


def main() -> None:
    p = argparse.ArgumentParser(description="Compara dos versiones en el mismo fold, época a época.")
    p.add_argument("referencia", help="versión (p. ej. cnn_v3) o carpeta de runs/")
    p.add_argument("nueva", help="versión (p. ej. cnn_v10) o carpeta de runs/")
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--modo", choices=list(MODOS), default="completo")
    args = p.parse_args()
    comparar(args.referencia, args.nueva, args.fold, args.modo)


if __name__ == "__main__":
    main()
