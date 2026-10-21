"""Auditoría de los datos (F1, punto 1 de la rúbrica): qué hay y si está bien, antes de interpretar nada.

    python -m src.data.auditoria                       # metadatos + todas las imágenes descargadas
    python -m src.data.auditoria --max-pacientes 200   # más rápido: muestra estratificada de train

Genera (todo se versiona, es la evidencia del punto 1):
    docs/resultados/1_auditoria/AUDITORIA.md       informe con las cifras y su lectura
    docs/resultados/1_auditoria/auditoria.json     las mismas cifras, para el informe y las diapositivas
    docs/resultados/1_auditoria/*.png              figuras

Qué comprueba y por qué:
1. Recuentos por conjunto, clase, cohorte y fold (metadatos; no hacen falta imágenes).
   Para describir el desbalance (30 % de pCR) y comprobar que los folds están equilibrados.
2. Referencias sin imagen: el AUC que se consigue sabiendo solo la cohorte o el subtipo.
   Es el listón con el que se compara la CNN (pregunta del profesor: ¿qué peso tiene la imagen?).
3. Integridad de las imágenes: que existan, se abran, sean PNG en gris de 256×256.
4. Orden de las fases: en el tejido debe cumplirse PRE < EARLY (el contraste ilumina). Si no,
   las fases estarían mal leídas. Y cuántas pacientes tienen lavado (LATE < EARLY).
5. Intensidades por cohorte: si una cohorte es más brillante o realza distinto, la red podría
   aprender a reconocer el hospital en lugar del tumor (sesgo que el enunciado pide discutir).
6. Realce por clase: si un indicador hecho a mano (realce medio en el tejido) ya separa algo las
   clases, hay señal en la imagen; su AUC es otra referencia para la CNN.
7. Ejemplos: un corte de una paciente con pCR y otra sin pCR de cada cohorte, con las tres fases
   y el mapa de realce (la guía pide mirarlas antes de programar).
8. Orientación: el corte central de 10 pacientes al azar por cohorte. Cada hospital orienta la
   mama a su manera, y eso es una pista del hospital que la red puede usar (D29).

Solo se usan etiquetas de TRAIN para cualquier cosa que mezcle imagen y clase. De test solo se
cuentan pacientes y cortes (describir su composición no es ajustar nada).
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
from sklearn.metrics import roc_auc_score

from src.data.caso import NOMBRE_COHORTE, RAIZ_DATOS, RAIZ_REPO, RESULTADOS, uc
from src.evaluation.metricas import coma

CARPETA = RESULTADOS / "1_auditoria"
SALIDA_MD = CARPETA / "AUDITORIA.md"
UMBRAL_TEJIDO = 0.1          # píxeles con PRE > 0,1 son tejido (el resto, aire); como en utils_caso
FASES = ("PRE", "EARLY", "LATE")
COLOR_FASE = {"PRE": "#2a78d6", "EARLY": "#eb6834", "LATE": "#1baf7a"}
COLOR_CLASE = {0: "#2a78d6", 1: "#eb6834"}
TINTA, TINTA_SUAVE, REJILLA = "#0b0b0b", "#52514e", "#e4e3df"


# --------------------------------------------------------------------------- #
# 1-2. Metadatos
# --------------------------------------------------------------------------- #

def pacientes_train_con_fold(samples: pd.DataFrame, patients: pd.DataFrame) -> pd.DataFrame:
    """Una fila por paciente de train con su fold, cohorte, subtipo y etiqueta."""
    folds = samples[samples.split == "train"].groupby("patient_id").fold.first()
    tabla = patients[patients.split == "train"].set_index("pid")
    tabla = tabla.join(folds, how="inner")
    tabla["subtipo"] = tabla.HR_HER2_STATUS.fillna("desconocido")
    return tabla


def recuentos(samples: pd.DataFrame, patients: pd.DataFrame) -> dict:
    por_split = {}
    for split, grupo in samples.groupby("split"):
        p = grupo.groupby("patient_id").pCR.first()
        por_split[split] = {"pacientes": len(p), "cortes": len(grupo), "pacientes_pcr": int(p.sum()),
                            "proporcion_pcr": float(p.mean())}
    cohorte = patients.groupby(["split", "dataset"]).pCR.agg(["count", "mean"]).reset_index()
    tr = pacientes_train_con_fold(samples, patients)
    por_fold = tr.groupby("fold").agg(pacientes=("pCR", "size"), proporcion_pcr=("pCR", "mean"))
    por_fold = por_fold.join(pd.crosstab(tr.fold, tr.dataset))
    cortes = samples.groupby("patient_id").size()
    return {
        "por_split": por_split,
        "por_cohorte": [{"split": r.split, "cohorte": r.dataset, "pacientes": int(r["count"]),
                         "proporcion_pcr": float(r["mean"])} for _, r in cohorte.iterrows()],
        "por_subtipo_train": {k: {"pacientes": int(v["count"]), "proporcion_pcr": float(v["mean"])}
                              for k, v in tr.groupby("subtipo").pCR.agg(["count", "mean"]).iterrows()},
        "por_fold": {int(k): {c: (float(v) if c == "proporcion_pcr" else int(v)) for c, v in fila.items()}
                     for k, fila in por_fold.iterrows()},
        "cortes_por_paciente": {"min": int(cortes.min()), "mediana": float(cortes.median()),
                                "max": int(cortes.max()), "con_menos_de_10": int((cortes < 10).sum())},
        "siempre_no_pcr_accuracy": float(1 - samples.groupby("patient_id").pCR.first().mean()),
    }


def referencias_sin_imagen(samples: pd.DataFrame, patients: pd.DataFrame) -> dict:
    """AUC por paciente usando solo un grupo (cohorte, subtipo o ambos), validado por folds.

    En cada fold, la probabilidad de una paciente es la tasa de pCR de su grupo calculada en los
    otros 4 folds. Es exactamente lo que haría un "modelo" que solo conociera esa variable.
    """
    tr = pacientes_train_con_fold(samples, patients)
    tr["cohorte_y_subtipo"] = tr.dataset + " · " + tr.subtipo
    salida = {}
    for variable in ("dataset", "subtipo", "cohorte_y_subtipo"):
        aucs = []
        for k in range(5):
            a, b = tr[tr.fold != k], tr[tr.fold == k]
            tasa = a.groupby(variable).pCR.mean()
            prob = b[variable].map(tasa).fillna(a.pCR.mean())
            aucs.append(float(roc_auc_score(b.pCR, prob)))
        nombre = {"dataset": "cohorte"}.get(variable, variable)
        salida[nombre] = {"auc_por_fold": aucs, "media": float(np.mean(aucs)), "desviacion": float(np.std(aucs, ddof=1))}
    return salida


# --------------------------------------------------------------------------- #
# 3-6. Imágenes
# --------------------------------------------------------------------------- #

def elegir_pacientes(tr: pd.DataFrame, maximo: int | None, semilla: int = 42) -> pd.Index:
    if maximo is None or maximo >= len(tr):
        return tr.index
    return tr.groupby(["dataset", "pCR"], group_keys=False).sample(frac=maximo / len(tr), random_state=semilla).index


def medir_cortes(filas: pd.DataFrame, raiz: Path = RAIZ_DATOS) -> tuple[pd.DataFrame, list[str]]:
    """Intensidad media de cada fase en el tejido, realce y lavado, corte a corte."""
    medidas, problemas = [], []
    for fila in filas.itertuples():
        try:
            x = uc.cargar_imagen(fila, raiz)
        except (OSError, ValueError, SyntaxError) as error:     # fichero que falta o corrupto
            problemas.append(f"{fila.sample_id}: no se puede leer ({type(error).__name__})")
            continue
        if x.shape != (3, 256, 256) or not np.isfinite(x).all():
            problemas.append(f"{fila.sample_id}: forma {x.shape} o valores no finitos")
            continue
        tejido = x[0] > UMBRAL_TEJIDO
        if tejido.sum() < 50:
            problemas.append(f"{fila.sample_id}: casi sin tejido ({int(tejido.sum())} píxeles)")
            continue
        pre, early, late = (float(c[tejido].mean()) for c in x)
        realce = x[1] - x[0]
        medidas.append({"sample_id": fila.sample_id, "patient_id": fila.patient_id, "pCR": int(fila.pCR),
                        "PRE": pre, "EARLY": early, "LATE": late, "realce": early - pre, "lavado": late - early,
                        "fraccion_realza": float((realce[tejido] > 0.1).mean())})
    return pd.DataFrame(medidas), problemas


def integridad_png(filas: pd.DataFrame, maximo: int = 300) -> list[str]:
    """Comprueba formato real (modo L, 256×256) en una muestra de ficheros."""
    from PIL import Image

    problemas = []
    for fila in filas.head(maximo).itertuples():
        for col in ("path_pre", "path_early", "path_late"):
            ruta = RAIZ_DATOS / getattr(fila, col)
            try:
                with Image.open(ruta) as png:
                    if png.format != "PNG" or png.mode != "L" or png.size != (256, 256):
                        problemas.append(f"{ruta.name}: {png.format} {png.mode} {png.size}")
            except (OSError, ValueError, SyntaxError) as error:
                problemas.append(f"{ruta.name}: {type(error).__name__}")
    return problemas


def resumen_imagenes(cortes: pd.DataFrame, tr: pd.DataFrame) -> dict:
    pac = cortes.groupby("patient_id").agg(PRE=("PRE", "mean"), EARLY=("EARLY", "mean"), LATE=("LATE", "mean"),
                                           realce=("realce", "mean"), fraccion_realza=("fraccion_realza", "mean"),
                                           pCR=("pCR", "first"))
    pac = pac.join(tr[["dataset"]])
    salida = {
        "pacientes": len(pac), "cortes": len(cortes),
        "pre_menor_que_early": float((pac.PRE < pac.EARLY).mean()),
        "lavado_late_menor_que_early": float((pac.LATE < pac.EARLY).mean()),
        "por_cohorte": {}, "auc_indicadores": {},
    }
    for cohorte, g in pac.groupby("dataset"):
        salida["por_cohorte"][cohorte] = {
            "pacientes": len(g), **{f: float(g[f].mean()) for f in ("PRE", "EARLY", "LATE", "realce")},
            "realce_pcr0": float(g.loc[g.pCR == 0, "realce"].mean()) if (g.pCR == 0).any() else None,
            "realce_pcr1": float(g.loc[g.pCR == 1, "realce"].mean()) if (g.pCR == 1).any() else None,
        }
    for indicador in ("realce", "fraccion_realza"):
        fila = {}
        for nombre, g in [("todas", pac), *pac.groupby("dataset")]:
            fila[nombre] = float(roc_auc_score(g.pCR, g[indicador])) if g.pCR.nunique() == 2 else None
        salida["auc_indicadores"][indicador] = fila
    return salida, pac


# --------------------------------------------------------------------------- #
# Figuras
# --------------------------------------------------------------------------- #

def _estilo(ax, titulo):
    ax.set_title(titulo, fontsize=10.5, fontweight="bold", color=TINTA, loc="left")
    ax.grid(True, axis="y", color=REJILLA, linewidth=0.8)
    ax.set_axisbelow(True)
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    for lado in ("left", "bottom"):
        ax.spines[lado].set_color(REJILLA)
    ax.tick_params(colors=TINTA_SUAVE, length=0)


def figura_intensidades(pac: pd.DataFrame, ruta: Path) -> Path:
    """Intensidad media en el tejido por fase y cohorte, y realce por clase y cohorte."""
    cohortes = [c for c in ("spy1", "spy2", "duke") if c in set(pac.dataset)]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    ancho = 0.25
    for i, fase in enumerate(FASES):
        datos = [pac.loc[pac.dataset == c, fase].values for c in cohortes]
        pos = np.arange(len(cohortes)) + (i - 1) * ancho
        caja = ax1.boxplot(datos, positions=pos, widths=ancho * 0.85, patch_artist=True, showfliers=False)
        for parte in caja["boxes"]:
            parte.set(facecolor=COLOR_FASE[fase], edgecolor="white", linewidth=1.5, alpha=0.9)
        for parte in caja["medians"]:
            parte.set(color="white", linewidth=1.5)
        for clave in ("whiskers", "caps"):
            for parte in caja[clave]:
                parte.set(color=COLOR_FASE[fase])
        ax1.plot([], [], color=COLOR_FASE[fase], linewidth=6, label=fase)
    ax1.set_xticks(range(len(cohortes)), [NOMBRE_COHORTE[c] for c in cohortes])
    ax1.set_ylabel("Intensidad media en el tejido", color=TINTA_SUAVE)
    ax1.legend(frameon=False, fontsize=8.5)
    _estilo(ax1, "Intensidad por fase y cohorte")

    for j, clase in enumerate((0, 1)):
        datos = [pac.loc[(pac.dataset == c) & (pac.pCR == clase), "realce"].values for c in cohortes]
        pos = np.arange(len(cohortes)) + (j - 0.5) * 0.35
        caja = ax2.boxplot(datos, positions=pos, widths=0.3, patch_artist=True, showfliers=False)
        for parte in caja["boxes"]:
            parte.set(facecolor=COLOR_CLASE[clase], edgecolor="white", linewidth=1.5, alpha=0.9)
        for parte in caja["medians"]:
            parte.set(color="white", linewidth=1.5)
        for clave in ("whiskers", "caps"):
            for parte in caja[clave]:
                parte.set(color=COLOR_CLASE[clase])
        ax2.plot([], [], color=COLOR_CLASE[clase], linewidth=6, label="pCR" if clase else "no pCR")
    ax2.set_xticks(range(len(cohortes)), [NOMBRE_COHORTE[c] for c in cohortes])
    ax2.set_ylabel("Realce medio (EARLY − PRE)", color=TINTA_SUAVE)
    ax2.legend(frameon=False, fontsize=8.5)
    _estilo(ax2, "Realce por clase y cohorte (train)")
    for ax in (ax1, ax2):
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: coma(v, 2)))
    fig.tight_layout()
    fig.savefig(ruta, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return ruta


def figura_ejemplos(samples_tr: pd.DataFrame, pac: pd.DataFrame, ruta: Path, semilla: int = 42) -> Path:
    """Un corte central de una paciente sin pCR y otra con pCR por cohorte: 3 fases + realce."""
    filas_fig = []
    rng = np.random.default_rng(semilla)
    for cohorte in ("spy1", "spy2", "duke"):
        for clase in (0, 1):
            ids = pac[(pac.dataset == cohorte) & (pac.pCR == clase)].index
            if len(ids):
                pid = ids[rng.integers(len(ids))]
                cortes = samples_tr[samples_tr.patient_id == pid].sort_values("slice_index")
                filas_fig.append((cohorte, clase, cortes.iloc[len(cortes) // 2]))
    if not filas_fig:
        return None
    fig, ejes = plt.subplots(len(filas_fig), 4, figsize=(9, 2.3 * len(filas_fig)), squeeze=False)
    for i, (cohorte, clase, fila) in enumerate(filas_fig):
        x = uc.cargar_imagen(fila)
        for j, fase in enumerate(FASES):
            ejes[i, j].imshow(x[j], cmap="gray", vmin=0, vmax=1)
        realce = ejes[i, 3].imshow(x[1] - x[0], cmap="RdBu_r", vmin=-0.5, vmax=0.5)
        for j, titulo in enumerate((*FASES, "EARLY − PRE")):
            ejes[i, j].set_xticks([])
            ejes[i, j].set_yticks([])
            if i == 0:
                ejes[i, j].set_title(titulo, fontsize=10, color=TINTA)
        ejes[i, 0].set_ylabel(f"{NOMBRE_COHORTE[cohorte]}\n{'pCR' if clase else 'no pCR'}", fontsize=9,
                              color=TINTA)
    fig.colorbar(realce, ax=ejes[:, 3].tolist(), fraction=0.05, label="realce")
    fig.suptitle("Ejemplos de train: corte central de una paciente por cohorte y clase", fontsize=11,
                 color=TINTA, x=0.01, ha="left", y=1.0)
    fig.subplots_adjust(top=0.96)
    fig.savefig(ruta, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return ruta


def figura_orientacion(samples_tr: pd.DataFrame, pac: pd.DataFrame, ruta: Path, por_cohorte: int = 10,
                       semilla: int = 0, raiz: Path = RAIZ_DATOS) -> Path | None:
    """EARLY del corte central de `por_cohorte` pacientes al azar de cada cohorte, en una fila por cohorte.

    Sirve para ver cómo orienta la mama cada hospital. En la revisión del 2026-10-21 (D29), en
    I-SPY1 apunta hacia abajo y en I-SPY2 y Duke a la izquierda o a la derecha: la orientación
    delata el hospital y no dice nada del tumor. Con semilla 0 salen las mismas pacientes que
    en el mosaico de esa revisión.
    """
    cortes = samples_tr[samples_tr.patient_id.isin(pac.index)]
    cortes = cortes.assign(cohorte=cortes.patient_id.map(pac.dataset))
    cohortes = [c for c in ("spy1", "spy2", "duke") if c in set(cortes.cohorte)]
    if not cohortes:
        return None
    rng = np.random.default_rng(semilla)
    fig, ejes = plt.subplots(len(cohortes), por_cohorte, figsize=(2 * por_cohorte, 2.2 * len(cohortes)),
                             squeeze=False)
    for fila_ejes, cohorte in zip(ejes, cohortes):
        pacientes = cortes[cortes.cohorte == cohorte].patient_id.unique()
        elegidas = rng.choice(pacientes, min(por_cohorte, len(pacientes)), replace=False)
        for ax, pid in zip(fila_ejes, elegidas):
            suyos = cortes[cortes.patient_id == pid].sort_values("slice_index")
            ax.imshow(uc.cargar_imagen(suyos.iloc[len(suyos) // 2], raiz)[1], cmap="gray", vmin=0, vmax=1)
            ax.set_title(pid[-10:], fontsize=7, color=TINTA_SUAVE)
        for ax in fila_ejes:
            ax.set_xticks([])
            ax.set_yticks([])
        fila_ejes[0].set_ylabel(NOMBRE_COHORTE[cohorte], fontsize=11, color=TINTA)
    fig.suptitle(f"Orientación por cohorte: EARLY del corte central de {por_cohorte} pacientes al azar",
                 fontsize=11, color=TINTA, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(ruta, dpi=100, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return ruta


# --------------------------------------------------------------------------- #
# Informe
# --------------------------------------------------------------------------- #

def _pct(x):
    return "—" if x is None else f"{coma(100 * x, 1)} %"


def informe(datos: dict) -> str:
    r, ref = datos["recuentos"], datos["referencias_sin_imagen"]
    lineas = [
        "# Auditoría de los datos",
        "",
        f"Generado automáticamente por `python -m src.data.auditoria` el {datos['fecha']}. No se edita a mano:",
        "se vuelve a generar. La explicación de por qué se mira cada cosa está en el código y en",
        "[`COMO_SE_HIZO.md`](../../COMO_SE_HIZO.md).",
        "",
        "## 1. Recuentos",
        "",
        "| Conjunto | Pacientes | Cortes | Pacientes con pCR |",
        "|---|---|---|---|",
    ]
    for split, v in r["por_split"].items():
        lineas.append(f"| {split} | {v['pacientes']} | {v['cortes']} | {v['pacientes_pcr']} ({_pct(v['proporcion_pcr'])}) |")
    lineas += ["", f"Decir siempre «no pCR» acierta el {_pct(r['siempre_no_pcr_accuracy'])} de las pacientes: la",
               "accuracy no sirve como métrica principal.", "", "**Por cohorte:**", "",
               "| Conjunto | Cohorte | Pacientes | Proporción de pCR |", "|---|---|---|---|"]
    for c in r["por_cohorte"]:
        lineas.append(f"| {c['split']} | {NOMBRE_COHORTE.get(c['cohorte'], c['cohorte'])} | {c['pacientes']} "
                      f"| {_pct(c['proporcion_pcr'])} |")
    lineas += ["", "**Por subtipo (train):**", "", "| Subtipo | Pacientes | Proporción de pCR |", "|---|---|---|"]
    for k, v in r["por_subtipo_train"].items():
        lineas.append(f"| {k} | {v['pacientes']} | {_pct(v['proporcion_pcr'])} |")
    lineas += ["", "**Por fold (train):** los folds deben parecerse entre sí.", "",
               "| Fold | Pacientes | Proporción de pCR | Duke | I-SPY1 | I-SPY2 |", "|---|---|---|---|---|---|"]
    for k, v in r["por_fold"].items():
        lineas.append(f"| {k} | {v['pacientes']} | {_pct(v['proporcion_pcr'])} | {v.get('duke', 0)} "
                      f"| {v.get('spy1', 0)} | {v.get('spy2', 0)} |")
    cp = r["cortes_por_paciente"]
    lineas += ["", (f"Cortes por paciente: mínimo {cp['min']}, mediana {coma(cp['mediana'], 0)}, "
                    f"máximo {cp['max']}; {cp['con_menos_de_10']} pacientes tienen menos de 10."), "",
               "## 2. Referencias sin imagen (train, validadas por folds)", "",
               "AUC por paciente de un «modelo» que solo conoce una variable: en cada fold, la probabilidad de",
               "una paciente es la tasa de pCR de su grupo en los otros 4. Es el listón con el que compararse.", "",
               "| Solo se conoce | AUC (media ± desviación de 5 folds) |", "|---|---|"]
    for k, v in ref.items():
        lineas.append(f"| {k.replace('_', ' ')} | {coma(v['media'], 3)} ± {coma(v['desviacion'], 3)} |")
    img = datos.get("imagenes")
    lineas += ["", "## 3. Imágenes", ""]
    if not img:
        lineas.append("_No había imágenes descargadas: solo se han auditado los metadatos._")
    else:
        lineas += [(f"Medido en {img['pacientes']} pacientes de train ({img['cortes']} cortes) "
                    f"{'(todas las descargadas)' if not datos['muestra'] else '(muestra estratificada)'}. "
                    f"Tejido = píxeles con PRE > {coma(UMBRAL_TEJIDO, 1)}."), "",
                   f"- **Integridad:** {len(datos['problemas'])} problemas de lectura o formato"
                   + (" (lista en `auditoria.json`, en esta misma carpeta)." if datos["problemas"] else "."),
                   f"- **Orden de las fases:** PRE < EARLY en el tejido en el {_pct(img['pre_menor_que_early'])} "
                   "de las pacientes"
                   + (": las fases están bien leídas." if img["pre_menor_que_early"] == 1 else
                      ". Revisar las que no lo cumplen."),
                   (f"- **Lavado:** LATE < EARLY en el {_pct(img['lavado_late_menor_que_early'])} de las "
                    "pacientes (la guía da un 17 %; no es un error, es *washout*)."), "",
                   "**Intensidad media en el tejido por cohorte:**", "",
                   "| Cohorte | Pacientes | PRE | EARLY | LATE | Realce | Realce sin pCR | Realce con pCR |",
                   "|---|---|---|---|---|---|---|---|"]
        for c, v in img["por_cohorte"].items():
            lineas.append(f"| {NOMBRE_COHORTE.get(c, c)} | {v['pacientes']} | {coma(v['PRE'], 3)} | "
                          f"{coma(v['EARLY'], 3)} | {coma(v['LATE'], 3)} | {coma(v['realce'], 3)} | "
                          f"{'—' if v['realce_pcr0'] is None else coma(v['realce_pcr0'], 3)} | "
                          f"{'—' if v['realce_pcr1'] is None else coma(v['realce_pcr1'], 3)} |")
        lineas += ["", "**¿Hay señal en la imagen con un indicador hecho a mano?** AUC por paciente de dos",
                   "indicadores sin entrenar nada (0,5 = ninguna señal; por debajo de 0,5, la relación va al revés):",
                   "", "| Indicador | Todas | " + " | ".join(NOMBRE_COHORTE.get(c, c) for c in img["por_cohorte"]) + " |",
                   "|---|---|" + "---|" * len(img["por_cohorte"])]
        for ind, v in img["auc_indicadores"].items():
            nombre = {"realce": "Realce medio en el tejido",
                      "fraccion_realza": "Fracción del tejido que realza (> 0,1)"}[ind]
            celdas = [("—" if v.get(c) is None else coma(v[c], 3)) for c in ["todas", *img["por_cohorte"]]]
            lineas.append(f"| {nombre} | " + " | ".join(celdas) + " |")
        lineas += ["", "![Intensidades por cohorte](intensidades.png)", "",
                   "![Ejemplos por cohorte y clase](ejemplos.png)", "",
                   "**Orientación por cohorte.** Cada hospital orienta la mama a su manera. En la revisión del",
                   "2026-10-21, en I-SPY1 apunta hacia abajo en las 10 pacientes; en I-SPY2 y Duke, a la izquierda",
                   "o a la derecha, y en Duke y algunas de I-SPY2 se ven el corazón y los pulmones (otro plano de",
                   "corte). La orientación no dice nada del tumor, pero sí del hospital (D29).", "",
                   "![Orientación por cohorte](orientacion.png)"]
    lineas += ["", "## Cómo leer esto", "",
               "- Si las cohortes difieren mucho en intensidad o realce, la red puede aprender a reconocer el",
               "  hospital. Por eso se miran los resultados por cohorte (rúbrica, punto 9) y se compara con la",
               "  referencia «solo cohorte».",
               "- Si un indicador hecho a mano ya da un AUC claramente por encima de 0,5, hay señal en el",
               "  realce y la CNN debería superarlo.",
               "- Si la CNN no supera «solo subtipo», la imagen aporta menos que una variable clínica que se",
               "  conoce antes del tratamiento: es una limitación honesta que hay que contar."]
    return "\n".join(lineas) + "\n"


def main() -> None:
    p = argparse.ArgumentParser(description="Auditoría de los datos (F1).")
    p.add_argument("--max-pacientes", type=int, default=None,
                   help="limitar las imágenes a una muestra estratificada de train")
    args = p.parse_args()

    samples, patients = uc.cargar_samples(), uc.cargar_patients()
    CARPETA.mkdir(parents=True, exist_ok=True)
    datos = {"fecha": datetime.now().astimezone().strftime("%Y-%m-%d %H:%M"), "recuentos": recuentos(samples, patients),
             "referencias_sin_imagen": referencias_sin_imagen(samples, patients), "problemas": [],
             "muestra": args.max_pacientes is not None}
    print("Metadatos auditados.")

    tr = pacientes_train_con_fold(samples, patients)
    samples_tr = samples[samples.split == "train"]
    disponibles = samples_tr[samples_tr.path_pre.map(lambda r: (RAIZ_DATOS / r).exists())]
    if disponibles.empty:
        print("No hay imágenes descargadas: solo metadatos.")
    else:
        con_imagen = tr.loc[tr.index.isin(disponibles.patient_id)]
        elegidas = elegir_pacientes(con_imagen, args.max_pacientes)
        filas = disponibles[disponibles.patient_id.isin(elegidas)]
        print(f"Midiendo {filas.patient_id.nunique()} pacientes ({len(filas)} cortes)...")
        cortes, problemas = medir_cortes(filas)
        datos["problemas"] = problemas + integridad_png(filas)
        if not cortes.empty:
            datos["imagenes"], pac = resumen_imagenes(cortes, tr)
            figura_intensidades(pac, CARPETA / "intensidades.png")
            figura_ejemplos(samples_tr, pac, CARPETA / "ejemplos.png")
            figura_orientacion(samples_tr, pac, CARPETA / "orientacion.png")

    (CARPETA / "auditoria.json").write_text(json.dumps(datos, indent=2, ensure_ascii=False), encoding="utf-8")
    SALIDA_MD.write_text(informe(datos), encoding="utf-8")
    print(f"Informe en {SALIDA_MD.relative_to(RAIZ_REPO)} y figuras en {CARPETA.relative_to(RAIZ_REPO)}/")


if __name__ == "__main__":
    main()
