"""App web del caso BreastDCEDL: probabilidad de pCR para un corte DCE-MRI.

    streamlit run app/app.py

Uso educativo, sin validez clínica. Usa EXACTAMENTE el mismo código que el pipeline:
- src/data/entrada.py   lectura y validación de las tres fases (misma normalización: PNG / 255)
- src/models/final.py   modelo final (conjunto de CNN: 5 folds por versión + calibración), en modo evaluación y
                        sin gradientes
Despliegue: app/README.md.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch

from src.data.entrada import FASES, TAMANO_MAXIMO, validar_corte
from src.models.final import cargar_modelo_final

EJEMPLOS = Path(__file__).resolve().parent / "ejemplos"

st.set_page_config(page_title="Predicción de pCR · BreastDCEDL", page_icon="🩺", layout="wide")

AVISO = (
    "**Uso exclusivamente educativo. Sin validez clínica. No es un dispositivo médico.** "
    "La probabilidad es una estimación de un modelo entrenado con datos retrospectivos; puede "
    "equivocarse en los dos sentidos. Una probabilidad alta **no** permite evitar la cirugía ni "
    "cambiar un tratamiento: la pCR solo se confirma con anatomía patológica."
)


@st.cache_resource(show_spinner="Cargando el modelo…")
def cargar():
    torch.manual_seed(0)
    return cargar_modelo_final()


def _num(x: float, d: int = 3) -> str:
    return f"{x:.{d}f}".replace(".", ",")


def _prob(p: float) -> str:
    """Probabilidad con 3 decimales; en los extremos, sin aparentar un 0 o un 1 exactos."""
    if p < 0.001:
        return "< 0,001"
    if p > 0.999:
        return "> 0,999"
    return _num(p)


def ejemplos() -> dict[str, dict]:
    """Cortes de ejemplo incluidos en app/ejemplos/<corte>/ (lista cerrada: no se aceptan rutas)."""
    esperado = {}
    if (EJEMPLOS / "esperado.json").exists():
        esperado = json.loads((EJEMPLOS / "esperado.json").read_text(encoding="utf-8"))
    salida = {}
    if not EJEMPLOS.exists():
        return salida
    for carpeta in sorted(p for p in EJEMPLOS.iterdir() if p.is_dir()):
        ficheros = sorted(carpeta.glob("*.png"))
        if len(ficheros) == 3:
            salida[carpeta.name] = {"ficheros": ficheros, **esperado.get(carpeta.name, {})}
    return salida


def figura_fases(x: np.ndarray):
    fig, ejes = plt.subplots(1, 4, figsize=(12, 3.3))
    for i, fase in enumerate(FASES):
        ejes[i].imshow(x[i], cmap="gray", vmin=0, vmax=1)
        ejes[i].set_title(fase)
    mapa = ejes[3].imshow(x[1] - x[0], cmap="RdBu_r", vmin=-0.5, vmax=0.5)
    ejes[3].set_title("Realce (EARLY − PRE)")
    for ax in ejes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(mapa, ax=ejes[3], fraction=0.046)
    fig.tight_layout()
    return fig


def tarjeta_modelo(manifiesto: dict, dispositivo: str) -> None:
    mi, ent = manifiesto["metricas_internas"], manifiesto["entrenamiento"]
    oof = mi["fuera_de_fold"]
    st.sidebar.header("Modelo")
    st.sidebar.markdown(
        f"- **Versión:** {manifiesto['version']} · huella `{manifiesto['huella_modelo']}`\n"
        f"- **Umbral:** {_num(manifiesto['umbral'])} ({manifiesto['criterio_umbral']})\n"
        f"- **Dispositivo de inferencia:** {dispositivo}\n"
        f"- **Conjunto:** {manifiesto['conjunto']}\n"
        f"- **Calibración:** Platt (a = {_num(manifiesto['calibracion']['a'])}, "
        f"b = {_num(manifiesto['calibracion']['b'])})"
    )
    st.sidebar.subheader("Entrenamiento")
    config = ent["config"]
    miembros = manifiesto.get("miembros") or [{"version": manifiesto["version"], "parametros": ent["parametros"],
                                               "receta": manifiesto.get("receta"), "pesos": manifiesto["pesos"]}]
    lineas = [f"- {len(miembros) * 5} CNN propias desde cero ({len(miembros)} versiones × 5 folds)"
              if len(miembros) > 1 else "- CNN propia desde cero (5 modelos, uno por fold)",
              (f"- Pérdida {ent['perdida']} · batch {config.get('batch_size')} · lr {config.get('learning_rate')} · "
               f"semilla {config.get('semilla')}")]
    for m in miembros:
        receta = m.get("receta") or {}
        lineas.append(f"- **{m['version']}:** {m['parametros']:,} parámetros".replace(",", ".")
                      + f" · aumento {receta.get('aumento') or 'no'} · épocas "
                      + ", ".join(str(f["epoca"]) for f in m["pesos"]))
    st.sidebar.markdown("\n".join(lineas))
    st.sidebar.subheader("Métricas internas (validación cruzada)")
    st.sidebar.markdown(
        f"- AUC por paciente: **{_num(mi['auc_val_media'])} ± {_num(mi['auc_val_desviacion'])}** "
        f"(entrenamiento {_num(mi['auc_train_media'])})\n"
        f"- Con el umbral, fuera de fold: sensibilidad {_num(oof['sensibilidad'])}, "
        f"especificidad {_num(oof['especificidad'])}, F1 macro {_num(oof['f1_macro'])}\n"
        f"- Error de Brier (calibración): {_num(mi['brier_calibrado'], 4)}"
    )
    st.sidebar.caption(f"Normalización de la entrada: {manifiesto['entrada']}. Idéntica al entrenamiento.")


def main() -> None:
    st.title("Predicción de respuesta patológica completa (pCR)")
    st.caption("Caso BreastDCEDL · CNN 2D entrenada desde cero con las fases PRE, EARLY y LATE de una DCE-MRI "
               "previa a la quimioterapia.")
    st.warning(AVISO, icon="⚠️")

    try:
        modelo, manifiesto = cargar()
    except (FileNotFoundError, ValueError) as error:
        st.error(f"No se puede cargar el modelo final: {error}")
        st.stop()
    dispositivo = str(next(modelo.parameters()).device)
    tarjeta_modelo(manifiesto, dispositivo)

    pestana_subir, pestana_ejemplo = st.tabs(["Subir las tres fases de un corte", "Elegir un ejemplo"])
    ficheros, esperado = None, None
    with pestana_subir:
        st.markdown(f"Sube **tres PNG en gris de 256×256** (máximo {TAMANO_MAXIMO // 2**20} MB cada uno) cuyos "
                    "nombres acaben en `_PRE.png`, `_EARLY.png` y `_LATE.png`, como en el dataset.")
        subidos = st.file_uploader("Fases del corte", type=["png"], accept_multiple_files=True,
                                   key="subidos")
        if subidos:
            ficheros = [(f.name, f.getvalue()) for f in subidos]
    with pestana_ejemplo:
        disponibles = ejemplos()
        if not disponibles:
            st.info("No hay ejemplos incluidos todavía (se añaden con `python -m app.preparar_ejemplos`).")
        else:
            elegido = st.selectbox("Corte de ejemplo", ["—", *disponibles])
            if elegido != "—" and not ficheros:
                ficheros = [(r.name, r.read_bytes()) for r in disponibles[elegido]["ficheros"]]
                esperado = disponibles[elegido].get("probabilidad_pipeline")

    if not ficheros:
        st.stop()

    x, errores, avisos = validar_corte(ficheros)
    for mensaje in errores:
        st.error(mensaje)
    if errores:
        st.stop()
    for mensaje in avisos:
        st.warning(mensaje)

    st.pyplot(figura_fases(x), clear_figure=True)

    inicio = time.perf_counter()
    with torch.no_grad():
        p = float(modelo.probabilidad(torch.from_numpy(x).unsqueeze(0).to(dispositivo))[0])
    milisegundos = (time.perf_counter() - inicio) * 1000
    etiqueta = "pCR" if p >= manifiesto["umbral"] else "no pCR"
    clase = "pCR (respuesta completa)" if etiqueta == "pCR" else "no pCR (enfermedad residual)"

    c1, c2, c3 = st.columns(3)
    c1.metric("Probabilidad de pCR", _prob(p))
    c2.metric("Clase según el umbral", etiqueta, help=clase)
    c3.metric("Umbral", _num(manifiesto["umbral"]))
    st.markdown(f"**Clase:** {clase} · **Tiempo de respuesta:** {milisegundos:.0f} ms · **Dispositivo:** "
                f"{dispositivo} · **Modelo:** {manifiesto['version']} (`{manifiesto['huella_modelo']}`)")
    if esperado is not None:
        coincide = abs(esperado - p) < 1e-5
        st.markdown(f"Probabilidad que da el pipeline para este ejemplo: {_num(esperado, 6)} → "
                    f"{'✅ coincide' if coincide else '❌ NO coincide'} (app: {_num(p, 6)}).")
    mi = manifiesto["metricas_internas"]
    st.info(
        f"**Incertidumbre.** El modelo ordena bien a una paciente con pCR frente a una sin pCR en torno al "
        f"{_num(100 * mi['auc_val_media'], 0)} % de las veces (AUC interno {_num(mi['auc_val_media'])} ± "
        f"{_num(mi['auc_val_desviacion'])}). Con el umbral elegido detecta "
        f"{_num(100 * mi['fuera_de_fold']['sensibilidad'], 0)} de cada 100 pCR y acierta "
        f"{_num(100 * mi['fuera_de_fold']['especificidad'], 0)} de cada 100 «no pCR». Esta predicción usa un solo "
        "corte; el modelo se evaluó por paciente con la media de sus cortes.",
        icon="ℹ️",
    )


main()
