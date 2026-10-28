"""Diagrama de bloques de una versión de la CNN, generado desde src/models/versiones.py.

    python -m src.models.dibujar cnn_v1

Guarda docs/figuras/<version>.png y .svg. La altura de cada caja es proporcional
al lado del mapa (256, 128, ...) y su anchura crece con el número de canales.
"""

from __future__ import annotations

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from src.data.caso import RAIZ_REPO
from src.data.recorte import LADO_RECORTE
from src.models.cnn import CANALES_POR_ENTRADA, LADO_ENTRADA
from src.models.construir import contar_parametros, crear_modelo, receta
from src.models.versiones import POR_DEFECTO, VERSIONES

COLORES = {
    "entrada": ("#dbe8f7", "#4a7fc1"),
    "bloque": ("#dff1e8", "#3f9a6e"),
    "resumen": ("#f3e6f7", "#9a5bb0"),
    "cabeza": ("#f3e6f7", "#9a5bb0"),
    "salida": ("#fbe9dc", "#d0743c"),
}
TEXTO = "#1f2933"
SUAVE = "#5b6773"


def etapas(nombre: str) -> list[dict]:
    """Lista de cajas a dibujar, con su texto y sus parámetros."""
    v = {**POR_DEFECTO, **VERSIONES[nombre]}
    modelo = crear_modelo(nombre)
    filtros = v["filtros"]
    conv = f"conv {v['kernel']}×{v['kernel']}, {{c}} filtros"
    capas = ["BatchNorm + ReLU" if v["batchnorm"] else "ReLU"]
    repeticion = f"(×{v['convs_por_bloque']})" if v["convs_por_bloque"] > 1 else None
    pool = "MaxPool 2×2" if v["pooling"] == "max" else "AvgPool 2×2"

    canales_entrada = CANALES_POR_ENTRADA[v["entrada"]]
    lineas_entrada = ["PRE · EARLY · LATE"]
    if v.get("normalizacion") == "pre":
        lineas_entrada += ["÷ media del PRE en el tejido", "(intensidad relativa, dentro)"]
    if v["entrada"] == "fases+realce":
        lineas_entrada += ["+ EARLY−PRE, LATE−EARLY", "(calculados dentro)"]
    lado = LADO_ENTRADA
    if receta(nombre).get("recorte") == "caja":            # D33: la red ve un recorte alrededor del tumor
        lado = LADO_RECORTE
        lineas_entrada += [f"recorte {LADO_RECORTE}×{LADO_RECORTE} en el tumor", "(con la caja; no desplegable)"]
    lista = [{"tipo": "entrada", "titulo": "Entrada", "lineas": lineas_entrada,
              "forma": f"{canales_entrada} @ {lado}×{lado}", "lado": lado,
              "canales": canales_entrada, "params": 0}]

    bloques = list(modelo.extractor)
    por_bloque = len(bloques) // len(filtros)
    for i, c in enumerate(filtros):
        lado //= 2
        modulos = bloques[i * por_bloque:(i + 1) * por_bloque]
        lista.append({"tipo": "bloque", "titulo": f"Bloque {i + 1}", "lineas": [conv.format(c=c), *capas, *([repeticion] if repeticion else []), pool],
                      "forma": f"{c} @ {lado}×{lado}", "lado": lado, "canales": c,
                      "params": sum(p.numel() for m in modulos for p in m.parameters())})

    if v["resumen"] == "global":
        n = filtros[-1]
        lista.append({"tipo": "resumen", "titulo": "Pooling global", "lineas": ["media de cada canal"],
                      "forma": f"vector {n}", "lado": 24, "canales": filtros[-1], "params": 0})
    elif v["resumen"] == "atencion":
        n = filtros[-1]
        lista.append({"tipo": "resumen", "titulo": "Pooling por atención",
                      "lineas": [f"un peso por posición ({lado}×{lado})", "media ponderada"],
                      "forma": f"vector {n}", "lado": 24, "canales": filtros[-1],
                      "params": sum(p.numel() for p in modelo.resumen.parameters())})
    elif v["resumen"] == "media+max":
        n = 2 * filtros[-1]
        lista.append({"tipo": "resumen", "titulo": "Media + máximo", "lineas": ["media de cada canal", "+ máximo de cada canal"],
                      "forma": f"vector {n}", "lado": 24, "canales": filtros[-1], "params": 0})
    else:
        n = filtros[-1] * lado * lado
        lista.append({"tipo": "resumen", "titulo": "Flatten", "lineas": ["todos los valores"],
                      "forma": f"vector {n:,}".replace(",", "."), "lado": 24, "canales": filtros[-1],
                      "params": 0})

    cabeza = []
    if v["cabeza_oculta"]:
        cabeza.append(f"Linear {n}→{v['cabeza_oculta']} + ReLU")
        n = v["cabeza_oculta"]
    if v["dropout"]:
        cabeza.append(f"Dropout {v['dropout']}")
    cabeza.append(f"Linear {n}→1")
    lista.append({"tipo": "cabeza", "titulo": "Cabeza", "lineas": cabeza, "forma": "1 logit",
                  "lado": 16, "canales": 8,
                  "params": sum(p.numel() for p in modelo.cabeza.parameters())})

    lista.append({"tipo": "salida", "titulo": "Salida", "lineas": ["sigmoide (solo al evaluar)"],
                  "forma": "P(pCR = 1)", "lado": 16, "canales": 8, "params": 0})
    for e in lista:
        e["total"] = contar_parametros(modelo)
    return lista


def dibujar(nombre: str) -> list:
    lista = etapas(nombre)
    total = lista[0]["total"]

    ancho_fig = 2.25 * len(lista) + 0.6
    fig, ax = plt.subplots(figsize=(ancho_fig, 5.6))
    ax.set_xlim(0, ancho_fig)
    ax.set_ylim(-3.05, 3.15)
    ax.axis("off")

    paso = (ancho_fig - 0.6) / len(lista)
    centros = [0.3 + paso * (i + 0.5) for i in range(len(lista))]
    for i, e in enumerate(lista):
        es_mapa = e["tipo"] in {"entrada", "bloque"}
        alto = 0.55 + 2.1 * (e["lado"] / LADO_ENTRADA) ** 0.5 if es_mapa else 0.9
        ancho = 1.05 + 0.04 * e["canales"] ** 0.5 if es_mapa else 1.25
        fondo, borde = COLORES[e["tipo"]]
        x0, y0 = centros[i] - ancho / 2, 0.55 - alto / 2
        ax.add_patch(FancyBboxPatch((x0, y0), ancho, alto, boxstyle="round,pad=0.02,rounding_size=0.06",
                                    facecolor=fondo, edgecolor=borde, linewidth=1.6))
        ax.text(centros[i], 2.75, e["titulo"], ha="center", va="center", fontsize=11,
                fontweight="bold", color=TEXTO)
        ax.text(centros[i], 0.55, e["forma"], ha="center", va="center", fontsize=9, color=TEXTO)
        for j, linea in enumerate(e["lineas"]):
            ax.text(centros[i], -1.2 - 0.32 * j, linea, ha="center", va="center", fontsize=8.2,
                    color=SUAVE)
        if e["params"]:
            ax.text(centros[i], -2.75, f"{e['params']:,} parám.".replace(",", "."), ha="center",
                    va="center", fontsize=8.4, color=borde, fontweight="bold")
        if i:
            ax.add_patch(FancyArrowPatch((centros[i - 1] + paso * 0.36, 0.55),
                                         (centros[i] - paso * 0.36, 0.55),
                                         arrowstyle="-|>", mutation_scale=12, color=SUAVE, linewidth=1.2))

    fig.suptitle(f"{nombre}  ·  {total:,} parámetros entrenables".replace(",", "."),
                 fontsize=14, fontweight="bold", color=TEXTO, y=0.99)
    como = receta(nombre)
    aumento = como["aumento"]
    nombre_aumento = {"geometrico": "geométrico",
                      "geometrico+giros90": "geométrico con giros de 90°",
                      "geometrico+escaner": "geométrico de escáner (zoom, brillo y ruido)"}.get(aumento, aumento)
    nota_receta = f" Entrenada con aumento {nombre_aumento} (solo en entrenamiento)." if aumento else ""
    if como.get("optimizador") == "adamw":
        nota_receta += f" Optimizador AdamW, weight decay {como['weight_decay']:g}."
    if como.get("learning_rate"):
        nota_receta += f" Learning rate {como['learning_rate']:g}."
    if como.get("cortes_por_paciente"):
        nota_receta += f" {como['cortes_por_paciente']} cortes por paciente y época."
    if como.get("programador") == "meseta":
        nota_receta += " Learning rate a la mitad si se estanca."
    fig.text(0.5, 0.015, "Los 3 canales de entrada son instantes de la DCE-MRI, no colores. "
             "Altura de cada caja según el tamaño del mapa; anchura según los canales." + nota_receta,
             ha="center", fontsize=8.5, color=SUAVE)

    carpeta = RAIZ_REPO / "docs" / "figuras"
    carpeta.mkdir(parents=True, exist_ok=True)
    rutas = [carpeta / f"{nombre}.png", carpeta / f"{nombre}.svg"]
    for ruta in rutas:
        fig.savefig(ruta, dpi=200, bbox_inches="tight", facecolor="white",
                    metadata={"Date": None} if ruta.suffix == ".svg" else None)
    plt.close(fig)
    return rutas


if __name__ == "__main__":
    for ruta in dibujar(sys.argv[1] if len(sys.argv) > 1 else "cnn_v1"):
        print(f"Guardado {ruta.relative_to(RAIZ_REPO)}")
