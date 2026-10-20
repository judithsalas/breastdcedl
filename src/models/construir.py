"""Construir la red de una versión y saber cosas de ella.

- `crear_modelo("cnn_v2")`: la CNN de esa versión, sin entrenar.
- `arquitectura` y `receta`: los parámetros de la red y cómo se entrena.
- `contar_parametros`: parámetros entrenables (diapositiva 3).
- `huella`: 8 caracteres que identifican la definición; si cambia, la versión se tocó después
  de entrenarla.
- `cambios`: qué cambia una versión respecto a la versión de la que sale.

Las versiones están en `versiones.py` y la red configurable en `cnn.py`.
"""

from __future__ import annotations

import hashlib
import json

from torch import nn

from src.models.cnn import CNN
from src.models.versiones import ORIGEN, POR_DEFECTO, VERSIONES


def arquitectura(nombre: str) -> dict:
    """Solo los parámetros de la red de una versión (sin la receta de entrenamiento)."""
    if nombre not in VERSIONES:
        raise ValueError(f"Modelo desconocido: {nombre!r}. Disponibles: {sorted(VERSIONES)}")
    return {k: v for k, v in VERSIONES[nombre].items() if k != "entrenamiento"}


def receta(nombre: str) -> dict:
    """Cómo se entrena una versión: lo que declara en `entrenamiento`, completado con POR_DEFECTO."""
    if nombre not in VERSIONES:
        raise ValueError(f"Modelo desconocido: {nombre!r}. Disponibles: {sorted(VERSIONES)}")
    return {**POR_DEFECTO["entrenamiento"], **VERSIONES[nombre].get("entrenamiento", {})}


def crear_modelo(nombre: str) -> nn.Module:
    return CNN(**arquitectura(nombre))


def contar_parametros(modelo: nn.Module) -> int:
    """Parámetros entrenables (el enunciado pide indicarlos en la diapositiva 3)."""
    return sum(p.numel() for p in modelo.parameters() if p.requires_grad)


def huella(definicion: str | dict | None) -> str | None:
    """Huella corta (8 caracteres) de la definición de una versión (red + receta).

    Se guarda con cada ejecución en el histórico. Si la huella de una ejecución no coincide con
    la de la definición actual, la versión se modificó después de entrenarla (no se debe: si
    cambia algo, es otra versión). Acepta el nombre de una versión o su diccionario, tal como se
    guarda en resultados.json (las tuplas pasan a listas: la huella no cambia).
    """
    if definicion is None:
        return None
    if isinstance(definicion, str):
        definicion = VERSIONES[definicion]
    texto = json.dumps(definicion, sort_keys=True, default=list)
    return hashlib.sha1(texto.encode("utf-8")).hexdigest()[:8]


def _completa(nombre: str) -> dict:
    """La definición con los valores por defecto explícitos, para comparar versiones."""
    v = VERSIONES[nombre]
    return {**POR_DEFECTO, **v, "entrenamiento": receta(nombre)}


def cambios(nombre: str) -> list[str]:
    """Qué cambia una versión respecto a su versión de origen, decisión a decisión."""
    origen = ORIGEN.get(nombre)
    if origen is None:
        return []
    antes, ahora = _completa(origen), _completa(nombre)
    salida = []
    for k in sorted(set(antes) | set(ahora)):
        if antes.get(k) == ahora.get(k):
            continue
        if isinstance(antes.get(k), dict) and isinstance(ahora.get(k), dict):
            salida += [f"{k}.{s}: {antes[k].get(s)} → {ahora[k].get(s)}"
                       for s in sorted(set(antes[k]) | set(ahora[k])) if antes[k].get(s) != ahora[k].get(s)]
        else:
            salida.append(f"{k}: {antes.get(k)} → {ahora.get(k)}")
    return salida
