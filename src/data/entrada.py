"""Lectura y validación de las 3 fases de un corte que llegan de fuera (app y predicción por línea de comandos).

La app web no puede fiarse de lo que le suben. Antes de pasar nada a la red se comprueba
(enunciado §9 y guía C6):
- que lleguen exactamente tres ficheros, uno por fase, reconocida por el sufijo del nombre
  (`_PRE.png`, `_EARLY.png`, `_LATE.png`, como en el dataset): ni fases ausentes ni duplicadas;
- que las tres sean del mismo corte (mismo nombre antes del sufijo) y no el mismo fichero repetido;
- que cada fichero no pase de 2 MB, sea de verdad un PNG (firma y decodificación completa), en
  gris de 8 bits (modo L) y de 256×256 píxeles;
- que los valores sean finitos.
Si algo falla, se devuelve un mensaje claro por cada problema y no se predice nada.

La conversión a números es **idéntica** a la del entrenamiento (utils_caso.cargar_fase): gris de
8 bits dividido entre 255, apilado en el orden PRE, EARLY, LATE. Un test comprueba que dan
exactamente lo mismo. Nada de lo subido se ejecuta ni se guarda en disco, y no se aceptan rutas.
"""

from __future__ import annotations

import hashlib
import io
import re

import numpy as np
from PIL import Image, UnidentifiedImageError

FASES = ("PRE", "EARLY", "LATE")
TAMANO_MAXIMO = 2 * 1024 * 1024          # bytes por fichero; un PNG del dataset ocupa unos 30–70 KB
LADO = 256
FIRMA_PNG = b"\x89PNG\r\n\x1a\n"
_SUFIJO = re.compile(r"^(?P<corte>.+)_(?P<fase>PRE|EARLY|LATE)\.png$", re.IGNORECASE)


def identificar(nombre: str) -> tuple[str, str] | None:
    """('ISPY1_1001_z012', 'EARLY') a partir de 'ISPY1_1001_z012_EARLY.png'; None si no encaja."""
    m = _SUFIJO.match(nombre.strip().split("/")[-1].split("\\")[-1])
    return (m["corte"], m["fase"].upper()) if m else None


def leer_fase(nombre: str, datos: bytes) -> tuple[np.ndarray | None, list[str]]:
    """Un PNG en memoria → (256, 256) float32 en [0, 1], o None y los errores encontrados."""
    errores = []
    if len(datos) > TAMANO_MAXIMO:
        return None, [f"{nombre}: ocupa {len(datos) / 1e6:.1f} MB; el máximo es {TAMANO_MAXIMO / 1e6:.0f} MB."]
    if not datos.startswith(FIRMA_PNG):
        return None, [f"{nombre}: no es un PNG (la firma del fichero no corresponde)."]
    try:
        with Image.open(io.BytesIO(datos)) as png:
            png.verify()                                   # comprueba la integridad sin decodificar
        with Image.open(io.BytesIO(datos)) as png:
            png.load()                                     # decodifica entero: detecta ficheros truncados
            if png.format != "PNG":
                errores.append(f"{nombre}: el formato es {png.format}, no PNG.")
            if png.mode != "L":
                errores.append(f"{nombre}: debe ser gris de 8 bits (modo L) y es modo {png.mode}.")
            if png.size != (LADO, LADO):
                errores.append(f"{nombre}: mide {png.size[0]}×{png.size[1]} y debe medir {LADO}×{LADO}.")
            if errores:
                return None, errores
            matriz = np.asarray(png.convert("L"), dtype=np.float32) / 255.0   # igual que utils_caso
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as error:
        return None, [f"{nombre}: el fichero está dañado o no se puede leer ({type(error).__name__})."]
    if not np.isfinite(matriz).all():
        return None, [f"{nombre}: contiene valores no finitos."]
    return matriz, []


def validar_corte(ficheros: list[tuple[str, bytes]]) -> tuple[np.ndarray | None, list[str], list[str]]:
    """Tres ficheros (nombre, bytes) → (tensor (3, 256, 256) PRE·EARLY·LATE, errores, avisos)."""
    errores, avisos = [], []
    if len(ficheros) != 3:
        errores.append(f"Hacen falta exactamente 3 ficheros (PRE, EARLY y LATE de un corte); han llegado {len(ficheros)}.")

    por_fase: dict[str, list[tuple[str, bytes]]] = {f: [] for f in FASES}
    cortes = set()
    for nombre, datos in ficheros:
        ident = identificar(nombre)
        if ident is None:
            errores.append(f"{nombre}: el nombre debe acabar en _PRE.png, _EARLY.png o _LATE.png "
                           "(como en el dataset) para saber qué fase es.")
            continue
        cortes.add(ident[0].lower())
        por_fase[ident[1]].append((nombre, datos))
    for fase, lista in por_fase.items():
        if len(lista) > 1:
            errores.append(f"Fase {fase} duplicada: {', '.join(n for n, _ in lista)}.")
        elif not lista and len(ficheros) <= 3:
            errores.append(f"Falta la fase {fase}.")
    if len(cortes) > 1:
        errores.append(f"Las tres fases deben ser del mismo corte y llegan de varios: {', '.join(sorted(cortes))}.")
    huellas = [hashlib.sha256(d).hexdigest() for _, d in ficheros]
    if len(set(huellas)) < len(huellas):
        errores.append("Dos de los ficheros son idénticos: una fase está repetida con otro nombre.")
    if errores:
        return None, errores, avisos

    planos = []
    for fase in FASES:
        nombre, datos = por_fase[fase][0]
        matriz, problemas = leer_fase(nombre, datos)
        errores += problemas
        planos.append(matriz)
    if errores:
        return None, errores, avisos

    x = np.stack(planos)
    tejido = x[0] > 0.1
    if tejido.sum() > 50 and x[1][tejido].mean() <= x[0][tejido].mean():
        avisos.append("En el tejido EARLY no es más brillante que PRE. En el dataset siempre lo es: "
                      "comprueba que las fases no estén cambiadas.")
    return x, errores, avisos
