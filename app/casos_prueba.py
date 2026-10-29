"""Ficheros para probar la app desplegada con entradas válidas e inválidas (lista de `app/README.md`).

    python -m app.casos_prueba

Parte de dos cortes de `app/ejemplos/` (de train) y deja en `pruebas_app/` (no se versiona) una carpeta
por prueba con los ficheros que hay que subir juntos a la app. Cada prueba se pasa además por la
misma validación que usa la app (`src/data/entrada.py`), y lo que responde queda en
`pruebas_app/LEEME.md`: es lo que la app tiene que enseñar. Así la lista de pruebas se puede repetir y
cualquiera puede comprobarla.

No hacen falta las imágenes del dataset ni la GPU: basta con los ejemplos de la app.
"""

from __future__ import annotations

import io
import json
import shutil
from pathlib import Path

from PIL import Image

from src.data.entrada import FASES, validar_corte

RAIZ = Path(__file__).resolve().parents[1]
EJEMPLOS = Path(__file__).resolve().parent / "ejemplos"
DESTINO = RAIZ / "pruebas_app"


def _cortes(ejemplos: Path) -> list[dict[str, Path]]:
    """Los cortes de ejemplo con sus tres fases: [{'PRE': ruta, 'EARLY': ruta, 'LATE': ruta}, ...]."""
    cortes = []
    for carpeta in sorted(p for p in ejemplos.iterdir() if p.is_dir()):
        fases = {f: carpeta / f"{carpeta.name}_{f}.png" for f in FASES}
        if all(r.exists() for r in fases.values()):
            cortes.append(fases)
    return cortes


def _png(imagen: Image.Image) -> bytes:
    salida = io.BytesIO()
    imagen.save(salida, format="PNG")
    return salida.getvalue()


def _casos(a: dict[str, Path], b: dict[str, Path]) -> list[tuple[str, str, dict[str, bytes]]]:
    """(carpeta, qué se prueba, {nombre de fichero: bytes}) de cada prueba."""
    leer = {f: a[f].read_bytes() for f in FASES}
    nombre = {f: a[f].name for f in FASES}
    pre = Image.open(io.BytesIO(leer["PRE"]))
    jpg = io.BytesIO()
    pre.save(jpg, format="JPEG")
    return [
        ("01_valido", "Las tres fases de un corte de ejemplo: tiene que predecir.", {nombre[f]: leer[f] for f in FASES}),
        ("02_dos_fases", "Solo PRE y EARLY.", {nombre[f]: leer[f] for f in ("PRE", "EARLY")}),
        ("03_misma_imagen_dos_veces", "La imagen EARLY subida también como LATE.",
         {nombre["PRE"]: leer["PRE"], nombre["EARLY"]: leer["EARLY"], nombre["LATE"]: leer["EARLY"]}),
        ("04_jpg_renombrado", "La fase PRE es un JPG con nombre .png.",
         {nombre["PRE"]: jpg.getvalue(), nombre["EARLY"]: leer["EARLY"], nombre["LATE"]: leer["LATE"]}),
        ("05_png_en_color", "La fase PRE guardada en color (RGB).",
         {nombre["PRE"]: _png(pre.convert("RGB")), nombre["EARLY"]: leer["EARLY"], nombre["LATE"]: leer["LATE"]}),
        ("06_otro_tamano", "La fase PRE reducida a 128×128.",
         {nombre["PRE"]: _png(pre.resize((128, 128))), nombre["EARLY"]: leer["EARLY"], nombre["LATE"]: leer["LATE"]}),
        ("07_png_danado", "La fase PRE cortada por la mitad (fichero dañado).",
         {nombre["PRE"]: leer["PRE"][: len(leer["PRE"]) // 2], nombre["EARLY"]: leer["EARLY"],
          nombre["LATE"]: leer["LATE"]}),
        ("08_fases_de_dos_cortes", "PRE y EARLY de un corte y LATE de otro.",
         {nombre["PRE"]: leer["PRE"], nombre["EARLY"]: leer["EARLY"], b["LATE"].name: b["LATE"].read_bytes()}),
        ("09_fases_cambiadas", "PRE y EARLY intercambiadas (cada imagen con el nombre de la otra).",
         {nombre["PRE"]: leer["EARLY"], nombre["EARLY"]: leer["PRE"], nombre["LATE"]: leer["LATE"]}),
    ]


def respuesta(ficheros: dict[str, bytes]) -> str:
    """Lo que responde la validación de la app a esos ficheros, en una línea."""
    _, errores, avisos = validar_corte(list(ficheros.items()))
    if errores:
        return "Error: " + " ".join(errores)
    return "Predice." + (" Aviso: " + " ".join(avisos) if avisos else "")


def crear(ejemplos: Path = EJEMPLOS, destino: Path = DESTINO) -> list[tuple[str, str, str]]:
    """Escribe las carpetas de prueba y su LEEME.md. Devuelve (carpeta, qué se prueba, respuesta)."""
    cortes = _cortes(ejemplos)
    if len(cortes) < 2:
        raise SystemExit(f"Hacen falta al menos 2 cortes en {ejemplos}. Se crean con: python -m app.preparar_ejemplos")
    a, b = cortes[0], cortes[1]
    if destino.exists():
        shutil.rmtree(destino)
    destino.mkdir(parents=True)
    filas = []
    for carpeta, que, ficheros in _casos(a, b):
        (destino / carpeta).mkdir()
        for nombre, datos in ficheros.items():
            (destino / carpeta / nombre).write_bytes(datos)
        filas.append((carpeta, que, respuesta(ficheros)))

    corte = a["PRE"].parent.name
    esperado = {}
    if (ejemplos / "esperado.json").exists():
        esperado = json.loads((ejemplos / "esperado.json").read_text(encoding="utf-8"))
    prob = esperado.get(corte, {}).get("probabilidad_pipeline")
    lineas = [
        "# Pruebas de la app con entradas válidas e inválidas",
        "",
        "Generado con `python -m app.casos_prueba` a partir de los ejemplos de `app/ejemplos/` (cortes de train).",
        "En cada carpeta están los ficheros que hay que subir **juntos** en la pestaña «Subir». La última",
        "columna es lo que responde la validación de la app con esos ficheros: la app desplegada tiene que",
        "enseñar lo mismo.",
        "",
        "| Carpeta | Qué se prueba | Qué tiene que responder | ¿Lo hace? |",
        "|---|---|---|---|",
    ]
    lineas += [f"| `{c}` | {q} | {r} | |" for c, q, r in filas]
    if prob is not None:
        lineas += ["", f"En `01_valido` (corte `{corte}`), la probabilidad del pipeline es **{prob:.6f}**; la app tiene",
                   "que dar la misma."]
    (destino / "LEEME.md").write_text("\n".join(lineas) + "\n", encoding="utf-8")
    return filas


def main() -> None:
    for carpeta, _, respuesta_app in crear():
        print(f"{carpeta:28s} {respuesta_app}")
    print(f"\nCarpetas en {DESTINO}; la lista, en {DESTINO / 'LEEME.md'}.")


if __name__ == "__main__":
    main()
