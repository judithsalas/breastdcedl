"""Validación de lo que se sube a la app: misma normalización que el entrenamiento y errores claros."""

import io

import numpy as np
import pytest
from PIL import Image

from src.data.caso import uc
from src.data.entrada import TAMANO_MAXIMO, identificar, leer_fase, validar_corte


def png(valores: np.ndarray, modo: str = "L", formato: str = "PNG") -> bytes:
    salida = io.BytesIO()
    Image.fromarray(valores).convert(modo).save(salida, format=formato)
    return salida.getvalue()


def corte(nombre="ISPY1_1001_z012", semilla=0):
    rng = np.random.default_rng(semilla)
    pre = rng.integers(20, 120, (256, 256), dtype=np.uint8)
    early = np.clip(pre.astype(int) + 60, 0, 255).astype(np.uint8)
    late = np.clip(pre.astype(int) + 40, 0, 255).astype(np.uint8)
    return [(f"{nombre}_PRE.png", png(pre)), (f"{nombre}_EARLY.png", png(early)), (f"{nombre}_LATE.png", png(late))]


def test_misma_normalizacion_que_el_entrenamiento(tmp_path):
    datos = corte()
    for nombre, contenido in datos:
        (tmp_path / nombre).write_bytes(contenido)
    x, errores, _ = validar_corte(datos)
    assert not errores
    for i, (nombre, _) in enumerate(datos):                       # orden PRE, EARLY, LATE
        assert np.array_equal(x[i], uc.cargar_fase(nombre, tmp_path))
    assert x.dtype == np.float32 and x.shape == (3, 256, 256)


def test_el_orden_de_subida_no_importa():
    datos = corte()
    x1, _, _ = validar_corte(datos)
    x2, _, _ = validar_corte(datos[::-1])
    assert np.array_equal(x1, x2)


def test_identificar_fase():
    assert identificar("carpeta/ISPY2_782334_z010_EARLY.png") == ("ISPY2_782334_z010", "EARLY")
    assert identificar("Breast_MRI_002_z050_late.PNG") == ("Breast_MRI_002_z050", "LATE")
    assert identificar("imagen.png") is None


@pytest.mark.parametrize("cambio, mensaje", [
    (lambda d: d[:2], "Falta la fase LATE"),
    (lambda d: [*d, d[0]], "duplicada"),
    (lambda d: [d[0], d[1], ("ISPY1_1001_z012_LATE.png", d[1][1])], "idénticos"),
    (lambda d: [d[0], d[1], ("OTRO_z001_LATE.png", d[2][1])], "mismo corte"),
    (lambda d: [d[0], d[1], ("ISPY1_1001_z012_FASE3.png", d[2][1])], "debe acabar en"),
])
def test_fases_ausentes_duplicadas_o_mal_nombradas(cambio, mensaje):
    x, errores, _ = validar_corte(cambio(corte()))
    assert x is None and any(mensaje in e for e in errores)


@pytest.mark.parametrize("contenido, mensaje", [
    (png(np.zeros((128, 128), np.uint8)), "128×128"),
    (png(np.zeros((256, 256, 3), np.uint8), modo="RGB"), "modo RGB"),
    (png(np.zeros((256, 256), np.uint8), formato="JPEG"), "no es un PNG"),
    (png(np.full((256, 256), 50, np.uint8))[:200], "dañado"),
    (b"\x89PNG\r\n\x1a\n" + b"0" * (TAMANO_MAXIMO + 1), "máximo"),
])
def test_ficheros_invalidos(contenido, mensaje):
    matriz, errores = leer_fase("x_PRE.png", contenido)
    assert matriz is None and any(mensaje in e for e in errores)


def test_aviso_si_las_fases_parecen_cambiadas():
    d = corte()
    cambiado = [("c_PRE.png", d[1][1]), ("c_EARLY.png", d[0][1]), ("c_LATE.png", d[2][1])]
    x, errores, avisos = validar_corte(cambiado)
    assert x is not None and not errores and any("cambiadas" in a for a in avisos)
