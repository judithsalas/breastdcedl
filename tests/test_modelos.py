"""Toda versión de la CNN debe convertir (B, 3, 256, 256) en (B, 1)."""

import pytest
import torch

from src.models.cnn import CNN
from src.models.construir import contar_parametros, crear_modelo
from src.models.versiones import VERSIONES


def salida(modelo):
    modelo.eval()
    with torch.no_grad():
        return modelo(torch.rand(2, 3, 256, 256))


@pytest.mark.parametrize("nombre", sorted(VERSIONES))
def test_versiones_dan_un_logit(nombre):
    assert salida(crear_modelo(nombre)).shape == (2, 1)


@pytest.mark.parametrize("opciones", [
    {"filtros": (8, 16, 32, 64), "convs_por_bloque": 2},
    {"filtros": (16, 32), "pooling": "avg", "batchnorm": False},
    {"filtros": (16, 32, 64), "resumen": "flatten"},
    {"filtros": (16, 32, 64), "cabeza_oculta": 32, "dropout": 0.3},
    {"filtros": (16, 32, 64), "kernel": 5},
])
def test_variantes_de_la_cnn(opciones):
    assert salida(CNN(**opciones)).shape == (2, 1)


def test_cnn_v1_es_pequena():
    assert contar_parametros(crear_modelo("cnn_v1")) < 30_000


def test_opciones_invalidas():
    with pytest.raises(ValueError):
        CNN(filtros=(16,), kernel=4)
    with pytest.raises(ValueError):
        crear_modelo("no_existe")


def test_entrada_con_realce():
    red = CNN(filtros=(8, 16), entrada="fases+realce")
    x = torch.rand(2, 3, 256, 256)
    dentro = red.preparar_entrada(x)
    assert dentro.shape == (2, 5, 256, 256)                       # recibe 3, ve 5
    assert torch.allclose(dentro[:, 3], x[:, 1] - x[:, 0])          # EARLY − PRE
    assert torch.allclose(dentro[:, 4], x[:, 2] - x[:, 1])          # LATE − EARLY
    assert salida(red).shape == (2, 1)
    with pytest.raises(ValueError):
        CNN(filtros=(8,), entrada="rgb")


def test_receta_y_arquitectura():
    from src.models.construir import arquitectura, receta
    assert receta("cnn_v1") == {"aumento": None}                    # las antiguas: sin aumento
    assert receta("cnn_v3") == {"aumento": "geometrico"}
    assert "entrenamiento" not in arquitectura("cnn_v3")            # la red no recibe la receta


def test_huellas_de_las_versiones_ya_entrenadas_no_cambian():
    # cnn_v1 se entrenó (R1, R2) con esta definición: si cambia, es otra versión.
    from src.models.construir import huella
    assert huella("cnn_v1") == "a5f7edc6"
