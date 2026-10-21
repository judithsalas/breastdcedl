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
    # Las antiguas: sin aumento y con Adam sin weight decay (lo que usaron al entrenarse)
    assert receta("cnn_v1") == {"aumento": None, "optimizador": "adam", "weight_decay": 0.0}
    assert receta("cnn_v3") == {"aumento": "geometrico", "optimizador": "adam", "weight_decay": 0.0}
    assert receta("cnn_v7") == {"aumento": "geometrico", "optimizador": "adamw", "weight_decay": 0.05}
    assert "entrenamiento" not in arquitectura("cnn_v3")            # la red no recibe la receta


def test_huellas_de_las_versiones_ya_entrenadas_no_cambian():
    # Se entrenaron con estas definiciones (R1, R2 y el plan del 2026-10-20): si cambian, son
    # otra versión. Las huellas coinciden con la columna huella de runs/historico.csv.
    from src.models.construir import huella
    assert huella("cnn_v1") == "a5f7edc6"
    assert huella("cnn_v2") == "cc7de9db"
    assert huella("cnn_v3") == "4bb03d5d"
    assert huella("cnn_v4") == "e7188804"


# --------------------------------------------------------------------------- #
# Piezas de la segunda ronda (D28)
# --------------------------------------------------------------------------- #

def test_normalizar_por_pre_quita_la_escala_del_escaner():
    """Si un escáner da todo un 50 % más brillante, la red normalizada ve lo mismo."""
    from src.models.cnn import normalizar_por_pre

    torch.manual_seed(0)
    x = torch.zeros(2, 3, 32, 32)
    x[:, :, 8:24, 8:24] = 0.3 + 0.3 * torch.rand(2, 3, 16, 16)       # tejido (> 0,1) y aire (0)
    a, b = normalizar_por_pre(x), normalizar_por_pre(1.5 * x)
    assert torch.allclose(a, b, atol=1e-6)
    tejido = x[:, 0] > 0.1
    assert torch.allclose(a[:, 0][tejido].view(2, -1).mean(dim=1), torch.ones(2), atol=1e-5)  # PRE ≈ 1
    assert torch.isfinite(normalizar_por_pre(torch.zeros(1, 3, 8, 8))).all()        # sin tejido: sin NaN


def test_red_normalizada_no_depende_de_la_escala():
    torch.manual_seed(0)
    red = CNN(filtros=(8, 16), normalizacion="pre").eval()
    x = torch.zeros(2, 3, 256, 256)
    x[:, :, 64:192, 64:192] = 0.3 + 0.3 * torch.rand(2, 3, 128, 128)
    with torch.no_grad():
        assert torch.allclose(red(x), red(1.5 * x), atol=1e-5)
    with pytest.raises(ValueError):
        CNN(filtros=(8,), normalizacion="zscore")


def test_pooling_por_atencion():
    from src.models.cnn import PoolingAtencion

    torch.manual_seed(0)
    pool = PoolingAtencion(canales=16)
    mapas = torch.rand(3, 16, 4, 4)
    salida = pool(mapas)
    assert salida.shape == (3, 16)
    assert pool.ultimos_pesos.shape == (3, 4, 4)
    assert torch.allclose(pool.ultimos_pesos.sum(dim=(1, 2)), torch.ones(3))     # los pesos suman 1
    # Con todos los pesos iguales sería la media global: la atención la generaliza
    with torch.no_grad():
        pool.w.weight.zero_()
        pool.w.bias.zero_()
    assert torch.allclose(pool(mapas), mapas.mean(dim=(2, 3)), atol=1e-6)


def test_crear_optimizador_segun_la_receta():
    from src.models.construir import crear_modelo, receta
    from src.training.entrenar import crear_optimizador

    adam = crear_optimizador(crear_modelo("cnn_v3"), 1e-3, receta("cnn_v3"))
    assert type(adam) is torch.optim.Adam and adam.param_groups[0]["weight_decay"] == 0
    red = crear_modelo("cnn_v7")
    adamw = crear_optimizador(red, 1e-3, receta("cnn_v7"))
    assert type(adamw) is torch.optim.AdamW
    con, sin = adamw.param_groups
    assert con["weight_decay"] == 0.05 and sin["weight_decay"] == 0.0
    assert all(p.ndim > 1 for p in con["params"]) and all(p.ndim <= 1 for p in sin["params"])
    assert len(con["params"]) + len(sin["params"]) == len(list(red.parameters()))


def test_diagrama_de_las_versiones_nuevas():
    from src.models.dibujar import etapas

    v8 = etapas("cnn_v8")
    textos = " ".join(" ".join(e["lineas"]) + e["titulo"] for e in v8)
    assert "Pooling por atención" in textos and "media del PRE" in textos and "Dropout 0.3" in textos
