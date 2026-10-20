"""El aumento de datos transforma igual las tres fases y no toca la intensidad."""

import pytest
import torch

from src.data.aumento import AumentoGeometrico, crear_aumento


def test_misma_transformacion_en_las_tres_fases():
    torch.manual_seed(0)
    base = torch.rand(1, 256, 256)
    for _ in range(10):
        y = AumentoGeometrico()(base.repeat(3, 1, 1))
        assert torch.equal(y[0], y[1]) and torch.equal(y[1], y[2])


def test_cambia_la_imagen_y_respeta_el_rango():
    torch.manual_seed(0)
    x = torch.rand(3, 256, 256)
    y = AumentoGeometrico(grados=10, p_volteo=1.0)(x)
    assert y.shape == x.shape and not torch.equal(x, y)
    assert float(y.min()) >= 0.0 and float(y.max()) <= 1.0


def test_crear_aumento():
    assert crear_aumento(None) is None
    assert isinstance(crear_aumento("geometrico"), AumentoGeometrico)
    with pytest.raises(ValueError):
        crear_aumento("color")
    with pytest.raises(ValueError):
        AumentoGeometrico()(torch.rand(1, 256, 256))
