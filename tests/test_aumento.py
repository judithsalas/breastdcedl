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


def test_giros90_salen_las_cuatro_orientaciones_y_las_fases_siguen_juntas():
    torch.manual_seed(0)
    base = torch.zeros(1, 8, 8)
    base[0, 0, :3] = 1.0                          # una marca en una esquina: cada giro la deja en otra
    giros = [torch.rot90(base, k, dims=(1, 2)) for k in range(4)]
    sin_mas = AumentoGeometrico(grados=0, desplazamiento=0, escala=(1.0, 1.0), p_volteo=0, giros90=True)
    vistas = set()
    for _ in range(40):
        y = sin_mas(base.repeat(3, 1, 1))
        assert torch.equal(y[0], y[1]) and torch.equal(y[1], y[2])
        vistas |= {k for k, g in enumerate(giros) if torch.allclose(y[:1], g, atol=1e-6)}
    assert vistas == {0, 1, 2, 3}


def test_crear_aumento():
    assert crear_aumento(None) is None
    assert isinstance(crear_aumento("geometrico"), AumentoGeometrico)
    assert not crear_aumento("geometrico").giros90 and "giros90" not in repr(crear_aumento("geometrico"))
    assert crear_aumento("geometrico+giros90").giros90
    with pytest.raises(ValueError):
        crear_aumento("color")
    with pytest.raises(ValueError):
        AumentoGeometrico()(torch.rand(1, 256, 256))
