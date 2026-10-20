"""Los modos rapido/completo existen y se pueden sobrescribir."""

import pytest

from src.config import MODOS, obtener_config


def test_existen_los_dos_modos():
    assert set(MODOS) == {"rapido", "completo"}


def test_completo_usa_todas_las_pacientes():
    assert obtener_config("completo").n_pacientes is None


def test_auc_de_entrenamiento_configurable():
    assert obtener_config("rapido").pacientes_auc_train == 0      # todas las del modo rápido
    assert obtener_config("completo", pacientes_auc_train=0).pacientes_auc_train == 0


def test_sobrescribir_un_valor():
    cfg = obtener_config("completo", batch_size=64)
    assert cfg.batch_size == 64
    assert MODOS["completo"].batch_size == 32  # el modo original no cambia


def test_none_no_sobrescribe():
    assert obtener_config("rapido", batch_size=None).batch_size == MODOS["rapido"].batch_size


def test_modo_o_parametro_desconocido():
    with pytest.raises(ValueError):
        obtener_config("uni")
    with pytest.raises(ValueError):
        obtener_config("rapido", tamano_lote=8)
