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


# --------------------------------------------------------------------------- #
# config.json (D31)
# --------------------------------------------------------------------------- #

def test_los_modos_de_config_json_son_los_de_siempre():
    """Las versiones ya entrenadas usaron estos valores: si cambian, dejan de ser comparables."""
    assert MODOS["completo"].como_dict() == {
        "modo": "completo", "n_pacientes": None, "fold_val": 0, "batch_size": 32, "epocas_max": 60,
        "paciencia": 10, "learning_rate": 1e-3, "pacientes_auc_train": 220, "semilla": 42, "num_workers": 4}
    assert MODOS["rapido"].como_dict() == {
        "modo": "rapido", "n_pacientes": 40, "fold_val": 0, "batch_size": 16, "epocas_max": 2,
        "paciencia": 2, "learning_rate": 1e-3, "pacientes_auc_train": 0, "semilla": 42, "num_workers": 0}


def test_evaluacion_y_comparacion_salen_de_config_json():
    from src.config import AGREGACIONES, COMPARACION, EVALUACION
    from src.data.caso import uc
    from src.evaluation.umbral import ESPECIFICIDAD_MINIMA
    from src.models.historial import SOBREAJUSTE
    from src.training.entrenar import METODO_AGREGACION, UMBRAL

    assert (METODO_AGREGACION, UMBRAL) == (EVALUACION["agregacion"], EVALUACION["umbral_entrenamiento"]) == ("mean", 0.5)
    assert ESPECIFICIDAD_MINIMA == EVALUACION["especificidad_minima"] == 0.8
    assert SOBREAJUSTE == COMPARACION["sobreajuste_maximo"] == 0.15
    assert set(AGREGACIONES) == set(uc.METODOS_AGREGACION)


def test_config_json_mal_escrito_da_un_mensaje_claro(tmp_path):
    import json

    from src.config import CONFIG, leer_config

    ruta = tmp_path / "config.json"
    ruta.write_text("{ esto no es json", encoding="utf-8")
    with pytest.raises(ValueError, match="no es un JSON válido"):
        leer_config(ruta)
    sin_paciencia = json.loads(json.dumps(CONFIG))
    del sin_paciencia["modos"]["completo"]["paciencia"]
    sin_paciencia["modos"]["completo"]["epocas"] = 60
    ruta.write_text(json.dumps(sin_paciencia), encoding="utf-8")
    with pytest.raises(ValueError, match=r"faltan \['paciencia'\], sobran \['epocas'\]"):
        leer_config(ruta)
    otra_agregacion = json.loads(json.dumps(CONFIG))
    otra_agregacion["evaluacion"]["agregacion"] = "moda"
    ruta.write_text(json.dumps(otra_agregacion), encoding="utf-8")
    with pytest.raises(ValueError, match="agregacion"):
        leer_config(ruta)


def test_solo_las_ejecuciones_estandar_cuentan():
    from src.config import es_estandar

    assert es_estandar("completo", 32, 0.001, 42)
    assert es_estandar("completo", None, float("nan"), 42)          # ejecuciones antiguas sin el dato
    assert not es_estandar("completo", 32, 0.0001, 42)                # una prueba con otro learning rate
    assert not es_estandar("completo", 16, 0.001, 42)
    assert es_estandar("rapido", 16, 0.001, 42) and not es_estandar("otro", 32, 0.001, 42)
