"""La sección experimento de config.json (D31): se comprueba antes de lanzar nada."""

import json

import pytest

from src.training.experimento import describir, leer_experimento


def escribir(tmp_path, **experimento):
    ruta = tmp_path / "prueba.json"
    ruta.write_text(json.dumps({"experimento": experimento}), encoding="utf-8")
    return ruta


def test_la_plantilla_de_config_json_es_valida():
    e = leer_experimento()
    assert e["modo"] in {"rapido", "completo"} and e["perdida"] in {"normal", "ponderada"}
    assert set(e["folds"]) <= set(range(5))


def test_valores_por_defecto_y_descripcion(tmp_path):
    e = leer_experimento(escribir(tmp_path, modelo="cnn_v3", modo="completo", cambios={"learning_rate": 0.0001}))
    assert e["folds"] == [0, 1, 2, 3, 4] and e["perdida"] == "normal"
    assert "learning_rate 0.001 → 0.0001" in describir(e)


@pytest.mark.parametrize("malo", [
    {"modelo": "cnn_v99", "modo": "completo"},
    {"modelo": "cnn_v3", "modo": "uni"},
    {"modelo": "cnn_v3", "modo": "completo", "perdida": "focal"},
    {"modelo": "cnn_v3", "modo": "completo", "folds": [5]},
    {"modelo": "cnn_v3", "modo": "completo", "folds": [0, 0]},
    {"modelo": "cnn_v3", "modo": "completo", "folds": []},
    {"modelo": "cnn_v3", "modo": "completo", "cambios": {"lr": 0.0001}},
    {"modelo": "cnn_v3", "modo": "completo", "cambios": {"fold_val": 2}},
])
def test_errores_con_mensaje(tmp_path, malo):
    with pytest.raises(SystemExit):
        leer_experimento(escribir(tmp_path, **malo))


def test_sin_seccion_experimento(tmp_path):
    ruta = tmp_path / "vacio.json"
    ruta.write_text("{}", encoding="utf-8")
    with pytest.raises(SystemExit):
        leer_experimento(ruta)
