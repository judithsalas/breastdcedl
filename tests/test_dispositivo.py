"""La detección de dispositivo funciona en cualquier máquina (también sin GPU)."""

from src.utils.dispositivo import describir_dispositivo, elegir_dispositivo


def test_elegir_dispositivo():
    assert elegir_dispositivo().type in {"cpu", "cuda"}


def test_describir_dispositivo():
    info = describir_dispositivo()
    assert info["torch"]
    assert info["backend"] == "cpu" or info["gpu"]
