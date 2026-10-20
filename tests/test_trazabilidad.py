"""Trazabilidad de las arquitecturas: toda versión tiene origen, estado y ficha en docs/ARQUITECTURAS.md.

Si se añade una versión en src/models/versiones.py sin declarar su ORIGEN y su ESTADO o sin abrir
su ficha, estos tests fallan (también en GitHub Actions).
"""

import shutil

import pandas as pd
import pytest

from src.data.caso import RAIZ_REPO
from src.models import historial
from src.models.construir import cambios, huella
from src.models.versiones import ESTADO, ESTADOS, ORIGEN, VERSIONES

DOCUMENTO = RAIZ_REPO / "docs" / "ARQUITECTURAS.md"


@pytest.mark.parametrize("nombre", sorted(VERSIONES))
def test_toda_version_tiene_origen_y_estado(nombre):
    assert nombre in ORIGEN, f"Falta ORIGEN['{nombre}'] en src/models/versiones.py"
    assert ORIGEN[nombre] is None or ORIGEN[nombre] in VERSIONES
    assert ESTADO.get(nombre) in ESTADOS, f"ESTADO['{nombre}'] debe ser uno de {ESTADOS}"


def test_la_genealogia_no_tiene_ciclos():
    for nombre in VERSIONES:
        vistos, actual = set(), nombre
        while actual is not None:
            assert actual not in vistos, f"Ciclo en ORIGEN a partir de {nombre}"
            vistos.add(actual)
            actual = ORIGEN[actual]


@pytest.mark.parametrize("nombre", sorted(VERSIONES))
def test_toda_version_tiene_ficha(nombre):
    lineas = DOCUMENTO.read_text(encoding="utf-8").splitlines()
    assert f"## {nombre}" in lineas, f"Falta la ficha '## {nombre}' en docs/ARQUITECTURAS.md"


def test_el_documento_tiene_las_marcas_automaticas():
    texto = DOCUMENTO.read_text(encoding="utf-8")
    for bloque in ("resumen", "genealogia", "ejecuciones"):
        assert f"<!-- auto:{bloque}:inicio -->" in texto and f"<!-- auto:{bloque}:fin -->" in texto


def test_huella():
    assert huella("cnn_v1") == huella(dict(VERSIONES["cnn_v1"]))
    # En resultados.json las tuplas se guardan como listas: la huella no debe cambiar.
    como_json = {**VERSIONES["cnn_v1"], "filtros": list(VERSIONES["cnn_v1"]["filtros"])}
    assert huella(como_json) == huella("cnn_v1")
    assert huella("cnn_v1") != huella("cnn_v2")
    assert huella(None) is None


def test_cambios_respecto_al_origen():
    assert cambios("cnn_v1") == []
    assert cambios("cnn_v2") == ["filtros: (16, 32, 64) → (16, 32, 64, 128)"]


def historico_falso(ruta):
    filas = [
        {"fecha": "2026-10-07 21:16", "ejecucion": "20261007-211625_cnn_v1_completo", "tipo": "simple",
         "modo": "completo", "modelo": "cnn_v1", "huella": "deadbeef", "perdida": "normal", "folds": "0",
         "epoca_elegida": 10, "auc_val": 0.585, "auc_val_desv": None, "auc_train": None,
         "diferencia_auc": None, "f1_macro": 0.41},
        {"fecha": "2026-10-16 10:00", "ejecucion": "20261016-100000_cv_cnn_v2_completo", "tipo": "cv",
         "modo": "completo", "modelo": "cnn_v2", "huella": huella("cnn_v2"), "perdida": "normal",
         "folds": "0,1,2,3,4", "epoca_elegida": 12.4, "auc_val": 0.612, "auc_val_desv": 0.031,
         "auc_train": 0.71, "diferencia_auc": 0.098, "f1_macro": 0.55},
        {"fecha": "2026-10-16 12:00", "ejecucion": "20261016-120000_cnn_v2_rapido", "tipo": "simple",
         "modo": "rapido", "modelo": "cnn_v2", "huella": huella("cnn_v2"), "perdida": "normal", "folds": "0",
         "epoca_elegida": 1, "auc_val": 1.0, "auc_val_desv": None, "auc_train": 1.0,
         "diferencia_auc": 0.0, "f1_macro": 0.4},
    ]
    pd.DataFrame(filas).to_csv(ruta, index=False)
    return ruta


def test_tablas_generadas(tmp_path):
    datos = historial.leer_historico(historico_falso(tmp_path / "historico.csv"))
    assert len(datos) == 2                                   # el modo rápido no cuenta

    resumen = historial.tabla_resumen(datos)
    assert "0,612 ± 0,031" in resumen
    assert "0,585 ⚠" in resumen                              # cnn_v1 cambió después de entrenarse
    assert "| 23.761 |" in resumen and "| 97.809 |" in resumen

    ejecuciones = historial.tabla_ejecuciones(datos)
    assert "⚠ cambió" in ejecuciones and "✓" in ejecuciones
    assert "cnn_v2_rapido" not in ejecuciones

    arbol = historial.genealogia()
    assert "cnn_v1 · descartada" in arbol and "└── cnn_v2" in arbol


def test_actualizar_documento_solo_toca_los_bloques(tmp_path):
    copia = tmp_path / "ARQUITECTURAS.md"
    shutil.copy(DOCUMENTO, copia)
    a_mano = [linea for linea in copia.read_text(encoding="utf-8").splitlines() if linea.startswith("### ")]

    historial.actualizar_documento(copia, historico_falso(tmp_path / "historico.csv"))
    texto = copia.read_text(encoding="utf-8")
    assert "20261016-100000_cv_cnn_v2_completo" in texto
    assert [linea for linea in texto.splitlines() if linea.startswith("### ")] == a_mano
    assert historial.actualizar_documento(copia, tmp_path / "historico.csv") is False   # ya al día


def test_copiar_figuras(tmp_path):
    ejecucion = tmp_path / "runs" / "20261016-100000_cnn_v2_completo"
    (ejecucion / "graficas").mkdir(parents=True)
    (ejecucion / "graficas" / "curvas.png").write_bytes(b"png")
    copiadas = historial.copiar_figuras(ejecucion, tmp_path / "figuras")
    assert [p.name for p in copiadas] == ["curvas.png"]
    assert (tmp_path / "figuras" / ejecucion.name / "curvas.png").exists()
