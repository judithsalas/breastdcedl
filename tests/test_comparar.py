"""Comparar dos versiones en un fold: encuentra las ejecuciones, lee lo que haya y dibuja."""

import subprocess
import sys

import pandas as pd
import pytest

from src.training import comparar


def ejecucion(raiz, nombre, aucs, con_pacientes=True):
    fold = raiz / nombre / "fold_0"
    fold.mkdir(parents=True)
    pd.DataFrame({"epoca": range(1, len(aucs) + 1), "auc": aucs, "auc_tr": [a + 0.1 for a in aucs],
                  "perdida_tr": 0.6, "perdida_va": 0.6,
                  "mejor": [a == max(aucs) for a in aucs]}).to_csv(fold / "historial.csv", index=False)
    if con_pacientes:
        pd.DataFrame({"patient_id": [f"P{i}" for i in range(8)], "cohorte": ["spy1", "spy2"] * 4,
                      "pCR": [0, 0, 1, 1] * 2, "prob_media": [0.1, 0.2, 0.7, 0.8, 0.3, 0.1, 0.6, 0.9]}).to_csv(
            fold / "predicciones_val_pacientes.csv", index=False)
    return raiz / nombre


def test_busca_la_ultima_validacion_cruzada_de_cada_version(tmp_path):
    vieja = ejecucion(tmp_path, "20261020-100000_cv_cnn_v1_completo", [0.5])
    nueva = ejecucion(tmp_path, "20261021-100000_cv_cnn_v1_completo", [0.5])
    ejecucion(tmp_path, "20261022-100000_cv_cnn_v1_completo_ponderada", [0.5])
    ejecucion(tmp_path, "20261022-100000_cv_cnn_v10_completo", [0.5])
    assert comparar.buscar_ejecucion("cnn_v1", raiz=tmp_path) == nueva       # ni la ponderada ni cnn_v10
    assert comparar.buscar_ejecucion(str(vieja), raiz=tmp_path) == vieja       # una ruta, tal cual
    with pytest.raises(SystemExit):
        comparar.buscar_ejecucion("cnn_v9", raiz=tmp_path)


def test_compara_aunque_la_nueva_vaya_por_la_mitad(tmp_path, capsys):
    ejecucion(tmp_path, "20261020-100000_cv_cnn_v3_completo", [0.56, 0.54, 0.52, 0.50])
    nueva = ejecucion(tmp_path, "20261021-100000_cv_cnn_v10_completo", [0.55, 0.60], con_pacientes=False)
    ruta = comparar.comparar("cnn_v3", "cnn_v10", raiz=tmp_path)
    assert ruta == nueva / "graficas" / "comparacion_fold0_con_cnn_v3.png" and ruta.exists()
    texto = capsys.readouterr().out
    assert "cnn_v3: 4 épocas · elegida la 1" in texto and "cnn_v10: 2 épocas · elegida la 2" in texto
    assert "¿delata el hospital?" in texto                    # la referencia tiene predicciones


def test_no_carga_torch():
    """Se usa mientras entrena: no debe cargar torch (en un proceso aparte, sin lo que importen otros tests)."""
    orden = "import sys, src.training.comparar; print('torch' in sys.modules)"
    salida = subprocess.run([sys.executable, "-c", orden], capture_output=True, text=True, check=True)
    assert salida.stdout.strip() == "False"
