"""Paquete de resultados para Claude: lleva lo pequeño y nunca imágenes ni pesos."""

import zipfile

from src.utils.para_claude import empaquetar, ficheros_para_claude


def test_lleva_resultados_y_no_imagenes_ni_pesos(tmp_path):
    for ruta in ["runs/historico.csv", "runs/20261021_cv_cnn_v2/resumen.json",
                 "runs/20261021_cv_cnn_v2/fold_0/resultados.json", "runs/20261021_cv_cnn_v2/fold_0/mejor.pt",
                 "runs/20261021_cv_cnn_v2/graficas/curvas_folds.png", "docs/resultados/1_auditoria/AUDITORIA.md",
                 "models/final/manifiesto.json", "models/final/fold_0.pt",
                 "breastdcedl/dataset/train/P1/P1_z000_PRE.png", "breastdcedl/metadata/patients.csv"]:
        (tmp_path / ruta).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / ruta).write_text("x", encoding="utf-8")
    elegidos = {f.as_posix() for f in ficheros_para_claude(tmp_path)}
    assert {"runs/historico.csv", "runs/20261021_cv_cnn_v2/resumen.json", "models/final/manifiesto.json",
            "runs/20261021_cv_cnn_v2/graficas/curvas_folds.png", "docs/resultados/1_auditoria/AUDITORIA.md"} <= elegidos
    assert not any(f.endswith(".pt") or f.startswith("breastdcedl/") for f in elegidos)

    destino, _ = empaquetar(tmp_path, tmp_path / "para_claude_prueba.zip")
    with zipfile.ZipFile(destino) as z:
        assert "estado_git.txt" in z.namelist() and "runs/historico.csv" in z.namelist()
