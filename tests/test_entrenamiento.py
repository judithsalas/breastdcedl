"""El bucle de entrenamiento funciona de principio a fin con datos sintéticos (sin imágenes)."""

import json

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.config import obtener_config
from src.training.entrenar import entrenar_modelo, muestra_estratificada, predecir


class RedMinima(nn.Module):
    """Solo para el test: no es una propuesta de arquitectura."""

    def __init__(self):
        super().__init__()
        self.capa = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(3, 1))

    def forward(self, x):
        return self.capa(x)


def datos_sinteticos(n_pacientes=6, cortes=3):
    filas = pd.DataFrame({
        "sample_id": [f"P{p}_z{c:03d}" for p in range(n_pacientes) for c in range(cortes)],
        "patient_id": [f"P{p}" for p in range(n_pacientes) for _ in range(cortes)],
        "pCR": [p % 2 for p in range(n_pacientes) for _ in range(cortes)],
    })
    x = torch.rand(len(filas), 3, 256, 256)
    y = torch.tensor(filas.pCR.values, dtype=torch.float32)
    return filas, DataLoader(TensorDataset(x, y), batch_size=4, shuffle=False)


def entrenar(tmp_path, epocas=3, paciencia=5, lr=1e-2, eval_train=True):
    torch.manual_seed(0)
    filas, cargador = datos_sinteticos()
    modelo = RedMinima()
    cfg = obtener_config("rapido", epocas_max=epocas, paciencia=paciencia)
    resultados = {"modelo": "red_minima"}
    entrenar_modelo(modelo, cargador, cargador, filas, nn.BCEWithLogitsLoss(),
                    torch.optim.SGD(modelo.parameters(), lr=lr) if lr == 0 else
                    torch.optim.Adam(modelo.parameters(), lr=lr), torch.device("cpu"),
                    cfg, tmp_path, resultados, eval_train=(cargador, filas) if eval_train else None)
    return filas, resultados


def test_predecir_conserva_el_orden_y_el_numero():
    filas, cargador = datos_sinteticos()
    probs, perdida = predecir(RedMinima(), cargador, torch.device("cpu"), nn.BCEWithLogitsLoss())
    assert len(probs) == len(filas)
    assert ((probs >= 0) & (probs <= 1)).all()
    assert perdida > 0


def test_bucle_completo(tmp_path):
    _, resultados = entrenar(tmp_path)
    assert len(resultados["epocas"]) == 3
    for fichero in ("mejor.pt", "historial.csv", "resultados.json",
                    "graficas/curvas.png", "graficas/matriz_confusion.png"):
        assert (tmp_path / fichero).exists(), fichero
    guardado = json.loads((tmp_path / "resultados.json").read_text(encoding="utf-8"))
    assert guardado["epocas"][-1]["epoca"] == 3


def test_auc_de_entrenamiento_y_diferencia(tmp_path):
    _, resultados = entrenar(tmp_path)
    fila = resultados["epocas"][0]
    assert 0 <= fila["auc_tr"] <= 1
    assert fila["diferencia_auc"] == fila["auc_tr"] - fila["auc"]
    historial = pd.read_csv(tmp_path / "historial.csv")
    assert {"auc_tr", "auc", "diferencia_auc", "f1_macro"} <= set(historial.columns)


def test_sin_eval_train_el_auc_de_entrenamiento_es_nan(tmp_path):
    _, resultados = entrenar(tmp_path, eval_train=False)
    assert pd.isna(resultados["epocas"][0]["auc_tr"])


def test_epoca_elegida_con_matriz_y_metricas_por_clase(tmp_path):
    filas, resultados = entrenar(tmp_path)
    e = resultados["mejor_epoca"]
    mc = e["matriz_confusion"]
    assert sum(mc.values()) == filas.patient_id.nunique()
    assert {"f1_pcr", "f1_no_pcr", "f1_macro", "auc_train", "diferencia_auc"} <= set(e)
    assert e["epoca"] == max(range(len(resultados["epocas"])),
                             key=lambda i: resultados["epocas"][i]["auc"]) + 1


def test_predicciones_de_la_epoca_elegida(tmp_path):
    filas, _ = entrenar(tmp_path)
    cortes = pd.read_csv(tmp_path / "predicciones_val.csv")
    pacientes = pd.read_csv(tmp_path / "predicciones_val_pacientes.csv")
    assert list(cortes.sample_id) == list(filas.sample_id)       # mismo orden que la validación
    assert len(pacientes) == filas.patient_id.nunique()
    medias = cortes.groupby("patient_id").prob.mean()
    assert (pacientes.set_index("patient_id").prob_media - medias).abs().max() < 1e-9


def test_parada_temprana(tmp_path):
    _, resultados = entrenar(tmp_path, epocas=20, paciencia=2, lr=0.0)
    assert len(resultados["epocas"]) == 3   # mejora en la 1, sin mejora en la 2 y la 3 -> para


def test_muestra_estratificada():
    filas = pd.DataFrame({"patient_id": [f"P{p}" for p in range(100) for _ in range(2)],
                          "pCR": [int(p < 30) for p in range(100) for _ in range(2)]})
    muestra = muestra_estratificada(filas, 20, semilla=42)
    pacientes = muestra.groupby("patient_id").pCR.first()
    assert len(pacientes) == 20 and pacientes.sum() == 6                    # 30 % de pCR
    assert muestra.equals(muestra_estratificada(filas, 20, semilla=42))      # reproducible
    assert len(muestra_estratificada(filas, 0, semilla=42)) == len(filas)    # 0 = todas
