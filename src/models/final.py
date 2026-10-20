"""El modelo final (D21): lo que usan la evaluación de test y la app, con el MISMO código.

Qué es:
- Un **conjunto** con los 5 modelos de la validación cruzada de la versión elegida (uno por fold,
  cada uno con su mejor época). Para un corte, cada modelo da un logit y se hace la media.
- Una **calibración de Platt**: logit_calibrado = a · logit_medio + b, con a y b ajustados con las
  predicciones fuera de fold (nunca con test). Así la probabilidad que se muestra se parece más a
  la frecuencia real de pCR.
- Todo lo necesario para reconstruirlo está en `models/final/manifiesto.json`: la arquitectura,
  los pesos (un fichero por fold, con su sha256), a y b, el umbral, la agregación y las métricas
  internas. No depende de `versiones.py`, así que la app funciona aunque se añadan versiones.

    from src.models.final import cargar_modelo_final
    modelo, manifiesto = cargar_modelo_final()
    p = modelo.probabilidad(x)      # x: (B, 3, 256, 256) en [0, 1] → (B,) probabilidad por corte
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
from torch import nn

from src.models.cnn import CNN

CARPETA_FINAL = Path(__file__).resolve().parents[2] / "models" / "final"


def sha256(ruta: Path) -> str:
    h = hashlib.sha256()
    with open(ruta, "rb") as f:
        for bloque in iter(lambda: f.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()


class ModeloFinal(nn.Module):
    """Media de los logits de varios modelos de la misma arquitectura + calibración de Platt."""

    def __init__(self, arquitectura: dict, estados: list[dict], platt_a: float = 1.0, platt_b: float = 0.0):
        super().__init__()
        if not estados:
            raise ValueError("El modelo final necesita al menos un modelo")
        self.redes = nn.ModuleList()
        for estado in estados:
            red = CNN(**arquitectura)
            red.load_state_dict(estado)
            self.redes.append(red)
        self.register_buffer("platt", torch.tensor([float(platt_a), float(platt_b)]))
        self.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Logit calibrado de cada corte: (B, 3, 256, 256) → (B,)."""
        logits = torch.stack([red(x).squeeze(1) for red in self.redes]).mean(dim=0)
        return self.platt[0] * logits + self.platt[1]

    @torch.no_grad()
    def probabilidad(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        return torch.sigmoid(self(x))


def huella_modelo(manifiesto: dict) -> str:
    """Identificador corto del modelo final: cambia si cambia cualquier peso, la calibración o el umbral."""
    partes = [f["sha256"] for f in manifiesto["pesos"]]
    partes += [json.dumps(manifiesto["calibracion"], sort_keys=True), str(manifiesto["umbral"])]
    return hashlib.sha256("|".join(partes).encode("utf-8")).hexdigest()[:12]


def cargar_modelo_final(carpeta: Path | str = CARPETA_FINAL, dispositivo: str | torch.device = "cpu"
                        ) -> tuple[ModeloFinal, dict]:
    """Carga y verifica el modelo final. Falla si algún fichero de pesos no coincide con su sha256."""
    carpeta = Path(carpeta)
    ruta = carpeta / "manifiesto.json"
    if not ruta.exists():
        raise FileNotFoundError(f"No hay modelo final en {carpeta}. Se crea con: python -m src.evaluation.cerrar_modelo")
    manifiesto = json.loads(ruta.read_text(encoding="utf-8"))
    estados = []
    for fichero in manifiesto["pesos"]:
        camino = carpeta / fichero["fichero"]
        if sha256(camino) != fichero["sha256"]:
            raise ValueError(f"{camino.name} no coincide con su sha256 del manifiesto: el modelo se ha modificado")
        estados.append(torch.load(camino, map_location="cpu", weights_only=True))
    cal = manifiesto["calibracion"]
    modelo = ModeloFinal(manifiesto["arquitectura"], estados, cal["a"], cal["b"]).to(dispositivo)
    return modelo, manifiesto
