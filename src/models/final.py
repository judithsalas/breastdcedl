"""El modelo final (D21): lo que usan la evaluación de test y la app, con el MISMO código.

Qué es:
- Un **conjunto** de CNN propias, todas desde cero (D36). Cada versión que entra (un «miembro»)
  aporta los 5 modelos de su validación cruzada, uno por fold, con su mejor época. Para un corte:
  se hace la media de los logits de los 5 modelos de cada versión, y después la media entre
  versiones (cada versión pesa lo mismo). Con una sola versión es el modelo de siempre (D21).
- Una **calibración de Platt**: logit_calibrado = a · logit_medio + b, con a y b ajustados con las
  predicciones fuera de fold (nunca con test). Así la probabilidad que se muestra se parece más a
  la frecuencia real de pCR.
- Todo lo necesario para reconstruirlo está en `experimentos/modelo_final/manifiesto.json`: la arquitectura de
  cada versión, los pesos (un fichero por fold y versión, con su sha256), a y b, el umbral, la
  agregación y las métricas internas. No depende de `versiones.py`, así que la app funciona aunque
  se añadan versiones.

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

CARPETA_FINAL = Path(__file__).resolve().parents[2] / "experimentos" / "modelo_final"


def sha256(ruta: Path) -> str:
    h = hashlib.sha256()
    with open(ruta, "rb") as f:
        for bloque in iter(lambda: f.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()


class ModeloFinal(nn.Module):
    """Conjunto de CNN + calibración de Platt.

    `miembros` = [(arquitectura, [estado de cada fold]), ...], uno por versión. El logit de un corte
    es la media, entre versiones, de la media de los logits de sus modelos.
    """

    def __init__(self, miembros: list[tuple[dict, list[dict]]], platt_a: float = 1.0, platt_b: float = 0.0):
        super().__init__()
        if not miembros or any(not estados for _, estados in miembros):
            raise ValueError("El modelo final necesita al menos un modelo por versión")
        self.redes = nn.ModuleList()
        self.grupos: list[list[int]] = []          # posiciones en self.redes de cada versión
        for arquitectura, estados in miembros:
            grupo = []
            for estado in estados:
                red = CNN(**arquitectura)
                red.load_state_dict(estado)
                grupo.append(len(self.redes))
                self.redes.append(red)
            self.grupos.append(grupo)
        self.register_buffer("platt", torch.tensor([float(platt_a), float(platt_b)]))
        self.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Logit calibrado de cada corte: (B, 3, 256, 256) → (B,)."""
        por_version = [torch.stack([self.redes[i](x).squeeze(1) for i in grupo]).mean(dim=0) for grupo in self.grupos]
        logits = torch.stack(por_version).mean(dim=0)
        return self.platt[0] * logits + self.platt[1]

    @torch.no_grad()
    def probabilidad(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        return torch.sigmoid(self(x))


def miembros(manifiesto: dict) -> list[dict]:
    """Las versiones del conjunto, cada una con su arquitectura y sus pesos.

    Un manifiesto de una sola versión (como los de antes de D36) es un conjunto de un miembro.
    """
    if "miembros" in manifiesto:
        return manifiesto["miembros"]
    return [{"version": manifiesto["version"], "arquitectura": manifiesto["arquitectura"],
             "pesos": manifiesto["pesos"]}]


def huella_modelo(manifiesto: dict) -> str:
    """Identificador corto del modelo final: cambia si cambia cualquier peso, la calibración o el umbral."""
    partes = [f["sha256"] for m in miembros(manifiesto) for f in m["pesos"]]
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
    partes = []
    for miembro in miembros(manifiesto):
        estados = []
        for fichero in miembro["pesos"]:
            camino = carpeta / fichero["fichero"]
            if sha256(camino) != fichero["sha256"]:
                raise ValueError(f"{fichero['fichero']} no coincide con su sha256 del manifiesto: el modelo se ha modificado")
            estados.append(torch.load(camino, map_location="cpu", weights_only=True))
        partes.append((miembro["arquitectura"], estados))
    cal = manifiesto["calibracion"]
    modelo = ModeloFinal(partes, cal["a"], cal["b"]).to(dispositivo)
    return modelo, manifiesto
