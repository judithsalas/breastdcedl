"""Predicción del pipeline para un corte suelto, por línea de comandos (misma lectura y modelo que la app).

    python -m src.models.predecir carpeta/ISPY1_1001_z012_PRE.png carpeta/ISPY1_1001_z012_EARLY.png \\
                                  carpeta/ISPY1_1001_z012_LATE.png

Sirve para comprobar en la defensa que la app da exactamente lo mismo que el pipeline: aquí y en
la app se usan src/data/entrada.py (lectura y validación) y src/models/final.py (modelo final).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

from src.data.entrada import validar_corte
from src.models.final import cargar_modelo_final


def predecir(ficheros: list[tuple[str, bytes]], carpeta_final: Path | None = None) -> dict:
    x, errores, avisos = validar_corte(ficheros)
    if errores:
        return {"errores": errores, "avisos": avisos}
    modelo, manifiesto = cargar_modelo_final(carpeta_final) if carpeta_final else cargar_modelo_final()
    p = float(modelo.probabilidad(torch.from_numpy(x).unsqueeze(0))[0])
    return {"probabilidad": p, "clase": "pCR" if p >= manifiesto["umbral"] else "no pCR",
            "umbral": manifiesto["umbral"], "modelo": manifiesto["version"],
            "huella_modelo": manifiesto["huella_modelo"], "errores": [], "avisos": avisos}


def main() -> None:
    rutas = [Path(r) for r in sys.argv[1:]]
    if not rutas:
        sys.exit("Uso: python -m src.models.predecir <corte>_PRE.png <corte>_EARLY.png <corte>_LATE.png")
    r = predecir([(ruta.name, ruta.read_bytes()) for ruta in rutas])
    for mensaje in r["errores"]:
        print(f"ERROR: {mensaje}")
    for mensaje in r["avisos"]:
        print(f"Aviso: {mensaje}")
    if not r["errores"]:
        print(f"P(pCR) = {r['probabilidad']:.6f} · umbral {r['umbral']:.3f} → {r['clase']} "
              f"(modelo {r['modelo']}, {r['huella_modelo']})")


if __name__ == "__main__":
    main()
