"""Lanza lo que diga la sección `experimento` de config.json (D31).

    python -m src.training.experimento                       # config.json de la raíz del repo
    python -m src.training.experimento --config prueba.json  # otro fichero con una sección experimento
    python -m src.training.experimento --ver                 # solo enseña qué se lanzaría

Es la plantilla para probar cosas sin tocar el código: qué versión, en qué modo, con qué pérdida,
en qué folds (por ejemplo solo el 0, unos 10 minutos) y con qué cambios sobre el modo. Para
seguirlo mientras entrena: `python -m src.training.comparar cnn_v3 <versión>`.

El plan de experimentos (`src.training.plan`) no usa esta sección: siempre entrena los 5 folds con
los valores estándar del modo, que son los que cuentan para comparar versiones (D16). Las pruebas
de un fold o con cambios en el batch, el learning rate o la semilla quedan en el histórico, pero
no cuentan para el criterio.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from src.config import CAMPOS_MODO, MODOS, RUTA_CONFIG
from src.models.versiones import VERSIONES

PERDIDAS = ("normal", "ponderada")


def leer_experimento(ruta: Path | str = RUTA_CONFIG) -> dict:
    """La sección experimento de un JSON, comprobada, con un mensaje claro si algo está mal."""
    ruta = Path(ruta)
    if not ruta.exists():
        raise SystemExit(f"No existe {ruta}.")
    try:
        datos = json.loads(ruta.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise SystemExit(f"{ruta.name} no es un JSON válido (línea {error.lineno}): {error.msg}") from None
    if "experimento" not in datos:
        raise SystemExit(f"{ruta.name} no tiene sección 'experimento'.")
    e = {"perdida": "normal", "folds": [0, 1, 2, 3, 4], "cambios": {}, "nota": "", **datos["experimento"]}
    if e.get("modelo") not in VERSIONES:
        raise SystemExit(f"experimento.modelo = {e.get('modelo')!r} no existe. Versiones: {list(VERSIONES)}")
    if e.get("modo") not in MODOS:
        raise SystemExit(f"experimento.modo debe ser uno de {list(MODOS)}")
    if e["perdida"] not in PERDIDAS:
        raise SystemExit(f"experimento.perdida debe ser una de {PERDIDAS}")
    folds = e["folds"]
    if not folds or len(set(folds)) != len(folds) or any(f not in range(5) for f in folds):
        raise SystemExit("experimento.folds debe ser una lista de folds distintos entre 0 y 4, por ejemplo [0].")
    desconocidos = set(e["cambios"]) - set(CAMPOS_MODO)
    if desconocidos:
        raise SystemExit(f"experimento.cambios tiene parámetros que no existen: {sorted(desconocidos)}. "
                         f"Se pueden cambiar: {list(CAMPOS_MODO)}")
    if "fold_val" in e["cambios"]:
        raise SystemExit("Los folds se eligen con experimento.folds, no con cambios.fold_val.")
    return e


def describir(e: dict) -> str:
    modo = MODOS[e["modo"]]
    cambios = ", ".join(f"{k} {getattr(modo, k)} → {v}" for k, v in e["cambios"].items()) or "ninguno"
    duracion = "unos 10 min por fold" if e["modo"] == "completo" else "unos minutos"
    return (f"{e['modelo']} · modo {e['modo']} · pérdida {e['perdida']} · folds {e['folds']} ({duracion})\n"
            f"  cambios sobre el modo: {cambios}\n  nota: {e['nota'] or '—'}")


def main() -> None:
    p = argparse.ArgumentParser(description="Lanza la sección experimento de config.json.")
    p.add_argument("--config", default=str(RUTA_CONFIG), help="JSON con una sección experimento")
    p.add_argument("--ver", action="store_true", help="solo enseñar qué se lanzaría")
    args = p.parse_args()

    e = leer_experimento(args.config)
    print(describir(e))
    if args.ver:
        return
    # Se importa aquí porque carga torch, y solo hace falta al entrenar (--ver no lo necesita).
    from src.training.validacion_cruzada import validar

    carpeta = validar(e["modo"], e["modelo"], ponderada=e["perdida"] == "ponderada",
                      nota=e["nota"] or f"experimento {Path(args.config).name}", cambios=e["cambios"],
                      folds=e["folds"])
    shutil.copy(args.config, carpeta / "config_usada.json")   # qué se lanzó exactamente, junto a sus resultados
    print(f"Configuración copiada en {carpeta / 'config_usada.json'}")


if __name__ == "__main__":
    main()
