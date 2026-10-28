"""Lanza lo que diga la sección `experimento` de config.json (D31, D34).

    python -m src.training.experimento                       # config.json de la raíz del repo
    python -m src.training.experimento --config prueba.json  # otro fichero con una sección experimento
    python -m src.training.experimento --ver                 # solo enseña qué se lanzaría

Es la plantilla para probar cosas sin tocar el código: qué versión, en qué modo, con qué pérdida,
en qué folds (por ejemplo solo el 0, unos 15 minutos) y con qué cambios sobre el modo.

`experimento` puede ser una prueba (un objeto) o una lista de pruebas, que se lanzan una detrás
de otra (D34). Así una noche de GPU prueba varias ideas en el fold 0. Para seguirlas:
`python -m src.training.comparar runs/<carpeta de cnn_v3> runs/<carpeta de la prueba>`.

El plan de experimentos (`src.training.plan`) no usa esta sección: siempre entrena los 5 folds con
los valores estándar del modo, que son los que cuentan para comparar versiones (D16). Las pruebas
de un fold o con cambios en el batch, el learning rate o la semilla quedan en el histórico, pero
no cuentan para el criterio.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.config import CAMPOS_MODO, MODOS, RUTA_CONFIG
from src.models.versiones import POR_DEFECTO, VERSIONES

PERDIDAS = ("normal", "ponderada")


def comprobar(e: dict, n: int | None = None) -> dict:
    """Una prueba, completada con los valores por defecto y comprobada (mensaje claro si algo falla)."""
    donde = "experimento" if n is None else f"experimento[{n}]"
    e = {"perdida": "normal", "folds": [0, 1, 2, 3, 4], "cambios": {}, "nota": "", **e}
    if e.get("modelo") not in VERSIONES:
        raise SystemExit(f"{donde}.modelo = {e.get('modelo')!r} no existe. Versiones: {list(VERSIONES)}")
    if e.get("modo") not in MODOS:
        raise SystemExit(f"{donde}.modo debe ser uno de {list(MODOS)}")
    if e["perdida"] not in PERDIDAS:
        raise SystemExit(f"{donde}.perdida debe ser una de {PERDIDAS}")
    folds = e["folds"]
    if not folds or len(set(folds)) != len(folds) or any(f not in range(5) for f in folds):
        raise SystemExit(f"{donde}.folds debe ser una lista de folds distintos entre 0 y 4, por ejemplo [0].")
    desconocidos = set(e["cambios"]) - set(CAMPOS_MODO)
    if desconocidos:
        raise SystemExit(f"{donde}.cambios tiene parámetros que no existen: {sorted(desconocidos)}. "
                         f"Se pueden cambiar: {list(CAMPOS_MODO)}")
    if "fold_val" in e["cambios"]:
        raise SystemExit("Los folds se eligen con experimento.folds, no con cambios.fold_val.")
    propio = {**POR_DEFECTO["entrenamiento"], **VERSIONES[e["modelo"]].get("entrenamiento", {})}["learning_rate"]
    if propio is not None and "learning_rate" in e["cambios"]:
        raise SystemExit(f"{donde}: {e['modelo']} fija su propio learning rate ({propio}); quita "
                         "learning_rate de cambios.")
    return e


def leer_experimento(ruta: Path | str = RUTA_CONFIG) -> list[dict]:
    """Las pruebas de la sección experimento de un JSON (una o una lista), comprobadas."""
    ruta = Path(ruta)
    if not ruta.exists():
        raise SystemExit(f"No existe {ruta}.")
    try:
        datos = json.loads(ruta.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise SystemExit(f"{ruta.name} no es un JSON válido (línea {error.lineno}): {error.msg}") from None
    if "experimento" not in datos:
        raise SystemExit(f"{ruta.name} no tiene sección 'experimento'.")
    seccion = datos["experimento"]
    if isinstance(seccion, dict):
        return [comprobar(seccion)]
    if not isinstance(seccion, list) or not seccion:
        raise SystemExit("experimento debe ser una prueba o una lista de pruebas.")
    return [comprobar(e, n) for n, e in enumerate(seccion)]


def describir(e: dict) -> str:
    modo = MODOS[e["modo"]]
    cambios = ", ".join(f"{k} {getattr(modo, k)} → {v}" for k, v in e["cambios"].items()) or "ninguno"
    duracion = "unos 15 min por fold" if e["modo"] == "completo" else "unos minutos"
    return (f"{e['modelo']} · modo {e['modo']} · pérdida {e['perdida']} · folds {e['folds']} ({duracion})\n"
            f"  cambios sobre el modo: {cambios}\n  nota: {e['nota'] or '—'}")


def main() -> None:
    p = argparse.ArgumentParser(description="Lanza la sección experimento de config.json.")
    p.add_argument("--config", default=str(RUTA_CONFIG), help="JSON con una sección experimento")
    p.add_argument("--ver", action="store_true", help="solo enseñar qué se lanzaría")
    args = p.parse_args()

    pruebas = leer_experimento(args.config)
    for n, e in enumerate(pruebas, 1):
        print(f"{n}/{len(pruebas)}. {describir(e)}")
    if args.ver:
        return
    # Se importa aquí porque carga torch, y solo hace falta al entrenar (--ver no lo necesita).
    from src.training.validacion_cruzada import validar

    for n, e in enumerate(pruebas, 1):
        print(f"\n{'=' * 72}\nPrueba {n} de {len(pruebas)}: {e['modelo']}, {e['nota'] or 'sin nota'}\n{'=' * 72}")
        carpeta = validar(e["modo"], e["modelo"], ponderada=e["perdida"] == "ponderada",
                          nota=e["nota"] or f"experimento {Path(args.config).name}", cambios=e["cambios"],
                          folds=e["folds"])
        (carpeta / "config_usada.json").write_text(json.dumps({"experimento": e}, indent=2, ensure_ascii=False),
                                                   encoding="utf-8")   # qué se lanzó exactamente
        print(f"Configuración copiada en {carpeta / 'config_usada.json'}")
    print(f"\nPruebas terminadas: {len(pruebas)}.")


if __name__ == "__main__":
    main()
