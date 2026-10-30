"""Lanza lo que diga la sección `experimento` de configs/config.json (D31, D34).

    python -m src.training.experimento                       # configs/config.json
    python -m src.training.experimento --config prueba.json  # otro fichero con una sección experimento
    python -m src.training.experimento --ver                 # solo enseña qué se lanzaría
    python -m src.training.experimento --repetir             # vuelve a entrenar también las ya hechas

Es la plantilla para probar cosas sin tocar el código: qué versión, en qué modo, con qué pérdida,
en qué folds (por ejemplo solo el 0, unos 15 minutos) y con qué cambios sobre el modo.

`experimento` puede ser una prueba (un objeto) o una lista de pruebas, que se lanzan una detrás
de otra (D34). Así una noche de GPU prueba varias ideas en el fold 0. Para seguirlas:
`python -m src.training.comparar runs/<carpeta de cnn_v3> runs/<carpeta de la prueba>`.

Es reanudable (D35): una prueba que ya terminó tiene su carpeta en runs/ con `config_usada.json`
igual a ella, y no se repite. Si la tarjeta se cuelga a mitad, se vuelve a lanzar el mismo comando
y sigue por la que falló.

Así se entrenó el modelo final: la sección `experimento` de configs/config.json lanza las 6 versiones en
modo `fijo` (6 épocas, D37) en los 5 folds. Sus ejecuciones están en experimentos/.

Para comparar versiones (D16) solo cuentan las validaciones cruzadas de 5 folds con los valores
estándar del modo. Las pruebas de un fold o con cambios en el batch, el learning rate o la semilla
quedan en el histórico, pero no cuentan para el criterio.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.config import CAMPOS_MODO, MODOS, RUTA_CONFIG
from src.models.versiones import POR_DEFECTO, VERSIONES

PERDIDAS = ("normal", "ponderada")
# Sin importar src.data.caso, que carga torch: --ver tiene que ser instantáneo.
CARPETA_RUNS = Path(__file__).resolve().parents[2] / "runs"


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


def ya_hecha(e: dict, carpeta_runs: Path = CARPETA_RUNS) -> Path | None:
    """La carpeta de una ejecución terminada de esta misma prueba, o None.

    `config_usada.json` se escribe al terminar la prueba, así que una que se cortó no cuenta.
    """
    for ruta in sorted(Path(carpeta_runs).glob("*/config_usada.json")):
        try:
            if json.loads(ruta.read_text(encoding="utf-8")).get("experimento") == e:
                return ruta.parent
        except (json.JSONDecodeError, OSError):
            continue
    return None


def describir(e: dict) -> str:
    modo = MODOS[e["modo"]]
    cambios = ", ".join(f"{k} {getattr(modo, k)} → {v}" for k, v in e["cambios"].items()) or "ninguno"
    duracion = {"completo": "unos 15 min por fold", "fijo": "unos 4 min por fold"}.get(e["modo"], "unos minutos")
    return (f"{e['modelo']} · modo {e['modo']} · pérdida {e['perdida']} · folds {e['folds']} ({duracion})\n"
            f"  cambios sobre el modo: {cambios}\n  nota: {e['nota'] or '—'}")


def main() -> None:
    p = argparse.ArgumentParser(description="Lanza la sección experimento de configs/config.json.")
    p.add_argument("--config", default=str(RUTA_CONFIG), help="JSON con una sección experimento")
    p.add_argument("--ver", action="store_true", help="solo enseñar qué se lanzaría")
    p.add_argument("--repetir", action="store_true", help="entrenar también las pruebas ya hechas")
    args = p.parse_args()

    pruebas = leer_experimento(args.config)
    hechas = {} if args.repetir else {n: c for n, e in enumerate(pruebas, 1) if (c := ya_hecha(e))}
    for n, e in enumerate(pruebas, 1):
        estado = f"  [hecha: {hechas[n].name}, no se repite]" if n in hechas else ""
        print(f"{n}/{len(pruebas)}. {describir(e)}{estado}")
    if args.ver:
        return
    # Se importa aquí porque carga torch, y solo hace falta al entrenar (--ver no lo necesita).
    from src.training.validacion_cruzada import validar

    for n, e in enumerate(pruebas, 1):
        if n in hechas:
            continue
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
