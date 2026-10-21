"""Plan de experimentos (D17): lo que se entrena en la GPU de la universidad, en orden.

    python -m src.training.plan                 # lanza todo lo que falte (se puede dejar de noche)
    python -m src.training.plan --ver           # solo muestra qué está hecho y qué falta
    python -m src.training.plan --modo rapido   # prueba de humo del plan entero (minutos)

Cada paso es una validación cruzada de 5 folds (src/training/validacion_cruzada.py). El plan es
**reanudable**: un paso se da por hecho si runs/historico.csv ya tiene su validación cruzada
completa (5 folds, misma pérdida, mismo modo y huella ✓). Si se corta la luz o falla algo, se
vuelve a lanzar el mismo comando y sigue por donde iba.

Por qué este orden (explicado en docs/COMO_SE_HIZO.md):
  Primera ronda (D17, hecha el 2026-10-20):
  1. cnn_v2  normal     ¿ayuda ver más contexto (4 bloques)?
  2. cnn_v3  normal     ¿ayuda el aumento de datos contra la memorización?
  3. cnn_v4  normal     ¿ayuda darle el realce hecho?
  Segunda ronda (D28, 2026-10-21), cada idea sobre cnn_v3 y las tres juntas:
  4. cnn_v5  normal     ¿ayuda la intensidad relativa al PRE (quitar la escala del hospital)?
  5. cnn_v6  normal     ¿ayuda el pooling por atención (mirar dónde está el tumor)?
  6. cnn_v7  normal     ¿ayuda regularizar (dropout + weight decay)?
  7. cnn_v8  normal     ¿suman las tres?
  Tercera ronda (D29, 2026-10-21):
  8. cnn_v10 normal     ¿ayuda quitar la pista de la orientación (giros de 90°)?
  9. mejor   ponderada  la que gane según el criterio, con pérdida ponderada (lo pide el
                        enunciado: comparar normal y ponderada con el modelo elegido). Si
                        sigue ganando cnn_v3, ya está hecha y no se repite.
Las versiones se comparan con pérdida normal para que solo cambie una cosa cada vez.
"""

from __future__ import annotations

import argparse
import time

from src.models.historial import HISTORICO, elegir_mejor, leer_historico, validaciones_validas
from src.training.validacion_cruzada import validar

PLAN: list[tuple[str, str]] = [
    ("cnn_v2", "normal"),
    ("cnn_v3", "normal"),
    ("cnn_v4", "normal"),
    ("cnn_v5", "normal"),
    ("cnn_v6", "normal"),
    ("cnn_v7", "normal"),
    ("cnn_v8", "normal"),
    ("cnn_v10", "normal"),
    ("mejor", "ponderada"),
]


def resolver(version: str, modo: str) -> str | None:
    """'mejor' → la versión ganadora según el criterio; cualquier otra, ella misma."""
    if version != "mejor":
        return version
    mejor = elegir_mejor(leer_historico(HISTORICO, modo))
    return None if mejor is None else mejor["version"]


def hecho(version: str, perdida: str, modo: str) -> bool:
    validas = validaciones_validas(leer_historico(HISTORICO, modo), perdida)
    return not validas.empty and version in set(validas.modelo)


def estado(modo: str) -> list[dict]:
    filas = []
    for paso, (version, perdida) in enumerate(PLAN, 1):
        real = resolver(version, modo)
        filas.append({"paso": paso, "plan": version, "version": real, "perdida": perdida,
                      "hecho": real is not None and hecho(real, perdida, modo)})
    return filas


def imprimir_estado(modo: str) -> None:
    print(f"Plan de experimentos (modo {modo}):")
    for f in estado(modo):
        etiqueta = f["plan"] if f["plan"] != "mejor" else f"mejor → {f['version'] or '¿? (se decide al final)'}"
        print(f"  {f['paso']}. {etiqueta:<32} {f['perdida']:<10} {'hecho' if f['hecho'] else 'pendiente'}")


def main() -> None:
    p = argparse.ArgumentParser(description="Lanza el plan de experimentos (reanudable).")
    p.add_argument("--modo", choices=["rapido", "completo"], default="completo")
    p.add_argument("--ver", action="store_true", help="solo mostrar el estado del plan")
    args = p.parse_args()

    imprimir_estado(args.modo)
    if args.ver:
        return
    inicio = time.perf_counter()
    for paso, (version, perdida) in enumerate(PLAN, 1):
        real = resolver(version, args.modo)
        if real is None:
            raise SystemExit(f"Paso {paso}: no se puede elegir la mejor versión todavía "
                             "(faltan validaciones cruzadas con pérdida normal).")
        if hecho(real, perdida, args.modo):
            continue
        print(f"\n{'=' * 72}\nPaso {paso} del plan: {real}, pérdida {perdida}\n{'=' * 72}")
        validar(args.modo, real, ponderada=perdida == "ponderada", nota=f"plan D17, paso {paso}")
    print(f"\nPlan terminado en {(time.perf_counter() - inicio) / 3600:.1f} h.")
    imprimir_estado(args.modo)
    mejor = elegir_mejor(leer_historico(HISTORICO, args.modo))
    if mejor:
        print(f"Mejor versión según el criterio: {mejor['version']}. Motivo: {mejor['motivo']}.")
        print("Siguiente paso: python -m src.evaluation.cerrar_modelo  (ver docs/COMO_SE_HIZO.md)")


if __name__ == "__main__":
    main()
