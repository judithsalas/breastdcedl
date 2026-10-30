"""Comprueba una arquitectura sin entrenar: forma de cada capa y parámetros.

    python -m src.models.comprobar cnn_v3
"""

from __future__ import annotations

import sys

import torch
from torch import nn

from src.models.construir import contar_parametros, crear_modelo
from src.models.versiones import VERSIONES


def resumen(modelo: nn.Module, tamano_lote: int = 2) -> torch.Tensor:
    """Pasa un lote aleatorio por la red e imprime la salida de cada capa."""
    filas = []

    def apuntar(nombre):
        def gancho(_modulo, _entrada, salida):
            if isinstance(salida, torch.Tensor):
                n = sum(p.numel() for p in _modulo.parameters(recurse=False))
                filas.append((nombre, type(_modulo).__name__, tuple(salida.shape), n))
        return gancho

    ganchos = [m.register_forward_hook(apuntar(n)) for n, m in modelo.named_modules()
               if n and not list(m.children())]
    modelo.eval()
    with torch.no_grad():
        salida = modelo(torch.rand(tamano_lote, 3, 256, 256))
    for g in ganchos:
        g.remove()

    print(f"{'capa':<28}{'tipo':<20}{'salida':<24}{'parámetros':>12}")
    print("-" * 84)
    print(f"{'entrada':<28}{'':<20}{(tamano_lote, 3, 256, 256)!s:<24}{'':>12}")
    for nombre, tipo, forma, n in filas:
        print(f"{nombre:<28}{tipo:<20}{forma!s:<24}{n:>12,}")
    print("-" * 84)
    print(f"Parámetros entrenables: {contar_parametros(modelo):,}")
    return salida


def main() -> None:
    nombre = sys.argv[1] if len(sys.argv) > 1 else "cnn_v3"
    print(f"Versiones disponibles: {sorted(VERSIONES)}\n")
    modelo = crear_modelo(nombre)
    salida = resumen(modelo)
    if tuple(salida.shape) != (2, 1):
        sys.exit(f"\nERROR: la salida debe ser (B, 1) y es {tuple(salida.shape)}")
    print("\nOK: entrada (B, 3, 256, 256) -> salida (B, 1)")


if __name__ == "__main__":
    main()
