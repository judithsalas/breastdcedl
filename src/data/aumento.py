"""Aumento de datos: pequeñas transformaciones aleatorias de cada corte durante el entrenamiento.

Por qué: con ~880 pacientes de entrenamiento la red acaba memorizando las imágenes (cnn_v1:
la pérdida de validación subía desde la época 3). Si en cada época ve el corte un poco girado,
desplazado o volteado, no puede memorizar píxeles concretos y tiene que aprender patrones que
se mantienen (la forma y el realce del tumor).

Reglas que cumple (guía, B5 y D2):
- La MISMA transformación para las tres fases: se aplica al tensor completo (3, H, W). Si se
  girase una fase y no las otras, EARLY − PRE dejaría de ser el realce.
- Solo geometría. Nada de brillo, contraste ni color: la intensidad relativa entre fases ES la
  señal (el realce), y los canales no son colores.
- Solo en entrenamiento. Validación, test y app ven la imagen tal cual.

Qué transformaciones y por qué estas:
- Volteo horizontal (50 %): una mama izquierda volteada parece una derecha; es anatomía real.
- Giro de hasta ±10°: el posicionamiento de la paciente varía un poco.
- Desplazamiento de hasta ±4 % (unos 10 píxeles): el recorte no siempre centra igual el tumor.
- Escala entre 0,95 y 1,05: tamaños de mama y resolución algo distintos entre centros.
No se usa volteo vertical: pondría la pared torácica donde no puede estar.
"""

from __future__ import annotations

import torch
from torchvision.transforms import InterpolationMode, v2

AUMENTOS = ("geometrico",)


class AumentoGeometrico:
    """Transformación geométrica aleatoria, idéntica para los 3 canales de un corte."""

    def __init__(self, grados: float = 10.0, desplazamiento: float = 0.04,
                 escala: tuple[float, float] = (0.95, 1.05), p_volteo: float = 0.5) -> None:
        self.parametros = {"grados": grados, "desplazamiento": desplazamiento, "escala": list(escala),
                           "p_volteo": p_volteo}
        self.transformacion = v2.Compose([
            v2.RandomHorizontalFlip(p=p_volteo),
            v2.RandomAffine(degrees=grados, translate=(desplazamiento, desplazamiento), scale=escala,
                            interpolation=InterpolationMode.BILINEAR, fill=0.0),
        ])

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[0] != 3:
            raise ValueError(f"Se espera un corte (3, H, W) y llega {tuple(x.shape)}")
        return self.transformacion(x).clamp(0.0, 1.0)

    def __repr__(self) -> str:
        return f"AumentoGeometrico({self.parametros})"


def crear_aumento(nombre: str | None):
    """El aumento que pide la receta de una versión (None = sin aumento)."""
    if nombre is None:
        return None
    if nombre == "geometrico":
        return AumentoGeometrico()
    raise ValueError(f"Aumento desconocido: {nombre!r}. Opciones: {AUMENTOS}")
