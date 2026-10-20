"""Las versiones de la CNN: AQUÍ se itera.

cnn_v1 y cnn_v2 fueron las primeras. Desde cnn_v3, cada versión prueba una idea
(D15); cada decisión está explicada en docs/COMO_SE_HIZO.md y en su ficha.

Cada versión es una lista de decisiones para la CNN de `src/models/cnn.py`, más una
"receta" de entrenamiento opcional (clave `entrenamiento`, ver POR_DEFECTO): una versión
es una idea completa, de la red y de cómo se entrena.

Para probar una idea nueva (pasos completos en docs/ARQUITECTURAS.md):
  1. Añadir una entrada (cnn_v5...) cambiando SOLO lo que se quiere comparar.
  2. Declarar de qué versión sale (ORIGEN) y su estado (ESTADO = "por entrenar").
  3. Abrir su ficha en docs/ARQUITECTURAS.md con la hipótesis ANTES de entrenar.
  4. Entrenar, completar la ficha y actualizar ESTADO.
No se modifican versiones ya entrenadas: si cambia algo, es otra versión (la huella
de cada ejecución en el histórico lo detecta). Los tests comprueban que toda versión
tiene ORIGEN, ESTADO y ficha.

Entrenar una versión:
    python -m src.models.comprobar cnn_v1
    python -m src.training.entrenar --modo rapido --modelo cnn_v1
"""

VERSIONES: dict[str, dict] = {
    # ---------------------------------------------------------------------- #
    # cnn_v1 · primera red (2026-10-07, Judith)
    # ---------------------------------------------------------------------- #
    "cnn_v1": {
        # 3 bloques: lo que recomienda la guía para empezar. Mapa final 32x32;
        # cada punto "ve" ~22x22 píxeles: la red detecta patrones locales de realce.
        "filtros": (16, 32, 64),
        # Kernel 3x3, estándar; padding automático para conservar el tamaño.
        "kernel": 3,
        # Una convolución por bloque: lo más simple para una primera versión.
        "convs_por_bloque": 1,
        # MaxPool: se queda con la respuesta más fuerte ("¿aparece el patrón?").
        "pooling": "max",
        # BatchNorm: entrenamiento más estable y menos sensible al learning rate.
        "batchnorm": True,
        # Pooling global: 64 valores en vez de 65.536 -> muchos menos parámetros
        # y menos sobreajuste con ~880 pacientes de entrenamiento.
        "resumen": "global",
        # Cabeza lineal directa, sin capa oculta ni dropout.
        "cabeza_oculta": None,
        "dropout": 0.0,
    },
    # ---------------------------------------------------------------------- #
    # cnn_v2 · un bloque más (2026-10-07, Judith)
    # ---------------------------------------------------------------------- #
    # cnn_v1 quedó al nivel del azar (ficha de cnn_v1 en docs/ARQUITECTURAS.md).
    # Hipótesis: con 3 bloques cada punto del mapa final ve solo ~22x22 píxeles.
    # ÚNICO cambio respecto a cnn_v1: un cuarto bloque de 128 filtros (mapa final
    # 16x16, cada punto ve ~46x46 píxeles). Todo lo demás, igual.
    "cnn_v2": {
        "filtros": (16, 32, 64, 128),
        "kernel": 3,
        "convs_por_bloque": 1,
        "pooling": "max",
        "batchnorm": True,
        "resumen": "global",
        "cabeza_oculta": None,
        "dropout": 0.0,
    },
    # ---------------------------------------------------------------------- #
    # cnn_v3 · cnn_v2 + aumento de datos (2026-10-18)
    # ---------------------------------------------------------------------- #
    # cnn_v1 memorizaba: la pérdida de validación subía desde la época 3. Con ~880
    # pacientes, la forma más eficaz de que no memorice es que no vea dos veces la
    # misma imagen. ÚNICO cambio respecto a cnn_v2: aumento geométrico (volteo
    # horizontal, giro de hasta 10°, desplazamiento y escala pequeños), igual para
    # las tres fases. La red es idéntica a cnn_v2.
    "cnn_v3": {
        "filtros": (16, 32, 64, 128),
        "kernel": 3,
        "convs_por_bloque": 1,
        "pooling": "max",
        "batchnorm": True,
        "resumen": "global",
        "cabeza_oculta": None,
        "dropout": 0.0,
        "entrenamiento": {"aumento": "geometrico"},
    },
    # ---------------------------------------------------------------------- #
    # cnn_v4 · cnn_v3 + realce explícito en la entrada (2026-10-18)
    # ---------------------------------------------------------------------- #
    # La señal del problema está en la DIFERENCIA entre fases (guía, B2). La red
    # podría aprender a restar, pero con pocos datos es más fácil dársela hecha.
    # ÚNICO cambio respecto a cnn_v3: la red añade dentro EARLY − PRE (realce) y
    # LATE − EARLY (lavado) como canales 4 y 5. Recibe lo mismo: las 3 fases.
    "cnn_v4": {
        "filtros": (16, 32, 64, 128),
        "kernel": 3,
        "convs_por_bloque": 1,
        "pooling": "max",
        "batchnorm": True,
        "resumen": "global",
        "cabeza_oculta": None,
        "dropout": 0.0,
        "entrada": "fases+realce",
        "entrenamiento": {"aumento": "geometrico"},
    },
}

# Valores que se usan cuando una versión no los declara (las antiguas no los tienen y
# así su huella no cambia).
POR_DEFECTO: dict = {
    "entrada": "fases",
    "entrenamiento": {"aumento": None},
}

# -------------------------------------------------------------------------- #
# Trazabilidad (la justificación completa está en docs/ARQUITECTURAS.md)
# -------------------------------------------------------------------------- #

# De qué versión sale cada una: su ficha explica qué cambia y por qué.
ORIGEN: dict[str, str | None] = {
    "cnn_v1": None,
    "cnn_v2": "cnn_v1",
    "cnn_v3": "cnn_v2",
    "cnn_v4": "cnn_v3",
}

# En qué punto está cada versión. Se justifica en su ficha.
ESTADOS = ("por entrenar", "en prueba", "descartada", "candidata", "final")
ESTADO: dict[str, str] = {
    "cnn_v1": "descartada",
    "cnn_v2": "por entrenar",
    "cnn_v3": "por entrenar",
    "cnn_v4": "por entrenar",
}
