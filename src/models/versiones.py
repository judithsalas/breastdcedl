"""Las versiones de la CNN que forman el modelo final.

El modelo final es un conjunto de 6 versiones (cnn_v3, cnn_v4, cnn_v5, cnn_v6, cnn_v7 y cnn_v10),
cada una entrenada en los 5 folds: 30 CNN propias, entrenadas desde cero. Sus ejecuciones están en
`experimentos/` y el modelo, en `experimentos/modelo_final/`.

Cada versión es una lista de decisiones para la CNN de `src/models/cnn.py`, más una "receta" de
entrenamiento opcional (clave `entrenamiento`, ver POR_DEFECTO): una versión es una idea completa,
de la red y de cómo se entrena. cnn_v3 es la red base y cada una de las otras cinco cambia una
sola cosa respecto a ella.

Una versión ya entrenada no se modifica: si cambia algo, es otra versión. La huella de cada
ejecución (`huella` en src/models/construir.py) lo detecta.

Comprobar y entrenar una versión:
    python -m src.models.comprobar cnn_v3
    python -m src.training.entrenar --modo rapido --modelo cnn_v3
"""

VERSIONES: dict[str, dict] = {
    # ---------------------------------------------------------------------- #
    # cnn_v3 · la red base: 4 bloques y aumento geométrico (2026-10-18)
    # ---------------------------------------------------------------------- #
    "cnn_v3": {
        # 4 bloques de 16, 32, 64 y 128 filtros. Mapa final de 16x16: cada punto
        # "ve" ~46x46 píxeles, lo bastante para un patrón de realce y su entorno.
        "filtros": (16, 32, 64, 128),
        # Kernel 3x3, estándar; padding automático para conservar el tamaño.
        "kernel": 3,
        # Una convolución por bloque: lo más simple.
        "convs_por_bloque": 1,
        # MaxPool: se queda con la respuesta más fuerte ("¿aparece el patrón?").
        "pooling": "max",
        # BatchNorm: entrenamiento más estable y menos sensible al learning rate.
        "batchnorm": True,
        # Pooling global: 128 valores en vez de 32.768 (128 x 16 x 16) -> muchos menos
        # parámetros y menos sobreajuste con ~880 pacientes de entrenamiento.
        "resumen": "global",
        # Cabeza lineal directa, sin capa oculta ni dropout.
        "cabeza_oculta": None,
        "dropout": 0.0,
        # Aumento geométrico (volteo horizontal, giro de hasta 10°, desplazamiento y
        # escala pequeños), igual para las tres fases: con tan pocas pacientes, que la
        # red no vea dos veces la misma imagen es lo más eficaz contra la memorización.
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
    # ====================================================================== #
    # cnn_v5, cnn_v6 y cnn_v7 (2026-10-21). cnn_v3 memoriza (sube el AUC de
    # entrenamiento, no el de validación) y su puntuación distingue en parte la
    # cohorte de cada paciente. Tres ideas, cada una por separado sobre cnn_v3.
    # ====================================================================== #
    # cnn_v5 · intensidad relativa a la propia paciente
    # ÚNICO cambio respecto a cnn_v3: dentro de la red, las tres fases se dividen
    # entre la intensidad media del PRE en el tejido del corte. Ataca el atajo de la
    # cohorte: cada escáner tiene su escala (auditoría: realce 0,149 en I-SPY2 y
    # ~0,11 en Duke e I-SPY1).
    "cnn_v5": {
        "filtros": (16, 32, 64, 128),
        "kernel": 3,
        "convs_por_bloque": 1,
        "pooling": "max",
        "batchnorm": True,
        "resumen": "global",
        "cabeza_oculta": None,
        "dropout": 0.0,
        "normalizacion": "pre",
        "entrenamiento": {"aumento": "geometrico"},
    },
    # cnn_v6 · mirar dónde está el tumor
    # ÚNICO cambio respecto a cnn_v3: el resumen final es un pooling por atención
    # (media ponderada de las 16x16 posiciones con pesos aprendidos) en vez de la
    # media global, que diluye un tumor pequeño entre el fondo (auditoría: el realce
    # medio de todo el tejido solo da AUC 0,54).
    "cnn_v6": {
        "filtros": (16, 32, 64, 128),
        "kernel": 3,
        "convs_por_bloque": 1,
        "pooling": "max",
        "batchnorm": True,
        "resumen": "atencion",
        "cabeza_oculta": None,
        "dropout": 0.0,
        "entrenamiento": {"aumento": "geometrico"},
    },
    # cnn_v7 · regularización
    # ÚNICO cambio respecto a cnn_v3 (en la idea; son dos piezas de la misma idea):
    # dropout 0,3 antes de la capa final y AdamW con weight decay 0,05. Las dos
    # herramientas estándar contra la memorización que la red aún no tenía.
    "cnn_v7": {
        "filtros": (16, 32, 64, 128),
        "kernel": 3,
        "convs_por_bloque": 1,
        "pooling": "max",
        "batchnorm": True,
        "resumen": "global",
        "cabeza_oculta": None,
        "dropout": 0.3,
        "entrenamiento": {"aumento": "geometrico", "optimizador": "adamw", "weight_decay": 0.05},
    },
    # ---------------------------------------------------------------------- #
    # cnn_v10 · quitar la pista de la orientación (2026-10-21)
    # ---------------------------------------------------------------------- #
    # ÚNICO cambio respecto a cnn_v3: el aumento añade giros de 0°, 90°, 180° o 270°
    # al azar. Cada cohorte orienta la mama a su manera (en I-SPY1 apunta hacia
    # abajo; en I-SPY2 y Duke, a los lados) y la puntuación de cnn_v3 reconoce I-SPY1
    # con AUC 0,28 (0,5 = no la reconoce). La red es idéntica a cnn_v3.
    "cnn_v10": {
        "filtros": (16, 32, 64, 128),
        "kernel": 3,
        "convs_por_bloque": 1,
        "pooling": "max",
        "batchnorm": True,
        "resumen": "global",
        "cabeza_oculta": None,
        "dropout": 0.0,
        "entrenamiento": {"aumento": "geometrico+giros90"},
    },
}

# Valores que se usan cuando una versión no los declara (así la huella de cada versión no
# depende de opciones que no usa).
POR_DEFECTO: dict = {
    "entrada": "fases",
    "normalizacion": None,
    # Adam sin weight decay, el learning rate del modo y todos los cortes en cada época.
    "entrenamiento": {"aumento": None, "optimizador": "adam", "weight_decay": 0.0, "learning_rate": None},
}

# -------------------------------------------------------------------------- #
# Trazabilidad
# -------------------------------------------------------------------------- #

# De qué versión sale cada una: cnn_v3 es la base y las demás cambian una cosa respecto a ella.
ORIGEN: dict[str, str | None] = {
    "cnn_v3": None,
    "cnn_v4": "cnn_v3",
    "cnn_v5": "cnn_v3",
    "cnn_v6": "cnn_v3",
    "cnn_v7": "cnn_v3",
    "cnn_v10": "cnn_v3",
}

# En qué punto está cada versión. Las 6 están en el modelo final: son las versiones con validación
# cruzada completa que no sobreajustan (diferencia de AUC train − val ≤ 0,15), una regla fijada
# antes de mirar cuánto da el conjunto.
ESTADOS = ("por entrenar", "en prueba", "descartada", "candidata", "final")
ESTADO: dict[str, str] = {
    "cnn_v3": "final",
    "cnn_v4": "final",
    "cnn_v5": "final",
    "cnn_v6": "final",
    "cnn_v7": "final",
    "cnn_v10": "final",
}

# Nombre corto de cada versión, para figuras y la app: qué la distingue, en pocas palabras.
NOMBRES: dict[str, str] = {
    "cnn_v3": "aumento geométrico",
    "cnn_v4": "con el realce dado",
    "cnn_v5": "intensidad relativa al PRE",
    "cnn_v6": "pooling por atención",
    "cnn_v7": "dropout y weight decay",
    "cnn_v10": "giros de 90°",
}
