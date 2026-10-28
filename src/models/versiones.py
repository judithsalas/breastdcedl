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
    # ====================================================================== #
    # Segunda ronda (2026-10-21, D28). Resultados de v2–v4: las tres en
    # ~0,57–0,59; la red memoriza (sube el AUC de entrenamiento, no el de
    # validación) y su puntuación distingue I-SPY2 del resto (AUC 0,58–0,66): en
    # parte reconoce el hospital. Tres ideas, cada una por separado sobre cnn_v3
    # (la candidata), y las tres juntas en cnn_v8.
    # ====================================================================== #
    # cnn_v5 · intensidad relativa a la propia paciente
    # ÚNICO cambio respecto a cnn_v3: dentro de la red, las tres fases se dividen
    # entre la intensidad media del PRE en el tejido del corte. Ataca el atajo del
    # hospital: cada escáner tiene su escala (auditoría: realce 0,149 en I-SPY2 y
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
    # cnn_v8 · las tres ideas juntas
    # cnn_v3 + normalización por el PRE + pooling por atención + regularización. Si
    # cada idea ayuda por separado, juntas deberían sumar; si no, las versiones
    # 5–7 dicen cuál sobra.
    "cnn_v8": {
        "filtros": (16, 32, 64, 128),
        "kernel": 3,
        "convs_por_bloque": 1,
        "pooling": "max",
        "batchnorm": True,
        "resumen": "atencion",
        "cabeza_oculta": None,
        "dropout": 0.3,
        "normalizacion": "pre",
        "entrenamiento": {"aumento": "geometrico", "optimizador": "adamw", "weight_decay": 0.05},
    },
    # ====================================================================== #
    # Tercera ronda (2026-10-21, D29). v5–v7 no mejoran a cnn_v3: la red no
    # mira el tumor (dentro de cada hospital, AUC 0,51–0,58) y reconoce el hospital.
    # cnn_v9 queda reservada para la atención guiada por la caja del tumor, que
    # depende de que el profesor permita usar sraw/eraw/scol/ecol al entrenar.
    # ====================================================================== #
    # cnn_v10 · quitar la pista de la orientación
    # ÚNICO cambio respecto a cnn_v3: el aumento añade giros de 0°, 90°, 180° o 270°
    # al azar. Cada hospital orienta la mama a su manera (en I-SPY1 apunta hacia
    # abajo; en I-SPY2 y Duke, a los lados) y la puntuación de cnn_v3 reconoce I-SPY1
    # con AUC 0,28 (0,5 = no lo reconoce). La red es idéntica a cnn_v3.
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
    # ====================================================================== #
    # Cuarta ronda (2026-10-21, D33): centrarse en el tumor.
    # ====================================================================== #
    # cnn_v11 · recorte alrededor del tumor
    # ÚNICO cambio respecto a cnn_v3: la red ve un cuadrado de 128×128 centrado en la caja
    # del tumor (patients.csv), sin cambiar la escala, en vez del corte entero de 256×256.
    # NO es desplegable: la app no tiene la caja. Mide cuánto se gana mirando solo el tumor.
    # Descartada sin entrenar (D34): la regla de la fase A no permite usar la caja del tumor
    # ni como entrada ni como supervisión. cnn_v9 (atención guiada por la caja), igual.
    "cnn_v11": {
        "filtros": (16, 32, 64, 128),
        "kernel": 3,
        "convs_por_bloque": 1,
        "pooling": "max",
        "batchnorm": True,
        "resumen": "global",
        "cabeza_oculta": None,
        "dropout": 0.0,
        "entrenamiento": {"aumento": "geometrico", "recorte": "caja"},
    },
    # ====================================================================== #
    # Quinta ronda (2026-10-28, D34). Solo imágenes y pCR (fase A): ni datos clínicos, ni
    # hospital, ni caja del tumor. Cada idea sobre cnn_v3, y se prueba primero en el fold 0.
    # ====================================================================== #
    # cnn_v12 · ver el tumor entero
    # ÚNICO cambio respecto a cnn_v3: dos convoluciones por bloque. El campo receptivo pasa de
    # 46 a 76 px: un tumor de I-SPY mide ~70 px y la red no lo ve entero (su puntuación no
    # tiene relación con el tamaño del tumor, R5).
    "cnn_v12": {
        "filtros": (16, 32, 64, 128),
        "kernel": 3,
        "convs_por_bloque": 2,
        "pooling": "max",
        "batchnorm": True,
        "resumen": "global",
        "cabeza_oculta": None,
        "dropout": 0.0,
        "entrenamiento": {"aumento": "geometrico"},
    },
    # cnn_v13 · no diluir la respuesta del tumor
    # ÚNICO cambio respecto a cnn_v12: el resumen une la media y el máximo de cada canal. El
    # máximo conserva la respuesta más fuerte, que en un tumor pequeño la media diluye.
    "cnn_v13": {
        "filtros": (16, 32, 64, 128),
        "kernel": 3,
        "convs_por_bloque": 2,
        "pooling": "max",
        "batchnorm": True,
        "resumen": "media+max",
        "cabeza_oculta": None,
        "dropout": 0.0,
        "entrenamiento": {"aumento": "geometrico"},
    },
    # cnn_v14 · que el escáner no sirva de pista
    # ÚNICO cambio respecto a cnn_v3: aumento "geometrico+escaner" (zoom 0,8–1,2, el mismo
    # brillo en las 3 fases y un poco de ruido). Imita lo que cambia de un hospital a otro.
    "cnn_v14": {
        "filtros": (16, 32, 64, 128),
        "kernel": 3,
        "convs_por_bloque": 1,
        "pooling": "max",
        "batchnorm": True,
        "resumen": "global",
        "cabeza_oculta": None,
        "dropout": 0.0,
        "entrenamiento": {"aumento": "geometrico+escaner"},
    },
    # cnn_v15 · no memorizar pacientes
    # ÚNICO cambio respecto a cnn_v3: cada época, 2 cortes al azar de cada paciente en vez de
    # los 10. Todas las pacientes pesan igual y cada época ve otros cortes.
    "cnn_v15": {
        "filtros": (16, 32, 64, 128),
        "kernel": 3,
        "convs_por_bloque": 1,
        "pooling": "max",
        "batchnorm": True,
        "resumen": "global",
        "cabeza_oculta": None,
        "dropout": 0.0,
        "entrenamiento": {"aumento": "geometrico", "cortes_por_paciente": 2},
    },
    # cnn_v16 · bajar el paso cuando se estanca
    # ÚNICO cambio respecto a cnn_v3: programador "meseta" (el learning rate se reduce a la
    # mitad si el AUC de validación no mejora en 3 épocas). La mejor época de cnn_v3 llega muy
    # pronto (1, 6, 5, 16 y 6) y luego la validación cae.
    "cnn_v16": {
        "filtros": (16, 32, 64, 128),
        "kernel": 3,
        "convs_por_bloque": 1,
        "pooling": "max",
        "batchnorm": True,
        "resumen": "global",
        "cabeza_oculta": None,
        "dropout": 0.0,
        "entrenamiento": {"aumento": "geometrico", "programador": "meseta"},
    },
    # ====================================================================== #
    # Sexta ronda (2026-10-28, D35). Misma regla de la fase A y mismo cribado en el fold 0.
    # ====================================================================== #
    # cnn_v17 · trabajar a 128×128
    # ÚNICO cambio respecto a cnn_v3: la red reduce la imagen a la mitad al entrar (media de 2×2
    # píxeles). Recibe lo mismo (256×256), pero cada neurona del final ve 92 px de la imagen
    # original en vez de 46, hay menos detalle fino que memorizar y va unas 4 veces más rápida.
    "cnn_v17": {
        "filtros": (16, 32, 64, 128),
        "kernel": 3,
        "convs_por_bloque": 1,
        "pooling": "max",
        "batchnorm": True,
        "resumen": "global",
        "cabeza_oculta": None,
        "dropout": 0.0,
        "reduccion": 2,
        "entrenamiento": {"aumento": "geometrico"},
    },
    # cnn_v18 · mapas cinéticos
    # ÚNICO cambio respecto a cnn_v3: además de las 3 fases, la red recibe el realce relativo al
    # PRE y el SER (lavado) de cada píxel, calculados dentro. Son divisiones, que la red no sabe
    # hacer sola, y son lo que mira la radiología (curvas de realce, volumen funcional de I-SPY).
    "cnn_v18": {
        "filtros": (16, 32, 64, 128),
        "kernel": 3,
        "convs_por_bloque": 1,
        "pooling": "max",
        "batchnorm": True,
        "resumen": "global",
        "cabeza_oculta": None,
        "dropout": 0.0,
        "entrada": "fases+cinetica",
        "entrenamiento": {"aumento": "geometrico"},
    },
}

# Valores que se usan cuando una versión no los declara (las antiguas no los tienen y
# así su huella no cambia).
POR_DEFECTO: dict = {
    "entrada": "fases",
    "normalizacion": None,
    "reduccion": 1,          # D35: 1 = la red trabaja a 256×256
    # Adam sin weight decay es lo que usaron cnn_v1 a cnn_v4.
    "entrenamiento": {"aumento": None, "optimizador": "adam", "weight_decay": 0.0, "recorte": None,
                      # D34: None = el learning rate del modo, todos los cortes cada época, sin programador
                      "learning_rate": None, "cortes_por_paciente": None, "programador": None},
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
    "cnn_v5": "cnn_v3",
    "cnn_v6": "cnn_v3",
    "cnn_v7": "cnn_v3",
    "cnn_v8": "cnn_v3",
    "cnn_v10": "cnn_v3",
    "cnn_v11": "cnn_v3",
    "cnn_v12": "cnn_v3",
    "cnn_v13": "cnn_v12",
    "cnn_v14": "cnn_v3",
    "cnn_v15": "cnn_v3",
    "cnn_v16": "cnn_v3",
    "cnn_v17": "cnn_v3",
    "cnn_v18": "cnn_v3",
}

# En qué punto está cada versión. Se justifica en su ficha.
ESTADOS = ("por entrenar", "en prueba", "descartada", "candidata", "final")
ESTADO: dict[str, str] = {
    "cnn_v1": "descartada",
    "cnn_v2": "descartada",     # empata con cnn_v3 pero sobreajusta (diferencia 0,162 > 0,15)
    "cnn_v3": "final",          # en el conjunto final (D36); la mejor según el criterio tras la primera ronda (2026-10-21)
    "cnn_v4": "final",          # en el conjunto final (D36); el realce explícito no mejora (0,572 ± 0,044)
    "cnn_v5": "final",          # en el conjunto final (D36); no mejora (0,574 ± 0,035) ni quita el atajo del hospital
    "cnn_v6": "final",          # en el conjunto final (D36); empata (0,591 ± 0,041) pero tiene más parámetros que cnn_v3
    "cnn_v7": "final",          # en el conjunto final (D36); empata (0,588 ± 0,042); no baja la diferencia train − val
    "cnn_v8": "descartada",     # empata (0,593 ± 0,025) pero sobreajusta (0,169 > 0,15)
    "cnn_v10": "final",         # en el conjunto final (D36); empata (0,581 ± 0,045); delata menos el hospital y memoriza menos
    "cnn_v11": "descartada",    # sin entrenar: usa la caja del tumor (regla de la fase A, D34)
    "cnn_v12": "descartada",    # fold 0: 0,579 < 0,61 (cribado D34); memoriza más rápido
    "cnn_v13": "por entrenar",
    "cnn_v14": "por entrenar",
    "cnn_v15": "por entrenar",
    "cnn_v16": "por entrenar",
    "cnn_v17": "por entrenar",
    "cnn_v18": "por entrenar",
}

# Nombre corto de cada versión, para figuras y la app: qué la distingue, en pocas palabras.
NOMBRES: dict[str, str] = {
    "cnn_v1": "primera red (3 bloques)",
    "cnn_v2": "4 bloques",
    "cnn_v3": "aumento geométrico",
    "cnn_v4": "con el realce dado",
    "cnn_v5": "intensidad relativa al PRE",
    "cnn_v6": "pooling por atención",
    "cnn_v7": "dropout y weight decay",
    "cnn_v8": "v5 + v6 + v7 juntas",
    "cnn_v10": "giros de 90°",
    "cnn_v11": "recorte con la caja del tumor",
    "cnn_v12": "dos convoluciones por bloque",
    "cnn_v13": "media + máximo",
    "cnn_v14": "aumento de escáner",
    "cnn_v15": "2 cortes por paciente y época",
    "cnn_v16": "learning rate que baja",
    "cnn_v17": "trabaja a 128×128",
    "cnn_v18": "mapas cinéticos",
}
