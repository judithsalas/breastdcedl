# Historial de arquitecturas

La historia del modelo: cada versión de la CNN que se prueba tiene aquí su **ficha**. La ficha
cuenta de qué versión sale, qué cambia y por qué, qué se esperaba ver, qué salió y qué se decidió.
Es la base de las diapositivas 3 y 4 y de las preguntas de la defensa sobre la arquitectura.

Las decisiones de proceso (entorno, tubería, datos) siguen en [`REGISTRO.md`](REGISTRO.md). Aquí
solo va lo que es de la red. Mapa completo de la documentación: D14 del registro.

## Cómo se usa

**Antes de entrenar una versión nueva:**
1. Añadirla en `src/models/versiones.py` cambiando **solo** lo que se quiere comparar.
2. Declarar en el mismo fichero su `ORIGEN` (de qué versión sale) y `ESTADO = "por entrenar"`.
3. Abrir su ficha aquí, copiando la [plantilla](#plantilla-para-una-versión-nueva), con el origen,
   el cambio, la hipótesis y **qué esperamos ver**. Se escribe antes de ver los resultados para no
   ajustar la explicación al resultado.

**Después de entrenar:**

4. Completar las ejecuciones, los resultados, la lectura y la decisión.
5. Actualizar `ESTADO` en `versiones.py`.
6. Copiar las gráficas que sirven de evidencia:
   `python -m src.models.historial --figuras runs/<ejecución>`. El comando imprime los enlaces para
   pegar en la ficha.

**Reglas:**
- Las tablas de abajo marcadas como automáticas se regeneran solas al terminar cada
  entrenamiento, o con `python -m src.models.historial`. No se editan a mano.
- Una versión ya entrenada no se modifica: si cambia algo, es otra versión. Cada ejecución guarda
  la **huella** de la definición que se entrenó. Si alguien cambia la versión después, su
  ejecución sale marcada con ⚠.
- cnn_v1 y cnn_v2 las decidió Judith. Desde el 2026-10-18 (D15) las decide Claude con su
  autorización: cada decisión está explicada desde cero en [`COMO_SE_HIZO.md`](COMO_SE_HIZO.md) y
  Judith puede revertirla. Los tests fallan si una versión no tiene `ORIGEN`, `ESTADO` o ficha.

**Estados:** `por entrenar` → `en prueba` → `descartada` o `candidata` → `final`.

**Criterio para comparar versiones** (fijado el 2026-10-18, D16; lo aplica el código:
`elegir_mejor` en `src/models/historial.py`):
1. Solo cuentan las validaciones cruzadas completas: modo `completo`, los 5 folds, pérdida
   normal y huella ✓ (la definición actual es la que se entrenó).
2. Se compara la **media del AUC por paciente en validación** de los 5 folds, no un fold suelto.
3. La mejor es la de mayor media. Pero toda versión cuya media quede a menos de una
   desviación de la mejor (la desviación entre folds de la mejor) se considera **empatada** con
   ella: con 219 pacientes por fold, el ruido de un fold es de ±0,04 y diferencias menores no
   significan nada.
4. Entre las empatadas se desempata en este orden:
   - primero las que **no sobreajustan** (diferencia de AUC entre entrenamiento y validación
     ≤ 0,15). La rúbrica pide descartar los modelos sobreentrenados; en las figuras del
     profesor, el bien entrenado tenía 0,10 y el sobreentrenado 0,38;
   - después, la **más simple** (menos parámetros);
   - si tienen los mismos parámetros, la más antigua (cambia menos cosas).
5. Si la ganadora sobreajusta (> 0,15), la siguiente versión debe atacar el sobreajuste.

**Referencias para leer los AUC:**

| Referencia | AUC |
|---|---|
| Azar | 0,50 |
| Solo la cohorte (R0, sin imagen) | 0,55 |
| Solo el subtipo HR/HER2 (R0, sin imagen) | 0,66 |
| El «médico»: edad, menopausia, HR y HER2 (R3, D25, sin imagen) | 0,69 |
| "Buen trabajo" en test según el profesor (cortes 2D, tres cohortes) | 0,60–0,70 |
| Fridman et al. en el mismo test (ViT preentrenado, otro método) | 0,72 |

## Resumen

Automático. AUC por paciente en validación de la **última** ejecución completa de cada tipo (no de
la mejor, para no escoger a posteriori). Todas las ejecuciones están en la tabla de abajo.
En 5 folds se da media ± desviación. ⚠ = la versión cambió después de esa ejecución.

<!-- auto:resumen:inicio -->
| Versión | Origen | Qué cambia | Parámetros | Estado | 1 fold · normal | 1 fold · ponderada | 5 folds · normal | 5 folds · ponderada | Train − val |
|---|---|---|---|---|---|---|---|---|---|
| [cnn_v1](#cnn_v1) | — | primera versión | 23.761 | descartada | — | — | — | — | — |
| [cnn_v2](#cnn_v2) | cnn_v1 | filtros: (16, 32, 64) → (16, 32, 64, 128) | 97.809 | por entrenar | — | — | — | — | — |
| [cnn_v3](#cnn_v3) | cnn_v2 | entrenamiento.aumento: None → geometrico | 97.809 | por entrenar | — | — | — | — | — |
| [cnn_v4](#cnn_v4) | cnn_v3 | entrada: fases → fases+realce | 98.097 | por entrenar | — | — | — | — | — |

**Según el criterio de comparación:** todavía no hay validaciones cruzadas completas con pérdida normal para comparar.
<!-- auto:resumen:fin -->

Las ejecuciones de cnn_v1 (R1, R2) se entrenaron antes de que existiera el histórico. Aparecen
aquí después de lanzar `python -m src.training.historico` en el PC de la universidad, donde
están sus carpetas. Mientras tanto, sus cifras están en la ficha.

## Genealogía

Automático: de qué versión sale cada una.

<!-- auto:genealogia:inicio -->
```
cnn_v1 · descartada
└── cnn_v2 · por entrenar
    └── cnn_v3 · por entrenar
        └── cnn_v4 · por entrenar
```
<!-- auto:genealogia:fin -->

## Ejecuciones

Automático: todas las ejecuciones en modo completo de `runs/historico.csv`. El detalle de cada
una está en `runs/<ejecución>/` (en la máquina donde se entrenó).

<!-- auto:ejecuciones:inicio -->
_Todavía no hay ejecuciones en modo completo en `runs/historico.csv`._
<!-- auto:ejecuciones:fin -->

---

## cnn_v1

**Primera red.** Origen: ninguno · Definida el 2026-10-07 · Diseñada por Judith ·
Estado: **descartada** (2026-10-07). Antes era D10, R1 y R2 del registro.

### Definición

Para poder iterar, la CNN es configurable (`src/models/cnn.py`) y cada versión es una lista de
decisiones en `src/models/versiones.py`.

| Decisión | cnn_v1 | Por qué |
|---|---|---|
| Bloques | 3 | La guía recomienda empezar con 3–4; red simple |
| Filtros | 16 → 32 → 64 | Pocos: red pequeña para pocos datos (~880 pacientes); término medio entre 8→16→32 y 32→64→128 |
| Kernel / padding / convs | 3×3, padding 1, 1 conv por bloque | Estándar y lo más simple; solo el pooling reduce el tamaño |
| Pooling | MaxPool 2×2 | Se queda con la respuesta más fuerte de cada zona |
| BatchNorm | Sí, tras cada conv (sin sesgo en la conv) | Entrenamiento más estable |
| De mapas a vector | Pooling global | 64 valores en vez de 65.536: muchos menos parámetros y menos sobreajuste |
| Cabeza | Linear(64 → 1), sin dropout | Con pooling global la cabeza ya es mínima (65 parámetros) |
| **Total** | **23.761 parámetros** | |

Tamaños: 3@256² → 16@128² → 32@64² → 64@32² → 64 → 1.
Diagrama: [`figuras/cnn_v1.svg`](figuras/cnn_v1.svg) (`python -m src.models.dibujar cnn_v1`).
Inicialización: la de PyTorch por defecto (pendiente de justificar, rúbrica punto 3).

### Hipótesis y qué esperábamos ver

Es la primera red, así que la hipótesis era que una CNN pequeña encuentra señal de realce en los
cortes. Se dejó escrita una consecuencia a vigilar: con 3 bloques cada punto del mapa final ve
~22×22 píxeles de 256×256, así que la red detecta patrones locales de realce y no la forma global
del tumor. "Si se queda corta, una versión con 4 bloques es la comparación natural."

### Ejecuciones

Fold 0 de validación: 219 pacientes, 64 con pCR. Agregación por media, umbral 0,5. Adam con
lr 1e-3, batch 32, paciencia 10, semilla 42. PC de la universidad (RX 6700 XT, ROCm 7.14).
Antes de D12, así que no se midió el AUC de entrenamiento.

| Id | Ejecución | Pérdida | Paró en | Época elegida | AUC val | Tiempo |
|---|---|---|---|---|---|---|
| R1 | `20261007-211625_cnn_v1_completo` | normal | 20 | 10 | 0,585 | 27 s/época (la primera 33 s) |
| R2 | `20261007-213119_cnn_v1_completo_ponderada` | ponderada (`pos_weight` = 2,40) | 20 | 10 | 0,572 | igual |

Gráficas: pendientes. Se generan en el PC de la uni con
`python -m src.training.graficas runs/2026100*` y se copian aquí con
`python -m src.models.historial --figuras runs/<ejecución>`.

### Resultados

**R1, pérdida normal:**
- Mejor AUC 0,585 (época 10). El resto de épocas, entre 0,51 y 0,58.
- Sensibilidad casi siempre 0 con umbral 0,5: la red predice "no pCR" para casi todas.
- Pérdida de entrenamiento de 0,603 a 0,524. La de validación sube desde la época 3 (de 0,60 a
  0,65–0,81): empieza a memorizar el entrenamiento sin generalizar.
- 0,60 es la pérdida de predecir siempre la proporción de pCR (29 %). La red empieza exactamente
  ahí y apenas aprende.

**R2, pérdida ponderada:**
- Mejor AUC 0,572 (época 10). El resto, entre 0,52 y 0,57.
- Sensibilidad entre 0,08 y 0,78 según la época, con la especificidad moviéndose al revés.
- Pérdida de entrenamiento de 0,973 a 0,833 (0,98 es la de un predictor constante con esta
  ponderación). La de validación sube desde la época 3 (de 0,97 a 1,09).

**Normal frente a ponderada** (lo que pide el enunciado):

| | Normal (R1) | Ponderada (R2) |
|---|---|---|
| Mejor AUC | 0,585 | 0,572 |
| Sensibilidad (umbral 0,5) | casi siempre 0 | 0,08–0,78, muy inestable |
| Especificidad (umbral 0,5) | casi siempre 1 | 0,26–0,94 |

- Con pérdida normal y 70 % de negativos, a la red le compensa dar probabilidades bajas a todas,
  así que con umbral 0,5 nunca predice pCR (sensibilidad 0).
- La ponderada multiplica por 2,40 el coste de fallar un positivo: las probabilidades suben, caen
  a ambos lados de 0,5 y aparece la sensibilidad.
- Pero **el AUC no cambia**: la ponderación desplaza el punto de corte, no la capacidad de
  ordenar a las pacientes. Si la red no distingue (AUC ≈ 0,55), ninguna pérdida lo arregla.
- Que la sensibilidad cambie tanto entre épocas indica que las probabilidades se agrupan cerca de
  0,5: pequeños cambios de la red mueven a muchas pacientes de un lado a otro.

### Lectura

- cnn_v1 no encuentra señal útil: su AUC está al nivel del azar.
- Con 219 pacientes, diferencias de ±0,04 son ruido, así que el "mejor" 0,585 no se distingue del
  resto de épocas.
- Comparado con R0 (añadido el 2026-10-14), está al nivel de "adivinar la cohorte" (0,55) y por
  debajo del subtipo solo (0,66).
- Nota (2026-10-14): la pérdida de entrenamiento baja mientras la de validación sube desde la
  época 3. Eso apunta más a sobreajuste que a falta de capacidad, aunque sin AUC de entrenamiento
  no se puede medir la diferencia.

### Decisión

**Descartada** con las dos pérdidas (2026-10-07). Aplica la regla de `CLAUDE.md`: si en 5–10
épocas no mejora el AUC por paciente en validación, se descarta.

### Qué aprendimos y qué lleva a la siguiente

- La pérdida ponderada no arregla una red que no ordena: cambia el umbral efectivo, no el AUC.
- Hipótesis para la siguiente versión: el campo receptivo (~22 px) es demasiado pequeño → cnn_v2.
- Hipótesis alternativa, que hay que vigilar: la red memoriza (sobreajuste) → regularizar en
  lugar de crecer.

---

## cnn_v2

**Un bloque más.** Origen: **cnn_v1** · Definida el 2026-10-07 · Elegida por Judith entre tres
opciones (4 bloques, 5 bloques, 2 convoluciones por bloque) · Estado: **por entrenar**.
Antes era D11 del registro.

### Definición

- **Único cambio respecto a cnn_v1:** un cuarto bloque de 128 filtros (16 → 32 → 64 → 128).
  Todo lo demás, igual.
- Tamaños: 3@256² → 16@128² → 32@64² → 64@32² → 128@16² → 128 → 1.
- **97.809 parámetros:** 4 veces cnn_v1, aún pequeña; la red de referencia del profesor tiene
  ~105.000.
- Diagrama: [`figuras/cnn_v2.svg`](figuras/cnn_v2.svg).

### Hipótesis

cnn_v1 no aprendía porque cada punto de su mapa final veía solo ~22×22 píxeles. Con 4 bloques el
mapa final es 16×16 y cada punto ve ~46×46 píxeles.

### Qué esperamos ver (antes de entrenar)

Escrito el 2026-10-14 a partir de D11 y fijado el 2026-10-18 (D16), antes de entrenar.
- **Si la hipótesis es cierta:** el AUC de validación sube claramente respecto a cnn_v1 (más que la
  desviación entre folds) y el de entrenamiento sube con él.
- **Si sube el AUC de entrenamiento pero no el de validación** (la diferencia crece época a época):
  es sobreajuste. El siguiente paso es regularizar (aumento de datos, dropout, weight decay), no
  crecer más.
- **Si no sube ninguno de los dos:** el problema no es el campo receptivo. Habría que revisar la
  entrada y el entrenamiento (tasa de aprendizaje, normalización, realce explícito).

### Ejecuciones

Pendientes: paso 1 del plan de experimentos (D17), validación cruzada de 5 folds con pérdida
normal (`python -m src.training.plan`). La ponderada se entrena solo con la versión que gane
(paso 4 del plan), para no gastar GPU en comparaciones que no cambian la decisión: en R1 y R2 la
ponderación movió la sensibilidad pero no el AUC.

### Resultados

Pendientes.

### Lectura

Pendiente.

### Decisión

Pendiente.

---

## cnn_v3

**cnn_v2 + aumento de datos.** Origen: **cnn_v2** · Definida el 2026-10-18 · Decidida por Claude
(D15) · Estado: **por entrenar**.

### Definición

- **Único cambio respecto a cnn_v2:** la receta de entrenamiento lleva aumento de datos
  geométrico (`entrenamiento = {"aumento": "geometrico"}`, código en `src/data/aumento.py`):
  - volteo horizontal (50 %);
  - giro de hasta ±10°;
  - desplazamiento de hasta ±4 % (unos 10 píxeles);
  - escala entre 0,95 y 1,05.
- La transformación es **idéntica para las tres fases** (se aplica al tensor completo) y solo
  en entrenamiento. No se toca la intensidad: el realce es la señal.
- No hay volteo vertical porque pondría la pared torácica donde no puede estar.
- La red es idéntica a cnn_v2: **97.809 parámetros**. Diagrama:
  [`figuras/cnn_v3.svg`](figuras/cnn_v3.svg).

### Hipótesis

cnn_v1 memorizaba: su pérdida de validación subía desde la época 3 mientras la de entrenamiento
bajaba. Con ~880 pacientes de entrenamiento, la forma más eficaz de evitarlo es que la red no vea
nunca dos veces exactamente la misma imagen. Si cada época ve el corte un poco girado, desplazado
o volteado, no puede memorizar píxeles y tiene que aprender lo que se mantiene: la forma y el
realce del tumor.

Se define ya, aunque cnn_v2 no esté entrenada, porque el plan (D17) entrena las versiones de
noche una tras otra. Tanto si cnn_v2 sobreajusta como si no, cnn_v3 responde a una pregunta útil:
si el aumento ayuda o estorba.

### Qué esperamos ver (antes de entrenar)

- **Si la hipótesis es cierta:**
  - la diferencia entre el AUC de entrenamiento y el de validación es claramente menor que en
    cnn_v2, porque la curva de entrenamiento sube más despacio;
  - el AUC de validación es igual o mejor;
  - la mejor época llega más tarde.
- **Si el AUC de validación baja respecto a cnn_v2 más que la desviación entre folds:** el
  aumento destruye información (por ejemplo, si la orientación importa). Se descarta y se prueba
  otra regularización sin aumento (dropout en la cabeza o weight decay).
- **Si no cambia nada:** la red no estaba memorizando de forma que el aumento lo corrija. El
  límite está en la señal o en cómo entra → cnn_v4 dirá si dar el realce hecho ayuda.

### Ejecuciones

Pendientes: paso 2 del plan de experimentos (D17).

### Resultados

Pendientes.

### Lectura

Pendiente.

### Decisión

Pendiente.

---

## cnn_v4

**cnn_v3 + realce explícito en la entrada.** Origen: **cnn_v3** · Definida el 2026-10-18 ·
Decidida por Claude (D15) · Estado: **por entrenar**.

### Definición

- **Único cambio respecto a cnn_v3:** `entrada = "fases+realce"`. La red recibe lo mismo (las 3
  fases en [0, 1]) y calcula **dentro** dos canales más, que pasan al primer bloque:
  - EARLY − PRE: el realce, donde entra el contraste;
  - LATE − EARLY: el lavado, negativo si el contraste se va.
- El primer bloque pasa de 432 a 720 pesos: **98.097 parámetros** en total. Diagrama:
  [`figuras/cnn_v4.svg`](figuras/cnn_v4.svg).
- Se calcula dentro de la red a propósito: la app le pasa las mismas 3 fases que el
  entrenamiento, así que no puede haber un preprocesado distinto entre los dos (el fallo que el
  profesor busca en la defensa).
- Mantiene el aumento de cnn_v3. El aumento se aplica a las fases antes de la resta, así que el
  realce sigue alineado.

### Hipótesis

La señal del problema está en la **cinética del contraste**: cuánto se ilumina el tejido y si
luego se lava (guía B2; el *washout* aparece en el 17 % de las pacientes y se asocia a lesiones
sospechosas). Una CNN que empieza con filtros al azar tiene que descubrir por sí sola que la
resta entre canales es lo importante, y con pocos datos puede no lograrlo. Dársela hecha es
conocimiento del dominio, no un modelo preentrenado, así que está permitido.

### Qué esperamos ver (antes de entrenar)

- **Si la hipótesis es cierta:** el AUC de validación sube respecto a cnn_v3 más que la
  desviación entre folds, y la mejor época llega antes, porque la señal es más fácil de
  encontrar.
- **Si queda igual:** la red ya aprendía la resta por su cuenta. Por el criterio de comparación
  se queda la más simple, cnn_v3.
- **Si baja:** los canales de diferencia añaden ruido (por ejemplo, realces negativos en cortes
  con las fases algo desalineadas). Se descarta.

### Ejecuciones

Pendientes: paso 3 del plan de experimentos (D17).

### Resultados

Pendientes.

### Lectura

Pendiente.

### Decisión

Pendiente.

---

## Modelo final

Pendiente de los resultados del plan. El **procedimiento** ya está decidido (D21 del registro;
explicado en [`COMO_SE_HIZO.md`](COMO_SE_HIZO.md)) y lo hace `python -m src.evaluation.cerrar_modelo`:
- **Versión:** la que gane con el criterio de comparación, con la pérdida (normal o ponderada) que
  dé más AUC medio en sus 5 folds.
- **Pesos:** el conjunto de los 5 modelos de su validación cruzada; se promedian sus logits.
- **Agregación:** media de los cortes.
- **Umbral:** el de mayor sensibilidad con especificidad ≥ 0,80, elegido con las predicciones
  fuera de fold.
- **Probabilidad calibrada** con Platt, ajustado también fuera de fold.

Cuando esté, aquí se anotará:
- qué versión se elige y por qué, con la comparación de la tabla de resumen;
- la configuración de entrenamiento;
- cómo se obtienen los pesos finales;
- la agregación y el umbral elegidos con las predicciones fuera de fold, con su criterio clínico;
- la ruta en `models/final/` y su checksum, que es el que mostrará la app.

---

## Plantilla para una versión nueva

Copiar desde aquí, sustituir `cnn_vN` y mantener el título de nivel 2 exactamente como el nombre
de la versión: los tests lo buscan y las tablas enlazan a él.

```markdown
## cnn_vN

**Nombre corto.** Origen: **cnn_vX** · Definida el AAAA-MM-DD · Decidida por Judith ·
Estado: **por entrenar**.

### Definición
- Único cambio respecto a cnn_vX: ...
- Tamaños: ...
- Parámetros: ... (python -m src.models.comprobar cnn_vN)
- Diagrama: figuras/cnn_vN.svg (python -m src.models.dibujar cnn_vN)

### Hipótesis
Qué problema de la versión de origen intenta resolver y por qué este cambio debería ayudar.

### Qué esperamos ver (antes de entrenar)
- Si la hipótesis es cierta: ...
- Si no lo es: ... y qué haríamos entonces.

### Ejecuciones
Nombres de las carpetas de runs/ (están en la tabla de ejecuciones) y gráficas copiadas
con python -m src.models.historial --figuras runs/<ejecución>.

### Resultados
Cifras clave: AUC val (media ± desviación), AUC train, diferencia, sensibilidad y
especificidad, normal frente a ponderada.

### Lectura
Qué significan los resultados frente a lo que esperábamos.

### Decisión
Descartada, candidata o final, con fecha y motivo (Judith).

### Qué aprendimos y qué lleva a la siguiente
```
