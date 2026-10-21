# Cómo se hizo: las decisiones del proyecto, explicadas desde cero

Este documento sirve para **entender y poder defender** cada decisión del proyecto. Para cada una
cuenta qué se decidió, por qué (explicado desde cero), qué alternativas había y qué contestar si
lo preguntan en la defensa.

**Quién decidió qué.**
- Del 7 al 15 de octubre, las decisiones las tomó Judith con ayuda de Claude: cnn_v1 y cnn_v2,
  el entorno, los modos rápido y completo y la tubería.
- Desde el 18 de octubre (D15), a petición de Judith, las decisiones técnicas las toma Claude y
  quedan explicadas aquí. Judith puede cambiar cualquiera; si lo hace, se actualiza este
  documento.

**Cómo leerlo.**
- Cada sección enlaza con el código y con su número de decisión (D…) en
  [`REGISTRO.md`](REGISTRO.md).
- La historia de cada red está en [`ARQUITECTURAS.md`](ARQUITECTURAS.md).
- Las cajas **«Si te preguntan»** son respuestas cortas para la defensa.

**Dónde está cada cosa (D26).** Lo que se lee está a la vista y lo que genera el código va junto
y en orden:
- `docs/`: los tres documentos para leer. Este (las decisiones), [`REGISTRO.md`](REGISTRO.md) (en
  qué punto estamos) y [`ARQUITECTURAS.md`](ARQUITECTURAS.md) (la historia de la red).
- `docs/resultados/`: lo que sale de los scripts, una carpeta por paso y en orden. Son
  `1_auditoria`, `2_cierre`, `3_clinico` y `4_test`. No se edita a mano: si algo cambia, se
  vuelve a ejecutar el script.
- `docs/figuras/`: diagramas de las redes y gráficas copiadas de los entrenamientos.
- `src/`: el código, por pasos. `data` revisa los datos, `models` tiene las redes, `training`
  entrena y `evaluation` cierra el modelo, compara con el médico y evalúa test.
  - Los `__init__.py` están vacíos: solo le dicen a Python que la carpeta es un paquete.
  - Lo común tiene su propio fichero: las rutas y el acceso a los datos del profesor en
    `src/data/caso.py`, y crear la red de una versión en `src/models/construir.py`.
- `runs/`: cada entrenamiento, completo. Solo vive en el ordenador donde se entrenó; a GitHub va
  `runs/historico.csv`.

> **Si te preguntan** «¿cómo está organizado el repositorio?»: el código va por pasos en `src/`,
> los resultados de cada paso se generan solos en `docs/resultados/` y las decisiones están en
> tres documentos. Así cualquier cifra del informe se puede seguir hasta el script que la produjo.

**Índice**
1. [El problema en una página](#1-el-problema-en-una-página)
2. [Datos: la paciente es la unidad](#2-datos-la-paciente-es-la-unidad)
3. [La red, versión a versión](#3-la-red-versión-a-versión)
4. [Entrenar y comparar](#4-entrenar-y-comparar)
5. [Cerrar el modelo](#5-cerrar-el-modelo)
6. [Test: una sola vez](#6-test-una-sola-vez)
7. [La app](#7-la-app)
8. [Las preguntas del profesor, respondidas](#8-las-preguntas-del-profesor-respondidas)
9. [Qué queda y en qué orden](#9-qué-queda-y-en-qué-orden)

---

## 1. El problema en una página

- **El contexto.** A algunas pacientes con cáncer de mama se les da quimioterapia *antes* de
  operarlas. Tras la cirugía, un patólogo mira el tejido. Si no queda cáncer invasivo es una
  **pCR** (respuesta patológica completa, y = 1); si queda algo, no pCR (y = 0).
- **Lo que hacemos.** Con una resonancia hecha **antes** de la quimio, estimar la probabilidad de
  pCR. Es predecir cómo responderá a un tratamiento que todavía no ha empezado, y por eso los
  resultados esperables son modestos. El profesor da estas referencias:
  - un AUC de 0,60–0,70 en test es un buen trabajo;
  - lo publicado con imagen pretratamiento está entre 0,70 y 0,80.
- **Los datos.** Cada corte tiene tres fases de una DCE-MRI: PRE (antes del contraste), EARLY
  (poco después) y LATE (más tarde). El contraste entra donde hay vasos, y los tumores tienen
  muchos. La señal está en la **diferencia** entre fases: el realce EARLY − PRE y el lavado
  LATE − EARLY.
- **El pipeline completo:**

```
3 PNG de un corte ─► tensor 3×256×256 en [0,1] ─► CNN ─► logit ─► (conjunto de 5 modelos,
calibración) ─► probabilidad del corte ─► media de los ~10 cortes de la paciente ─► umbral ─► clase
```

---

## 2. Datos: la paciente es la unidad

### 2.1 Separar por paciente, no por corte (regla del caso)

- **Qué:** se usa la columna `fold` de `samples.csv`, que reparte a las 1.097 pacientes de train
  en 5 grupos con la misma proporción de pCR. Una paciente está en un solo grupo.
- **Por qué:** los ~10 cortes de una paciente se parecen mucho. Si unos se usan para entrenar y
  otros para validar, la red "reconoce" a la paciente y la validación sale inflada sin significar
  nada. Es una **fuga de datos** e invalida el trabajo. `uc.particion` además comprueba que no
  haya solape, y un test lo vuelve a comprobar en GitHub.
- **Alternativa descartada:** repartir los 10.945 cortes al azar (prohibido).

> **Si te preguntan** «¿cómo evitas la fuga?»: la unidad es la paciente. Uso los folds ya
> calculados por paciente y estratificados por pCR; ninguna paciente está en dos conjuntos, y lo
> comprueban el código y un test automático.

### 2.2 Normalización: solo dividir entre 255 (D8)

- **Qué:** cada PNG (gris, 8 bits) se divide entre 255 → valores en [0, 1]. No se hace nada más.
- **Por qué:** los autores del dataset ya normalizaron las **tres fases con la misma ventana**
  (percentiles 1–99 calculados juntos). Gracias a eso EARLY − PRE es un realce de verdad. Si
  normalizáramos cada fase por separado, destruiríamos esa comparación.
- **Las otras reglas:** los tres canales no son colores, así que nada de normalización de
  ImageNet ni cambios de color.
- **En la app:** la normalización es la misma, y un test comprueba que la app lee exactamente
  igual que el entrenamiento (`tests/test_entrada.py`).

> **Si te preguntan** «¿normalizas igual en la app?»: sí, y está comprobado. La app y el
> entrenamiento dividen entre 255. Un test compara valor a valor la lectura de la app con la de
> `utils_caso`, y en los ejemplos de la app se ve que la probabilidad coincide con la del pipeline.

### 2.3 Auditoría de los datos (D20, rúbrica punto 1)

`python -m src.data.auditoria` genera [`AUDITORIA.md`](resultados/1_auditoria/AUDITORIA.md). Mira estas cosas:

1. **Recuentos:** pacientes y cortes por conjunto, clase, cohorte, subtipo y fold. Sirve para
   describir el desbalance (30 % de pCR) y comprobar que los folds se parecen.
2. **Integridad:** que todos los PNG existan, se abran y sean gris de 8 bits de 256×256.
3. **Orden de las fases:** en el tejido debe cumplirse PRE < EARLY (el contraste ilumina). Si no
   se cumpliera, estaríamos leyendo las fases cambiadas.
4. **Lavado:** el porcentaje de pacientes con LATE < EARLY. La guía dice que el 17 %; no es un
   error, es el *washout*.
5. **Intensidades por cohorte:** si un hospital tiene imágenes más brillantes o realces distintos,
   la red podría aprender a reconocer el hospital en lugar del tumor.
6. **Realce por clase:** el AUC de dos indicadores hechos a mano, sin entrenar nada (realce medio
   y fracción del tejido que realza). Si ya separan algo, hay señal en la imagen y la CNN debe
   superarlos.
7. **Ejemplos:** una paciente con pCR y otra sin pCR de cada cohorte, con sus tres fases y el
   mapa de realce.
8. **Orientación** (añadida el 21 de octubre, D29): el corte central de 10 pacientes al azar por
   cohorte. Sirvió para ver que cada hospital orienta la mama a su manera (§3.10).

Todo lo que mezcla imagen y etiqueta usa **solo train**. De test solo se cuenta cuántas pacientes
hay.

### 2.4 Referencias sin imagen (R0)

- **Qué:** el AUC de un "modelo" que solo conoce la cohorte (≈ 0,55) o el subtipo HR/HER2
  (≈ 0,66), validado por folds.
- **Por qué:** es el listón con el que compararse.
  - Si la CNN no supera "solo cohorte", puede estar aprendiendo el hospital.
  - Si no supera "solo subtipo", la imagen aporta menos que un dato clínico que ya se conoce
    antes del tratamiento.
- **Límite:** el profesor dice que estas variables sirven para comparar, no como modelo. Por eso
  no entran en la red ni en la app. R0 se amplía en §2.5 con un "médico" que usa varias
  variables a la vez.

### 2.5 El modelo del médico, sin imagen (D25)

**La pregunta (de Judith, 18 de octubre).** ¿La red se entrena solo con las fotos o también con los
datos clínicos? La idea era tener un modelo que hiciera de médico con lo que sabe (edad,
menopausia, HR/HER2, análisis de sangre…), solo para comparar.

**Qué se decidió.**
- **La CNN se entrena solo con las imágenes**, y la app solo recibe imágenes. Lo dice el
  enunciado (§7: la entrada son las tres fases) y lo repite el correo del profesor: las variables
  de `patients.csv` sirven *"de comparación, no de modelo principal"*.
- **El "médico" es un modelo aparte**, sin imagen. Mira lo que se sabe antes de la quimioterapia
  por la historia clínica y la biopsia: **edad, menopausia, receptores hormonales (HR) y HER2**.
  Código: `src/evaluation/modelo_clinico.py`; informe:
  [`MODELO_CLINICO.md`](resultados/3_clinico/MODELO_CLINICO.md).
- **No hay análisis de sangre**: el dataset no los trae, ni tampoco peso, IMC, tabaco o alcohol
  (lo dice la guía). Ese médico no se puede hacer con estos datos.

**Qué es, desde cero.** Una **regresión logística**: suma las variables, cada una con su peso, y
convierte la suma en una probabilidad entre 0 y 1. Es el modelo más sencillo y más interpretable
para una respuesta sí/no. Cada peso se lee como una **razón de odds**: por cuánto se multiplica la
"apuesta" a favor de la pCR al cambiar la variable. Por ejemplo, 2 significa el doble y 0,5 la
mitad.

**Por qué este modelo y no otro.**
- Con cuatro variables y ~1.100 pacientes no hace falta nada más complejo. Un modelo más flexible
  (un bosque aleatorio, una red) apenas ganaría y se entendería peor.
- Las variables se eligieron **antes de mirar resultados**, por criterio clínico: lo que sabe el
  médico. No se probaron combinaciones para quedarse con la mejor. Quitar la menopausia daba 0,005
  más de AUC, muy por debajo de la desviación (0,024), y eso es ruido.

**Qué variables NO se usan, y por qué.**

| Columnas | Por qué no |
|---|---|
| raza (`race_white`, `race_black`) | la guía lo dice: tratarla como causa es un error metodológico |
| cohorte (`dataset`) | es del dataset, no de la paciente; R0 ya mide cuánto acierta sola |
| adquisición (`n_z`, `slice_thick`, …) | son del escáner, no de la paciente |
| máscara (`mask_*`, `sraw`, …) | salen de la imagen |
| volumen tumoral (`tum_vol`) | sale de la segmentación de la imagen (la guía lo pone con la máscara, no con lo clínico) y en Duke tiene otra escala (mediana 6 frente a 15 en I-SPY), señal de que no se mide igual. Se prueba solo como variante aparte, sin llevarla a test |
| `HR_HER2_STATUS`, `TripleNeg`… | son combinaciones de HR y HER2, que ya están |

**Los datos que faltan.**
- La **menopausia falta en todas las pacientes de I-SPY1** (el 100 %).
  - Si se rellenara con un valor fijo o con un indicador de "falta el dato", el modelo podría
    reconocer I-SPY1 por ese hueco: aprendería la cohorte, no la paciente.
  - Por eso se rellena como lo haría un médico que no tiene el dato: con la probabilidad de estar
    en la menopausia **según la edad**. Se aprende en train: la media de edad es 42 años sin
    menopausia y 58 con ella.
- **HR y HER2** faltan en el 2–3 % de I-SPY1, y la **edad** en el 0,4 % de I-SPY2. Se rellenan
  con la mediana.
- **Todo lo que se rellena se aprende solo con los folds de entrenamiento:** la paciente de
  validación no influye ni en su propia imputación.

**Cómo se valida.**
- Con **los mismos 5 folds que la CNN**: cada paciente la predice el modelo que no la vio. Así la
  comparación es justa.
- Umbral con **el mismo criterio que la CNN**: la mayor sensibilidad con especificidad ≥ 0,80.

**Resultados en train, fuera de fold (datos reales, 1.097 pacientes).**

| Modelo | AUC (5 folds) |
|---|---|
| Solo la cohorte (R0) | 0,549 ± 0,027 |
| Solo el subtipo HR/HER2 (R0) | 0,657 ± 0,028 |
| **Médico: edad, menopausia, HR, HER2** | **0,689 ± 0,024** |
| Médico + volumen tumoral (variante) | 0,711 ± 0,030 |

- **Lo que más pesa es la biopsia.**
  - HER2+ multiplica la odds de pCR por 2,7 (IC 2,1–3,8).
  - HR+ la reduce a 0,30 (IC 0,22–0,38): los tumores hormonodependientes responden peor a la
    quimioterapia.
  - Edad y menopausia apenas pesan una vez sabido el subtipo: sus intervalos cruzan el 1.
- **Por cohorte:** 0,80 en I-SPY1, 0,72 en Duke y 0,66 en I-SPY2.
- **Diferencia train–validación de 0,01:** un modelo tan simple no sobreajusta.

**El listón que tiene que superar la CNN es alto.** 0,69 está en la franja que el profesor
considera "buen trabajo" para la CNN (0,60–0,70). Puede que la CNN no lo supere sola, y eso sería
un resultado honesto y defendible. En la bibliografía la clínica también pesa mucho: en Joo et
al. 2021 daba 0,83 sola y la imagen la subía a 0,89.

**Comparación con la CNN** (se completa cuando la CNN esté cerrada). Con la CNN se hacen tres cosas:
1. **CNN contra el médico:** diferencia de AUC en las mismas pacientes, con intervalo por
   **bootstrap pareado**.
   - En cada remuestreo se toman las mismas pacientes para los dos modelos. Así el intervalo mide
     la diferencia entre ellos y no lo fácil o difícil que sea la muestra.
   - Si el intervalo cruza el 0, no hay pruebas de que uno sea mejor.
2. **CNN + médico (modelo combinado):** la misma regresión logística con una variable más, la
   puntuación de la CNN.
   - La pregunta clave es si **(CNN + médico) − médico** es mayor que 0: **eso es lo que añade
     la imagen** a lo que ya sabe el médico.
   - Se entrena con las predicciones fuera de fold de la CNN: cada paciente, puntuada por un
     modelo que no la vio. A esto se le llama apilado (*stacking*), y es honesto.
3. **¿La CNN repite el subtipo?** Dos pistas:
   - la correlación entre la puntuación de la CNN y la del médico;
   - el AUC de la CNN **dentro de cada subtipo**. Dentro de un subtipo el médico casi no
     distingue; si la CNN sí, la imagen aporta algo más que la biopsia.

**Test.** El médico y el combinado se ajustan con train y quedan congelados en
`models/clinico/modelo_clinico.json` (números en JSON, sin pickle). Se evalúan en test **en la
misma evaluación única que la CNN** (`evaluar_test`).
- `evaluar_test` se niega a empezar si falta ese fichero, o si el combinado se hizo con otra CNN
  (lo comprueba con la huella del modelo). Si no, la comparación en test quedaría fuera de la
  evaluación única.
- El volumen no va a test: es una variante de train, no el médico.

**Límites.**
- Es un "médico" simplificado: un oncólogo sabe más (grado, Ki-67, tamaño clínico, ganglios), y
  aquí no está.
- HR y HER2 se definen con criterios algo distintos en cada cohorte (Fridman et al.).
- En test, al combinado se le da la puntuación del conjunto de 5 modelos de la CNN, que es algo
  más estable que la de un solo modelo, con la que se entrenó. Es la misma pequeña diferencia que
  ya tiene el umbral de la CNN (§5.2).

> **Si te preguntan** «¿por qué no metéis los datos clínicos en la red?»: porque el caso pide una
> CNN que prediga con las tres fases, y el profesor quiere las variables clínicas como
> comparación. Además, la app recibe solo imágenes. Lo que sí hacemos es medir cuánto añade la
> imagen al médico con un modelo combinado, aparte.
>
> «¿Por qué regresión logística?»: cuatro variables, mil pacientes y una respuesta sí/no. Es el
> modelo estándar en clínica, no sobreajusta (0,01 de diferencia train–validación) y cada peso
> se explica como una razón de odds.
>
> «¿Y los análisis de sangre?»: no están en el dataset. Tampoco peso, IMC ni tabaco.

---

## 3. La red, versión a versión

Toda la historia, con hipótesis y resultados, está en [`ARQUITECTURAS.md`](ARQUITECTURAS.md). Aquí
va el porqué de cada pieza.

### 3.1 Las piezas de cnn_v1 (diseño de Judith, D10)

| Pieza | Qué hace | Por qué |
|---|---|---|
| Convolución 3×3 | pasa un filtro pequeño por la imagen y detecta patrones locales | es el tamaño estándar: dos 3×3 seguidas ven lo mismo que una 5×5 con menos parámetros |
| BatchNorm | normaliza la salida de cada capa durante el entrenamiento | entrena más estable y depende menos de la tasa de aprendizaje y de la inicialización |
| ReLU | deja pasar lo positivo y anula lo negativo | da a la red capacidad no lineal; es la activación estándar |
| MaxPool 2×2 | reduce el mapa a la mitad quedándose con el máximo de cada zona | resume ("¿aparece el patrón aquí?") y amplía lo que ve cada neurona |
| Pooling global | hace la media de cada mapa y deja un número por canal | 64 números en vez de 65.536: muchos menos parámetros y menos sobreajuste |
| Cabeza lineal → 1 logit | combina los 64 números en una salida | el enunciado pide una única salida (un logit) |

- **Tamaño:** 3 bloques con 16 → 32 → 64 filtros y 23.761 parámetros. Es pequeña a propósito:
  solo hay ~880 pacientes para entrenar.
- **No hay sigmoide al final de la red** porque `BCEWithLogitsLoss` la incluye por dentro, y así es
  más estable numéricamente. Al predecir se aplica la sigmoide.

### 3.2 Qué enseñó cnn_v1 (R1, R2)

- **El AUC se quedó en 0,585 con pérdida normal y 0,572 con ponderada:** al nivel del azar y de
  "adivinar la cohorte".
- **Estaba memorizando:** la pérdida de entrenamiento bajaba y la de validación subía desde la
  época 3. Aprendía detalles de las pacientes de entrenamiento que no sirven con otras.
- **La pérdida ponderada no arregla una red que no ordena bien:** cambia la sensibilidad (ver
  §4.1), pero no el AUC.

### 3.3 cnn_v2: ver más contexto (diseño de Judith, D11)

- **Qué cambia:** un cuarto bloque de 128 filtros (97.809 parámetros).
- **Qué es el campo receptivo:** cuántos píxeles de la imagen original "ve" cada punto del mapa
  final. Con 3 bloques, unos 22×22 de 256×256. Con 4, unos 46×46: un trozo del tumor y su
  entorno.
- **Hipótesis:** cnn_v1 no veía lo suficiente.
- **Riesgo:** si el problema era la memorización, más capacidad puede empeorarlo. Por eso se
  prueba también cnn_v3.

### 3.4 cnn_v3: aumento de datos (D18)

- **Qué cambia:** la red es igual que cnn_v2, pero en cada época cada corte se ve ligeramente
  transformado:
  - volteo horizontal la mitad de las veces;
  - giro de hasta ±10°;
  - desplazamiento de hasta un 4 %;
  - escala entre 0,95 y 1,05.
- **Por qué:** si la red nunca ve dos veces exactamente la misma imagen, no puede memorizar
  píxeles y tiene que aprender lo que se mantiene, que es el tumor y su realce. Es la forma más
  eficaz de combatir el sobreajuste con pocos datos.
- **Las tres reglas que cumple:**
  1. **La misma transformación para las tres fases.** Si se girase EARLY y no PRE, la resta ya no
     sería el realce. Un test comprueba que tres canales iguales siguen siendo iguales tras
     transformarlos.
  2. **Solo geometría.** Cambiar el brillo alteraría la señal (el realce es una diferencia de
     brillo entre fases).
  3. **Solo en entrenamiento.** Validación, test y app ven la imagen tal cual.
- **Por qué no hay volteo vertical:** pondría la pared torácica donde no puede estar. El volteo
  horizontal sí es realista: una mama izquierda volteada parece una derecha.
  - **Corregido el 21 de octubre (D29, §3.10):** en este dataset la pared torácica no está siempre
    en el mismo lado. Cada hospital orienta la mama a su manera, y cnn_v10 gira también 90°.
- **Por qué se define antes de ver cnn_v2:** para entrenar todo de una vez por la noche (§4.6).
  Además, cualquier resultado de cnn_v2 hace útil a cnn_v3.

### 3.5 cnn_v4: darle el realce hecho (D19)

- **Qué cambia:** la red recibe las 3 fases y calcula **dentro** EARLY − PRE y LATE − EARLY, que
  añade como dos canales más. Son 98.097 parámetros.
- **Por qué:** la red podría aprender a restar canales por su cuenta, pero empieza con filtros al
  azar y hay pocos datos. Darle la resta hecha es poner conocimiento del dominio (la cinética del
  contraste), y está permitido: no es un modelo preentrenado ni transfer learning.
- **Por qué dentro de la red y no al preparar los datos:** la app le pasa las mismas 3 fases que
  el entrenamiento, así que es imposible preprocesar distinto. Es justo el fallo que el profesor
  busca en la defensa.

### 3.6 Una idea por versión, y este orden

- **Una idea por versión.** Cada versión cambia **una sola cosa** respecto a su origen
  (v1 → v2 → v3 → v4). Así, si algo mejora o empeora, se sabe por qué. Si cnn_v4 cambiara a la vez
  el tamaño, el aumento y el realce, no sabríamos qué funcionó.
- **El orden va de lo general a lo específico:**
  1. capacidad (v2);
  2. regularización (v3);
  3. entrada con conocimiento del dominio (v4).
- La segunda ronda (v5–v8, §3.9) sale de los resultados de la primera: cada versión prueba una idea
  sobre la candidata, cnn_v3, y cnn_v8 las junta.
- La tercera (cnn_v10, §3.10) sale de mirar por qué la segunda no mejoró.

### 3.7 Inicialización, optimizador y demás hiperparámetros (D8, D24)

| Decisión | Valor | Por qué |
|---|---|---|
| Inicialización | la de PyTorch: uniforme de Kaiming en convoluciones y lineales; BatchNorm con peso 1 y sesgo 0 | Kaiming está pensada para ReLU; con BatchNorm detrás de cada convolución, la escala inicial apenas importa porque BN renormaliza |
| Optimizador | Adam, tasa de aprendizaje 0,001 | adapta el paso de cada parámetro; 0,001 es el valor por defecto que suele funcionar con BatchNorm |
| Lote | 32 cortes, fijo | cabe de sobra en la GPU, y fijarlo hace comparables todos los entrenamientos |
| Épocas | hasta 60, parando si en 10 no mejora el AUC de validación | da margen a redes que aprenden despacio (con aumento) sin entrenar de más |
| Semilla | 42 en todo | reproducibilidad: el mismo comando da el mismo resultado (salvo pequeñas diferencias de la GPU) |

> **Si te preguntan** «¿por qué no usaste ResNet?»: el enunciado lo prohíbe. Hay que construir la
> CNN desde cero, sin arquitecturas de catálogo ni pesos preentrenados. Por eso la red es mía,
> pequeña y adaptada a pocos datos.

### 3.8 Qué enseñó la primera ronda (cnn_v2, v3 y v4)

Se entrenaron la noche del 20 de octubre: cada validación cruzada tardó unos 45 minutos. Detalle,
fold a fold, en las fichas de [`ARQUITECTURAS.md`](ARQUITECTURAS.md).

| Versión | AUC validación (5 folds) | AUC entrenamiento | Diferencia | ¿Delata el hospital?* |
|---|---|---|---|---|
| cnn_v2 | 0,593 ± 0,032 | 0,755 | 0,162 | 0,58 |
| cnn_v3 (aumento) | 0,576 ± 0,037 | 0,656 | 0,080 | **0,66** |
| cnn_v4 (realce) | 0,572 ± 0,044 | 0,698 | 0,126 | 0,61 |

\* AUC de la puntuación de la red para distinguir I-SPY2 del resto. 0,5 = no sabe de qué hospital
es la paciente.

**Lo que dicen, en cuatro frases:**
1. **Las tres empatan** (las diferencias son menores que una desviación). Gana cnn_v3 por el
   criterio: cnn_v2 empata pero sobreajusta, y cnn_v4 empata pero es más compleja.
2. **La red apenas aprende algo que valga para pacientes nuevas.** Desde la primera época, el AUC
   de validación se mueve entre 0,50 y 0,60 sin subir; solo sube el de entrenamiento. Es la figura
   «MAL» del correo del profesor, en suave. El aumento de datos frena la memorización (la
   diferencia baja de 0,16 a 0,08), pero no hace aprender más.
3. **En parte reconoce el hospital.** Su puntuación distingue I-SPY2 (más pCR y más realce) con AUC
   0,66, y dentro de Duke e I-SPY1 se queda en ~0,50–0,53.
4. **No repite el subtipo** (AUC ~0,45–0,50 para HER2 y HR), y la poca señal que tiene está en
   HER2+ y triple negativo (~0,58–0,63). Por eso, junto al médico, puede sumar (§2.5).

**Una cautela sobre las cifras.** En cada fold se queda la mejor época de 13–23 medidas sobre las
mismas pacientes de validación. Elegir el máximo de muchas medidas con ruido infla un poco el
resultado: época a época, la validación ronda 0,53–0,58. La cifra honesta será la de test, que se
mide una sola vez. Las comparaciones entre versiones siguen siendo justas, porque todas tienen el
mismo sesgo.

> **Si te preguntan** «¿tu red aprende biología o el hospital?»: lo medimos. La puntuación de
> cnn_v3 distingue I-SPY2 del resto con AUC 0,66, así que en parte aprende el hospital; dentro de
> Duke e I-SPY1 está casi al azar. Por eso la segunda ronda empieza por quitarle ese atajo (§3.9).
> Lo comprobamos porque desde ese día cada validación cruzada lo calcula sola («¿delata el
> hospital?»).

### 3.9 Segunda ronda: cnn_v5 a cnn_v8 (D28)

Judith pidió intentar acercarse a 0,70 sin sobreentrenar. Con lo de §3.8, hay tres problemas y una
idea para cada uno, todas sobre cnn_v3. Como cada validación cruzada tarda ~45 minutos, en una
noche caben las tres por separado y además las tres juntas.

**cnn_v5 · Intensidad relativa a la propia paciente (contra el atajo del hospital).**
- **Qué hace:** dentro de la red, divide las tres fases de cada corte entre la intensidad media del
  PRE en su tejido. El tejido pasa a valer ~1 en PRE, y un 1,4 en EARLY significa «realza un 40 %».
- **Por qué:** cada escáner tiene su escala, y el dataset no las armoniza (Fridman et al.). En la
  auditoría, el realce medio es 0,149 en I-SPY2 y ~0,11 en Duke e I-SPY1. Con la intensidad
  relativa, el brillo del escáner deja de servir de pista. Es la idea del «realce relativo» que se
  usa en radiología.
- **Por qué dentro de la red:** igual que el realce de cnn_v4, la app recibe las mismas 3 fases y
  no puede haber una normalización distinta (lo que el profesor comprueba en la defensa).
- **En otro trabajo con este mismo dataset** (THDA-ResNet, sobre HER2), solo cambiar cómo se
  normalizaba la intensidad movía el AUC entre 0,70 y 0,74, y entre 0,60 y 0,66 en otro hospital.

**cnn_v6 · Pooling por atención (contra la dilución del tumor).**
- **El problema:** el tumor ocupa una zona pequeña del corte y realzan también el corazón y la
  pared torácica. Las versiones anteriores terminan con la **media** de las 16×16 = 256 posiciones
  del último mapa: cada posición pesa 1/256 y las pocas del tumor se pierden entre el fondo. La
  auditoría lo apoya: el realce medio de todo el tejido solo da AUC 0,54.
- **Qué hace la atención** (Ilse, Tomczak y Welling, 2018): la red calcula un **peso** para cada
  posición y hace una **media ponderada**. Los pesos salen de una mini-red de dos capas,
  `a_k = softmax(w · tanh(V · h_k))`, y suman 1. Si todos fueran iguales, sería la media de
  antes: la atención es una generalización de la media global.
- **Coste:** 4.161 parámetros más (101.970 en total).
- **De regalo:** los pesos se pueden dibujar encima de la imagen y ver **dónde mira la red**. Si
  mira el tumor, es un argumento fuerte en la defensa; si mira el corazón, también es un hallazgo.

**cnn_v7 · Regularización (contra la memorización).**
- **Dropout 0,3:** al entrenar, en cada paso se apaga al azar el 30 % de las 128 entradas de la
  capa final. La red no puede apoyar la decisión en unas pocas neuronas concretas. Al evaluar no se
  apaga nada.
- **Weight decay con AdamW (0,05):** en cada paso los pesos se encogen un poco hacia 0. Los pesos
  grandes son los que permiten memorizar detalles de unas pocas pacientes. AdamW (Loshchilov y
  Hutter, 2019) aplica ese encogimiento aparte del paso de Adam, que es como funciona bien. No se
  encogen los sesgos ni BatchNorm, que no memorizan.
- **Por qué ahora:** la red no tenía ninguna de las dos herramientas estándar contra el
  sobreajuste.

**cnn_v8 · Las tres juntas.** Si cada idea ayuda algo, juntas deberían sumar. Rompe a propósito la
regla de «una idea por versión»: se entrena a la vez que v5–v7, y ellas dicen qué aporta cada
pieza.

**Cómo se elegirá:** con el mismo criterio de siempre (D16). Además se mirará «¿delata el
hospital?» y el AUC dentro de cada hospital. Si una versión baja un poco la media pero deja de
reconocer el hospital, es más honesta, y eso se contará en el informe.

**Siendo realistas:** la red encuentra poca señal en un corte 2D. Una mejora de 0,02–0,05 sería
un buen resultado; llegar a 0,70 con la imagen sola es poco probable con estos datos. La vía
honesta hacia ~0,70 es la comparación **CNN + médico** (§2.5): el médico solo ya da 0,69 y la CNN
aporta información que el subtipo no tiene.

> **Si te preguntan** «¿por qué no hiciste la red más grande?»: porque el problema era que
> memorizaba, no que se quedara corta. De cnn_v1 a cnn_v2 cuadruplicamos los parámetros y el AUC de
> validación apenas cambió (0,585 → 0,593), mientras el de entrenamiento se disparaba. Más grande
> habría memorizado más.
>
> «¿Por qué atención y no otra cosa?»: el tumor es pequeño y la media global lo diluye. La atención
> deja que la red pese más donde importa, cuesta solo 4.000 parámetros y además enseña dónde mira.

### 3.10 Por qué la segunda ronda no mejoró, y cnn_v10 (D29)

**Lo que salió** (21 de octubre, cifras del registro del entrenamiento; la lectura completa irá en
las fichas cuando llegue el zip de `para_claude`):

| Versión | AUC validación (5 folds) | ¿Delata el hospital? Duke · I-SPY1 · I-SPY2 |
|---|---|---|
| cnn_v3 (referencia) | 0,576 ± 0,037 | 0,42 · 0,28 · 0,66 |
| cnn_v5 (intensidad relativa) | 0,574 ± 0,035 | 0,41 · 0,41 · 0,61 |
| cnn_v6 (atención) | 0,591 ± 0,041 | 0,45 · 0,32 · 0,61 |
| cnn_v7 (dropout y weight decay) | 0,588 ± 0,042 | 0,42 · 0,40 · 0,61 |

cnn_v8 se cayó por un fallo de la tarjeta gráfica, no del código: v5–v7 funcionaron con las mismas
piezas (D29). Se relanza.

**Ninguna mejora es clara:** todas están a menos de una desviación de cnn_v3. Para averiguar por qué,
se miraron otra vez las predicciones fuera de fold de cnn_v3 junto con los metadatos:

1. **Dentro de cada hospital, la red casi acierta por azar:** Duke 0,51, I-SPY1 0,53, I-SPY2 0,58.
2. **Su puntuación no tiene que ver con el tamaño del tumor** (correlación de Spearman −0,05 con el
   tamaño de su caja). Y eso que el tamaño de la caja, él solo, da AUC 0,59 en I-SPY2. Una red que
   mirase el tumor aprendería al menos su tamaño. Tampoco capta el subtipo (§3.8).
3. **Memoriza en vez de aprender.** La mejor época de cada fold fue la 1, 6, 5, 16 y 6. Después,
   el AUC de entrenamiento sube hasta 0,71–0,87 y el de validación no se mueve.
4. **El tumor es pequeño:** su caja ocupa de mediana un 7–8 % de la imagen en I-SPY y un 3 % en Duke.
   Con una etiqueta por paciente y unas 880 pacientes de entrenamiento, la red no descubre sola qué
   parte de la imagen importa. La atención de cnn_v6 tenía que encontrar el tumor sin ninguna pista.
5. **Sí reconoce el hospital, y no solo porque cambie la pCR entre hospitales.**
   - Si la puntuación solo reflejara la pCR, «¿delata el hospital?» saldría 0,49–0,51 ± 0,02–0,03.
     cnn_v3 da 0,66 para I-SPY2 y 0,28 para I-SPY1.
   - Pero eso aporta poco al AUC global: 0,574 en total frente a 0,560 de media dentro de cada
     hospital.
   - Es un síntoma, no la causa principal.

**Por qué normalizar la intensidad (cnn_v5) no quitó el atajo.** Al mirar 10 pacientes al azar de
cada hospital ([`orientacion.png`](resultados/1_auditoria/orientacion.png), nueva figura de la
auditoría), se ve que **cada hospital orienta la mama a su manera**:
- en I-SPY1 apunta hacia abajo (las 10);
- en I-SPY2 y Duke, a la izquierda o a la derecha;
- en Duke y en alguna de I-SPY2 se ven el corazón y los pulmones.

Nuestro aumento ya volteaba izquierda-derecha, así que «izquierda o derecha» no da pista, pero
«hacia abajo» delata I-SPY1. Eso explica su 0,28.

**cnn_v10 · cnn_v3 + giros de 90°.**
- **Qué hace:** al entrenar, cada corte se gira además 0°, 90°, 180° o 270° al azar, igual en las tres
  fases (`geometrico+giros90` en `src/data/aumento.py`). Con el volteo, salen las 8 orientaciones
  del cuadrado con la misma probabilidad, así que la orientación deja de servir de pista.
- **No cambia nada fuera del entrenamiento:** ni la red, ni la app, ni la validación, ni el test.
  No necesita permiso del profesor.
- **Siendo realistas:**
  - quita un atajo, pero no le enseña a la red dónde está el tumor;
  - puede que el AUC no suba, porque el atajo «ayudaba» (I-SPY1 tiene menos pCR que la media);
  - aun así, una red que no reconoce el hospital es más honesta y se puede defender mejor
    (rúbrica, punto 9).
- **Si empata con cnn_v3, por el criterio gana cnn_v3** (mismos parámetros, más antigua). El criterio
  está fijado desde el 18 de octubre y no se cambia después de ver resultados. Que deje de reconocer
  el hospital se contará en el informe igualmente.

**cnn_v9 (reservada) · atención guiada por la caja del tumor.** Es la idea que ataca el problema 4:
- **Al entrenar:** una pérdida extra castiga que la atención (la de cnn_v6) se vaya fuera de la caja
  del tumor, ampliada un poco porque el tejido de alrededor también informa (Braman et al. 2017).
- **Al predecir:** la red no necesita la caja, así que la app y la validación privada no cambian.
- **En la bibliografía:** guiar así la atención es una técnica conocida contra los atajos (Ross et
  al. 2017, «Right for the right reasons»; Li et al. 2018, GAIN).
- **Por qué está pendiente:** la caja sale de `patients.csv`, y el profesor dijo que esas variables
  no entran en la CNN. Por eso se le pregunta antes (D29).
- **Antes de entrenarla:** comprobar que la caja cae sobre el tumor en los PNG. Las coordenadas
  están en el tamaño original de cada resonancia (de 128 a 512 píxeles), no en 256.

**Lo que no se hace, y por qué** (preguntas de Judith del 21 de octubre):
- **Quitar el fondo negro:** vale 0 y la red aprende enseguida a no hacerle caso. Recortar al borde
  de la mama sí agrandaría el tumor, pero cada imagen quedaría a una escala distinta y se perdería
  el tamaño real. No es prioritario.
- **Ajustar brillo o contraste:**
  - El brillo de cada hospital ya se quitó en cnn_v5, y no mejoró.
  - Ecualizar o subir el contraste de cada imagen la cambia de forma distinta en cada paciente y
    rompe la comparación entre fases, que es la señal (§2.2).
- **Las imágenes no son en color.** Son PNG en gris (la auditoría lo comprueba en las 10.945). El
  rojo y azul de las figuras es el mapa de realce pintado para verlo mejor; la red no ve esos
  colores.

> **Si te preguntan** «¿tu red reconoce el hospital?»: sí, y lo medimos. Lo más claro es I-SPY1:
> la red lo distingue con AUC 0,28 (0,5 sería no distinguirlo), porque allí la mama se orienta
> hacia abajo y en los otros hospitales hacia los lados. Por eso cnn_v10 entrena con giros de 90°,
> para que la orientación deje de servir de pista. Y lo comprobamos con la medida «¿delata el
> hospital?» antes y después.
>
> «¿Por qué no recortaste alrededor del tumor?»: porque la caja del tumor viene de `patients.csv`,
> que no puede entrar en la CNN, y la app y la validación privada no la tendrían. Lo que planteamos
> es usarla solo al entrenar, para guiar la atención, y se lo preguntamos al profesor antes.

---

## 4. Entrenar y comparar

### 4.1 Pérdida normal frente a ponderada (rúbrica punto 6)

- **Pérdida normal:** BCE (entropía cruzada binaria). Penaliza igual equivocarse con un positivo
  que con un negativo. Como el 70 % son negativos, a la red le compensa dar probabilidades bajas a
  todas: con umbral 0,5 casi nunca dice pCR, y la **sensibilidad queda cerca de 0** (R1).
- **Pérdida ponderada:** `pos_weight = N0/N1 ≈ 2,40` multiplica por 2,4 el coste de fallar un
  positivo. Las probabilidades suben, ya caen a los dos lados de 0,5 y **la sensibilidad aparece**
  (R2).
- **Pero el AUC no cambia,** porque la ponderación desplaza el punto de corte, no la capacidad de
  ordenar a las pacientes.
- **Un detalle matemático útil:** con `pos_weight = w`, la red aprende a dar un logit inflado en
  unos ln(w) ≈ 0,88. Eso es lo que corrige después la calibración (§5.5).

> **Si te preguntan** «¿qué le pasa a la sensibilidad?»: con la normal, casi 0 a umbral 0,5,
> porque la clase mayoritaria domina; con la ponderada sube, porque fallar un positivo cuesta 2,4
> veces más. El AUC es casi igual en las dos: la ponderación no hace que la red distinga mejor,
> solo mueve el corte. Por eso el umbral se elige aparte, con validación.

### 4.2 Época elegida y parada temprana

- **Qué:** tras cada época se mide el AUC por paciente en validación, se guarda la mejor época y
  se para si en 10 épocas no mejora.
- **Matiz que hay que saber decir:** la época se elige mirando ese mismo fold, así que su AUC es
  un poco optimista. Por eso la cifra honesta es la de test, medida una sola vez.

### 4.3 AUC de entrenamiento sobre 220 pacientes (D12, D16)

- **Qué:** en cada época se mide también el AUC en una muestra fija de 220 pacientes de
  entrenamiento.
- **Por qué:** la rúbrica pide demostrar el sobreajuste con curvas de entrenamiento y validación
  juntas, y la diferencia en la época elegida. Una diferencia grande (0,38 en el ejemplo del
  profesor) es sobreajuste.
- **Por qué 220:** es el tamaño de un fold, así que los dos AUC tienen un ruido parecido y su
  diferencia es comparable. Usar todas las pacientes alargaría cada época un 80 %.

### 4.4 Validación cruzada de 5 folds (rúbrica punto 2)

- **Qué:** la misma versión se entrena 5 veces, cada vez dejando un fold distinto para validar.
  Se da la **media ± desviación** de los 5 AUC.
- **Por qué:** con 219 pacientes, el AUC de un fold suelto tiene un ruido de ±0,04, y dos
  versiones pueden parecer distintas por azar. La media de 5 es más estable, y la desviación dice
  cuánto varía.
- **De regalo:** cada paciente de train queda predicha una vez por el modelo que no la vio. Son
  las predicciones **fuera de fold**, la base de §5.

### 4.5 Criterio para comparar versiones (D16)

Lo aplica el código (`elegir_mejor`), así que no depende de mirar la tabla con buenos ojos.

1. Solo cuentan validaciones cruzadas completas (5 folds), con pérdida normal y la definición
   actual de la versión (huella ✓).
2. La de mayor AUC medio es la referencia.
3. Las que quedan a menos de **una desviación** de ella están **empatadas**: no hay pruebas de que
   sean peores.
4. Entre las empatadas se desempata así:
   - primero, las que **no sobreajustan** (diferencia de AUC entre entrenamiento y validación
     ≤ 0,15), porque la rúbrica pide descartar los modelos sobreentrenados;
   - después, la **más simple** (menos parámetros);
   - si siguen empatadas, la más antigua.

**Ejemplo.** cnn_v2 da 0,630 con diferencia 0,25; cnn_v3, 0,640 con diferencia 0,08; cnn_v4,
0,645 ± 0,03 con diferencia 0,07. Las tres empatan (todas están por encima de 0,615). cnn_v2 queda
fuera por sobreajustar. Entre cnn_v3 y cnn_v4 gana cnn_v3, que tiene menos parámetros.

> **Si te preguntan** «¿por qué te quedas con esa red y no con la de AUC más alto?»: porque la
> diferencia está dentro del ruido (menos de una desviación entre folds). Ante la duda, elijo la
> que no sobreajusta y la más simple: generaliza mejor y es más fácil de justificar.

### 4.6 Plan de experimentos (D17)

`python -m src.training.plan` entrena de noche, una tras otra, estas validaciones cruzadas:

| Paso | Qué | Pregunta que responde |
|---|---|---|
| 1 | cnn_v2, pérdida normal | ¿ayuda ver más contexto? |
| 2 | cnn_v3, pérdida normal | ¿el aumento corrige la memorización? |
| 3 | cnn_v4, pérdida normal | ¿ayuda darle el realce hecho? |
| 4 | cnn_v5, pérdida normal | ¿ayuda la intensidad relativa al PRE (quitar el atajo del hospital)? |
| 5 | cnn_v6, pérdida normal | ¿ayuda el pooling por atención (no diluir el tumor)? |
| 6 | cnn_v7, pérdida normal | ¿ayudan el dropout y el weight decay (no memorizar)? |
| 7 | cnn_v8, pérdida normal | ¿suman las tres ideas? |
| 8 | cnn_v10, pérdida normal | ¿ayuda quitar la pista de la orientación (giros de 90°)? |
| 9 | la ganadora, pérdida ponderada | el enunciado pide comparar normal y ponderada con el modelo elegido |

Los pasos 1–3 y la ponderada de cnn_v3 se hicieron la noche del 20 de octubre (§3.8). Los pasos 4–7
se añadieron el 21 (D28, §3.9) y el paso 8 ese mismo día por la tarde (D29, §3.10). Si al final
sigue ganando cnn_v3, el paso 9 ya está hecho.

- **Por qué se comparan con pérdida normal:** para que entre versiones solo cambie una cosa.
- **Por qué la ponderada solo con la ganadora:** en R1 y R2 la ponderación movió la sensibilidad
  y no el AUC, así que entrenar la ponderada de todas no cambiaría la elección y gastaría GPU.
- **Es reanudable:** si se corta, se lanza otra vez y sigue donde iba (mira el histórico).
- **Duración medida:** unos 45 minutos por paso en la RX 6700 XT (~30 s por época). La primera
  ronda entera tardó unas 3 horas.

### 4.7 Histórico, gráficas y huella (D13, D14)

- **Histórico:** cada ejecución deja una fila en `runs/historico.csv` y sus gráficas en
  `graficas/` (curvas de entrenamiento y validación, matriz de confusión).
- **Tablas automáticas:** `ARQUITECTURAS.md` tiene una tabla resumen y la genealogía que se
  actualizan solas.
- **Huella:** cada versión tiene una de 8 caracteres. Si una versión ya entrenada se modificara,
  sus ejecuciones saldrían marcadas con ⚠.
- **Para qué sirve todo esto:** se puede seguir cualquier cifra del informe hasta el entrenamiento
  exacto que la produjo.

### 4.8 Ver pronto si una versión va bien: el fold 0 contra la referencia (D30)

Judith pidió no esperar 45 minutos para saber si una versión nueva va bien. Ya había dos gráficas que
se redibujan al acabar cada época:
- `runs/<ejecución>/fold_0/graficas/curvas.png`: entrenamiento y validación del fold 0, dos líneas
  por panel, como las figuras «BIEN / MAL» del profesor;
- `runs/<ejecución>/graficas/curvas_folds.png`: los folds terminados, superpuestos.

Faltaba comparar con la referencia. `python -m src.training.comparar cnn_v3 cnn_v10` lo hace:
- **Qué dibuja:** cuatro paneles del fold 0, con una línea por versión en cada uno:
  - AUC por paciente en validación y en entrenamiento;
  - pérdida en validación y en entrenamiento.

  Se guarda en la carpeta de la nueva, en `graficas/comparacion_fold0_con_cnn_v3.png`.
- **Por qué AUC y pérdida, y no la curva ROC:** para seguir un entrenamiento importa cómo cambian
  época a época. El sobreajuste se ve cuando la pérdida de entrenamiento baja y la de validación
  sube, y el AUC de validación dice si ordena mejor a las pacientes. La curva ROC es de un modelo
  ya cerrado: sirve para elegir y explicar el umbral (§5.4) y para comparar con el médico en test
  (§6), no para seguir épocas.
- **Qué imprime:** la época elegida de cada una y, en las pacientes de ese fold, el AUC dentro de
  cada hospital y «¿delata el hospital?».
- **Se puede usar mientras entrena:** el fold 0 es el primero que se entrena (~10 minutos) y su
  historial se guarda cada época. El script no carga torch ni toca la tarjeta gráfica.
- **No cambia el plan:** el plan sigue entrenando los 5 folds. Si en el fold 0 la nueva va claramente
  peor durante 5–10 épocas, se puede parar y ahorrar el resto (regla de `CLAUDE.md`).

**Cuidado al leerlo:**
- Un fold tiene ~220 pacientes y su AUC baila unas ±0,04 de una ejecución a otra. En las de un solo
  hospital hay aún menos pacientes (unas 21 de I-SPY1 por fold).
- Solo dice algo una diferencia grande (≥ 0,05) y que se mantiene varias épocas.
- Para decidir cuentan los 5 folds (D16, rúbrica punto 2). Por eso el plan no se cambia a «solo el
  fold 0».

> **Si te preguntan** «¿por qué no entrenaste solo un fold para ir más rápido?»: lo usamos para
> mirar pronto, pero no para decidir. Un fold baila ±0,04, más que las diferencias que buscábamos
> entre versiones. Las decisiones salen de los 5 folds, con media y desviación.

### 4.9 Un solo fichero de configuración: `config.json` (D31)

Judith pidió tener todos los parámetros en un fichero JSON, como una plantilla. Está en la raíz del
repo, [`config.json`](../config.json), y tiene cuatro secciones:

| Sección | Qué lleva | ¿Se toca? |
|---|---|---|
| `experimento` | qué versión, modo, pérdida, folds, cambios sobre el modo y nota | sí: es la plantilla para probar cosas |
| `modos` | `rapido` y `completo`: pacientes, fold, batch, épocas, paciencia, learning rate, AUC de entrenamiento, semilla, procesos de carga | no, salvo `num_workers`: es el estándar con el que se comparan las versiones |
| `evaluacion` | agregación por paciente, umbral durante el entrenamiento y especificidad mínima del umbral final | solo antes de cerrar el modelo, y con motivo (D21) |
| `comparacion` | diferencia de AUC a partir de la cual una versión sobreajusta (0,15) | no: es el criterio fijado (D16) |

Al final del fichero, `_ayuda` explica cada parámetro en una línea.

**Cómo se usa para probar algo:**
1. Editar `experimento` en `config.json`. Por ejemplo, cnn_v10 solo en el fold 0:
   `"modelo": "cnn_v10", "folds": [0]`. Para otro learning rate, `"cambios": {"learning_rate": 0.0001}`.
2. Comprobar qué se va a lanzar: `python -m src.training.experimento --ver`.
3. Lanzarlo: `nohup python -m src.training.experimento > experimento.log 2>&1 &`.
4. Seguirlo con `python -m src.training.comparar cnn_v3 cnn_v10` (§4.8).

En la carpeta de la ejecución se guarda una copia, `config_usada.json`, para saber siempre qué se
lanzó. También se puede escribir otro JSON solo con la sección `experimento` y lanzarlo con
`--config otro.json`.

**Qué no está en `config.json`, a propósito:** lo que define una versión de la red. Son las capas, el
pooling, el dropout, el aumento de datos, el optimizador y el weight decay, y están en
`src/models/versiones.py`. Cambiarlos es crear otra versión, con su ficha, su huella y su hipótesis
(D14). Si estuvieran en el JSON, cambiar un número cambiaría cnn_v3 sin que nada lo registrara.

**Una protección nueva en el criterio.** Con el JSON es muy fácil lanzar, por ejemplo, cnn_v3 con
otro learning rate en los 5 folds. Esa ejecución no debe pasar por «la» cnn_v3. Por eso el criterio
(D16) y el plan solo cuentan las validaciones cruzadas con el batch, el learning rate y la semilla
de su modo (`es_estandar` en `src/config.py`). Las demás quedan en el histórico como pruebas. Si una
prueba sale bien, se convierte en versión.

**Qué no cambió:** los valores de los modos son los mismos que antes en `src/config.py` (un test lo
comprueba), así que todas las ejecuciones anteriores siguen siendo comparables.

> **Si te preguntan** «¿dónde están los hiperparámetros?»: los de ejecución, en `config.json`, con
> una explicación de cada uno. Los de cada arquitectura, en `src/models/versiones.py`, porque cada
> versión es una idea con su ficha. Y cada entrenamiento guarda la configuración que usó, así que
> cualquier cifra se puede reproducir.

---

## 5. Cerrar el modelo

`python -m src.evaluation.cerrar_modelo` hace todo esto y lo deja en
[`CIERRE_MODELO.md`](resultados/2_cierre/CIERRE_MODELO.md) (D21).

### 5.1 Todo se decide con predicciones fuera de fold

- **Qué es "fuera de fold":** cada paciente de train está predicha por el modelo del fold en el
  que estaba en validación, es decir, por un modelo que no la vio. Son 1.097 predicciones
  "honestas" sin tocar test.
- **Qué se decide con ellas:** la calibración, la agregación y el umbral.
- **Por qué no con test:** usar test para elegir cualquiera de las tres cosas invalida el trabajo.

### 5.2 El modelo final es el conjunto de los 5 modelos de la validación cruzada

- **Qué:** para un corte, cada uno de los 5 modelos (uno por fold) da un logit, y se hace la
  media.
- **Por qué es mejor que reentrenar con todo train:**
  - cada uno de los 5 eligió su época con su propio fold de validación, lo que es una parada
    temprana honesta; reentrenando con todo no habría con qué parar y habría que adivinar el
    número de épocas;
  - la media de 5 modelos que vieron datos distintos es más estable y suele calibrar mejor;
  - el coste es despreciable: 5 redes de ~100.000 parámetros predicen en milisegundos en CPU.
- **El enunciado pide una única salida:** se cumple, porque la media de los logits es un único
  logit por corte.
- **Límite que hay que saber contar:** el umbral y la calibración se ajustaron con predicciones de
  modelos sueltos (las fuera de fold), y el conjunto da probabilidades algo menos extremas. La
  diferencia es pequeña, y ajustarlos con el conjunto sobre train sería trampa, porque cada modelo
  ya vio el 80 % de train.

### 5.3 Agregación: la media de los cortes

- **Qué:** la probabilidad de una paciente es la media de las de sus cortes. Es lo que indica el
  profesor.
- **Alternativas:** el máximo (basta un corte sospechoso; da más falsos positivos), la mediana o
  el voto. Se calculan para comparar en el informe del cierre, pero se queda la media salvo que
  otra sea claramente mejor.
- **Por qué la media:** usa todos los cortes y es robusta.

### 5.4 Umbral: máxima sensibilidad con especificidad ≥ 0,80

- **Qué hace el umbral:** convierte la probabilidad en una clase. No sale del modelo: es una
  **decisión clínica** sobre qué error preocupa más.
- **Los dos errores posibles:**
  - **Falso positivo:** decir "probablemente pCR" a quien tiene tumor residual. Es el error
    peligroso si alguien relajara el tratamiento o la vigilancia; el enunciado (§2.2) avisa
    justo de ese mal uso.
  - **Falso negativo:** decir "probablemente no pCR" a quien sí responde. En este contexto de
    investigación cuesta menos: la paciente recibe el tratamiento previsto igualmente.
- **Por eso se limita primero el falso positivo:** especificidad ≥ 0,80, es decir, como mucho
  1 de cada 5 pacientes sin pCR recibe un "pCR" equivocado. Dentro de ese límite se detectan
  tantas pCR como se pueda.
- **Por qué 0,80 y no 0,95 como Fridman:** con 0,95 su modelo solo detecta el 27 % de las pCR.
  0,80 es un compromiso razonable.
- **Un detalle técnico:** entre dos probabilidades observadas la matriz de confusión no cambia,
  así que muchos umbrales dan el mismo resultado. Se toma el **punto medio** de ese intervalo y no
  un extremo. Un umbral pegado al borde cambiaría de resultado con cualquier variación mínima de
  las probabilidades, por ejemplo entre el modelo de un fold y el conjunto de los 5.
- **Alternativas** (se calculan para comparar):
  - Youden: máximo de sensibilidad + especificidad − 1; trata igual los dos errores;
  - máxima F1 macro;
  - 0,5.

> **Si te preguntan** «¿qué umbral usas y por qué?»: el que da más sensibilidad con especificidad
> de al menos 0,80, elegido con las predicciones fuera de fold de train. El error que más me
> preocupa es el falso positivo, porque podría llevar a relajar un tratamiento. Si el uso fuera
> otro, por ejemplo buscar candidatas a un ensayo, el umbral cambiaría.

> **Si te preguntan** «¿qué F1 es la óptima?»: ninguna F1 sola. La F1 de la clase pCR engaña:
> decir siempre "pCR" ya da 0,46, más que el modelo de Fridman (0,39). Hay que dar la F1 de las
> dos clases y la F1 macro, y elegir el umbral por el coste clínico de cada error, no por
> maximizar una F1.

### 5.5 Calibración de Platt

- **Qué es la calibración:** una probabilidad está calibrada si, de todas las pacientes a las
  que el modelo da 0,3, alrededor del 30 % logran pCR de verdad.
- **Por qué hace falta:** las redes no suelen estar calibradas, y con pérdida ponderada las
  probabilidades salen infladas (§4.1).
- **Qué es Platt:** una regresión logística de una variable sobre el logit,
  logit_calibrado = a · logit + b, ajustada con las predicciones fuera de fold:
  - `a < 1` corrige el exceso de confianza;
  - `b` desplaza (con pérdida ponderada, b ≈ −0,88 deshace la inflación).
- **Qué no cambia:** el orden de las pacientes, así que el AUC apenas se mueve.
- **Cómo se mide:**
  - con el **error de Brier** (la media de (probabilidad − resultado)²; 0 es perfecto y 0,25 es
    decir siempre 0,5);
  - con la gráfica de calibración por deciles, comparando antes y después.

> **Si te preguntan** «¿está calibrada la probabilidad?»: la he calibrado con Platt usando las
> predicciones fuera de fold, y lo muestro con el Brier y la gráfica de calibración, antes y
> después, y luego en test. Aun así, con 176 pacientes de test la calibración se estima con mucho
> ruido.

### 5.6 Un corte en la app, una paciente en la evaluación

- **El matiz:** el modelo se evalúa **por paciente** (media de sus cortes), pero en la defensa se
  carga **un corte** cada vez. La app da la probabilidad de ese corte y le aplica el mismo umbral.
- **Por qué está bien así:** lo que se verifica en la defensa es que la app dé exactamente lo
  mismo que el pipeline para ese corte. Las métricas de rendimiento son por paciente.

---

## 6. Test: una sola vez

`python -m src.evaluation.evaluar_test` deja el resultado en `docs/resultados/4_test/RESULTADO_TEST.md`
(D22).

- **Usa el modelo final tal cual:** el mismo código que la app y el umbral del cierre. No ajusta
  nada.
- **Se bloquea:** si ya existe el resultado, se niega a repetir. Con `--repetir "motivo"` repite,
  pero el motivo queda escrito en el informe. Volver atrás tras ver test invalida los resultados,
  así que tiene que quedar a la vista.
- **IC del 95 % por bootstrap:** se repite el cálculo 2.000 veces remuestreando **pacientes**
  (no cortes, porque la unidad es la paciente) y se toman los percentiles 2,5 y 97,5. Con 176
  pacientes, el IC del AUC mide unos ±0,08, como dice el profesor.
- **Comparaciones:**
  - con decir siempre "no pCR" (accuracy del 70 %, F1 macro de 0,41) y siempre "pCR";
  - con las métricas fuera de fold (cuánto se pierde con pacientes nuevas);
  - con el médico sin imagen y con CNN + médico, en las mismas pacientes (§2.5, D25);
  - con la bibliografía.
- **Más allá de la cifra (rúbrica punto 9):** resultados por cohorte, calibración y los casos más
  claramente fallados, con sus tres fases y el realce, para mirar en qué se equivoca.

> **Si te preguntan** «¿cuántas veces miraste test?»: una. El script se bloquea solo después de
> la primera vez, y el modelo, el umbral y la agregación estaban cerrados antes, con validación
> fuera de fold.

---

## 7. La app

### 7.1 Por qué Streamlit y Streamlit Community Cloud (D23)

- **Streamlit** monta una interfaz en Python puro con mucho control sobre lo que se muestra. Hace
  falta enseñar mucha información a la vez: las tres fases, el realce, la probabilidad, el umbral,
  la huella del modelo, el dispositivo, el tiempo, el entrenamiento y las métricas.
- **Community Cloud** despliega gratis desde el repo de GitHub y da una URL fija.
- **Alternativas descartadas:**
  - Gradio en Hugging Face Spaces (también válido, pero habría que mantener el código en otro
    repositorio);
  - un servidor propio (hay que mantenerlo y no aporta nada aquí).

### 7.2 Mismo código que el pipeline

- **Lectura y validación:** `src/data/entrada.py`, compartido con la predicción por línea de
  comandos.
- **Modelo:** `src/models/final.py`, compartido con la evaluación de test.
- **Qué valida la entrada:**
  - 3 ficheros, uno por fase, reconocida por el sufijo del nombre: ni ausentes ni duplicadas;
  - las tres del mismo corte y no el mismo fichero repetido;
  - PNG de verdad (firma y decodificación completa);
  - gris de 8 bits, 256×256, como mucho 2 MB;
  - valores finitos.
- **Si las fases parecen cambiadas** (EARLY no más brillante que PRE), avisa.
- **Seguridad:** no acepta rutas, no guarda ni ejecuta nada de lo subido, y los ejemplos son una
  lista cerrada de cortes de train.

### 7.3 La defensa

- **Qué hará el profesor:** cargar 5 cortes de su validación privada sin tocar el código ni
  reiniciar.
- **Lo que verificará y cómo estamos preparados:**
  - **Que la predicción coincida con la del pipeline:** `python -m src.models.predecir` da la
    del pipeline para cualquier corte, y en los ejemplos la app enseña ✅ si coincide.
  - **Que la normalización sea la misma:** está comprobado por test.
  - **Que los errores se gestionen bien:** hay un mensaje claro para cada fallo de la lista de
    `app/README.md`.
- **Antes de la defensa:** abrir la URL para despertar la app.

---

## 8. Las preguntas del profesor, respondidas

**¿Qué error es más grave, un falso positivo o un falso negativo? ¿Cambia según el uso?**
- Aquí el falso positivo, porque podría llevar a relajar un tratamiento o la vigilancia. Por eso
  el umbral exige especificidad ≥ 0,80.
- Sí cambia según el uso. Si el modelo sirviera para preseleccionar candidatas a un ensayo de
  tratamientos alternativos para las que probablemente no responderán, preocuparía más pasar por
  alto a esas pacientes, y el umbral se elegiría de otra forma. Por eso el umbral es una decisión
  explícita y no un 0,5 automático.

**¿La red aprende biología tumoral o diferencias entre cohortes?**
- No se puede descartar del todo. Hay tres pistas:
  1. el AUC por cohorte (si solo funciona en una, sospechoso);
  2. la comparación con "solo cohorte" (0,55): hay que superarla con claridad;
  3. la auditoría de intensidades por cohorte.
- Hay además motivos para sospechar: las cohortes difieren en escáneres, tiempos de adquisición,
  criterios de HR/HER2 y en cómo se eligieron los cortes (Duke usa *bounding box*). El modelo de
  Fridman da 0,54 en Duke.

**¿Cómo afecta tener diez observaciones correlacionadas por paciente?**
- Los cortes de una paciente no son independientes: son ~1.100 pacientes, no ~11.000 casos.
- Consecuencias, y qué hacemos con cada una:
  - la partición tiene que ser por paciente, o hay fuga;
  - las métricas y sus intervalos se calculan por paciente (el bootstrap remuestrea pacientes);
  - hay que agregar los cortes;
  - la red puede memorizar a una paciente con sus 10 cortes, de ahí el aumento de datos.

**¿Está calibrada la probabilidad?** Ver §5.5. Está calibrada con Platt fuera de fold, se mide con
el Brier y la gráfica por deciles, y se comprueba en test.

**¿Qué evidencia faltaría para estudiar la utilidad clínica, la equidad y la generalización?**
- Validación externa prospectiva en otros hospitales y escáneres.
- Más pacientes de test, para estrechar los intervalos.
- Resultados por subgrupos (subtipo, edad, raza), sin tratar raza como variable causal.
- Calibración en la población de destino.
- Comparación con lo que ya saben los clínicos (subtipo, tamaño).
- Un estudio de impacto: ¿cambia decisiones?, ¿mejora resultados?
- Un buen resultado retrospectivo no demuestra utilidad clínica.

**¿Tendríais todos los datos del CSV en la práctica clínica?**
- No todos. La edad, la menopausia y el subtipo (HR, HER2) sí se conocen antes del tratamiento,
  por la historia y la biopsia. La cohorte, los parámetros de adquisición, la máscara y el volumen
  segmentado son del dataset o salen de la imagen, no son información clínica. Y no hay análisis
  de sangre, peso, IMC ni tabaco.
- Por eso las variables del CSV solo sirven para comparar (R0 y el médico de §2.5) y no entran
  en la CNN ni en la app. El médico usa solo las que un clínico tendría de verdad.

**Si es diagnóstico por imagen, ¿qué peso tiene la imagen?**
- Se mide de dos formas (§2.5, D25):
  1. **CNN contra el médico sin imagen** (edad, menopausia, HR, HER2: 0,69 en train). Si la CNN
     no lo supera, la imagen aporta menos que la biopsia, y es una conclusión honesta.
  2. **(CNN + médico) − médico**, con su intervalo por bootstrap pareado: lo que la imagen
     **añade** a lo que ya sabe el médico. Es la respuesta más directa a la pregunta.
- En la bibliografía pasa lo mismo: en Joo et al. la clínica sola ya daba 0,83 y la imagen añadía
  hasta 0,89.

**¿Aporta algo predecir pCR = 1?**
- Por sí sola no cambia nada: la paciente se opera igual y la pCR solo la confirma la anatomía
  patológica.
- Su posible valor es de investigación: estratificar el riesgo, enriquecer ensayos o identificar
  pronto a las que probablemente no responderán (usos del artículo de Fridman).

**¿Qué F1 es la óptima?** Ver §5.4: ninguna sola. Hay que dar las dos clases y la macro.

**¿Qué umbral estáis tomando?** Ver §5.4.

**¿Por qué cortes 2D y no el volumen 3D?**
- El dataset docente da ~10 cortes por paciente en PNG, y una CNN 2D es lo que pide el
  enunciado.
- El 3D aprovecharía mejor la forma del tumor, pero necesita más memoria y más datos. Queda como
  limitación y trabajo futuro.

---

## 9. Qué queda y en qué orden

Todo se ejecuta en el **PC de la universidad** (tiene las imágenes y la GPU), desde la carpeta del
repo con el entorno `cancer` activado. Después de cada paso, `git add`, `git commit` y `git push`.

**Qué está comprobado ya.** Desde la nube no hay acceso al dataset. Por eso el 18 de octubre todo
el flujo se probó de principio a fin con **imágenes sintéticas** con un realce plantado para
distinguir las clases:
- el plan completo (4 validaciones cruzadas);
- el cierre del modelo, que reproduce las predicciones del entrenamiento con una diferencia de
  1e-17;
- la evaluación de test con su bloqueo;
- el modelo del médico (§2.5), este con los datos clínicos reales de train, y su comparación con
  la CNN de imágenes sintéticas;
- la predicción por línea de comandos;
- la app en un navegador: un ejemplo coincide con el pipeline y una fase que falta da error.

Además hay más de 120 tests automáticos. Esa prueba destapó y corrigió un fallo real: el umbral quedaba
pegado al borde de su intervalo (§5.4). Lo que falta es ejecutarlo con los datos reales: los
números que salgan serán los de verdad.

1. **Poner al día** (5 min):
   ```bash
   git pull && pip install -r requirements.txt && ruff check . && pytest -q
   python -m src.training.historico          # R1 y R2 entran en el histórico (cierra la duda 4)
   python -m src.training.graficas runs/*    # sus curvas
   python -m src.models.historial --figuras runs/20261007-211625_cnn_v1_completo runs/20261007-213119_cnn_v1_completo_ponderada
   ```
2. **Auditoría** (unos minutos): `python -m src.data.auditoria` → `docs/resultados/1_auditoria/`.
   **Hecha el 20 de octubre.** Hay que repetirla una vez para la figura nueva de orientación
   (`orientacion.png`, D29), cuando no esté entrenando nada.
3. **Plan de experimentos** (una noche): `python -m src.training.plan`. Antes,
   `python -m src.training.plan --modo rapido` para comprobar en minutos que todo funciona.
   `python -m src.training.plan --ver` dice qué falta. Si se corta, se vuelve a lanzar.
   - **Primera ronda (cnn_v2–v4) hecha el 20 de octubre** (§3.8).
   - **Segunda ronda (cnn_v5–v8, §3.9):** se lanza con el mismo comando, una sola vez, y que siga
     aunque se cierre VS Code: `nohup python -m src.training.plan > plan.log 2>&1 &`. Se salta lo
     ya hecho. cnn_v5–v7 hechas el 21 de octubre; cnn_v8 se cayó por la tarjeta gráfica y se
     relanzó tras reiniciar el PC.
   - **Tercera ronda (cnn_v10, §3.10):** el mismo comando, cuando termine cnn_v8.
   - **Para mirar pronto si va bien** (§4.8), en cuanto haya unas épocas del fold 0:
     `python -m src.training.comparar cnn_v3 cnn_v10`.
   - **Probar algo en un solo fold o con otros valores** (§4.9): editar la sección `experimento` de
     `config.json` y lanzar `nohup python -m src.training.experimento > experimento.log 2>&1 &`.
   - **Mientras entrena, no lanzar nada que use la tarjeta**, y vigilar solo con
     `grep -a "Paso\|Fold\|época" plan.log | tail -3`.
   - **Si se cae con «unspecified launch failure»,** la tarjeta se ha quedado colgada: hay que
     reiniciar el PC. Si se repite, bajar `num_workers` de 4 a 2 en `config.json` (modo `completo`).
4. **Pasarle los resultados a Claude:** `python -m src.utils.para_claude` crea en la carpeta del
   repo un zip pequeño (`para_claude_<fecha>.zip`) que se envía por el chat.
   - Lleva el histórico, el resumen y las gráficas de cada ejecución, los informes de
     `docs/resultados/` y el estado de Git.
   - No lleva imágenes ni pesos. Claude no tiene las imágenes ni le hacen falta: entrenar se hace
     en el PC de la uni, y para leer resultados basta con estos ficheros (D27).
   - Con eso Claude escribe la lectura y la decisión de cada ficha y actualiza este documento con
     los resultados reales.
5. **Cerrar el modelo:** `python -m src.evaluation.cerrar_modelo` → `models/final/` y
   `docs/resultados/2_cierre/`. Después:
   - `python -m src.evaluation.modelo_clinico` (un minuto): comparación con el médico y modelo
     combinado (§2.5) → `docs/resultados/3_clinico/` y `models/clinico/`. **Tiene que ir antes de
     test**, que si no se niega a empezar.
   - `python -m app.preparar_ejemplos`.
6. **Desplegar la app** siguiendo `app/README.md`, y hacer la lista de pruebas válidas e
   inválidas.
7. **Test, una sola vez:** `python -m src.evaluation.evaluar_test` → `docs/resultados/4_test/`,
   con la CNN, el médico y CNN + médico en las mismas pacientes.
8. **Informe:** decisiones (este documento), métricas, matriz de confusión, limitaciones, uso de
   GPU (memoria, tiempo por época y total, batch: están en `resultados.json` y en el histórico) y
   las preguntas de la §8.
9. **Diapositivas**, exactamente 5 y lo último:
   1. problema clínico y pCR;
   2. datos, partición por paciente y clases (auditoría);
   3. arquitectura final con tamaños y parámetros (el diagrama de `docs/figuras/`);
   4. entrenamiento, decisiones y resultados internos (curvas, tabla de versiones, umbral);
   5. app, demo, limitaciones y conclusiones.
