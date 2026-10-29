# Validación interna de la calibración y del umbral

Generado por `python -m src.evaluation.validacion_interna` para el modelo `86efe7b4b112`. Solo train; test no se toca.
Por qué: D40 en [`REGISTRO.md`](../../REGISTRO.md) y `COMO_SE_HIZO.md` §5.8.

En el cierre, Platt y el umbral se eligieron con todas las pacientes de train (fuera de fold) y se midieron con esas mismas.
Aquí, cada fold recibe un Platt y un umbral elegidos sin él, como una paciente nueva.

## 1. ¿Sirve la probabilidad? Brier (1097 pacientes, 322 con pCR)

El Brier es el error cuadrático medio entre la probabilidad y lo que pasó (0 es perfecto). La referencia es decir a todas la tasa de pCR (29,4 %).

| Probabilidad | Brier |
|---|---|
| Constante: la tasa de pCR a todas | 0,2074 |
| Constante, con la tasa de los otros folds | 0,2077 |
| CNN sin calibrar | 0,2093 |
| CNN con Platt ajustado con las mismas pacientes (cifra del cierre) | 0,2068 |
| **CNN con Platt ajustado sin el fold** | **0,2079** |
| Médico (fuera de fold) | 0,1890 |

**Lectura:** la probabilidad de la CNN, calibrada sin el fold, no bate a la probabilidad constante: como probabilidad, dice tanto como la tasa de pCR. Lo que ordena (el AUC) es otra cosa, y se mide aparte.

| Fold | a | b | Umbral |
|---|---|---|---|
| 0 | 0,179 | -0,700 | 0,307 |
| 1 | 0,395 | -0,465 | 0,324 |
| 2 | 0,154 | -0,685 | 0,312 |
| 3 | 0,208 | -0,725 | 0,296 |
| 4 | 0,089 | -0,776 | 0,303 |
| todos (modelo final) | 0,187 | -0,688 | 0,307 |

## 2. El umbral (mayor sensibilidad con especificidad ≥ 0,80)

| Cómo se eligió | Sensibilidad | Especificidad | Precisión pCR | VP · FP · FN · VN |
|---|---|---|---|---|
| CNN, con las mismas pacientes (cifra del cierre) | 0,283 | 0,801 | 0,371 | 91 · 154 · 231 · 621 |
| **CNN, sin el fold que se mide** | 0,280 | 0,805 | 0,373 | 90 · 151 · 232 · 624 |
| Médico, con las mismas pacientes | 0,398 | 0,803 | 0,456 | 128 · 153 · 194 · 622 |
| Médico, sin el fold que se mide | 0,398 | 0,803 | 0,456 | 128 · 153 · 194 · 622 |

El criterio de especificidad ≥ 0,80 es una elección ilustrativa del trabajo (limitar los «pCR» equivocados), no un requisito clínico validado.

## 3. Por qué la cifra principal es la media de los folds

| AUC | Valor |
|---|---|
| Media de los 5 folds (no depende de la calibración) | 0,554 |
| Todas juntas, con un Platt para todos | 0,546 |
| Todas juntas, con un Platt por fold | 0,528 |

Con todas las pacientes juntas se comparan probabilidades de modelos distintos, y el resultado cambia según cómo se calibre cada uno.
La media de los folds solo compara pacientes predichas por el mismo modelo.

## 4. Dispersión de la probabilidad

Fuera de fold, cada paciente la predicen las redes de su fold (6, una por versión): la desviación del logit entre pacientes es 0,114, y el 90 % de las probabilidades está entre 0,261 y 0,324.
En la app y en test predicen todas (30). La media de más redes varía menos, así que en test la probabilidad puede salir más concentrada y con menos pacientes por encima del umbral. `evaluar_test` lo mide.
