# Registro del proyecto

Documento vivo: estado actual, hoja de ruta, criterios del profesor, decisiones de proceso,
referencias y dudas abiertas. Se actualiza en cada sesión de trabajo. Servirá de base para el
informe y las diapositivas.

La historia de la red (una ficha por arquitectura) está en [`ARQUITECTURAS.md`](ARQUITECTURAS.md).
Qué va en cada documento: [D14](#d14--mapa-de-la-documentación-y-trazabilidad-de-las-arquitecturas-2026-10-15).

---

## Estado actual

**En una frase (2026-10-21, noche):** la segunda ronda (cnn_v5–v8) no mejora a cnn_v3, que sigue
siendo la candidata. La red no mira el tumor y reconoce el hospital, en parte por cómo orienta cada
uno la mama. cnn_v10 (giros de 90°) se está entrenando, y se pregunta al profesor por usar la caja del
tumor al entrenar (D29).

**Qué tenemos**
- **Las redes:** cnn_v1, v2, v4, v5, v6, v7 y v8 descartadas; **cnn_v3 candidata** (0,576 ± 0,037,
  sin sobreajuste); cnn_v10 en prueba (D29). Historia completa en
  [`ARQUITECTURAS.md`](ARQUITECTURAS.md).
- **Lo que enseñó la primera ronda** (R4): la red apenas generaliza, en parte reconoce el hospital
  (su puntuación distingue I-SPY2 con AUC 0,66) y no repite el subtipo.
- **Lo que enseñó la segunda** (R5):
  - dentro de cada hospital, la red casi acierta por azar (0,51–0,58);
  - su puntuación no tiene que ver con el tamaño del tumor;
  - reconoce I-SPY1 por la orientación de la mama.
- **Todo el camino, hecho en código y probado con imágenes inventadas:** revisar los datos →
  entrenar y comparar versiones → cerrar el modelo → compararlo con el médico → test una vez →
  app.
- **El médico sin imagen** (D25): AUC 0,689 en train. Es el listón que la CNN tiene que superar.
- **La app**, a falta del modelo final y de publicarla.
- Las decisiones, explicadas desde cero en [`COMO_SE_HIZO.md`](COMO_SE_HIZO.md). Desde el
  2026-10-18 las toma Claude y Judith las revisa (D15).

**Siguiente paso:** en el PC de la uni, cuando termine cnn_v10 (paso 8 del plan):
1. copiar los cambios de R5 y D32, comprobarlos y subirlos;
2. repetir la auditoría para la figura de orientación;
3. pasar a Claude el zip de `para_claude`. Con cnn_v10 se decide si se cierra el modelo con cnn_v3.

Además, Judith pregunta al profesor si se puede usar la caja del tumor solo al entrenar (duda 7).

**Pasos hasta la entrega** (comandos explicados en
[`COMO_SE_HIZO.md` §9](COMO_SE_HIZO.md#9-qué-queda-y-en-qué-orden))
1. PC de la uni: `git pull`, `pytest -q`, y R1/R2 al histórico (`python -m src.training.historico`,
   `python -m src.training.graficas runs/*`).
2. Auditoría: `python -m src.data.auditoria` → `docs/resultados/1_auditoria/`. ✅ 2026-10-20.
3. Plan de experimentos, de noche: `python -m src.training.plan`. ✅ Primera ronda 2026-10-20;
   segunda ronda 2026-10-21 (cnn_v5–v8, R5); tercera ronda (cnn_v10) en curso.
4. Pasar a Claude los resultados: `python -m src.utils.para_claude` y enviar el zip por el chat
   (D27). Con eso escribe las lecturas y decisiones de las fichas.
5. Cierre: `python -m src.evaluation.cerrar_modelo`, después `python -m src.evaluation.modelo_clinico`
   (antes de test, D25) y `python -m app.preparar_ejemplos`.
6. Desplegar la app (`app/README.md`).
7. Test, una sola vez: `python -m src.evaluation.evaluar_test`.
8. Informe y 5 diapositivas.

Estado de cada punto de la rúbrica:
[Criterios de evaluación](#criterios-de-evaluación-del-profesor-correo-del-2026-10-14).

**Historial de lo hecho**
- Repositorio con estructura, material del profesor (sin imágenes), entorno compartido, modos
  `rapido`/`completo`, detección de dispositivo, tests y CI.
- Portátil configurado (2026-10-07): Miniconda, entorno `cancer` con Python 3.12.15, torch
  2.14.0+cpu, 5 pacientes descargadas, 11 tests OK. Entorno aislado con `PYTHONNOUSERSITE=1`.
- PC Universidad en marcha (2026-10-07): R1 y R2 se entrenaron allí (RX 6700 XT, ROCm).
- Tubería de entrenamiento (`src/training/entrenar.py`), CNN configurable (`src/models/cnn.py`)
  con sus versiones en `src/models/versiones.py`, y comprobador de arquitecturas
  (`src/models/comprobar.py`).
- **cnn_v1 diseñada** (D10): 23.761 parámetros. Probada en modo rápido en CPU: ~12 s por época
  con 500 cortes, así que una época completa (~11.000 cortes) tardaría ~5 min en CPU.
- **Primer entrenamiento completo de cnn_v1** en el PC de la universidad (R1): funciona en la GPU
  (ROCm), 27 s por época. AUC por paciente máximo 0,585; prácticamente al nivel del azar.
- **cnn_v1 con pérdida ponderada** (R2): AUC máximo 0,572. Igual que R1: **cnn_v1 descartada**.
- **cnn_v2 definida** (D11): cnn_v1 + un cuarto bloque de 128 filtros; 97.809 parámetros. Sin entrenar.
- (2026-10-14) **Correo del profesor** con lo que se evalúa, resultados razonables, bibliografía y
  una rúbrica de 10 puntos: resumido en [Criterios de evaluación](#criterios-de-evaluación-del-profesor-correo-del-2026-10-14).
- (2026-10-14) **Referencias sin imagen** con los metadatos de train (R0): la cohorte sola da AUC
  ≈ 0,55 y el subtipo solo ≈ 0,66.
- (2026-10-14) **Tubería ampliada** (D12): AUC de entrenamiento por época, predicciones guardadas,
  matriz de confusión y métricas por clase, validación cruzada de 5 folds.
- (2026-10-14) **Histórico de ejecuciones y gráficas aparte** en `runs/` (D13).
- (2026-10-14) **Artículos de Fridman et al. leídos** (Scientific Data y preprint): ver
  [Referencias](#referencias).
- (2026-10-15) **Historial de arquitecturas** en [`ARQUITECTURAS.md`](ARQUITECTURAS.md), con
  trazabilidad comprobada por tests y huella de cada ejecución (D14). D10, D11, R1 y R2 viven ahora
  en las fichas de cnn_v1 y cnn_v2.
- (2026-10-18) **Modo de trabajo nuevo** (D15) y **decisiones pendientes cerradas** (D16).
- (2026-10-18) **cnn_v3** (aumento de datos) y **cnn_v4** (realce explícito) definidas, con su
  ficha (D18, D19), y **plan de experimentos** reanudable para la GPU (D17).
- (2026-10-18) **Auditoría de datos automatizada** (D20), **cierre del modelo** (D21),
  **evaluación de test bloqueada a una vez** (D22) y **app Streamlit** (D23). Inicialización
  justificada (D24).
- (2026-10-18) [`COMO_SE_HIZO.md`](COMO_SE_HIZO.md): todas las decisiones explicadas desde cero,
  con alternativas y respuestas para la defensa.
- (2026-10-18) **Modelo clínico de comparación, "el médico"** (D25): edad, menopausia, HR y HER2,
  sin imagen. AUC 0,689 ± 0,024 en train (R3). La CNN y la app siguen siendo solo imagen.
- (2026-10-20) **Limpieza de carpetas** (D26): en `docs/` quedan solo los tres documentos para
  leer; todo lo que generan los scripts va a `docs/resultados/`, una carpeta por paso. Fuera
  `notebooks/` (vacía) y la guía de instalación de una sola vez del PC de la uni.
- (2026-10-21) **Primera ronda de entrenamiento con datos reales** (R4): cnn_v2–v4 empatan en
  ~0,57–0,59; candidata cnn_v3. **Segunda ronda definida** (D28): cnn_v5 (intensidad relativa al
  PRE), cnn_v6 (pooling por atención), cnn_v7 (dropout y weight decay) y cnn_v8 (las tres).
- (2026-10-20) **El repositorio del proyecto es `judithsalas/breastdcedl`** (en el PC de la uni,
  `~/Escritorio/JUDITH/breastdcedl`, donde están las imágenes). `caso1_breast-cancer` deja de usarse.
  Pasar resultados a Claude sin imágenes: `python -m src.utils.para_claude` (D27).
- (2026-10-21) **Segunda ronda entrenada** (cnn_v5–v8, R5): ninguna mejora clara; sigue cnn_v3. cnn_v8 se cayó
  porque la tarjeta gráfica se quedó colgada; hubo que reiniciar el PC con el botón. **Tercera
  ronda definida** (D29): cnn_v10 (giros de 90°), figura de orientación en la auditoría y cnn_v9
  reservada a la espera del profesor. **Comparar el fold 0 con la referencia** mientras entrena
  (D30). **Parámetros en `config.json`** y pruebas de un fold con `src.training.experimento` (D31).

---

## Hoja de ruta

Basada en el plan de trabajo del enunciado (§13) y la guía, con fases añadidas por nosotras.

| Fase | Contenido | Estado |
|---|---|---|
| F0 | Configuración: repo, entorno en las dos máquinas, descarga de datos | hecho (portátil y uni) |
| F1 | **Auditoría y visualización de datos** (añadida): pacientes, clases, cohortes, canales, rangos, ejemplos PRE/EARLY/LATE y realce, comprobación PRE < EARLY | código hecho (D20); falta ejecutarlo en la uni |
| F2 | Tubería de entrenamiento: Dataset/DataLoader, bucle de entrenamiento, modos, evaluación por paciente, registro automático de cada ejecución (config, máquina, tiempos, métricas). Ampliada (D12, D13): AUC de entrenamiento, predicciones, matriz de confusión, validación cruzada, histórico y gráficas | hecho |
| F3 | **Diseño de la CNN** e iteración: baselines, pérdida normal vs ponderada (`pos_weight`), sobreajuste, descartar redes que no mejoren en 5–10 épocas | cnn_v2–v4 definidas y plan listo (D17); falta entrenar |
| F4 | Cerrar el modelo con validación interna: validación cruzada por paciente, método de agregación y umbral (nunca con test) | código hecho (D21) |
| F5 | Evaluación final en test, **una sola vez**: métricas por paciente y matriz de confusión. Dibujo de la red. Pesos en `models/final/` | código hecho y bloqueado a una vez (D22) |
| F6 | Informe: decisiones, métricas, matriz de confusión, limitaciones, uso de GPU, preguntas de discusión | pendiente |
| F7 | Aplicación web desplegada con URL, pruebas con entradas válidas e inválidas | app hecha (D23); falta desplegar |
| F8 | Presentación: **exactamente 5 diapositivas** (lo último) | pendiente |
| F9 | Defensa: 5 muestras privadas en la app | pendiente |

Requisitos de la app (enunciado §9, guía C6), para no olvidarlos en F7: cargar las tres fases
de un corte o elegir un ejemplo; mostrar PRE, EARLY, LATE y el mapa de realce; validar formato
(PNG 256×256 en grises, fases ausentes o duplicadas, ficheros corruptos, valores no finitos);
mostrar probabilidad, clase, umbral, versión/checksum del modelo, dispositivo, tiempo de
respuesta, parámetros de entrenamiento y métricas internas; aviso de uso educativo sin validez
clínica; `model.eval()` sin gradientes; sin rutas arbitrarias ni ejecución de contenido subido;
límite de tamaño y formato; normalización idéntica a la del entrenamiento.

---

## Criterios de evaluación del profesor (correo del 2026-10-14)

Resumen del correo con sugerencias para completar el caso. El correo y sus dos figuras no están
en el repo (los tiene Judith). Es fuente de verdad junto al enunciado y la guía (ver `CLAUDE.md`).

**Qué se evalúa.** Una CNN propia, entrenada desde cero, que prediga pCR con las tres fases DCE
(PRE, EARLY, LATE). La unidad de evaluación es la paciente (media de sus cortes), con el umbral
elegido en validación y test evaluado una sola vez, al final. Las variables de `patients.csv`
pueden servir **de comparación, no de modelo principal**.

**Resultados razonables** (AUC por paciente en un conjunto de prueba independiente):

| Estudio | Qué usa | AUC |
|---|---|---|
| Fridman et al. 2025 (BreastDCEDL) | mismo test que el nuestro, solo imagen | 0,72 (precision 0,70; sensibilidad 0,27; especificidad 0,95) |
| Braman et al. 2017 | radiómica DCE intra y peritumoral, test de 39 pacientes | 0,74 |
| Cain et al. 2019 | radiómica DCE, solo triple negativo y HER2+, un centro | 0,71 (IC95 0,58–0,83) |
| Liu et al. 2019 | radiómica T2 + difusión + T1 con contraste, cuatro hospitales | 0,79; 0,86 con clínica |
| Joo et al. 2021 | deep learning con imagen y clínica, un centro | 0,89; solo clínica 0,83 |
| Fridman et al. 2025 | solo I-SPY2, con clínica | 0,85; subgrupo HR+/HER2− (40 pacientes) 0,94 |

- Con imagen pretratamiento y test independiente, lo publicado está entre **0,70 y 0,80**. Las
  cifras de 0,85–0,94 usan clínica, una sola cohorte o subgrupos pequeños.
- Con cortes 2D y tres cohortes, **un AUC de 0,60 a 0,70 en test es un buen trabajo**.
- Si aparece 0,90 en validación, buscar una fuga antes de celebrarlo.
- Con 176 pacientes de test, el IC del AUC es de unos **±0,08**: 0,02 de diferencia entre dos
  modelos no significa nada.

**Figura "MAL / BIEN" del correo.** Modelo sobreentrenado: la pérdida de validación se dispara,
el AUC de entrenamiento llega a 1 y el de validación no sube (0,95 train / 0,59 validación / 0,56
test: diferencia entre entrenamiento y test de 0,38). Modelo bien entrenado, 5 folds: las curvas de
entrenamiento y validación bajan y suben juntas (0,74 / 0,66 / 0,64: diferencia de 0,10).

**Preguntas que plantea** (para el informe y la defensa):
- ¿Tendremos de verdad todos los datos del CSV en la práctica clínica?
- Si es diagnóstico por imagen, ¿qué peso tiene la imagen?
- ¿Hemos mirado el artículo? (ver [Referencias](#referencias))
- ¿Aporta realmente algo predecir pCR = 1?
- ¿Qué F1 es la óptima? ¿Precision y recall?
- ¿Qué umbral estamos tomando?

**Rúbrica: qué tiene un análisis de 10.** Lista de comprobación del trabajo; el estado se
actualiza en cada sesión.

| # | Criterio | Qué pide | Estado (2026-10-14) | Fase |
|---|---|---|---|---|
| 1 | Datos entendidos | Tres fases y realce mirados antes de programar; desbalance (30 % pCR) y cohortes descritos | **Hecho** (2026-10-20): auditoría con las 1.097 pacientes en `docs/resultados/1_auditoria/` | F1 |
| 2 | Partición por paciente | Ninguna paciente en dos conjuntos; validación cruzada de 5 folds con media ± desviación | **Hecho** para cnn_v2–v8 (R4, R5), con comparación pareada en las mismas pacientes | F2–F4 |
| 3 | Arquitectura propia justificada | Capas, tamaños y parámetros explicados; desde cero; las fases no son RGB | Hecho para v1–v4 (fichas en `ARQUITECTURAS.md`); inicialización justificada (D24) | F3 |
| 4 | Entrenamiento reproducible | Hiperparámetros, semilla, criterio de parada y época elegida documentados | Hecho: se guardan en `resultados.json` y en el histórico (D13); cada versión, con su historia en `ARQUITECTURAS.md` (D14) | F2 |
| 5 | Sobreajuste controlado y demostrado | Curvas de train y validación juntas por época; diferencia de AUC en la época elegida; sobreentrenados descartados | **Demostrado** con cnn_v2–v8: curvas de los 5 folds en las fichas; cnn_v2 (0,162) y cnn_v8 (0,169) descartadas por sobreajuste; comparador del fold 0 con pérdida y AUC (D30) | F3–F4 |
| 6 | Pérdida normal frente a ponderada | Comparadas, explicando qué le pasa a la sensibilidad | **Hecho** con cnn_v1 y con la candidata cnn_v3: el AUC no cambia (0,576 frente a 0,574) y la sensibilidad con umbral 0,5 pasa de 0,02 a 0,56 | F3–F4 |
| 7 | Umbral y agregación justificados | Elegidos con validación y con criterio clínico explícito (qué error es más grave) | Código listo (D21): media de cortes y umbral con especificidad ≥ 0,80 | F4 |
| 8 | Test completo, una sola vez | Matriz de confusión, AUC con IC, precision, recall y F1 de cada clase, F1 macro, comparación con «siempre no pCR» | Código listo y bloqueado a una vez (D22), con IC por bootstrap | F5 |
| 9 | Más allá de la cifra global | Resultados por cohorte, calibración y casos fallados mirados con sus imágenes | Código listo: por cohorte, calibración y casos fallados con imágenes (D21, D22). Sesgo por hospital medido en cada validación cruzada y explicado en parte por la orientación de la mama (R5, D29) | F4–F6 |
| 10 | Comparación y honestidad | Bibliografía, limitaciones y una app que reproduce exactamente el pipeline | Bibliografía recogida; app hecha con el mismo código que el pipeline (D23); comparación con un médico sin imagen y con CNN + médico, en train y en la evaluación única de test (D25, R3) | F4–F7 |

---

## Decisiones

Formato: qué se decide, por qué y alternativas descartadas.

### D1 · Un entorno conda común, PyTorch instalado aparte en cada máquina (2026-10-07)
- `environment.yml` (Python 3.12) y `requirements.txt` con versiones fijas, compartidos.
- PyTorch 2.14.0 / torchvision 0.29.0 en ambas máquinas: variante CPU en el portátil, ROCm 7.14
  en la universidad.
- **Por qué:** un entorno conda no se puede copiar entre sistemas operativos; lo que sí se
  comparte es la receta. PyTorch tiene un paquete distinto por hardware, así que no puede ir en
  un `requirements.txt` común.
- **Descartado:** el `requirements.txt` de la plantilla (es un `pip freeze` de Linux con
  paquetes `nvidia-*` que fallan en Windows e incluye librerías que no usamos).

### D2 · El portátil trabaja en CPU (2026-10-07)
- **Por qué:** su GPU AMD es de gama baja; ROCm en Windows no la soporta y DirectML obligaría a
  otra versión de PyTorch distinta de la de la universidad.
- Los entrenamientos reales se hacen en el PC de la universidad. GPU en la nube (Colab) queda
  como plan B, a valorar en F3 según los tiempos reales.

### D3 · Modos de ejecución `rapido` y `completo` (2026-10-07)
- El modo indica el tipo de ejecución, no la máquina; el dispositivo se detecta solo.
- `rapido`: 40 pacientes, 2 épocas, batch 16; sus métricas no cuentan.
- `completo`: todas las pacientes, hasta 60 épocas, parada si el AUC por paciente no mejora en
  10, **batch fijo 32** para que todos los entrenamientos sean comparables.
- Valores provisionales: se ajustarán al medir tiempos y memoria reales.

### D4 · Estructura del repositorio a partir de la plantilla UAX (2026-10-07)
- Se toma de la plantilla: carpetas `src/`, `tests/`, `notebooks/`, `models/`, `docs/`,
  configuración versionada, CI con ruff + pytest y commits convencionales.
- (2026-10-20, D26) `notebooks/` se quitó: siempre estuvo vacía y la auditoría es un script.
- `CLAUDE.md` (contexto para Claude) separado de este registro.

### D5 · Se versiona el material del profesor, no las imágenes (2026-10-07)
- En Git: guía, scripts, metadatos, enunciado y figuras de `breastdcedl/` (~9 MB).
- Fuera de Git: `breastdcedl/dataset/` (38.109 PNG). Se descarga con `descargar_datos.py`, que
  salta los ficheros que ya existen.
- Ventaja extra: los tests de metadatos (sin solape de pacientes, etiqueta constante) corren
  en GitHub Actions sin necesitar las imágenes.

### D6 · Solo se versionan los pesos del modelo final (2026-10-07)
- `models/final/*.pt` se sube a GitHub (pocos MB); el resto de checkpoints, no.

### D7 · Aplazar la auditoría de datos (F1) para probar una primera red en clase (2026-10-07)
- Se adelanta la tubería de entrenamiento (F2) y la primera red (F3) para aprovechar la GPU de
  la clase.
- **Condición:** F1 se hace antes de sacar conclusiones de cualquier resultado. Los primeros
  entrenamientos sirven para comprobar que todo funciona y como referencia, no para decidir.

### D8 · Decisiones provisionales de la tubería de entrenamiento (2026-10-07)
Pendientes de revisar por Judith; cada una se puede cambiar.
- **Entrada:** el tensor (3, 256, 256) en [0, 1] tal como lo da `utils_caso` (PNG / 255), sin
  más normalización. La app deberá usar exactamente lo mismo.
- **Sin aumento de datos** por ahora: es una decisión a tomar en F3 (siempre sobre el tensor
  completo, nunca canal a canal).
- **Optimizador Adam**, lr 1e-3 (valor por defecto razonable para empezar).
- **Métrica de selección y de parada:** AUC por paciente en validación, porque no depende del
  umbral y no la engaña el desbalance como la accuracy.
- **Agregación por paciente:** media de probabilidades; **umbral 0,5**. Ambos provisionales: se
  eligen en F4 con validación interna.
- **Pérdida:** normal por defecto; `--ponderada` usa `pos_weight = N0/N1` calculado sobre las
  filas de entrenamiento (2,40 con todos los datos del fold 0).
- **Modo rápido:** usa solo pacientes ya descargadas, para poder probar con una descarga parcial.
  Sus métricas no significan nada (p. ej. 10 pacientes de validación con 1 pCR).

### D9 · Entorno aislado de paquetes externos (2026-10-07)
- En el portátil, pip encontró paquetes en la carpeta de usuario de Python
  (`AppData\Roaming\Python`) y no los instaló en el entorno. Se fija `PYTHONNOUSERSITE=1` en el
  entorno para que solo use lo suyo y sea reproducible. Igual en la universidad.
- En el portátil se instaló Miniconda (no Miniforge); los dos sirven igual. Al crear el entorno
  hubo que aceptar los términos de uso de los canales de Anaconda.

### D10 · Primera arquitectura: cnn_v1 (2026-10-07)
Movida el 2026-10-15 a la ficha de cnn_v1 en [`ARQUITECTURAS.md`](ARQUITECTURAS.md#cnn_v1) (D14).
Se mantiene el número para que las referencias antiguas sigan valiendo.

### D11 · Segunda arquitectura: cnn_v2 (2026-10-07)
Movida el 2026-10-15 a la ficha de cnn_v2 en [`ARQUITECTURAS.md`](ARQUITECTURAS.md#cnn_v2) (D14).

### D12 · Tubería ampliada para la rúbrica del profesor (2026-10-14)
Judith aprobó hacer estos cambios el 2026-10-14. Los valores marcados como provisionales esperan
su revisión. Sin ellos, las ejecuciones nuevas no servirían para los puntos 2, 5, 7, 8 y 9 de la
rúbrica y habría que repetirlas.
- **AUC por paciente de entrenamiento en cada época** (punto 5). Se mide en modo evaluación sobre
  una **muestra fija y estratificada de 220 pacientes** de entrenamiento (provisional;
  `pacientes_auc_train` en `src/config.py`, desde D31 en `config.json`; `--pacientes-auc-train`; 0 = todas, que es lo que usa
  el modo rápido).
  - **Por qué 220:** es el tamaño de un fold, así que el AUC de entrenamiento y el de validación
    tienen un ruido parecido y su diferencia es comparable. Además alarga poco cada época: unos
    2.200 cortes más, frente a 8.700 si se usan todas.
  - **Descartado:** usar todas las pacientes de entrenamiento (exacto, pero ~80 % más de tiempo por
    época) o mirar solo la pérdida de entrenamiento (no da AUC).
- **Predicciones de la época elegida guardadas** (puntos 7 y 9): `predicciones_val.csv` (cada
  corte) y `predicciones_val_pacientes.csv` (cada paciente: media, mediana, máximo y cohorte). Con
  ellas se eligen agregación y umbral, se estudia la calibración y se miran las cohortes sin
  reentrenar.
- **Matriz de confusión y métricas por clase de la época elegida** (punto 8 y petición de Judith):
  precision, recall y F1 de pCR y de no pCR, y F1 macro, en `resultados.json` y en figura.
  - Se calculan en `src/evaluation/metricas.py` a partir de la matriz de `uc.evaluar_por_paciente`
    (no se toca el material del profesor).
  - La F1 se calcula como 2·VP / (2·VP + FP + FN): vale 0, y no "indefinida", si el modelo nunca
    predice esa clase.
  - Incluye las referencias triviales "siempre no pCR" y "siempre pCR".
- **Validación cruzada de 5 folds** (punto 2): `python -m src.training.validacion_cruzada`.
  - Entrena la misma versión en los 5 folds con la misma configuración y semilla.
  - Resume cada fold y da media ± desviación (muestral, ddof = 1).
  - Junta las predicciones **fuera de fold** de las 1.097 pacientes, con su matriz, el AUC por
    cohorte y las referencias triviales. Es la base de F4.
- **Limitación conocida:** en cada fold, la época elegida se escoge mirando ese mismo fold, así
  que su AUC de validación es algo optimista. Lo corrige test, evaluado una sola vez.
- **Organización del código:** lo que hacía `main()` pasa a `ejecutar()`, que reutiliza la
  validación cruzada. La inicialización y el orden de los lotes no cambian respecto a R1 y R2.
- **Coste estimado** para cnn_v1 en la GPU de la uni: ~27 s por época (R1) más ~20 % por el AUC de
  entrenamiento; 5 folds × 20–25 épocas ≈ 1–1,5 h por configuración. Hay que confirmarlo con la
  primera ejecución.

### D13 · Histórico de ejecuciones y gráficas aparte (2026-10-14)
Petición de Judith: en `runs/` hace falta un histórico y las gráficas aparte.
- **`runs/historico.csv`**: una fila por ejecución.
  - Columnas: fecha, tipo, modo, modelo, parámetros, pérdida, folds, época elegida, épocas, AUC de
    validación (± desviación en validación cruzada), AUC de entrenamiento, diferencia,
    sensibilidad, especificidad, F1 por clase y macro, umbral, agregación, batch, lr, semilla,
    tiempos, dispositivo, versión de torch y nota.
  - Una validación cruzada ocupa una sola fila, con la media de sus folds.
  - Se actualiza solo al terminar cada ejecución. `python -m src.training.historico` lo rehace
    mirando `runs/`, también con las ejecuciones antiguas (R1, R2): su época elegida se saca del
    historial y el AUC de entrenamiento queda vacío.
- **El histórico se versiona** (provisional, a confirmar por Judith). Es la única excepción dentro
  de `runs/` (`.gitignore`: `runs/*` y `!runs/historico.csv`).
  - **Por qué:** los entrenamientos reales se hacen en la uni y el informe se puede escribir en
    cualquier máquina; lo que no está en GitHub no existe en la otra.
  - Al actualizar se recalculan las filas de las carpetas de esta máquina y se conservan las demás,
    así que no se pierden filas al juntar las dos máquinas.
  - Si las dos máquinas suben cambios a la vez, git puede dar conflicto en el CSV: se dejan las
    filas de ambas y se vuelve a lanzar `python -m src.training.historico`.
  - **Para no versionarlo:** volver a poner `runs/` en `.gitignore`.
- **Gráficas aparte, en `graficas/` dentro de cada ejecución:**
  - `curvas.png`: pérdida y AUC por paciente por época, entrenamiento y validación juntos, con la
    época elegida y la diferencia de AUC en esa época.
  - `matriz_confusion.png`: matriz de la época elegida.
  - En una validación cruzada: `curvas_folds.png` (todos los folds superpuestos, como la figura
    "BIEN" del profesor) y `matriz_confusion_oof.png`.
  - Se redibujan al acabar cada época, así que se puede seguir un entrenamiento largo abriendo el
    PNG.
  - `python -m src.training.graficas runs/<carpeta>` las rehace, también para R1 y R2.
  - Las que vayan al informe se copian a `docs/figuras/`, porque `runs/` no se versiona.
- **Formato de las figuras:** coma decimal; azul = entrenamiento, naranja = validación, como en las
  figuras del profesor. Paleta comprobada para daltonismo.

### D14 · Mapa de la documentación y trazabilidad de las arquitecturas (2026-10-15)
Petición de Judith: comprobar que la estructura y la trazabilidad están bien y tener un historial
de las arquitecturas que se prueban, para ver con cuál se queda y por qué se da cada paso.

**Diagnóstico antes del cambio:**
- La historia de las redes estaba repartida en cuatro sitios: comentarios de `versiones.py`,
  D10–D11, R1–R2 y la tabla del README. Ningún hilo unía versión → hipótesis → resultado →
  decisión → siguiente versión.
- Este registro mezclaba decisiones de proceso y de arquitectura y ya tenía ~460 líneas.
- Nada obligaba a documentar una versión nueva ni impedía cambiar una ya entrenada: era solo una
  norma.
- Las hipótesis no tenían "qué esperamos ver" escrito antes de entrenar.
- La evidencia (gráficas) se quedaba en `runs/`, que no se versiona.

**Cambio:**
- **`docs/ARQUITECTURAS.md`** (nuevo): una ficha por versión con origen, definición, hipótesis,
  qué esperamos ver, ejecuciones, resultados, lectura, decisión y qué aprendimos. Incluye una
  tabla resumen, la genealogía y una tabla de ejecuciones (las tres automáticas), una plantilla, el
  criterio para comparar versiones y la sección del modelo final.
- D10, D11, R1 y R2 se mueven a las fichas de cnn_v1 y cnn_v2 sin perder contenido. Aquí quedan
  punteros con el mismo número.
- **`versiones.py`** declara el `ORIGEN` y el `ESTADO` de cada versión. Los tests obligan a que
  toda versión tenga origen, estado y ficha, y a que la genealogía no tenga ciclos.
- **Huella:** cada ejecución guarda en el histórico una huella de 8 caracteres de la definición
  que se entrenó. Si no coincide con la actual, las tablas la marcan con ⚠. Así la regla "una
  versión entrenada no se modifica" se comprueba.
- **`python -m src.models.historial`** regenera las partes automáticas. Se lanza solo tras cada
  entrenamiento y con `python -m src.training.historico`.
- **`--figuras runs/<ejecución>`** copia sus gráficas a `docs/figuras/ejecuciones/`, que sí se
  versiona, para que la evidencia viaje con el repo.

**Mapa de la documentación.** Cada cosa tiene un único sitio.

| Dónde | Qué guarda | Cuándo se toca |
|---|---|---|
| `docs/REGISTRO.md` | Estado, hoja de ruta, criterios del profesor y rúbrica, decisiones de proceso (D…), resultados que no son de una red (R0, R3), referencias y dudas | Cada sesión |
| `docs/ARQUITECTURAS.md` | Historia de la red: una ficha por versión, tablas automáticas y modelo final | Antes y después de entrenar cada versión |
| `src/models/versiones.py` | Definición exacta de cada versión, su `ORIGEN` y su `ESTADO` | Al crear una versión o cambiar su estado |
| `runs/historico.csv` | Una fila por ejecución, con métricas, huella, máquina y versión de torch | Automático |
| `runs/<ejecución>/` | Todo el detalle: configuración, épocas, predicciones, pesos y gráficas (no versionado) | Automático |
| `docs/figuras/` | Diagramas de las redes y gráficas que sirven de evidencia (copiadas de `runs/`) | Al cerrar una versión |
| `docs/resultados/` | Lo que generan los scripts, una carpeta por paso: `1_auditoria`, `2_cierre`, `3_clinico`, `4_test` (D26) | Automático; no se edita a mano |
| `CLAUDE.md` | Reglas de trabajo y del caso | Cuando cambian las reglas |
| `README.md` | Cómo instalar y usar el repo | Cuando cambian los comandos |

**Cadena de trazabilidad de un resultado:**
1. La ficha da la decisión y por qué, y nombra la ejecución.
2. La fila del histórico da sus métricas, la huella, la máquina y la versión de torch.
3. `runs/<ejecución>/` tiene la configuración y las épocas.
4. `versiones.py` tiene la definición; la huella ✓ confirma que es la misma que se entrenó.
5. `docs/figuras/ejecuciones/<ejecución>/` guarda las gráficas.

**Pendiente de Judith** (resuelto el 2026-10-18 en D16): fijar el criterio para comparar versiones (propuesta en
`ARQUITECTURAS.md`) y revisar "qué esperamos ver" de cnn_v2 antes de entrenarla.

### D15 · Modo de trabajo: Claude decide, Judith revisa y aprende (2026-10-18)
- Petición de Judith: avanzar rápido; que Claude tome las decisiones que considere más acertadas
  y las deje explicadas en un documento para poder consultarlas, entenderlas y aprender.
- Desde hoy Claude decide lo técnico, arquitecturas incluidas (desde cnn_v3), y lo implementa.
  Judith ejecuta en la uni, revisa, puede revertir cualquier decisión, sube a GitHub y defiende.
- Cada decisión se explica desde cero en [`COMO_SE_HIZO.md`](COMO_SE_HIZO.md), con alternativas y
  respuestas para la defensa. El enunciado exige que la alumna entienda y defienda sus
  decisiones: el documento existe para eso.

### D16 · Se cierran las decisiones pendientes (2026-10-18)
- **AUC de entrenamiento sobre 220 pacientes** (D12): se mantiene (ruido comparable al de un fold
  de validación y ~20 % más de tiempo por época, frente a ~80 % con todas).
- **`runs/historico.csv` se versiona** (D13): se mantiene.
- **Criterio para comparar versiones:** fijado e implementado (`elegir_mejor` en
  `src/models/historial.py`; texto en `ARQUITECTURAS.md`):
  - media del AUC de 5 folds, con pérdida normal y huella ✓;
  - empate si la diferencia es menor que una desviación;
  - entre empatadas, primero las que no sobreajustan (diferencia ≤ 0,15), luego la de menos
    parámetros, luego la más antigua.
- **"Qué esperamos ver" de cnn_v2:** aceptado tal como estaba.

### D17 · Plan de experimentos con lanzador reanudable (2026-10-18)
- `python -m src.training.plan` lanza en orden, como validaciones cruzadas de 5 folds:
  1. cnn_v2 normal;
  2. cnn_v3 normal;
  3. cnn_v4 normal;
  4. la mejor según el criterio, con pérdida ponderada.
- Es reanudable: salta lo que ya está en el histórico.
- Estimación (sin medir aún): 1,5–2 h por paso en la RX 6700 XT, una noche en total.
- Por qué este orden y por qué la ponderada solo para la ganadora: `COMO_SE_HIZO.md` §4.6.

### D18 · Receta de entrenamiento por versión y aumento de datos geométrico (2026-10-18)
- Una versión puede declarar `entrenamiento = {"aumento": ...}`. Así queda en su huella: si
  cambia la receta, es otra versión. Las versiones antiguas no la declaran y su huella no cambia.
- `src/data/aumento.py`:
  - volteo horizontal, giro de ±10°, desplazamiento de ±4 % y escala de 0,95–1,05;
  - igual en las 3 fases y solo en entrenamiento;
  - sin cambios de intensidad ni volteo vertical.
- Lo usa cnn_v3.

### D19 · Realce explícito dentro de la red (2026-10-18)
- La CNN acepta `entrada = "fases+realce"`: recibe las 3 fases y añade dentro EARLY − PRE y
  LATE − EARLY.
- Al calcularse dentro, la app no puede preprocesar distinto. Lo usa cnn_v4.

### D20 · Auditoría de datos automatizada (2026-10-18)
- `python -m src.data.auditoria` genera `AUDITORIA.md`, `auditoria.json` y figuras (desde D26,
  en `docs/resultados/1_auditoria/`). Incluye:
  - recuentos por conjunto, cohorte, subtipo y fold;
  - referencias sin imagen (R0, ahora reproducibles en el repo);
  - integridad de los PNG;
  - PRE < EARLY y lavado;
  - intensidades por cohorte;
  - realce por clase, con el AUC de indicadores hechos a mano;
  - ejemplos visuales.
- Solo se cruzan imagen y etiqueta en train. Cubre el punto 1 de la rúbrica.

### D21 · Cierre del modelo (2026-10-18)
`python -m src.evaluation.cerrar_modelo`, todo con predicciones fuera de fold:
- **Versión:** la ganadora del criterio. Pérdida ponderada solo si supera a la normal en más de
  una desviación.
- **Modelo final:** el conjunto de los 5 modelos de su validación cruzada, con la media de los
  logits.
- **Calibración de Platt** ajustada fuera de fold.
- **Agregación:** media de los cortes.
- **Umbral:** máxima sensibilidad con especificidad ≥ 0,80.
- **Exportación** a `models/final/`: pesos por fold y `manifiesto.json` con sha256, calibración,
  umbral y métricas internas.
- **Informe** `CIERRE_MODELO.md` (desde D26, en `docs/resultados/2_cierre/`).
- Comprueba que los modelos recargados reproducen las predicciones del entrenamiento.
- Por qué cada cosa: `COMO_SE_HIZO.md` §5.

### D22 · Evaluación de test única (2026-10-18)
- `python -m src.evaluation.evaluar_test` usa el modelo final tal cual. Calcula:
  - matriz de confusión;
  - AUC, sensibilidad y especificidad con IC del 95 % por bootstrap de pacientes;
  - precision, recall y F1 por clase, y F1 macro;
  - las referencias triviales;
  - resultados por cohorte, calibración y casos fallados con sus imágenes.
- **Se niega a repetir** si ya existe `resultado_test.json` (desde D26, en `docs/resultados/4_test/`). Con `--repetir "motivo"`
  repite, pero el motivo queda escrito en el informe.

### D23 · App web: Streamlit en Streamlit Community Cloud (2026-10-18)
- `app/app.py` usa exactamente el código del pipeline:
  - `src/data/entrada.py`, que valida y lee los ficheros con la misma normalización (comprobada
    por test);
  - `src/models/final.py`, el modelo final.
- Valida: 3 ficheros, fases por sufijo (ni ausentes ni duplicadas), mismo corte, PNG real, gris
  de 8 bits, 256×256, ≤ 2 MB y valores finitos.
- Muestra todo lo que pide el enunciado.
- Ejemplos de train con la probabilidad del pipeline para comprobar que coincide.
- `python -m src.models.predecir` da la predicción del pipeline para cualquier corte.
- Despliegue gratuito desde el repo de GitHub (`app/README.md`). Alternativas descartadas:
  `COMO_SE_HIZO.md` §7.

### D24 · Inicialización de los pesos: la de PyTorch por defecto, justificada (2026-10-18)
- Convoluciones y capas lineales: uniforme de Kaiming (He). BatchNorm: peso 1 y sesgo 0.
- Con BatchNorm detrás de cada convolución, la escala inicial de los pesos apenas influye (BN
  renormaliza cada capa), y Kaiming está pensada para ReLU. Una inicialización a mano no aportaría.
- Cubre la parte de "inicialización" del enunciado (§7) y de la rúbrica (punto 3).

### D25 · Modelo clínico de comparación: "el médico", sin imagen (2026-10-18)
Pregunta de Judith: ¿la red se entrena solo con las fotos o también con los datos clínicos? La
idea era un modelo "como un médico" (edad, menopausia, HR/HER2, análisis de sangre…) solo para
comparar.
- **La CNN y la app siguen siendo solo imagen** (enunciado §7; correo del profesor: las variables
  de `patients.csv` son "de comparación, no de modelo principal").
- **Modelo aparte** `src/evaluation/modelo_clinico.py`, una regresión logística con lo que se
  sabe antes de la quimioterapia: edad, menopausia, HR y HER2.
  - No hay análisis de sangre, peso, IMC ni tabaco en el dataset.
  - Fuera: raza (guía), cohorte, adquisición y máscara.
  - El volumen tumoral sale de la imagen y en Duke tiene otra escala: solo como variante, sin
    test.
- **Datos que faltan:** se rellenan con lo aprendido en los folds de entrenamiento.
  - Las variables, con la mediana.
  - La menopausia (falta entera en I-SPY1), con su probabilidad según la edad.
  - Sin indicador de "falta", que delataría la cohorte.
- **Validación:** los mismos 5 folds que la CNN y el mismo criterio de umbral.
- **Con la CNN cerrada:**
  - diferencia de AUC CNN − médico, con bootstrap pareado de pacientes;
  - combinado CNN + médico, por apilado con las predicciones fuera de fold, para medir lo que
    añade la imagen;
  - AUC de la CNN dentro de cada subtipo.
- **Test:** médico y combinado quedan congelados en `models/clinico/modelo_clinico.json` (se
  versiona: excepción en `.gitignore`). Se evalúan en la misma evaluación única que la CNN.
  `evaluar_test` se niega a empezar si falta el fichero o si se hizo con otra CNN (huella).
- Cambios de apoyo:
  - `cerrar_modelo` guarda `oof_pacientes.csv` y la huella en `cierre.json`, junto a su informe;
  - `metricas.py` gana `diferencia_auc` (bootstrap pareado) y `auc_por_rangos`, igual que
    sklearn pero más rápido para el bootstrap (comprobado por test).
- Orden nuevo: plan → cerrar_modelo → **modelo_clinico** → evaluar_test.
- Por qué cada cosa: `COMO_SE_HIZO.md` §2.5. Resultados en R3.
- **Transparencia sobre test.** Al probar el flujo de principio a fin, `evaluar_test` se ejecutó
  en la copia con imágenes sintéticas. Esa copia lleva los metadatos reales, así que el médico se
  evaluó una vez contra las etiquetas reales de test.
  - El modelo ya estaba fijado antes: variables, imputación y regularización.
  - No se ha cambiado nada después, ni se cambiará por lo visto.
  - La cifra oficial es la de la evaluación única de Judith. El médico es determinista, así que
    saldrá igual.

### D26 · Limpieza de carpetas (2026-10-20)
Judith no veía clara la distribución de carpetas. Criterio: **lo que se lee, a la vista; lo que
genera el código, junto y en orden**.
- **`docs/` solo tiene los tres documentos para leer:** `REGISTRO.md` (estado y diario),
  `COMO_SE_HIZO.md` (decisiones explicadas) y `ARQUITECTURAS.md` (historia de la red).
- **`docs/resultados/`** guarda todo lo que generan los scripts. Hay una carpeta por paso,
  numerada en el orden en que se ejecutan, con el informe y sus figuras al lado:
  `1_auditoria/`, `2_cierre/`, `3_clinico/` y `4_test/`.
  - Antes había cinco informes sueltos en `docs/`, cada uno con su carpeta de figuras aparte.
  - La ruta está en un solo sitio en el código: `RESULTADOS`, en `src/data/caso.py`.
- **`docs/figuras/`** sigue igual: diagramas de las redes y gráficas copiadas de los
  entrenamientos.
- **Fuera** (sigue en el historial de Git, por si hiciera falta):
  - `notebooks/`, que siempre estuvo vacía: la auditoría es un script (D20).
  - `docs/PUESTA_EN_MARCHA_UNI.md`, instrucciones de una sola vez para pasar el PC de la uni de
    la carpeta antigua al repo clonado. Lo que sigue sirviendo (entorno, ROCm, descarga,
    comprobaciones) ya estaba en el README; se añadió allí una línea para mover las imágenes en
    vez de descargarlas otra vez.
- `src/` no se toca: ya va por pasos (data → models → training → evaluation) y moverlo cambiaría
  los imports sin ganar claridad.
- **README** con una tabla «Por dónde empezar» y el mapa de carpetas en palabras sencillas.
  «Estado actual» de este registro, reescrito en corto: una frase, qué tenemos, el siguiente
  paso y los pasos que quedan.
- `tests/test_estructura.py` comprueba que se mantiene el orden. En `docs/` solo pueden estar
  los tres documentos, `figuras/` y `resultados/`, y cada script tiene que escribir en su carpeta
  numerada.
- `limpieza.py` (de un solo uso, se borra solo) hace la misma reorganización en el repo de
  Judith: mueve los informes que ya hubiera generado y borra lo que sobra. Al final Judith
  prefirió sustituir el repo entero por una copia limpia (`git rm` de lo versionado y copiar la
  nueva), que deja intactos `.git/`, `runs/` y las imágenes.
- **`__init__.py` vacíos** (petición de Judith, el mismo día). Solo sirven para que Python trate
  cada carpeta como un paquete. El código que tenían pasa a módulos con nombre:
  - `src/data/__init__.py` → `src/data/caso.py`: rutas, acceso a `utils_caso` y nombres de las
    cohortes;
  - `src/models/__init__.py` → `src/models/construir.py`: `crear_modelo`, `receta`, `huella` y
    `cambios`.
  - Los imports dicen ahora de dónde viene cada cosa, por ejemplo
    `from src.models.construir import crear_modelo`.
  - Nada cambia en lo que hace el código: las huellas de las versiones son las mismas y los
    tests pasan igual. Un test comprueba que los `__init__.py` siguen vacíos.

### D27 · Pasar resultados a Claude sin las imágenes (2026-10-20)
- Claude trabaja en la nube sin `breastdcedl/dataset/`. Son ~38.000 PNG, y además el bucket del
  profesor está bloqueado desde la nube. No hacen falta: entrenar y evaluar se hace en el PC de la
  uni, y para leer resultados y decidir bastan ficheros pequeños.
- `python -m src.utils.para_claude` junta en `para_claude_<fecha>.zip` lo que hace falta:
  - `runs/historico.csv` y cada ejecución de `runs/` sin los pesos `.pt`;
  - `docs/resultados/`;
  - los manifiestos de `models/final/` y `models/clinico/`;
  - `estado_git.txt`, con los últimos commits y lo que está sin subir.
- Nunca mete nada de `breastdcedl/` ni pesos. El zip está en `.gitignore`.
- Si alguna vez Claude necesita ver imágenes concretas, bastan las figuras de la auditoría
  (`ejemplos.png`) o los PNG de unas pocas pacientes de train.
- Test: `tests/test_para_claude.py`.

### D28 · Segunda ronda de arquitecturas: cnn_v5 a cnn_v8 (2026-10-21)
Judith pidió subir el AUC hacia 0,70 sin sobreentrenar. Con los resultados de la primera ronda
(R4) hay tres problemas, y para cada uno una versión sobre la candidata cnn_v3:
- **cnn_v5 · intensidad relativa al PRE**, contra el atajo del hospital. Dentro de la red, las
  tres fases se dividen entre la intensidad media del PRE en el tejido del corte
  (`normalizar_por_pre` en `src/models/cnn.py`).
- **cnn_v6 · pooling por atención** (Ilse et al. 2018), contra la dilución del tumor pequeño en la
  media global. Media ponderada de las 16×16 posiciones con pesos aprendidos (`PoolingAtencion`);
  4.161 parámetros más. Los pesos se pueden dibujar para ver dónde mira la red.
- **cnn_v7 · regularización**, contra la memorización: dropout 0,3 en la cabeza y AdamW con weight
  decay 0,05 (sin decay en sesgos ni BatchNorm). La receta de entrenamiento admite ahora
  `optimizador` y `weight_decay`; v1–v4 siguen con Adam sin decay (sus huellas no cambian).
- **cnn_v8 · las tres juntas.** Rompe a propósito «una idea por versión»: se entrena a la vez que
  v5–v7, que dicen qué aporta cada pieza. Así basta una noche (~45 min por validación cruzada).
- **No se crece la red:** de v1 a v2 los parámetros se multiplicaron por 4 sin mejorar la
  validación.
- **No se usan las coordenadas del tumor** de `patients.csv` (`sraw`, `eraw`, `scol`, `ecol`) para
  guiar la atención. Podrían ayudar, pero el profesor pide que esas variables solo sirvan para
  comparar. Queda como pregunta para él si v6 no mejora.
- **Diagnóstico nuevo en cada validación cruzada:** «¿delata el hospital?», el AUC de la
  puntuación para distinguir cada cohorte del resto (`senal_de_hospital`, en `resumen.json` y en
  pantalla).
- El plan (D17) añade los pasos 4–7. La ponderada se hace solo con la ganadora (paso 8; si sigue
  siendo cnn_v3, ya está hecha).
- El motivo del criterio de comparación dice ahora qué desempate decidió.
- Por qué cada cosa, desde cero: `COMO_SE_HIZO.md` §3.8 y §3.9; hipótesis y «qué esperamos ver» de
  cada versión en sus fichas (escritas antes de entrenar).

### D29 · Tercera ronda: quitar la pista de la orientación (cnn_v10) y preguntar por la caja del tumor (2026-10-21)
cnn_v5–v7 no mejoran a cnn_v3 (R5). Al buscar por qué, con las predicciones fuera de fold de cnn_v3
y los metadatos (R5), el freno principal es que **la red no mira el tumor**:
- dentro de cada hospital acierta casi por azar;
- su puntuación no tiene que ver con el tamaño del tumor;
- y el tumor ocupa un 3–8 % de la imagen.

Además reconoce el hospital, y el mosaico de 10 pacientes por cohorte enseña una de las pistas:
**cada hospital orienta la mama a su manera** (I-SPY1 hacia abajo; I-SPY2 y Duke, a los lados).

Se decide:
- **cnn_v10 = cnn_v3 + giros de 90° al azar al entrenar** (`geometrico+giros90` en
  `src/data/aumento.py`). Con el volteo horizontal, las 8 orientaciones salen con la misma
  probabilidad y la orientación deja de servir de pista. Una idea, sobre la candidata. No cambia la
  red ni la app. Paso 8 del plan; la ponderada pasa a ser el paso 9.
- **La auditoría dibuja la orientación** (`orientacion.png`, `figura_orientacion`): es evidencia del
  sesgo entre hospitales para el informe (rúbrica, punto 9). Con semilla 0 salen las mismas 30
  pacientes que en el mosaico que se miró.
- **Deja de valer «no voltear en vertical»** (D18) para cnn_v10: la pared torácica no está siempre
  en el mismo lado. cnn_v3 no cambia (su huella es la misma).
- **cnn_v9 queda reservada** para la atención guiada por la caja del tumor:
  - al entrenar, una pérdida extra que castiga la atención fuera de la caja (`sraw`, `eraw`, `scol`,
    `ecol`);
  - al predecir no hace falta la caja.

  Es la única idea que ataca el freno principal, pero la caja sale de `patients.csv`, que no entra en
  la CNN (correo del profesor). Judith se lo pregunta antes (duda 7). Si dice que sí, antes de
  entrenar hay que comprobar sobre unos PNG que la caja cae en el tumor: las coordenadas están en el
  tamaño original de cada resonancia (`n_xy` de 128 a 512), no en 256.
- **Lo que no se hace:**
  - **Quitar el fondo o recortar a la mama:** el fondo vale 0 y no estorba; recortar cambia la escala
    de cada imagen.
  - **Ajustar brillo o contraste:** el brillo de cada hospital ya se probó en cnn_v5; el contraste
    por imagen rompe la comparación entre fases.
  - **Cambiar el criterio de comparación (D16)** después de ver resultados: si cnn_v10 empata con
    cnn_v3, sigue ganando cnn_v3.
- **Tests:** las huellas de cnn_v5–v8 quedan fijadas (ya se entrenaron con esas definiciones), y hay
  tests nuevos para los giros y para la figura.
- Por qué cada cosa, desde cero: `COMO_SE_HIZO.md` §3.10; hipótesis y «qué esperamos ver» en la ficha
  de cnn_v10.

### D30 · Comparar el fold 0 con la referencia mientras entrena (2026-10-21)
Judith pidió ver pronto, con dos líneas, si una versión nueva va bien, sin esperar a los 5 folds.
- **`python -m src.training.comparar <referencia> <nueva> [--fold k]`** (`src/training/comparar.py`):
  - cuatro paneles por época, con una línea por versión: AUC por paciente y pérdida, en validación y
    en entrenamiento (la curva ROC no sirve para seguir épocas; es del modelo cerrado);
  - imprime la época elegida y, en las pacientes del fold, el AUC por hospital y «¿delata el
    hospital?»;
  - guarda `graficas/comparacion_fold<k>_con_<referencia>.png` en la carpeta de la nueva, así que va en
    el zip de `para_claude`.
- **Busca sola** la última validación cruzada de cada versión con pérdida normal, o acepta rutas.
- **Funciona con un fold a medias** (el historial se guarda cada época) y **no carga torch**: se puede
  usar mientras entrena. Por eso `auc_por_cohorte` y `senal_de_hospital` pasan de
  `validacion_cruzada.py` a `src/evaluation/metricas.py`, sin cambiar lo que calculan.
- **El plan no cambia a «solo el fold 0»:** un fold baila ±0,04, más que las diferencias que se buscan;
  las decisiones siguen saliendo de los 5 folds (D16, rúbrica punto 2). El fold 0 sirve para parar
  pronto una versión que vaya claramente peor (regla de `CLAUDE.md`).
- Ya existían, y se redibujan cada época: `fold_0/graficas/curvas.png` (entrenamiento frente a
  validación, dos líneas) y `graficas/curvas_folds.png` (todos los folds).
- Por qué, desde cero: `COMO_SE_HIZO.md` §4.8. Test: `tests/test_comparar.py`.

### D31 · Un solo fichero de configuración, `config.json` (2026-10-21)
Judith pidió tener los parámetros en un JSON, como plantilla.
- **`config.json` en la raíz**, con cuatro secciones:
  - `experimento`: versión, modo, pérdida, folds, cambios sobre el modo y nota;
  - `modos`: `rapido` y `completo`, con los mismos valores que antes tenía `src/config.py`;
  - `evaluacion`: agregación, umbral durante el entrenamiento y especificidad mínima del umbral final;
  - `comparacion`: límite de sobreajuste (0,15).

  `_ayuda` explica cada parámetro.
- **`src/config.py` lo lee y lo comprueba:** si falta o sobra un parámetro, o el JSON está mal escrito,
  da un mensaje claro. `entrenar.py` (agregación y umbral), `umbral.py` (especificidad mínima) e
  `historial.py` (sobreajuste) toman los valores de ahí; ya no hay números sueltos en el código.
- **`python -m src.training.experimento [--config otro.json] [--ver]`** lanza la sección
  `experimento` (un fold, o los que se digan) y guarda `config_usada.json` en la carpeta de la
  ejecución. El plan no la usa: sigue entrenando los 5 folds con el modo estándar.
- **No va en el JSON lo que define una versión** (capas, dropout, aumento, optimizador, weight decay):
  sigue en `versiones.py`, con su huella. Cambiarlo es otra versión.
- **Protección en el criterio:** `validaciones_validas` solo cuenta validaciones cruzadas con el
  batch, el learning rate y la semilla de su modo (`es_estandar`). Así una prueba con otro learning
  rate no sustituye a la versión. Las épocas y la paciencia no están en el histórico y no se
  comprueban.
- **Probado:** tests nuevos (`test_config.py`, `test_experimento.py` y uno en `test_plan.py`), y una
  ejecución de prueba con imágenes sintéticas (cnn_v10, fold 0, learning rate cambiado). El
  histórico guarda el learning rate cambiado y el comparador la encuentra.
- Por qué, desde cero: `COMO_SE_HIZO.md` §4.9.

### D32 · Dos correcciones de las tablas automáticas (2026-10-21)
Salieron al meter los resultados de la segunda ronda.
- **Las pruebas de un fold salían en la columna «5 folds»** de la tabla de resumen de
  `ARQUITECTURAS.md`. Pasaba con cnn_v10, que llevaba solo el fold 0 terminado. Ahora la columna «5
  folds» solo recoge validaciones cruzadas con los 5 folds, y lo demás va a «1 fold» (`_cv_completa`
  en `historial.py`).
- **El histórico guardaba 4 cifras** y la tabla volvía a redondear: el 0,57449 de cnn_v5 salía 0,575.
  Ahora guarda 6 (`historico.py`). No cambia ninguna decisión.
- En el comparador (D30), el eje de la pérdida pasa a 3 decimales: con 2 se repetían las etiquetas.

---

## Resultados

Los resultados de cada arquitectura están en su ficha de [`ARQUITECTURAS.md`](ARQUITECTURAS.md)
(D14). Aquí quedan los que no son de una red concreta. Cada ejecución tiene su detalle en
`runs/<ejecución>/` (no versionado) y una fila en `runs/historico.csv` (D13).

### R0 · Referencias sin imagen, con los metadatos de train (2026-10-14)
Comparación permitida por el correo del profesor (las variables de `patients.csv` sirven para
comparar, no como modelo). Solo train; test no se ha mirado.
- **Método:** en cada fold, la probabilidad de una paciente es la tasa de pCR de su grupo
  calculada en los otros 4 folds; AUC por paciente en el fold, media de los 5.
- **Solo la cohorte:** AUC 0,549 de media (0,534 en el fold 0).
- **Solo el subtipo (HR/HER2):** AUC 0,657 de media (0,697 en el fold 0).
- **pCR por cohorte (train):** Duke 21 % (209 pacientes), I-SPY1 25 % (104), I-SPY2 32 % (784).
- **pCR por subtipo (train):** HER2+ 43 % (287), triple negativo 38 % (354), HR+/HER2− 14 % (453).
- **Folds:** equilibrados por cohorte (~42 Duke, ~21 I-SPY1 y ~157 I-SPY2 cada uno). Casi todas las
  pacientes tienen 10 cortes (mínimo 5).
- **Lectura:** cnn_v1 (0,585) está al nivel de "adivinar la cohorte" y por debajo de una variable
  clínica de tres categorías. Sirve para las preguntas del profesor sobre el peso de la imagen y
  sobre si la red aprende biología o diferencias entre cohortes. HR y HER2 se conocen antes del
  tratamiento (biopsia); la cohorte no es información clínica.
- Se calculó con un script provisional fuera del repo; desde D20 lo rehace `src/data/auditoria.py`.

### R3 · Modelo del médico en train, fuera de fold (2026-10-18)
`python -m src.evaluation.modelo_clinico` con los metadatos reales; detalle en
[`MODELO_CLINICO.md`](resultados/3_clinico/MODELO_CLINICO.md). Solo train.
- **Médico (edad, menopausia, HR, HER2):** AUC 0,689 ± 0,024 (entrenamiento 0,700; diferencia
  0,010).
  - Fuera de fold: 0,685 (IC 0,65–0,72).
  - Con especificidad ≥ 0,80: sensibilidad 0,40.
- **Médico + volumen** (variante, no va a test): 0,711 ± 0,030.
- **Razones de odds** (todo train, IC por bootstrap):
  - HER2+ 2,74 (2,06–3,84);
  - HR+ 0,30 (0,22–0,38);
  - edad 0,84 por cada 10 años (0,69–1,03);
  - menopausia 0,96 (0,62–1,45).
- **Por cohorte:** I-SPY1 0,80; Duke 0,72; I-SPY2 0,66.
- **Lectura:**
  - Supera a "solo subtipo" (0,66) sobre todo porque usa HR y HER2 por separado. El subtipo de
    tres categorías junta los HER2+ con y sin receptores hormonales.
  - Edad y menopausia aportan poco: sus intervalos cruzan el 1.
  - Es el listón que la CNN debe superar. Está en la franja de 0,60–0,70 que el profesor
    considera buen trabajo para la CNN, así que puede que la CNN sola no lo supere.
- Pendiente: la comparación con la CNN y el combinado, al volver a ejecutarlo tras
  `cerrar_modelo`.

### R4 · Primera ronda con datos reales: cnn_v2, v3 y v4 (2026-10-20)
Plan D17 en el PC de la uni, 5 folds cada una, ~45 min por validación cruzada. Detalle fold a
fold y curvas en las fichas de [`ARQUITECTURAS.md`](ARQUITECTURAS.md).

| Versión | AUC val (5 folds) | AUC train | Diferencia | Fuera de fold | Duke · I-SPY1 · I-SPY2 | ¿Delata I-SPY2? |
|---|---|---|---|---|---|---|
| cnn_v2 | 0,593 ± 0,032 | 0,755 | 0,162 | 0,595 | 0,59 · 0,49 · 0,60 | 0,58 |
| cnn_v3 | 0,576 ± 0,037 | 0,656 | 0,080 | 0,574 | 0,51 · 0,53 · 0,58 | 0,66 |
| cnn_v4 | 0,572 ± 0,044 | 0,698 | 0,126 | 0,547 | 0,51 · 0,54 · 0,55 | 0,61 |
| cnn_v3 ponderada | 0,574 ± 0,030 | 0,668 | 0,094 | 0,543 | 0,50 · 0,50 · 0,54 | 0,62 |

- **Gana cnn_v3** por el criterio: empata con cnn_v2, que sobreajusta (0,162 > 0,15); cnn_v4
  empata pero tiene más parámetros.
- **La red apenas generaliza:** el AUC de validación se mueve entre 0,50 y 0,60 desde la primera
  época y no sube; solo sube el de entrenamiento.
- **En parte reconoce el hospital** (columna «¿delata I-SPY2?»: 0,58–0,66). Dentro de Duke e
  I-SPY1, ~0,50–0,54.
- **No repite el subtipo** (AUC para HER2+ y HR+ ~0,45–0,50). Por subtipo (cnn_v3): HER2+ 0,61,
  triple negativo 0,58, HR+/HER2− 0,50.
- **Normal frente a ponderada (cnn_v3):** el mismo AUC; la sensibilidad con umbral 0,5 pasa de
  0,02 a 0,56 y la especificidad de 0,99 a 0,53.
- Las cifras por fold son la mejor época de cada fold sobre su validación, algo optimistas.
  Época a época, la validación ronda 0,53–0,58. Ver `COMO_SE_HIZO.md` §3.8.
- Calculado con el zip `para_claude_20261021-1210` (D27).

### R5 · Segunda ronda: cnn_v5 a cnn_v8, y por qué no mejoran (2026-10-21)
Calculado con el zip `para_claude_20261021-2051` (D27). Detalle fold a fold, curvas, lectura y
decisión en las fichas de [`ARQUITECTURAS.md`](ARQUITECTURAS.md).

| Versión | AUC val (5 folds) | AUC train | Diferencia | Fuera de fold | Duke · I-SPY1 · I-SPY2 | ¿Delata el hospital? Duke · I-SPY1 · I-SPY2 | Frente a cnn_v3 (pareado) |
|---|---|---|---|---|---|---|---|
| cnn_v3 (R4) | 0,576 ± 0,037 | 0,656 | 0,080 | 0,574 | 0,51 · 0,53 · 0,58 | 0,42 · 0,28 · 0,66 | — |
| cnn_v5 | 0,574 ± 0,035 | 0,678 | 0,104 | 0,566 | 0,52 · 0,42 · 0,59 | 0,41 · 0,41 · 0,61 | −0,009 (−0,046 a +0,026) |
| cnn_v6 | 0,591 ± 0,041 | 0,656 | 0,065 | 0,569 | 0,54 · 0,59 · 0,56 | 0,45 · 0,32 · 0,61 | −0,005 (−0,039 a +0,027) |
| cnn_v7 | 0,588 ± 0,042 | 0,691 | 0,103 | 0,557 | 0,53 · 0,57 · 0,55 | 0,42 · 0,40 · 0,61 | −0,017 (−0,053 a +0,019) |
| cnn_v8 | 0,593 ± 0,025 | 0,762 | **0,169** | 0,587 | 0,53 · 0,47 · 0,60 | 0,44 · 0,39 · 0,59 | +0,013 (−0,029 a +0,053) |

- **Ninguna mejora clara:** todas a menos de una desviación de cnn_v3, y todos los intervalos
  pareados cruzan el 0. **Sigue ganando cnn_v3**; v5–v8, descartadas.
  - cnn_v8 empata, pero sobreajusta (0,169 > 0,15).
  - cnn_v6 empata, pero tiene más parámetros.
  - cnn_v5 y cnn_v7 empatan con los mismos parámetros, y gana la más antigua.
- **Ninguna repite el subtipo** (AUC 0,47–0,51 para HER2+ y HR+). Por subtipo, HER2+ 0,57–0,62,
  triple negativo 0,56–0,60, HR+/HER2− 0,51–0,55.
- **Las versiones no coinciden entre sí:** Spearman de 0,33 a 0,62 entre las puntuaciones fuera de fold
  de cnn_v3, v5, v6, v7 y v8. Buena parte de lo que aprende cada una es ruido.
- **Ruido del propio entrenamiento:** el fold 0 de cnn_v8, entrenado dos veces con la misma semilla
  (antes y después de la caída), dio 0,558 y 0,583.
- **cnn_v8 se cayó** con `CUDA error: unspecified launch failure`: la tarjeta se quedó colgada.
  - Quedaron procesos en estado D que no se podían matar, y `rocm-smi` no leía la tarjeta.
  - Se arregló reiniciando el PC con el botón: la cuenta `aulas` no puede reiniciar con `sudo`.
  - No es del código: v5–v7 usan las mismas piezas.
  - Causas probables: la RX 6700 XT no tiene soporte oficial en ROCm (va con
    `HSA_OVERRIDE_GFX_VERSION`), el PC estaba sin memoria y llevaba 14 días sin reiniciar.
- **Análisis de por qué no mejoran**, con las predicciones fuera de fold de cnn_v3 (R4) y los
  metadatos (solo train):
  - **AUC dentro de cada hospital:** Duke 0,505, I-SPY1 0,527, I-SPY2 0,579. La media ponderada
    (0,560) queda 0,014 por debajo del global (0,574): reconocer el hospital aporta poco al AUC.
  - **¿Delata el hospital por algo más que la pCR?** Se simuló una puntuación con el mismo AUC que
    solo dependiera de la pCR: daría 0,49 ± 0,02 (Duke), 0,50 ± 0,03 (I-SPY1) y 0,51 ± 0,02
    (I-SPY2). cnn_v3 da 0,42, 0,28 y 0,66, así que sí reconoce el hospital, sobre todo I-SPY1.
  - **Tamaño del tumor:**
    - la puntuación casi no tiene relación con el tamaño de la caja del tumor (Spearman −0,03 con el
      área en mm²);
    - en cambio, el tamaño solo da AUC 0,51 en Duke, 0,56 en I-SPY1 y 0,59 en I-SPY2;
    - la caja ocupa de mediana un 3,3 % de la imagen en Duke, 7,2 % en I-SPY1 y 7,9 % en I-SPY2.
  - **Épocas:** la mejor época de cnn_v3 por fold fue la 1, 6, 5, 16 y 6. El AUC de entrenamiento
    llega a 0,71–0,87; el de validación en la última época queda en 0,46–0,61.
  - **Orientación** (mosaico de 10 pacientes por cohorte, D29): en I-SPY1 la mama apunta hacia abajo
    en las 10; en I-SPY2 y Duke, a la izquierda o a la derecha. En Duke y alguna de I-SPY2 se ven el
    corazón y los pulmones.
- **Lectura:**
  - El freno principal es que la red no encuentra el tumor.
  - Reconocer el hospital es un síntoma, y en parte se explica por la orientación.
  - Normalizar la intensidad (cnn_v5) no podía quitar eso.
- **Siguientes pasos:** cnn_v10, y cnn_v9 si el profesor deja usar la caja (D29).

### R1, R2 · cnn_v1 con pérdida normal y ponderada (2026-10-07)
Movidos el 2026-10-15 a la ficha de cnn_v1 en [`ARQUITECTURAS.md`](ARQUITECTURAS.md#cnn_v1),
con la comparación entre pérdida normal y ponderada (D14).

---

## Referencias

### Fridman et al. 2026, *Scientific Data* 13:264 — el artículo que cita el enunciado
"BreastDCEDL: A standardized deep learning-ready breast DCE-MRI dataset of 2,070 patients",
[doi:10.1038/s41597-026-06589-6](https://doi.org/10.1038/s41597-026-06589-6). Leído el
2026-10-14 (PDF de Judith, no está en el repo).
- **Es un *Data Descriptor*:** describe el dataset y **no trae resultados de predicción de pCR**.
  Las cifras del correo (AUC 0,72, etc.) salen del preprint (abajo).
- **Particiones originales:**
  - train 1.532 pacientes (1.099 con pCR, 29,3 %);
  - validación 268 (177 con pCR, 29,9 %);
  - test 270 (176 con pCR, 30,1 %).

  Nuestro test es su test (176). La validación privada del profesor es su validación (177).
  Nuestro train tiene 1.097 porque se excluyeron 2 pacientes Duke (`excluded_patients.csv`).
- **Diferencias entre cohortes** (para discutir el sesgo, preguntas del correo y D5 de la guía):
  - Resolución temporal de 60 s en Duke y 120 s en I-SPY. En Duke se tomó como EARLY la segunda
    adquisición postcontraste para igualar tiempos.
  - Duke viene de 1 centro e I-SPY2 de más de 22. Escáneres GE, Siemens y Philips en distinta
    proporción.
  - **No se armonizaron intensidades** entre centros.
  - Umbrales distintos para HR (≥ 1 % en I-SPY2, Allred > 3 en Duke, ≥ 10 % en I-SPY1) y para HER2
    (FISH > 2,2 en Duke, ≥ 2,0 en I-SPY).
  - Duke tiene *bounding box* en lugar de máscara (por eso sus cortes se eligieron distinto).
- **Usos que proponen para una predicción de pCR pretratamiento:** estratificar el riesgo
  (intensificar o desescalar), enriquecer ensayos clínicos e identificar pronto a las pacientes que
  probablemente no responderán. Sirve para la pregunta del profesor sobre si predecir pCR = 1
  aporta algo, y para elegir el criterio del umbral.
- **Limitación que reconocen:** la RM durante o después de la quimioterapia predice mejor que la
  pretratamiento, y combinar imagen con variables clínicas mejora el resultado.
- La licencia del artículo es CC BY-NC-ND 4.0. Es la del texto, no la de los datos, así que no
  resuelve la duda 1.

### Fridman et al. 2025, preprint arXiv:2506.12190 (v3) — de aquí salen las cifras del correo
"BreastDCEDL: A Comprehensive Breast Cancer DCE-MRI Dataset and Transformer Implementation for
Treatment Response Prediction", <https://arxiv.org/abs/2506.12190>.
- **Modelo:** un Vision Transformer sobre cortes 2D, con las tres fases fusionadas **como RGB**
  y pasadas a 8 bits "por compatibilidad con modelos preentrenados". **No es comparable en
  método:** aquí está prohibido preentrenar y tratar las fases como colores.
- **No describen** la agregación por paciente, el umbral ni el entrenamiento (optimizador,
  épocas, pérdida). Tampoco dan F1.
- **Tabla 2, "Overall Test Set", N = 177:** AUC 0,72; accuracy 0,75; sensibilidad 0,27;
  especificidad 0,95; precision 0,70; VPN 0,75.
  - Por cohorte: I-SPY2 0,78 (99), I-SPY1 0,68 (35), **Duke 0,54** (43). Su modelo está al nivel
    del azar en Duke.
  - Por subtipo: HER2+ 0,68; triple negativo 0,63; HR+/HER2− 0,64.
  - Con variables clínicas, solo I-SPY2: 0,85; HR+/HER2− (40 pacientes): 0,94.
- **Lo que da esa tabla con nuestra prevalencia de test** (53 pCR / 123 no pCR):
  - Su F1 de pCR es 0,39, **menor** que la de decir siempre "pCR" (0,46).
  - Su F1 macro es 0,61, frente a 0,41 de decir siempre "no pCR".

  La F1 de la clase pCR, sola, no sirve como criterio. Ver la pregunta del profesor sobre qué F1
  es la óptima.
- **Inconsistencia:** la tabla dice N = 177 en test, pero su propio texto (y el artículo
  publicado) dicen test 176 y validación 177. Ver duda 6.

---

## Dudas y contradicciones abiertas

1. **Licencia de los datos.** `breastdcedl/LICENSE` (bucket del profesor) dice **CC BY 4.0**;
   la guía y el enunciado dicen **CC BY-NC 4.0**. Hasta aclararlo, aplicamos la más restrictiva
   (CC BY-NC 4.0: sin uso comercial). Conviene preguntar al profesor.
2. **`ACLARACION_CORTES.md` no existe en el bucket** aunque la guía la cita. Su contenido
   esencial está en la guía (sección B4): no usar `mask_start`/`mask_end` para filtrar cortes en
   I-SPY.
3. **Dos versiones de `GUIA.md`.** La del bucket es más antigua; la que tiene Judith (la que está
   en el repo) coincide con los recuentos por fold de `samples.csv`. Se usa la de Judith.
4. **Versión de torch en la universidad:** pendiente de confirmar que es 2.14.0 (podría ser
   2.14.1 si se instaló la última). Se cierra con la columna `torch` del histórico (D13) tras
   `python -m src.training.historico` en el PC de la uni, o con `maquina.torch` del
   `resultados.json` de R1.
5. **Licencia de nuestro código** (p. ej. Apache 2.0 como la plantilla, o MIT): pendiente de
   decidir antes de publicar.
6. **¿Sobre qué conjunto está la tabla del preprint?** Dice "test, N = 177", pero el test tiene 176
   pacientes con pCR (el nuestro) y la validación privada 177. Por cohorte da 99/43/35; nuestro test
   es 99/42/35. No cambia nada práctico: 0,72 es la referencia en una población equivalente. El
   correo dice "mismo test que el vuestro, 177 pacientes"; si sale en la defensa, conviene
   mencionar la discrepancia.
7. **¿Se puede usar la caja del tumor (`sraw`, `eraw`, `scol`, `ecol`) solo al entrenar?**
   (2026-10-21, D29). El correo dice que las variables de `patients.csv` no entran en la CNN. La
   caja no entraría al predecir: solo guiaría la atención durante el entrenamiento (cnn_v9). Judith
   se lo pregunta al profesor. Mientras no conteste, cnn_v9 no se define.
