# Experimentos del modelo final

Una **copia exacta** de las 6 ejecuciones con las que se construyó el modelo final, más el modelo
final y el modelo clínico de comparación. Los entrenamientos se hacen en `runs/`, que se queda en
local; aquí se copian sin cambiar nada.

## Las 6 ejecuciones

Cada una es la validación cruzada de una versión en los 5 folds, en modo `fijo`: 6 épocas y se
usa la última, sin elegirla con la validación. Se lanzaron con `python -m src.training.experimento`
(la sección `experimento` de `configs/config.json`).

| Versión | Ejecución | AUC val (5 folds) | Train − val | Parámetros |
|---|---|---|---|---|
| cnn_v3 | `20261028-195129_cv_cnn_v3_fijo` | 0,549 ± 0,038 | 0,117 | 97.809 |
| cnn_v10 | `20261028-200725_cv_cnn_v10_fijo` | 0,547 ± 0,036 | 0,092 | 97.809 |
| cnn_v6 | `20261028-202321_cv_cnn_v6_fijo` | 0,565 ± 0,040 | 0,106 | 101.970 |
| cnn_v7 | `20261028-203931_cv_cnn_v7_fijo` | 0,552 ± 0,049 | 0,099 | 97.809 |
| cnn_v5 | `20261028-205529_cv_cnn_v5_fijo` | 0,542 ± 0,055 | 0,119 | 97.809 |
| cnn_v4 | `20261028-211140_cv_cnn_v4_fijo` | 0,549 ± 0,049 | 0,111 | 98.097 |

Qué hay en cada carpeta:

| Fichero | Qué tiene |
|---|---|
| `config_usada.json` | la configuración con la que se lanzó: versión, modo, pérdida, folds y cambios |
| `fold_k/resultados.json` | la definición de la red y su receta de entrenamiento, la configuración completa del modo, la máquina, las métricas de cada época y las de la época usada |
| `fold_k/historial.csv` | las métricas por época, en tabla |
| `fold_k/mejor.pt` | los pesos de ese fold (época 6) |
| `fold_k/predicciones_val*.csv` | la probabilidad de cada corte y de cada paciente de validación |
| `fold_k/graficas/` | `curvas.png` (pérdida y AUC por época, entrenamiento y validación) y `matriz_confusion.png` |
| `resumen.json`, `resumen.csv` | media ± desviación de los 5 folds |
| `predicciones_oof*.csv` | todas las pacientes de train, cada una predicha por el fold en el que estaba en validación |
| `graficas/` | `curvas_folds.png` (los 5 folds superpuestos) y `matriz_confusion_oof.png` |

## El modelo final: `modelo_final/`

- `manifiesto.json`: todo lo necesario para reconstruirlo: la arquitectura y la receta de cada
  versión, los 30 ficheros de pesos con su sha256, la calibración de Platt (a = 0,187,
  b = −0,688), la agregación (media de los cortes de cada paciente), el umbral (0,307) y las
  métricas internas. Huella del modelo: `86efe7b4b112`.
- `<versión>/fold_k.pt`: los pesos de las 30 CNN. Son los de `fold_k/mejor.pt` de las ejecuciones
  de arriba, guardados solos.

Lo carga `src/models/final.py`, que comprueba el sha256 de cada fichero antes de usarlo. Lo usan
la evaluación de test, `src.models.predecir` y la app web (exportado a ONNX).

## El modelo clínico: `modelo_clinico/`

`modelo_clinico.json`: la regresión logística del «médico» (edad, menopausia, HR y HER2, sin
imagen), en JSON y sin pickle. Solo sirve de comparación: no entra en la CNN ni en la app.
Resultados en [`docs/resultados/3_clinico/`](../docs/resultados/3_clinico/).
