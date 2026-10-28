# Cierre del modelo

Generado por `python -m src.evaluation.cerrar_modelo` el 2026-10-28 18:37.
Todo con predicciones **fuera de fold** de train; test no se ha usado. Explicación de cada
decisión: [`COMO_SE_HIZO.md`](../../COMO_SE_HIZO.md).

## 1. Qué modelo

- **Modelo:** conjunto de 6 CNN. Motivo: todas las versiones con validación cruzada completa que no sobreajustan (diferencia ≤ 0,15): cnn_v3, cnn_v4, cnn_v5, cnn_v6, cnn_v7, cnn_v10.
- **Pérdida:** normal. Motivo: las versiones se compararon con la pérdida normal; con cnn_v3, la ponderada da el mismo AUC (rúbrica, punto 6).

| Versión | AUC val (5 folds) | Train − val | Parámetros | Validación cruzada |
|---|---|---|---|---|
| cnn_v3 | 0,576 ± 0,037 | 0,080 | 97.809 | `20261020-192927_cv_cnn_v3_completo` |
| cnn_v4 | 0,572 ± 0,044 | 0,126 | 98.097 | `20261020-201130_cv_cnn_v4_completo` |
| cnn_v5 | 0,574 ± 0,035 | 0,104 | 97.809 | `20261021-135616_cv_cnn_v5_completo` |
| cnn_v6 | 0,591 ± 0,041 | 0,065 | 101.970 | `20261021-144306_cv_cnn_v6_completo` |
| cnn_v7 | 0,588 ± 0,042 | 0,103 | 97.809 | `20261021-152323_cv_cnn_v7_completo` |
| cnn_v10 | 0,581 ± 0,045 | 0,047 | 97.809 | `20261021-203548_cv_cnn_v10_completo` |
| ~~cnn_v2~~ | — | — | — | no entra: sobreajusta (diferencia 0,162 > 0,15) |
| ~~cnn_v8~~ | — | — | — | no entra: sobreajusta (diferencia 0,169 > 0,15) |

- **AUC en validación (5 folds):** 0,596 ± 0,040; en entrenamiento 0,668; diferencia media 0,088. AUC de entrenamiento y diferencia: media de las versiones del conjunto (cada una se midió en su validación cruzada).
- **Modelo final:** media de los logits de los 5 modelos de cada versión (uno por fold) y media entre las 6 versiones: 30 CNN propias desde cero (huella del modelo `5c1dbb39aef8`).

## 2. Calibración (Platt)

logit calibrado = 0,235 · logit + (-0,608). Error de Brier por paciente: 0,2078 sin calibrar → 0,2062 calibrado (menos es mejor; 0,25 sería decir siempre 0,5).

## 3. Agregación por paciente (1097 pacientes, 322 con pCR)

| Agregación | AUC | IC 95 % (bootstrap) |
|---|---|---|
| media **(elegida)** | 0,578 | 0,541–0,614 |
| mediana | 0,577 | 0,540–0,614 |
| máximo | 0,583 | 0,546–0,620 |
| voto | 0,554 | 0,523–0,588 |

## 4. Umbral

| Criterio | Umbral | Sensibilidad | Especificidad | Precision pCR | F1 pCR | F1 no pCR | F1 macro |
|---|---|---|---|---|---|---|---|
| máxima sensibilidad con especificidad ≥ 0,80 **(elegido)** | 0,312 | 0,264 | 0,813 | 0,370 | 0,308 | 0,767 | 0,538 |
| máximo de sensibilidad + especificidad − 1 | 0,286 | 0,786 | 0,355 | 0,336 | 0,471 | 0,492 | 0,481 |
| máxima F1 macro | 0,302 | 0,463 | 0,652 | 0,356 | 0,402 | 0,695 | 0,549 |
| umbral por defecto | 0,500 | 0,000 | 0,999 | 0,000 | 0,000 | 0,827 | 0,414 |

Con el umbral elegido, fuera de fold: VP 85, FN 237, FP 145, VN 630.

## 5. Por cohorte (fuera de fold)

| Cohorte | Pacientes | Con pCR | AUC | IC 95 % |
|---|---|---|---|---|
| Duke | 209 | 44 | 0,528 | 0,436–0,616 |
| I-SPY1 | 104 | 26 | 0,578 | 0,443–0,713 |
| I-SPY2 | 784 | 252 | 0,575 | 0,533–0,617 |

## 6. Exportación

| Fichero | Época | AUC del fold | sha256 |
|---|---|---|---|
| `models/final/cnn_v3/fold_0.pt` | 1 | 0,566 | `7b9ecf9d88b01e5a…` |
| `models/final/cnn_v3/fold_1.pt` | 6 | 0,537 | `70a4916e0d2432b6…` |
| `models/final/cnn_v3/fold_2.pt` | 5 | 0,635 | `5b18599b70cbd0cc…` |
| `models/final/cnn_v3/fold_3.pt` | 16 | 0,563 | `2bbf7bdb6131bd83…` |
| `models/final/cnn_v3/fold_4.pt` | 6 | 0,579 | `2a1ac74dae0b4681…` |
| `models/final/cnn_v4/fold_0.pt` | 6 | 0,564 | `2414ca6736a89ab8…` |
| `models/final/cnn_v4/fold_1.pt` | 3 | 0,534 | `11aeaf7886bb7daa…` |
| `models/final/cnn_v4/fold_2.pt` | 5 | 0,609 | `aee180c86ef6f3a7…` |
| `models/final/cnn_v4/fold_3.pt` | 28 | 0,528 | `50c6c3c063176cb6…` |
| `models/final/cnn_v4/fold_4.pt` | 5 | 0,626 | `be17acdaf19cfd12…` |
| `models/final/cnn_v5/fold_0.pt` | 10 | 0,539 | `c0f712b0265198c6…` |
| `models/final/cnn_v5/fold_1.pt` | 13 | 0,557 | `a82250626418c276…` |
| `models/final/cnn_v5/fold_2.pt` | 5 | 0,605 | `ad53b837d48a0fc7…` |
| `models/final/cnn_v5/fold_3.pt` | 10 | 0,553 | `cd65f79152a5f72d…` |
| `models/final/cnn_v5/fold_4.pt` | 6 | 0,619 | `2ae977f3c6ad6be7…` |
| `models/final/cnn_v6/fold_0.pt` | 1 | 0,580 | `14cf5e6498ad298c…` |
| `models/final/cnn_v6/fold_1.pt` | 6 | 0,540 | `a9ef81d4e0e7c02b…` |
| `models/final/cnn_v6/fold_2.pt` | 4 | 0,649 | `1b12764550baec70…` |
| `models/final/cnn_v6/fold_3.pt` | 13 | 0,577 | `843c5258ba8d9a2a…` |
| `models/final/cnn_v6/fold_4.pt` | 6 | 0,611 | `95c0d64b274f2406…` |
| `models/final/cnn_v7/fold_0.pt` | 1 | 0,563 | `02bafe62204f51aa…` |
| `models/final/cnn_v7/fold_1.pt` | 3 | 0,547 | `75c08d9710089034…` |
| `models/final/cnn_v7/fold_2.pt` | 17 | 0,642 | `287561314ffb5085…` |
| `models/final/cnn_v7/fold_3.pt` | 29 | 0,563 | `aba59268495bdaa4…` |
| `models/final/cnn_v7/fold_4.pt` | 6 | 0,626 | `629e87f521ee5e7a…` |
| `models/final/cnn_v10/fold_0.pt` | 10 | 0,587 | `9e6a7e5fdd9f2f85…` |
| `models/final/cnn_v10/fold_1.pt` | 4 | 0,545 | `6b5fe6c1c00eccca…` |
| `models/final/cnn_v10/fold_2.pt` | 15 | 0,632 | `20d974cd3528c552…` |
| `models/final/cnn_v10/fold_3.pt` | 14 | 0,526 | `64fb5d9d5fbdc870…` |
| `models/final/cnn_v10/fold_4.pt` | 6 | 0,616 | `2b2b5350dd136a58…` |

Comprobación: cada modelo recargado desde `models/final/` reproduce las predicciones del entrenamiento con una diferencia máxima de 3.0e-07.

![Cierre del modelo](cierre.png)
