# Modelo clínico de comparación (sin imagen)

Generado por `python -m src.evaluation.modelo_clinico` el 2026-10-28 18:38. Solo train, con
los mismos 5 folds que la CNN; test no se ha usado (se evalúa una sola vez, junto con la CNN, en `evaluar_test`).
Por qué y cómo: D25 en [`COMO_SE_HIZO.md`](../../COMO_SE_HIZO.md#25-el-modelo-del-médico-sin-imagen-d25).

**Para qué sirve.** Es lo que sabría un médico antes de la quimioterapia sin mirar la resonancia. Si la CNN no lo
supera, la imagen (tal como la usamos) aporta menos que la biopsia. **No entra en la CNN ni en la app**: el profesor
pide estas variables como comparación, no como modelo principal.

## 1. Qué sabe el médico

| Variable | Para qué | % que falta en train: Duke · I-SPY1 · I-SPY2 |
|---|---|---|
| `age` · Edad | edad de la paciente | 0,0 % · 0,0 % · 0,4 % |
| `menopause` · Posmenopausia | estado menopáusico (1 = posmenopausia) | 0,0 % · 100,0 % · 3,8 % |
| `HR` · Receptores hormonales (HR+) | receptores hormonales en la biopsia | 0,0 % · 1,9 % · 0,0 % |
| `HER2` · HER2+ | HER2 en la biopsia | 0,0 % · 2,9 % · 0,0 % |
| `tum_vol` · Volumen tumoral | solo en la variante (sale de la imagen) | 0,5 % · 0,0 % · 0,0 % |

Lo que falta se completa con lo aprendido en los folds de entrenamiento: la mediana; la menopausia, con la
probabilidad de posmenopausia según la edad (falta entera en I-SPY1). Sin indicador de «falta el dato», que en
I-SPY1 delataría la cohorte.

**El dataset no trae análisis de sangre, peso, IMC, tabaco ni alcohol** (guía). Lo que no se usa:

| Columnas | Por qué no |
|---|---|
| `race_white, race_black` | raza: tratarla como causa es un error metodológico (guía, D4) |
| `dataset` | cohorte: es del dataset, no de la paciente (R0 mide cuánto acierta sola) |
| `n_xy, n_z, n_times, pre, post_early, post_late, slice_thick, xy_spacing` | adquisición de la resonancia: no es información clínica |
| `mask_start, mask_end, sraw, eraw, scol, ecol` | localización de la máscara: sale de la imagen |
| `tum_vol` | volumen tumoral: sale de la segmentación de la imagen y en Duke tiene otra escala; solo como variante aparte |
| `HR_HER2_STATUS, TripleNeg, HER2pos, HRposHER2neg` | son combinaciones de HR y HER2, que ya están |

## 2. Resultados en validación cruzada (1097 pacientes, 322 con pCR)

| Modelo | AUC validación (5 folds) | AUC entrenamiento | Diferencia | AUC fuera de fold (IC 95 %) | Sensibilidad | Especificidad |
|---|---|---|---|---|---|---|
| Solo la cohorte (R0) | 0,549 ± 0,027 | — | — | — | — | — |
| Solo el subtipo HR/HER2 (R0) | 0,657 ± 0,028 | — | — | — | — | — |
| **Médico: edad, menopausia, HR, HER2** | 0,689 ± 0,024 | 0,700 | 0,010 | 0,685 (0,65–0,72) | 0,398 | 0,803 |
| Médico + volumen tumoral | 0,711 ± 0,030 | 0,720 | 0,010 | 0,707 (0,67–0,74) | 0,444 | 0,801 |
| CNN (solo imagen) | 0,596 ± 0,040 | — | — | 0,578 (0,54–0,61) | 0,264 | 0,813 |
| CNN + médico | 0,699 ± 0,022 | 0,710 | 0,011 | 0,693 (0,66–0,73) | 0,463 | 0,801 |

Sensibilidad y especificidad con el umbral de cada modelo elegido con el mismo criterio que la CNN (mayor sensibilidad con especificidad ≥ 0,80).
El volumen se prueba solo como variante: sale de la segmentación de la imagen y en Duke tiene otra escala, así que no es «lo que sabe el médico» y no se lleva a test.

![ROC fuera de fold](roc_fuera_de_fold.png)

## 3. Qué pesa cada variable (modelo ajustado con todo train)

| Variable | Cambio | Razón de odds de pCR (IC 95 %, bootstrap) |
|---|---|---|
| Edad | 10 años más | 0,84 (0,69–1,03) |
| Posmenopausia | sí frente a no | 0,96 (0,62–1,45) |
| Receptores hormonales (HR+) | positivo frente a negativo | 0,30 (0,22–0,38) |
| HER2+ | positivo frente a negativo | 2,74 (2,06–3,84) |

Razón de odds > 1: más probable la pCR; < 1: menos; si el intervalo cruza el 1, no se puede afirmar que influya. Son asociaciones en este dataset, no causas.

![Razones de odds](razones_de_odds.png)

## 4. CNN frente al médico (1097 pacientes, fuera de fold)

| Comparación | Diferencia de AUC (bootstrap pareado de pacientes) |
|---|---|
| CNN − médico | -0,108 (IC 95 % -0,157 a -0,059) |
| (CNN + médico) − médico: lo que **añade la imagen** | 0,008 (IC 95 % -0,005 a 0,020) |
| (CNN + médico) − CNN: lo que añade la clínica | 0,116 (IC 95 % 0,072 a 0,161) |

Correlación de Spearman entre la puntuación de la CNN y la del médico: 0,02. Si fuera alta, la CNN estaría
repitiendo sobre todo el subtipo; si es baja, mira otra cosa.

**AUC de la CNN dentro de cada subtipo.** Dentro de un subtipo el médico apenas distingue (solo le
quedan edad y menopausia). Si la CNN separa ahí, la imagen aporta algo más que el subtipo.

| Subtipo | Pacientes | Con pCR | AUC de la CNN |
|---|---|---|---|
| HER2pos | 287 | 123 | 0,628 |
| HRposHER2neg | 453 | 64 | 0,516 |
| TripleNeg | 354 | 134 | 0,589 |

**Por cohorte (AUC fuera de fold).**

| Cohorte | Pacientes | Con pCR | CNN | Médico | CNN + médico |
|---|---|---|---|---|---|
| Duke | 209 | 44 | 0,528 | 0,717 | 0,714 |
| I-SPY1 | 104 | 26 | 0,578 | 0,796 | 0,774 |
| I-SPY2 | 784 | 252 | 0,575 | 0,663 | 0,673 |

El combinado usa la puntuación fuera de fold de la CNN (cada paciente, predicha por el modelo que no la vio):
es apilado honesto. En test se le da la puntuación del conjunto de 5 modelos, que es algo más estable.

## 5. Cómo leerlo

- Es el listón que la CNN tiene que superar para decir que la imagen aporta algo que no sabe ya el médico.
- Las diferencias de AUC se juzgan por su intervalo: si cruza el 0, no hay pruebas de que un modelo sea mejor que otro.
- En la bibliografía pasa lo mismo: en Joo et al. 2021 la clínica sola daba 0,83 y la imagen la subía a 0,89.
- Modelo guardado en `models/clinico/modelo_clinico.json` (huella `1d963df5`, hecho con la CNN `5c1dbb39aef8`): lo usa `evaluar_test`.
