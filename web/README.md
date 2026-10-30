---
title: Predicción de pCR en RM de mama
emoji: 🩻
colorFrom: gray
colorTo: yellow
sdk: static
app_file: index.html
pinned: false
license: cc-by-nc-4.0
short_description: CNN propia que estima la pCR en RM de mama (educativo)
---

# Predicción de pCR en resonancia de mama

App web del caso BreastDCEDL (Aprendizaje Automático, UAX). Una CNN 2D entrenada desde cero mira las tres fases
(PRE, EARLY y LATE) de un corte de DCE-MRI de mama hecho antes de la quimioterapia y estima la probabilidad de
respuesta patológica completa (pCR).

**Uso educativo, sin validez clínica.** No es un dispositivo médico.

Tres páginas:
- **Predicción** (`index.html`): sube las tres fases de un corte o elige un ejemplo; da la probabilidad, la
  clase y el mapa de qué zonas la suben o la bajan.
- **La red** (`red.html`): la arquitectura pieza a pieza, con un corte atravesándola, sus mapas de
  características reales y los parámetros de cada capa.
- **Resultados** (`resultados.html`): la CNN frente a un modelo clínico sin imagen y a los dos juntos, con
  predicciones fuera de fold de las pacientes de entrenamiento (sin identificadores).

- **Modelo:** conjunto de 30 CNN propias (6 versiones × 5 folds), entrenadas desde cero con PyTorch. Se ejecuta
  en el navegador de quien usa la página, con ONNX Runtime Web: las imágenes no salen de su ordenador.
- **Mismo cálculo que el pipeline:** `modelo/modelo.onnx` se exporta del modelo final con
  `python -m src.models.exportar_web`. Lleva dentro la normalización del entrenamiento (dividir entre 255), las
  30 redes, la media de logits y la calibración. Antes de publicarlo se comprueba que da lo mismo que PyTorch.
- **Validación de la entrada:** `js/validar.js` aplica las mismas reglas y los mismos mensajes que
  `src/data/entrada.py`, y un test lo comprueba con ficheros válidos e inválidos.
- **Código y resultados:** <https://github.com/judithsalas/breastdcedl> (la app está en la carpeta `web/`).

Imágenes de ejemplo: dataset docente derivado de BreastDCEDL (Fridman et al., *Scientific Data*, 2026), con
licencia CC BY-NC 4.0. Son de pacientes de entrenamiento; ninguna es de test ni de la validación privada.

## Cómo se publica

Esta carpeta se sube tal cual a un Space estático de Hugging Face (gratis). Pasos en el `README.md` del
repositorio, sección «App web».
