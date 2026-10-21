# CLAUDE.md

Contexto y reglas para Claude en este repositorio. **Leer al empezar cada sesión**, junto con
[`docs/REGISTRO.md`](docs/REGISTRO.md) (estado, hoja de ruta, decisiones de proceso) y
[`docs/ARQUITECTURAS.md`](docs/ARQUITECTURAS.md) (historia de la red, una ficha por versión).
Qué va en cada documento: D14 del registro. Las decisiones explicadas desde cero, para que Judith
pueda entenderlas y defenderlas: [`docs/COMO_SE_HIZO.md`](docs/COMO_SE_HIZO.md).

## El proyecto

Caso BreastDCEDL: CNN 2D desde cero (PyTorch) que predice pCR a partir de las tres fases
DCE-MRI (PRE, EARLY, LATE) de un corte, con evaluación **por paciente**. Entregables:
repositorio reproducible, pesos, informe, app web desplegada con URL y 5 diapositivas.

Fuentes de verdad, por orden:
1. Enunciado: `breastdcedl/documentation/caso_breastdcedl.pdf`
2. Guía: `breastdcedl/GUIA.md`
3. Correo del profesor del 2026-10-14: qué se evalúa, resultados razonables, bibliografía y la
   **rúbrica de 10 puntos**, que es la lista de comprobación del trabajo. Resumido en
   `docs/REGISTRO.md` → "Criterios de evaluación del profesor" (el correo lo tiene Judith).
4. Presentación del profesor, artículos de Fridman et al. (*Scientific Data* 2026 y preprint
   arXiv 2506.12190, resumidos en `docs/REGISTRO.md` → "Referencias") y diapositivas "De la GPU a
   GitHub" (no están en el repo; Judith las tiene).

## Reglas de trabajo con Judith (obligatorias)

- **Modo de trabajo desde el 2026-10-18 (D15): Claude decide, Judith revisa y aprende.** Judith
  pidió avanzar rápido: Claude toma las decisiones técnicas que considera más acertadas
  (arquitecturas incluidas, desde cnn_v3) y las implementa. Cada decisión se explica en
  `docs/COMO_SE_HIZO.md` desde cero: qué se decidió, por qué, qué alternativas había y qué
  responder si preguntan. Judith puede revertir cualquier decisión. cnn_v1 y cnn_v2 son diseño
  suyo. Las versiones viven en `src/models/versiones.py` (una por idea; las ya entrenadas no se
  modifican).
- **Lo que sigue siendo de Judith:** ejecutar en el PC de la uni (entrenamientos, test), subir
  a GitHub, entregar y defender. El enunciado exige que entienda y defienda sus decisiones: por
  eso cada decisión nueva DEBE tener su entrada en `docs/COMO_SE_HIZO.md`, con el mismo nivel de
  detalle que las existentes.
- **Cada versión, trazada** (pasos en `docs/ARQUITECTURAS.md` → "Cómo se usa"):
  - Antes de entrenar: entrada en `versiones.py` con su `ORIGEN` y `ESTADO`, y ficha con la
    hipótesis y "qué esperamos ver".
  - Después: ejecuciones, resultados, lectura, decisión y `ESTADO` actualizado.
  - Las gráficas que sirven de evidencia se copian con
    `python -m src.models.historial --figuras runs/<ejecución>`.
  - Los tests fallan si falta el origen, el estado o la ficha.
- **Ninguna decisión sin explicarla.** Antes de que Judith suba algo a GitHub, se le explica qué
  se ha hecho y por qué (y queda escrito en `docs/COMO_SE_HIZO.md` y en el registro).
- **Ser objetivo:** si algo no se puede hacer, está mal o contradice los documentos, decirlo
  con los motivos. Si hay contradicciones entre documentos o dudas, preguntar.
- Una arquitectura se descarta si en 5–10 épocas no mejora el AUC por paciente en validación.
- Las versiones se comparan con el criterio de `docs/ARQUITECTURAS.md` (D16), que aplica el código
  (`elegir_mejor`): media del AUC de 5 folds, empate dentro de una desviación, y entre empatadas
  primero las que no sobreajustan y después la más simple.
- Los resultados que van al informe salen de validación cruzada de 5 folds, con media ±
  desviación (rúbrica, punto 2): `python -m src.training.plan` o
  `python -m src.training.validacion_cruzada`.
- El orden de trabajo hasta la entrega está en `docs/COMO_SE_HIZO.md` → "Qué queda y en qué
  orden": plan → cierre del modelo → modelo clínico de comparación (D25) → test una vez → app →
  informe → diapositivas.
- Toda decisión, cambio de plan o resultado relevante se anota en `docs/REGISTRO.md`, y el estado
  de la rúbrica se actualiza allí.
- Las diapositivas (exactamente 5) son lo último que se hace.

## Reglas del caso que invalidan el trabajo

1. Mezclar cortes de una misma paciente entre entrenamiento y validación → usar siempre `fold`
   / `uc.particion`.
2. Usar test para ajustar algo (modelo, hiperparámetros, umbral, método de agregación).
   Test se evalúa **una sola vez**, al final.
3. Modelos preentrenados, arquitecturas de catálogo o transfer learning.
4. Publicar la validación privada del profesor o sus etiquetas.
5. Trabajo individual.

Las variables de `patients.csv` no entran en la CNN ni en la app (correo del profesor): solo en el
modelo clínico de comparación (D25, `src/evaluation/modelo_clinico.py`). La caja del tumor se usa en
cnn_v11 (D33) solo para medir cuánto se gana mirando el tumor: esas versiones no son desplegables
(`desplegable` en `historial.py`) y no pueden ser el modelo final.

Recordatorios técnicos: los 3 canales son instantes temporales, no RGB (nada de normalización
ImageNet ni ColorJitter); aumentos geométricos al tensor completo; `shuffle=False` al evaluar;
la normalización de la app debe ser idéntica a la del entrenamiento; no reportar solo accuracy
(predecir siempre pCR=0 da 70,6 %).

**Claude no tiene las imágenes** (`breastdcedl/dataset/`, ~38.000 PNG; el bucket está bloqueado
desde la nube). Tiene todo lo demás de `breastdcedl/` (guía, enunciado, metadatos, `utils_caso.py`).
Para leer resultados, Judith ejecuta `python -m src.utils.para_claude` y envía el zip (D27). Para
probar el flujo, Claude usa imágenes sintéticas.

## Dos máquinas

Repositorio del proyecto: **`judithsalas/breastdcedl`** (en el PC de la uni,
`~/Escritorio/JUDITH/breastdcedl`). El antiguo `caso1_breast-cancer` ya no se usa (2026-10-20).

| | Portátil | PC Universidad |
|---|---|---|
| SO | Windows | Ubuntu |
| Cálculo | CPU (GPU AMD integrada/baja, no se usa) | AMD RX 6700 XT 12 GB, ROCm |
| PyTorch | torch 2.14.0 CPU | torch 2.14.0 ROCm 7.14 + `HSA_OVERRIDE_GFX_VERSION=10.3.0` |

Mismo entorno conda `cancer` (Python 3.12) y mismo `requirements.txt` en ambas. PyTorch no
está en `requirements.txt` porque el paquete depende del hardware. El código no distingue
máquinas: `src/utils/dispositivo.py` detecta GPU o CPU. Los modos `rapido`/`completo`
(`config.json`, D31) indican tipo de ejecución, no máquina.

## Convenciones

- Código y documentación en español. Commits con Conventional Commits en español
  (`feat(models): ...`, `docs: ...`).
- Material del profesor en `breastdcedl/`: se usa tal cual, no se modifica (excluido de ruff).
- `breastdcedl/dataset/` nunca se versiona.
- De `models/` solo se versionan `models/final/` y `models/clinico/` (el JSON del modelo
  clínico, D25).
- De `runs/` solo se versiona `runs/historico.csv` (D13); las gráficas del informe se copian a
  `docs/figuras/`.
- Orden de carpetas (D26):
  - en `docs/` solo los tres documentos para leer (`REGISTRO.md`, `COMO_SE_HIZO.md`,
    `ARQUITECTURAS.md`), más `figuras/` y `resultados/`;
  - todo lo que genera un script va a `docs/resultados/<n>_<paso>/`, con el informe y sus
    figuras juntos (ruta en `RESULTADOS`, `src/data/caso.py`);
  - `tests/test_estructura.py` lo comprueba.
- Los `__init__.py` se dejan **vacíos** (petición de Judith, 2026-10-20). El código va en módulos
  con nombre: rutas y acceso a `utils_caso` en `src/data/caso.py`; crear una versión de la red,
  su huella y sus cambios en `src/models/construir.py`. Lo comprueba `tests/test_estructura.py`.
- Parámetros de ejecución (modos, evaluación, plantilla de experimentos) en `config.json` (D31).
  Lo que define una versión de la red, en `src/models/versiones.py`. No se meten números sueltos en
  el código.
- Antes de subir: `ruff check .` y `pytest -q`.
- Claude trabaja en la nube sobre una copia del repo y entrega un zip con los ficheros
  cambiados. Si hay que mover o borrar ficheros, el zip trae además un script de un solo uso.
  Judith lo copia encima del repo, comprueba (`ruff`, `pytest`) y sube a GitHub; en la otra
  máquina, `git pull`.
