# App web: predicción de pCR para un corte

Interfaz Streamlit sobre el **mismo código** del pipeline:
- `src/data/entrada.py` lee y valida los tres PNG con la misma normalización que el entrenamiento
  (gris de 8 bits / 255);
- `src/models/final.py` carga el modelo final de `models/final/`: conjunto de 5 folds y
  calibración de Platt, en modo evaluación y sin gradientes.

Por qué Streamlit y Streamlit Community Cloud: D23 del registro y `docs/COMO_SE_HIZO.md`.

## Qué cumple del enunciado (§9) y de la guía (C6)

| Requisito | Dónde |
|---|---|
| Cargar las tres fases de un corte o elegir un ejemplo | pestañas "Subir" y "Elegir un ejemplo" |
| Mostrar PRE, EARLY, LATE por separado y el mapa de realce | figura de 4 paneles |
| Validar forma, tipo y valores; errores claros con ficheros corruptos, que no sean PNG de 256×256, o con fases ausentes o duplicadas | `src/data/entrada.py` (tests en `tests/test_entrada.py`) |
| Probabilidad de pCR y clase según el umbral | métricas principales |
| Umbral y versión o checksum del modelo | métricas y barra lateral (huella del modelo) |
| Dispositivo de inferencia y tiempo de respuesta | línea bajo las métricas |
| Parámetros del entrenamiento y métricas internas | barra lateral |
| Aviso visible de uso educativo, incertidumbre y ausencia de validez clínica | aviso arriba y recuadro de incertidumbre |
| Sin rutas arbitrarias ni ejecución de contenido subido; límite de tamaño y formato | solo subidas en memoria y ejemplos de una lista cerrada; 2 MB por PNG (`.streamlit/config.toml` y `src/data/entrada.py`) |
| `model.eval()`, sin gradientes, resultado reproducible | `ModeloFinal` (en eval, `torch.no_grad`) |
| Normalización idéntica a la del entrenamiento | test `test_misma_normalizacion_que_el_entrenamiento` y comprobación con los ejemplos |
| Sin validación privada ni sus etiquetas | los ejemplos son de train (`app/ejemplos/LEEME.md`) |

## Probarla en local

```bash
pip install -r app/requirements.txt     # en el entorno cancer ya está casi todo; añade streamlit
streamlit run app/app.py
```

Antes hace falta el modelo final (`python -m src.evaluation.cerrar_modelo`). Los ejemplos se
generan con `python -m app.preparar_ejemplos`.

## Desplegar (Streamlit Community Cloud, gratis)

1. En el PC de la uni, con el plan terminado:
   ```bash
   python -m src.evaluation.cerrar_modelo     # crea models/final/
   python -m app.preparar_ejemplos            # crea app/ejemplos/
   git add models/final app/ejemplos docs && git commit -m "feat(app): modelo final y ejemplos" && git push
   ```
2. Entrar en <https://share.streamlit.io> con la cuenta de GitHub y elegir **Create app**.
3. Repositorio `judithsalas/breastdcedl`, rama `main`, fichero principal `app/app.py`.
4. En **Advanced settings**, elegir Python **3.12** (la misma versión que el entorno `cancer`).
5. **Deploy**. Streamlit instala `app/requirements.txt` (PyTorch CPU 2.14.0) y da una URL del
   tipo `https://<nombre>.streamlit.app`.
6. Comprobar en la URL las pruebas de la lista de abajo y apuntar la URL en el README y en el
   informe.

Las apps gratuitas se duermen si nadie las usa durante unos días. **Antes de la defensa, abrir la
URL** para despertarla y esperar a que cargue. No volver a desplegar ni reiniciar durante la
defensa (el enunciado lo prohíbe).

## Pruebas con entradas válidas e inválidas (antes de entregar)

- [ ] Un ejemplo: la probabilidad de la app coincide con la del pipeline (✅ en pantalla).
- [ ] Las tres fases de un corte de test subidas a mano: mismo resultado que
      `python -m src.models.predecir <PRE> <EARLY> <LATE>`.
- [ ] Solo dos fases → "Falta la fase …".
- [ ] La misma fase dos veces → "Fase … duplicada".
- [ ] Un JPG renombrado a .png → "no es un PNG".
- [ ] Un PNG en color o de otro tamaño → mensaje de modo o de tamaño.
- [ ] Un PNG cortado (fichero dañado) → "dañado".
- [ ] Fases de dos cortes distintos → "mismo corte".
