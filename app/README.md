# App web: predicción de pCR para un corte

> **La app publicada ya no es esta (desde el 2026-10-29).** Es la página de `web/`, en un Space estático de
> Hugging Face, con el mismo modelo en ONNX ejecutado en el navegador (pasos en el `README.md` principal,
> «App web»). Esta carpeta guarda los **ejemplos** (`ejemplos/`, que también usa `web/`), las **pruebas de
> entrada** (`casos_prueba.py`) y una **versión local en Python** con Streamlit, que usa el mismo modelo y la
> misma validación. Lo de abajo describe esa versión local.

Interfaz Streamlit sobre el **mismo código** del pipeline:
- `src/data/entrada.py` lee y valida los tres PNG con la misma normalización que el entrenamiento
  (gris de 8 bits / 255);
- `src/models/final.py` carga el modelo final de `experimentos/modelo_final/`: conjunto de CNN propias (los 5
  modelos de cada versión, uno por fold) y calibración de Platt, en modo evaluación y sin
  gradientes.

Streamlit se eligió porque la app queda en Python, con el mismo código del pipeline, y Streamlit
Community Cloud la publica gratis desde el repositorio.

## Qué cumple del enunciado (§9) y de la guía (C6)

| Requisito | Dónde |
|---|---|
| Cargar las tres fases de un corte o elegir un ejemplo | pestañas "Subir" y "Elegir un ejemplo" |
| Mostrar PRE, EARLY, LATE por separado y el mapa de realce | figura de 4 paneles |
| Validar forma, tipo y valores; errores claros con ficheros corruptos, que no sean PNG de 256×256, o con fases ausentes o duplicadas | `src/data/entrada.py` (comprobado con tests en local) |
| Probabilidad de pCR y clase según el umbral | métricas principales |
| Umbral y versión o checksum del modelo | métricas y barra lateral (huella del modelo) |
| Dispositivo de inferencia y tiempo de respuesta | línea bajo las métricas |
| Parámetros del entrenamiento y métricas internas | barra lateral |
| Aviso visible de uso educativo, incertidumbre y ausencia de validez clínica | aviso arriba y recuadro de incertidumbre |
| Sin rutas arbitrarias ni ejecución de contenido subido; límite de tamaño y formato | solo subidas en memoria y ejemplos de una lista cerrada; 2 MB por PNG (`.streamlit/config.toml` y `src/data/entrada.py`) |
| `model.eval()`, sin gradientes, resultado reproducible | `ModeloFinal` (en eval, `torch.no_grad`) |
| Normalización idéntica a la del entrenamiento | un test en local y la comprobación con los ejemplos |
| Sin validación privada ni sus etiquetas | los ejemplos son de train (`app/ejemplos/LEEME.md`) |

## Probarla en local

```bash
pip install -r app/requirements.txt     # en el entorno cancer ya está casi todo; añade streamlit
streamlit run app/app.py
```

Antes hace falta el modelo final (`python -m src.evaluation.cerrar_modelo`). Los ejemplos se
generan con `python -m app.preparar_ejemplos`.

## Desplegar (Streamlit Community Cloud, gratis)

1. Tener en el repositorio `experimentos/modelo_final/` (el modelo final) y `app/ejemplos/` (se crean con
   `python -m src.evaluation.cerrar_modelo` y `python -m app.preparar_ejemplos`).
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

Los ficheros de casi todas las pruebas se crean con `python -m app.casos_prueba` (a partir de
`app/ejemplos/`; no hacen falta las imágenes del dataset). Deja en `pruebas_app/` (no se versiona) una
carpeta por prueba, con los ficheros que hay que subir juntos, y en `pruebas_app/LEEME.md` lo que la
app tiene que responder a cada una, con una columna para marcarla.

- [ ] Un ejemplo de la pestaña «Elegir un ejemplo»: la probabilidad de la app coincide con la del
      pipeline (✅ en pantalla).
- [ ] `01_valido`: predice, y da la misma probabilidad que el pipeline (está en el LEEME).
- [ ] Un corte de train que no sea de los ejemplos, subido a mano: mismo resultado que
      `python -m src.models.predecir <PRE> <EARLY> <LATE>`.
- [ ] `02_dos_fases` → "Falta la fase LATE".
- [ ] `03_misma_imagen_dos_veces` → "Dos de los ficheros son idénticos".
- [ ] `04_jpg_renombrado` → "no es un PNG".
- [ ] `05_png_en_color` y `06_otro_tamano` → mensaje de modo o de tamaño.
- [ ] `07_png_danado` → "dañado".
- [ ] `08_fases_de_dos_cortes` → "mismo corte".
- [ ] `09_fases_cambiadas` → predice, con el aviso de que EARLY no es más brillante que PRE.
