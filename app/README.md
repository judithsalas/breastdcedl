# Ejemplos y pruebas de la app web

La app publicada es la página de `web/`, en un Space estático de Hugging Face (pasos en el `README.md`
principal, «App web»). Esta carpeta guarda lo que la página necesita para sus ejemplos y para probarla:

- `ejemplos/`: seis cortes de train, con la probabilidad que les da el pipeline (`esperado.json`).
  `python -m src.models.exportar_web` los copia a `web/ejemplos/`, y la página marca ✓ cuando su
  probabilidad coincide con la del pipeline. Se generan con `python -m app.preparar_ejemplos`, que
  necesita las imágenes del dataset.
- `casos_prueba.py`: crea los ficheros para probar la app con entradas válidas e inválidas.

## Pruebas con entradas válidas e inválidas (antes de entregar)

`python -m app.casos_prueba` parte de `app/ejemplos/` (no hacen falta las imágenes del dataset) y deja en
`pruebas_app/` (no se versiona) una carpeta por prueba, con los ficheros que hay que subir juntos, y en
`pruebas_app/LEEME.md` lo que la app tiene que responder a cada una, con una columna para marcarla.

- [ ] Un ejemplo de «Elegir un ejemplo»: la probabilidad de la app coincide con la del pipeline (✓ en
      pantalla).
- [ ] `01_valido`: predice, y da la misma probabilidad que el pipeline (está en el LEEME).
- [ ] Un corte de train que no sea de los ejemplos, subido a mano: mismo resultado que
      `python -m src.models.predecir <PRE> <EARLY> <LATE>`.
- [ ] `02_dos_fases` → coloca las dos imágenes y dice "Falta LATE: pon esa imagen en su hueco."
- [ ] `03_misma_imagen_dos_veces` → "Dos de los ficheros son idénticos".
- [ ] `04_jpg_renombrado` → "no es un PNG".
- [ ] `05_png_en_color` y `06_otro_tamano` → mensaje de modo o de tamaño.
- [ ] `07_png_danado` → "dañado".
- [ ] `08_fases_de_dos_cortes` → "mismo corte".
- [ ] `09_fases_cambiadas` → predice, con el aviso de que EARLY no es más brillante que PRE.
