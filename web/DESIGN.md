# DESIGN.md · App web de pCR en RM de mama

Sistema de diseño de las tres páginas de `web/`. Los valores viven en `estilos.css` como variables;
las páginas usan los nombres, nunca valores sueltos. Sale del tema, de las convenciones de las estaciones de
lectura de radiología y de dos revisiones de diseño de la página publicada.

## Para quién y qué tiene que transmitir

- **Quién lo usa:** el profesor y el tribunal de Aprendizaje Automático, y compañeros. Cargan un corte,
  miran la predicción y revisan cómo se hizo el modelo.
- **Sensación:** precisión clínica y calma. Una estación de lectura de radiología con un informe al lado,
  no una página de marketing.
- **Lo memorable:** el visor. La imagen manda; todo lo demás es discreto.

## Principios

1. **Lo que se mira va en monitores negros; lo que se lee, en la sala.** Imágenes y gráficas sobre
   `--monitor`; texto y controles sobre `--sala` y `--panel`.
2. **Un solo acento en la interfaz:** el ámbar del gadolinio, para lo interactivo y lo actual (lo
   elegido, la acción principal, los enlaces, la página en la que estás, el corte en la escala).
3. **Los colores de datos solo viven dentro de imágenes y gráficas** y cada uno significa una sola cosa.
4. **Jerarquía de producto:** títulos contenidos, etiquetas en minúscula y datos alineados. Sin
   mayúsculas de adorno, sin «·» para unir datos y sin flechas en los textos.
5. **Las dos clases se llaman siempre «pCR» y «no pCR»**: en etiquetas, leyendas, matrices, tablas y
   textos («pacientes pCR», «pacientes no pCR»). Nunca «con pCR», «sin ella», «lograron pCR» ni «las demás».

## Color

| Token | Valor | Papel |
|---|---|---|
| `--sala` | `#0b1116` | fondo de la página |
| `--panel` | `#10181f` | barra superior, informe, paneles de texto |
| `--panel-2` | `#16202a` | hover, campos, chips |
| `--monitor` | `#05080b` | fondo de imágenes y gráficas |
| `--linea` | `#223039` | bordes finos y separadores |
| `--linea-2` | `#33424d` | bordes de controles |
| `--tinta` | `#e6edf2` | texto principal (16:1 sobre `--sala`) |
| `--tinta-2` | `#9eb0bd` | texto secundario (8,5:1) |
| `--tinta-3` | `#7c8f9d` | etiquetas y pies (5,7:1) |
| `--acento` | `#f2a541` | acento único: selección, botón principal (texto `--sala` encima, 9,3:1), página actual, el corte |
| `--acento-texto` | `#f7bd6b` | enlaces y textos en acento (11:1) |
| `--foco` | `#ffffff` | anillo de foco del teclado |
| `--exito` / `--aviso` / `--error` | `#6fd3a2` / `#e5c07b` / `#f28b82` | coincide con el pipeline / avisos de la entrada / errores |

**Colores de datos** (solo en imágenes y gráficas):

| Qué | Colores |
|---|---|
| Realce (EARLY − PRE) | escala «inferno» |
| Mapa «Qué empuja» | rojo `#ef5b5b` sube, azul `#3b8beb` baja (convención de SHAP) |
| CNN | verde azulado `#1b9cb1` (y su rampa en las activaciones de «La red») |
| Médico | violeta `#9085e9` |
| CNN + médico | magenta `#d55181` |

Las tres series están validadas contra `--monitor`: ΔE con daltonismo ≥ 8,1, sin daltonismo ≥ 15,9 y
contraste ≥ 3:1.

## Tipografía

- **Schibsted Grotesk** (700, 800): títulos y cifras. Grotesca compacta y firme.
- **Atkinson Hyperlegible Next** (400, 600, 700): todo lo demás. Del Braille Institute, pensada para no
  confundir letras ni cifras; encaja con un contexto médico y de accesibilidad.

Escala (razón ≈ 1,25, interlineado 1,5 en texto y 1,15 en títulos):

| Token | Tamaño | Uso |
|---|---|---|
| `--t-xs` | 12,8 px | pies, etiquetas, ejes |
| `--t-s` | 14,4 px | controles, tablas, notas |
| `--t-m` | 16 px | texto |
| `--t-l` | 20 px | títulos de panel y de gráfica |
| `--t-xl` | 25 px | títulos de sección |
| `--t-xxl` | 40 px | título de página y cifras de «Resultados» |
| `--t-cifra` | 72 px | la probabilidad del resultado |

Líneas de texto de 60 a 75 caracteres (`max-width: 68ch`). Las cifras del texto son tabulares (se alinean
en columnas); las de los títulos y cifras grandes, proporcionales, para que no abran huecos.

## Espacio

Escala de 4 px: `--e-1` 4 · `--e-2` 8 · `--e-3` 12 · `--e-4` 16 · `--e-5` 24 · `--e-6` 32 · `--e-7` 48 ·
`--e-8` 64. Secciones separadas por `--e-8` (escritorio) y `--e-7` (móvil). Un solo ancho de página,
`--ancho-pagina` 1.320 px, con `--margen` a los lados.

## Forma y profundidad

- Radios: `--radio-s` 4 px (el visor, las imágenes y las miniaturas) y `--radio` 8 px (paneles, tarjetas de
  gráficas, botones, filtros, campos).
  Las píldoras (999 px) solo para etiquetas de estado: la pastilla del aviso y la clase del resultado.
- Una sombra, `--sombra`, solo para lo que flota (información emergente, curva de realce, barra de filtros).
  Los monitores no llevan sombra: un borde de 1 px.

## Componentes

- **Barra superior:** marca (icono del visor) y nombre, las tres páginas como pestañas con subrayado
  `--acento` en la actual, y a la derecha la pastilla «Uso educativo, sin validez clínica», que lleva al
  aviso completo del pie. Hasta 900 px dice solo «Sin validez clínica», y en el móvil va sin icono, para que
  la barra no se parta.
- **Botones:** principal (fondo `--acento`, texto `--sala`), secundario (borde `--linea-2`) y discreto
  (solo texto, como un enlace). Hover: un paso más claro. Foco: anillo `--foco`. Desactivado: 40 % de
  opacidad. Como mucho un principal a la vista: «Subir un corte» en Predicción y «Siguiente» en los pasos de «La red».
- **Enlaces:** `--acento-texto`, subrayados.
- **Informe del resultado:** panel con título, la clase como etiqueta («pCR» / «no pCR»), la cifra, la
  escala y los datos del modelo en dos columnas. Debajo de la clase, siempre, la fiabilidad del modelo
  (`.fiabilidad-corte`, sobre `--panel-2`): una cifra de salud no va nunca sin su incertidumbre al lado.
- **Monitores:** gráficas e imágenes sobre `--monitor`, con borde `--linea`; sin sombra.
- **Cifras destacadas** («Resultados», «Cómo de fiable es»): raya neutra `--linea-2` encima, la cifra en
  la letra de los títulos y, si es de un modelo, un punto de su color junto al nombre (nunca la raya).
- **Gráficas agrupadas por pregunta:** un título de sección (h2) por pregunta y las tarjetas debajo (h3).
  En pantallas estrechas, los nombres de las filas van encima de su fila y el eje lleva las marcas que caben.
- **Visor:** datos en las cuatro esquinas sobre un velo oscuro degradado arriba y abajo, para que se lean
  sobre cualquier imagen.
- **Miniaturas y selector de cortes:** imagen cuadrada, nombre debajo; elegida = contorno `--acento`.
- **Por dónde empezar:** bajo el título de Predicción, el botón principal «Subir un corte» y el enlace «o
  elegir un ejemplo». En tablet y móvil, donde el título queda debajo del visor, el mismo par va encima de la
  imagen, con la línea que dice qué es. Es el único botón principal de la página.
- **Huecos para subir un corte:** tres recuadros en fila, PRE, EARLY y LATE, cada uno con su
  monitor cuadrado, el nombre de la fase, qué momento es y el fichero. Vacío: borde discontinuo `--linea-2`
  e icono de subir; al arrastrar encima o al pasar el cursor, borde `--acento`; con imagen, borde continuo,
  la miniatura y un botón × redondo de 28 px. Debajo, una sola línea con lo que tiene que cumplir cada
  imagen y «Quitar las tres imágenes» (botón discreto). Sin explicaciones de nombres de fichero ni de cómo
  valida la página. El foco del teclado rodea todo el recuadro. Una imagen que no pasa la validación pone
  su hueco con borde y nombre `--error`.
- **Carga sin saltos:** una página que se dibuja con datos que llegan después (Resultados) se monta invisible
  (`data-cargando` en `body`, con una línea «Cargando…» bajo el título) y aparece de una vez.
- **Pie:** aviso completo, el proyecto y los datos, en tres columnas.

## Iconos e imágenes

- Las imágenes son las resonancias del dataset: nada de fotos de archivo ni ilustraciones.
- Iconos: muy pocos, de trazo de 1,75 px al estilo Lucide (información, subir), en línea con el texto.
- Sin flechas en los textos: «129 pesos, sale 1 logit», no «129 pesos → 1 logit».

## Accesibilidad (WCAG 2.1 AA)

- Texto ≥ 4,5:1 (también el de las gráficas SVG y el de las esquinas del visor). El foco y lo que marca un
  estado (el contorno de lo elegido, el punto de los filtros) ≥ 3:1.
- Foco del teclado siempre visible (`--foco`); orden del tabulador igual que el de lectura.
- Controles de 44 px de alto como mínimo (botones, filtros, «Ver los datos», deslizador, pestañas); la
  pastilla del aviso, 32 px.
- Cada gráfica con nombre (`role="img"` y `aria-label`) y sus valores en una tabla («Ver los datos»).
- Nada depende solo del color: leyendas y etiquetas directas en las gráficas, texto junto a cada clase.
- Sin scroll horizontal a 320 px; sin animaciones si se piden menos (`prefers-reduced-motion`).
- Se comprueba con axe-core y Playwright antes de publicar (pruebas locales, que no se suben).
