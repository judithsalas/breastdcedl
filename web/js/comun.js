// Piezas comunes a las tres páginas de la app web: formato de números, colores y lectura de los datos.

export const num = (x, d = 3) => (x == null || Number.isNaN(x) ? "—" : Number(x).toFixed(d).replace(".", ",").replace("-", "−"));
export const pct = (x, d = 0) => (x == null ? "—" : `${(100 * x).toFixed(d).replace(".", ",")} %`);
// Separador de miles siempre, también con 4 cifras (4.608), como en el resto de la documentación
export const miles = (n) => (n == null ? "—" : String(Math.round(Number(n))).replace(/\B(?=(\d{3})+(?!\d))/g, "."));
export const $ = (id) => document.getElementById(id);

export function nodo(etiqueta, texto, clase) {
  const n = document.createElement(etiqueta);
  if (texto != null) n.textContent = texto;
  if (clase) n.className = clase;
  return n;
}

export async function leerJSON(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url} (${r.status})`);
  return r.json();
}

// Colores de los tres modelos de la comparación (paleta categórica validada: orden fijo, nunca por rango)
// Los tonos de la paleta para fondo oscuro, validados contra el fondo de las gráficas (--monitor) (CVD ΔE ≥ 9,4)
export const COLORES = { cnn: "#1b9cb1", medico: "#9085e9", combinado: "#d55181" };
export const NOMBRES_CORTOS = { cnn: "CNN", medico: "Médico", combinado: "CNN + médico" };
export const COHORTES = { spy1: "I-SPY1", spy2: "I-SPY2", duke: "Duke" };

// Rampas para pintar en la caja de luz
function rampa(anclas) {
  const rgb = anclas.map((h) => [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16)));
  const tabla = new Uint8ClampedArray(256 * 3);
  for (let i = 0; i < 256; i++) {
    const t = (i / 255) * (rgb.length - 1);
    const a = Math.floor(t), b = Math.min(a + 1, rgb.length - 1), f = t - a;
    for (let c = 0; c < 3; c++) tabla[i * 3 + c] = rgb[a][c] + (rgb[b][c] - rgb[a][c]) * f;
  }
  return tabla;
}
export const INFERNO = rampa(["#000004", "#1b0c41", "#4a0c6b", "#781c6d", "#a52c60", "#cf4446", "#ed6925", "#fb9b06", "#fcffa4"]);
// Activaciones de la red: un solo tono (azul verdoso), de la caja de luz al blanco
export const ACTIVACION = rampa(["#0c1217", "#123544", "#1d5f73", "#3b8fa3", "#7cc3cf", "#d5f1f2"]);

/** Pinta una matriz (alto × ancho) en un canvas con una rampa; valores ya entre 0 y 1. */
export function pintarMatriz(canvas, valores, ancho, alto, tabla) {
  canvas.width = ancho;
  canvas.height = alto;
  const ctx = canvas.getContext("2d");
  const img = ctx.createImageData(ancho, alto);
  for (let i = 0; i < ancho * alto; i++) {
    const k = Math.round(Math.min(1, Math.max(0, valores[i])) * 255) * 3;
    img.data[4 * i] = tabla[k];
    img.data[4 * i + 1] = tabla[k + 1];
    img.data[4 * i + 2] = tabla[k + 2];
    img.data[4 * i + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
  return canvas;
}

/** Pinta un plano gris (Uint8, 256×256) tal cual. */
export function pintarGris(canvas, gris, lado = 256) {
  canvas.width = lado;
  canvas.height = lado;
  const ctx = canvas.getContext("2d");
  const img = ctx.createImageData(lado, lado);
  for (let i = 0; i < gris.length; i++) {
    img.data[4 * i] = img.data[4 * i + 1] = img.data[4 * i + 2] = gris[i];
    img.data[4 * i + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
  return canvas;
}

/** Los PNG de un ejemplo, listos para validarCorte. */
export async function ficherosDeEjemplo(e) {
  return Promise.all(e.ficheros.map(async (nombre) => {
    const r = await fetch(`ejemplos/${encodeURIComponent(e.corte)}/${encodeURIComponent(nombre)}`);
    return { nombre, bytes: new Uint8Array(await r.arrayBuffer()) };
  }));
}

/**
 * Mapa de contribuciones (lado × lado) sobre una fase en gris: rojo, la zona sube la probabilidad de pCR; azul, la
 * baja (la convención de SHAP; el ámbar queda para el caso actual).
 * La intensidad es relativa al valor más grande del mapa (en valor absoluto).
 */
export function lienzoContribucion(mapa, lado, fondo) {
  const canvas = pintarGris(document.createElement("canvas"), fondo);
  let max = 1e-9;
  for (const v of mapa) max = Math.max(max, Math.abs(v));
  // El mapa se pinta a su tamaño (16×16) y se amplía suavizado: cada zona se ve como una mancha, no como un bloque
  const pequeno = document.createElement("canvas");
  pequeno.width = pequeno.height = lado;
  const pctx = pequeno.getContext("2d");
  const img = pctx.createImageData(lado, lado);
  for (let i = 0; i < lado * lado; i++) {
    const v = mapa[i] / max;
    const [r, g, b] = v >= 0 ? [239, 91, 91] : [59, 139, 235];
    img.data.set([r, g, b, Math.round(255 * 0.8 * Math.abs(v))], 4 * i);
  }
  pctx.putImageData(img, 0, 0);
  const ctx = canvas.getContext("2d");
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = "high";
  ctx.drawImage(pequeno, 0, 0, 256, 256);
  return canvas;
}
