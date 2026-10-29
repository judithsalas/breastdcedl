// Lector de PNG sin dependencias, para la app web (D38).
//
// Hace lo mismo que Pillow en src/data/entrada.py: comprueba la firma, que cada trozo esté entero y con
// su CRC, que los datos de imagen se descompriman enteros y que los filtros de cada fila sean válidos.
// Si algo falla, lanza un error («dañado»). Devuelve el modo con el nombre que le da Pillow ("L", "RGB",
// "I;16"…) y, si es gris de 8, 4 o 2 bits, los píxeles de 0 a 255 tal cual (como Image.convert("L")).
// No usa el navegador para pintar la imagen: un <canvas> puede aplicar corrección de color y cambiar
// los valores, y aquí tienen que ser exactamente los del fichero.

export const FIRMA = Uint8Array.from([137, 80, 78, 71, 13, 10, 26, 10]);

const TABLA_CRC = (() => {
  const tabla = new Uint32Array(256);
  for (let n = 0; n < 256; n++) {
    let c = n;
    for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    tabla[n] = c >>> 0;
  }
  return tabla;
})();

function crc32(bytes, inicio, fin) {
  let c = 0xffffffff;
  for (let i = inicio; i < fin; i++) c = TABLA_CRC[(c ^ bytes[i]) & 0xff] ^ (c >>> 8);
  return (c ^ 0xffffffff) >>> 0;
}

// Nombre del modo en Pillow según (profundidad, tipo de color) del PNG (PngImagePlugin._MODES).
const MODOS = {
  "1,0": "1", "2,0": "L", "4,0": "L", "8,0": "L", "16,0": "I;16",
  "8,2": "RGB", "16,2": "RGB",
  "1,3": "P", "2,3": "P", "4,3": "P", "8,3": "P",
  "8,4": "LA", "16,4": "LA",
  "8,6": "RGBA", "16,6": "RGBA",
};
const CANALES = { 0: 1, 2: 3, 3: 1, 4: 2, 6: 4 };
// Pasadas del entrelazado Adam7: [x0, y0, paso x, paso y].
const ADAM7 = [[0, 0, 8, 8], [4, 0, 8, 8], [0, 4, 4, 8], [2, 0, 4, 4], [0, 2, 2, 4], [1, 0, 2, 2], [0, 1, 1, 2]];

export class PngDanado extends Error {}

export function tieneFirma(bytes) {
  if (bytes.length < FIRMA.length) return false;
  return FIRMA.every((b, i) => bytes[i] === b);
}

function leerTrozos(bytes) {
  const vista = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const trozos = [];
  let pos = FIRMA.length;
  for (;;) {
    if (pos + 12 > bytes.length) throw new PngDanado("el fichero se corta antes del final");
    const largo = vista.getUint32(pos);
    const tipo = String.fromCharCode(...bytes.subarray(pos + 4, pos + 8));
    if (!/^[A-Za-z]{4}$/.test(tipo)) throw new PngDanado("trozo con un tipo no válido");
    const fin = pos + 8 + largo;
    if (fin + 4 > bytes.length) throw new PngDanado("el fichero se corta antes del final");
    if (crc32(bytes, pos + 4, fin) !== vista.getUint32(fin)) throw new PngDanado(`CRC incorrecto en ${tipo}`);
    trozos.push({ tipo, datos: bytes.subarray(pos + 8, fin) });
    pos = fin + 4;
    if (tipo === "IEND") return trozos;
  }
}

async function descomprimir(datos) {
  const flujo = new Blob([datos]).stream().pipeThrough(new DecompressionStream("deflate"));
  const partes = [];
  let total = 0;
  const lector = flujo.getReader();
  for (;;) {
    const { value, done } = await lector.read();
    if (done) break;
    partes.push(value);
    total += value.length;
  }
  const salida = new Uint8Array(total);
  let pos = 0;
  for (const p of partes) {
    salida.set(p, pos);
    pos += p.length;
  }
  return salida;
}

function paeth(a, b, c) {
  const p = a + b - c;
  const pa = Math.abs(p - a), pb = Math.abs(p - b), pc = Math.abs(p - c);
  return pa <= pb && pa <= pc ? a : pb <= pc ? b : c;
}

// Quita los filtros de una imagen (o de una pasada de Adam7) de `alto` filas de `bytesFila` bytes.
function desfiltrar(crudo, pos, alto, bytesFila, bpp) {
  const salida = new Uint8Array(alto * bytesFila);
  for (let y = 0; y < alto; y++) {
    const filtro = crudo[pos++];
    if (filtro > 4) throw new PngDanado(`filtro de fila desconocido (${filtro})`);
    const fila = y * bytesFila, previa = fila - bytesFila;
    for (let i = 0; i < bytesFila; i++) {
      const x = crudo[pos++];
      const a = i >= bpp ? salida[fila + i - bpp] : 0;
      const b = y > 0 ? salida[previa + i] : 0;
      const c = y > 0 && i >= bpp ? salida[previa + i - bpp] : 0;
      const v = filtro === 0 ? x : filtro === 1 ? x + a : filtro === 2 ? x + b
        : filtro === 3 ? x + ((a + b) >> 1) : x + paeth(a, b, c);
      salida[fila + i] = v & 0xff;
    }
  }
  return { datos: salida, pos };
}

// Gris de 8, 4 o 2 bits → valores de 0 a 255 (Pillow: ×17 con 4 bits, ×85 con 2 bits).
function grisFila(datos, fila, bytesFila, ancho, profundidad) {
  const salida = new Uint8Array(ancho);
  if (profundidad === 8) return datos.subarray(fila * bytesFila, fila * bytesFila + ancho);
  const porByte = 8 / profundidad, mascara = (1 << profundidad) - 1, escala = 255 / mascara;
  for (let x = 0; x < ancho; x++) {
    const byte = datos[fila * bytesFila + Math.floor(x / porByte)];
    const desplazamiento = 8 - profundidad * (1 + (x % porByte));
    salida[x] = ((byte >> desplazamiento) & mascara) * escala;
  }
  return salida;
}

/**
 * Lee un PNG entero. Devuelve { ancho, alto, modo, gris } (gris: Uint8Array ancho×alto, solo en modo "L").
 * Lanza PngDanado si el fichero está roto o cortado.
 */
export async function leerPNG(bytes) {
  if (!tieneFirma(bytes)) throw new PngDanado("no tiene la firma de PNG");
  const trozos = leerTrozos(bytes);
  if (trozos[0].tipo !== "IHDR" || trozos[0].datos.length !== 13) throw new PngDanado("falta la cabecera IHDR");
  const h = new DataView(trozos[0].datos.buffer, trozos[0].datos.byteOffset, 13);
  const ancho = h.getUint32(0), alto = h.getUint32(4);
  const profundidad = h.getUint8(8), tipoColor = h.getUint8(9), entrelazado = h.getUint8(12);
  const modo = MODOS[`${profundidad},${tipoColor}`];
  if (!modo || ancho === 0 || alto === 0 || entrelazado > 1) throw new PngDanado("cabecera IHDR no válida");

  const idat = trozos.filter((t) => t.tipo === "IDAT");
  if (!idat.length) throw new PngDanado("no hay datos de imagen");
  const juntos = new Uint8Array(idat.reduce((s, t) => s + t.datos.length, 0));
  let pos = 0;
  for (const t of idat) {
    juntos.set(t.datos, pos);
    pos += t.datos.length;
  }
  let crudo;
  try {
    crudo = await descomprimir(juntos);
  } catch {
    throw new PngDanado("los datos de imagen no se pueden descomprimir");
  }

  const bitsPixel = profundidad * CANALES[tipoColor];
  const bpp = Math.max(1, bitsPixel >> 3);
  const pasadas = entrelazado ? ADAM7 : [[0, 0, 1, 1]];
  const necesario = pasadas.reduce((s, [x0, y0, dx, dy]) => {
    const w = Math.ceil((ancho - x0) / dx), hh = Math.ceil((alto - y0) / dy);
    return w > 0 && hh > 0 ? s + hh * (1 + Math.ceil((w * bitsPixel) / 8)) : s;
  }, 0);
  if (crudo.length < necesario) throw new PngDanado("faltan datos de imagen");

  const gris = modo === "L" ? new Uint8Array(ancho * alto) : null;
  pos = 0;
  for (const [x0, y0, dx, dy] of pasadas) {
    const w = Math.ceil((ancho - x0) / dx), hh = Math.ceil((alto - y0) / dy);
    if (w <= 0 || hh <= 0) continue;
    const bytesFila = Math.ceil((w * bitsPixel) / 8);
    const r = desfiltrar(crudo, pos, hh, bytesFila, bpp);
    pos = r.pos;
    if (!gris) continue;
    for (let y = 0; y < hh; y++) {
      const fila = grisFila(r.datos, y, bytesFila, w, profundidad);
      for (let x = 0; x < w; x++) gris[(y0 + y * dy) * ancho + x0 + x * dx] = fila[x];
    }
  }
  return { ancho, alto, modo, gris };
}
