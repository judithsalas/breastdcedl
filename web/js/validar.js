// Validación de las 3 fases de un corte en la app web (D38): la misma que src/data/entrada.py.
//
// Mismas reglas, en el mismo orden y con los mismos mensajes. tests/test_web.py lo comprueba: pasa por
// aquí (con Node) y por entrada.py los mismos ficheros, válidos e inválidos, y exige los mismos errores,
// los mismos avisos y los mismos píxeles.
//
// No se normaliza nada: devuelve los píxeles enteros (0–255) en el orden PRE, EARLY, LATE, y la división
// entre 255 la hace el propio modelo ONNX, igual que en el entrenamiento.

import { leerPNG, tieneFirma } from "./png.js";

export const FASES = ["PRE", "EARLY", "LATE"];
export const TAMANO_MAXIMO = 2 * 1024 * 1024; // bytes por fichero; un PNG del dataset ocupa unos 30–70 KB
export const LADO = 256;
const SUFIJO = /^(.+)_(PRE|EARLY|LATE)\.png$/i;

/** ["ISPY1_1001_z012", "EARLY"] a partir de "ISPY1_1001_z012_EARLY.png"; null si no encaja. */
export function identificar(nombre) {
  const base = nombre.trim().split("/").pop().split("\\").pop();
  const m = SUFIJO.exec(base);
  return m ? [m[1], m[2].toUpperCase()] : null;
}

async function sha256(bytes) {
  const huella = await crypto.subtle.digest("SHA-256", bytes);
  return [...new Uint8Array(huella)].map((b) => b.toString(16).padStart(2, "0")).join("");
}

/** Un PNG en memoria → { gris: Uint8Array(256×256) | null, errores }. */
export async function leerFase(nombre, bytes) {
  if (bytes.length > TAMANO_MAXIMO) {
    return { gris: null, errores: [`${nombre}: ocupa ${(bytes.length / 1e6).toFixed(1)} MB; el máximo es ${(TAMANO_MAXIMO / 1e6).toFixed(0)} MB.`] };
  }
  if (!tieneFirma(bytes)) {
    return { gris: null, errores: [`${nombre}: no es un PNG (la firma del fichero no corresponde).`] };
  }
  let png;
  try {
    png = await leerPNG(bytes);
  } catch {
    return { gris: null, errores: [`${nombre}: el fichero está dañado o no se puede leer.`] };
  }
  const errores = [];
  if (png.modo !== "L") errores.push(`${nombre}: debe ser gris de 8 bits (modo L) y es modo ${png.modo}.`);
  if (png.ancho !== LADO || png.alto !== LADO) {
    errores.push(`${nombre}: mide ${png.ancho}×${png.alto} y debe medir ${LADO}×${LADO}.`);
  }
  return errores.length ? { gris: null, errores } : { gris: png.gris, errores };
}

/**
 * Tres ficheros [{ nombre, bytes }] → { pixeles: Uint8Array(3×256×256) PRE·EARLY·LATE | null, errores, avisos }.
 */
export async function validarCorte(ficheros) {
  const errores = [], avisos = [];
  if (ficheros.length !== 3) {
    errores.push(`Hacen falta exactamente 3 ficheros (PRE, EARLY y LATE de un corte); han llegado ${ficheros.length}.`);
  }
  const porFase = Object.fromEntries(FASES.map((f) => [f, []]));
  const cortes = new Set();
  for (const f of ficheros) {
    const ident = identificar(f.nombre);
    if (!ident) {
      errores.push(`${f.nombre}: el nombre debe acabar en _PRE.png, _EARLY.png o _LATE.png (como en el dataset) para saber qué fase es.`);
      continue;
    }
    cortes.add(ident[0].toLowerCase());
    porFase[ident[1]].push(f);
  }
  for (const fase of FASES) {
    const lista = porFase[fase];
    if (lista.length > 1) errores.push(`Fase ${fase} duplicada: ${lista.map((f) => f.nombre).join(", ")}.`);
    else if (!lista.length && ficheros.length <= 3) errores.push(`Falta la fase ${fase}.`);
  }
  if (cortes.size > 1) {
    errores.push(`Las tres fases deben ser del mismo corte y llegan de varios: ${[...cortes].sort().join(", ")}.`);
  }
  const huellas = await Promise.all(ficheros.map((f) => sha256(f.bytes)));
  if (new Set(huellas).size < huellas.length) {
    errores.push("Dos de los ficheros son idénticos: una fase está repetida con otro nombre.");
  }
  if (errores.length) return { pixeles: null, errores, avisos };

  const planos = [];
  for (const fase of FASES) {
    const f = porFase[fase][0];
    const r = await leerFase(f.nombre, f.bytes);
    errores.push(...r.errores);
    planos.push(r.gris);
  }
  if (errores.length) return { pixeles: null, errores, avisos };

  const n = LADO * LADO;
  const pixeles = new Uint8Array(3 * n);
  planos.forEach((p, i) => pixeles.set(p, i * n));
  // Aviso si las fases parecen cambiadas: en el tejido (PRE > 0,1), EARLY debe ser más brillante que PRE.
  let tejido = 0, sumaPre = 0, sumaEarly = 0;
  for (let i = 0; i < n; i++) {
    if (pixeles[i] / 255 > 0.1) {
      tejido++;
      sumaPre += pixeles[i];
      sumaEarly += pixeles[n + i];
    }
  }
  if (tejido > 50 && sumaEarly <= sumaPre) {
    avisos.push("En el tejido EARLY no es más brillante que PRE. En el dataset siempre lo es: comprueba que las fases no estén cambiadas.");
  }
  return { pixeles, errores, avisos };
}
