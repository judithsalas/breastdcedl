// El modelo final en el navegador (D38), con ONNX Runtime Web.
//
// modelo.onnx lo genera `python -m src.models.exportar_web` a partir de los pesos de models/final/: dentro van
// la división entre 255, las 30 CNN, la media de logits, la calibración de Platt y la sigmoide. Aquí solo se le
// pasan los píxeles enteros del corte. Antes de usarlo se comprueba que el fichero descargado tiene el sha256
// que dice info.json, y la sesión es solo de inferencia (sin gradientes), con el modelo exportado en modo eval.

export const VERSION_ORT = "1.20.1";
const CDNS = [
  `https://cdn.jsdelivr.net/npm/onnxruntime-web@${VERSION_ORT}/dist/`,
  `https://unpkg.com/onnxruntime-web@${VERSION_ORT}/dist/`,
];

function cargarScript(url) {
  return new Promise((ok, mal) => {
    const s = document.createElement("script");
    s.src = url;
    s.crossOrigin = "anonymous";
    s.onload = ok;
    s.onerror = () => {
      s.remove();
      mal(new Error(`No se pudo cargar ${url}`));
    };
    document.head.append(s);
  });
}

let promesaOrt = null;

/** ONNX Runtime Web, cargado una sola vez por página (de jsDelivr o, si falla, de unpkg). */
export function cargarOrt() {
  promesaOrt ??= (async () => {
    for (const base of CDNS) {
      try {
        await cargarScript(`${base}ort.wasm.min.js`);
        window.ort.env.wasm.wasmPaths = base;
        return window.ort;
      } catch {
        // se prueba el siguiente
      }
    }
    promesaOrt = null;
    throw new Error("No se pudo descargar ONNX Runtime Web. Comprueba la conexión y recarga la página.");
  })();
  return promesaOrt;
}

export async function descargar(url, alProgresar) {
  const respuesta = await fetch(url);
  if (!respuesta.ok) throw new Error(`No se pudo descargar ${url} (${respuesta.status}).`);
  const total = Number(respuesta.headers.get("content-length")) || 0;
  const lector = respuesta.body.getReader();
  const partes = [];
  let recibidos = 0;
  for (;;) {
    const { value, done } = await lector.read();
    if (done) break;
    partes.push(value);
    recibidos += value.length;
    if (alProgresar) alProgresar(recibidos, total);
  }
  const bytes = new Uint8Array(recibidos);
  let pos = 0;
  for (const p of partes) {
    bytes.set(p, pos);
    pos += p.length;
  }
  return bytes;
}

export async function sha256(bytes) {
  const huella = await crypto.subtle.digest("SHA-256", bytes);
  return [...new Uint8Array(huella)].map((b) => b.toString(16).padStart(2, "0")).join("");
}

// Hugging Face sirve los .onnx con «Cache-Control: no-store», así que el navegador los volvería a descargar en cada
// visita. Se guardan en la caché del navegador con su sha256 en la clave: solo se reutilizan si son el mismo fichero
// y se comprueban al leerlos. Si la caché no está disponible (modo privado, sin espacio), se descarga y ya está.
const CACHE = "pcr-rm-mama-modelos";
async function descargarGuardado(url, huella, alProgresar) {
  let cache = null;
  try {
    cache = await caches.open(CACHE);
  } catch {
    cache = null;
  }
  const clave = new URL(`${url}?sha256=${huella}`, location.href).href;
  if (cache) {
    try {
      const guardada = await cache.match(clave);
      if (guardada) {
        const bytes = new Uint8Array(await guardada.arrayBuffer());
        if ((await sha256(bytes)) === huella) {
          if (alProgresar) alProgresar(bytes.length, bytes.length);
          return bytes;
        }
        await cache.delete(clave);
      }
    } catch {
      /* se descarga */
    }
  }
  const bytes = await descargar(url, alProgresar);
  if (cache && (await sha256(bytes)) === huella) {
    try {
      const base = clave.split("?")[0];
      for (const r of await cache.keys()) if (r.url.split("?")[0] === base && r.url !== clave) await cache.delete(r);
      await cache.put(clave, new Response(bytes));
    } catch {
      /* el navegador puede no dejar guardarlo: la próxima vez se descarga otra vez */
    }
  }
  return bytes;
}

/** Descarga y comprueba el modelo. Devuelve { predecir(pixeles) → { probabilidad, ms }, dispositivo }. */
export async function cargarModelo(info, alProgresar) {
  const [ort, bytes] = await Promise.all([cargarOrt(), descargarGuardado(`modelo/${info.onnx.fichero}`, info.onnx.sha256, alProgresar)]);
  const huella = await sha256(bytes);
  if (huella !== info.onnx.sha256) {
    throw new Error(`El fichero del modelo no coincide con su sha256 (${huella.slice(0, 12)}… en vez de ${info.onnx.sha256.slice(0, 12)}…).`);
  }
  const hilos = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1;
  ort.env.wasm.numThreads = hilos;
  const sesion = await ort.InferenceSession.create(bytes, { executionProviders: ["wasm"], graphOptimizationLevel: "all" });
  const dispositivo = `CPU de este navegador (WebAssembly${hilos > 1 ? `, ${hilos} hilos` : ""}), ONNX Runtime Web ${VERSION_ORT}`;

  async function predecir(pixeles) {
    const entrada = new ort.Tensor("uint8", pixeles, [1, 3, 256, 256]);
    const inicio = performance.now();
    const salida = await sesion.run({ fases: entrada });
    const ms = performance.now() - inicio;
    const mapa = salida.mapa ? { datos: Float32Array.from(salida.mapa.data), lado: salida.mapa.dims[2] } : null;
    return { probabilidad: Number(salida.prob.data[0]), ms, mapa };
  }
  return { predecir, dispositivo, huella };
}

/** Una red suelta (una_red.onnx), comprobada con su sha256: devuelve todas sus salidas como tensores de ORT. */
export async function cargarUnaRed(descripcion) {
  const [ort, bytes] = await Promise.all([cargarOrt(), descargarGuardado(`modelo/${descripcion.fichero}`, descripcion.sha256)]);
  if ((await sha256(bytes)) !== descripcion.sha256) throw new Error("El fichero de la red no coincide con su sha256.");
  ort.env.wasm.numThreads = 1;
  const sesion = await ort.InferenceSession.create(bytes, { executionProviders: ["wasm"] });
  return async (pixeles) => sesion.run({ fases: new ort.Tensor("uint8", pixeles, [1, 3, 256, 256]) });
}
