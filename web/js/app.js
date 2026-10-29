// Interfaz de la app web (D38). La validación está en validar.js (igual que src/data/entrada.py) y el modelo en
// modelo.js (el ONNX exportado del modelo final). Aquí solo se dibuja y se enseña.
// Todo el texto que viene de fuera (nombres de fichero) se escribe con textContent: nunca se interpreta como HTML.

import { lienzoContribucion } from "./comun.js";
import { cargarModelo } from "./modelo.js";
import { FASES, LADO, TAMANO_MAXIMO, validarCorte } from "./validar.js";

const LECTURA_MAXIMA = 50 * 1024 * 1024; // ficheros más grandes ni se leen: se rechazan por tamaño
const COHORTES = { spy1: "I-SPY1", spy2: "I-SPY2", duke: "Duke" };
// Mapa de color «inferno» para el realce (de matplotlib): negro → violeta → rojo → naranja → amarillo pálido.
const INFERNO = ["#000004", "#1b0c41", "#4a0c6b", "#781c6d", "#a52c60", "#cf4446", "#ed6925", "#fb9b06", "#fcffa4"];
const REALCE_MAXIMO = 0.4; // EARLY − PRE (en [0, 1]) que se pinta con el color más claro

const $ = (id) => document.getElementById(id);
const estado = { info: null, modelo: null, modeloListo: null, peticion: 0 };

// ---------- Formato ----------
const AUMENTOS = { geometrico: "geométrico", "geometrico+giros90": "geométrico y giros de 90°", "geometrico+escaner": "geométrico y de escáner" };
const SUPER = { "-": "⁻", 0: "⁰", 1: "¹", 2: "²", 3: "³", 4: "⁴", 5: "⁵", 6: "⁶", 7: "⁷", 8: "⁸", 9: "⁹" };
function cientifica(x) {
  if (!x) return "0";
  const exp = Math.floor(Math.log10(Math.abs(x)));
  const mant = Math.round(x / 10 ** exp);
  return `${mant}·10${String(exp).replace(/./g, (c) => SUPER[c])}`;
}
const num = (x, d = 3) => (x == null || Number.isNaN(x) ? "—" : x.toFixed(d).replace(".", ","));
const pct = (x) => `${Math.round(100 * x)} %`;
const miles = (n) => (n == null ? "—" : String(Math.round(Number(n))).replace(/\B(?=(\d{3})+(?!\d))/g, "."));
function prob(p) {
  if (p < 0.001) return "< 0,001";
  if (p > 0.999) return "> 0,999";
  return num(p);
}
function nodo(etiqueta, texto, clase) {
  const n = document.createElement(etiqueta);
  if (texto != null) n.textContent = texto;
  if (clase) n.className = clase;
  return n;
}

// ---------- Imágenes ----------
function hexARgb(hex) {
  return [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16));
}
const LUT = (() => {
  const anclas = INFERNO.map(hexARgb);
  const tabla = new Uint8ClampedArray(256 * 3);
  for (let i = 0; i < 256; i++) {
    const t = (i / 255) * (anclas.length - 1);
    const a = Math.floor(t), b = Math.min(a + 1, anclas.length - 1), f = t - a;
    for (let c = 0; c < 3; c++) tabla[i * 3 + c] = anclas[a][c] + (anclas[b][c] - anclas[a][c]) * f;
  }
  return tabla;
})();

function pintar(canvas, rellenar) {
  const ctx = canvas.getContext("2d");
  const imagen = ctx.createImageData(LADO, LADO);
  rellenar(imagen.data);
  ctx.putImageData(imagen, 0, 0);
}
function pintarGris(canvas, gris) {
  pintar(canvas, (d) => {
    for (let i = 0; i < gris.length; i++) {
      d[4 * i] = d[4 * i + 1] = d[4 * i + 2] = gris[i];
      d[4 * i + 3] = 255;
    }
  });
}
function pintarRealce(canvas, pre, early) {
  pintar(canvas, (d) => {
    for (let i = 0; i < pre.length; i++) {
      const t = Math.min(1, Math.max(0, (early[i] - pre[i]) / 255 / REALCE_MAXIMO));
      const k = Math.round(t * 255) * 3;
      d[4 * i] = LUT[k];
      d[4 * i + 1] = LUT[k + 1];
      d[4 * i + 2] = LUT[k + 2];
      d[4 * i + 3] = 255;
    }
  });
}
function pintarFases(pixeles) {
  const n = LADO * LADO;
  const planos = FASES.map((_, i) => pixeles.subarray(i * n, (i + 1) * n));
  const lienzos = document.querySelectorAll("#fases canvas");
  planos.forEach((p, i) => pintarGris(lienzos[i], p));
  pintarRealce(lienzos[3], planos[0], planos[1]);
  lienzos[4].getContext("2d").clearRect(0, 0, LADO, LADO);    // el mapa llega con la predicción
  estado.early = planos[1];
  document.querySelector(".caja-luz").setAttribute("data-lleno", "");
}
function vaciarFases() {
  for (const c of document.querySelectorAll("#fases canvas")) c.getContext("2d").clearRect(0, 0, LADO, LADO);
  document.querySelector(".caja-luz").removeAttribute("data-lleno");
}

// ---------- Mensajes ----------
function mensajes(lista) {
  const caja = $("mensajes");
  caja.replaceChildren(...lista.map(([tipo, texto]) => nodo("p", texto, `mensaje mensaje--${tipo}`)));
}

// ---------- La regla de probabilidad ----------
const SVG = "http://www.w3.org/2000/svg";
function svg(etiqueta, atributos, texto) {
  const n = document.createElementNS(SVG, etiqueta);
  for (const [k, v] of Object.entries(atributos)) n.setAttribute(k, v);
  if (texto != null) n.textContent = texto;
  return n;
}
function dibujarRegla(p) {
  const { umbral, referencia } = estado.info;
  const franja = referencia && referencia.franja_90;
  const regla = $("regla");
  // El ancho del dibujo sigue al de la pantalla, para que el texto no se encoja en el móvil
  const W = Math.round(Math.max(320, Math.min(640, regla.getBoundingClientRect().width || 600)));
  regla.setAttribute("viewBox", `0 0 ${W} 150`);
  const X0 = 20, X1 = W - 20;
  const arriba = (v) => X0 + v * (X1 - X0);
  let lo = Math.min(p, umbral, franja ? franja[0] : umbral) - 0.02;
  let hi = Math.max(p, umbral, franja ? franja[1] : umbral) + 0.02;
  if (hi - lo < 0.08) {
    const c = (hi + lo) / 2;
    lo = c - 0.04;
    hi = c + 0.04;
  }
  lo = Math.max(0, lo);
  hi = Math.min(1, hi);
  const abajo = (v) => X0 + ((v - lo) / (hi - lo)) * (X1 - X0);
  const linea = "#9fb0bd", tinta = "#18232e", ambar = "#f0a33a";
  const hijos = [
    // Escala entera, de 0 a 1
    svg("line", { x1: X0, x2: X1, y1: 26, y2: 26, stroke: linea, "stroke-width": 2 }),
    ...[0, 0.5, 1].flatMap((v) => [
      svg("line", { x1: arriba(v), x2: arriba(v), y1: 21, y2: 31, stroke: linea, "stroke-width": 2 }),
      svg("text", { x: arriba(v), y: 14, "text-anchor": "middle" }, num(v, v === 0.5 ? 1 : 0)),
    ]),
    svg("rect", { x: arriba(lo), y: 20, width: Math.max(2, arriba(hi) - arriba(lo)), height: 12, fill: "none", stroke: tinta, "stroke-width": 1.5, rx: 2 }),
    svg("circle", { cx: arriba(p), cy: 26, r: 4, fill: ambar }),
    // Lupa: de la ventana de arriba a la escala ampliada
    svg("path", { d: `M${arriba(lo)} 32 L${X0} 84 M${arriba(hi)} 32 L${X1} 84`, stroke: linea, "stroke-width": 1, fill: "none" }),
  ];
  if (franja) {
    hijos.push(svg("rect", { x: abajo(franja[0]), y: 86, width: abajo(franja[1]) - abajo(franja[0]), height: 26, fill: "#d9e1e7", rx: 2 }));
  }
  hijos.push(
    svg("line", { x1: X0, x2: X1, y1: 99, y2: 99, stroke: tinta, "stroke-width": 1.5 }),
    svg("text", { x: X0, y: 132, "text-anchor": "start" }, num(lo, 2)),
    svg("text", { x: X1, y: 132, "text-anchor": "end" }, num(hi, 2)),
    svg("line", { x1: abajo(umbral), x2: abajo(umbral), y1: 80, y2: 118, stroke: tinta, "stroke-width": 2, "stroke-dasharray": "4 3" }),
    svg("text", { x: abajo(umbral), y: 132, "text-anchor": "middle", class: "regla__fuerte" }, `umbral ${num(umbral)}`),
  );
  const aguja = svg("g", { class: "regla__aguja" });
  aguja.append(
    svg("line", { x1: 0, x2: 0, y1: 84, y2: 114, stroke: ambar, "stroke-width": 3 }),
    svg("circle", { cx: 0, cy: 99, r: 7, fill: ambar, stroke: "#fff", "stroke-width": 2 }),
    svg("text", { x: 0, y: 74, "text-anchor": "middle", class: "regla__corte" }, `este corte, ${prob(p)}`),
  );
  hijos.push(aguja);
  regla.replaceChildren(...hijos);
  aguja.style.transition = "none";
  aguja.style.transform = `translateX(${abajo(lo)}px)`;
  requestAnimationFrame(() => requestAnimationFrame(() => {
    aguja.style.transition = "transform 0.7s cubic-bezier(0.2, 0.7, 0.2, 1)";
    aguja.style.transform = `translateX(${abajo(p)}px)`;
  }));
}

// ---------- Resultado ----------
function mostrarResultado(r, esperado) {
  const { umbral } = estado.info;
  const panel = $("resultado");
  panel.dataset.estado = "lleno";
  $("probabilidad").textContent = prob(r.probabilidad);
  const clase = $("clase");
  const esPcr = r.probabilidad >= umbral;
  clase.replaceChildren(
    nodo("strong", esPcr ? "pCR" : "no pCR"),
    document.createTextNode(esPcr
      ? `: está por encima del umbral (${num(umbral)}), así que el modelo lo clasifica como respuesta completa.`
      : `: está por debajo del umbral (${num(umbral)}), así que el modelo lo clasifica como enfermedad residual.`),
  );
  dibujarRegla(r.probabilidad);
  if (r.mapa && estado.early) {
    const lienzo = document.querySelectorAll("#fases canvas")[4];
    lienzo.getContext("2d").drawImage(lienzoContribucion(r.mapa.datos, r.mapa.lado, estado.early), 0, 0);
  }
  $("dato-umbral").textContent = `${num(umbral)} (${estado.info.criterio_umbral})`;
  $("dato-huella").textContent = estado.info.huella_modelo;
  $("dato-dispositivo").textContent = estado.modelo.dispositivo;
  $("dato-tiempo").textContent = `${Math.round(r.ms)} ms`;
  const coincide = $("coincide");
  if (esperado == null) {
    coincide.hidden = true;
  } else {
    const ok = Math.abs(esperado - r.probabilidad) < 1e-5;
    coincide.hidden = false;
    coincide.dataset.ok = ok ? "si" : "no";
    coincide.textContent = `${ok ? "✓ Coincide" : "✗ No coincide"} con el pipeline de Python: ${num(r.probabilidad, 6)} aquí y ${num(esperado, 6)} en el pipeline.`;
  }
}

async function analizar(ficheros, esperado = null) {
  const peticion = ++estado.peticion;
  const r = await validarCorte(ficheros);
  if (peticion !== estado.peticion) return;
  const lista = [...r.errores.map((m) => ["error", m]), ...r.avisos.map((m) => ["aviso", m])];
  if (r.errores.length) {
    mensajes(lista);
    vaciarFases();
    $("resultado").dataset.estado = "vacio";
    return;
  }
  pintarFases(r.pixeles);
  const panel = $("resultado");
  if (!estado.modelo) {
    mensajes([...lista, ["info", "El modelo todavía se está descargando; la predicción saldrá en cuanto termine."]]);
  } else {
    mensajes(lista);
  }
  if (panel.dataset.estado === "lleno") panel.dataset.estado = "calculando";
  try {
    await estado.modeloListo;
    const resultado = await estado.modelo.predecir(r.pixeles);
    if (peticion !== estado.peticion) return;
    mensajes(lista);
    mostrarResultado(resultado, esperado);
    if (window.matchMedia("(max-width: 900px)").matches) {
      const suave = !window.matchMedia("(prefers-reduced-motion: reduce)").matches;
      panel.scrollIntoView({ behavior: suave ? "smooth" : "auto", block: "start" });
    }
  } catch (error) {
    mensajes([...lista, ["error", `No se pudo calcular la predicción: ${error.message}`]]);
  }
}

// ---------- Subir ficheros ----------
async function leerFicheros(lista) {
  const grandes = lista.filter((f) => f.size > LECTURA_MAXIMA);
  if (grandes.length) {
    mensajes(grandes.map((f) => ["error", `${f.name}: ocupa ${(f.size / 1e6).toFixed(1)} MB; el máximo es ${(TAMANO_MAXIMO / 1e6).toFixed(0)} MB.`]));
    return;
  }
  const ficheros = await Promise.all(lista.map(async (f) => ({ nombre: f.name, bytes: new Uint8Array(await f.arrayBuffer()) })));
  for (const b of document.querySelectorAll(".ejemplos button")) b.setAttribute("aria-pressed", "false");
  await analizar(ficheros);
}

function prepararSubida() {
  const entrada = $("ficheros"), zona = $("soltar");
  entrada.addEventListener("change", () => {
    const lista = [...entrada.files];
    entrada.value = "";
    if (lista.length) leerFicheros(lista);
  });
  zona.addEventListener("dragover", (e) => {
    e.preventDefault();
    zona.setAttribute("data-encima", "");
  });
  zona.addEventListener("dragleave", () => zona.removeAttribute("data-encima"));
  zona.addEventListener("drop", (e) => {
    e.preventDefault();
    zona.removeAttribute("data-encima");
    const lista = [...e.dataTransfer.files];
    if (lista.length) leerFicheros(lista);
  });
}

// ---------- Pestañas ----------
function prepararPestanas() {
  const pestanas = [$("pestana-subir"), $("pestana-ejemplo")];
  const elegir = (activa) => {
    for (const p of pestanas) {
      const si = p === activa;
      p.setAttribute("aria-selected", String(si));
      p.tabIndex = si ? 0 : -1;
      $(p.getAttribute("aria-controls")).hidden = !si;
    }
  };
  pestanas.forEach((p, i) => {
    p.addEventListener("click", () => elegir(p));
    p.addEventListener("keydown", (e) => {
      if (e.key !== "ArrowRight" && e.key !== "ArrowLeft") return;
      const otra = pestanas[(i + (e.key === "ArrowRight" ? 1 : pestanas.length - 1)) % pestanas.length];
      elegir(otra);
      otra.focus();
    });
  });
}

// ---------- Ejemplos ----------
async function prepararEjemplos() {
  const ul = $("ejemplos");
  let lista;
  try {
    const r = await fetch("ejemplos/ejemplos.json");
    if (!r.ok) throw new Error(String(r.status));
    lista = await r.json();
  } catch {
    ul.replaceChildren(nodo("li", "No se han podido cargar los ejemplos.", "nota"));
    return;
  }
  ul.replaceChildren(...lista.map((e) => {
    const boton = nodo("button");
    boton.type = "button";
    boton.setAttribute("aria-pressed", "false");
    const etiqueta = e.pcr == null ? "" : e.pcr ? "etiqueta real: pCR" : "etiqueta real: no pCR";
    boton.append(nodo("span", COHORTES[e.cohorte] || e.cohorte || "Ejemplo", "ejemplo__cohorte"),
      nodo("span", etiqueta, "ejemplo__etiqueta"), nodo("span", e.corte, "ejemplo__corte"));
    boton.addEventListener("click", async () => {
      for (const b of ul.querySelectorAll("button")) b.setAttribute("aria-pressed", String(b === boton));
      const ficheros = await Promise.all(e.ficheros.map(async (nombre) => {
        const r = await fetch(`ejemplos/${encodeURIComponent(e.corte)}/${encodeURIComponent(nombre)}`);
        return { nombre, bytes: new Uint8Array(await r.arrayBuffer()) };
      }));
      await analizar(ficheros, e.probabilidad_pipeline);
    });
    const li = nodo("li");
    li.append(boton);
    return li;
  }));
}

// ---------- Datos del modelo ----------
function lista(dl, filas) {
  dl.replaceChildren(...filas.flatMap(([t, d]) => [nodo("dt", t), nodo("dd", d)]));
}
function rellenarModelo(info) {
  const m = info.metricas, e = info.entrenamiento, ref = info.referencia || {};
  $("ficha-huella").textContent = info.huella_modelo;
  $("ficha-umbral").textContent = num(info.umbral);
  $("modelo-que").textContent = `${e.redes} redes convolucionales propias (${e.versiones.length} versiones × 5 folds), `
    + "entrenadas desde cero con PyTorch. Para un corte, cada versión hace la media de los logits de sus redes, y después "
    + `se hace la media entre versiones. La probabilidad se calibra con Platt (a = ${num(info.calibracion.a)}, `
    + `b = ${num(info.calibracion.b)}), y el umbral es el de ${info.criterio_umbral}, elegido con validación cruzada.`;
  lista($("modelo-entrenamiento"), [
    ["Entrada", info.entrada],
    ["Parámetros", `${miles(e.parametros)} en ${e.redes} redes`],
    ["Optimizador", `${e.optimizador}, learning rate ${String(e.learning_rate).replace(".", ",")}, batch ${e.batch}`],
    ["Épocas", e.eleccion_epoca === "ultima" ? `${e.epocas}, fijadas de antemano; se usa la última` : `${e.epocas} como máximo`],
    ["Pérdida", e.perdida === "normal" ? "entropía cruzada binaria, sin pesos" : e.perdida],
    ["Semilla", String(e.semilla)],
    ["Entrenado en", e.gpu || "—"],
  ]);
  const filas = [
    ["AUC por paciente", `${num(m.auc_val)} ± ${num(m.auc_val_desviacion)} (folds: ${(m.auc_val_por_fold || []).map((v) => num(v)).join("; ")})`],
    ["AUC en entrenamiento", num(m.auc_train)],
  ];
  if (m.auc_oof != null) filas.push(["Todas juntas, fuera de fold", `${num(m.auc_oof)} (IC 95 % ${num(m.auc_oof_ic95[0], 2)}–${num(m.auc_oof_ic95[1], 2)})`]);
  const vi = info.validacion_interna;
  filas.push(
    ["Con el umbral", `sensibilidad ${num(m.sensibilidad)}, especificidad ${num(m.especificidad)}, precisión pCR ${num(m.precision_pcr)}`
      + " (con las mismas pacientes con que se eligió)"],
  );
  if (vi) filas.push(["Con el umbral, elegido sin el fold", `sensibilidad ${num(vi.sensibilidad)}, especificidad ${num(vi.especificidad)}, precisión pCR ${num(vi.precision_pcr)}`]);
  filas.push(["F1 macro", num(m.f1_macro)],
    ["Error de Brier", `${num(m.brier, 4)}${vi ? ` (${num(vi.brier_platt_cruzado, 4)} calibrando sin el fold; decir a todas la tasa de pCR da ${num(vi.brier_constante, 4)})` : ""}`]);
  if (ref.medico_auc != null) filas.push(["Médico sin imagen", `${num(ref.medico_auc)} (edad, menopausia, HR y HER2)`]);
  lista($("modelo-metricas"), filas);
  $("modelo-versiones").tBodies[0].replaceChildren(...e.versiones.map((v) => {
    const tr = nodo("tr");
    tr.append(nodo("td", v.version), nodo("td", v.que_cambia || "—"), nodo("td", miles(v.parametros)),
      nodo("td", AUMENTOS[v.aumento] || v.aumento || "no"), nodo("td", v.auc_val == null ? "—" : `${num(v.auc_val)} ± ${num(v.auc_val_desviacion)}`));
    return tr;
  }));
  const o = info.onnx;
  $("modelo-onnx").textContent = `En esta página el modelo es un fichero ONNX (${num(o.bytes / 1e6, 1)} MB, sha256 ${o.sha256.slice(0, 12)}…), `
    + `exportado de PyTorch ${o.torch} con la división entre 255 dentro. Antes de publicarlo se comprobó que da lo mismo que `
    + `PyTorch en ${o.comprobacion.cortes} cortes (diferencia máxima ${cientifica(o.comprobacion.diferencia_maxima)}).`;

  const partes = [`En validación cruzada, el modelo puntúa más alto a la paciente con pCR en ${pct(m.auc_val)} de las parejas `
    + `(AUC ${num(m.auc_val)} ± ${num(m.auc_val_desviacion)}; 0,5 sería el azar).`,
  `Con el umbral detecta ${Math.round(100 * m.sensibilidad)} de cada 100 pCR y acierta ${Math.round(100 * m.especificidad)} de cada 100 «no pCR»; `
    + `de cada 100 pacientes que marca como pCR, ${Math.round(100 * m.precision_pcr)} lo son. Son cifras internas; con pacientes nuevas pueden ser peores.`];
  if (ref.medico_auc != null) partes.push(`Un médico que solo sabe la edad, la menopausia y el subtipo (HR, HER2) llega a ${num(ref.medico_auc)}.`);
  if (ref.franja_90) partes.push(`Al 90 % de las pacientes las sitúa entre ${num(ref.franja_90[0], 2)} y ${num(ref.franja_90[1], 2)}: `
    + `la probabilidad apenas se aparta de la tasa de pCR de entrenamiento (${pct(ref.tasa_pcr)}).`);
  partes.push("Esta predicción usa un solo corte; el modelo se evaluó por paciente, con la media de sus cortes.");
  $("texto-incertidumbre").textContent = partes.join(" ");
}

// ---------- Arranque ----------
async function arrancar() {
  prepararPestanas();
  prepararSubida();
  prepararEjemplos();
  try {
    const r = await fetch("modelo/info.json");
    if (!r.ok) throw new Error(`info.json (${r.status})`);
    estado.info = await r.json();
    rellenarModelo(estado.info);
  } catch (error) {
    $("ficha-estado").textContent = "No disponible";
    mensajes([["error", `No se pudo leer la información del modelo: ${error.message}`]]);
    return;
  }
  const ficha = $("ficha-estado");
  estado.modeloListo = cargarModelo(estado.info, (recibidos, total) => {
    ficha.textContent = total ? `Descargando ${Math.round((100 * recibidos) / total)} %` : "Descargando…";
  }).then((modelo) => {
    estado.modelo = modelo;
    ficha.textContent = "Listo, en tu navegador";
  }).catch((error) => {
    ficha.textContent = "No se pudo cargar";
    mensajes([["error", error.message]]);
    throw error;
  });
}

arrancar();
