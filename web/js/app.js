// Interfaz de la app web (D38). La validación está en validar.js (igual que src/data/entrada.py) y el modelo en
// modelo.js (el ONNX exportado del modelo final). Aquí solo se dibuja y se enseña.
// Todo el texto que viene de fuera (nombres de fichero) se escribe con textContent: nunca se interpreta como HTML.

import { lienzoContribucion } from "./comun.js";
import { cargarModelo } from "./modelo.js";
import { FASES, LADO, TAMANO_MAXIMO, identificar, validarCorte } from "./validar.js";

const LECTURA_MAXIMA = 50 * 1024 * 1024; // ficheros más grandes ni se leen: se rechazan por tamaño
const COHORTES = { spy1: "I-SPY1", spy2: "I-SPY2", duke: "Duke" };
// Mapa de color «inferno» para el realce (de matplotlib): negro → violeta → rojo → naranja → amarillo pálido.
const INFERNO = ["#000004", "#1b0c41", "#4a0c6b", "#781c6d", "#a52c60", "#cf4446", "#ed6925", "#fb9b06", "#fcffa4"];
const REALCE_MAXIMO = 0.4; // EARLY − PRE (en [0, 1]) que se pinta con el color más claro

const $ = (id) => document.getElementById(id);
const estado = { info: null, modelo: null, modeloListo: null, peticion: 0, vista: "EARLY", corte: null, ultima: null };

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
  estado.planos = planos;
  $("visor").setAttribute("data-lleno", "");
  mostrarVista(estado.vista);
}
function vaciarFases() {
  for (const c of document.querySelectorAll("#fases canvas")) c.getContext("2d").clearRect(0, 0, LADO, LADO);
  estado.planos = null;
  $("visor").removeAttribute("data-lleno");
  $("fases-vacio").textContent = "Elige un corte o sube los tuyos.";
  mostrarVista(estado.vista);
}

// ---------- El visor: lo elegido, en grande, con los datos en las esquinas como en una estación de radiología ----------
function mostrarVista(fase) {
  estado.vista = fase;
  const botones = [...document.querySelectorAll("#fases .fase")];
  const boton = botones.find((b) => b.dataset.fase === fase) || botones[1];
  for (const b of botones) b.setAttribute("aria-pressed", String(b === boton));
  const visor = $("visor-lienzo"), ctx = visor.getContext("2d");
  ctx.clearRect(0, 0, LADO, LADO);
  ctx.drawImage(boton.querySelector("canvas"), 0, 0);
  const nombre = boton.querySelector(".fase__nombre").textContent;
  const notas = { PRE: "antes del contraste", EARLY: "el contraste acaba de llegar", LATE: "fase tardía",
    REALCE: "EARLY − PRE: dónde entra el contraste", MAPA: "zonas que suben (rojo) o bajan (azul) la probabilidad" };
  $("esquina-sd").replaceChildren(nodo("strong", nombre), nodo("span", notas[fase] || ""));
  visor.setAttribute("aria-label", `${nombre}, ampliado`);
  $("visor").dataset.vista = fase;
  esquinas();
}
function esquinas() {
  const c = estado.corte;
  $("esquina-si").replaceChildren(...(c ? [nodo("strong", c.titulo), nodo("span", c.subtitulo || "")] : []));
  $("esquina-ii").replaceChildren(...(c ? [...(c.etiqueta ? [nodo("strong", c.etiqueta)] : []),
    nodo("span", window.matchMedia("(hover: hover)").matches ? "pasa el cursor: curva de realce" : "toca la imagen: curva de realce")] : []));
  const r = estado.ultima;
  let abajo = [];
  if (c && r) abajo = [nodo("span", "probabilidad de pCR"), nodo("strong", prob(r.probabilidad))];
  else if (c) abajo = [nodo("span", "calculando…")];
  $("esquina-id").replaceChildren(...abajo);
}
function prepararVisor() {
  for (const b of document.querySelectorAll("#fases .fase")) b.addEventListener("click", () => mostrarVista(b.dataset.fase));
  const pantalla = $("visor");
  const mover = (e) => curvaDeRealce(e, pantalla);
  pantalla.addEventListener("pointermove", mover);
  pantalla.addEventListener("pointerdown", mover);
  pantalla.addEventListener("pointerleave", () => { $("cinetica").hidden = true; $("mira").hidden = true; });
}

// ---------- La curva de realce de un punto (lo que mira un radiólogo en una DCE-MRI) ----------
// Intensidad media de un cuadrado de 5 × 5 píxeles en PRE, EARLY y LATE. Con ella, el realce inicial
// (EARLY − PRE) / PRE y lo que pasa después (LATE − EARLY) / EARLY: si sigue subiendo (persistente), se queda
// (meseta) o baja (lavado). Es la clasificación de Kuhl (1999). Aquí es orientativa: los PNG están reescalados a
// 8 bits y no son las intensidades originales de la resonancia.
function mediaLocal(plano, x, y) {
  let suma = 0, n = 0;
  for (let dy = -2; dy <= 2; dy++) for (let dx = -2; dx <= 2; dx++) {
    const xx = x + dx, yy = y + dy;
    if (xx < 0 || yy < 0 || xx >= LADO || yy >= LADO) continue;
    suma += plano[yy * LADO + xx];
    n++;
  }
  return suma / n;
}
function curvaDeRealce(evento, pantalla) {
  const caja = $("cinetica");
  if (!estado.planos) { caja.hidden = true; return; }
  const r = pantalla.getBoundingClientRect();
  const fx = (evento.clientX - r.left) / r.width, fy = (evento.clientY - r.top) / r.height;
  if (fx < 0 || fy < 0 || fx > 1 || fy > 1) { caja.hidden = true; return; }
  const x = Math.min(LADO - 1, Math.floor(fx * LADO)), y = Math.min(LADO - 1, Math.floor(fy * LADO));
  const [pre, early, late] = estado.planos.map((p) => mediaLocal(p, x, y));
  const W = 196, H = 92, x0 = 12, x1 = W - 12, y0 = 10, y1 = H - 22;
  const Y = (v) => y1 - (v / 255) * (y1 - y0);
  const X = [x0, (x0 + x1) / 2, x1];
  const puntos = [pre, early, late];
  const hijos = [svg("line", { x1: x0, x2: x1, y1, y2: y1, class: "cinetica__base" })];
  hijos.push(svg("polyline", { points: puntos.map((v, i) => `${X[i]},${Y(v)}`).join(" "), class: "cinetica__linea" }));
  puntos.forEach((v, i) => hijos.push(svg("circle", { cx: X[i], cy: Y(v), r: 3.5, class: "cinetica__punto" })));
  ["PRE", "EARLY", "LATE"].forEach((n, i) => hijos.push(svg("text", { x: X[i], y: H - 6, "text-anchor": i === 0 ? "start" : i === 2 ? "end" : "middle" }, n)));
  $("cinetica-grafica").replaceChildren(...hijos);
  let texto, tipo = "";
  if (pre < 0.08 * 255 && early < 0.12 * 255) {
    texto = "Fuera del tejido";
  } else {
    const subida = (early - pre) / Math.max(pre, 1), despues = (late - early) / Math.max(early, 1);
    texto = `Realce inicial ${subida >= 0 ? "+" : "−"}${Math.round(Math.abs(subida) * 100)} %`;
    if (subida < 0.1) tipo = "casi no capta contraste";
    else if (despues > 0.1) tipo = "después sigue subiendo: persistente";
    else if (despues < -0.1) tipo = "después baja: lavado";
    else tipo = "después se mantiene: meseta";
  }
  $("cinetica-texto").replaceChildren(nodo("strong", texto), ...(tipo ? [nodo("span", tipo)] : []));
  caja.hidden = false;
  // La caja va al lado del cursor, sin salirse del visor
  const cw = caja.offsetWidth, ch = caja.offsetHeight;
  let left = evento.clientX - r.left + 18, top = evento.clientY - r.top + 18;
  if (left + cw > r.width - 8) left = evento.clientX - r.left - cw - 18;
  if (top + ch > r.height - 8) top = evento.clientY - r.top - ch - 18;
  caja.style.transform = `translate(${Math.max(8, left)}px, ${Math.max(8, top)}px)`;
  const mira = $("mira");
  mira.hidden = false;
  mira.style.transform = `translate(${evento.clientX - r.left}px, ${evento.clientY - r.top}px)`;
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
  const linea = "#4d5f6c", tinta = "#e6edf2", ambar = "#f2a541", fondo = "#10181f";   // los tokens de DESIGN.md
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
    hijos.push(svg("rect", { x: abajo(franja[0]), y: 86, width: abajo(franja[1]) - abajo(franja[0]), height: 26, fill: "#26333d", rx: 2 }));
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
    svg("circle", { cx: 0, cy: 99, r: 7, fill: ambar, stroke: fondo, "stroke-width": 2 }),
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
  panel.setAttribute("data-hubo", "");
  estado.ultima = r;
  $("probabilidad").textContent = prob(r.probabilidad);
  $("regla-umbral").textContent = textoUmbral(umbral);
  const clase = $("clase");
  const esPcr = r.probabilidad >= umbral;
  clase.replaceChildren(
    nodo("strong", esPcr ? "pCR" : "no pCR"),
    document.createTextNode(esPcr
      ? `: la probabilidad está por encima del umbral (${num(umbral)}). Si el modelo acertara, tras la quimioterapia no quedaría cáncer invasivo.`
      : `: la probabilidad está por debajo del umbral (${num(umbral)}). Si el modelo acertara, tras la quimioterapia quedaría cáncer invasivo.`),
  );
  $("fiabilidad-corte").textContent = fiabilidad();
  const chip = $("clase-chip");
  chip.hidden = false;
  chip.textContent = esPcr ? "pCR" : "no pCR";
  chip.dataset.clase = esPcr ? "pcr" : "no";
  dibujarRegla(r.probabilidad);
  if (r.mapa && estado.early) {
    const lienzo = document.querySelectorAll("#fases canvas")[4];
    lienzo.getContext("2d").drawImage(lienzoContribucion(r.mapa.datos, r.mapa.lado, estado.early), 0, 0);
  }
  mostrarVista(estado.vista);
  $("dato-dispositivo").textContent = `${estado.modelo.dispositivo}. Solo inferencia: el modelo se exportó en modo evaluación `
    + "(model.eval()) y aquí no se calculan gradientes.";
  $("dato-tiempo").textContent = `${Math.round(r.ms)} ms`;
  const coincide = $("coincide");
  if (esperado == null) {
    coincide.hidden = true;
  } else {
    const ok = Math.abs(esperado - r.probabilidad) < 1e-5;
    coincide.hidden = false;
    coincide.dataset.ok = ok ? "si" : "no";
    coincide.replaceChildren(nodo("span", ok ? "✓" : "✗", "coincide__marca"),
      document.createTextNode(ok ? ` Igual que en el código de Python del proyecto (${num(esperado, 6)})`
        : ` Distinto del código de Python del proyecto: ${num(r.probabilidad, 6)} aquí y ${num(esperado, 6)} allí`));
  }
}

// Cuánto acierta el modelo, junto al resultado: con la cifra de test si ya se evaluó, si no con la de validación
function fiabilidad() {
  const t = estado.info?.test?.modelos?.cnn;
  const auc = t ? t.auc : estado.info?.metricas?.auc_val;
  if (auc == null) return "";
  const donde = t ? "En test, con pacientes nuevas," : "En validación cruzada,";
  return auc < 0.6
    ? `${donde} su AUC es ${num(auc, 2)}: acierta poco más que el azar (0,5). Tómalo como un ejemplo de cómo funciona, no como una predicción.`
    : `${donde} su AUC es ${num(auc, 2)} (0,5 es el azar y 1, acertar siempre). Es un modelo educativo: no sirve para decidir nada.`;
}

// Por qué el umbral no es 0,5, en palabras: sale de la especificidad mínima con que se eligió (D49)
function textoUmbral(umbral) {
  const m = /especificidad\s*≥\s*([\d,.]+)/.exec(estado.info?.criterio_umbral || "");
  if (!m) return `El umbral, ${num(umbral)}, se eligió con las pacientes de entrenamiento.`;
  const falsos = 1 - Number(m[1].replace(",", "."));
  const cada = Math.round(1 / falsos);
  const proporcion = Math.abs(1 / falsos - cada) < 1e-6 ? `1 de cada ${cada}` : `${Math.round(100 * falsos)} de cada 100`;
  return `El umbral es ${num(umbral)} y no 0,5: se eligió con las pacientes de entrenamiento para que, como mucho, `
    + `${proporcion} pacientes no pCR salga como pCR.`;
}

// Si la persona eligió el corte más abajo, se le sube a ver el resultado. En pantallas anchas, al visor (imagen y
// resultado van juntos); en las estrechas, al panel del resultado, que está debajo de la imagen (D49)
function subirAlVisor() {
  const estrecha = window.matchMedia("(max-width: 1000px)").matches;
  const destino = estrecha ? $("resultado") : document.querySelector(".sala");
  const arriba = destino.getBoundingClientRect().top;
  if (estrecha ? arriba >= 0 && arriba < window.innerHeight / 3 : arriba >= -40) return;
  const suave = !window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  destino.scrollIntoView({ behavior: suave ? "smooth" : "auto", block: "start" });
}

// Qué dice el título del resultado: de qué corte es
function tituloResultado() {
  const c = estado.corte;
  $("titulo-resultado").textContent = !c ? "Resultado" : c.ejemplo ? "Resultado del ejemplo" : "Resultado de tu corte";
}

// La entrada, ya validada: forma, tipo y los valores que tiene de verdad (lo que pide el enunciado, a la vista)
function textoEntrada(pixeles) {
  let min = 255, max = 0;
  for (let i = 0; i < pixeles.length; i++) {
    const v = pixeles[i];
    if (v < min) min = v;
    if (v > max) max = v;
  }
  return `PRE, EARLY y LATE: 3 × ${LADO} × ${LADO} píxeles, gris de 8 bits, valores de ${min} a ${max}. Dentro del modelo `
    + "se dividen entre 255, como en el entrenamiento.";
}

async function analizar(ficheros, esperado = null, corte = null, subir = true, { traducir = (m) => m, avisos = [] } = {}) {
  const peticion = ++estado.peticion;
  const r = await validarCorte(ficheros);
  if (peticion !== estado.peticion) return null;
  const lista = [...r.errores.map((m) => ["error", traducir(m)]), ...avisos.map((m) => ["aviso", m]),
    ...r.avisos.map((m) => ["aviso", traducir(m)])];
  const panel = $("resultado");
  if (r.errores.length) {
    mensajes(lista);
    estado.corte = null;
    estado.ultima = null;
    vaciarFases();
    panel.dataset.estado = "vacio";
    panel.removeAttribute("data-hubo");
    $("clase-chip").hidden = true;
    $("dato-entrada").textContent = "—";
    tituloResultado();
    $("resultado-espera").textContent = "Corrige los ficheros (los mensajes están en «Elige un corte») o elige un ejemplo.";
    return r;
  }
  estado.corte = corte || { titulo: "Tu corte", subtitulo: ficheros.map((f) => f.nombre).find((n) => /_PRE\.png$/i.test(n))?.replace(/_PRE\.png$/i, "") };
  estado.ultima = null;
  tituloResultado();
  $("dato-entrada").textContent = textoEntrada(r.pixeles);
  pintarFases(r.pixeles);
  if (subir) subirAlVisor();
  if (!estado.modelo) {
    mensajes([...lista, ["info", "El modelo todavía se está descargando; la predicción saldrá en cuanto termine."]]);
    const mb = estado.info?.onnx?.bytes ? ` (${num(estado.info.onnx.bytes / 1e6, 0)} MB)` : "";
    $("resultado-espera").textContent = `Calculando… La primera vez el navegador descarga el modelo${mb} y el motor que lo ejecuta; `
      + "después quedan guardados y todo se calcula aquí.";
  } else {
    mensajes(lista);
  }
  panel.dataset.estado = "calculando";
  try {
    await estado.modeloListo;
    const resultado = await estado.modelo.predecir(r.pixeles);
    if (peticion !== estado.peticion) return r;
    if (!Number.isFinite(resultado.probabilidad)) throw new Error("el modelo ha devuelto un valor no finito (NaN o infinito)");
    mensajes(lista);
    mostrarResultado(resultado, esperado);
  } catch (error) {
    mensajes([...lista, ["error", `No se pudo calcular la predicción: ${error.message}`]]);
    panel.dataset.estado = "vacio";
    $("resultado-espera").textContent = "No se pudo calcular la predicción (el motivo está en «Elige un corte»).";
  }
  return r;
}

// ---------- Subir un corte propio: un hueco por fase (D47) ----------
// La fase la marca el hueco, no el nombre del fichero. Para validar, cada fichero se nombra por dentro como en el
// dataset (<corte>_<FASE>.png) y pasa por la MISMA validación que en Python (validar.js = entrada.py); los mensajes
// se devuelven con el nombre que puso la persona.
const huecos = { PRE: null, EARLY: null, LATE: null };
const iconosHueco = {};

function pintarHueco(fase) {
  const caja = document.querySelector(`.hueco[data-fase="${fase}"]`);
  const f = huecos[fase];
  caja.toggleAttribute("data-lleno", Boolean(f));
  caja.removeAttribute("data-error");
  caja.querySelector(".hueco__fichero").textContent = f ? f.nombre : "Elige o arrastra";
  const vista = caja.querySelector(".hueco__vista");
  if (f) {
    const img = nodo("img");
    img.alt = "";
    img.src = f.url;
    vista.replaceChildren(img);
  } else {
    vista.innerHTML = iconosHueco[fase];
  }
  caja.querySelector(".hueco__quitar").hidden = !f;
  $("vaciar-huecos").hidden = !FASES.some((x) => huecos[x]);
}

function quitarDeHueco(fase) {
  if (huecos[fase]) URL.revokeObjectURL(huecos[fase].url);
  huecos[fase] = null;
  pintarHueco(fase);
}

async function ponerEnHueco(fase, fichero) {
  if (fichero.size > LECTURA_MAXIMA) {
    mensajes([["error", `${fichero.name}: ocupa ${(fichero.size / 1e6).toFixed(1)} MB; el máximo es ${(TAMANO_MAXIMO / 1e6).toFixed(0)} MB.`]]);
    return;
  }
  const bytes = new Uint8Array(await fichero.arrayBuffer());
  quitarDeHueco(fase);
  huecos[fase] = { nombre: fichero.name, bytes, url: URL.createObjectURL(new Blob([bytes], { type: "image/png" })) };
  pintarHueco(fase);
}

// Con las tres fases puestas, se calcula; si falta alguna, se dice cuál
async function despuesDeColocar() {
  const faltan = FASES.filter((f) => !huecos[f]);
  if (faltan.length) {
    mensajes([["info", `Falta${faltan.length > 1 ? "n" : ""} ${faltan.join(" y ")}: pon ${faltan.length > 1 ? "esas imágenes" : "esa imagen"} en su hueco.`]]);
    return;
  }
  const ids = FASES.map((f) => identificar(huecos[f].nombre));
  const base = ids.find(Boolean)?.[0] || "tu_corte";
  const avisos = [];
  const internos = FASES.map((fase, i) => {
    const id = ids[i];
    if (id && id[1] !== fase) avisos.push(`${huecos[fase].nombre} está en el hueco ${fase}, pero su nombre dice ${id[1]}: se usa como ${fase}.`);
    return { nombre: `${id ? id[0] : base}_${fase}.png`, bytes: huecos[fase].bytes, original: huecos[fase].nombre, fase };
  });
  // Los mensajes de la validación hablan del nombre interno: se cambia, de una pasada, por el que puso la persona
  const porNombre = new Map(internos.map((f) => [f.nombre, f.original === f.nombre ? f.nombre : `${f.original} (${f.fase})`]));
  const patron = new RegExp([...porNombre.keys()].map((n) => n.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")).join("|"), "g");
  const traducir = (m) => m.replace(patron, (n) => porNombre.get(n));
  for (const b of document.querySelectorAll(".ejemplos button")) b.setAttribute("aria-pressed", "false");
  const r = await analizar(internos.map(({ nombre, bytes }) => ({ nombre, bytes })), null,
    { titulo: "Tu corte", subtitulo: base === "tu_corte" ? "" : base }, true, { traducir, avisos });
  // Si un error es de una imagen concreta (empieza por su nombre), se marca su hueco
  if (r) for (const f of internos) {
    const mala = r.errores.some((m) => m.startsWith(`${f.nombre}:`));
    document.querySelector(`.hueco[data-fase="${f.fase}"]`).toggleAttribute("data-error", mala);
  }
}

// Varios ficheros soltados a la vez (o uno fuera de los huecos): si sus nombres dicen la fase, una de cada, se colocan
// solos (D47); si no, se pide ponerlos en su hueco. Sin explicaciones de nombres: quien sube no tiene por qué saberlas (D48)
async function leerVarios(lista) {
  const fases = lista.map((f) => identificar(f.name)?.[1]);
  const colocables = lista.length <= 3 && fases.every(Boolean) && new Set(fases).size === fases.length;
  if (!colocables) {
    mensajes([["info", "Suelta cada imagen sobre su hueco: PRE, EARLY o LATE."]]);
    return;
  }
  for (let i = 0; i < lista.length; i++) await ponerEnHueco(fases[i], lista[i]);
  await despuesDeColocar();
}

function prepararSubida() {
  for (const caja of document.querySelectorAll(".hueco")) {
    const fase = caja.dataset.fase;
    iconosHueco[fase] = caja.querySelector(".hueco__vista").innerHTML;
    const entrada = caja.querySelector("input");
    entrada.addEventListener("change", async () => {
      const lista = [...entrada.files];
      entrada.value = "";
      if (!lista.length) return;
      // Si en el diálogo se eligen las tres a la vez, se colocan por su nombre, como al arrastrarlas juntas (D49)
      if (lista.length > 1) {
        await leerVarios(lista);
        return;
      }
      await ponerEnHueco(fase, lista[0]);
      await despuesDeColocar();
    });
    caja.addEventListener("dragover", (e) => {
      e.preventDefault();
      e.stopPropagation();
      caja.setAttribute("data-encima", "");
    });
    caja.addEventListener("dragleave", () => caja.removeAttribute("data-encima"));
    caja.addEventListener("drop", async (e) => {
      e.preventDefault();
      e.stopPropagation();
      caja.removeAttribute("data-encima");
      const lista = [...e.dataTransfer.files];
      if (lista.length > 1) await leerVarios(lista);
      else if (lista.length) {
        await ponerEnHueco(fase, lista[0]);
        await despuesDeColocar();
      }
    });
    caja.querySelector(".hueco__quitar").addEventListener("click", () => {
      quitarDeHueco(fase);
      mensajes([]);
    });
  }
  const zona = $("subir");
  zona.addEventListener("dragover", (e) => e.preventDefault());
  zona.addEventListener("drop", (e) => {
    e.preventDefault();
    const lista = [...e.dataTransfer.files];
    if (lista.length) leerVarios(lista);
  });
  $("vaciar-huecos").addEventListener("click", () => {
    for (const f of FASES) quitarDeHueco(f);
    mensajes([]);
  });
  // «Subir un corte», junto al título: lleva a los huecos y deja el foco en el primero vacío (D49)
  for (const enlace of document.querySelectorAll("[data-ir-a-subir]")) {
    enlace.addEventListener("click", (e) => {
      e.preventDefault();
      const suave = !window.matchMedia("(prefers-reduced-motion: reduce)").matches;
      $("subir").scrollIntoView({ behavior: suave ? "smooth" : "auto", block: "center" });
      const vacio = FASES.find((f) => !huecos[f]) || "PRE";
      document.querySelector(`.hueco[data-fase="${vacio}"] input`).focus({ preventScroll: true });
    });
  }
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
  const botones = lista.map((e) => {
    const boton = nodo("button");
    boton.type = "button";
    boton.className = "ejemplo";
    boton.setAttribute("aria-pressed", "false");
    const cohorte = COHORTES[e.cohorte] || e.cohorte || "Ejemplo";
    const etiqueta = e.pcr == null ? "" : e.pcr ? "etiqueta real: pCR" : "etiqueta real: no\u00a0pCR";
    // La miniatura es solo para mirar: la predicción lee los PNG con su propio lector (validar.js)
    const miniatura = nodo("img");
    const early = e.ficheros.find((n) => /_EARLY\.png$/i.test(n)) || e.ficheros[0];
    miniatura.src = `ejemplos/${encodeURIComponent(e.corte)}/${encodeURIComponent(early)}`;
    miniatura.alt = "";
    miniatura.loading = "lazy";
    // Bajo la miniatura, solo la clase: la entradilla de la sección ya dice que es la etiqueta real
    boton.append(miniatura, nodo("span", cohorte, "ejemplo__cohorte"),
      nodo("span", e.pcr == null ? "" : e.pcr ? "pCR" : "no\u00a0pCR", "ejemplo__etiqueta"));
    boton.title = e.corte;
    boton.setAttribute("aria-label", [cohorte, e.corte, etiqueta].filter(Boolean).join(", "));
    boton.addEventListener("click", async (evento) => {
      for (const b of ul.querySelectorAll("button")) b.setAttribute("aria-pressed", String(b === boton));
      const ficheros = await Promise.all(e.ficheros.map(async (nombre) => {
        const r = await fetch(`ejemplos/${encodeURIComponent(e.corte)}/${encodeURIComponent(nombre)}`);
        return { nombre, bytes: new Uint8Array(await r.arrayBuffer()) };
      }));
      await analizar(ficheros, e.probabilidad_pipeline, { titulo: `Ejemplo de ${cohorte}`, subtitulo: e.corte, etiqueta, ejemplo: true },
        evento.isTrusted);
    });
    const li = nodo("li");
    li.append(boton);
    return li;
  });
  ul.replaceChildren(...botones);
  // Al abrir la página, el primer ejemplo: que lo primero que se vea sea una resonancia de verdad
  botones[0]?.querySelector("button").click();
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
    ["Entrada", String(info.entrada).replace(/\s*→\s*\[0, 1\]/, " (de 0 a 1)")],
    // e.parametros suma una red de cada versión; las 30 redes del conjunto tienen 5 veces más (D50)
    ["Parámetros", `${miles(e.parametros)} por cada fold (una red de cada versión); ${miles(e.parametros * e.redes / e.versiones.length)} en las ${e.redes} redes`],
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

  // Las cuatro cifras que lo resumen, y debajo el detalle. Con test evaluado (D44), van primero las de test
  const t = info.test;
  const tc = t?.modelos?.cnn, tm = t?.modelos?.medico;
  const cifras = [];
  if (tc) {
    cifras.push([num(tc.auc), `AUC de la CNN en test: ${t.pacientes} pacientes nuevas, una sola vez (IC 95 % ${num(tc.ic95[0], 2)}–${num(tc.ic95[1], 2)})`]);
    cifras.push([num(m.auc_val), `AUC de la CNN en validación cruzada, ± ${num(m.auc_val_desviacion)} (0,5 sería el azar)`]);
    if (tm) cifras.push([num(tm.auc), "AUC en test de un médico que solo sabe la edad, la menopausia y el subtipo"]);
    const { VP: vp, FN: fn } = tc.matriz_confusion;
    cifras.push([`${vp} de ${vp + fn}`, `pCR de test que detecta con el umbral (especificidad ${num(tc.especificidad, 2)})`]);
  } else {
    cifras.push([num(m.auc_val), `AUC de la CNN por paciente, ± ${num(m.auc_val_desviacion)} (0,5 sería el azar)`]);
    if (ref.medico_auc != null) cifras.push([num(ref.medico_auc), "AUC de un médico que solo sabe la edad, la menopausia y el subtipo"]);
    cifras.push([`${Math.round(100 * m.precision_pcr)} de 100`, "de las pacientes que marca como pCR, lo son"]);
    if (ref.franja_90) cifras.push([`${num(ref.franja_90[0], 2)}–${num(ref.franja_90[1], 2)}`, "probabilidad que da al 90 % de las pacientes"]);
  }
  $("fiable-cifras").replaceChildren(...cifras.map(([valor, texto]) => {
    const d = nodo("div");
    d.append(nodo("dt", texto), nodo("dd", valor));
    return d;
  }));
  const partes = [`El AUC dice en qué proporción de las parejas de pacientes (una pCR y otra no pCR) el modelo puntúa más alto `
    + `a la pCR: 0,5 es el azar y 1, acertar siempre.`];
  if (tc) {
    partes.push(`En test, con pacientes que ningún modelo había visto, da ${num(tc.auc)}, casi lo mismo que en validación cruzada `
      + `(${num(m.auc_val)}): la estimación interna se sostiene. Pero con ${t.pacientes} pacientes el intervalo incluye el 0,5, así que en test `
      + "no se puede afirmar que acierte más que el azar.");
  } else {
    partes.push(`Son cifras internas, de validación cruzada; con pacientes nuevas pueden ser peores. `
      + `Con el umbral detecta ${Math.round(100 * m.sensibilidad)} de cada 100 pCR y acierta ${Math.round(100 * m.especificidad)} de cada 100 «no pCR».`);
  }
  if (ref.franja_90) partes.push(`La probabilidad apenas se aparta de la tasa de pCR de entrenamiento (${pct(ref.tasa_pcr)}).`);
  partes.push("Esta predicción usa un solo corte; el modelo se evaluó por paciente, con la media de sus cortes.");
  $("texto-incertidumbre").textContent = partes.join(" ");

  // «El modelo»: el resultado de test, con todo lo que pide el enunciado (matriz de confusión, AUC con IC, F1)
  if (tc) {
    $("modelo-test-bloque").hidden = false;
    const fecha = (t.fecha || "").slice(0, 10).split("-").reverse().join("/");
    $("modelo-test-nota").textContent = `${t.pacientes} pacientes (${t.pacientes_pcr} pCR) que no se usaron para nada antes. Evaluado una sola vez`
      + `${fecha ? `, el ${fecha}` : ""}, con el modelo ya cerrado.`;
    const { VP: vp, FN: fn, FP: fp, VN: vn } = tc.matriz_confusion;
    const filasTest = [
      ["AUC por paciente", `${num(tc.auc)} (IC 95 % ${num(tc.ic95[0], 2)}–${num(tc.ic95[1], 2)})`],
      ["Con el umbral", `sensibilidad ${num(tc.sensibilidad)}, especificidad ${num(tc.especificidad)}, precisión pCR ${num(tc.precision_pcr)}`],
      ["Matriz de confusión", `de ${vp + fn} pacientes pCR, dice pCR a ${vp} y no pCR a ${fn}; de ${fp + vn} no pCR, dice pCR a ${fp} y no pCR a ${vn}`],
      ["F1 macro", `${num(tc.f1_macro)} (decir siempre «no pCR»: ${num(t.siempre_no_pcr?.f1_macro)})`],
      ["Error de Brier", `${num(tc.brier, 4)} (decir a todas la tasa de pCR: ${num(t.brier_constante, 4)})`],
    ];
    if (tm) filasTest.push(["Médico sin imagen", `${num(tm.auc)} (IC 95 % ${num(tm.ic95[0], 2)}–${num(tm.ic95[1], 2)})`]);
    const dif = t.comparacion?.combinado_menos_medico;
    if (dif) filasTest.push(["Lo que añade la imagen al médico", `${dif.diferencia > 0 ? "+" : ""}${num(dif.diferencia)} `
      + `(IC 95 % ${num(dif.ic95[0])} a ${num(dif.ic95[1])})`]);
    lista($("modelo-test"), filasTest);
  }
}

// ---------- Arranque ----------
async function arrancar() {
  prepararVisor();
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
