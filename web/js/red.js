// Página «La red» (D39): la arquitectura, pieza a pieza, con un corte de verdad atravesándola.
//
// Los mapas de características son reales: salen de la primera red del conjunto (cnn_v3, fold 0), exportada a
// ONNX con lo que sale de cada bloque (una_red.onnx) y ejecutada aquí. Los tamaños, parámetros y campos
// receptivos vienen de info.json, que los calcula exportar_web recorriendo la red de PyTorch capa a capa.

import { cargarUnaRed } from "./modelo.js";
import { validarCorte } from "./validar.js";
import {
  $, ACTIVACION, COHORTES, INFERNO, ficherosDeEjemplo, leerJSON, lienzoContribucion, miles, nodo, num, pintarGris,
  pintarMatriz,
} from "./comun.js";

const SVG = "http://www.w3.org/2000/svg";
const N = 256 * 256;
const estado = { info: null, ejemplos: [], ejecutar: null, ejemplo: null, pixeles: null, salidas: null, pieza: "entrada" };

function svg(etiqueta, atributos = {}, texto) {
  const n = document.createElementNS(SVG, etiqueta);
  for (const [k, v] of Object.entries(atributos)) n.setAttribute(k, v);
  if (texto != null) n.textContent = texto;
  return n;
}

// ---------- Datos de cada bloque ----------
function bloque(k) {
  const t = estado.salidas[`bloque${k}`];
  const [, canales, alto, ancho] = t.dims;
  return { datos: t.data, canales, alto, ancho };
}
function canal(b, c) {
  const n = b.alto * b.ancho;
  return b.datos.subarray(c * n, (c + 1) * n);
}
function normalizado(valores) {
  let max = 0;
  for (const v of valores) if (v > max) max = v;
  const salida = new Float32Array(valores.length);
  if (max > 0) for (let i = 0; i < valores.length; i++) salida[i] = valores[i] / max;
  return salida;
}
/** Canales ordenados de más a menos activación media (los que más «responden» a este corte, primero). */
function ordenCanales(b) {
  const medias = Array.from({ length: b.canales }, (_, c) => {
    let s = 0;
    for (const v of canal(b, c)) s += v;
    return [c, s / (b.alto * b.ancho)];
  });
  return medias.sort((x, y) => y[1] - x[1]);
}
function lienzoCanal(b, c) {
  return pintarMatriz(document.createElement("canvas"), normalizado(canal(b, c)), b.ancho, b.alto, ACTIVACION);
}
function plano(i) {
  return estado.pixeles.subarray(i * N, (i + 1) * N);
}
function lienzoFase(i) {
  return pintarGris(document.createElement("canvas"), plano(i));
}
function lienzoRealce() {
  const pre = plano(0), early = plano(1);
  const v = new Float32Array(N);
  for (let i = 0; i < N; i++) v[i] = (early[i] - pre[i]) / 255 / 0.4;
  return pintarMatriz(document.createElement("canvas"), v, 256, 256, INFERNO);
}
// ---------- El diagrama ----------
const PIEZAS = ["entrada", "b1", "b2", "b3", "b4", "media", "lineal", "conjunto", "calibracion", "umbral"];
// Posición (x), lado y separación entre tarjetas de cada pila; el resto de piezas, en X
const GEOMETRIA = {
  entrada: { x: 28, lado: 150, paso: 14, n: 3 }, b1: { x: 270, lado: 112, paso: 8, n: 5 },
  b2: { x: 456, lado: 90, paso: 6.5, n: 5 }, b3: { x: 606, lado: 70, paso: 5, n: 5 }, b4: { x: 734, lado: 54, paso: 5, n: 5 },
};
const X = { media: 866, lineal: 950, conjunto: 1018, prob: 1178, clase: 1292 };
const CENTRO = 225;
const BASE = 352;                  // línea de los nombres, la misma para todas las piezas
const ZONA = { y: 112, alto: 296 }; // el rectángulo que se marca al elegir una pieza
const derecha = (id) => GEOMETRIA[id].x + GEOMETRIA[id].lado + (GEOMETRIA[id].n - 1) * GEOMETRIA[id].paso;

function pila(grupo, x, lado, lienzos, desplazamiento) {
  // De atrás adelante: el último lienzo de la lista queda delante, abajo a la izquierda
  const n = lienzos.length;
  lienzos.forEach((c, i) => {
    const k = n - 1 - i;
    const px = x + k * desplazamiento, py = CENTRO - lado / 2 - k * desplazamiento;
    grupo.append(svg("rect", { x: px - 1, y: py - 1, width: lado + 2, height: lado + 2, fill: "#2b3842", rx: 2 }));
    const img = svg("image", { x: px, y: py, width: lado, height: lado, preserveAspectRatio: "none" });
    img.setAttribute("href", c.toDataURL());
    grupo.append(img);
  });
}

function etiquetas(grupo, x, y, nombre, detalle) {
  grupo.append(svg("text", { x, y, class: "dg-nombre" }, nombre));
  detalle.split("\n").forEach((linea, i) => grupo.append(svg("text", { x, y: y + 18 + i * 16, class: "dg-detalle" }, linea)));
}

function flecha(raiz, x1, x2, arriba) {
  raiz.append(svg("line", { x1, x2, y1: CENTRO, y2: CENTRO, class: "dg-linea" }));
  raiz.append(svg("path", { d: `M${x2 - 6} ${CENTRO - 4} L${x2} ${CENTRO} L${x2 - 6} ${CENTRO + 4}`, class: "dg-linea" }));
  arriba.forEach((t, i) => raiz.append(svg("text", { x: (x1 + x2) / 2, y: 52 + i * 15, class: "dg-op", "text-anchor": "middle" }, t)));
}
function zona(g, x, ancho) {
  g.prepend(svg("rect", { x, y: ZONA.y, width: ancho, height: ZONA.alto, class: "dg-zona" }));
}

function grupoPieza(raiz, id, etiqueta) {
  const g = svg("g", { class: "pieza", "data-pieza": id, tabindex: 0, role: "button", "aria-label": etiqueta });
  g.addEventListener("click", () => elegirPieza(id));
  g.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      elegirPieza(id);
    }
  });
  raiz.append(g);
  return g;
}

function dibujarDiagrama() {
  const raiz = $("diagrama");
  raiz.replaceChildren();
  const red = estado.info.entrenamiento.versiones[0].red;
  const prob = estado.ejemplo.probabilidad_pipeline;
  const umbral = estado.info.umbral;

  // Entrada: las tres fases apiladas (PRE delante)
  {
    const g = grupoPieza(raiz, "entrada", "Entrada: las tres fases del corte");
    const { x, lado, paso } = GEOMETRIA.entrada;
    pila(g, x, lado, [lienzoFase(2), lienzoFase(1), lienzoFase(0)], paso);
    etiquetas(g, x, BASE, "Entrada", "PRE, EARLY y LATE\n3 × 256 × 256");
    zona(g, x - 12, derecha("entrada") - x + 24);
  }
  // Cuatro bloques de convolución con sus mapas reales
  red.bloques.forEach((b, i) => {
    const id = `b${i + 1}`;
    const { x, lado, paso, n } = GEOMETRIA[id];
    const datos = bloque(i + 1);
    const lienzos = ordenCanales(datos).slice(0, n).reverse().map(([c]) => lienzoCanal(datos, c));
    const g = grupoPieza(raiz, id, `Bloque ${i + 1}: ${b.salida[0]} mapas de ${b.salida[1]} × ${b.salida[2]}`);
    pila(g, x, lado, lienzos, paso);
    const [c, h, w] = b.salida;
    etiquetas(g, x, BASE, `Bloque ${i + 1}`, `${c} mapas\n${h} × ${w}`);
    zona(g, x - 12, Math.max(derecha(id) - x + 24, 96));
  });
  // Flechas con la operación de cada bloque
  const op = ["conv 3×3", "BN, ReLU", "máx. 2×2"];
  flecha(raiz, derecha("entrada") + 8, GEOMETRIA.b1.x - 8, op);
  flecha(raiz, derecha("b1") + 8, GEOMETRIA.b2.x - 8, op);
  flecha(raiz, derecha("b2") + 8, GEOMETRIA.b3.x - 8, op);
  flecha(raiz, derecha("b3") + 8, GEOMETRIA.b4.x - 8, op);

  // Media global: el vector de 128 (se dibujan 12 puntos, con la intensidad de 12 canales)
  const b4 = bloque(4);
  const medias = ordenCanales(b4);
  const maxMedia = medias[0][1] || 1;
  const yPunto = (i) => CENTRO - 82 + i * 15;
  {
    const g = grupoPieza(raiz, "media", "Media global: un vector de 128 valores");
    const x = X.media;
    flecha(raiz, derecha("b4") + 8, x - 14, ["media", "de cada mapa"]);
    for (let i = 0; i < 12; i++) {
      const [, m] = medias[Math.round((i * (medias.length - 1)) / 11)];
      g.append(svg("circle", { cx: x, cy: yPunto(i), r: 5.5, fill: "#7cc3cf", "fill-opacity": 0.25 + 0.75 * (m / maxMedia) }));
    }
    etiquetas(g, x - 22, BASE, "Media", "vector\nde 128");
    zona(g, x - 32, 72);
  }
  // Capa lineal: una neurona, un logit
  {
    const g = grupoPieza(raiz, "lineal", "Capa lineal: de 128 valores a un logit");
    const x = X.lineal;
    for (let i = 0; i < 12; i++) g.append(svg("line", { x1: X.media + 6, y1: yPunto(i), x2: x - 18, y2: CENTRO, class: "dg-conexion" }));
    g.append(svg("circle", { cx: x, cy: CENTRO, r: 18, class: "dg-nodo" }));
    g.append(svg("text", { x, y: CENTRO + 5, "text-anchor": "middle", class: "dg-nodo-texto" }, "Σ"));
    etiquetas(g, x - 22, BASE, "Lineal", "129 pesos\n→ 1 logit");
    zona(g, x - 32, 76);
  }
  // El conjunto: 6 versiones × 5 folds; la red dibujada es la primera
  {
    const g = grupoPieza(raiz, "conjunto", "Conjunto: media de las 30 redes");
    const x0 = X.conjunto + 12, y0 = CENTRO - 42;
    flecha(raiz, X.lineal + 22, X.conjunto - 6, ["×30"]);
    estado.info.entrenamiento.versiones.forEach((v, fila) => {
      for (let col = 0; col < 5; col++) {
        const esta = fila === 0 && col === 0;
        g.append(svg("circle", { cx: x0 + col * 17, cy: y0 + fila * 17, r: 6, class: esta ? "dg-red dg-red--esta" : "dg-red" }));
      }
    });
    etiquetas(g, X.conjunto, BASE, "Conjunto", "6 versiones × 5\nmedia de logits");
    zona(g, X.conjunto - 12, 120);
  }
  // Calibración y probabilidad
  {
    const g = grupoPieza(raiz, "calibracion", `Calibración y sigmoide: probabilidad ${num(prob)}`);
    const x = X.prob;
    flecha(raiz, X.conjunto + 106, x - 34, ["Platt", "y sigmoide"]);
    g.append(svg("circle", { cx: x, cy: CENTRO, r: 30, class: "dg-nodo dg-nodo--prob" }));
    g.append(svg("text", { x, y: CENTRO + 6, "text-anchor": "middle", class: "dg-nodo-texto dg-nodo-texto--prob" }, num(prob)));
    etiquetas(g, x - 34, BASE, "Probabilidad", "de pCR de\neste corte");
    zona(g, x - 46, 100);
  }
  // Umbral y clase
  {
    const esPcr = prob >= umbral;
    const g = grupoPieza(raiz, "umbral", `Umbral ${num(umbral)}: clase ${esPcr ? "pCR" : "no pCR"}`);
    const x = X.clase;
    flecha(raiz, X.prob + 34, x - 44, [`≥ ${num(umbral)}?`]);
    g.append(svg("rect", { x: x - 40, y: CENTRO - 17, width: 82, height: 34, rx: 17, class: esPcr ? "dg-clase dg-clase--pcr" : "dg-clase" }));
    g.append(svg("text", { x: x + 1, y: CENTRO + 5, "text-anchor": "middle", class: "dg-clase-texto" }, esPcr ? "pCR" : "no pCR"));
    etiquetas(g, x - 26, BASE, "Clase", `umbral\n${num(umbral)}`);
    zona(g, x - 48, 96);
  }
  // Llaves de abajo: qué parte es cada cosa
  const llave = (x1, x2, texto) => {
    raiz.append(svg("path", { d: `M${x1} 420 L${x1} 428 L${x2} 428 L${x2} 420`, class: "dg-llave" }));
    raiz.append(svg("text", { x: (x1 + x2) / 2, y: 450, "text-anchor": "middle", class: "dg-llave-texto" }, texto));
  };
  llave(18, derecha("b4") + 10, "Extracción de características: cuatro bloques");
  llave(X.media - 32, X.lineal + 44, "Una red decide");
  llave(X.conjunto - 12, 1334, "Conjunto, calibración y umbral");
  marcarPieza();
}

function marcarPieza() {
  for (const g of document.querySelectorAll(".pieza")) g.classList.toggle("pieza--elegida", g.dataset.pieza === estado.pieza);
}

// ---------- El panel de detalle ----------
function parametros(filas) {
  $("detalle-parametros").replaceChildren(...filas.flatMap(([t, d]) => [nodo("dt", t), nodo("dd", d)]));
}
function explicacion(parrafos) {
  $("detalle-explicacion").replaceChildren(...parrafos.map((p) => nodo("p", p)));
}
function figura(lienzo, pie, clase = "") {
  const f = nodo("figure", null, `fig ${clase}`);
  f.append(lienzo, nodo("figcaption", pie));
  return f;
}

function detalleEntrada() {
  $("detalle-titulo").textContent = "Las tres fases del corte";
  explicacion([
    "Una resonancia de mama con contraste se repite varias veces: antes de inyectar el contraste (PRE), poco después (EARLY) y un rato más tarde (LATE). El contraste entra donde hay muchos vasos, y los tumores tienen muchos: por eso brillan en EARLY.",
    "La red recibe las tres a la vez, como los tres canales de una imagen en color. Pero no son colores: son tres momentos. Cada píxel se divide entre 255 para llevarlo a [0, 1], igual que en el entrenamiento.",
    "El mapa de realce (EARLY − PRE) no entra en esta red: se enseña para ver lo que ella tiene que descubrir sola. Una de las versiones del conjunto (cnn_v4) sí lo recibe hecho.",
  ]);
  parametros([["Forma", "3 × 256 × 256"], ["Valores", "de 0 a 1 (píxel / 255)"], ["Parámetros", "0"]]);
  const v = nodo("div", null, "rejilla-fases");
  v.append(figura(lienzoFase(0), "PRE"), figura(lienzoFase(1), "EARLY"), figura(lienzoFase(2), "LATE"),
    figura(lienzoRealce(), "Realce: EARLY − PRE"));
  $("detalle-visual").replaceChildren(v);
}

function detalleBloque(k) {
  const b = estado.info.entrenamiento.versiones[0].red.bloques[k - 1];
  const conv = b.convoluciones[0];
  const [c, h, w] = b.salida;
  const [ce, he] = b.entrada;
  $("detalle-titulo").textContent = `Bloque ${k}: ${c} mapas de ${h} × ${w}`;
  explicacion([
    `Convolución 3 × 3 con ${conv.filtros} filtros. Cada filtro es una plantilla de 3 × 3 × ${conv.entra} = ${9 * conv.entra} pesos que recorre la imagen y se enciende donde encuentra su patrón. Al principio son bordes y contrastes; más adentro, combinaciones de los anteriores.`,
    "BatchNorm reescala cada mapa para que el entrenamiento sea estable, y ReLU deja pasar solo lo positivo. Después, el máximo de cada cuadrado de 2 × 2 reduce el mapa a la mitad: se pierde detalle y se gana campo de visión.",
    `Cada punto de estos mapas depende, como mucho, de un cuadrado de ${b.campo_receptivo} × ${b.campo_receptivo} píxeles del corte original (su campo receptivo teórico). En la práctica pesa más el centro del cuadrado que los bordes. Pasa el ratón por un mapa para verlo.`,
  ]);
  parametros([
    ["Entra", `${ce} × ${he} × ${he}`], ["Sale", `${c} × ${h} × ${w}`],
    ["Pesos de la convolución", `3 × 3 × ${conv.entra} × ${conv.filtros} = ${miles(conv.pesos)} (sin sesgo: lo pone BatchNorm)`],
    ["BatchNorm", `${miles(b.batchnorm)} (escala y desplazamiento de cada filtro)`],
    ["Parámetros del bloque", miles(b.parametros)],
    ["Campo receptivo teórico", `${b.campo_receptivo} × ${b.campo_receptivo} píxeles; un paso de ${b.salto} píxeles`],
  ]);
  // Visual: el corte con el campo receptivo, y todos los mapas ordenados por activación
  const datos = bloque(k);
  const v = nodo("div", null, "mapas");
  const base = nodo("div", null, "mapas__base");
  const imagen = lienzoFase(1);
  const capa = document.createElement("canvas");
  capa.width = capa.height = 256;
  capa.className = "mapas__capa";
  base.append(imagen, capa, nodo("p", "EARLY, con el campo receptivo", "mapas__pie"));
  const rejilla = nodo("div", null, `mapas__rejilla mapas__rejilla--${Math.min(c, 64) > 32 ? "densa" : "normal"}`);
  const mostrados = ordenCanales(datos).slice(0, Math.min(c, 64));
  for (const [ci] of mostrados) {
    const lienzo = lienzoCanal(datos, ci);
    lienzo.title = `Filtro ${ci + 1}`;
    lienzo.addEventListener("pointermove", (e) => {
      const r = lienzo.getBoundingClientRect();
      const x = Math.floor(((e.clientX - r.left) / r.width) * w), y = Math.floor(((e.clientY - r.top) / r.height) * h);
      const centroX = (x + 0.5) * b.salto, centroY = (y + 0.5) * b.salto;
      const ctx = capa.getContext("2d");
      ctx.clearRect(0, 0, 256, 256);
      ctx.strokeStyle = "#f0a33a";
      ctx.lineWidth = 2;
      ctx.strokeRect(centroX - b.campo_receptivo / 2, centroY - b.campo_receptivo / 2, b.campo_receptivo, b.campo_receptivo);
    });
    rejilla.append(lienzo);
  }
  const nota = c > 64 ? `Los 64 filtros que más se activan con este corte, de ${c}.` : `Los ${c} filtros, de más a menos activos con este corte.`;
  v.append(base, nodo("div", null, "mapas__lado"));
  v.lastChild.append(rejilla, nodo("p", nota, "mapas__pie"));
  $("detalle-visual").replaceChildren(v);
}

function detalleMedia() {
  const red = estado.info.entrenamiento.versiones;
  const atencion = red.find((v) => v.red && v.red.resumen.tipo === "atención");
  $("detalle-titulo").textContent = "Media global: un número por mapa";
  explicacion([
    "Cada uno de los 128 mapas del último bloque se resume en su media. Queda un vector de 128 números que describe el corte entero, sin importar dónde está cada cosa: un tumor arriba o abajo cuenta igual.",
    "No aprende nada (0 parámetros) y obliga a la red a ser sencilla, lo que ayuda a no memorizar.",
    atencion ? `En ${atencion.version}, en vez de la media normal hay una media ponderada que aprende dónde mirar (pooling por atención): ${miles(atencion.red.resumen.parametros)} parámetros más.` : "",
  ].filter(Boolean));
  parametros([["Entra", "128 × 16 × 16"], ["Sale", "vector de 128"], ["Parámetros", "0"]]);
  // Visual: las 128 medias, como barras
  const medias = ordenCanales(bloque(4));
  const max = medias[0][1] || 1;
  const ancho = 520, alto = 160;
  const s = svg("svg", { viewBox: `0 0 ${ancho} ${alto + 24}`, class: "barras", role: "img", "aria-label": "Las 128 medias, de mayor a menor" });
  medias.forEach(([, m], i) => {
    const h = (m / max) * alto;
    s.append(svg("rect", { x: 4 + i * 4, y: alto - h, width: 3, height: Math.max(0.5, h), fill: "#3b8fa3", rx: 1 }));
  });
  s.append(svg("line", { x1: 0, x2: ancho, y1: alto + 0.5, y2: alto + 0.5, stroke: "#3a4954" }));
  s.append(svg("text", { x: 4, y: alto + 18, class: "barras__texto" }, "las 128 medias de este corte, de mayor a menor"));
  $("detalle-visual").replaceChildren(figura(s, "Este vector es todo lo que la red sabe del corte cuando decide."));
}

function detalleLineal() {
  const red = estado.info.entrenamiento.versiones;
  const dropout = red.find((v) => v.red && v.red.cabeza.some((c) => c.capa === "Dropout"));
  const logit = estado.salidas.logit.data[0];
  const mapa = estado.salidas.mapa;
  $("detalle-titulo").textContent = "Una neurona: de 128 números a un logit";
  explicacion([
    "Multiplica los 128 valores por 128 pesos, los suma y añade un sesgo: 129 parámetros. Sale el logit, un número sin límites; si es positivo, la red se inclina hacia pCR.",
    `Esta red, con este corte, da un logit de ${num(logit, 2)}. Como antes hay una media, el logit se reparte exactamente entre las 16 × 16 zonas del corte: es el mapa de la derecha (rojo, sube la probabilidad de pCR; azul, la baja).`,
    dropout ? `En ${dropout.version}, durante el entrenamiento se apaga al azar el 30 % de los 128 valores (dropout) y los pesos se frenan con weight decay, para que no memorice.` : "",
  ].filter(Boolean));
  parametros([["Entra", "vector de 128"], ["Sale", "1 logit"], ["Parámetros", "128 pesos + 1 sesgo = 129"], ["Logit de esta red", num(logit, 3)]]);
  const lienzo = lienzoContribucion(mapa.data, mapa.dims[1], plano(1));
  $("detalle-visual").replaceChildren(figura(lienzo, "Qué zonas suben (rojo) o bajan (azul) el logit de esta red. Su suma más el sesgo es el logit.", "fig--grande"));
}

function detalleConjunto() {
  const versiones = estado.info.entrenamiento.versiones;
  $("detalle-titulo").textContent = "Treinta redes que votan";
  explicacion([
    "La red de arriba es una de treinta: 6 versiones × 5 folds. Para un corte, cada versión hace la media de los logits de sus cinco redes, y después se hace la media de las seis versiones. Cada versión pesa lo mismo.",
    "Ninguna versión sola ganaba a las demás, y cada una se equivoca en pacientes distintas. Juntas, la predicción depende menos de cómo salió un entrenamiento concreto.",
    "Las seis versiones entran por una regla fijada antes de ver el resultado: todas las que tienen validación cruzada completa y no sobreajustan.",
  ]);
  parametros([["Redes", String(estado.info.entrenamiento.redes)], ["Parámetros en total", miles(estado.info.entrenamiento.parametros)],
    ["Agregación", "media de logits por versión y luego entre versiones"]]);
  // Visual: AUC de validación de cada una de las 30 redes
  const tabla = nodo("table", null, "tabla tabla--treinta");
  const cab = nodo("tr");
  cab.append(nodo("th", "Versión"), ...[1, 2, 3, 4, 5].map((k) => nodo("th", `fold ${k}`)));
  const thead = nodo("thead");
  thead.append(cab);
  const tbody = nodo("tbody");
  for (const v of versiones) {
    const tr = nodo("tr");
    tr.append(nodo("th", v.version));
    (v.auc_val_por_fold || []).forEach((a) => {
      const td = nodo("td", num(a));
      td.style.setProperty("--intensidad", String(Math.max(0, Math.min(1, (a - 0.45) / 0.2))));
      tr.append(td);
    });
    tbody.append(tr);
  }
  tabla.append(thead, tbody);
  $("detalle-visual").replaceChildren(figura(tabla, "AUC por paciente de cada red en su fold de validación (pacientes que no vio). Más oscuro, más alto."));
}

function detalleCalibracion() {
  const { a, b } = estado.info.calibracion;
  const prob = estado.ejemplo.probabilidad_pipeline;
  const ref = estado.info.referencia || {};
  $("detalle-titulo").textContent = "De logit a probabilidad";
  explicacion([
    `El logit medio se recalibra con Platt: logit calibrado = a · logit + b, con a = ${num(a)} y b = ${num(b)}, ajustados con las predicciones fuera de fold (a cada paciente la predijeron las 6 redes de su fold; aquí se aplica a la media de las 30). Después, la sigmoide lo convierte en una probabilidad entre 0 y 1.`,
    `Que a sea tan pequeño quiere decir que la red sabe poco: la calibración aprieta todas las probabilidades cerca de la tasa de pCR de entrenamiento (${ref.tasa_pcr ? num(100 * ref.tasa_pcr, 0) + " %" : "29 %"}). No aparenta una seguridad que no tiene, pero tampoco dice mucho más que esa tasa: su error de Brier es casi el de decir a todas el ${ref.tasa_pcr ? num(100 * ref.tasa_pcr, 0) + " %" : "29 %"}.`,
  ]);
  parametros([["a", num(a, 3)], ["b", num(b, 3)], ["Probabilidad de este corte", num(prob, 4)],
    ...(ref.franja_90 ? [["El 90 % de las pacientes", `entre ${num(ref.franja_90[0], 2)} y ${num(ref.franja_90[1], 2)}`]] : [])]);
  // Visual: la curva sigmoide de la calibración, con el corte encima
  const W = 520, H = 290, x0 = 44, y0 = 16, ancho = W - x0 - 12, alto = H - y0 - 52;
  const logitMin = -6, logitMax = 6;
  const X = (l) => x0 + ((l - logitMin) / (logitMax - logitMin)) * ancho;
  const Y = (p) => y0 + (1 - p) * alto;
  const s = svg("svg", { viewBox: `0 0 ${W} ${H}`, class: "curva", role: "img", "aria-label": "Probabilidad según el logit medio de las 30 redes" });
  for (const p of [0, 0.25, 0.5, 0.75, 1]) {
    s.append(svg("line", { x1: x0, x2: x0 + ancho, y1: Y(p), y2: Y(p), class: "curva__rejilla" }));
    s.append(svg("text", { x: x0 - 8, y: Y(p) + 4, "text-anchor": "end", class: "curva__texto" }, num(p, 2)));
  }
  if (ref.franja_90) {
    // Los logits de las pacientes de entrenamiento: donde cae el 90 %, sombreado
    const inv = (p) => (Math.log(p / (1 - p)) - b) / a;
    const l0 = Math.max(logitMin, inv(ref.franja_90[0])), l1 = Math.min(logitMax, inv(ref.franja_90[1]));
    s.append(svg("rect", { x: X(l0), y: y0, width: Math.max(2, X(l1) - X(l0)), height: alto, class: "curva__franja" }));
    s.append(svg("text", { x: X(l1) + 6, y: y0 + 14, class: "curva__texto" }, "aquí cae el 90 % de las pacientes"));
  }
  let d = "";
  for (let l = logitMin; l <= logitMax + 1e-9; l += 0.1) d += `${d ? "L" : "M"}${X(l).toFixed(1)} ${Y(1 / (1 + Math.exp(-(a * l + b)))).toFixed(1)} `;
  s.append(svg("path", { d, class: "curva__linea" }));
  const lCorte = (Math.log(prob / (1 - prob)) - b) / a;
  if (lCorte > logitMin && lCorte < logitMax) {
    s.append(svg("circle", { cx: X(lCorte), cy: Y(prob), r: 6, class: "curva__punto" }));
    s.append(svg("text", { x: X(lCorte) - 10, y: Y(prob) - 14, "text-anchor": "end", class: "curva__texto curva__texto--fuerte" }, `este corte, ${num(prob)}`));
  }
  for (const l of [-6, -3, 0, 3, 6]) s.append(svg("text", { x: X(l), y: y0 + alto + 18, "text-anchor": "middle", class: "curva__texto" }, num(l, 0)));
  s.append(svg("text", { x: x0 + ancho / 2, y: H - 6, "text-anchor": "middle", class: "curva__texto curva__texto--eje" }, "logit medio de las 30 redes"));
  $("detalle-visual").replaceChildren(figura(s, "Con a tan pequeño, la curva es casi plana: logits muy distintos dan probabilidades parecidas."));
}

function detalleUmbral() {
  const m = estado.info.metricas;
  const umbral = estado.info.umbral;
  const prob = estado.ejemplo.probabilidad_pipeline;
  $("detalle-titulo").textContent = "El umbral: de probabilidad a clase";
  explicacion([
    `Si la probabilidad es ${num(umbral)} o más, la clase es pCR. El umbral no sale de la red: es una decisión. Se eligió, con las predicciones fuera de fold, el que detecta más pCR sin bajar del 80 % de especificidad.`,
    "La idea es que como mucho 1 de cada 5 pacientes sin pCR reciba un «pCR» equivocado, el error que más preocuparía si alguien pensara en relajar un tratamiento. El 80 % es una elección de este trabajo, no un criterio clínico validado.",
    `Aun así, de cada 100 pacientes que marca como pCR, solo ${Math.round(100 * m.precision_pcr)} lo son. Por eso no sirve para decidir un tratamiento.`,
    `Este corte da ${num(prob)}: ${prob >= umbral ? "por encima, pCR" : "por debajo, no pCR"}.`,
  ]);
  const vi = estado.info.validacion_interna;
  parametros([["Umbral", num(umbral, 4)], ["Criterio", estado.info.criterio_umbral],
    ["Sensibilidad", `${num(m.sensibilidad)}${vi ? `; ${num(vi.sensibilidad)} eligiendo el umbral sin el fold` : ""}`],
    ["Especificidad", `${num(m.especificidad)}${vi ? `; ${num(vi.especificidad)} sin el fold` : ""}`],
    ["Precisión pCR", `${num(m.precision_pcr)}${vi ? `; ${num(vi.precision_pcr)} sin el fold` : ""}`]]);
  $("detalle-visual").replaceChildren(nodo("p", "La página «Resultados» deja mover el umbral y ver cómo cambian los aciertos y los errores de cada modelo.", "nota"));
}

const DETALLES = {
  entrada: ["Paso 1 de 10", detalleEntrada], b1: ["Paso 2 de 10", () => detalleBloque(1)], b2: ["Paso 3 de 10", () => detalleBloque(2)],
  b3: ["Paso 4 de 10", () => detalleBloque(3)], b4: ["Paso 5 de 10", () => detalleBloque(4)], media: ["Paso 6 de 10", detalleMedia],
  lineal: ["Paso 7 de 10", detalleLineal], conjunto: ["Paso 8 de 10", detalleConjunto], calibracion: ["Paso 9 de 10", detalleCalibracion],
  umbral: ["Paso 10 de 10", detalleUmbral],
};

function elegirPieza(id) {
  estado.pieza = id;
  marcarPieza();
  const [paso, rellenar] = DETALLES[id];
  $("detalle-paso").textContent = paso;
  rellenar();
}

// ---------- Versiones, folds y aumento ----------
const DONDE = {
  cnn_v3: ["Entrenamiento: aumento geométrico (la base)", ["entrada"]],
  cnn_v4: ["Entrada: + realce y lavado (5 canales)", ["entrada", "b1"]],
  cnn_v5: ["Entrada: intensidad relativa al PRE", ["entrada"]],
  cnn_v6: ["Resumen: media ponderada por atención", ["media"]],
  cnn_v7: ["Cabeza: dropout y weight decay", ["lineal"]],
  cnn_v10: ["Entrenamiento: + giros de 90°", ["entrada"]],
};
function tablaVersiones() {
  const cuerpo = $("tabla-versiones").tBodies[0];
  cuerpo.replaceChildren(...estado.info.entrenamiento.versiones.map((v) => {
    const [donde, piezas] = DONDE[v.version] || ["—", []];
    const tr = nodo("tr");
    tr.tabIndex = 0;
    const th = nodo("th", v.version);
    th.scope = "row";
    tr.append(th, nodo("td", v.que_cambia || "—"), nodo("td", donde), nodo("td", miles(v.parametros)),
      nodo("td", v.auc_val == null ? "—" : `${num(v.auc_val)} ± ${num(v.auc_val_desviacion)}`));
    const resaltar = (si) => {
      for (const g of document.querySelectorAll(".pieza")) g.classList.toggle("pieza--resaltada", si && piezas.includes(g.dataset.pieza));
    };
    tr.addEventListener("pointerenter", () => resaltar(true));
    tr.addEventListener("pointerleave", () => resaltar(false));
    tr.addEventListener("focus", () => resaltar(true));
    tr.addEventListener("blur", () => resaltar(false));
    return tr;
  }));
}

function folds() {
  const caja = $("folds");
  caja.replaceChildren();
  caja.append(nodo("span", "", "folds__esquina"), ...[1, 2, 3, 4, 5].map((k) => nodo("span", `fold ${k}`, "folds__cab")));
  for (let red = 0; red < 5; red++) {
    caja.append(nodo("span", `red ${red + 1}`, "folds__fila"));
    for (let k = 0; k < 5; k++) {
      const celda = nodo("span", k === red ? "mide" : "aprende", k === red ? "folds__celda folds__celda--val" : "folds__celda");
      caja.append(celda);
    }
  }
  const e = estado.info.entrenamiento;
  $("receta").replaceChildren(...[
    ["Pacientes por red", "unas 878 para aprender y 219 para medir"],
    ["Optimizador", `Adam, learning rate ${String(e.learning_rate).replace(".", ",")}, lotes de ${e.batch} cortes`],
    ["Épocas", `${e.epocas}, fijadas de antemano; se usa la última`],
    ["Pérdida", "entropía cruzada binaria"],
    ["Semilla", String(e.semilla)],
    ["Máquina", `${e.gpu || "GPU"}, unos 16 minutos por versión`],
  ].flatMap(([t, d]) => [nodo("dt", t), nodo("dd", d)]));
}

function aumento() {
  const caja = $("aumento");
  const original = lienzoFase(1);
  const sorteos = [{ nombre: "original" }];
  for (let i = 0; i < 6; i++) {
    sorteos.push({
      grados: (Math.random() * 2 - 1) * 10, dx: (Math.random() * 2 - 1) * 0.04 * 256, dy: (Math.random() * 2 - 1) * 0.04 * 256,
      escala: 0.95 + Math.random() * 0.1, espejo: Math.random() < 0.5, giro90: i === 5 ? 90 * (1 + Math.floor(Math.random() * 3)) : 0,
    });
  }
  caja.replaceChildren(...sorteos.map((s) => {
    const c = document.createElement("canvas");
    c.width = c.height = 256;
    const ctx = c.getContext("2d");
    ctx.fillStyle = "#000";
    ctx.fillRect(0, 0, 256, 256);
    ctx.translate(128 + (s.dx || 0), 128 + (s.dy || 0));
    ctx.rotate((((s.grados || 0) + (s.giro90 || 0)) * Math.PI) / 180);
    ctx.scale((s.espejo ? -1 : 1) * (s.escala || 1), s.escala || 1);
    ctx.drawImage(original, -128, -128);
    const pie = s.nombre || [`${s.grados >= 0 ? "+" : "−"}${Math.abs(s.grados).toFixed(0)}°`, `zoom ×${num(s.escala, 2)}`,
      s.espejo ? "espejo" : null, s.giro90 ? `giro de ${s.giro90}° (cnn_v10)` : null].filter(Boolean).join(", ");
    return figura(c, pie);
  }));
}

// ---------- Arranque ----------
async function usarEjemplo(e) {
  estado.ejemplo = e;
  for (const b of document.querySelectorAll("#selector button")) b.setAttribute("aria-pressed", String(b.dataset.corte === e.corte));
  const r = await validarCorte(await ficherosDeEjemplo(e));
  if (r.errores.length) throw new Error(r.errores.join(" "));
  estado.pixeles = r.pixeles;
  estado.salidas = await estado.ejecutar(r.pixeles);
  dibujarDiagrama();
  elegirPieza(estado.pieza);
  aumento();
}

function selector() {
  $("selector").replaceChildren(...estado.ejemplos.map((e) => {
    const li = nodo("li");
    const b = nodo("button", null, "ejemplo");
    b.type = "button";
    b.dataset.corte = e.corte;
    b.setAttribute("aria-pressed", "false");
    b.setAttribute("aria-label", `${COHORTES[e.cohorte] || e.cohorte}, ${e.corte}`);
    const img = document.createElement("img");
    img.alt = "";
    img.src = `ejemplos/${encodeURIComponent(e.corte)}/${encodeURIComponent(e.ficheros[1])}`;
    const etiqueta = e.pcr == null ? "" : e.pcr ? "etiqueta real: pCR" : "etiqueta real: no\u00a0pCR";
    b.append(img, nodo("span", COHORTES[e.cohorte] || e.cohorte, "ejemplo__cohorte"), nodo("span", etiqueta, "ejemplo__etiqueta"));
    b.addEventListener("click", () => usarEjemplo(e).catch((err) => { $("diagrama-estado").textContent = err.message; }));
    li.append(b);
    return li;
  }));
}

async function arrancar() {
  try {
    [estado.info, estado.ejemplos] = await Promise.all([leerJSON("modelo/info.json"), leerJSON("ejemplos/ejemplos.json")]);
    selector();
    tablaVersiones();
    folds();
    $("otra-vez").addEventListener("click", aumento);
    estado.ejecutar = await cargarUnaRed(estado.info.una_red);
    await usarEjemplo(estado.ejemplos[0]);
    $("diagrama-estado").textContent = window.matchMedia("(max-width: 900px)").matches
      ? "Desliza a los lados para ver toda la red y pincha en cualquier pieza."
      : "Pincha en cualquier pieza del diagrama.";
  } catch (error) {
    $("diagrama-estado").textContent = `No se pudo cargar la red: ${error.message}`;
  }
}

arrancar();
