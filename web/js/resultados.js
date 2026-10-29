// Página «Resultados» (D39): la CNN frente al médico y frente a los dos juntos, con las gráficas que recomiendan
// las guías para evaluar modelos clínicos. Los datos los prepara src/models/datos_web.py (datos/resultados.json),
// todos de entrenamiento y fuera de fold. Las gráficas se dibujan con D3 al ancho de su tarjeta.

import { $, COLORES, NOMBRES_CORTOS, leerJSON, miles, nodo, num } from "./comun.js";

const d3 = window.d3;
const MODELOS = ["cnn", "medico", "combinado"];
const estado = { datos: null, visibles: new Set(MODELOS), umbral: 0.3 };
const TINTA = "#18232e", TINTA_2 = "#4f5d6a", REJILLA = "#e4e8eb", EJE = "#c3ccd3", REFERENCIA = "#9aa6b0";

// ---------- Utilidades ----------
const visibles = () => MODELOS.filter((m) => estado.visibles.has(m) && estado.datos.modelos_datos?.[m]);
const tooltip = $("tooltip");
function mostrarTooltip(evento, lineas) {
  tooltip.replaceChildren(...lineas.map(([color, texto, fuerte]) => {
    const p = nodo("p", null, "tooltip__linea");
    if (color) {
      const s = nodo("span", null, "tooltip__muestra");
      s.style.background = color;
      p.append(s);
    }
    p.append(fuerte ? nodo("strong", texto) : document.createTextNode(texto));
    return p;
  }));
  tooltip.hidden = false;
  const margen = 14, ancho = tooltip.offsetWidth, alto = tooltip.offsetHeight;
  let x = evento.clientX + margen, y = evento.clientY + margen;
  if (x + ancho > window.innerWidth - 8) x = evento.clientX - ancho - margen;
  if (y + alto > window.innerHeight - 8) y = evento.clientY - alto - margen;
  tooltip.style.left = `${x}px`;
  tooltip.style.top = `${y}px`;
}
const ocultarTooltip = () => { tooltip.hidden = true; };

function lienzo(contenedor, alto, margen = { arriba: 16, derecha: 18, abajo: 44, izquierda: 52 }) {
  const ancho = Math.max(280, contenedor.clientWidth);
  const s = d3.select(contenedor).append("svg").attr("width", ancho).attr("height", alto).attr("viewBox", `0 0 ${ancho} ${alto}`)
    .attr("role", "img");
  const g = s.append("g").attr("transform", `translate(${margen.izquierda},${margen.arriba})`);
  return { s, g, ancho: ancho - margen.izquierda - margen.derecha, alto: alto - margen.arriba - margen.abajo, margen };
}

function ejes(g, x, y, ancho, alto, { titx, tity, fx = (v) => num(v, 1), fy = (v) => num(v, 1), nx = 5, ny = 5 } = {}) {
  g.append("g").attr("class", "rejilla").selectAll("line").data(y.ticks(ny)).join("line")
    .attr("x1", 0).attr("x2", ancho).attr("y1", (d) => y(d)).attr("y2", (d) => y(d));
  const ex = g.append("g").attr("class", "eje").attr("transform", `translate(0,${alto})`).call(d3.axisBottom(x).ticks(nx).tickFormat(fx).tickSizeOuter(0));
  const ey = g.append("g").attr("class", "eje").call(d3.axisLeft(y).ticks(ny).tickFormat(fy).tickSizeOuter(0));
  ey.select(".domain").remove();
  if (titx) ex.append("text").attr("class", "eje__titulo").attr("x", ancho / 2).attr("y", 36).attr("text-anchor", "middle").text(titx);
  if (tity) ey.append("text").attr("class", "eje__titulo").attr("transform", "rotate(-90)").attr("x", -alto / 2).attr("y", -40).attr("text-anchor", "middle").text(tity);
}

function tabla(contenedor, cabecera, filas) {
  const t = nodo("table", null, "tabla tabla--datos");
  const tr = nodo("tr");
  tr.append(...cabecera.map((c) => nodo("th", c)));
  const thead = nodo("thead");
  thead.append(tr);
  const tbody = nodo("tbody");
  for (const f of filas) {
    const fila = nodo("tr");
    fila.append(...f.map((c) => nodo("td", c)));
    tbody.append(fila);
  }
  t.append(thead, tbody);
  contenedor.replaceChildren(t);
}

function leyenda(contenedor, elementos) {
  const l = nodo("div", null, "leyenda");
  for (const [color, texto, forma = "linea"] of elementos) {
    const e = nodo("span", null, "leyenda__elemento");
    const m = nodo("span", null, `leyenda__muestra leyenda__muestra--${forma}`);
    m.style.setProperty("--color", color);
    e.append(m, document.createTextNode(texto));
    l.append(e);
  }
  contenedor.append(l);
}

// ---------- Cifras de arriba ----------
function cifras() {
  const d = estado.datos;
  const caja = $("cifras");
  caja.replaceChildren(...MODELOS.map((m) => {
    const r = d.modelos_datos[m];
    const t = nodo("div", null, "cifra-modelo");
    const cab = nodo("p", null, "cifra-modelo__nombre");
    const muestra = nodo("span", null, "cifra-modelo__muestra");
    muestra.style.background = COLORES[m];
    cab.append(muestra, document.createTextNode(NOMBRES_CORTOS[m]));
    t.append(cab, nodo("p", num(r.auc_val), "cifra-modelo__valor"),
      nodo("p", `AUC, media de 5 folds ± ${num(r.auc_val_desviacion)}`, "cifra-modelo__detalle"),
      nodo("p", r.auc_val_ic95 ? `IC 95 % ${num(r.auc_val_ic95[0], 2)}–${num(r.auc_val_ic95[1], 2)}` : "", "cifra-modelo__detalle"),
      nodo("p", `Todas juntas: ${num(r.auc_oof)}`, "cifra-modelo__detalle cifra-modelo__detalle--tenue"));
    return t;
  }));
  // La diferencia se calcula con la misma cifra que los recuadros (media de folds), así que cuadra con ellos
  const c = d.comparacion.combinado_menos_medico;
  const signo = (v) => `${v > 0 ? "+" : ""}${num(v)}`;
  const cruza = c.ic95[0] <= 0 && c.ic95[1] >= 0;
  const suma = (c.por_fold || []).filter((v) => v > 0).length;
  const nota = nodo("div", null, "cifra-nota");
  nota.append(
    nodo("p", "Lo que añade la imagen al médico", "cifra-nota__titulo"),
    nodo("p", signo(c.diferencia), "cifra-nota__valor"),
    nodo("p", `${num(d.modelos_datos.combinado.auc_val)} − ${num(d.modelos_datos.medico.auc_val)}. IC 95 % ${signo(c.ic95[0])} a ${signo(c.ic95[1])}`
      + `${cruza ? ": incluye el 0, así que no se puede afirmar que añada algo" : ""}. Suma en ${suma} de ${(c.por_fold || []).length} folds. `
      + `Saber solo la cohorte da ${num(d.auc_solo_hospital)}; el azar, 0,5.`, "cifra-nota__texto"),
  );
  caja.append(nota);
}

function filtros() {
  $("filtros").replaceChildren(...MODELOS.map((m) => {
    const b = nodo("button", null, "chip");
    b.type = "button";
    b.setAttribute("aria-pressed", "true");
    const s = nodo("span", null, "chip__muestra");
    s.style.background = COLORES[m];
    b.append(s, document.createTextNode(NOMBRES_CORTOS[m]));
    b.addEventListener("click", () => {
      if (estado.visibles.has(m) && estado.visibles.size === 1) return;   // siempre queda al menos uno
      estado.visibles.has(m) ? estado.visibles.delete(m) : estado.visibles.add(m);
      b.setAttribute("aria-pressed", String(estado.visibles.has(m)));
      dibujarTodo();
    });
    return b;
  }));
}

// ---------- ROC ----------
function roc(c) {
  const { g, ancho, alto } = lienzo(c, Math.min(440, c.clientWidth * 0.9 + 40));
  const x = d3.scaleLinear([0, 1], [0, ancho]), y = d3.scaleLinear([0, 1], [alto, 0]);
  ejes(g, x, y, ancho, alto, { titx: "1 − especificidad (no pCR marcadas como pCR)", tity: "Sensibilidad (pCR detectadas)" });
  g.append("line").attr("class", "referencia").attr("x1", 0).attr("y1", alto).attr("x2", ancho).attr("y2", 0);
  g.append("text").attr("class", "anotacion").attr("x", x(0.74)).attr("y", y(0.7) + 18).attr("transform", `rotate(${-Math.atan2(alto, ancho) * 180 / Math.PI} ${x(0.74)} ${y(0.7) + 18})`).text("azar");
  const linea = d3.line().x((p) => x(p[0])).y((p) => y(p[1]));
  for (const m of visibles()) {
    const r = estado.datos.modelos_datos[m];
    g.append("path").attr("d", linea(r.roc)).attr("fill", "none").attr("stroke", COLORES[m]).attr("stroke-width", 2).attr("stroke-linejoin", "round");
    if (r.con_umbral) {
      const fx = 1 - r.con_umbral.especificidad, fy = r.con_umbral.sensibilidad;
      g.append("circle").attr("class", "marca").attr("cx", x(fx)).attr("cy", y(fy)).attr("r", 5).attr("fill", COLORES[m]);
    }
  }
  // Pasar el ratón: para cada modelo, el punto de su curva a esa altura de falsos positivos
  const guia = g.append("line").attr("class", "guia").attr("y1", 0).attr("y2", alto).style("display", "none");
  const marcas = g.append("g");
  g.append("rect").attr("width", ancho).attr("height", alto).attr("fill", "transparent")
    .on("pointermove", (e) => {
      const fx = Math.max(0, Math.min(1, x.invert(d3.pointer(e)[0])));
      guia.style("display", null).attr("x1", x(fx)).attr("x2", x(fx));
      const lineas = [[null, `1 − especificidad ${num(fx, 2)}`, true]];
      marcas.selectAll("*").remove();
      for (const m of visibles()) {
        const puntos = estado.datos.modelos_datos[m].roc;
        const p = puntos.reduce((a, b) => (Math.abs(b[0] - fx) < Math.abs(a[0] - fx) ? b : a));
        marcas.append("circle").attr("cx", x(p[0])).attr("cy", y(p[1])).attr("r", 4.5).attr("fill", COLORES[m]).attr("stroke", "#fff").attr("stroke-width", 2);
        lineas.push([COLORES[m], `${NOMBRES_CORTOS[m]}: sensibilidad ${num(p[1], 2)}, umbral ${num(p[2], 3)}`]);
      }
      mostrarTooltip(e, lineas);
    })
    .on("pointerleave", () => { guia.style("display", "none"); marcas.selectAll("*").remove(); ocultarTooltip(); });
  leyenda(c, [...visibles().map((m) => [COLORES[m], `${NOMBRES_CORTOS[m]} (AUC de la curva ${num(estado.datos.modelos_datos[m].auc_oof)})`]),
    [TINTA, "el umbral elegido de cada modelo", "punto"]]);
  c.append(nodo("p", "Cada curva junta a todas las pacientes, así que su AUC es el de todas juntas. La cifra principal de cada modelo es la media de los 5 folds (arriba): solo compara pacientes predichas por la misma red.", "nota nota--grafica"));
  tabla(document.querySelector('[data-tabla="roc"]'), ["Modelo", "AUC media de folds", "IC 95 %", "AUC todas juntas", "IC 95 %"],
    MODELOS.map((m) => {
      const r = estado.datos.modelos_datos[m];
      const ic = r.auc_val_ic95 ? `${num(r.auc_val_ic95[0])}–${num(r.auc_val_ic95[1])}` : "—";
      return [NOMBRES_CORTOS[m], `${num(r.auc_val)} ± ${num(r.auc_val_desviacion)}`, ic, num(r.auc_oof), `${num(r.auc_oof_ic95[0])}–${num(r.auc_oof_ic95[1])}`];
    }));
}

// ---------- Calibración ----------
function calibracion(c) {
  const { g, ancho, alto } = lienzo(c, Math.min(440, c.clientWidth * 0.9 + 40));
  const max = 0.8;
  const x = d3.scaleLinear([0, max], [0, ancho]), y = d3.scaleLinear([0, max], [alto, 0]);
  ejes(g, x, y, ancho, alto, { titx: "Probabilidad que da el modelo (media del grupo)", tity: "Proporción real de pCR", fx: (v) => num(v, 1), fy: (v) => num(v, 1) });
  g.append("line").attr("class", "referencia").attr("x1", 0).attr("y1", alto).attr("x2", ancho).attr("y2", 0);
  g.append("text").attr("class", "anotacion").attr("x", x(0.6)).attr("y", y(0.62)).attr("text-anchor", "end").text("calibración perfecta");
  const tasa = estado.datos.tasa_pcr;
  g.append("line").attr("class", "referencia referencia--tenue").attr("x1", 0).attr("x2", ancho).attr("y1", y(tasa)).attr("y2", y(tasa));
  g.append("text").attr("class", "anotacion").attr("x", ancho - 4).attr("y", y(tasa) - 6).attr("text-anchor", "end").text(`tasa de pCR, ${num(100 * tasa, 0)} %`);
  const visiblesM = visibles();
  visiblesM.forEach((m, k) => {
    const puntos = estado.datos.modelos_datos[m].calibracion;
    const desplaza = (k - (visiblesM.length - 1) / 2) * 3;
    const grupo = g.append("g");
    grupo.append("path").attr("d", d3.line().x((p) => x(p.predicha) + desplaza).y((p) => y(p.observada))(puntos))
      .attr("fill", "none").attr("stroke", COLORES[m]).attr("stroke-width", 1.5).attr("stroke-opacity", 0.55);
    grupo.selectAll("line.ic").data(puntos).join("line").attr("class", "ic")
      .attr("x1", (p) => x(p.predicha) + desplaza).attr("x2", (p) => x(p.predicha) + desplaza)
      .attr("y1", (p) => y(p.ic95[0])).attr("y2", (p) => y(Math.min(max, p.ic95[1]))).attr("stroke", COLORES[m]).attr("stroke-width", 1.5).attr("stroke-opacity", 0.6);
    grupo.selectAll("circle").data(puntos).join("circle").attr("class", "marca")
      .attr("cx", (p) => x(p.predicha) + desplaza).attr("cy", (p) => y(p.observada)).attr("r", 5).attr("fill", COLORES[m])
      .on("pointermove", (e, p) => mostrarTooltip(e, [[COLORES[m], NOMBRES_CORTOS[m], true],
        [null, `Decía ${num(p.predicha)}; lograron pCR ${num(p.observada)} (IC ${num(p.ic95[0], 2)}–${num(p.ic95[1], 2)})`],
        [null, `${p.pacientes} pacientes`]]))
      .on("pointerleave", ocultarTooltip);
  });
  leyenda(c, visibles().map((m) => [COLORES[m], NOMBRES_CORTOS[m], "punto"]));
  const d = estado.datos, vi = d.validacion_interna;
  if (d.brier_constante != null) {
    let texto = `Error de Brier (cuanto menos, mejor): decir a todas la tasa de pCR, ${num(d.brier_constante, 4)}; `
      + MODELOS.map((m) => `${NOMBRES_CORTOS[m]}, ${num(d.modelos_datos[m].brier, 4)}`).join("; ") + ".";
    if (vi) {
      texto += ` La CNN se calibró con estas mismas pacientes. Con la calibración elegida sin el fold que se mide, su Brier es ${num(vi.brier_platt_cruzado, 4)}: `
        + (vi.brier_platt_cruzado < d.brier_constante ? "bate por poco a decir a todas la tasa de pCR." : "no bate a decir a todas la tasa de pCR. Como probabilidad, dice lo mismo que esa tasa.");
    }
    c.append(nodo("p", texto, "nota nota--grafica"));
  }
  tabla(document.querySelector('[data-tabla="calibracion"]'), ["Modelo", "Grupo", "Predicha", "Observada", "IC 95 %", "Pacientes"],
    MODELOS.flatMap((m) => estado.datos.modelos_datos[m].calibracion.map((p, i) =>
      [NOMBRES_CORTOS[m], String(i + 1), num(p.predicha), num(p.observada), `${num(p.ic95[0])}–${num(p.ic95[1])}`, String(p.pacientes)])));
}

// ---------- Distribución del riesgo (múltiplos pequeños) ----------
function distribucion(c) {
  const bordes = estado.datos.histograma_bordes;
  const ms = visibles();
  const altoPanel = 120;
  const { g, ancho } = lienzo(c, ms.length * altoPanel + 50, { arriba: 10, derecha: 18, abajo: 40, izquierda: 52 });
  const x = d3.scaleLinear([0, 1], [0, ancho]);
  ms.forEach((m, i) => {
    const h = estado.datos.modelos_datos[m].histograma;
    const totPcr = d3.sum(h.pcr), totNo = d3.sum(h.no_pcr);
    const fp = h.pcr.map((v) => v / totPcr), fn = h.no_pcr.map((v) => v / totNo);
    const y = d3.scaleLinear([0, d3.max([...fp, ...fn])], [altoPanel - 26, 20]);
    const panel = g.append("g").attr("transform", `translate(0,${i * altoPanel})`);
    panel.append("line").attr("class", "base").attr("x1", 0).attr("x2", ancho).attr("y1", altoPanel - 26).attr("y2", altoPanel - 26);
    const area = d3.area().curve(d3.curveStepAfter).x((_, k) => x(bordes[k])).y0(altoPanel - 26).y1((v) => y(v));
    const linea = d3.line().curve(d3.curveStepAfter).x((_, k) => x(bordes[k])).y((v) => y(v));
    panel.append("path").attr("d", area([...fp, fp[fp.length - 1]])).attr("fill", COLORES[m]).attr("fill-opacity", 0.28);
    panel.append("path").attr("d", linea([...fp, fp[fp.length - 1]])).attr("fill", "none").attr("stroke", COLORES[m]).attr("stroke-width", 2);
    panel.append("path").attr("d", linea([...fn, fn[fn.length - 1]])).attr("fill", "none").attr("stroke", TINTA).attr("stroke-width", 1.5).attr("stroke-dasharray", null);
    panel.append("text").attr("class", "panel-titulo").attr("x", 0).attr("y", 12).text(NOMBRES_CORTOS[m]);
    panel.append("rect").attr("width", ancho).attr("height", altoPanel - 26).attr("fill", "transparent")
      .on("pointermove", (e) => {
        const k = Math.max(0, Math.min(bordes.length - 2, Math.floor(x.invert(d3.pointer(e)[0]) * (bordes.length - 1))));
        mostrarTooltip(e, [[COLORES[m], `${NOMBRES_CORTOS[m]}, probabilidad ${num(bordes[k], 3)}–${num(bordes[k + 1], 3)}`, true],
          [COLORES[m], `con pCR: ${h.pcr[k]} pacientes`], [TINTA, `sin pCR: ${h.no_pcr[k]} pacientes`]]);
      })
      .on("pointerleave", ocultarTooltip);
  });
  const eje = g.append("g").attr("class", "eje").attr("transform", `translate(0,${ms.length * altoPanel - 26})`)
    .call(d3.axisBottom(x).ticks(5).tickFormat((v) => num(v, 1)).tickSizeOuter(0));
  eje.append("text").attr("class", "eje__titulo").attr("x", ancho / 2).attr("y", 36).attr("text-anchor", "middle").text("Probabilidad de pCR que da el modelo");
  leyenda(c, [[TINTA_2, "con pCR (relleno, color del modelo)", "area"], [TINTA, "sin pCR (línea negra)"]]);
  const filas = [];
  for (const m of MODELOS) {
    const h = estado.datos.modelos_datos[m].histograma;
    h.pcr.forEach((v, k) => { if (v || h.no_pcr[k]) filas.push([NOMBRES_CORTOS[m], `${num(bordes[k], 3)}–${num(bordes[k + 1], 3)}`, String(v), String(h.no_pcr[k])]); });
  }
  tabla(document.querySelector('[data-tabla="distribucion"]'), ["Modelo", "Probabilidad", "Con pCR", "Sin pCR"], filas);
}

// ---------- Curva de decisión ----------
function decision(c) {
  const d = estado.datos;
  const u = d.decision_umbrales;
  const { g, ancho, alto } = lienzo(c, Math.min(420, c.clientWidth * 0.85 + 40));
  const todos = [...visibles().flatMap((m) => d.modelos_datos[m].decision), ...d.decision_todas];
  const x = d3.scaleLinear([u[0], u[u.length - 1]], [0, ancho]);
  const y = d3.scaleLinear([Math.max(-0.05, d3.min(todos)), Math.max(0.32, d3.max(todos))], [alto, 0]).nice();
  ejes(g, x, y, ancho, alto, { titx: "Umbral de riesgo a partir del cual se actuaría", tity: "Beneficio neto", fx: (v) => num(v, 1), fy: (v) => num(v, 2) });
  g.append("line").attr("class", "referencia").attr("x1", 0).attr("x2", ancho).attr("y1", y(0)).attr("y2", y(0));
  const linea = d3.line().defined((v) => v != null && v >= y.domain()[0]).x((_, k) => x(u[k])).y((v) => y(v));
  g.append("path").attr("d", linea(d.decision_todas)).attr("fill", "none").attr("stroke", REFERENCIA).attr("stroke-width", 2);
  g.append("text").attr("class", "anotacion").attr("x", x(0.07)).attr("y", y(d.decision_todas[2]) - 8).text("actuar con todas");
  g.append("text").attr("class", "anotacion").attr("x", ancho - 4).attr("y", y(0) - 6).attr("text-anchor", "end").text("con ninguna");
  for (const m of visibles()) {
    g.append("path").attr("d", linea(d.modelos_datos[m].decision)).attr("fill", "none").attr("stroke", COLORES[m]).attr("stroke-width", 2);
  }
  const guia = g.append("line").attr("class", "guia").attr("y1", 0).attr("y2", alto).style("display", "none");
  g.append("rect").attr("width", ancho).attr("height", alto).attr("fill", "transparent")
    .on("pointermove", (e) => {
      const k = d3.minIndex(u, (v) => Math.abs(x(v) - d3.pointer(e)[0]));
      guia.style("display", null).attr("x1", x(u[k])).attr("x2", x(u[k]));
      mostrarTooltip(e, [[null, `Umbral ${num(u[k], 2)}`, true],
        ...visibles().map((m) => [COLORES[m], `${NOMBRES_CORTOS[m]}: ${num(d.modelos_datos[m].decision[k], 3)}`]),
        [REFERENCIA, `Actuar con todas: ${num(d.decision_todas[k], 3)}`]]);
    })
    .on("pointerleave", () => { guia.style("display", "none"); ocultarTooltip(); });
  leyenda(c, [...visibles().map((m) => [COLORES[m], NOMBRES_CORTOS[m]]), [REFERENCIA, "actuar con todas"]]);
  tabla(document.querySelector('[data-tabla="decision"]'), ["Umbral", ...MODELOS.map((m) => NOMBRES_CORTOS[m]), "Todas"],
    u.filter((_, k) => k % 5 === 0).map((v) => {
      const k = u.indexOf(v);
      return [num(v, 2), ...MODELOS.map((m) => num(d.modelos_datos[m].decision[k], 3)), num(d.decision_todas[k], 3)];
    }));
}

// ---------- Umbral interactivo ----------
function confusion(puntos, umbral) {
  let vp = 0, fn = 0, fp = 0, vn = 0;
  for (const [p, y] of puntos) {
    if (p >= umbral) y ? vp++ : fp++;
    else y ? fn++ : vn++;
  }
  return { vp, fn, fp, vn };
}
function matrices(usarPropio = false) {
  const caja = $("matrices");
  const u = estado.umbral;
  $("valor-umbral").textContent = usarPropio ? "el de cada modelo" : num(u);
  caja.replaceChildren(...visibles().map((m) => {
    const r = estado.datos.modelos_datos[m];
    const umbral = usarPropio && r.umbral != null ? r.umbral : u;
    const { vp, fn, fp, vn } = confusion(r.puntos, umbral);
    const sens = vp / (vp + fn), espec = vn / (vn + fp), vpp = vp + fp ? vp / (vp + fp) : NaN, vpn = vn + fn ? vn / (vn + fn) : NaN;
    const t = nodo("div", null, "matriz");
    const cab = nodo("p", null, "matriz__titulo");
    const s = nodo("span", null, "chip__muestra");
    s.style.background = COLORES[m];
    cab.append(s, document.createTextNode(`${NOMBRES_CORTOS[m]}, umbral ${num(umbral)}`));
    const rejilla = nodo("div", null, "matriz__rejilla");
    const total = vp + fn + fp + vn;
    const celda = (valor, texto, clase) => {
      const c = nodo("div", null, `matriz__celda ${clase}`);
      c.style.setProperty("--peso", String(valor / total));
      c.append(nodo("span", miles(valor), "matriz__numero"), nodo("span", texto, "matriz__texto"));
      return c;
    };
    rejilla.append(
      nodo("span", "", "matriz__esquina"), nodo("span", "dice pCR", "matriz__cab"), nodo("span", "dice no pCR", "matriz__cab"),
      nodo("span", "con pCR", "matriz__fila"), celda(vp, "aciertos", "matriz__celda--bien"), celda(fn, "se le escapan", "matriz__celda--mal"),
      nodo("span", "sin pCR", "matriz__fila"), celda(fp, "falsas alarmas", "matriz__celda--mal"), celda(vn, "aciertos", "matriz__celda--bien"),
    );
    const metricas = nodo("dl", null, "matriz__metricas");
    for (const [k, v] of [["Sensibilidad", sens], ["Especificidad", espec], ["Valor predictivo +", vpp], ["Valor predictivo −", vpn]]) {
      metricas.append(nodo("dt", k), nodo("dd", num(v)));
    }
    t.append(cab, rejilla, metricas);
    return t;
  }));
}

function notaUmbral() {
  const vi = estado.datos.validacion_interna;
  const base = "El umbral de cada modelo se eligió con estas mismas pacientes, así que sus cifras podrían salir algo mejor de lo que serán con pacientes nuevas. "
    + "El 80 % de especificidad es una elección de este trabajo para limitar los «pCR» equivocados, no un criterio clínico validado.";
  if (!vi) { $("umbral-nota").textContent = base; return; }
  const f = (r) => `sensibilidad ${num(r.sensibilidad)}, especificidad ${num(r.especificidad)} y valor predictivo + ${num(r.precision_pcr)}`;
  $("umbral-nota").textContent = `${base} Eligiendo el umbral (y la calibración) sin el fold que se mide, la CNN da ${f(vi.cnn_cruzado)}`
    + (vi.medico_cruzado ? `; el médico, ${f(vi.medico_cruzado)}.` : ".");
}

// ---------- Subgrupos (bosque) ----------
function subgrupos(c) {
  const d = estado.datos;
  const filas = [...d.por_subtipo.map((f) => ({ ...f, tipo: "Subtipo" })), ...d.por_hospital.map((f) => ({ ...f, tipo: "Cohorte" }))];
  const ms = visibles();
  const altoFila = 18 + ms.length * 9;
  const { g, ancho, alto } = lienzo(c, filas.length * altoFila + 70, { arriba: 10, derecha: 18, abajo: 44, izquierda: 150 });
  const x = d3.scaleLinear([0.3, 0.95], [0, ancho]);
  g.append("g").attr("class", "rejilla").selectAll("line").data(x.ticks(6)).join("line").attr("x1", (v) => x(v)).attr("x2", (v) => x(v)).attr("y1", 0).attr("y2", alto);
  g.append("line").attr("class", "referencia").attr("x1", x(0.5)).attr("x2", x(0.5)).attr("y1", 0).attr("y2", alto);
  g.append("text").attr("class", "anotacion").attr("x", x(0.5) + 4).attr("y", alto - 4).text("azar");
  const eje = g.append("g").attr("class", "eje").attr("transform", `translate(0,${alto})`).call(d3.axisBottom(x).ticks(6).tickFormat((v) => num(v, 1)).tickSizeOuter(0));
  eje.append("text").attr("class", "eje__titulo").attr("x", ancho / 2).attr("y", 36).attr("text-anchor", "middle").text("AUC dentro del grupo (IC 95 %)");
  filas.forEach((f, i) => {
    const y0 = i * altoFila + 8;
    if (i === d.por_subtipo.length) g.append("line").attr("class", "separador").attr("x1", -140).attr("x2", ancho).attr("y1", y0 - 6).attr("y2", y0 - 6);
    g.append("text").attr("class", "fila-titulo").attr("x", -12).attr("y", y0 + (ms.length * 9) / 2 + 2).attr("text-anchor", "end").text(f.grupo);
    g.append("text").attr("class", "fila-detalle").attr("x", -12).attr("y", y0 + (ms.length * 9) / 2 + 15).attr("text-anchor", "end").text(`${f.pacientes} pacientes, ${f.pcr} pCR`);
    ms.forEach((m, k) => {
      const v = f[m];
      const y = y0 + k * 9 + 4;
      g.append("line").attr("x1", x(v.ic95[0])).attr("x2", x(v.ic95[1])).attr("y1", y).attr("y2", y).attr("stroke", COLORES[m]).attr("stroke-width", 2).attr("stroke-linecap", "round");
      g.append("circle").attr("class", "marca").attr("cx", x(v.auc)).attr("cy", y).attr("r", 4.5).attr("fill", COLORES[m])
        .on("pointermove", (e) => mostrarTooltip(e, [[COLORES[m], `${NOMBRES_CORTOS[m]}, ${f.grupo}`, true],
          [null, `AUC ${num(v.auc)} (IC ${num(v.ic95[0])}–${num(v.ic95[1])})`], [null, `${f.pacientes} pacientes, ${f.pcr} con pCR`]]))
        .on("pointerleave", ocultarTooltip);
    });
  });
  leyenda(c, ms.map((m) => [COLORES[m], NOMBRES_CORTOS[m], "punto"]));
  tabla(document.querySelector('[data-tabla="subgrupos"]'), ["Grupo", "Pacientes", "pCR", ...MODELOS.map((m) => NOMBRES_CORTOS[m])],
    filas.map((f) => [f.grupo, String(f.pacientes), String(f.pcr), ...MODELOS.map((m) => `${num(f[m].auc)} (${num(f[m].ic95[0])}–${num(f[m].ic95[1])})`)]));
}

// ---------- Folds ----------
function folds(c) {
  const { g, ancho, alto } = lienzo(c, Math.min(340, c.clientWidth * 0.7 + 40));
  const x = d3.scalePoint([1, 2, 3, 4, 5], [0, ancho]).padding(0.4);
  const y = d3.scaleLinear([0.45, 0.8], [alto, 0]);
  g.append("g").attr("class", "rejilla").selectAll("line").data(y.ticks(5)).join("line").attr("x1", 0).attr("x2", ancho).attr("y1", (v) => y(v)).attr("y2", (v) => y(v));
  g.append("g").attr("class", "eje").attr("transform", `translate(0,${alto})`).call(d3.axisBottom(x).tickFormat((v) => `fold ${v}`).tickSizeOuter(0));
  g.append("g").attr("class", "eje").call(d3.axisLeft(y).ticks(5).tickFormat((v) => num(v, 2)).tickSizeOuter(0)).select(".domain").remove();
  g.append("line").attr("class", "referencia").attr("x1", 0).attr("x2", ancho).attr("y1", y(0.5)).attr("y2", y(0.5));
  for (const m of visibles()) {
    const v = estado.datos.modelos_datos[m].auc_por_fold;
    g.append("path").attr("d", d3.line().x((_, k) => x(k + 1)).y((a) => y(a))(v)).attr("fill", "none").attr("stroke", COLORES[m]).attr("stroke-width", 2);
    g.selectAll(null).data(v).join("circle").attr("class", "marca").attr("cx", (_, k) => x(k + 1)).attr("cy", (a) => y(a)).attr("r", 4.5).attr("fill", COLORES[m])
      .on("pointermove", (e, a) => mostrarTooltip(e, [[COLORES[m], NOMBRES_CORTOS[m], true], [null, `AUC ${num(a)}`]]))
      .on("pointerleave", ocultarTooltip);
  }
  leyenda(c, visibles().map((m) => [COLORES[m], `${NOMBRES_CORTOS[m]} (media ${num(estado.datos.modelos_datos[m].auc_val)})`]));
  tabla(document.querySelector('[data-tabla="folds"]'), ["Modelo", "Fold 1", "Fold 2", "Fold 3", "Fold 4", "Fold 5"],
    MODELOS.map((m) => [NOMBRES_CORTOS[m], ...estado.datos.modelos_datos[m].auc_por_fold.map((a) => num(a))]));
}

// ---------- Razones de odds ----------
function odds(c) {
  const o = estado.datos.razones_odds;
  const variables = o.combinado.map((v) => v.clave);
  const fila = (m, clave) => o[m].find((v) => v.clave === clave);
  const ms = ["medico", "combinado"].filter((m) => estado.visibles.has(m));
  const altoFila = 16 + Math.max(1, ms.length) * 10;
  const { g, ancho, alto } = lienzo(c, variables.length * altoFila + 64, { arriba: 10, derecha: 18, abajo: 44, izquierda: Math.min(230, c.clientWidth * 0.42) });
  const x = d3.scaleLog([0.15, 6], [0, ancho]);
  g.append("g").attr("class", "rejilla").selectAll("line").data([0.25, 0.5, 1, 2, 4]).join("line").attr("x1", (v) => x(v)).attr("x2", (v) => x(v)).attr("y1", 0).attr("y2", alto);
  g.append("line").attr("class", "referencia").attr("x1", x(1)).attr("x2", x(1)).attr("y1", 0).attr("y2", alto);
  const eje = g.append("g").attr("class", "eje").attr("transform", `translate(0,${alto})`).call(d3.axisBottom(x).tickValues([0.25, 0.5, 1, 2, 4]).tickFormat((v) => num(v, v < 1 ? 2 : 0)).tickSizeOuter(0));
  eje.append("text").attr("class", "eje__titulo").attr("x", ancho / 2).attr("y", 36).attr("text-anchor", "middle").text("Razón de odds de pCR (escala logarítmica)");
  variables.forEach((clave, i) => {
    const y0 = i * altoFila + 8;
    g.append("text").attr("class", "fila-titulo").attr("x", -12).attr("y", y0 + (ms.length * 10) / 2 + 3).attr("text-anchor", "end").text(fila("combinado", clave).variable);
    ms.forEach((m, k) => {
      const v = fila(m, clave);
      if (!v) return;
      const y = y0 + k * 10 + 4;
      g.append("line").attr("x1", x(Math.max(0.15, v.ic95[0]))).attr("x2", x(Math.min(6, v.ic95[1]))).attr("y1", y).attr("y2", y).attr("stroke", COLORES[m]).attr("stroke-width", 2).attr("stroke-linecap", "round");
      g.append("circle").attr("class", "marca").attr("cx", x(v.or)).attr("cy", y).attr("r", 4.5).attr("fill", COLORES[m])
        .on("pointermove", (e) => mostrarTooltip(e, [[COLORES[m], `${NOMBRES_CORTOS[m]}: ${v.variable}`, true], [null, `Razón de odds ${num(v.or, 2)} (IC ${num(v.ic95[0], 2)}–${num(v.ic95[1], 2)})`]]))
        .on("pointerleave", ocultarTooltip);
    });
  });
  leyenda(c, ms.map((m) => [COLORES[m], NOMBRES_CORTOS[m], "punto"]));
  tabla(document.querySelector('[data-tabla="odds"]'), ["Variable", "Médico", "CNN + médico"],
    variables.map((clave) => [fila("combinado", clave).variable, ...["medico", "combinado"].map((m) => {
      const v = fila(m, clave);
      return v ? `${num(v.or, 2)} (${num(v.ic95[0], 2)}–${num(v.ic95[1], 2)})` : "—";
    })]));
}

// ---------- El sesgo de la época ----------
function sesgo(c) {
  const filas = estado.datos.sesgo_epoca;
  if (!filas.length) { c.replaceChildren(nodo("p", "Sin datos del histórico.", "nota")); return; }
  const { g, ancho, alto } = lienzo(c, filas.length * 30 + 70, { arriba: 10, derecha: 24, abajo: 44, izquierda: 80 });
  const x = d3.scaleLinear([0.52, 0.61], [0, ancho]);
  const y = d3.scaleBand(filas.map((f) => f.version), [0, alto]).padding(0.4);
  g.append("g").attr("class", "rejilla").selectAll("line").data(x.ticks(5)).join("line").attr("x1", (v) => x(v)).attr("x2", (v) => x(v)).attr("y1", 0).attr("y2", alto);
  const eje = g.append("g").attr("class", "eje").attr("transform", `translate(0,${alto})`).call(d3.axisBottom(x).ticks(5).tickFormat((v) => num(v, 2)).tickSizeOuter(0));
  eje.append("text").attr("class", "eje__titulo").attr("x", ancho / 2).attr("y", 36).attr("text-anchor", "middle").text("AUC de validación (media de 5 folds)");
  for (const f of filas) {
    const cy = y(f.version) + y.bandwidth() / 2;
    g.append("text").attr("class", "fila-titulo").attr("x", -12).attr("y", cy + 4).attr("text-anchor", "end").text(f.version);
    g.append("line").attr("x1", x(f.fija)).attr("x2", x(f.elegida)).attr("y1", cy).attr("y2", cy).attr("stroke", EJE).attr("stroke-width", 2);
    g.append("circle").attr("cx", x(f.elegida)).attr("cy", cy).attr("r", 5).attr("fill", "#fff").attr("stroke", TINTA_2).attr("stroke-width", 2)
      .on("pointermove", (e) => mostrarTooltip(e, [[null, f.version, true], [null, `Con la época elegida: ${num(f.elegida)}`], [null, `Con 6 épocas fijas: ${num(f.fija)}`]]))
      .on("pointerleave", ocultarTooltip);
    g.append("circle").attr("class", "marca").attr("cx", x(f.fija)).attr("cy", cy).attr("r", 5).attr("fill", COLORES.cnn)
      .on("pointermove", (e) => mostrarTooltip(e, [[null, f.version, true], [null, `Con 6 épocas fijas: ${num(f.fija)}`], [null, `Bajó ${num(f.elegida - f.fija)}`]]))
      .on("pointerleave", ocultarTooltip);
  }
  leyenda(c, [[TINTA_2, "eligiendo la mejor época con la validación", "hueco"], [COLORES.cnn, "6 épocas fijadas de antemano", "punto"]]);
  tabla(document.querySelector('[data-tabla="sesgo"]'), ["Versión", "Época elegida", "6 fijas", "Diferencia"],
    filas.map((f) => [f.version, num(f.elegida), num(f.fija), num(f.elegida - f.fija)]));
}

// ---------- Entrenamiento (múltiplos pequeños) ----------
function entrenamiento(c) {
  const versiones = estado.datos.entrenamiento;
  if (!versiones.length) { c.replaceChildren(nodo("p", "Las curvas salen de runs/, que solo está en el PC donde se entrenó.", "nota")); return; }
  const columnas = c.clientWidth > 900 ? 3 : c.clientWidth > 560 ? 2 : 1;
  const contenedor = nodo("div", null, "multiples");
  contenedor.style.gridTemplateColumns = `repeat(${columnas}, minmax(0, 1fr))`;
  c.append(contenedor);
  for (const v of versiones) {
    const caja = nodo("div", null, "multiples__panel");
    caja.append(nodo("p", `${v.version}`, "multiples__titulo"));
    contenedor.append(caja);
    const { g, ancho, alto } = lienzo(caja, 200, { arriba: 8, derecha: 12, abajo: 30, izquierda: 40 });
    const x = d3.scaleLinear(d3.extent(v.epoca), [0, ancho]);
    const y = d3.scaleLinear([0.4, 0.8], [alto, 0]);
    g.append("g").attr("class", "rejilla").selectAll("line").data(y.ticks(4)).join("line").attr("x1", 0).attr("x2", ancho).attr("y1", (d) => y(d)).attr("y2", (d) => y(d));
    g.append("g").attr("class", "eje").attr("transform", `translate(0,${alto})`).call(d3.axisBottom(x).ticks(v.epoca.length).tickFormat(d3.format("d")).tickSizeOuter(0));
    g.append("g").attr("class", "eje").call(d3.axisLeft(y).ticks(4).tickFormat((d) => num(d, 1)).tickSizeOuter(0)).select(".domain").remove();
    g.append("line").attr("class", "referencia").attr("x1", 0).attr("x2", ancho).attr("y1", y(0.5)).attr("y2", y(0.5));
    for (const [serie, color] of [["auc_train", TINTA_2], ["auc_val", COLORES.cnn]]) {
      const s = v[serie];
      if (!s) continue;
      const banda = d3.area().x((_, k) => x(v.epoca[k])).y0((m, k) => y(Math.max(0.4, m - s.desviacion[k]))).y1((m, k) => y(Math.min(0.8, m + s.desviacion[k])));
      g.append("path").attr("d", banda(s.media)).attr("fill", color).attr("fill-opacity", 0.12);
      g.append("path").attr("d", d3.line().x((_, k) => x(v.epoca[k])).y((m) => y(m))(s.media)).attr("fill", "none").attr("stroke", color).attr("stroke-width", 2);
      g.selectAll(null).data(s.media).join("circle").attr("cx", (_, k) => x(v.epoca[k])).attr("cy", (m) => y(m)).attr("r", 3.5).attr("fill", color)
        .on("pointermove", (e, m) => {
          const k = s.media.indexOf(m);
          mostrarTooltip(e, [[color, `${v.version}, época ${v.epoca[k]}`, true], [null, `${serie === "auc_val" ? "Validación" : "Entrenamiento"}: ${num(m)} ± ${num(s.desviacion[k])}`]]);
        })
        .on("pointerleave", ocultarTooltip);
    }
  }
  leyenda(c, [[TINTA_2, "entrenamiento"], [COLORES.cnn, "validación (pacientes que no vio)"]]);
}

// ---------- Bibliografía ----------
function bibliografia(c) {
  const d = estado.datos;
  const filas = [
    { estudio: "Nuestra CNN, solo imagen", que: "validación cruzada en train, media de folds", auc: d.modelos_datos.cnn.auc_val, propio: "cnn" },
    { estudio: "Nuestro médico, sin imagen", que: "validación cruzada en train, media de folds", auc: d.modelos_datos.medico.auc_val, propio: "medico" },
    { estudio: "Nuestra CNN + médico", que: "validación cruzada en train, media de folds", auc: d.modelos_datos.combinado.auc_val, propio: "combinado" },
    ...d.bibliografia,
  ];
  const { g, ancho, alto, margen } = lienzo(c, filas.length * 34 + 64, { arriba: 10, derecha: 24, abajo: 44, izquierda: Math.min(330, c.clientWidth * 0.45) });
  const x = d3.scaleLinear([0.45, 0.95], [0, ancho]);
  const y = d3.scaleBand(filas.map((_, i) => i), [0, alto]).padding(0.35);
  g.append("g").attr("class", "rejilla").selectAll("line").data(x.ticks(6)).join("line").attr("x1", (v) => x(v)).attr("x2", (v) => x(v)).attr("y1", 0).attr("y2", alto);
  g.append("line").attr("class", "referencia").attr("x1", x(0.5)).attr("x2", x(0.5)).attr("y1", 0).attr("y2", alto);
  const eje = g.append("g").attr("class", "eje").attr("transform", `translate(0,${alto})`).call(d3.axisBottom(x).ticks(6).tickFormat((v) => num(v, 2)).tickSizeOuter(0));
  eje.append("text").attr("class", "eje__titulo").attr("x", ancho / 2).attr("y", 36).attr("text-anchor", "middle").text("AUC");
  filas.forEach((f, i) => {
    const cy = y(i) + y.bandwidth() / 2;
    if (i === 3) g.append("line").attr("class", "separador").attr("x1", -margen.izquierda + 8).attr("x2", ancho).attr("y1", y(i) - 6).attr("y2", y(i) - 6);
    g.append("text").attr("class", "fila-titulo").attr("x", -12).attr("y", cy).attr("text-anchor", "end").text(f.estudio);
    g.append("text").attr("class", "fila-detalle").attr("x", -12).attr("y", cy + 13).attr("text-anchor", "end").text(f.que);
    const color = f.propio ? COLORES[f.propio] : TINTA_2;
    if (f.auc_min != null) {
      g.append("line").attr("x1", x(f.auc_min)).attr("x2", x(f.auc)).attr("y1", cy).attr("y2", cy).attr("stroke", color).attr("stroke-width", 6).attr("stroke-linecap", "round").attr("stroke-opacity", 0.5);
    }
    g.append("circle").attr("cx", x(f.auc)).attr("cy", cy).attr("r", 5).attr("fill", f.propio ? color : "#fff").attr("stroke", color).attr("stroke-width", 2);
    g.append("text").attr("class", "etiqueta-serie").attr("x", x(f.auc) + 10).attr("y", cy + 4)
      .text(f.auc_min != null ? `${num(f.auc_min, 2)}–${num(f.auc, 2)}` : num(f.auc, f.propio ? 3 : 2));
  });
}

// ---------- Todo ----------
const GRAFICAS = { roc, calibracion, distribucion, decision, subgrupos, folds, odds, sesgo, entrenamiento, bibliografia };
function dibujarTodo() {
  for (const c of document.querySelectorAll("[data-grafica]")) {
    c.replaceChildren();
    const f = GRAFICAS[c.dataset.grafica];
    if (!f) continue;
    if (!estado.datos.modelos_datos && !["sesgo", "entrenamiento"].includes(c.dataset.grafica)) continue;
    try {
      f(c);
    } catch (error) {
      c.replaceChildren(nodo("p", `No se pudo dibujar: ${error.message}`, "nota"));
    }
  }
  if (estado.datos.modelos_datos) matrices();
}

async function arrancar() {
  try {
    estado.datos = await leerJSON("datos/resultados.json");
  } catch (error) {
    $("aviso-datos").hidden = false;
    $("aviso-datos").textContent = `No se pudieron leer los resultados: ${error.message}`;
    return;
  }
  if (!estado.datos.modelos_datos) {
    $("aviso-datos").hidden = false;
    $("aviso-datos").textContent = `Faltan las predicciones de la comparación: ${estado.datos.aviso || "vuelve a exportar la app"}.`;
  } else {
    cifras();
    filtros();
    estado.umbral = estado.datos.modelos_datos.cnn.umbral ?? 0.3;
    $("deslizador").value = String(estado.umbral);
    $("deslizador").addEventListener("input", (e) => { estado.umbral = Number(e.target.value); matrices(); });
    $("umbral-propio").addEventListener("click", () => matrices(true));
    notaUmbral();
  }
  dibujarTodo();
  let espera;
  window.addEventListener("resize", () => { clearTimeout(espera); espera = setTimeout(dibujarTodo, 200); });
}

arrancar();
