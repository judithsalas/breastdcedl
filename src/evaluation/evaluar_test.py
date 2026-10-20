"""Evaluación final en test (F5, D22): UNA sola vez, con el modelo ya cerrado.

    python -m src.evaluation.evaluar_test

Reglas del caso que hace cumplir este script:
- Usa el modelo final tal cual está en models/final/ (el mismo código que la app,
  src/models/final.py), con el umbral y la agregación decididos en el cierre. No ajusta nada.
- **Se bloquea después de la primera vez.** Si ya existe docs/resultados/4_test/resultado_test.json, se niega
  a repetir. Solo se puede repetir con `--repetir "motivo"`, y el motivo queda escrito en el
  informe: si se vuelve atrás a tocar el modelo después de mirar test, los resultados dejan de
  valer, y hay que poder contarlo.

Qué calcula (rúbrica, puntos 8 y 9):
- Matriz de confusión por paciente; AUC con intervalo de confianza del 95 % por bootstrap de
  pacientes; sensibilidad y especificidad con su intervalo; precision, recall y F1 de cada clase;
  F1 macro; accuracy.
- Comparación con decir siempre «no pCR» y siempre «pCR», y con las métricas internas (fuera de
  fold) para ver cuánto se pierde fuera de train.
- Resultados por cohorte, calibración (Brier y deciles) y los casos más claramente fallados,
  mirados con sus imágenes (figura con las tres fases y el realce).
- Comparación con el modelo clínico (D25, "el médico": edad, menopausia, HR, HER2) y con el
  combinado CNN + médico, en las mismas pacientes y en esta misma evaluación única. Por eso se
  niega a empezar si falta models/clinico/modelo_clinico.json o si se hizo con otra CNN
  (`--sin-clinico` evalúa solo la CNN, pero entonces la comparación ya no podría hacerse en test).

Salida (se versiona), todo en docs/resultados/4_test/: RESULTADO_TEST.md, resultado_test.json,
predicciones_test.csv (por corte), predicciones_test_pacientes.csv (por paciente, con los tres
modelos) y las figuras.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.data.caso import NOMBRE_COHORTE, RAIZ_REPO, RESULTADOS, uc
from src.evaluation import modelo_clinico as mc
from src.evaluation.metricas import (
    auc_con_ic,
    brier,
    coma,
    dibujar_matriz_confusion,
    diferencia_auc,
    ic_bootstrap,
    metricas_con_umbral,
    referencias_triviales,
    tabla_calibracion,
)
from src.models.final import CARPETA_FINAL, cargar_modelo_final

CARPETA_TEST = RESULTADOS / "4_test"
TINTA = "#0b0b0b"


def predecir_cortes(modelo, filas: pd.DataFrame, lote: int = 64, raiz: Path | None = None) -> np.ndarray:
    """Probabilidad calibrada de cada corte, en el orden de `filas` (shuffle=False)."""
    datos = uc.BreastDCEDataset(filas) if raiz is None else uc.BreastDCEDataset(filas, raiz)
    salida = []
    with torch.no_grad():
        for x, _ in DataLoader(datos, batch_size=lote, shuffle=False):
            salida.append(modelo.probabilidad(x).cpu().numpy())
    return np.concatenate(salida)


def evaluar(pacientes: pd.DataFrame, umbral: float) -> dict:
    """Todas las métricas por paciente (columnas pCR, prob y cohorte)."""
    y, p = pacientes.pCR.to_numpy(), pacientes.prob.to_numpy()
    m = metricas_con_umbral(y, p, umbral)

    def sens(yy, pp):
        return metricas_con_umbral(yy, pp, umbral)["sensibilidad"]

    def espe(yy, pp):
        return metricas_con_umbral(yy, pp, umbral)["especificidad"]

    n_pcr = int(y.sum())
    resultado = {
        "pacientes": len(y), "pacientes_pcr": n_pcr, "umbral": umbral, **m,
        **auc_con_ic(y, p),
        "sensibilidad_ic95": list(ic_bootstrap(y, p, sens)), "especificidad_ic95": list(ic_bootstrap(y, p, espe)),
        "brier": brier(y, p), "calibracion": tabla_calibracion(y, p),
        "referencias_triviales": referencias_triviales(n_pcr, len(y) - n_pcr),
        "por_cohorte": {},
    }
    if "cohorte" in pacientes:
        for c, g in pacientes.groupby("cohorte"):
            mc = metricas_con_umbral(g.pCR.to_numpy(), g.prob.to_numpy(), umbral)
            resultado["por_cohorte"][str(c)] = {"pacientes": len(g), "pcr": int(g.pCR.sum()),
                                                **auc_con_ic(g.pCR.to_numpy(), g.prob.to_numpy()),
                                                "sensibilidad": mc["sensibilidad"],
                                                "especificidad": mc["especificidad"]}
    return resultado


def casos_fallados(pacientes: pd.DataFrame, umbral: float, n: int = 3) -> pd.DataFrame:
    """Los falsos negativos con menor probabilidad y los falsos positivos con mayor probabilidad."""
    fn = pacientes[(pacientes.pCR == 1) & (pacientes.prob < umbral)].nsmallest(n, "prob").assign(error="FN")
    fp = pacientes[(pacientes.pCR == 0) & (pacientes.prob >= umbral)].nlargest(n, "prob").assign(error="FP")
    return pd.concat([fn, fp])


def figura_casos(casos: pd.DataFrame, cortes: pd.DataFrame, ruta: Path, raiz: Path | None = None) -> Path | None:
    if casos.empty:
        return None
    fig, ejes = plt.subplots(len(casos), 4, figsize=(9, 2.3 * len(casos)), squeeze=False)
    for i, (pid, caso) in enumerate(casos.iterrows()):
        suyos = cortes[cortes.patient_id == pid]
        fila = suyos.loc[(suyos.prob - caso.prob).abs().idxmin()]   # el corte más representativo
        x = uc.cargar_imagen(fila) if raiz is None else uc.cargar_imagen(fila, raiz)
        for j, titulo in enumerate(("PRE", "EARLY", "LATE")):
            ejes[i, j].imshow(x[j], cmap="gray", vmin=0, vmax=1)
            if i == 0:
                ejes[i, j].set_title(titulo, fontsize=10, color=TINTA)
        ejes[i, 3].imshow(x[1] - x[0], cmap="RdBu_r", vmin=-0.5, vmax=0.5)
        if i == 0:
            ejes[i, 3].set_title("EARLY − PRE", fontsize=10, color=TINTA)
        for ax in ejes[i]:
            ax.set_xticks([])
            ax.set_yticks([])
        cohorte = NOMBRE_COHORTE.get(caso.get("cohorte"), caso.get("cohorte", ""))
        ejes[i, 0].set_ylabel(f"{caso.error} · {cohorte}\np = {coma(caso.prob, 2)}", fontsize=9)
    fig.subplots_adjust(top=0.95)
    fig.suptitle("Casos más claramente fallados en test (FN: tenía pCR; FP: no la tenía)", fontsize=11,
                 x=0.01, ha="left", color=TINTA, y=0.99)
    fig.savefig(ruta, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return ruta


def figura_calibracion(resultado: dict, ruta: Path) -> Path:
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    t = resultado["calibracion"]
    ax.plot([g["prob_media"] for g in t], [g["proporcion_real"] for g in t], marker="o", color="#2a78d6", lw=1.6)
    ax.plot([0, 1], [0, 1], color="#52514e", lw=0.8, ls="--")
    ax.set_xlabel("Probabilidad media predicha", color="#52514e")
    ax.set_ylabel("Proporción real de pCR", color="#52514e")
    ax.set_title(f"Calibración en test (Brier {coma(resultado['brier'], 3)})", fontsize=10.5, loc="left",
                 fontweight="bold", color=TINTA)
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    ax.grid(True, color="#e4e3df")
    fig.savefig(ruta, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return ruta


def comprobar_clinico(manifiesto: dict, ruta: Path | None = None) -> dict:
    """El modelo clínico guardado (D25), comprobando que su combinado se hizo con ESTA CNN."""
    try:
        guardado = mc.cargar(mc.MODELO_JSON if ruta is None else ruta)
    except FileNotFoundError:
        raise SystemExit("Falta el modelo clínico de comparación (D25). Antes de test:\n"
                         "    python -m src.evaluation.modelo_clinico\n"
                         "(--sin-clinico evalúa solo la CNN, pero la comparación ya no podría hacerse en test).") from None
    if guardado.get("huella_cnn") != manifiesto["huella_modelo"]:
        raise SystemExit(f"El modelo clínico se hizo con la CNN {guardado.get('huella_cnn')} y el modelo final es "
                         f"{manifiesto['huella_modelo']}.\nVuelve a ejecutar python -m src.evaluation.modelo_clinico "
                         "(después de cerrar_modelo).")
    return guardado


def comparar_con_clinico(pacientes: pd.DataFrame, clinicos: pd.DataFrame, guardado: dict) -> tuple[dict, pd.DataFrame]:
    """CNN, médico y CNN + médico en las mismas pacientes de test, cada uno con su umbral de train."""
    pred = mc.predecir(clinicos, pacientes.prob, guardado)
    salida = {"huella_clinico": guardado["huella"], "modelos": {}, "diferencias": {}}
    for nombre, columna in (("medico", "prob_medico"), ("combinado", "prob_combinado")):
        if columna not in pred:
            continue
        r = evaluar(pacientes.assign(prob=pred[columna]), guardado[nombre].umbral)
        salida["modelos"][nombre] = {k: r[k] for k in ("umbral", "auc", "ic95", "sensibilidad", "especificidad",
                                                        "precision_pcr", "f1_macro", "brier", "matriz_confusion",
                                                        "por_cohorte")}
    y = pacientes.pCR.to_numpy()
    salida["diferencias"]["cnn_menos_medico"] = diferencia_auc(y, pacientes.prob, pred.prob_medico)
    if "prob_combinado" in pred:
        salida["diferencias"]["combinado_menos_medico"] = diferencia_auc(y, pred.prob_combinado, pred.prob_medico)
    return salida, pred


def figura_roc_comparada(pacientes: pd.DataFrame, pred: pd.DataFrame, ruta: Path) -> Path:
    y = pacientes.pCR.to_numpy()
    curvas = [("CNN (solo imagen)", y, pacientes.prob.to_numpy(), mc.NARANJA, "-"),
              ("Médico (sin imagen)", y, pred.prob_medico.to_numpy(), mc.AZUL, "-")]
    if "prob_combinado" in pred:
        curvas.append(("CNN + médico", y, pred.prob_combinado.to_numpy(), mc.VERDE, "-"))
    return mc.figura_roc(curvas, "ROC por paciente en test", ruta)


def seccion_clinica(r: dict) -> list[str]:
    c = r["comparacion_clinica"]
    lineas = ["## Comparación con el médico (modelo clínico sin imagen, D25)", "",
              ("Mismas pacientes de test. Cada modelo con su umbral elegido en train con el mismo criterio. "
               "El médico usa edad,"),
              ("menopausia, HR y HER2; no entra en la app. Detalle en [`MODELO_CLINICO.md`](../3_clinico/MODELO_CLINICO.md)."), "",
              "| Modelo | AUC (IC 95 %) | Sensibilidad | Especificidad | Precision pCR | F1 macro | Brier |",
              "|---|---|---|---|---|---|---|",
              (f"| CNN (solo imagen) | {coma(r['auc'], 3)} ({coma(r['ic95'][0], 2)}–{coma(r['ic95'][1], 2)}) | "
               f"{coma(r['sensibilidad'], 3)} | {coma(r['especificidad'], 3)} | {_v(r['precision_pcr'])} | "
               f"{coma(r['f1_macro'], 3)} | {coma(r['brier'], 3)} |")]
    for nombre, etiqueta in (("medico", "Médico (sin imagen)"), ("combinado", "CNN + médico")):
        if nombre in c["modelos"]:
            m = c["modelos"][nombre]
            lineas.append(f"| {etiqueta} | {coma(m['auc'], 3)} ({coma(m['ic95'][0], 2)}–{coma(m['ic95'][1], 2)}) | "
                          f"{coma(m['sensibilidad'], 3)} | {coma(m['especificidad'], 3)} | {_v(m['precision_pcr'])} | "
                          f"{coma(m['f1_macro'], 3)} | {coma(m['brier'], 3)} |")
    lineas += ["", "| Diferencia de AUC (bootstrap pareado de pacientes) | Valor (IC 95 %) |", "|---|---|"]
    for clave, etiqueta in (("cnn_menos_medico", "CNN − médico"),
                            ("combinado_menos_medico", "(CNN + médico) − médico: lo que añade la imagen")):
        if clave in c["diferencias"]:
            d = c["diferencias"][clave]
            lineas.append(f"| {etiqueta} | {coma(d['diferencia'], 3)} ({coma(d['ic95'][0], 3)} a "
                          f"{coma(d['ic95'][1], 3)}) |")
    lineas += ["", "Si el intervalo de una diferencia cruza el 0, con 176 pacientes no hay pruebas de que un modelo sea",
               "mejor que el otro.", "", "![ROC comparada](roc_test_comparada.png)", ""]
    return lineas


def _v(x):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else coma(x, 3)


def informe(resultado: dict, manifiesto: dict, motivo_repeticion: str | None) -> str:
    r, mi = resultado, manifiesto["metricas_internas"]
    tr = r["referencias_triviales"]
    lineas = [
        "# Resultado en test", "",
        f"Evaluado el {r['fecha']} con `python -m src.evaluation.evaluar_test`. **Evaluación única:** el modelo",
        (f"estaba cerrado (versión {manifiesto['version']}, huella `{manifiesto['huella_modelo']}`, umbral "
        f"{coma(r['umbral'], 3)}, media de cortes) y no se ha tocado después."), "",
    ]
    if motivo_repeticion:
        lineas += [f"> **Atención: esta evaluación se ha repetido.** Motivo declarado: {motivo_repeticion}", ""]
    lineas += [
        f"## Métricas por paciente ({r['pacientes']} pacientes, {r['pacientes_pcr']} con pCR)", "",
        "| Métrica | Test | Fuera de fold (train) | Siempre «no pCR» | Siempre «pCR» |", "|---|---|---|---|---|",
        (f"| AUC | {coma(r['auc'], 3)} (IC 95 % {coma(r['ic95'][0], 3)}–{coma(r['ic95'][1], 3)}) | "
        f"{coma(mi['fuera_de_fold']['auc']['auc'], 3)} | 0,500 | 0,500 |"),
        (f"| Sensibilidad (recall pCR) | {coma(r['sensibilidad'], 3)} (IC {coma(r['sensibilidad_ic95'][0], 2)}–"
        f"{coma(r['sensibilidad_ic95'][1], 2)}) | {coma(mi['fuera_de_fold']['sensibilidad'], 3)} | 0 | 1 |"),
        (f"| Especificidad (recall no pCR) | {coma(r['especificidad'], 3)} (IC {coma(r['especificidad_ic95'][0], 2)}–"
        f"{coma(r['especificidad_ic95'][1], 2)}) | {coma(mi['fuera_de_fold']['especificidad'], 3)} | 1 | 0 |"),
    ]
    for clave, nombre in (("precision_pcr", "Precision pCR"), ("precision_no_pcr", "Precision no pCR (VPN)"),
                          ("f1_pcr", "F1 pCR"), ("f1_no_pcr", "F1 no pCR"), ("f1_macro", "F1 macro"),
                          ("accuracy", "Accuracy")):
        interno = mi["fuera_de_fold"].get(clave)
        lineas.append(f"| {nombre} | {_v(r[clave])} | {_v(interno)} | {_v(tr['siempre_no_pcr'][clave])} | "
                      f"{_v(tr['siempre_pcr'][clave])} |")
    m = r["matriz_confusion"]
    lineas += ["", (f"Matriz de confusión: VP {m['VP']}, FN {m['FN']}, FP {m['FP']}, VN {m['VN']}. "
               f"Error de Brier: {coma(r['brier'], 4)}."), "",
               "![Matriz de confusión](matriz_confusion_test.png)", "",
               "## Por cohorte", "", "| Cohorte | Pacientes | Con pCR | AUC (IC 95 %) | Sensibilidad | Especificidad |",
               "|---|---|---|---|---|---|"]
    for c, v in r["por_cohorte"].items():
        lineas.append(f"| {NOMBRE_COHORTE.get(c, c)} | {v['pacientes']} | {v['pcr']} | {coma(v['auc'], 3)} ({coma(v['ic95'][0], 2)}–"
                      f"{coma(v['ic95'][1], 2)}) | {coma(v['sensibilidad'], 3)} | {coma(v['especificidad'], 3)} |")
    lineas += ["", "Con tan pocas pacientes por cohorte los intervalos son muy anchos: las diferencias entre",
               "cohortes se describen, no se afirman.", ""]
    if r.get("comparacion_clinica"):
        lineas += seccion_clinica(r)
    lineas += ["## Calibración y casos fallados", "", "![Calibración](calibracion_test.png)", "",
               "![Casos fallados](casos_fallados_test.png)", "",
               "## Cómo leerlo", "",
               "- Con 176 pacientes, el IC del AUC mide unos ±0,08: una diferencia de 0,02 con otro modelo no",
               "  significa nada (correo del profesor).",
               "- Referencias: lo publicado con imagen pretratamiento y test independiente está entre 0,70 y",
               "  0,80; con cortes 2D y tres cohortes, 0,60–0,70 es un buen trabajo; Fridman et al. obtienen",
               "  0,72 en este mismo test con un ViT preentrenado (método no permitido aquí).",
               "- La diferencia entre fuera de fold y test indica cuánto se pierde con pacientes nuevas."]
    return "\n".join(lineas) + "\n"


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluación final en test (una sola vez).")
    p.add_argument("--repetir", metavar="MOTIVO", help="repetir aunque ya se haya evaluado, dejando el motivo escrito")
    p.add_argument("--sin-clinico", action="store_true",
                   help="evaluar solo la CNN, sin la comparación con el modelo clínico (D25)")
    args = p.parse_args()

    resultado_json = CARPETA_TEST / "resultado_test.json"
    if resultado_json.exists() and not args.repetir:
        raise SystemExit("Test ya se evaluó (docs/resultados/4_test/resultado_test.json). La regla es una sola vez.\n"
                         "Si de verdad hay que repetir, usa --repetir \"motivo\": quedará escrito en el informe.")

    modelo, manifiesto = cargar_modelo_final(CARPETA_FINAL)
    clinico = None if args.sin_clinico else comprobar_clinico(manifiesto)   # antes de mirar test
    test = uc.conjunto_test(uc.cargar_samples()).reset_index(drop=True)
    print(f"Evaluando {test.patient_id.nunique()} pacientes de test ({len(test)} cortes) con el modelo "
          f"{manifiesto['version']} ({manifiesto['huella_modelo']})...")
    cortes = test[["sample_id", "patient_id", "pCR", "path_pre", "path_early", "path_late"]].copy()
    cortes["prob"] = predecir_cortes(modelo, test)

    from src.training.entrenar import cohortes
    g = cortes.groupby("patient_id")
    pacientes = pd.DataFrame({"pCR": g.pCR.first(), "prob": g.prob.mean(), "n_cortes": g.size()})
    pacientes["cohorte"] = pacientes.index.map(cohortes())

    resultado = {"fecha": datetime.now().astimezone().strftime("%Y-%m-%d %H:%M"),
                 "modelo": manifiesto["version"], "huella_modelo": manifiesto["huella_modelo"],
                 **evaluar(pacientes, manifiesto["umbral"]), "repeticion": args.repetir}

    CARPETA_TEST.mkdir(parents=True, exist_ok=True)
    tabla_pacientes = pacientes.copy()
    if clinico is not None:
        clinicos = uc.cargar_patients().set_index("pid")
        resultado["comparacion_clinica"], pred = comparar_con_clinico(pacientes, clinicos, clinico)
        tabla_pacientes = tabla_pacientes.join(pred)
        figura_roc_comparada(pacientes, pred, CARPETA_TEST / "roc_test_comparada.png")
    tabla_pacientes.to_csv(CARPETA_TEST / "predicciones_test_pacientes.csv", index_label="patient_id")
    casos = casos_fallados(pacientes, manifiesto["umbral"])
    resultado["casos_fallados"] = [{"paciente": pid, "error": c.error, "cohorte": c.get("cohorte"),
                                    "prob": float(c.prob)} for pid, c in casos.iterrows()]
    dibujar_matriz_confusion(resultado["matriz_confusion"], CARPETA_TEST / "matriz_confusion_test.png",
                             f"Test · {manifiesto['version']}",
                             f"{resultado['pacientes']} pacientes · umbral {coma(manifiesto['umbral'], 3)}")
    figura_calibracion(resultado, CARPETA_TEST / "calibracion_test.png")
    figura_casos(casos, cortes, CARPETA_TEST / "casos_fallados_test.png")
    cortes.drop(columns=["path_pre", "path_early", "path_late"]).to_csv(CARPETA_TEST / "predicciones_test.csv",
                                                                         index=False)
    resultado_json.write_text(json.dumps(resultado, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    (CARPETA_TEST / "RESULTADO_TEST.md").write_text(informe(resultado, manifiesto, args.repetir), encoding="utf-8")
    print(f"AUC test {resultado['auc']:.3f} (IC {resultado['ic95'][0]:.3f}–{resultado['ic95'][1]:.3f}) · "
          f"sensibilidad {resultado['sensibilidad']:.3f} · especificidad {resultado['especificidad']:.3f}")
    if resultado.get("comparacion_clinica"):
        for nombre, m in resultado["comparacion_clinica"]["modelos"].items():
            print(f"  {mc.NOMBRE_MODELO[nombre]}: AUC {m['auc']:.3f} (IC {m['ic95'][0]:.3f}–{m['ic95'][1]:.3f})")
    print(f"Informe en {(CARPETA_TEST / 'RESULTADO_TEST.md').relative_to(RAIZ_REPO)}")


if __name__ == "__main__":
    main()
