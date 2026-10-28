"""Entrenamiento de una arquitectura con evaluación por paciente.

    python -m src.training.entrenar --modo rapido   --modelo cnn_v1
    python -m src.training.entrenar --modo completo --modelo cnn_v1
    python -m src.training.entrenar --modo completo --modelo cnn_v1 --ponderada   # pos_weight = N0/N1

Para entrenar en los 5 folds de una vez: `python -m src.training.validacion_cruzada`
(mismas opciones).

Cada ejecución crea una carpeta en runs/ (no se versiona) con:
    resultados.json                  configuración, máquina, métricas por época, tiempos,
                                     memoria y la época elegida (`mejor_epoca`): matriz de
                                     confusión, métricas por clase y diferencia de AUC
                                     entre entrenamiento y validación
    historial.csv                    las métricas por época, en tabla (para las curvas)
    mejor.pt                         pesos de la época elegida
    predicciones_val.csv             probabilidad de cada corte de validación en la época elegida
    predicciones_val_pacientes.csv   lo mismo por paciente (media, mediana, máximo) con su cohorte
    graficas/                        curvas.png (pérdida y AUC por época, entrenamiento y
                                     validación) y matriz_confusion.png (época elegida); ver
                                     src/training/graficas.py
y añade o actualiza su fila en runs/historico.csv (ver src/training/historico.py) y las tablas
automáticas de docs/ARQUITECTURAS.md (ver src/models/historial.py).

Qué se mide tras cada época:
- Validación: se predice cada corte, se agrega por paciente (media de probabilidades,
  umbral 0,5) con uc.evaluar_por_paciente y se calculan AUC, sensibilidad,
  especificidad, matriz de confusión y precision/recall/F1 de cada clase.
- Entrenamiento: el AUC por paciente sobre una muestra fija y estratificada de
  pacientes de entrenamiento (cfg.pacientes_auc_train), en modo evaluación. Con él se
  ve el sobreajuste: la diferencia de AUC entre entrenamiento y validación.
El criterio para quedarse con una época y para parar es el AUC por paciente en
validación. El conjunto test no se toca aquí.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import replace
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.config import EVALUACION, MODOS, Config, obtener_config
from src.data.aumento import crear_aumento
from src.data.caso import RAIZ_DATOS, RAIZ_REPO, uc
from src.data.muestreo import MuestreoPorPaciente
from src.data.recorte import crear_dataset
from src.evaluation.metricas import metricas_por_clase
from src.models.construir import contar_parametros, crear_modelo, receta
from src.models.historial import actualizar_documento
from src.models.versiones import VERSIONES
from src.training import historico
from src.training.graficas import dibujar_graficas
from src.utils.dispositivo import describir_dispositivo, elegir_dispositivo
from src.utils.semilla import fijar_semilla

CARPETA_RUNS = RAIZ_REPO / "runs"
# De config.json (D31). Sirven para las métricas de cada época; el umbral y la agregación
# definitivos se eligen al cerrar el modelo, con validación (D21).
METODO_AGREGACION = EVALUACION["agregacion"]
UMBRAL = EVALUACION["umbral_entrenamiento"]


# --------------------------------------------------------------------------- #
# Datos
# --------------------------------------------------------------------------- #

def pacientes_con_imagenes(filas: pd.DataFrame) -> pd.DataFrame:
    """Filas de las pacientes cuyas imágenes están descargadas (todas sus fases)."""
    existe = filas.apply(
        lambda f: all((RAIZ_DATOS / f[c]).exists() for c in ("path_pre", "path_early", "path_late")),
        axis=1,
    )
    completas = existe.groupby(filas.patient_id).all()
    return filas[filas.patient_id.isin(completas[completas].index)]


def elegir_pacientes(filas: pd.DataFrame, n: int, semilla: int) -> pd.DataFrame:
    """Hasta n pacientes al azar (reproducible), con todos sus cortes."""
    ids = pd.Series(sorted(filas.patient_id.unique()))
    elegidas = ids.sample(n=min(n, len(ids)), random_state=semilla)
    return filas[filas.patient_id.isin(elegidas)]


def muestra_estratificada(filas: pd.DataFrame, n: int, semilla: int) -> pd.DataFrame:
    """Unas n pacientes al azar (reproducible) con la misma proporción de pCR; n = 0 → todas.

    Se usa para medir el AUC de entrenamiento sin recorrer todo el entrenamiento cada
    época. La muestra es fija durante todo el entrenamiento.
    """
    pacientes = filas.groupby("patient_id").pCR.first()
    if n <= 0 or n >= len(pacientes):
        return filas
    elegidas = pacientes.groupby(pacientes).sample(frac=n / len(pacientes), random_state=semilla).index
    return filas[filas.patient_id.isin(elegidas)]


def preparar_datos(cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Partición por paciente (columna fold) y, en modo rápido, un subconjunto."""
    entrena, valida = uc.particion(uc.cargar_samples(), fold_val=cfg.fold_val)

    if cfg.n_pacientes is not None:
        # Solo pacientes ya descargadas, para poder probar con una descarga parcial.
        entrena = elegir_pacientes(pacientes_con_imagenes(entrena), cfg.n_pacientes, cfg.semilla)
        valida = elegir_pacientes(pacientes_con_imagenes(valida), max(cfg.n_pacientes // 4, 2), cfg.semilla)
        if entrena.empty or valida.empty:
            raise SystemExit(
                "No hay suficientes pacientes descargadas para el modo rápido "
                f"(entrena: {entrena.patient_id.nunique()}, valida: {valida.patient_id.nunique()}).\n"
                "Descarga más: python breastdcedl/descargar_datos.py --destino breastdcedl --pacientes 100"
            )
    else:
        faltan = [r for r in pd.concat([entrena, valida]).path_pre if not (RAIZ_DATOS / r).exists()]
        if faltan:
            raise SystemExit(
                f"Faltan imágenes de {len(faltan)} cortes para el modo completo.\n"
                "Descarga todo: python breastdcedl/descargar_datos.py --destino breastdcedl"
            )

    return entrena.reset_index(drop=True), valida.reset_index(drop=True)


@lru_cache(maxsize=1)
def cohortes() -> dict:
    """Cohorte de cada paciente (spy1, spy2, duke), de patients.csv."""
    return uc.cargar_patients().set_index("pid").dataset.to_dict()


def tabla_predicciones(probabilidades: np.ndarray, filas: pd.DataFrame) -> pd.DataFrame:
    """Una fila por corte: identificadores, etiqueta y probabilidad predicha."""
    columnas = [c for c in ("sample_id", "patient_id", "fold", "pCR") if c in filas.columns]
    return filas[columnas].reset_index(drop=True).assign(prob=np.asarray(probabilidades, dtype=float))


def tabla_por_paciente(predicciones: pd.DataFrame) -> pd.DataFrame:
    """Una fila por paciente, con las agregaciones que se compararán en F4.

    `prob_media` es exactamente la que usa uc.evaluar_por_paciente con metodo="mean".
    El voto no se guarda porque depende del umbral: se calcula desde los cortes.
    """
    g = predicciones.groupby("patient_id")
    tabla = pd.DataFrame({
        "pCR": g.pCR.first(),
        "n_cortes": g.size(),
        "prob_media": g.prob.mean(),
        "prob_mediana": g.prob.median(),
        "prob_max": g.prob.max(),
    })
    tabla.insert(0, "cohorte", tabla.index.map(cohortes()))
    if "fold" in predicciones.columns:
        tabla.insert(0, "fold", g.fold.first())
    return tabla.reset_index()


# --------------------------------------------------------------------------- #
# Entrenamiento y evaluación
# --------------------------------------------------------------------------- #

def predecir(modelo: nn.Module, cargador: DataLoader, dispositivo: torch.device,
             criterio: nn.Module | None = None) -> tuple[np.ndarray, float]:
    """Probabilidad de pCR de cada corte, en el orden del cargador (shuffle=False)."""
    modelo.eval()
    probabilidades, perdida_total, n = [], 0.0, 0
    with torch.no_grad():
        for x, y in cargador:
            x, y = x.to(dispositivo), y.to(dispositivo)
            logits = modelo(x).squeeze(1)
            if criterio is not None:
                perdida_total += criterio(logits, y).item() * len(y)
                n += len(y)
            probabilidades.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(probabilidades), (perdida_total / n if n else float("nan"))


def evaluar(probabilidades: np.ndarray, filas: pd.DataFrame) -> dict:
    """Métricas por paciente: las de uc.evaluar_por_paciente más las de cada clase."""
    m = uc.evaluar_por_paciente(probabilidades, filas, umbral=UMBRAL, metodo=METODO_AGREGACION)
    m.update(metricas_por_clase(m["matriz_confusion"]))
    return m


def crear_optimizador(modelo: nn.Module, lr: float, receta_version: dict) -> torch.optim.Optimizer:
    """El optimizador que pide la receta de la versión.

    - "adam": Adam sin weight decay (cnn_v1 a cnn_v4).
    - "adamw": AdamW (Loshchilov y Hutter, 2019), que aplica el weight decay aparte del paso de
      Adam. El weight decay empuja los pesos hacia 0 en cada paso: la red no puede apoyarse en
      pesos muy grandes para memorizar detalles de unas pocas pacientes. Solo se aplica a los
      pesos de convoluciones y capas lineales; no a los sesgos ni a BatchNorm, que no
      memorizan y a los que encoger solo estorba (práctica habitual).
    """
    if receta_version.get("optimizador", "adam") == "adam":
        return torch.optim.Adam(modelo.parameters(), lr=lr)
    if receta_version["optimizador"] != "adamw":
        raise ValueError(f"Optimizador desconocido: {receta_version['optimizador']!r}")
    con_decay = [p for p in modelo.parameters() if p.requires_grad and p.ndim > 1]
    sin_decay = [p for p in modelo.parameters() if p.requires_grad and p.ndim <= 1]
    return torch.optim.AdamW([{"params": con_decay, "weight_decay": receta_version["weight_decay"]},
                              {"params": sin_decay, "weight_decay": 0.0}], lr=lr)


PROGRAMADORES = (None, "meseta")


def crear_programador(optimizador: torch.optim.Optimizer, nombre: str | None):
    """El programador del learning rate que pide la receta (D34).

    - None: el learning rate no cambia en todo el entrenamiento (cnn_v1 a cnn_v15).
    - "meseta": ReduceLROnPlateau sobre el AUC por paciente en validación: si en 3 épocas no
      mejora, el learning rate se reduce a la mitad. La parada temprana espera 10 épocas, así que
      da tiempo a probar 2–3 pasos más pequeños antes de parar.
    """
    if nombre is None:
        return None
    if nombre == "meseta":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizador, mode="max", factor=0.5, patience=3)
    raise ValueError(f"Programador desconocido: {nombre!r}. Opciones: {PROGRAMADORES}")


def entrenar_una_epoca(modelo, cargador, criterio, optimizador, dispositivo) -> float:
    modelo.train()
    perdida_total, n = 0.0, 0
    total, inicio = len(cargador), time.perf_counter()
    for i, (x, y) in enumerate(cargador, 1):
        x, y = x.to(dispositivo), y.to(dispositivo)
        optimizador.zero_grad()
        perdida = criterio(modelo(x).squeeze(1), y)
        perdida.backward()
        optimizador.step()
        perdida_total += perdida.item() * len(y)
        n += len(y)
        if i % max(1, total // 20) == 0 or i == total:   # progreso dentro de la época
            restante = (time.perf_counter() - inicio) / i * (total - i)
            print(f"\r  entrenando lote {i}/{total} ({i / total:.0%}), quedan ~{restante:.0f} s ",
                  end="", flush=True)
    print("\r" + " " * 60 + "\r", end="", flush=True)
    return perdida_total / n


def entrenar_modelo(modelo, cargador_tr, cargador_va, valida, criterio, optimizador,
                    dispositivo, cfg: Config, carpeta: Path, resultados: dict,
                    eval_train: tuple[DataLoader, pd.DataFrame] | None = None, programador=None) -> dict:
    """Bucle de épocas con parada temprana por AUC por paciente en validación.

    `eval_train` = (cargador con shuffle=False, filas) de la muestra de entrenamiento
    sobre la que se mide el AUC de entrenamiento. Si es None, no se mide.
    """
    mejor_auc, sin_mejora = -math.inf, 0
    ultima = cfg.eleccion_epoca == "ultima"    # D37: se guarda la última época, sin mirar la validación
    resultados["epocas"] = []
    inicio = time.perf_counter()

    for epoca in range(1, cfg.epocas_max + 1):
        t0 = time.perf_counter()
        perdida_tr = entrenar_una_epoca(modelo, cargador_tr, criterio, optimizador, dispositivo)
        probs, perdida_va = predecir(modelo, cargador_va, dispositivo, criterio)
        m = evaluar(probs, valida)
        auc_tr = float("nan")
        if eval_train is not None:
            probs_tr, _ = predecir(modelo, eval_train[0], dispositivo)
            auc_tr = evaluar(probs_tr, eval_train[1])["auc"]
        segundos = time.perf_counter() - t0

        auc = m["auc"]
        lr_epoca = optimizador.param_groups[0]["lr"]          # el de esta época, antes de que el programador lo baje
        if programador is not None and not math.isnan(auc):
            programador.step(auc)
        mejora = ultima or (not math.isnan(auc) and auc > mejor_auc)
        if mejora:
            mejor_auc, sin_mejora = auc, 0
            torch.save({"modelo": resultados["modelo"], "arquitectura": resultados.get("arquitectura"),
                        "estado": modelo.state_dict(), "epoca": epoca, "auc_val": auc, "config": cfg.como_dict()},
                       carpeta / "mejor.pt")
            guardar_predicciones(tabla_predicciones(probs, valida), carpeta)
            resultados["mejor_epoca"] = {
                "epoca": epoca, "auc_val": auc, "auc_train": auc_tr, "diferencia_auc": auc_tr - auc,
                "perdida_tr": perdida_tr, "perdida_va": perdida_va,
                "agregacion": METODO_AGREGACION, "umbral": UMBRAL, "pacientes": m["pacientes"],
                "matriz_confusion": m["matriz_confusion"],
                "sensibilidad": m["sensibilidad"], "especificidad": m["especificidad"],
                **metricas_por_clase(m["matriz_confusion"]),
            }
        else:
            sin_mejora += 1

        fila = {"epoca": epoca, "perdida_tr": perdida_tr, "perdida_va": perdida_va,
                "auc_tr": auc_tr, "auc": auc, "diferencia_auc": auc_tr - auc,
                "sensibilidad": m["sensibilidad"], "especificidad": m["especificidad"],
                "accuracy": m["accuracy"], "f1_pcr": m["f1_pcr"], "f1_no_pcr": m["f1_no_pcr"],
                "f1_macro": m["f1_macro"], **m["matriz_confusion"], "segundos": segundos, "mejor": mejora,
                "lr": lr_epoca}
        resultados["epocas"].append(fila)
        print(f"época {epoca:>3}/{cfg.epocas_max} | pérdida tr {perdida_tr:.4f} va {perdida_va:.4f} | "
              f"AUC tr {auc_tr:.3f} va {auc:.3f} | sens {m['sensibilidad']:.2f} "
              f"espec {m['especificidad']:.2f} | {segundos:.0f} s{'  *' if mejora else ''}")

        resultados["mejor_auc_val"] = None if mejor_auc == -math.inf else mejor_auc
        resultados["tiempo_total_s"] = time.perf_counter() - inicio
        if dispositivo.type == "cuda":
            resultados["memoria_max_gb"] = round(torch.cuda.max_memory_allocated() / 1024**3, 2)
        guardar(resultados, carpeta)
        dibujar_graficas(carpeta)   # graficas/: curvas hasta esta época y matriz de la elegida

        if not ultima and sin_mejora >= cfg.paciencia:
            print(f"Parada: {cfg.paciencia} épocas sin mejorar el AUC por paciente.")
            break

    return resultados


def guardar_predicciones(predicciones: pd.DataFrame, carpeta: Path) -> None:
    predicciones.to_csv(carpeta / "predicciones_val.csv", index=False)
    tabla_por_paciente(predicciones).to_csv(carpeta / "predicciones_val_pacientes.csv", index=False)


def guardar(resultados: dict, carpeta: Path) -> None:
    (carpeta / "resultados.json").write_text(json.dumps(resultados, indent=2, ensure_ascii=False),
                                              encoding="utf-8")
    if resultados.get("epocas"):
        with open(carpeta / "historial.csv", "w", newline="", encoding="utf-8") as f:
            escritor = csv.DictWriter(f, fieldnames=list(resultados["epocas"][0]))
            escritor.writeheader()
            escritor.writerows(resultados["epocas"])


# --------------------------------------------------------------------------- #
# Una ejecución completa (la usan main() y la validación cruzada)
# --------------------------------------------------------------------------- #

def ejecutar(cfg: Config, nombre_modelo: str, ponderada: bool = False, nota: str = "",
             carpeta: Path | None = None, registrar: bool = True) -> dict:
    """Entrena `nombre_modelo` con el fold de validación `cfg.fold_val` y lo guarda todo en `carpeta`
    (por defecto, runs/<fecha>_<modelo>_<modo>[_ponderada]).

    Con `registrar` se actualiza runs/historico.csv al terminar. La validación cruzada lo
    desactiva en cada fold y registra una sola fila con la media de los folds.
    """
    fijar_semilla(cfg.semilla)
    dispositivo = elegir_dispositivo()
    entrena, valida = preparar_datos(cfg)
    muestra_tr = muestra_estratificada(entrena, cfg.pacientes_auc_train, cfg.semilla).reset_index(drop=True)

    modelo = crear_modelo(nombre_modelo).to(dispositivo)
    como_entrenar = receta(nombre_modelo)
    if como_entrenar["learning_rate"] is not None:       # la versión fija su propio learning rate (D34)
        if cfg.learning_rate != MODOS[cfg.modo].learning_rate:
            raise SystemExit(f"{nombre_modelo} fija su learning rate ({como_entrenar['learning_rate']}); "
                             "no se puede cambiar desde la configuración.")
        cfg = replace(cfg, learning_rate=como_entrenar["learning_rate"])
    aumento = crear_aumento(como_entrenar["aumento"])   # solo para el cargador de entrenamiento
    peso_positivos = uc.pos_weight(entrena) if ponderada else None
    criterio = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(peso_positivos, device=dispositivo) if peso_positivos else None)
    optimizador = crear_optimizador(modelo, cfg.learning_rate, como_entrenar)

    usa_gpu = dispositivo.type == "cuda"
    opciones = {"batch_size": cfg.batch_size, "num_workers": cfg.num_workers, "pin_memory": usa_gpu,
                "persistent_workers": cfg.num_workers > 0}
    recorte = como_entrenar["recorte"]                  # None = el corte entero; "caja" = alrededor del tumor (D33)
    if como_entrenar["cortes_por_paciente"]:             # muestreo equilibrado por paciente (D34)
        muestreo = MuestreoPorPaciente(entrena, como_entrenar["cortes_por_paciente"], cfg.semilla)
        cargador_tr = DataLoader(crear_dataset(entrena, recorte, transform=aumento), sampler=muestreo, **opciones)
    else:
        cargador_tr = DataLoader(crear_dataset(entrena, recorte, transform=aumento), shuffle=True,
                                 generator=torch.Generator().manual_seed(cfg.semilla), **opciones)
    programador = crear_programador(optimizador, como_entrenar["programador"])
    cargador_va = DataLoader(crear_dataset(valida, recorte), shuffle=False, **opciones)
    cargador_tr_eval = DataLoader(crear_dataset(muestra_tr, recorte), shuffle=False, **opciones)

    if carpeta is None:
        sufijo = "_ponderada" if ponderada else ""
        carpeta = CARPETA_RUNS / f"{datetime.now().astimezone():%Y%m%d-%H%M%S}_{nombre_modelo}_{cfg.modo}{sufijo}"
    carpeta.mkdir(parents=True, exist_ok=True)

    resultados = {
        "ejecucion": carpeta.relative_to(CARPETA_RUNS).as_posix() if carpeta.is_relative_to(CARPETA_RUNS)
        else carpeta.name,
        "modelo": nombre_modelo,
        "arquitectura": VERSIONES[nombre_modelo],
        "receta": {**como_entrenar, "detalle_aumento": repr(aumento) if aumento else None},
        "parametros": contar_parametros(modelo),
        "perdida": "ponderada" if ponderada else "normal",
        "pos_weight": peso_positivos,
        "agregacion": METODO_AGREGACION,
        "umbral": UMBRAL,
        "nota": nota,
        "config": cfg.como_dict(),
        "maquina": describir_dispositivo(dispositivo),
        "datos": {
            "pacientes_entrena": int(entrena.patient_id.nunique()), "cortes_entrena": len(entrena),
            "pacientes_valida": int(valida.patient_id.nunique()), "cortes_valida": len(valida),
            "pcr_pacientes_valida": int(valida.groupby("patient_id").pCR.first().sum()),
            "pacientes_auc_train": int(muestra_tr.patient_id.nunique()),
        },
    }

    print(f"Ejecución: {resultados['ejecucion']}")
    print(f"Modelo {nombre_modelo}: {resultados['parametros']:,} parámetros | dispositivo: "
          f"{resultados['maquina']['backend']} {resultados['maquina']['gpu'] or ''}")
    print(f"Entrena: {resultados['datos']['pacientes_entrena']} pacientes / {len(entrena)} cortes | "
          f"Valida (fold {cfg.fold_val}): {resultados['datos']['pacientes_valida']} pacientes / "
          f"{len(valida)} cortes | AUC de entrenamiento sobre {resultados['datos']['pacientes_auc_train']} pacientes")
    optim_texto = (f"AdamW, weight decay {como_entrenar['weight_decay']:g}"
                   if como_entrenar["optimizador"] == "adamw" else "Adam")
    print(f"Pérdida {resultados['perdida']} | aumento {como_entrenar['aumento'] or 'no'} | "
          f"recorte {como_entrenar['recorte'] or 'no'} | "
          f"cortes por paciente {como_entrenar['cortes_por_paciente'] or 'todos'} | "
          f"programador {como_entrenar['programador'] or 'no'} | {optim_texto} | "
          f"batch {cfg.batch_size} | lr {cfg.learning_rate} | "
          + (f"{cfg.epocas_max} épocas fijas, se usa la última (D37)\n" if cfg.eleccion_epoca == "ultima"
             else f"hasta {cfg.epocas_max} épocas, paciencia {cfg.paciencia}\n"))

    entrenar_modelo(modelo, cargador_tr, cargador_va, valida, criterio, optimizador,
                    dispositivo, cfg, carpeta, resultados, eval_train=(cargador_tr_eval, muestra_tr),
                    programador=programador)

    e = resultados.get("mejor_epoca")
    if e:
        mc = e["matriz_confusion"]
        cual = "usada (la última, fija)" if cfg.eleccion_epoca == "ultima" else "elegida"
        print(f"\nÉpoca {cual} {e['epoca']}: AUC va {e['auc_val']:.3f} | AUC tr {e['auc_train']:.3f} | "
              f"diferencia {e['diferencia_auc']:+.3f}")
        print(f"Umbral {UMBRAL}: VP {mc['VP']} FN {mc['FN']} FP {mc['FP']} VN {mc['VN']} | "
              f"F1 pCR {e['f1_pcr']:.2f} · F1 no pCR {e['f1_no_pcr']:.2f} · F1 macro {e['f1_macro']:.2f}")
    print(f"Resultados en {carpeta.relative_to(RAIZ_REPO) if carpeta.is_relative_to(RAIZ_REPO) else carpeta}"
          " (gráficas en graficas/)")
    if registrar:
        registrar_ejecucion()
    return resultados


def registrar_ejecucion() -> None:
    """Actualiza runs/historico.csv y las tablas automáticas de docs/ARQUITECTURAS.md."""
    print(f"Histórico actualizado: {historico.actualizar(CARPETA_RUNS).relative_to(RAIZ_REPO)}")
    try:
        if actualizar_documento():
            print("Tablas de docs/ARQUITECTURAS.md actualizadas: completa la ficha de la versión.")
    except ValueError as error:      # el documento no tiene las marcas: no se toca
        print(f"Aviso: no se actualizó docs/ARQUITECTURAS.md ({error})")


# --------------------------------------------------------------------------- #
# Programa principal
# --------------------------------------------------------------------------- #

def anadir_argumentos(p: argparse.ArgumentParser, con_fold: bool = True) -> argparse.ArgumentParser:
    """Argumentos comunes a entrenar.py y validacion_cruzada.py."""
    p.add_argument("--modo", choices=list(MODOS), required=True)
    p.add_argument("--modelo", required=True, help="nombre registrado en src/models (p. ej. cnn_v1)")
    p.add_argument("--ponderada", action="store_true", help="BCEWithLogitsLoss con pos_weight = N0/N1")
    p.add_argument("--nota", default="", help="comentario libre que se guarda con los resultados")
    # Sobrescribir valores del modo (si no se indican, se usan los de src/config.py)
    p.add_argument("--batch", dest="batch_size", type=int)
    p.add_argument("--epocas", dest="epocas_max", type=int)
    p.add_argument("--paciencia", type=int)
    p.add_argument("--lr", dest="learning_rate", type=float)
    if con_fold:
        p.add_argument("--fold", dest="fold_val", type=int)
    p.add_argument("--pacientes", dest="n_pacientes", type=int)
    p.add_argument("--pacientes-auc-train", dest="pacientes_auc_train", type=int,
                   help="pacientes de entrenamiento para el AUC de entrenamiento (0 = todas)")
    p.add_argument("--workers", dest="num_workers", type=int)
    p.add_argument("--semilla", type=int)
    return p


def cambios_de_config(args: argparse.Namespace) -> dict:
    """Los argumentos que sobrescriben valores de la configuración."""
    return {k: v for k, v in vars(args).items() if k not in {"modo", "modelo", "ponderada", "nota", "folds"}}


def main() -> None:
    args = anadir_argumentos(argparse.ArgumentParser(
        description="Entrena una CNN con evaluación por paciente.")).parse_args()
    cfg = obtener_config(args.modo, **cambios_de_config(args))
    ejecutar(cfg, args.modelo, args.ponderada, args.nota)


if __name__ == "__main__":
    main()
