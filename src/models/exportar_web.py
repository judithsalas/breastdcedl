"""Prepara la app web (D38): el modelo final en ONNX, sus datos y los ejemplos, en `web/`.

    python -m src.models.exportar_web

La app de Hugging Face es una página estática: el modelo se ejecuta en el navegador de quien la usa,
con ONNX Runtime Web. Este script hace todo lo que la página necesita, a partir del modelo cerrado:

1. **`web/modelo/modelo.onnx`**: el modelo final COMPLETO en un solo grafo. Entra el corte tal cual
   sale de los PNG (3 × 256 × 256 enteros de 0 a 255, PRE, EARLY y LATE) y sale la probabilidad de
   pCR. Dentro van, en este orden y con el mismo código de PyTorch:
   - la normalización del entrenamiento: dividir entre 255 (`src/data/entrada.py`, `utils_caso`);
   - las 30 CNN y sus medias de logits (`ModeloFinal`, D36);
   - la calibración de Platt y la sigmoide.
   Así la página no normaliza ni promedia nada por su cuenta: solo lee los píxeles del PNG.
2. **Comprobación**: cada corte de ejemplo se predice con PyTorch (como la app de Python y
   `src.models.predecir`) y con el ONNX. Si difieren en más de 1e-5, no se exporta nada.
3. **`web/modelo/info.json`**: umbral, huella, entrenamiento y métricas internas, que la página
   enseña (enunciado §9). Si test ya se evaluó con este modelo, también su resultado (D44): se copia de
   `docs/resultados/4_test/`, aquí no se evalúa nada.
4. **`web/ejemplos/`**: los cortes de `app/ejemplos/` (de train), con la probabilidad del pipeline,
   para que la página enseñe ✅ si coincide.

Se ejecuta en el PC de la uni después de `cerrar_modelo`, `modelo_clinico` y `app.preparar_ejemplos`.
Después, la carpeta `web/` se sube tal cual al Space de Hugging Face (`web/README.md`).
"""

from __future__ import annotations

import hashlib
import json
import shutil
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from src.data.caso import RAIZ_DATOS, RAIZ_REPO, RESULTADOS
from src.data.entrada import FASES, validar_corte
from src.models.cnn import CNN, PoolingAtencion
from src.models.final import CARPETA_FINAL, cargar_modelo_final, miembros
from src.models.versiones import NOMBRES

WEB = RAIZ_REPO / "web"
EJEMPLOS_APP = RAIZ_REPO / "app" / "ejemplos"
CLINICO = RAIZ_REPO / "experimentos" / "modelo_clinico" / "modelo_clinico.json"
TOLERANCIA = 1e-5
LADO = 256


def admite_mapa(red: nn.Module) -> bool:
    """¿Se puede repartir el logit de esta red entre las posiciones del último mapa, de forma exacta?

    Sí cuando, tras el extractor, hay una media (global o ponderada por atención) y una sola capa lineal:
    entonces logit = Σ_posiciones peso_posición · (w · mapa_posición) + b (dropout no hace nada al evaluar).
    """
    return (isinstance(red, CNN) and isinstance(red.cabeza[-1], nn.Linear)
            and not any(isinstance(m, nn.Linear) for m in list(red.cabeza)[:-1])
            and isinstance(red.resumen, (PoolingAtencion, nn.Sequential)))


def logit_y_mapa(red: CNN, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Logit de una red (B,) y cuánto aporta cada posición del último mapa (B, h, w).

    El logit se calcula con las mismas capas que `red(x)`. El mapa suma, junto con el sesgo de la última
    capa, exactamente el logit: es una descomposición, no una aproximación (como CAM, Zhou et al. 2016).
    """
    mapas = red.extractor(red.preparar_entrada(x))
    logit = red.cabeza(red.resumen(mapas)).squeeze(1)
    if not admite_mapa(red):
        return logit, torch.zeros(mapas.shape[0], mapas.shape[2], mapas.shape[3])
    b, _, alto, ancho = mapas.shape
    por_posicion = torch.einsum("bchw,c->bhw", mapas, red.cabeza[-1].weight[0])
    if isinstance(red.resumen, PoolingAtencion):
        h = mapas.flatten(2).transpose(1, 2)
        pesos = torch.softmax(red.resumen.w(torch.tanh(red.resumen.V(h))), dim=1).view(b, alto, ancho)
    else:
        pesos = torch.full_like(por_posicion, 1.0 / (alto * ancho))
    return logit, por_posicion * pesos


def a_rejilla(mapa: torch.Tensor, factor: int) -> torch.Tensor:
    """(B, h, h) → (B, h·factor, h·factor) repitiendo cada celda y dividiendo su valor: la suma no cambia."""
    if factor == 1:
        return mapa
    return mapa.repeat_interleave(factor, dim=1).repeat_interleave(factor, dim=2) / (factor * factor)


class ModeloWeb(nn.Module):
    """El modelo final tal como lo usa la página.

    Entra: píxeles enteros (B, 3, 256, 256). Salen:
    - `prob` (B,): la probabilidad de pCR, con las mismas operaciones y en el mismo orden que ModeloFinal;
    - `mapa` (B, 16, 16): cuánto empuja cada zona del corte el logit calibrado (positivo: hacia pCR).
      Es la media de los mapas de las 30 redes, igual que el logit es la media de sus logits.
    """

    def __init__(self, modelo: nn.Module):
        super().__init__()
        self.modelo = modelo.eval()
        # Todas las redes del conjunto acaban hoy en 16×16; si alguna acabara más fina, se lleva a la rejilla
        # más fina repartiendo cada valor entre sus celdas, para que el mapa siga sumando el logit.
        with torch.no_grad():
            prueba = torch.zeros(1, 3, LADO, LADO)
            lados = [int(logit_y_mapa(r, prueba)[1].shape[-1]) for r in self.modelo.redes]
        self.factores = [max(lados) // n for n in lados]

    def forward(self, pixeles: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = pixeles.to(torch.float32) / 255.0          # la normalización del entrenamiento, dentro del grafo
        logits, mapas = [], []
        for grupo in self.modelo.grupos:
            partes = [logit_y_mapa(self.modelo.redes[i], x) for i in grupo]
            partes = [(lg, a_rejilla(m, self.factores[i])) for i, (lg, m) in zip(grupo, partes)]
            logits.append(torch.stack([p[0] for p in partes]).mean(dim=0))
            mapas.append(torch.stack([p[1] for p in partes]).mean(dim=0))
        a, b = self.modelo.platt[0], self.modelo.platt[1]
        prob = torch.sigmoid(a * torch.stack(logits).mean(dim=0) + b)
        return prob, a * torch.stack(mapas).mean(dim=0)


class RedPorDentro(nn.Module):
    """Una sola red del conjunto, abierta: lo que sale de cada bloque, para dibujarlo en la página «La red»."""

    def __init__(self, red: CNN):
        super().__init__()
        self.red = red.eval()

    def forward(self, pixeles: torch.Tensor):
        x = self.red.preparar_entrada(pixeles.to(torch.float32) / 255.0)
        bloques = []
        for capa in self.red.extractor:
            x = capa(x)
            if isinstance(capa, (nn.MaxPool2d, nn.AvgPool2d)):
                bloques.append(x)
        logit = self.red.cabeza(self.red.resumen(x)).squeeze(1)
        _, mapa = logit_y_mapa(self.red, pixeles.to(torch.float32) / 255.0)
        return (*bloques, mapa, logit)


def _exportar(modulo: nn.Module, ruta: Path, salidas: list[str]) -> Path:
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ejemplo = torch.zeros(1, 3, LADO, LADO, dtype=torch.uint8)
    # El exportador clásico (TorchScript) basta para estas redes, que no tienen control de flujo; avisa de que
    # en versiones futuras de PyTorch el recomendado será el nuevo, pero con torch 2.14 funciona y es estable.
    with torch.no_grad(), warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        torch.onnx.export(modulo.eval(), (ejemplo,), str(ruta), input_names=["fases"], output_names=salidas,
                          opset_version=17, dynamo=False)
    return ruta


def exportar_onnx(modelo: nn.Module, ruta: Path) -> Path:
    """Guarda el modelo web en ONNX (opset 17), en modo evaluación y sin gradientes."""
    return _exportar(ModeloWeb(modelo), ruta, ["prob", "mapa"])


def exportar_una_red(red: CNN, ruta: Path) -> Path:
    """La primera red del conjunto, con lo que sale de cada bloque (para la página «La red»)."""
    n = sum(isinstance(m, (nn.MaxPool2d, nn.AvgPool2d)) for m in red.extractor)
    return _exportar(RedPorDentro(red), ruta, [f"bloque{i + 1}" for i in range(n)] + ["mapa", "logit"])


def a_pixeles(x: np.ndarray) -> np.ndarray:
    """(3, 256, 256) en [0, 1] (como lo deja validar_corte) → enteros de 0 a 255, los del PNG."""
    return np.rint(np.asarray(x, dtype=np.float64) * 255).astype(np.uint8)


def probabilidad_onnx(sesion, pixeles: np.ndarray) -> float:
    return float(sesion.run(None, {"fases": pixeles[None].astype(np.uint8)})[0][0])


def sesion_onnx(ruta: Path):
    import onnxruntime as ort

    return ort.InferenceSession(str(ruta), providers=["CPUExecutionProvider"])


def sha256(ruta: Path) -> str:
    return hashlib.sha256(Path(ruta).read_bytes()).hexdigest()


def leer_ejemplos(carpeta: Path = EJEMPLOS_APP) -> list[dict]:
    """Los cortes de ejemplo (carpeta por corte, con sus 3 PNG) y lo que dio el pipeline al crearlos."""
    esperado = {}
    if (carpeta / "esperado.json").exists():
        esperado = json.loads((carpeta / "esperado.json").read_text(encoding="utf-8"))
    salida = []
    for sub in sorted(p for p in Path(carpeta).iterdir() if p.is_dir()) if Path(carpeta).exists() else []:
        rutas = [sub / f"{sub.name}_{f}.png" for f in FASES]
        if all(r.exists() for r in rutas):
            salida.append({"corte": sub.name, "rutas": rutas, **esperado.get(sub.name, {})})
    return salida


def etiquetas_de_train() -> dict[str, int]:
    """pCR de cada corte de train (para enseñar la etiqueta real de los ejemplos, que son de train)."""
    ruta = RAIZ_DATOS / "metadata" / "samples.csv"
    if not ruta.exists():
        return {}
    samples = pd.read_csv(ruta, usecols=["sample_id", "split", "pCR"])
    train = samples[samples.split == "train"]
    return dict(zip(train.sample_id, train.pCR.astype(int)))


def sesgo_medio(modelo: nn.Module) -> float:
    """El sesgo de la última capa, promediado como los logits (media por versión y luego entre versiones)."""
    with torch.no_grad():
        por_version = [torch.stack([modelo.redes[i].cabeza[-1].bias[0] for i in g]).mean() for g in modelo.grupos]
        return float(torch.stack(por_version).mean())


def comprobar_mapa(modelo: nn.Module, sesion, pixeles: np.ndarray) -> tuple[float, float]:
    """(diferencia entre el mapa de ONNX y el de PyTorch, error de la descomposición: Σ mapa + constante − logit)."""
    with torch.no_grad():
        prob, mapa = ModeloWeb(modelo)(torch.from_numpy(pixeles).unsqueeze(0))
    _, mapa_onnx = sesion.run(None, {"fases": pixeles[None].astype(np.uint8)})
    a, b = (float(v) for v in modelo.platt)
    p = float(prob[0])
    reconstruido = float(mapa[0].sum()) + a * sesgo_medio(modelo) + b
    return float(np.abs(mapa_onnx[0] - mapa[0].numpy()).max()), abs(reconstruido - float(np.log(p / (1 - p))))


def comprobar(modelo: nn.Module, sesion, ejemplos: list[dict], semilla: int = 0) -> dict:
    """PyTorch frente a ONNX en los ejemplos y en cortes aleatorios. Falla si difieren más de TOLERANCIA.

    También el mapa de la página: que el de ONNX sea el de PyTorch y que sume el logit (descomposición exacta).
    """
    diferencias, filas, mapas = [], [], []
    for e in ejemplos:
        x, errores, _ = validar_corte([(r.name, r.read_bytes()) for r in e["rutas"]])
        if errores:
            raise SystemExit(f"El ejemplo {e['corte']} no pasa la validación: {errores}")
        with torch.no_grad():
            p_torch = float(modelo.probabilidad(torch.from_numpy(x).unsqueeze(0))[0])
        p_onnx = probabilidad_onnx(sesion, a_pixeles(x))
        previa = e.get("probabilidad_pipeline")
        if previa is not None and abs(previa - p_torch) > TOLERANCIA:
            raise SystemExit(f"El ejemplo {e['corte']} se hizo con otro modelo (pipeline {previa:.6f}, ahora "
                             f"{p_torch:.6f}). Rehaz los ejemplos: python -m app.preparar_ejemplos")
        diferencias.append(abs(p_torch - p_onnx))
        mapas.append(comprobar_mapa(modelo, sesion, a_pixeles(x)))
        filas.append({"corte": e["corte"], "torch": p_torch, "onnx": p_onnx})
    rng = np.random.default_rng(semilla)
    for _ in range(3):
        pixeles = rng.integers(0, 256, (3, LADO, LADO), dtype=np.uint8)
        with torch.no_grad():
            p_torch = float(modelo.probabilidad(torch.from_numpy(pixeles.astype(np.float32) / 255.0).unsqueeze(0))[0])
        diferencias.append(abs(p_torch - probabilidad_onnx(sesion, pixeles)))
        mapas.append(comprobar_mapa(modelo, sesion, pixeles))
    maxima = max(diferencias)
    if maxima > TOLERANCIA:
        raise SystemExit(f"El ONNX no reproduce a PyTorch: diferencia máxima {maxima:.2e} (> {TOLERANCIA:.0e}).")
    mapa_max, suma_max = max(m[0] for m in mapas), max(m[1] for m in mapas)
    if mapa_max > TOLERANCIA or suma_max > 1e-3:
        raise SystemExit(f"El mapa no cuadra: ONNX frente a PyTorch {mapa_max:.1e}, suma frente al logit {suma_max:.1e}.")
    return {"diferencia_maxima": maxima, "cortes": len(diferencias), "ejemplos": filas,
            "mapa_diferencia_maxima": mapa_max, "mapa_suma_error": suma_max}


def referencia_oof(manifiesto: dict, resultados: Path = RESULTADOS) -> dict:
    """Tasa de pCR y la franja donde cae el 90 % de las pacientes de train (fuera de fold), si es de este modelo."""
    cierre = resultados / "2_cierre" / "cierre.json"
    oof = resultados / "2_cierre" / "oof_pacientes.csv"
    if not (cierre.exists() and oof.exists()):
        return {}
    if json.loads(cierre.read_text(encoding="utf-8")).get("huella_modelo") != manifiesto["huella_modelo"]:
        return {}
    tabla = pd.read_csv(oof)
    return {"tasa_pcr": float(tabla.pCR.mean()), "pacientes": len(tabla),
            "franja_90": [float(tabla.prob.quantile(0.05)), float(tabla.prob.quantile(0.95))]}


def resumen_validacion_interna(manifiesto: dict, resultados: Path = RESULTADOS) -> dict | None:
    """Brier y umbral con Platt y umbral elegidos sin el fold que se mide (D40), si ya se calcularon."""
    from src.evaluation.validacion_interna import leer

    v = leer(manifiesto, Path(resultados) / "2_cierre" / "validacion_interna.json")
    if v is None:
        return None
    c = v["umbral"]["cruzado"]
    return {"brier_constante": v["brier"]["constante"], "brier_platt_cruzado": v["brier"]["platt_cruzado"],
            "sensibilidad": c["sensibilidad"], "especificidad": c["especificidad"], "precision_pcr": c["precision_pcr"]}


def referencia_medico(manifiesto: dict, ruta: Path = CLINICO) -> dict:
    if not ruta.exists():
        return {}
    clinico = json.loads(ruta.read_text(encoding="utf-8"))
    if clinico.get("huella_cnn") != manifiesto["huella_modelo"]:
        return {}
    v = clinico.get("validacion", {})
    return {"medico_auc": v.get("medico", {}).get("auc_val"), "cnn_mas_medico_auc": v.get("combinado", {}).get("auc_val")}


def anatomia(red: CNN) -> dict:
    """Capa a capa, lo que hace la red con un corte: formas, parámetros y campo receptivo (para «La red»)."""
    canales = red.preparar_entrada(torch.zeros(1, 3, LADO, LADO)).shape[1]
    lado, salto, campo = LADO, 1, 1
    bloques, actual = [], None
    for capa in red.extractor:
        if actual is None:
            actual = {"entrada": [canales, lado, lado], "convoluciones": [], "parametros": 0}
        if isinstance(capa, nn.Conv2d):
            k = capa.kernel_size[0]
            campo += (k - 1) * salto
            n = sum(p.numel() for p in capa.parameters())
            actual["convoluciones"].append({"kernel": k, "filtros": capa.out_channels, "entra": capa.in_channels,
                                            "pesos": n, "sesgo": capa.bias is not None})
            actual["parametros"] += n
            canales = capa.out_channels
        elif isinstance(capa, nn.BatchNorm2d):
            n = sum(p.numel() for p in capa.parameters())
            actual["batchnorm"] = n
            actual["parametros"] += n
        elif isinstance(capa, (nn.MaxPool2d, nn.AvgPool2d)):
            campo += salto
            salto *= 2
            lado //= 2
            actual.update(pooling="máximo 2×2" if isinstance(capa, nn.MaxPool2d) else "media 2×2",
                          salida=[canales, lado, lado], campo_receptivo=campo, salto=salto)
            bloques.append(actual)
            actual = None
    resumen = ("atención" if isinstance(red.resumen, PoolingAtencion)
               else "media global" if isinstance(red.resumen, nn.Sequential) else type(red.resumen).__name__)
    cabeza = [{"capa": type(c).__name__, "parametros": sum(p.numel() for p in c.parameters()),
               **({"p": c.p} if isinstance(c, nn.Dropout) else {})} for c in red.cabeza]
    return {"entrada": red.entrada, "normalizacion": red.normalizacion, "bloques": bloques,
            "resumen": {"tipo": resumen, "parametros": sum(p.numel() for p in red.resumen.parameters()),
                        "vector": canales},
            "cabeza": cabeza, "parametros": sum(p.numel() for p in red.parameters())}


def info_web(manifiesto: dict, onnx: Path, comprobacion: dict, resultados: Path = RESULTADOS,
             clinico: Path = CLINICO) -> dict:
    """Lo que la página enseña del modelo (enunciado §9): umbral, huella, entrenamiento y métricas internas.

    Y, si ya se hizo con este modelo, el resultado de la evaluación única de test (R13, D44).
    """
    from src.models.datos_web import resumen_test

    mi, ent = manifiesto["metricas_internas"], manifiesto["entrenamiento"]
    config = ent.get("config", {})
    versiones = []
    for m in miembros(manifiesto):
        receta = m.get("receta") or {}
        arquitectura = m.get("arquitectura") or {}
        versiones.append({"version": m["version"], "que_cambia": NOMBRES.get(m["version"], ""),
                          "parametros": m.get("parametros"), "aumento": receta.get("aumento"),
                          "detalle_aumento": receta.get("detalle_aumento"), "weight_decay": receta.get("weight_decay"),
                          "arquitectura": arquitectura, "red": anatomia(CNN(**arquitectura)) if arquitectura else None,
                          "epocas": [f.get("epoca") for f in m["pesos"]],
                          "auc_val_por_fold": [f.get("auc_val_fold") for f in m["pesos"]],
                          "auc_val": m.get("auc_val_media"), "auc_val_desviacion": m.get("auc_val_desviacion"),
                          "auc_train": m.get("auc_train_media"), "validacion_cruzada": m.get("validacion_cruzada")})
    oof = mi.get("fuera_de_fold", {})
    return {
        "huella_modelo": manifiesto["huella_modelo"],
        "fecha_cierre": manifiesto.get("fecha"),
        "modelo": manifiesto.get("version"),
        "conjunto": manifiesto.get("conjunto"),
        "entrada": manifiesto.get("entrada"),
        "umbral": manifiesto["umbral"],
        "criterio_umbral": manifiesto.get("criterio_umbral"),
        "calibracion": {k: manifiesto["calibracion"][k] for k in ("a", "b")},
        "onnx": {"fichero": onnx.name, "sha256": sha256(onnx), "bytes": onnx.stat().st_size, "opset": 17,
                 "torch": torch.__version__,
                 "comprobacion": {k: comprobacion[k] for k in ("diferencia_maxima", "cortes", "mapa_diferencia_maxima",
                                                               "mapa_suma_error") if k in comprobacion}},
        "entrenamiento": {
            "redes": ent.get("redes"), "parametros": ent.get("parametros"), "perdida": ent.get("perdida"),
            "batch": config.get("batch_size"), "learning_rate": config.get("learning_rate"),
            "semilla": config.get("semilla"), "epocas": config.get("epocas_max"),
            "eleccion_epoca": config.get("eleccion_epoca"), "optimizador": "Adam",
            "gpu": (ent.get("maquina") or {}).get("gpu"), "versiones": versiones,
        },
        "metricas": {
            "auc_val": mi.get("auc_val_media"), "auc_val_desviacion": mi.get("auc_val_desviacion"),
            "auc_val_por_fold": mi.get("auc_val_por_fold"), "auc_train": mi.get("auc_train_media"),
            "auc_oof": (oof.get("auc") or {}).get("auc"), "auc_oof_ic95": (oof.get("auc") or {}).get("ic95"),
            "sensibilidad": oof.get("sensibilidad"), "especificidad": oof.get("especificidad"),
            "precision_pcr": oof.get("precision_pcr"), "f1_macro": oof.get("f1_macro"),
            "brier": mi.get("brier_calibrado"),
        },
        "referencia": {**referencia_oof(manifiesto, resultados), **referencia_medico(manifiesto, clinico)},
        "validacion_interna": resumen_validacion_interna(manifiesto, resultados),
        "test": resumen_test(manifiesto, resultados),
    }


def copiar_ejemplos(ejemplos: list[dict], comprobacion: dict, destino: Path, origen: Path = EJEMPLOS_APP) -> list[dict]:
    """Copia los PNG de cada ejemplo y escribe ejemplos.json (la página no puede listar carpetas)."""
    if destino.exists():
        shutil.rmtree(destino)
    destino.mkdir(parents=True)
    etiquetas = etiquetas_de_train()
    por_corte = {f["corte"]: f for f in comprobacion["ejemplos"]}
    lista = []
    for e in ejemplos:
        (destino / e["corte"]).mkdir()
        for r in e["rutas"]:
            shutil.copy2(r, destino / e["corte"] / r.name)
        lista.append({"corte": e["corte"], "cohorte": e.get("cohorte"), "pcr": etiquetas.get(e["corte"]),
                      "ficheros": [r.name for r in e["rutas"]],
                      "probabilidad_pipeline": por_corte[e["corte"]]["torch"]})
    (destino / "ejemplos.json").write_text(json.dumps(lista, indent=2, ensure_ascii=False), encoding="utf-8")
    if (origen / "LEEME.md").exists():
        shutil.copy2(origen / "LEEME.md", destino / "LEEME.md")
    return lista


def exportar(carpeta_final: Path = CARPETA_FINAL, web: Path = WEB, ejemplos_app: Path = EJEMPLOS_APP,
             resultados: Path = RESULTADOS, clinico: Path = CLINICO,
             runs: Path = RAIZ_REPO / "experimentos") -> dict:
    """Todo lo que necesita la página. Devuelve info.json."""
    modelo, manifiesto = cargar_modelo_final(carpeta_final)
    modelo.eval()
    ruta = exportar_onnx(modelo, Path(web) / "modelo" / "modelo.onnx")
    ejemplos = leer_ejemplos(ejemplos_app)
    comprobacion = comprobar(modelo, sesion_onnx(ruta), ejemplos)
    info = info_web(manifiesto, ruta, comprobacion, resultados, clinico)
    info["una_red"] = exportar_una_red_comprobada(modelo, manifiesto, Path(web) / "modelo" / "una_red.onnx", ejemplos)
    (Path(web) / "modelo" / "info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    copiar_ejemplos(ejemplos, comprobacion, Path(web) / "ejemplos", ejemplos_app)
    from src.models.datos_web import escribir_resultados

    escribir_resultados(manifiesto, Path(web) / "datos" / "resultados.json", resultados, runs)
    return info


def exportar_una_red_comprobada(modelo: nn.Module, manifiesto: dict, ruta: Path, ejemplos: list[dict]) -> dict:
    """Exporta la primera red del conjunto (fold 0 de la primera versión) y comprueba su logit con PyTorch."""
    red = modelo.redes[modelo.grupos[0][0]]
    exportar_una_red(red, ruta)
    sesion = sesion_onnx(ruta)
    maxima = 0.0
    for e in ejemplos[:3]:
        x, _, _ = validar_corte([(r.name, r.read_bytes()) for r in e["rutas"]])
        with torch.no_grad():
            esperado = float(red(torch.from_numpy(x).unsqueeze(0))[0, 0])
        maxima = max(maxima, abs(float(sesion.run(["logit"], {"fases": a_pixeles(x)[None]})[0][0]) - esperado))
    if maxima > 1e-4:
        raise SystemExit(f"una_red.onnx no reproduce a su red: diferencia {maxima:.1e}")
    primero = miembros(manifiesto)[0]
    return {"fichero": ruta.name, "sha256": sha256(ruta), "bytes": ruta.stat().st_size, "version": primero["version"],
            "fold": 0, "diferencia_logit": maxima, "salidas": [o.name for o in sesion.get_outputs()]}


def main() -> None:
    info = exportar()
    o = info["onnx"]
    print(f"Modelo {info['modelo']} (huella {info['huella_modelo']}) → web/modelo/{o['fichero']} "
          f"({o['bytes'] / 1e6:.1f} MB, sha256 {o['sha256'][:12]}…)")
    print(f"PyTorch frente a ONNX en {o['comprobacion']['cortes']} cortes: diferencia máxima "
          f"{o['comprobacion']['diferencia_maxima']:.1e} (límite {TOLERANCIA:.0e})")
    c = o["comprobacion"]
    if "mapa_diferencia_maxima" in c:
        print(f"Mapa «Qué empuja»: diferencia con PyTorch {c['mapa_diferencia_maxima']:.1e}, error de la suma "
              f"{c['mapa_suma_error']:.1e}")
    u = info["una_red"]
    print(f"Red de «La red»: {u['version']} fold {u['fold']}, diferencia de logit con PyTorch {u['diferencia_logit']:.1e}")
    datos = json.loads((WEB / "datos" / "resultados.json").read_text(encoding="utf-8"))
    if datos.get("aviso"):
        print(f"AVISO en «Resultados»: {datos['aviso']}")
    else:
        print("«Resultados»: " + " · ".join(f"{k} {v['auc_val']:.3f}" for k, v in datos["modelos_datos"].items()))
    if info.get("validacion_interna") is None:
        print("AVISO: falta la validación interna (python -m src.evaluation.validacion_interna); la web sale sin ella.")
    test = info.get("test")
    if test is None:
        print("Test: aún no evaluado con este modelo; la web sale sin él.")
    else:
        print(f"Test en la web ({test['pacientes']} pacientes, {test['fecha']}): "
              + " · ".join(f"{k} {v['auc']:.3f}" for k, v in test["modelos"].items()))
    n = len(json.loads((WEB / "ejemplos" / "ejemplos.json").read_text(encoding="utf-8")))
    print(f"{n} ejemplos en web/ejemplos/. Siguiente paso: subir web/ al Space (web/README.md).")


if __name__ == "__main__":
    main()
