"""Historial de arquitecturas: rellena las partes automáticas de docs/ARQUITECTURAS.md.

    python -m src.models.historial                            # regenera tablas y genealogía
    python -m src.models.historial --figuras runs/<carpeta>   # copia sus gráficas a docs/figuras/

Lo escrito a mano (hipótesis, lectura y decisión de cada ficha) no se toca. Solo se reescribe lo
que hay entre las marcas `<!-- auto:NOMBRE:inicio -->` y `<!-- auto:NOMBRE:fin -->`:

    auto:resumen      una fila por versión: origen, qué cambia, parámetros, estado y el AUC de su
                      última ejecución completa de cada tipo (1 fold / 5 folds, normal / ponderada)
    auto:genealogia   árbol de versiones: de cuál sale cada una
    auto:ejecuciones  todas las ejecuciones en modo completo, con su huella

Los números salen de runs/historico.csv (D13); el modo rápido no cuenta. Se ejecuta solo al
terminar cada entrenamiento. A mano, tras un `git pull` que traiga filas nuevas del histórico.

Se muestra la ÚLTIMA ejecución de cada tipo, no la mejor, para no escoger a posteriori. Todas
están en la tabla de ejecuciones.
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

import pandas as pd

from src.config import COMPARACION, es_estandar
from src.data.caso import RAIZ_REPO
from src.evaluation.metricas import coma
from src.models.construir import cambios, contar_parametros, crear_modelo, huella
from src.models.versiones import ESTADO, ORIGEN, VERSIONES

DOCUMENTO = RAIZ_REPO / "docs" / "ARQUITECTURAS.md"
HISTORICO = RAIZ_REPO / "runs" / "historico.csv"
FIGURAS = RAIZ_REPO / "docs" / "figuras" / "ejecuciones"


def leer_historico(ruta: Path = HISTORICO, modo: str = "completo") -> pd.DataFrame:
    """Ejecuciones de un modo del histórico (por defecto, las completas; vacío si aún no hay)."""
    if not Path(ruta).exists():
        return pd.DataFrame()
    tabla = pd.read_csv(ruta, dtype={"huella": str, "folds": str})
    return tabla[tabla.modo == modo].sort_values("ejecucion")


TODOS_LOS_FOLDS = "0,1,2,3,4"
# Diferencia de AUC train − val a partir de la cual se considera que sobreajusta (config.json, D16).
SOBREAJUSTE = COMPARACION["sobreajuste_maximo"]


def validaciones_validas(historico: pd.DataFrame, perdida: str = "normal") -> pd.DataFrame:
    """Última validación cruzada de 5 folds de cada versión, con huella ✓, la pérdida pedida y el
    batch, el learning rate y la semilla estándar de su modo (config.json, D31)."""
    if historico.empty:
        return historico
    filas = historico[(historico.tipo == "cv") & (historico.perdida == perdida)
                      & (historico.folds.astype(str) == TODOS_LOS_FOLDS)
                      & historico.modelo.isin(list(VERSIONES))]
    filas = filas[filas.apply(lambda f: _huella_ok(f) is True, axis=1)] if len(filas) else filas
    filas = filas[filas.apply(lambda f: es_estandar(f.modo, f.get("batch"), f.get("lr"), f.get("semilla")),
                              axis=1)] if len(filas) else filas
    return filas.groupby("modelo").tail(1)


def elegir_mejor(historico: pd.DataFrame) -> dict | None:
    """Aplica el criterio de comparación de docs/ARQUITECTURAS.md (D16).

    1. Validaciones cruzadas completas de 5 folds, pérdida normal, huella ✓.
    2. La de mayor AUC medio en validación es la referencia.
    3. Empatan con ella las que quedan a menos de una desviación (la de la referencia).
    4. Entre las empatadas, primero las que no sobreajustan (diferencia train − val ≤ 0,15);
       después, la de menos parámetros; a igualdad, la más antigua.
    """
    candidatas = validaciones_validas(historico, "normal")
    if candidatas.empty:
        return None
    referencia = candidatas.loc[candidatas.auc_val.idxmax()]
    margen = 0.0 if pd.isna(referencia.auc_val_desv) else float(referencia.auc_val_desv)
    empatadas = candidatas[candidatas.auc_val >= referencia.auc_val - margen]
    orden = list(VERSIONES)
    def sobreajusta(modelo: str) -> bool:
        dif = empatadas.loc[empatadas.modelo == modelo, "diferencia_auc"].iloc[0]
        return bool(not pd.isna(dif) and dif > SOBREAJUSTE)

    ganadora = min(empatadas.modelo,
                   key=lambda m: (sobreajusta(m), contar_parametros(crear_modelo(m)), orden.index(m)))
    fila = candidatas[candidatas.modelo == ganadora].iloc[0]
    if ganadora == referencia.modelo:
        motivo = f"mayor AUC medio ({coma(referencia.auc_val, 3)} ± {coma(margen, 3)}, {referencia.modelo})"
    else:
        if sobreajusta(referencia.modelo) and not sobreajusta(ganadora):
            desempate = (f"{referencia.modelo} sobreajusta (diferencia {coma(referencia.diferencia_auc, 3)} > "
                         f"{coma(SOBREAJUSTE, 2)}) y {ganadora} no ({coma(fila.diferencia_auc, 3)})")
        elif contar_parametros(crear_modelo(ganadora)) < contar_parametros(crear_modelo(referencia.modelo)):
            desempate = f"{ganadora} tiene menos parámetros"
        else:
            desempate = f"{ganadora} es más antigua (cambia menos cosas)"
        motivo = (f"empata con {referencia.modelo} ({coma(referencia.auc_val, 3)} ± {coma(margen, 3)}); "
                  f"{desempate}")
    return {"version": ganadora, "auc_val": float(fila.auc_val), "auc_val_desv": fila.auc_val_desv,
            "empatadas": sorted(empatadas.modelo, key=orden.index), "referencia": referencia.modelo,
            "motivo": motivo, "ejecucion": fila.ejecucion}


def _num(valor, decimales: int = 3) -> str:
    return "—" if pd.isna(valor) else coma(float(valor), decimales)


def _auc(fila: pd.Series) -> str:
    texto = _num(fila.auc_val)
    if fila.tipo == "cv" and not pd.isna(fila.get("auc_val_desv")):
        texto += f" ± {_num(fila.auc_val_desv)}"
    return texto + ("" if _huella_ok(fila) is not False else " ⚠")


def _huella_ok(fila: pd.Series) -> bool | None:
    """True si la ejecución usó la definición actual, False si no, None si no se sabe."""
    if fila.modelo not in VERSIONES or pd.isna(fila.get("huella")):
        return None
    return fila.huella == huella(fila.modelo)


def _ultima(historico: pd.DataFrame, modelo: str, tipo: str, perdida: str) -> pd.Series | None:
    if historico.empty:
        return None
    filas = historico[(historico.modelo == modelo) & (historico.tipo == tipo) & (historico.perdida == perdida)]
    return None if filas.empty else filas.iloc[-1]


def tabla_resumen(historico: pd.DataFrame) -> str:
    """Una fila por versión, en el orden en que se definieron."""
    lineas = [
        ("| Versión | Origen | Qué cambia | Parámetros | Estado | 1 fold · normal | 1 fold · ponderada "
         "| 5 folds · normal | 5 folds · ponderada | Train − val |"),
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for nombre in VERSIONES:
        celdas = []
        for tipo in ("simple", "cv"):
            for perdida in ("normal", "ponderada"):
                fila = _ultima(historico, nombre, tipo, perdida)
                celdas.append("—" if fila is None else _auc(fila))
        # Diferencia train − val de la ejecución más informativa: 5 folds si la hay.
        referencia = next((f for f in (_ultima(historico, nombre, t, p) for t in ("cv", "simple")
                                       for p in ("normal", "ponderada")) if f is not None), None)
        diferencia = "—" if referencia is None else _num(referencia.diferencia_auc)
        origen = ORIGEN.get(nombre)
        parametros = f"{contar_parametros(crear_modelo(nombre)):,}".replace(",", ".")
        lineas.append(
            f"| [{nombre}](#{nombre}) | {origen or '—'} | {'; '.join(cambios(nombre)) or 'primera versión'} "
            f"| {parametros} | {ESTADO.get(nombre, '¿?')} | " + " | ".join(celdas) + f" | {diferencia} |"
        )
    mejor = elegir_mejor(historico)
    lineas.append("")
    if mejor is None:
        lineas.append("**Según el criterio de comparación:** todavía no hay validaciones cruzadas completas "
                      "con pérdida normal para comparar.")
    else:
        lineas.append(f"**Según el criterio de comparación:** la mejor es **{mejor['version']}**: "
                      f"{mejor['motivo']}. Empatadas: {', '.join(mejor['empatadas'])}.")
    return "\n".join(lineas)


def genealogia() -> str:
    """Árbol de versiones a partir de ORIGEN."""
    hijos: dict[str | None, list[str]] = {}
    for nombre in VERSIONES:
        hijos.setdefault(ORIGEN.get(nombre), []).append(nombre)

    lineas: list[str] = []

    def rama(nombre: str, prefijo: str, ultimo: bool, raiz: bool) -> None:
        conector = "" if raiz else ("└── " if ultimo else "├── ")
        lineas.append(f"{prefijo}{conector}{nombre} · {ESTADO.get(nombre, '¿?')}")
        siguiente = prefijo if raiz else prefijo + ("    " if ultimo else "│   ")
        for i, hijo in enumerate(hijos.get(nombre, [])):
            rama(hijo, siguiente, i == len(hijos[nombre]) - 1, False)

    for raiz in hijos.get(None, []):
        rama(raiz, "", True, True)
    return "```\n" + "\n".join(lineas) + "\n```"


def tabla_ejecuciones(historico: pd.DataFrame) -> str:
    """Todas las ejecuciones en modo completo, de la más antigua a la más reciente."""
    if historico.empty:
        return "_Todavía no hay ejecuciones en modo completo en `runs/historico.csv`._"
    lineas = [
        ("| Fecha | Ejecución | Versión | Pérdida | Folds | Época elegida | AUC val | AUC train "
         "| Train − val | F1 macro | Huella |"),
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for _, f in historico.iterrows():
        ok = _huella_ok(f)
        marca = {True: f"`{f.huella}` ✓", False: f"`{f.huella}` ⚠ cambió", None: "—"}[ok]
        epoca = _num(f.epoca_elegida, 1 if f.tipo == "cv" else 0)
        lineas.append(
            f"| {f.fecha} | `{f.ejecucion}` | {f.modelo} | {f.perdida} | {f.folds} | {epoca} | {_auc(f)} "
            f"| {_num(f.auc_train)} | {_num(f.diferencia_auc)} | {_num(f.f1_macro)} | {marca} |"
        )
    return "\n".join(lineas)


def reemplazar_bloque(texto: str, nombre: str, contenido: str) -> str:
    """Sustituye lo que hay entre las marcas de un bloque automático."""
    patron = re.compile(rf"(<!-- auto:{nombre}:inicio -->).*?(<!-- auto:{nombre}:fin -->)", re.DOTALL)
    if not patron.search(texto):
        raise ValueError(f"Faltan las marcas del bloque auto:{nombre} en el documento")
    return patron.sub(lambda m: f"{m.group(1)}\n{contenido}\n{m.group(2)}", texto)


def actualizar_documento(documento: Path = DOCUMENTO, historico: Path = HISTORICO) -> bool:
    """Regenera los bloques automáticos. Devuelve True si el documento ha cambiado."""
    documento = Path(documento)
    if not documento.exists():
        return False
    datos = leer_historico(historico)
    texto = documento.read_text(encoding="utf-8")
    nuevo = texto
    for nombre, contenido in (("resumen", tabla_resumen(datos)), ("genealogia", genealogia()),
                              ("ejecuciones", tabla_ejecuciones(datos))):
        nuevo = reemplazar_bloque(nuevo, nombre, contenido)
    if nuevo != texto:
        documento.write_text(nuevo, encoding="utf-8")
    return nuevo != texto


def copiar_figuras(carpeta: Path, destino: Path = FIGURAS) -> list[Path]:
    """Copia graficas/*.png de una ejecución a docs/figuras/ejecuciones/<ejecución>/ (sí se versiona)."""
    carpeta = Path(carpeta)
    origen = carpeta / "graficas"
    if not origen.is_dir():
        raise SystemExit(f"{carpeta} no tiene graficas/. Créalas con: python -m src.training.graficas {carpeta}")
    salida = Path(destino) / carpeta.name
    salida.mkdir(parents=True, exist_ok=True)
    copiadas = []
    for png in sorted(origen.glob("*.png")):
        shutil.copy2(png, salida / png.name)
        copiadas.append(salida / png.name)
    return copiadas


def main() -> None:
    p = argparse.ArgumentParser(description="Partes automáticas del historial de arquitecturas.")
    p.add_argument("--figuras", nargs="+", metavar="CARPETA",
                   help="ejecuciones de runs/ cuyas gráficas se copian a docs/figuras/ejecuciones/")
    args = p.parse_args()

    for carpeta in args.figuras or []:
        print(f"Gráficas de {Path(carpeta).name} (enlaces para su ficha en docs/ARQUITECTURAS.md):")
        for ruta in copiar_figuras(Path(carpeta)):
            relativa = ruta.relative_to(RAIZ_REPO / "docs").as_posix()
            print(f"  ![{ruta.stem}]({relativa})")

    cambiado = actualizar_documento()
    print(f"docs/ARQUITECTURAS.md {'actualizado' if cambiado else 'ya estaba al día'}.")
    print(genealogia())


if __name__ == "__main__":
    main()
