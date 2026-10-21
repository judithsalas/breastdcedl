"""Recorte de cada corte alrededor del tumor, con su caja de `patients.csv` (D33).

    python -m src.data.recorte          # comprueba que las cajas caen sobre el tumor en los PNG

Por qué: el tumor ocupa un 3–8 % del corte (R5) y la red no consigue encontrarlo sola. Casi todos
los trabajos de pCR con buenos resultados le dan a la red una región centrada en el tumor en vez
del corte entero. Aquí se recorta un cuadrado de LADO_RECORTE × LADO_RECORTE píxeles centrado en
la caja del tumor, SIN cambiar la escala: el tamaño del tumor también es información (R5).

La caja (`sraw`, `eraw` filas; `scol`, `ecol` columnas) viene en píxeles del volumen original, de
`n_xy` × `n_xy`. Los PNG son 256 × 256, así que se multiplica por 256 / n_xy. Es la caja 3D del
tumor: en cada corte, el tumor queda dentro.

Límite importante: la app y la validación privada del profesor NO tienen la caja. Una versión
entrenada con este recorte no se puede desplegar tal cual: sirve para medir cuánto se gana
centrándose en el tumor (cnn_v11). Por eso el criterio de comparación no la elige como modelo
final (`desplegable` en `src/models/historial.py`) y `cerrar_modelo` se niega a cerrarla.

La comprobación (`python -m src.data.recorte`) mira, en el corte central de pacientes al azar, si
el realce (EARLY − PRE) es mayor dentro de la caja que en el resto del tejido. Lo mide con la caja
tal cual y traspuesta (filas por columnas), por si las coordenadas vinieran al revés, y dibuja
las dos en docs/resultados/1_auditoria/cajas.png.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.data.caso import NOMBRE_COHORTE, RAIZ_DATOS, RESULTADOS, uc

LADO_PNG = 256
LADO_RECORTE = 128          # ventana del recorte: cabe la caja del ~90 % de los tumores (R5)
RECORTES = ("caja",)
UMBRAL_TEJIDO = 0.1         # el mismo que la auditoría y cnn.py
FIGURA = RESULTADOS / "1_auditoria" / "cajas.png"


def cajas_en_png(patients: pd.DataFrame) -> dict[str, tuple[float, float, float, float]]:
    """Caja de cada paciente en píxeles del PNG: (fila inicial, fila final, columna inicial, columna final)."""
    p = patients.dropna(subset=["sraw", "eraw", "scol", "ecol", "n_xy"])
    escala = LADO_PNG / p.n_xy
    return {pid: (float(a), float(b), float(c), float(d)) for pid, a, b, c, d in
            zip(p.pid, p.sraw * escala, p.eraw * escala, p.scol * escala, p.ecol * escala)}


def ventana(caja: tuple[float, float, float, float], lado: int = LADO_RECORTE,
            tamano: int = LADO_PNG) -> tuple[int, int]:
    """Esquina superior izquierda de un cuadrado de `lado` centrado en la caja y dentro de la imagen."""
    f0, f1, c0, c1 = caja
    fila = round((f0 + f1) / 2 - lado / 2)
    columna = round((c0 + c1) / 2 - lado / 2)
    return min(max(fila, 0), tamano - lado), min(max(columna, 0), tamano - lado)


def recortar(x: torch.Tensor | np.ndarray, caja, lado: int = LADO_RECORTE):
    """Recorta las tres fases a la vez (misma ventana): (3, 256, 256) → (3, lado, lado)."""
    fila, columna = ventana(caja, lado, x.shape[-1])
    return x[..., fila:fila + lado, columna:columna + lado]


class DatasetRecortado(Dataset):
    """Como uc.BreastDCEDataset, pero cada corte se recorta alrededor del tumor de su paciente.

    El aumento (`transform`) se aplica DESPUÉS de recortar, a las tres fases a la vez.
    """

    def __init__(self, filas: pd.DataFrame, cajas: dict, transform=None, raiz=RAIZ_DATOS,
                 lado: int = LADO_RECORTE) -> None:
        self.base = uc.BreastDCEDataset(filas, raiz)
        faltan = sorted(set(self.base.filas.patient_id) - set(cajas))
        if faltan:
            raise ValueError(f"{len(faltan)} pacientes sin caja del tumor (p. ej. {faltan[:3]})")
        self.pacientes = self.base.filas.patient_id.to_numpy()
        self.cajas, self.transform, self.lado = cajas, transform, lado

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, i: int):
        x, y = self.base[i]
        x = recortar(x, self.cajas[self.pacientes[i]], self.lado).contiguous()
        if self.transform is not None:
            x = self.transform(x)
        return x, y


def crear_dataset(filas: pd.DataFrame, recorte: str | None, transform=None, raiz=RAIZ_DATOS) -> Dataset:
    """El Dataset que pide la receta de una versión: el de siempre, o recortado por la caja."""
    if recorte is None:
        return uc.BreastDCEDataset(filas, raiz, transform=transform)
    if recorte == "caja":
        return DatasetRecortado(filas, cajas_en_png(uc.cargar_patients(raiz)), transform, raiz)
    raise ValueError(f"Recorte desconocido: {recorte!r}. Opciones: {RECORTES}")


# --------------------------------------------------------------------------- #
# Comprobación: ¿la caja cae sobre el tumor en los PNG?
# --------------------------------------------------------------------------- #

def traspuesta(caja):
    f0, f1, c0, c1 = caja
    return c0, c1, f0, f1


def realce_dentro_fuera(x: np.ndarray, caja) -> float | None:
    """Realce medio (EARLY − PRE, sin negativos) dentro de la caja dividido entre el del resto del tejido."""
    realce = np.clip(x[1] - x[0], 0, None)
    tejido = np.maximum(x[0], x[1]) > UMBRAL_TEJIDO
    dentro = np.zeros_like(tejido)
    f0, f1, c0, c1 = (round(v) for v in caja)
    dentro[max(f0, 0):max(f1, 0), max(c0, 0):max(c1, 0)] = True
    a, b = realce[dentro & tejido], realce[~dentro & tejido]
    if a.size < 20 or b.size < 20 or b.mean() <= 0:
        return None
    return float(a.mean() / b.mean())


def comprobar(samples: pd.DataFrame, patients: pd.DataFrame, n: int = 150, semilla: int = 0,
              raiz=RAIZ_DATOS) -> dict:
    """Mediana del cociente dentro / fuera con la caja tal cual y traspuesta, en n pacientes de train."""
    cajas = cajas_en_png(patients)
    tr = samples[(samples.split == "train") & samples.patient_id.isin(cajas)
                 & samples.path_pre.map(lambda r: (raiz / r).exists())]
    pacientes = tr.patient_id.unique()
    rng = np.random.default_rng(semilla)
    elegidas = rng.choice(pacientes, min(n, len(pacientes)), replace=False)
    cocientes = {"tal cual": [], "traspuesta": []}
    for pid in elegidas:
        suyos = tr[tr.patient_id == pid].sort_values("slice_index")
        x = uc.cargar_imagen(suyos.iloc[len(suyos) // 2], raiz)
        for nombre, caja in (("tal cual", cajas[pid]), ("traspuesta", traspuesta(cajas[pid]))):
            q = realce_dentro_fuera(x, caja)
            if q is not None:
                cocientes[nombre].append(q)
    return {nombre: {"pacientes": len(v), "mediana": float(np.median(v)) if v else float("nan"),
                     "mayor_que_1": float(np.mean(np.array(v) > 1)) if v else float("nan")}
            for nombre, v in cocientes.items()}


def figura(samples: pd.DataFrame, patients: pd.DataFrame, ruta=FIGURA, por_cohorte: int = 4,
           semilla: int = 0, raiz=RAIZ_DATOS):
    """EARLY del corte central con la caja tal cual (verde) y traspuesta (naranja, discontinua)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    cajas = cajas_en_png(patients)
    cohorte = patients.set_index("pid").dataset
    tr = samples[(samples.split == "train") & samples.patient_id.isin(cajas)
                 & samples.path_pre.map(lambda r: (raiz / r).exists())]
    rng = np.random.default_rng(semilla)
    cohortes = [c for c in ("spy1", "spy2", "duke") if c in set(tr.patient_id.map(cohorte))]
    fig, ejes = plt.subplots(len(cohortes), por_cohorte, figsize=(3 * por_cohorte, 3.1 * len(cohortes)),
                             squeeze=False)
    for fila_ejes, c in zip(ejes, cohortes):
        pacientes = tr[tr.patient_id.map(cohorte) == c].patient_id.unique()
        for ax, pid in zip(fila_ejes, rng.choice(pacientes, min(por_cohorte, len(pacientes)), replace=False)):
            suyos = tr[tr.patient_id == pid].sort_values("slice_index")
            ax.imshow(uc.cargar_imagen(suyos.iloc[len(suyos) // 2], raiz)[1], cmap="gray", vmin=0, vmax=1)
            for caja, color, estilo in ((cajas[pid], "#2fbf71", "-"), (traspuesta(cajas[pid]), "#eb6834", "--")):
                f0, f1, c0, c1 = caja
                ax.add_patch(Rectangle((c0, f0), c1 - c0, f1 - f0, fill=False, color=color, lw=1.6, ls=estilo))
            fila, columna = ventana(cajas[pid])
            ax.add_patch(Rectangle((columna, fila), LADO_RECORTE, LADO_RECORTE, fill=False, color="#4a7fc1",
                                   lw=1, ls=":"))
            ax.set_title(pid[-10:], fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
        fila_ejes[0].set_ylabel(NOMBRE_COHORTE[c], fontsize=11)
    fig.suptitle("Caja del tumor: verde = tal cual · naranja = traspuesta · azul punteado = recorte de "
                 f"{LADO_RECORTE}×{LADO_RECORTE}", fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    ruta.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ruta, dpi=110, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return ruta


def main() -> None:
    p = argparse.ArgumentParser(description="Comprueba que las cajas del tumor caen sobre el tumor en los PNG.")
    p.add_argument("--pacientes", type=int, default=150)
    args = p.parse_args()
    samples, patients = uc.cargar_samples(), uc.cargar_patients()
    r = comprobar(samples, patients, args.pacientes)
    print("Realce dentro de la caja / realce en el resto del tejido (corte central; > 1 = la caja realza más):")
    for nombre, v in r.items():
        print(f"  caja {nombre:<10}: mediana {v['mediana']:.2f} · mayor que 1 en el {100 * v['mayor_que_1']:.0f} % "
              f"({v['pacientes']} pacientes)")
    bien, mal = r["tal cual"]["mediana"], r["traspuesta"]["mediana"]
    if bien > 1.3 and bien > mal:
        print("OK: la caja tal cual cae sobre la zona que realza. Se puede entrenar con el recorte.")
    else:
        print("CUIDADO: la caja tal cual no cae claramente sobre la zona que realza. No entrenes aún: "
              "manda la figura a Claude.")
    print(f"Figura: {figura(samples, patients)}")


if __name__ == "__main__":
    main()
