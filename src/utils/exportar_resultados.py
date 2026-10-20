"""Empaqueta los resultados en un zip pequeño, sin imágenes ni pesos.

    python -m src.utils.exportar_resultados

Sirve para revisar los resultados sin copiar las imágenes del dataset (son ~38.000 PNG y no hacen falta: los
entrenamientos se hacen en el PC de la uni). Lo que hace falta para leer resultados y decidir son
ficheros pequeños, y este script los junta en un zip en la carpeta del repo:

- runs/historico.csv y, de cada ejecución de runs/, todo menos los pesos (*.pt): configuración y
  métricas (resultados.json, resumen.json), épocas (historial.csv), predicciones y gráficas;
- docs/resultados/ entero (auditoría, cierre, médico y test, si ya existen);
- los manifiestos de models/final/ y models/clinico/ (sin pesos);
- estado_git.txt: los últimos commits y lo que está sin subir, para saber en qué punto está el
  repo.

Nunca mete breastdcedl/ (ni imágenes ni metadatos) ni ficheros .pt. El zip
(resultados_<fecha>.zip) está en .gitignore: no se sube a GitHub.
"""

from __future__ import annotations

import subprocess
import zipfile
from datetime import datetime
from pathlib import Path

from src.data.caso import RAIZ_REPO

AVISO_MB = 25            # por encima, avisar: puede ser grande para enviarlo


def ficheros_a_exportar(raiz: Path = RAIZ_REPO) -> list[Path]:
    """Lo que entra en el zip, como rutas relativas a la raíz del repo."""
    elegidos: list[Path] = []
    runs = raiz / "runs"
    if runs.is_dir():
        elegidos += [p for p in runs.rglob("*") if p.is_file() and p.suffix != ".pt"]
    resultados = raiz / "docs" / "resultados"
    if resultados.is_dir():
        elegidos += [p for p in resultados.rglob("*") if p.is_file()]
    for manifiesto in (raiz / "models" / "final" / "manifiesto.json", raiz / "models" / "clinico" / "modelo_clinico.json"):
        if manifiesto.exists():
            elegidos.append(manifiesto)
    return sorted({p.relative_to(raiz) for p in elegidos if "__pycache__" not in p.parts})


def estado_git(raiz: Path = RAIZ_REPO) -> str:
    """Últimos commits y cambios sin subir (o un aviso si git no está disponible)."""
    partes = []
    for orden in (["git", "log", "--oneline", "-10"], ["git", "status", "--short"]):
        try:
            salida = subprocess.run(orden, cwd=raiz, capture_output=True, text=True, timeout=30, check=False)
            partes.append(f"$ {' '.join(orden)}\n{salida.stdout or salida.stderr}")
        except (OSError, subprocess.SubprocessError) as error:
            partes.append(f"$ {' '.join(orden)}\n(no se pudo ejecutar: {error})")
    return "\n".join(partes)


def empaquetar(raiz: Path = RAIZ_REPO, destino: Path | None = None) -> tuple[Path, list[Path]]:
    fecha = datetime.now().astimezone()
    destino = destino or raiz / f"resultados_{fecha:%Y%m%d-%H%M}.zip"
    ficheros = ficheros_a_exportar(raiz)
    with zipfile.ZipFile(destino, "w", zipfile.ZIP_DEFLATED) as z:
        for relativo in ficheros:
            z.write(raiz / relativo, relativo.as_posix())
        z.writestr("estado_git.txt", f"Generado el {fecha:%Y-%m-%d %H:%M}\n\n{estado_git(raiz)}")
    return destino, ficheros


def main() -> None:
    destino, ficheros = empaquetar()
    mb = destino.stat().st_size / 2**20
    ejecuciones = sorted({f.parts[1] for f in ficheros if f.parts[0] == "runs" and len(f.parts) > 2})
    print(f"{destino.name}: {len(ficheros)} ficheros, {mb:.1f} MB")
    print(f"  ejecuciones de runs/: {len(ejecuciones)}")
    print(f"  informes de docs/resultados/: {sum(1 for f in ficheros if f.suffix == '.md' and f.parts[0] == 'docs')}")
    if mb > AVISO_MB:
        print(f"  Aviso: pesa más de {AVISO_MB} MB.")
    print("No se sube a GitHub (está en .gitignore).")


if __name__ == "__main__":
    main()
