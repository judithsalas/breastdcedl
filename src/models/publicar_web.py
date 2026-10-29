"""Sube la app web (`web/`) a un Space estático de Hugging Face (D38).

    python -m src.models.publicar_web <usuario>/breastdcedl-pcr

Antes, una sola vez:
1. crear una cuenta en https://huggingface.co y un token de escritura (Settings → Access Tokens → «Write»);
2. guardarlo en el ordenador: `python -c "from huggingface_hub import login; login()"` y pegar el token.

Crea el Space si no existe (estático, gratis), comprueba que `web/` está completa y que el ONNX es el que
dice `info.json`, y sube la carpeta. Un Space estático solo sirve ficheros: no hay servidor que se duerma.
No sube `web/pruebas/` ni `web/package.json`, que solo sirven para los tests.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

from src.models.exportar_web import WEB

NO_SE_SUBEN = ["pruebas/*", "package.json"]


def comprobar_carpeta(web: Path = WEB) -> dict:
    """Que estén el modelo, su info y los ejemplos, y que el ONNX no haya cambiado desde la exportación."""
    necesarios = ("index.html", "red.html", "resultados.html", "README.md", "modelo/modelo.onnx", "modelo/una_red.onnx",
                  "modelo/info.json", "ejemplos/ejemplos.json", "datos/resultados.json")
    faltan = [r for r in necesarios if not (web / r).exists()]
    if faltan:
        raise SystemExit(f"Faltan en web/: {', '.join(faltan)}. Créalos con: python -m src.models.exportar_web")
    info = json.loads((web / "modelo" / "info.json").read_text(encoding="utf-8"))
    huella = hashlib.sha256((web / "modelo" / "modelo.onnx").read_bytes()).hexdigest()
    if huella != info["onnx"]["sha256"]:
        raise SystemExit("web/modelo/modelo.onnx no es el de info.json. Vuelve a exportar: python -m src.models.exportar_web")
    resultados = json.loads((web / "datos" / "resultados.json").read_text(encoding="utf-8"))
    if resultados.get("huella_modelo") != info["huella_modelo"]:
        raise SystemExit("web/datos/resultados.json es de otro modelo. Vuelve a exportar: python -m src.models.exportar_web")
    return info


def publicar(space: str, web: Path = WEB) -> str:
    from huggingface_hub import HfApi

    info = comprobar_carpeta(web)
    api = HfApi()
    api.create_repo(space, repo_type="space", space_sdk="static", exist_ok=True)
    api.upload_folder(repo_id=space, folder_path=str(web), repo_type="space", ignore_patterns=NO_SE_SUBEN,
                      commit_message=f"App web con el modelo {info['huella_modelo']}")
    return f"https://huggingface.co/spaces/{space}"


def main() -> None:
    if len(sys.argv) != 2 or "/" not in sys.argv[1]:
        sys.exit("Uso: python -m src.models.publicar_web <usuario>/<nombre-del-space>")
    url = publicar(sys.argv[1])
    print(f"Subido. La app estará en {url} en uno o dos minutos.")


if __name__ == "__main__":
    main()
