"""Orden de las carpetas (D26): lo que se lee arriba en docs/, lo que genera el código en docs/resultados/."""

from src.data import auditoria
from src.data.caso import RAIZ_REPO, RESULTADOS
from src.evaluation import cerrar_modelo, evaluar_test, modelo_clinico

DOCUMENTOS_PARA_LEER = {"COMO_SE_HIZO.md", "ARQUITECTURAS.md", "REGISTRO.md"}


def test_cada_paso_guarda_en_su_carpeta_numerada():
    assert auditoria.CARPETA == RESULTADOS / "1_auditoria" and auditoria.SALIDA_MD.parent == auditoria.CARPETA
    assert cerrar_modelo.FIGURAS == RESULTADOS / "2_cierre" and cerrar_modelo.INFORME.parent == cerrar_modelo.FIGURAS
    assert modelo_clinico.FIGURAS == RESULTADOS / "3_clinico" and modelo_clinico.INFORME.parent == modelo_clinico.FIGURAS
    assert modelo_clinico.OOF_CNN.parent == cerrar_modelo.FIGURAS
    assert evaluar_test.CARPETA_TEST == RESULTADOS / "4_test"


def test_en_docs_solo_quedan_los_documentos_para_leer():
    sueltos = {p.name for p in (RAIZ_REPO / "docs").glob("*.md")}
    assert sueltos == DOCUMENTOS_PARA_LEER, (
        f"Ficheros sueltos en docs/: {sorted(sueltos - DOCUMENTOS_PARA_LEER)}. Los informes generados van en "
        "docs/resultados/<n>_<paso>/ (D26); si vienen de antes de la limpieza, muévelos allí o vuelve a generarlos")
    carpetas = {p.name for p in (RAIZ_REPO / "docs").iterdir() if p.is_dir()}
    assert carpetas <= {"figuras", "resultados"}, f"Carpetas de más en docs/: {sorted(carpetas - {'figuras', 'resultados'})}"


def test_no_quedan_carpetas_vacias_de_la_plantilla():
    assert not (RAIZ_REPO / "notebooks").exists(), "notebooks/ ya no se usa (D26): la auditoría es un script"


def test_los_init_estan_vacios():
    """Los __init__.py solo marcan carpetas como paquetes: el código vive en módulos con nombre."""
    con_algo = [p.relative_to(RAIZ_REPO).as_posix() for carpeta in ("src", "tests", "app")
                for p in (RAIZ_REPO / carpeta).rglob("__init__.py") if p.read_bytes().strip()]
    assert not con_algo, f"Estos __init__.py tienen contenido y deben estar vacíos: {con_algo}"
