"""Comprobaciones de los metadatos. No necesitan las imágenes: corren en CPU y en GitHub Actions."""

from src.data.caso import uc


def test_recuentos():
    samples = uc.cargar_samples()
    assert len(samples) == 12703
    assert samples.patient_id.nunique() == 1273


def test_ninguna_paciente_en_dos_splits():
    samples = uc.cargar_samples()
    assert (samples.groupby("patient_id").split.nunique() == 1).all()


def test_etiqueta_constante_por_paciente():
    samples = uc.cargar_samples()
    assert (samples.groupby("patient_id").pCR.nunique() == 1).all()


def test_folds_sin_solape():
    samples = uc.cargar_samples()
    for k in range(5):
        entrena, valida = uc.particion(samples, fold_val=k)
        assert not set(entrena.patient_id) & set(valida.patient_id)
