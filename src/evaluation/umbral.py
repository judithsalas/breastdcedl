"""Elección del umbral de decisión (D21), siempre con predicciones de validación, nunca de test.

El umbral convierte una probabilidad en una clase (pCR / no pCR). No sale del modelo: es una
decisión clínica sobre qué error preocupa más.

Criterio elegido: **la mayor sensibilidad posible con una especificidad de al menos 0,80**.
- Un falso positivo es decirle a una paciente que probablemente responderá por completo cuando no
  va a ser así. Es el error más peligroso si alguien usara la predicción para relajar el
  tratamiento o la vigilancia (el enunciado, §2.2, avisa precisamente de ese uso).
- Exigir especificidad ≥ 0,80 limita los falsos positivos a como mucho 1 de cada 5 pacientes sin
  pCR; dentro de ese límite se detectan tantas pCR como se pueda.
- Es un punto intermedio: el modelo de referencia del artículo (Fridman) trabaja con 0,95 de
  especificidad pero solo 0,27 de sensibilidad; con 0,80 se detectan bastantes más pCR.

Si varios umbrales dan exactamente el mismo resultado (pasa siempre: entre dos probabilidades
observadas la matriz de confusión no cambia), se toma el **punto medio** de ese intervalo y no un
extremo. Un umbral pegado al borde cambia de resultado con cualquier variación mínima de las
probabilidades (por ejemplo, entre el modelo de un fold y el conjunto de los 5).

Se calculan también, solo para comparar, el umbral de Youden (máximo de sensibilidad +
especificidad − 1), el de máxima F1 macro y el de 0,5.
"""

from __future__ import annotations

import numpy as np

from src.evaluation.metricas import coma, metricas_con_umbral

ESPECIFICIDAD_MINIMA = 0.80


def candidatos(p: np.ndarray) -> np.ndarray:
    """Umbrales posibles: cada probabilidad observada (y un poco por encima de la máxima)."""
    valores = np.unique(np.asarray(p, float))
    return np.append(valores, valores.max() + 1e-6)


def umbral_por_especificidad(y: np.ndarray, p: np.ndarray, especificidad_minima: float = ESPECIFICIDAD_MINIMA
                             ) -> dict:
    """Mayor sensibilidad con especificidad ≥ mínimo; a igual sensibilidad, mayor especificidad.

    Todos los umbrales con la misma sensibilidad y especificidad forman un intervalo (la
    sensibilidad solo baja y la especificidad solo sube al subir el umbral), y se devuelve su
    punto medio.
    """
    valores = candidatos(p)
    metricas = [metricas_con_umbral(y, p, t) for t in valores]
    validas = [i for i, m in enumerate(metricas) if m["especificidad"] >= especificidad_minima]
    if not validas:
        return None
    objetivo = max((metricas[i]["sensibilidad"], metricas[i]["especificidad"]) for i in validas)
    empatados = [i for i in validas if (metricas[i]["sensibilidad"], metricas[i]["especificidad"]) == objetivo]
    # Con el umbral en (valores[i-1], valores[i]] se predicen como pCR las mismas pacientes que con
    # valores[i]: el intervalo completo va desde la probabilidad observada anterior (sin incluirla).
    inferior = valores[min(empatados) - 1] if min(empatados) > 0 else 0.0
    return metricas_con_umbral(y, p, (inferior + valores[max(empatados)]) / 2)


def umbral_youden(y: np.ndarray, p: np.ndarray) -> dict:
    return max((metricas_con_umbral(y, p, t) for t in candidatos(p)),
               key=lambda m: m["sensibilidad"] + m["especificidad"])


def umbral_f1_macro(y: np.ndarray, p: np.ndarray) -> dict:
    return max((metricas_con_umbral(y, p, t) for t in candidatos(p)), key=lambda m: m["f1_macro"])


def comparar_umbrales(y: np.ndarray, p: np.ndarray) -> dict:
    """El umbral elegido y, para comparar, los de otros criterios."""
    return {
        "elegido": {"criterio": f"máxima sensibilidad con especificidad ≥ {coma(ESPECIFICIDAD_MINIMA, 2)}",
                    **umbral_por_especificidad(y, p)},
        "youden": {"criterio": "máximo de sensibilidad + especificidad − 1", **umbral_youden(y, p)},
        "f1_macro": {"criterio": "máxima F1 macro", **umbral_f1_macro(y, p)},
        "0,5": {"criterio": "umbral por defecto", **metricas_con_umbral(y, p, 0.5)},
    }
