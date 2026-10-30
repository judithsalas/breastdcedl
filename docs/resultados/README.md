# Resultados (los genera el código)

Aquí se guarda lo que producen los scripts: informes, figuras y tablas. **No se edita a mano**:
si algo cambia, se vuelve a ejecutar su script y se sobrescribe.

Hay una carpeta por paso, numerada en el orden en que se ejecutan:

| Carpeta | Qué contiene | Se crea con |
|---|---|---|
| `1_auditoria/` | Cómo son los datos: recuentos, cohortes, intensidades y ejemplos | `python -m src.data.auditoria` |
| `2_cierre/` | El modelo final: versión elegida, calibración, umbral y métricas internas | `python -m src.evaluation.cerrar_modelo` |
| `3_clinico/` | El «médico» sin imagen y la comparación con la CNN | `python -m src.evaluation.modelo_clinico` |
| `4_test/` | El resultado en test, que se calcula **una sola vez** | `python -m src.evaluation.evaluar_test` |

Cada carpeta tiene su informe (`.md`) con las figuras al lado. Qué hace cada paso y por qué: en el
README principal y al principio de cada script.
