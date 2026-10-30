# Extensión de 500 respuestas: versión del póster

Esta carpeta conserva la revisión operacional **v2 general**, sin mezclar
variantes posteriores de LOT u observables distintos.

- [Respuestas](../../data/multiprompt_2026-10-28/responses.json): 500 textos, semillas y finalización.
- [Prompts](../../data/multiprompt_2026-10-28/prompts.json) y [protocolo de adquisición](../../data/multiprompt_2026-10-28/protocol.json).
- [Etiquetas y evidencia](semantic_review_v2.json): propuestas post hoc asistidas por IA, no revisión humana independiente.
- [Criterio](CRITERIO.md): significado de 0/1/X y procedencia.
- [Curva por token](auroc_por_token_diagonal.csv): global, intraprompt y soporte, decoder diagonal.
- [Protocolo de ajuste](protocol.json): PCA, pesos, particiones y controles.
- `metrics.csv`, `primary_metrics.json`: métricas por modelo y evaluación.
- `predicciones_terminales_500.csv`: predicciones al terminar cada respuesta.
- `conditional_terminal_controls.csv`: controles terminales, no curvas temporales.

Los 67 X se excluyen de pérdidas y métricas. Predicciones sobre X no son nuevas
etiquetas. Los tokens no son respuestas independientes. Los tensores completos
y parámetros entrenados no forman parte de esta instantánea. El notebook
ligero reproduce la figura, no el entrenamiento.
