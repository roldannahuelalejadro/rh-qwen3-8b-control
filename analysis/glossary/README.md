# Resultados del glosario

Esta carpeta conserva los resultados ligeros del análisis y la revisión manual.
Los notebooks están en [`../../notebooks/glossary/`](../../notebooks/glossary/README.md).

- `revision_manual.json`: 100 decisiones manuales, con hashes e historial.
- `revision_manual_Nahuel_2026-10-24_original.json`: exportación original preservada.
- `criterios_revision_manual.md`: reglas operacionales comunicadas por el revisor.
- `clasificacion_100.csv`: propuestas automáticas y etiquetas humanas separadas.
- `explorador_glosario_100.html`: visor autónomo con las 100 respuestas, botones de
  revisión, guardado local e importación/exportación de decisiones.
- `decoder_v1/`: métricas out-of-fold, particiones, hiperparámetros, PCA y b,g,H.
- `decoder_v2_checks/`: curvas densas, diagnósticos del intercepto, comparación
  cúbica y parámetros por fold; incluye el tensor cúbico simétrico.
- `publication_manifest.json`: hashes de los artefactos y correspondencia entre
  scripts originales y publicados. Solo se adaptaron rutas al organizar el repo.

La cache `decoder_v1/a_lj.npy` se regenera y no se versiona. Los arrays originales
`h` se obtienen del dataset público:

- Dataset: `NahuelRoldan/qwen3-8b-glossary-full-hidden-states-2026-10-23`.
- Revisión fijada: `54a33c4b473df6b65538ef0b7d89a72324fc56a1`.
- Subcarpeta del dataset: `cloud_glossary_2026-10-23`.

## Resultado principal de esta corrida

Evaluación fuera de entrenamiento por rollout, j=192, 72 respuestas etiquetadas:

| Decoder | AUROC | Log-loss |
|---|---:|---:|
| Constante | 0.500 | 0.604 |
| Lineal | 0.629 | 0.578 |
| Cuadrático diagonal | 0.690 | 0.553 |
| Cuadrático completo | 0.670 | 0.583 |
| Cúbico completo | 0.681 | 0.563 |

El cúbico es un seguimiento exploratorio de la prueba cuadrática. Estas cifras
no establecen superioridad estadística ni necesidad de los términos cruzados.
El modelo final ajustado sobre los 72 casos no se usa para reportar rendimiento:
las evaluaciones proceden de los archivos `*_oof.csv`.

En la sensibilidad al intercepto, cinco inicializaciones produjeron
probabilidades finales con diferencia máxima de 1.35e-5. Fijar el intercepto
es una restricción distinta: se reajustaron los otros coeficientes y se
conservaron las curvas para comparar su influencia durante la generación.

## Comprobaciones

```bash
python scripts/validate_glossary_analysis.py
node scripts/test_glossary_manual_review.cjs
```

Estas comprobaciones no necesitan activaciones grandes. Con los arrays y la
cache regenerada pueden ejecutarse también `check_glossary_decoder.py` y
`check_glossary_extensions.py` para comprobar promedios, coeficientes,
reconstrucción de probabilidades y ejes de los tensores.
