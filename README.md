# ¿Las activaciones internas permiten anticipar reward hacking?

**Qwen3-8B · Representaciones reducidas · Grupo 13 · Investigathon 2026**

[Póster completo](reports/poster_grupo_13.pdf) · [Galería ampliada](docs/GALERIA.md) · [Método y resultados](docs/RESULTADOS_POSTER.md) · [Notebook del póster](notebooks/poster/01_resultados_y_contraste.ipynb) · [Datos y etiquetas](analysis/multiprompt_500/README.md)

Estudiamos si un resumen de las activaciones internas de un modelo de lenguaje
permite distinguir respuestas con conductas relacionadas con el **reward hacking
(RH)**: satisfacer una métrica aparente sacrificando el objetivo de la tarea.

> **Resultado principal.** Observamos discriminación exploratoria durante la
> generación. Al ampliar las tareas, una métrica global alta no se sostuvo de
> manera uniforme dentro de cada prompt. **No demostramos una señal general de
> RH ni anticipación antes de generar la respuesta.**

## El recorrido

| Etapa | Diseño | Qué estudiamos |
| --- | --- | --- |
| **Control de precisión** | 400 respuestas: 10 prompts, original/SFT, NF4/BF16 | Concordancia de coordenadas iniciales |
| **Glosario** | 100 respuestas, original INT8; 51 RH, 21 no RH, 28 X | Discriminación en una tarea con etiquetas manuales |
| **Extensión** | 500 respuestas: 10 prompts × 50, original NF4 | AUROC global frente a discriminación intraprompt |

Las corridas, precisiones y versiones de etiquetas se mantienen separadas.
**X significa no clasificable**, no una respuesta negativa.

## Un prompt, dos respuestas

Promediamos los estados ocultos de cada capa sobre el prefijo disponible,
los proyectamos mediante PCA y ajustamos un pequeño decoder logístico.
Los ejemplos siguientes son respuestas reservadas del ajuste; no prueban
por sí solos que la señal detectada sea exclusivamente RH.

![Glosario: respuestas, trayectorias y AUROC](docs/assets/glosario.png)

**72 respuestas clasificables**, no miles de observaciones independientes por
contar tokens. El decoder usa todos los pasos válidos, con igual peso total por
respuesta. PCA, escalado y selección de regularización usan solo entrenamiento.

[Notebooks históricos](notebooks/glossary/README.md) · [Ajuste con todos los pasos](analysis/glossary/allj_v3/)

## Qué cambia al mezclar tareas

![AUROC global e intraprompt durante la generación](docs/assets/global_vs_intraprompt.png)

El decoder diagonal alcanzó un **AUROC global cercano a 0,9**, pero el promedio
dentro de cada prompt fue menor y variable, con una ventana cercana a **0,73
alrededor de j = 200**. La cohorte cambia al finalizar respuestas: las curvas
no comparan siempre el mismo conjunto de casos.

![Distribución de etiquetas en las tareas con ambas clases](docs/assets/contraste_por_tarea.png)

Solo tres tareas presentaron ambas clases; dos tenían **un único positivo**.
El resultado global puede aprovechar diferencias entre tareas y el contraste
intraprompt es frágil. Las etiquetas de esta extensión son **propuestas
operacionales post hoc asistidas por IA**, no validación humana independiente
ni las etiquetas manuales del glosario de 100 respuestas.

[Criterio y procedencia](analysis/multiprompt_500/CRITERIO.md) · [Curvas numéricas](analysis/multiprompt_500/auroc_por_token_diagonal.csv)

## Más allá del póster

### Original y SFT: control de cuantización

![Concordancia inicial NF4/BF16 en ambos checkpoints](docs/assets/gallery/precision_ambos_checkpoints.png)

Seis componentes por cada uno de diez prompts, con una base fija dentro de cada
checkpoint. Los 60 puntos de cada panel **no son 60 observaciones independientes**.
La concordancia geométrica no demuestra equivalencia conductual.

### ¿Qué cambia al conservar más componentes PCA?

![Comparación K=1 a K=6, AUROC y log-loss en j=192](docs/assets/gallery/componentes_K1_K6.png)

Mismos 72 rollouts del glosario, con todos los pasos en el ajuste del decoder.
Más componentes no mejoran uniformemente todas las métricas ni todos los órdenes.
Es una comparación exploratoria, no una elección de K confirmada en datos nuevos.

### ¿Puede ser una señal del vocabulario?

![Controles léxicos frente al decoder K2](docs/assets/gallery/control_lexico.png)

Las palabras de la respuesta también predicen parcialmente las etiquetas del
glosario. Esto hace plausible un atajo léxico, pero no prueba que sea el mecanismo
causal del decoder. El panel muestra promedios por partición; el punto final es
retrospectivo y los controles textuales no tienen idéntico esquema de ajuste.

**[Abrir la galería completa →](docs/GALERIA.md)**: nueve figuras adicionales,
incluyendo sensibilidad al intercepto, trayectorias para distintos K, término
cúbico, diferencias por capa y diagnósticos por tarea.

## Explorar sin GPU

- **Leer:** [póster PDF](reports/poster_grupo_13.pdf) y [método](docs/RESULTADOS_POSTER.md).
- **Revisar respuestas:** [500 textos originales](data/multiprompt_2026-10-28/responses.json), [prompts](data/multiprompt_2026-10-28/prompts.json) y [etiquetas con evidencia](analysis/multiprompt_500/semantic_review_v2.json).
- **Recrear la figura:** [notebook ligero](notebooks/poster/01_resultados_y_contraste.ipynb), sin descargar la LLM ni los tensores.
- **Explorar el piloto:** descargar [viewer/index.html](viewer/index.html) y abrirlo en el navegador. GitHub muestra su código, no ejecuta el HTML.
- **Activaciones del glosario de 100 respuestas:** [dataset en Hugging Face](https://huggingface.co/datasets/NahuelRoldan/qwen3-8b-glossary-full-hidden-states-2026-10-23). Los tensores grandes no están en Git.

```bash
python -m pip install numpy pandas matplotlib jupyter
jupyter notebook notebooks/poster/01_resultados_y_contraste.ipynb
```

## Mapa del repositorio

```text
reports/                       Póster y documentos
docs/assets/                   Figuras del póster
docs/RESULTADOS_POSTER.md       Método y límites
notebooks/poster/              Resultados finales, sin GPU
notebooks/glossary/            Análisis históricos del glosario
analysis/multiprompt_500/      Métricas, etiquetas y evidencia
data/multiprompt_2026-10-28/   500 respuestas y protocolo
data/control_2026-10-21/       Piloto original/SFT y NF4/BF16
analysis/glossary/             Resultados del glosario
viewer/                       Explorador autónomo del piloto
acquisition/                  Fuentes archivadas
```

## Alcance y reproducibilidad

El decoder es una **regresión logística con características polinómicas**, no
una PINN ni un autoencoder de reconstrucción. La matriz cuadrática modela un
score: no demuestra dinámica hamiltoniana en la LLM. SAE es una alternativa
futura; aquí usamos PCA.

Esta actualización permite leer los resultados y regenerar la figura del póster.
**No incluye los tensores completos de la corrida de 500 respuestas**: el notebook
ligero no reproduce la adquisición ni reentrena el decoder. Los notebooks
históricos conservan su alcance y rutas, documentados en sus carpetas.

Los escenarios proceden de *School of Reward Hacks*. El SFT público no se
confirmó como el checkpoint exacto del artículo. Los protocolos documentan
versiones y configuración; la correlación de coordenadas no prueba equivalencia
conductual entre precisiones.

[Control inicial](docs/CONTROL_PRECISION.md) · [Ficha de datos](docs/DATA_CARD.md) · [Procedencia](docs/PROVENANCE.md)

**Equipo del póster:** Nahuel Roldan, Lihuen Manzano y Juan Pablo Ance.
No se añade una licencia general sin acuerdo de los autores; los recursos de
terceros mantienen sus condiciones. Las consultas pueden hacerse mediante issues.
