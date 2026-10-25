# Notebooks del experimento del glosario

Los notebooks incluyen sus resultados y figuras: pueden leerse sin descargar
activaciones ni ejecutar inferencia. Conservan el orden direccion–paso–capa
de `Xi[i,j,ell]` y las etiquetas manuales de Nahuel.

1. [Revisión de 100 respuestas](01_revision_RH_glosario_100.ipynb): propuestas
   automáticas, revisión humana, respuestas completas y tabla EOS/LOT.
2. [Dos casos seleccionados](02_casos_r0001_RH_r0026_no_RH.ipynb): r0001 como RH
   y r0026 como no RH; solo contexto y acceso a tensores.
3. [Decoder PCA cuadrático](03_decoder_PCA_cuadratico.ipynb): promedios
   acumulativos, una componente por capa, regresión logística L2 y evaluación
   anidada por rollout. Comparación constante/lineal/diagonal/completa.
4. [Comprobaciones y extensión cúbica](04_ajuste_trayectorias_constante_cubico.ipynb):
   método de ajuste, curvas para cada token, constante inicial frente a fija y
   comparación de un término cúbico.

## Ejecutarlos

Desde la raíz del repositorio:

```bash
python -m pip install -r requirements-analysis.txt
python scripts/fetch_glossary_activations.py --rollouts r0001 r0026
```

El segundo comando descarga alrededor de 450 MiB, no los 21 GiB completos.
Con eso se puede ejecutar el notebook 02. El notebook 01 necesita únicamente
los JSON y tokens que ya están en el repo.

Para repetir los ajustes de 03 y 04, descargar los arrays completos y ejecutar
03 antes de 04:

```bash
python scripts/fetch_glossary_activations.py --all
```

Hace falta almacenamiento para unos 21 GiB de activaciones y 281 MiB para la
cache derivada `analysis/glossary/decoder_v1/a_lj.npy`. Esos arrays están
excluidos de Git. No se descarga ni carga la LLM y no se necesita GPU.

Si ya tenés la carpeta original, podés enlazar sus arrays sin copiarlos:

```bash
python scripts/fetch_glossary_activations.py --all --source "RUTA/cloud_glossary_2026-10-23"
```

Se usan enlaces duros en el mismo volumen. Los arrays originales deben
mantenerse inmutables: escribir sobre uno de los enlaces también cambia el
original. Los notebooks abren las activaciones en modo de solo lectura.

Abrir los notebooks con VS Code/Jupyter y elegir el entorno instalado. Las
rutas se resuelven buscando la raíz del repo, también si el directorio de
trabajo inicial es `notebooks/glossary/`.

## Interpretar los resultados

Las bases PCA y la regularización se ajustan dentro de cada entrenamiento.
Todos los prefijos de una respuesta se mantienen en el mismo fold. Los 28 X
no se convierten en negativos. Las etiquetas incluyen juicios manuales sobre
truncamiento y repetición: consultar los criterios antes de interpretar RH.

El entrenamiento usa j=0,32,64,128,192. Las curvas del notebook 04 muestran
todos los j, pero después de 192 representan extrapolación temporal. No hay
evaluación entre prompts distintos ni un score inicial diferente por semilla:
el prompt y su representación inicial son compartidos.
