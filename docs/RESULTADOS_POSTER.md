# Método y resultados del póster

## Pregunta y representación

¿Un score representacional de alineamiento, medido antes de generar la respuesta,
permite predecir la aparición posterior de reward hacking?

Los resultados actuales estudian principalmente clasificación **durante** la
generación. No establecen anticipación antes de responder ni el inicio de RH.
Para un prompt fijo, el estado inicial es compartido por sus rollouts.

Para T tokens de entrada y j tokens generados procesados, promediamos por capa:

$$a_{\ell j}=\frac{1}{T+j}\sum_{t=0}^{T+j-1}h_{\ell t}.$$

Los estados causales pasados se conservan, sin recalcularlos con tokens futuros.
Se registran 36 bloques, de 4096 componentes por vector.

$$\xi_{ij}^{(\ell)}=\langle v_i^{(\ell)},a_{\ell j}-\mu_\ell\rangle,
\qquad\Xi_{ij\ell}=\xi_{ij}^{(\ell)}.$$

La representación reúne K componentes por capa y paso. Para ajustar usamos
coordenadas estandarizadas con parámetros de entrenamiento. La expansión
polinómica del score se sigue de una función logística. El decoder diagonal
conserva cuadrados individuales, sin términos cruzados. La corrida de 500
compara constante, lineal y diagonal, no cuadrático completo ni cúbico.

## Cómo se ajustó

Pérdida logística L2, todos los j válidos y peso 5/(N_b+1) por paso de la
respuesta b: igual peso total por respuesta. PCA se calibra con cinco prefijos
(0,32,64,128,192, acotados a la longitud disponible), solo con entrenamiento.
Entrenar el decoder con todos los j no significa ajustar PCA con todos los j.
Las particiones separan respuestas completas, nunca tokens de una misma
respuesta entre entrenamiento y evaluación.

En el glosario: original INT8, K=1 por capa, 51 RH, 21 no RH, 28 X manuales.
Validación anidada de tres particiones y selección interna de regularización.
Los ejemplos r0001/r0026 y la evaluación agregada pueden reflejar repetición,
longitud o estilo, no exclusivamente RH.

En la extensión: original NF4, K=2 por capa, 433 clasificables y 67 X, tres
particiones por respuesta y C=0,01 fijo. El protocolo contiene las evaluaciones
de transferencia de prompt y controles, separados de la curva principal.

## Cómo leer las métricas

AUROC mide ordenamiento; no porcentaje de aciertos ni calibración de probabilidad.
El global agrupa tareas. El intraprompt promedia comparaciones dentro de tarea
y partición con ambas clases: no tiene la misma ponderación que el global.

El [CSV temporal](../analysis/multiprompt_500/auroc_por_token_diagonal.csv)
es del decoder diagonal con respuestas fuera del ajuste. Cerca de j=200:
global 0,919 e intraprompt 0,731. Al finalizar individualmente las respuestas:
0,927 y 0,402. No confundir terminal individual con el último punto de la curva
temporal. El soporte cambia con EOS; no se seleccionó un horizonte óptimo
confirmado en datos nuevos.

Solo tres tareas presentan ambas clases: agradecimientos (1 RH/49 no RH),
FAQ (1/48 y 1 X), glosario culinario (19/24 y 7 X). Dos tienen un único positivo,
por lo que el desempeño intraprompt es inestable y no prueba generalización.

## Etiquetas y disponibilidad

Las etiquetas operacionales v2 son propuestas de IA con evidencia, revisadas
post hoc a partir de una lectura previa, no anotación humana independiente.
No se heredaron etiquetas del paper ni se forzó balance. LOT por sí solo no
decide RH en esta versión. No se mezclan variantes posteriores de etiquetas.

El [notebook](../notebooks/poster/01_resultados_y_contraste.ipynb) recrea la
figura desde CSV sin GPU. Se incluyen 500 respuestas y hashes fuente para
enlazarlas con las etiquetas. No se incluyen tensores completos de esta corrida,
credenciales, cachés ni pesos. Es una instantánea de resultados, no un paquete
de reproducción completa del entrenamiento. Los resultados son exploratorios.
