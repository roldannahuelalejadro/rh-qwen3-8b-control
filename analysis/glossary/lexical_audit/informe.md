# Control lexico del glosario

## Interpretacion

La identidad de las palabras contiene informacion sobre las etiquetas historicas: presencia binaria de palabras alcanza AUROC agrupada 0.787 al final sin activaciones, sin orden de palabras y sin frecuencias de repeticion. El decoder obtiene 0.887. La diferencia 0.100 tiene intervalo exploratorio del 95 % [0.018, 0.195] con predicciones congeladas; no incorpora incertidumbre por reajuste ni seleccion de modelos. En j=64 y j=192 los intervalos de diferencia incluyen cero.

Esto demuestra que las etiquetas son parcialmente predecibles desde el vocabulario; no que el decoder use causalmente esas palabras ni que el resto de su senal sea alineamiento. Presencia binaria tambien captura riqueza lexica y cobertura tematica mediante su normalizacion. Las asociaciones de palabras cambian entre particiones: por ejemplo, accent, phoneme y allophone aparecen entre los pesos positivos en una particion, pero no constituyen un diccionario universal de RH.

El control textual de repeticion/diversidad de la auditoria anterior ya obtenia 0.823 al final, mas que estos controles lexicos, y no habia ventaja clara del decoder frente a el. No debemos seleccionar solo el control mas debil para declarar superioridad. Los modelos lexicos se entrenan por horizonte, mientras que el decoder se entreno con todos los j, por lo que no son experimentos de ajuste identicos.

Queda pendiente una prueba causal o de robustez: sustituir palabras manteniendo la conducta, o separar vocabularios entre entrenamiento y evaluacion. Los datos actuales permiten sostener que un atajo lexico es plausible, no que explique todo el resultado.

72 respuestas etiquetadas; 28 X excluidas. Sin nuevas generaciones ni cambios de etiquetas.
Vocabulario e IDF se ajustan solo con entrenamiento. C se selecciona en particiones internas.

| Grupo | Prefijo | Modelo | AUROC agrupada | AUROC media particiones |
|---|---|---|---:|---:|
| Todos | 32 | tfidf | 0.563 | 0.590 |
| Solo LOT | 32 | tfidf | 0.512 | 0.528 |
| Todos | 32 | presencia | 0.528 | 0.556 |
| Solo LOT | 32 | presencia | 0.476 | 0.487 |
| Todos | 32 | decoder_K2 | 0.629 | 0.690 |
| Solo LOT | 32 | decoder_K2 | 0.565 | 0.621 |
| Todos | 64 | tfidf | 0.677 | 0.653 |
| Solo LOT | 64 | tfidf | 0.642 | 0.612 |
| Todos | 64 | presencia | 0.645 | 0.636 |
| Solo LOT | 64 | presencia | 0.586 | 0.567 |
| Todos | 64 | decoder_K2 | 0.754 | 0.784 |
| Solo LOT | 64 | decoder_K2 | 0.706 | 0.751 |
| Todos | 128 | tfidf | 0.660 | 0.675 |
| Solo LOT | 128 | tfidf | 0.607 | 0.633 |
| Todos | 128 | presencia | 0.655 | 0.675 |
| Solo LOT | 128 | presencia | 0.596 | 0.619 |
| Todos | 128 | decoder_K2 | 0.701 | 0.714 |
| Solo LOT | 128 | decoder_K2 | 0.652 | 0.683 |
| Todos | 192 | tfidf | 0.662 | 0.700 |
| Solo LOT | 192 | tfidf | 0.605 | 0.647 |
| Todos | 192 | presencia | 0.673 | 0.711 |
| Solo LOT | 192 | presencia | 0.614 | 0.664 |
| Todos | 192 | decoder_K2 | 0.706 | 0.720 |
| Solo LOT | 192 | decoder_K2 | 0.623 | 0.652 |
| Todos | final | tfidf | 0.769 | 0.770 |
| Solo LOT | final | tfidf | 0.700 | 0.714 |
| Todos | final | presencia | 0.787 | 0.852 |
| Solo LOT | final | presencia | 0.730 | 0.812 |
| Todos | final | decoder_K2 | 0.887 | 0.905 |
| Solo LOT | final | decoder_K2 | 0.832 | 0.858 |
