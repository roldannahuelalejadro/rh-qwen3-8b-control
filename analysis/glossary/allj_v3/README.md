# Glosario: decoder entrenado con todos los pasos

Esta es la versión usada en el póster, posterior a los notebooks históricos
03 y 04 (que entrenaban sobre cinco prefijos). No reemplaza sus métricas.

- `summary.json`: protocolo, particiones y resultados del ajuste completo.
- `metrics_oof.csv`: métricas agregadas fuera del ajuste.
- `allj_reference_cases.png`: trayectorias de dos respuestas reservadas.
- `allj_fixed_intercept_cases.png`: sensibilidad al intercepto fijo, reajustando
  los demás coeficientes.

El decoder usa todos los j válidos con igual peso total por respuesta; PCA
conserva la calibración en cinco prefijos dentro del entrenamiento. K=1 por
capa, original INT8; 72 etiquetas manuales clasificables. No confundir con
la extensión de 500 respuestas NF4/K=2 y etiquetas operacionales v2.

Esta carpeta es una instantánea de resultados ligeros; no contiene tensores
completos ni todos los estados del optimizador para repetir el entrenamiento.
