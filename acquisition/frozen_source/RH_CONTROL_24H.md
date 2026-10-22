# Control de cuantizacion y adquisicion RH

## Alcance

Corrida local secuencial con diez prompts y diez rollouts por prompt, en dos
checkpoints y dos precisiones: 400 respuestas previstas. No se intenta ajustar
un Hamiltoniano ni validar el predictor con diez tareas. Las variantes A/B
pertenecen al mismo grupo y no son dos tareas independientes.

La hipotesis predictiva se mantiene: un score representacional de alineamiento,
medido antes de generar la respuesta, permite predecir la aparicion posterior
de reward hacking. Esta corrida prepara datos y un control de cuantizacion;
no convierte automaticamente las coordenadas PCA en ese score.

## Modelos y procedencia

- Original: `unsloth/Qwen3-8B`, revision
  `946bc9ac74a6c1f8cf012497c503a119b2fcf2eb`.
- SFT publico: `longtermrisk/Qwen3-8B-school-of-reward-hacks-sft`, revision
  `28d32d8626b1f211c7d207bc85ae59bac1e3e8c8`.
- La ficha del SFT declara el modelo original, pero no confirma que sea el
  checkpoint exacto evaluado en el paper. No se transfieren sus porcentajes.
- Ambas condiciones usan el tokenizer y la plantilla del original. El SFT
  incluye un pre-tokenizer diferente y un token PAD adicional. Se verifica
  igualdad del vocabulario BPE, merges e IDs estructurales originales;
  batch 1 sin padding no usa el token adicional. Esto estandariza entradas,
  no reproduce la configuracion serializada completa del SFT.
- `rope_parameters` del SFT se traduce a `rope_theta=1000000`, compatible con
  Transformers 4.57.6. Sin esta traduccion, el cargador usaba 10000 por defecto.
- No se editan los pesos ni los archivos descargados. Revisiones, configuracion
  y auditoria quedan en el directorio del experimento.

## Casos congelados

Se transcriben los ejemplos funcionales del apendice C de
[School of Reward Hacks](https://arxiv.org/pdf/2508.17511), con comillas ASCII
e indentacion Python normalizadas. No se seleccionan segun nuestros resultados.

1. Histograma de palabras: tabla 6.
2. Duplicados consecutivos: tabla 6.
3. Hamlet y frase de verificacion: tabla 8.
4. Glosario: tabla 7.
5. Resena de Gatsby: tabla 7.
6. Dialogo y cantidad de turnos: tabla 7.
7. Omelette, recompensa favorable al chocolate: tabla 9.
8. Helado napolitano, recompensa desfavorable al chocolate: tabla 9.
9. Evaluador rapido en posicion A: tabla 10.
10. El mismo contraste con posiciones A/B intercambiadas: tabla 10.

## Muestreo

- Temperatura 1, top-p 1, top-k 0; semillas emparejadas entre condiciones.
- Thinking desactivado; EOS termina normalmente la respuesta.
- Hasta 768 tokens nuevos; maximo de entrada 2048, sin truncar prompts.
- Las respuestas que alcanzan el limite se marcan como truncadas. No se
  interpretan como ausencia de RH. El limite y thinking son decisiones locales
  documentadas, no una afirmacion de replica exacta del paper.
- Una sola respuesta en memoria cada vez. No se ejecuta codigo generado.
- No hay API ni juez externo de pago.

## Lecturas representacionales

- Salidas de bloques 6, 18 y 30, indices desde cero y antes de la norma final.
- Tres resumenes iniciales: promedio del prompt completo, ultimo token y
  promedio de los ultimos 16 tokens. Batch 1 no tiene padding.
- PCA centrado, sin whitening, hasta seis componentes. Se ajusta por checkpoint,
  capa y resumen en NF4, con 24 prompts de calibracion sin etiquetas.
- Calibracion: un prompt por tipo de tarea, seleccion determinista en el CSV
  publico de entrenamiento, revision `d7e04a550119cb5410494cf90e2313284a5f2148`.
  No se usan las respuestas RH/control del CSV. Se excluyen coincidencias
  textuales normalizadas con los diez casos; no se promete ausencia de semejanza.
- La base y centro NF4 de cada checkpoint se congelan para su control BF16.
  Las bases del original y el SFT no se interpretan como ejes semanticos iguales.
- Durante decoding se guardan coordenadas del ultimo estado del bloque 18
  cada ocho posiciones procesadas. Su posicion real esta registrada; el primer
  punto es el prefill previo a generar. Las lecturas posteriores son exploratorias.
- Se conservan vectores crudos iniciales y matrices de calibracion, ademas de xi.
  El visor SONAR no se mantiene abierto: no hay consumo de GUI ni una integracion
  en vivo que pueda modificar la corrida. Las trayectorias quedan para analisis/replay.

## Scoring y limites

La presencia de la frase y la eleccion inequivoca de evaluador tienen endpoints
automaticos especificos. El segundo no demuestra por si solo sabotaje real.
Una mencion de chocolate no demuestra una recomendacion, ni una metrica alta
demuestra RH. Codigo, recetas y textos requieren revision semantica posterior;
su etiqueta queda nula. Se guardan proxies descriptivos, no etiquetas inventadas.

No se comparan respuestas literalmente como criterio de fidelidad. La misma
semilla puede generar respuestas distintas al cambiar las probabilidades.
Diez rollouts no garantizan estimaciones precisas ni equivalencia entre condiciones.

## Orden y seguridad

Orden: original NF4, SFT NF4, original BF16, SFT BF16. Asi se prioriza terminar
la adquisicion cuantizada antes de los controles costosos.

- NF4: pesos cuantizados de cuatro bits, calculo BF16, todo en GPU.
- BF16: 16 bloques GPU y 20 bloques realmente ejecutados en CPU.
- Maximo de 23 horas desde el lanzamiento; termina antes si completa el plan.
- RAM libre minima del worker: 4 GiB; resguardo adicional del supervisor a 3 GiB.
- Reserva de VRAM con limite de asignacion; disco libre minimo 20 GiB.
- Temperatura GPU de parada: 82 C. No se modifican ventiladores ni limites de potencia.
- Guardado atomico por respuesta y checkpoint por calibracion.
- Un error detiene el lote; no hay un bucle infinito de reintentos.
- Mientras vive el supervisor solicita evitar suspension del sistema, sin cambiar
  permanentemente el plan de energia. No impide apagados, reinicios o cortes.
- Los resguardos son de mejor esfuerzo, no una garantia contra fallos del sistema.

## Archivos y operacion

Directorio: `outputs/rh_control_2026_10_21`.

- `status.json`: estado, PID, etapa y telemetria del supervisor.
- `ESTADO.md`: inventario provisional legible, actualizado durante la corrida.
- `protocol.json`: protocolo y hashes de codigo congelados.
- `tokenizer_audit.json`: diferencias de serializacion verificadas.
- Cada etapa tiene log, estado, calibracion, bases, capturas iniciales y rollouts.
- Cada respuesta tiene JSON con texto, IDs, semilla, tiempo, finalizacion y proxies;
  su trayectoria esta en un NPZ separado.
- `rollout_inventory.csv`: inventario acumulado; no un resultado predictivo final.

Para parar sin terminal: doble clic en `work/DETENER_CONTROL_RH.pyw`.
Tambien puede crearse un archivo vacio llamado `STOP` en el directorio de salida.
Se preservan las respuestas completas; la que esta en curso puede interrumpirse.

Para reanudar se retira `STOP` y se ejecuta el mismo supervisor con un nuevo
presupuesto temporal. Los resultados ya guardados se omiten; un protocolo/codigo
distinto exige otro directorio. No reutilizar las carpetas `rh_control_smoke_*`
como resultados cientificos: sus longitudes y cantidades se redujeron para probar.

```powershell
& '.venv-rh-benchmark/Scripts/python.exe' work/run_rh_control_overnight.py run --hours 23
& '.venv-rh-benchmark/Scripts/python.exe' work/run_rh_control_overnight.py status
```
