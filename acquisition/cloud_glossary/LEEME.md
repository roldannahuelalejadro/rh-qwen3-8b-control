# Experimento del glosario en la GPU remota

El programa corre en Ubuntu, en la T4 de la nube. Tu PC envia archivos y ordenes
por SSH; los calculos y resultados permanecen en la computadora remota.
Cerrar SSH no termina un proceso iniciado con tmux.

## Configuracion

- Original `unsloth/Qwen3-8B`, revision fijada en `run.py`.
- Pesos en LLM.int8; modulos no cuantizados y capturas en FP16.
- Prompt del glosario sin cambios, plantilla original, thinking desactivado.
- Temperatura 1, top-p 1, top-k 0, hasta 768 tokens, una respuesta a la vez.
- Semilla `230000 + indice del rollout`, indice desde cero.
- La etiqueta RH queda pendiente de revision manual.

## Archivos de cada respuesta

`prompt_h.npy` guarda una vez las activaciones del prompt en todos los bloques.
Cada carpeta `r0001`, `r0002`, etc., contiene:

- `generated_h.npy`: [posicion, bloque, componente], FP16.
- `generated_ids.npy`: identificadores de los tokens generados.
- `result.json`: texto, semilla, longitud, finalizacion, tiempos y hashes.
- `progress.json`: ultimo avance registrado, util ante interrupciones.

Las posiciones validas son exclusivamente `[:valid_positions]`. Las restantes
son ceros reservados; no deben entrar en promedios ni analisis.
Se guarda la salida de cada bloque antes de la normalizacion final.
El estado de la posicion j ya incorpora el token j y predice el token j+1.
El ultimo token, incluido EOS, recibe un forward adicional para guardar su
estado. Ese forward no produce otro token.

## Ejecucion en Ubuntu

La carpeta remota es `~/cloud_glossary`. `setup.sh` crea el entorno y descarga
solo JSON y safetensors del checkpoint fijado, sin duplicar una copia exportada.

```bash
cd ~/cloud_glossary
bash setup.sh
bash launch.sh smoke
```

La prueba usa dos respuestas de 32 tokens. Tras comprobarla, iniciar el primer
lote de 20 respuestas completas en una sesion persistente:

```bash
tmux new-session -d -s glossary-run 'cd ~/cloud_glossary && bash launch.sh acquire 20 > acquire.log 2>&1'
tail -n 20 ~/cloud_glossary/acquire.log
```

`acquire.exit` contiene el codigo de salida cuando termina. Cero significa que
terminaron la adquisicion y la verificacion. El objetivo puede ampliarse a 40,
60 o 100 con el mismo comando; las respuestas completas se verifican y omiten.
No modificar el script o entorno entre lotes: el protocolo comprueba cambios.

Se reservan 12 GiB de disco y se detiene la corrida si quedan menos de 3 GiB
de RAM disponible. El checkpoint tiene 36 bloques y 4096 componentes por vector.
Cada respuesta completa reserva 216 MiB de activaciones (4,22 GiB por lote de 20).
Antes de ampliar el lote, comprobar espacio y descargar los resultados.

Para detener la captura de forma controlada:

```bash
touch ~/cloud_glossary/acquisition/STOP
```

Una respuesta interrumpida queda incompleta, sin etiqueta ni `result.json`.
Debe inspeccionarse y moverse a una carpeta de parciales antes de reanudar;
el programa no la sobreescribe. Los rollouts completos se conservan.

## Descargar los resultados

La corrida programada el 23 de octubre tiene objetivo de 100 rollouts.
`after_smoke.sh` espera el exito de la prueba corta y lanza ese objetivo.
`receive.py` corre en segundo plano en Windows: descarga cada carpeta terminada,
comprueba hashes, dimensiones y valores finitos, y envia un recibo a la nube.
Solo entonces `archive.py` retira la copia remota de `generated_h.npy`.
La copia completa queda en `output/cloud_glossary_2026-10-23` del workspace.
Los metadatos y recibos permanecen en la nube. La verificacion remota reconoce
estos recibos; la verificacion completa de los arrays se hace en el destino.

Tu PC debe permanecer encendida, conectada y sin suspension para descargar.
Si el receptor se interrumpe, la nube conserva los archivos pendientes y la
adquisicion se detiene al alcanzar su reserva de disco. Para continuar tras
una parada hay que revisar el log y reanudar la corrida y el receptor.

Desde PowerShell local, usar `scp -r` con la misma clave y `known_hosts` que
SSH. El origen es `ubuntu@3.141.99.235:~/cloud_glossary/acquisition` y el
destino una carpeta local con espacio suficiente. Comprobar los hashes con
`verify.py` antes de considerar cualquier limpieza del almacenamiento remoto.
Nunca incluir claves privadas en el repositorio o en la carpeta del experimento.
