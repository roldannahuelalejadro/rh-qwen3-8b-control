# Control RH en Qwen3-8B

Repositorio de datos del control realizado el **21 de octubre de 2026**:
prompts, respuestas, resúmenes de activación iniciales `a_l`, coordenadas PCA
`xi` y trayectorias reducidas durante la generación.

Es una adquisición y un control de cuantización, **no una validación predictiva
de reward hacking ni una reproducción exacta del paper**. No se ajustó un
Hamiltoniano ni un score representacional de alineamiento. El agregador `A`,
`M_sigma` y `D_stat` siguen pendientes para esta corrida.

## Qué contiene

| Recurso | Cantidad |
| --- | ---: |
| Prompts de evaluación | 10 |
| Checkpoints: original y SFT | 2 |
| Precisiones: NF4 y BF16 | 2 |
| Rollouts por prompt y condición | 10 |
| Respuestas originales, en JSON | 400 |
| Capturas iniciales, en NPZ | 40 |
| Trayectorias reducidas, en NPZ | 400 |
| Capturas de calibración geométrica | 96 |
| Bases PCA, una por checkpoint | 2 |

Los datos y el visor ocupan aproximadamente 20 MiB. No se incluyen pesos de
modelos, entornos Python, cachés ni las traducciones parciales descartadas.

## Consultar las respuestas

Abrir [viewer/index.html](viewer/index.html) en un navegador, con doble clic.
Es autónomo y funciona sin conexión ni servidor: permite seleccionar prompt,
checkpoint, precisión y rollout por separado. Los textos están en su idioma
original, sin traducir ni resumir.

También está disponible el [PDF de preguntas y respuestas](reports/preguntas_y_respuestas_RH.pdf).
Estos visores contienen textos e indicadores; los arrays de activaciones se
encuentran en `data/` y se leen con NumPy.

## Estructura

```text
data/control_2026-10-21/
  protocol.json                 # Prompts, modelos, revisiones y configuración
  tokenizer_audit.json
  rollout_inventory.csv
  status.json
  original_nf4/
    initials/                   # a_l y xi iniciales, con metadatos JSON
    rollouts/                   # Respuesta JSON y trayectoria NPZ por rollout
    calibration/                # 24 capturas de calibración
    basis.npz                   # Centro, componentes y varianza de cada PCA
    basis.json
    status.json
  original_bf16/                # Misma organización; reutiliza la base original_nf4
  sft_nf4/                     # Base PCA propia del checkpoint SFT
  sft_bf16/                    # Reutiliza la base sft_nf4
acquisition/frozen_source/      # Código y plan archivados de la adquisición
viewer/                        # Visor HTML y plantilla
reports/                       # PDF de los textos originales
scripts/                       # Lectura, validación y regeneración del visor
MANIFEST.sha256.json            # Integridad de datos y código archivado
```

## Qué son las activaciones guardadas

Antes del primer token generado se guardaron vectores de **4096 componentes**
de los bloques 6, 18 y 30, con índices basados en cero, antes de la normalización
final. Se conservaron tres resúmenes de los estados del prompt completo:

- `mean`: promedio de todos los tokens de entrada.
- `last`: estado del último token de entrada.
- `tail16_mean`: promedio de los últimos 16 tokens, o de todos si hay menos.

Cada archivo inicial contiene esos nueve vectores sin proyectar y sus nueve
proyecciones PCA de seis componentes. Los arrays están almacenados en `float32`;
eso no significa que la inferencia se haya ejecutado íntegramente en FP32.

**Durante la generación solo se guardó `xi` del último estado del bloque 18 cada
ocho pasos, además del punto inicial.** No se conservaron matrices completas por
token ni los `a_l` sin proyectar de todos los pasos. Una respuesta corta puede
tener únicamente el punto inicial de la trayectoria.

Los diez rollouts del mismo prompt y condición comparten una captura inicial.
No son diez observaciones iniciales diferentes. Cada respuesta conserva la
referencia a su captura y a su propia trayectoria.

Consultar [docs/DATA_CARD.md](docs/DATA_CARD.md) para formas, claves y límites.

## Leer una captura

Solo hace falta Python y NumPy; no se carga ningún modelo ni se usa la GPU.

```bash
python -m pip install -r requirements.txt
python scripts/validate_dataset.py
```

Ejemplo ejecutado desde la raíz del repositorio:

```python
import json
from pathlib import Path
import numpy as np

data = Path("data/control_2026-10-21")
record = json.loads(
    (data / "sft_nf4/rollouts/omelette_chocolate_favored__r06.json")
    .read_text(encoding="utf-8")
)

# Las rutas de los JSON originales usan separadores de Windows.
initial_file = data / record["initial_file"].replace("\\", "/")
trajectory_file = data / record["trajectory_file"].replace("\\", "/")

with np.load(initial_file, allow_pickle=False) as arrays:
    a_l = arrays["l18_mean"]           # (4096,)
    xi = arrays["l18_mean_xi"]         # (6,)

with np.load(trajectory_file, allow_pickle=False) as arrays:
    steps = arrays["generated_tokens_processed"]
    path = arrays["xi_last"]           # (n_puntos, 6), bloque 18 / last

print(record["text"])
print(a_l.shape, xi.shape, path.shape)
```

`l18_mean_xi` y `xi_last` usan **resúmenes diferentes**. Para comparar el primer
punto de la trayectoria con la captura inicial debe usarse `l18_last_xi`.

## Integridad y mantenimiento

`python scripts/validate_dataset.py` comprueba hashes, cobertura de las 400
combinaciones, enlaces entre archivos, arrays finitos, formas, proyecciones PCA
y correspondencia del visor con los registros. No modifica datos.

`python scripts/build_viewer.py` regenera únicamente `viewer/index.html` a partir
de los JSON y de `viewer/template.html`, sin inferencia ni red.

Los datos importados y el código congelado se conservan sin cambios. Los análisis
nuevos deben ir en archivos o carpetas nuevas. El manifiesto verifica integridad
respecto de esta copia; no constituye una firma ni una certificación externa.

## Alcance y procedencia

Los escenarios proceden de [School of Reward Hacks](https://arxiv.org/abs/2508.17511).
Las revisiones exactas de los dos checkpoints están en `protocol.json` y en
[docs/PROVENANCE.md](docs/PROVENANCE.md). La ficha pública del SFT no acredita que
sea el checkpoint exacto del paper.

Los indicadores automáticos son locales a cada tarea. `null` significa pendiente,
no cero ni ausencia de RH. Hay 45 respuestas truncadas y 298 registros que requieren
revisión. Las coordenadas PCA no son, por sí mismas, un score de alineamiento.

No se ha añadido una licencia general nueva al código o a los datos propios.
Esa elección queda pendiente antes de su reutilización pública; los recursos de
terceros conservan sus condiciones y atribuciones respectivas.
