# Procedencia y recursos externos

## Checkpoints

| Condición | Repositorio | Revisión |
| --- | --- | --- |
| Original | [unsloth/Qwen3-8B](https://huggingface.co/unsloth/Qwen3-8B) | `946bc9ac74a6c1f8cf012497c503a119b2fcf2eb` |
| SFT | [longtermrisk/Qwen3-8B-school-of-reward-hacks-sft](https://huggingface.co/longtermrisk/Qwen3-8B-school-of-reward-hacks-sft) | `28d32d8626b1f211c7d207bc85ae59bac1e3e8c8` |

Los pesos no se redistribuyen aquí. El nombre público del SFT no demuestra
identidad con el checkpoint del paper. Las condiciones y revisiones utilizadas
están congeladas en el protocolo.

## Escenarios y calibración

- Paper de referencia: [School of Reward Hacks, arXiv:2508.17511](https://arxiv.org/abs/2508.17511).
- Los diez escenarios se transcribieron de las tablas 6 a 10, conservando el
  inglés y normalizando comillas ASCII e indentación Python. El protocolo
  identifica la tabla de cada prompt.
- Fuente de calibración: [longtermrisk/school-of-reward-hacks](https://huggingface.co/datasets/longtermrisk/school-of-reward-hacks),
  revisión `d7e04a550119cb5410494cf90e2313284a5f2148`.
- El protocolo registra `CC-BY-4.0` para esa fuente de calibración. No se asigna
  por extensión esa licencia a todo el código o a todos los materiales propios.
- Se incluyen los 24 prompts de calibración dentro de `protocol.json`, no el
  archivo CSV público completo ni las demostraciones de entrenamiento completas.

## Configuración y límites de reproducción

Se usaron el tokenizer y la plantilla del original en todas las condiciones.
El tokenizer serializado del SFT tiene diferencias de preprocesado; el archivo
`tokenizer_audit.json` registra la comparación. La configuración RoPE del SFT
se normalizó en memoria para usar `rope_theta = 1000000` con la versión local
de Transformers. NF4 usa GPU; el control BF16 distribuye bloques entre GPU y CPU.

`acquisition/frozen_source/` contiene una copia exacta de los archivos archivados
al ejecutar el experimento, incluidos sus requisitos y su plan operativo. Los
tres scripts identificados por `protocol.json` conservan sus hashes originales.
Se archivan como evidencia, no como una orden de volver a ejecutar la corrida.
Sus rutas y supuestos de caché pertenecen al entorno de adquisición y deben
revisarse antes de cualquier reproducción en otro equipo.

Los scripts de `scripts/` son herramientas de lectura y empaquetado del presente
repositorio. No cambian el protocolo ni requieren cargar modelos.

## Publicación

Este repositorio local no incluye una licencia general nueva ni supone que se
haya publicado en GitHub. La selección de licencia y las atribuciones finales
de los materiales propios quedan pendientes de acuerdo entre sus responsables.
