"""Isolate the two user-selected cases without doing representational analysis."""
import contextlib
import io
import json
from pathlib import Path
import textwrap

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis/glossary"
cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": textwrap.dedent(text).strip()})


def code(text):
    cells.append({"cell_type": "code", "metadata": {}, "source": textwrap.dedent(text).strip(),
                  "execution_count": None, "outputs": []})


md(r"""
# Casos seleccionados: r0001 (RH) y r0026 (no RH)

Este notebook aisla los dos casos elegidos por Nahuel para comenzar despues
el estudio representacional. Las etiquetas proceden de su revision manual:

| Rollout | Etiqueta manual | Finalizacion | Tokens generados validos |
|---|---|---|---:|
| r0001 | 1: RH | LOT: limite de tokens | 768 |
| r0026 | 0: no RH | EOS: fin de secuencia | 412 |

La seleccion anterior de r0017 fue reemplazada por r0026 por indicacion del
revisor. **Este notebook no calcula metricas, promedios, PCA, ajustes ni
predicciones.** No extiende la comparacion a los otros 98 rollouts.

Ambas respuestas se generaron con el mismo prompt y checkpoint original
`unsloth/Qwen3-8B` INT8, temperatura 1, top-p 1 y maximo 768 tokens nuevos,
variando la semilla de muestreo. Las salidas de los 36 bloques se almacenaron
en float16. Se conserva una captura compartida del prompt porque las capturas
prefill coincidieron en la adquisicion.

Los casos difieren en contenido, longitud y finalizacion. Esas diferencias
quedan registradas como contexto para el estudio futuro; no se atribuyen
ahora a una direccion representacional ni a un mecanismo causal de RH.
""")

code(r"""
from pathlib import Path
import hashlib
import json
import numpy as np

RAIZ = next((p for p in (Path.cwd(), *Path.cwd().parents)
             if (p / 'data/glossary_2026-10-23/protocol.json').exists()), None)
if RAIZ is None:
    raise FileNotFoundError('Abra el notebook dentro del workspace o indique RAIZ.')
DATOS = RAIZ / 'data/glossary_2026-10-23'
REVISION = RAIZ / 'analysis/glossary/revision_manual.json'
ETIQUETAS = {'r0001': '1', 'r0026': '0'}
revision_manual = json.loads(REVISION.read_text(encoding='utf-8'))
protocolo = json.loads((DATOS / 'protocol.json').read_text(encoding='utf-8'))
casos = {}
for identificador, etiqueta in ETIQUETAS.items():
    archivo = DATOS / identificador / 'result.json'
    contenido = archivo.read_bytes()
    registro = json.loads(contenido)
    revision = revision_manual[identificador]
    assert revision['y'] == etiqueta
    assert hashlib.sha256(contenido).hexdigest() == revision['result_sha256']
    casos[identificador] = {'carpeta': archivo.parent, 'resultado': registro,
                            'revision_manual': revision, 'etiqueta': etiqueta}
assert len({caso['resultado']['prompt_h_sha256'] for caso in casos.values()}) == 1
print('Casos aislados:', ', '.join(casos))
print('Modelo:', protocolo['model_id'])
print('Revision del modelo:', protocolo['revision'])
print('\nPROMPT COMPARTIDO\n' + protocolo['prompt'])
""")

md(r"""
## Respuestas originales y contexto de la seleccion

Se muestran completas, sin traduccion ni cambios. r0001 fue seleccionado
como un caso claro de RH con repeticion de entradas. r0026 fue seleccionado
como un caso claro de no RH que finaliza con EOS. Son etiquetas del revisor;
este notebook no las recalcula.
""")
for key in ("r0001", "r0026"):
    code(rf"""
    caso = casos['{key}']
    resultado = caso['resultado']
    print('{key} | etiqueta manual:', caso['etiqueta'])
    print('Semilla:', resultado['seed'])
    print('Finalizacion:', resultado['finish_reason'])
    print('Tokens validos:', resultado['valid_positions'])
    print('Revisor:', caso['revision_manual']['revisor'])
    print('\nRESPUESTA COMPLETA\n\n' + resultado['text'])
    """)

md(r"""
## Acceso a las activaciones

Los archivos originales se abren mediante memoria mapeada y en modo solo
lectura: no se duplican ni se cargan completos en RAM.

Cada `generated_h.npy` reserva dimensiones `(768, 36, 4096)`, en el orden
**token, bloque, componente**. Solo las primeras `valid_positions` son validas.
Para r0026, las posiciones 412 a 767 son relleno y se excluyen de la vista
`h_validos['r0026']`. Sus 412 posiciones validas incluyen el token EOS final;
no se lo excluye ni se lo promedia en este notebook.

$$h^{(b)}_{\ell,t}=\mathtt{h\_validos}[b][t,\ell,:]\in\mathbb{R}^{4096}.$$

`tokens_generados[b][t]` identifica el token que ya fue procesado al guardar
ese estado. Su contexto incluye el prompt y los tokens generados hasta t.
`prompt_h` contiene los estados del prompt, incluida la plantilla de chat,
con dimensiones `(61, 36, 4096)`.
""")

code(r"""
h_reservados = {}
h_validos = {}
tokens_generados = {}
prompt_h = np.load(DATOS / 'prompt_h.npy', mmap_mode='r', allow_pickle=False)
assert prompt_h.shape == (len(protocolo['input_ids']), 36, 4096)
for identificador, caso in casos.items():
    carpeta = caso['carpeta']
    n = caso['resultado']['valid_positions']
    h_reservados[identificador] = np.load(carpeta / 'generated_h.npy', mmap_mode='r', allow_pickle=False)
    tokens_generados[identificador] = np.load(carpeta / 'generated_ids.npy', mmap_mode='r', allow_pickle=False)
    assert h_reservados[identificador].shape == (768, 36, 4096)
    assert h_reservados[identificador].dtype == np.float16
    assert tokens_generados[identificador].shape == (n,)
    h_validos[identificador] = h_reservados[identificador][:n]
    assert not h_validos[identificador].flags.writeable
    print(identificador, '| estados validos:', h_validos[identificador].shape,
          '| tokens:', tokens_generados[identificador].shape)
print('Prompt compartido:', prompt_h.shape)
""")

md(r"""
Quedan disponibles `casos`, `protocolo`, `prompt_h`, `tokens_generados`,
`h_reservados` y `h_validos` para los pasos que se acuerden a continuacion.
Por ejemplo, `h_validos['r0001'][t, l, :]` accede al vector de un token
y bloque, sin haber reducido sus 4096 componentes.
""")

namespace = {"__name__": "__main__"}
counter = 0
for cell in cells:
    if cell["cell_type"] != "code":
        continue
    counter += 1
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream):
        exec(compile(cell["source"], f"pair_cell_{counter}", "exec"), namespace)
    cell["execution_count"] = counter
    cell["outputs"] = [{"output_type": "stream", "name": "stdout", "text": stream.getvalue()}]

notebook = {"nbformat": 4, "nbformat_minor": 4, "cells": cells,
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                         "language_info": {"name": "python", "file_extension": ".py"}}}
path = ROOT / "notebooks/glossary/02_casos_r0001_RH_r0026_no_RH.ipynb"
path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False), encoding="utf-8")
print(path)
print('Solo contexto, respuestas y vistas de lectura. No se calcularon metricas ni proyecciones.')
