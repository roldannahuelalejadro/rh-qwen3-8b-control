"""Refresh report presentation using saved fits; do not rerun training."""
import base64
import csv
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from glossary_decoder_extensions import case_intercept_curves,figures

OUT=ROOT/'analysis/glossary/decoder_v2_checks'
PATH=ROOT/'notebooks/glossary/04_ajuste_trayectorias_constante_cubico.ipynb'
for name in ('r0001','r0026'):
    path=OUT/f'{name}_Xi_dense.npz'
    with np.load(path,allow_pickle=False) as f:
        saved={key:f[key] for key in f.files}
    if saved['Xi'].shape==(1,36,len(saved['j'])):
        saved['Xi']=saved['Xi'].transpose(0,2,1)
        np.savez_compressed(path,**saved)
    assert saved['Xi'].shape==(1,len(saved['j']),36)
def read(name):
    return list(csv.DictReader((OUT/name).open(encoding='utf-8-sig')))
metrics=read('metrics_dense_oof.csv')
for r in metrics:
    r['j']=int(r['j'])
    for k in ('AUROC','log_loss','Brier','balanced_accuracy'):r[k]=float(r[k])
intercept=read('metrics_intercept_oof.csv')
for r in intercept:
    for k in ('j','initial_or_fixed_beta','AUROC','Brier','log_loss'):r[k]=float(r[k])
traces=read('intercept_convergence.csv')
for r in traces:
    for k in ('fold','initial','iteration','objective','beta','max_abs_gradient'):r[k]=float(r[k])
cases=read('case_curves_every_token_oof.csv')
for r in cases:
    r['j']=int(r['j']);r['extrapolation']=r['extrapolation']=='True'
    for k in ('constant','linear','diagonal','quadratic','cubic'):r[k]=float(r[k])
result=dict(metrics=metrics,intercept_metrics=intercept,traces=traces,cases=cases,
            intercept_case_curves=case_intercept_curves(OUT))
paths=figures(result,OUT)
notebook=json.loads(PATH.read_text(encoding='utf-8'))
notebook['cells']=[c for c in notebook['cells'] if not any('image/png' in o.get('data',{}) for o in c.get('outputs',[]))]
for p in paths:
    source=f"from IPython.display import Image, display\ndisplay(Image(filename=str(SALIDA/'{p.name}')))"
    compile(source,'figure','exec')
    notebook['cells'].append(dict(cell_type='code',metadata={},execution_count=None,source=source,
        outputs=[dict(output_type='display_data',metadata={},data={
            'image/png':base64.b64encode(p.read_bytes()).decode(),'text/plain':p.name})]))
notebook['cells'].append(dict(cell_type='markdown',metadata={},source='''## Lectura de esta corrida

Las curvas de ambos casos contienen todos los j; el entrenamiento sigue usando
cinco prefijos. A partir de j=193 se muestra extrapolacion temporal, no evaluacion
en un horizonte usado durante el entrenamiento. Los dos casos son ejemplos
seleccionados y no sustituyen la evaluacion sobre los 72 rollouts.

Las cinco inicializaciones convergieron con una diferencia maxima de probabilidad
de 1.35e-5 (entre 60 y 511 iteraciones). Esto respalda insensibilidad a la
inicializacion en esta optimizacion, no ausencia de sesgo de muestreo. La constante
fija cambia el problema: su influencia por token se muestra en la quinta figura.

En j=192, AUROC/log-loss son: diagonal 0.690/0.553, completo 0.670/0.583 y
cubico 0.681/0.563. El cubico mejora frente al completo, pero no supera al
diagonal en esas dos metricas. Su exactitud balanceada a umbral 0.5 es 0.580,
frente a 0.532 diagonal: los criterios no ordenan exactamente igual los modelos.
La muestra y el esquema de seguimiento no justifican declarar necesario el
tensor cubico. No hay prueba de generalizacion entre prompts.
'''))
PATH.write_text(json.dumps(notebook,indent=1,ensure_ascii=False),encoding='utf-8')
print(PATH)
