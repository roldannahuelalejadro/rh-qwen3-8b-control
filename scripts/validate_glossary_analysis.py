"""Validate published glossary artifacts without downloading hidden states."""
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data/glossary_2026-10-23'
ANALYSIS=ROOT/'analysis/glossary'
manual=json.loads((ANALYSIS/'revision_manual.json').read_text(encoding='utf-8'))
assert set(manual)=={f'r{i:04d}' for i in range(1,101)}
assert Counter(r['y'] for r in manual.values())==Counter({'1':51,'0':21,'X':28})
assert manual==json.loads((ANALYSIS/'revision_manual_Nahuel_2026-10-24_original.json').read_text(encoding='utf-8'))
for name,review in manual.items():
    path=DATA/name/'result.json'
    assert hashlib.sha256(path.read_bytes()).hexdigest()==review['result_sha256']
    r=json.loads(path.read_text(encoding='utf-8'))
    tokens=np.load(path.parent/'generated_ids.npy',allow_pickle=False)
    assert tokens.shape==(r['valid_positions'],)
    assert hashlib.sha256((path.parent/'generated_ids.npy').read_bytes()).hexdigest()==r['files_sha256']['generated_ids.npy']
    assert r['rh_label'] is None
manifest=json.loads((ANALYSIS/'publication_manifest.json').read_text(encoding='utf-8'))
for relative,expected in manifest['files'].items():
    assert hashlib.sha256((ROOT/relative).read_bytes()).hexdigest()==expected,relative
for name,entry in manifest['sources'].items():
    path=ROOT/'scripts'/name
    assert hashlib.sha256(path.read_bytes()).hexdigest()==entry['published_sha256'],name
for path in (ROOT/'notebooks/glossary').glob('*.ipynb'):
    nb=json.loads(path.read_text(encoding='utf-8'))
    assert nb['nbformat']==4
    for cell in nb['cells']:
        if cell['cell_type']=='code':
            source=cell['source']
            source=''.join(source) if isinstance(source,list) else source
            assert 'output/cloud_glossary_2026-10-23' not in source
            assert 'output/analisis_glosario_100' not in source
            compile(source,path.name,'exec')
folds=json.loads((ANALYSIS/'decoder_v1/folds_and_hyperparameters.json').read_text())['outer']
eligible={name for name,r in manual.items() if r['y']!='X'}
seen=[]
for fold in folds:
    tr,te=set(fold['training']),set(fold['test'])
    assert not tr&te
    assert tr|te==eligible
    seen.extend(te)
assert len(seen)==len(set(seen))==72
with (ANALYSIS/'clasificacion_100.csv').open(encoding='utf-8-sig') as f:
    rows=list(csv.DictReader(f))
assert len(rows)==100 and all(r['y_manual']==manual[r['rollout']]['y'] for r in rows)
print('OK: 100 manual labels and response/token hashes, publication manifest, four notebooks, and disjoint rollout folds.')
