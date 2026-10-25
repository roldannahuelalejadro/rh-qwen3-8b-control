"""Independent checks of exported decoder parameters, grouping and prefix means."""
import csv
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'analysis/glossary/decoder_v1'
DATA = ROOT / 'data/glossary_2026-10-23'
manual = json.loads((OUT.parent / 'revision_manual.json').read_text(encoding='utf-8'))
steps = (0, 32, 64, 128, 192)
classified = {name for name, r in manual.items() if r['y'] != 'X'}
folds = json.loads((OUT / 'folds_and_hyperparameters.json').read_text())['outer']
seen = []
for fold in folds:
    tr, te = set(fold['training']), set(fold['test'])
    assert not tr & te
    assert tr | te == classified
    seen.extend(te)
assert len(seen) == len(set(seen)) == 72
assert set(seen) == classified

raw = np.load(OUT / 'a_lj.npy', mmap_mode='r', allow_pickle=False)
prompt = np.load(DATA / 'prompt_h.npy', mmap_mode='r', allow_pickle=False)
for name in ('r0001', 'r0026'):
    b = int(name[1:]) - 1
    hidden = np.load(DATA / name / 'generated_h.npy', mmap_mode='r', allow_pickle=False)
    for t, j in enumerate(steps):
        direct = (prompt.sum(axis=0, dtype=np.float64) + hidden[:j].sum(axis=0, dtype=np.float64)) / (len(prompt) + j)
        assert np.array_equal(raw[b, t], direct.astype(np.float32))

xi = np.load(OUT / 'Xi_final.npy', mmap_mode='r', allow_pickle=False)
assert xi.shape == (100, 1, 5, 36)
assert np.array_equal(xi[:, :, 0], np.broadcast_to(xi[0, :, 0], xi[:, :, 0].shape))
rows = list(csv.DictReader((OUT / 'predictions_final_in_sample_or_X.csv').open(encoding='utf-8-sig')))
for kind in ('linear', 'diagonal', 'quadratic'):
    params = np.load(OUT / f'decoder_{kind}.npz', allow_pickle=False)
    assert np.allclose(params['H'], params['H'].T)
    x = np.asarray(xi[:, 0]).reshape(-1, 36)
    logits = params['b0'] + x @ params['g'] + 0.5*np.einsum('ni,ij,nj->n', x, params['H'], x)
    predicted = 1 / (1 + np.exp(-logits))
    assert np.allclose(predicted, [float(r[kind]) for r in rows], atol=1e-8)
    if kind == 'linear':
        assert not np.any(params['H'])
    if kind == 'diagonal':
        assert not np.any(params['H'] - np.diag(np.diag(params['H'])))

oof = list(csv.DictReader((OUT / 'predictions_oof.csv').open(encoding='utf-8-sig')))
assert len(oof) == 72 * 5
for row in oof:
    assert row['rollout'] in classified
    assert row['y'] == manual[row['rollout']]['y']
    assert row['rollout'] in folds[int(row['fold']) - 1]['test']
notebook = json.loads((ROOT / 'notebooks/glossary/03_decoder_PCA_cuadratico.ipynb').read_text(encoding='utf-8'))
for cell in notebook['cells']:
    if cell['cell_type'] == 'code':
        source = cell['source']
        compile(''.join(source) if isinstance(source, list) else source, 'notebook_cell', 'exec')
print('OK: rollout disjoint folds, X excluded, literal cumulative means, Xi order, shared initial code, symmetric H, recovered probabilities, notebook code.')
