"""Check saved cubic tensors, dense horizons and intercept experiments."""
import csv
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'analysis/glossary/decoder_v2_checks'
def read(name):
    return list(csv.DictReader((OUT/name).open(encoding='utf-8-sig')))
cases=read('case_curves_every_token_oof.csv')
manual=json.loads((OUT.parent/'revision_manual.json').read_text(encoding='utf-8'))
for name,n in (('r0001',768),('r0026',412)):
    case=[r for r in cases if r['rollout']==name]
    assert [int(r['j']) for r in case]==list(range(n+1))
    assert all(r['y']==manual[name]['y'] for r in case)
    assert all((r['extrapolation']=='True')==(int(r['j'])>192) for r in case)
    with np.load(OUT/f'{name}_Xi_dense.npz',allow_pickle=False) as f:
        x=f['Xi'][0]
        assert f['Xi'].shape==(1,n+1,36)
        fold=int(f['fold'])
    with np.load(OUT/f'fold{fold}_encoder.npz',allow_pickle=False) as enc:
        z=(x[[0,32,64,128,192]]-enc['score_mean'])/enc['score_scale']
    with np.load(OUT/f'fold{fold}_cubic.npz',allow_pickle=False) as params:
        tensor=params['T_standardized']
        assert tensor.shape==(36,36,36)
        assert np.allclose(tensor,tensor.swapaxes(0,1))
        assert np.allclose(tensor,tensor.swapaxes(1,2))
        direct=params['b0_standardized']+(np.prod(z[:,None,:]**params['powers'][None,:,:],axis=2)@params['polynomial_coefficients'])
        reconstructed=params['b0_standardized']+z@params['g_standardized']+.5*np.einsum('ni,ij,nj->n',z,params['H_standardized'],z)
        reconstructed+=np.einsum('ni,ijk,nj,nk->n',z,tensor,z,z,optimize=True)/6
        assert np.allclose(direct,reconstructed,atol=1e-8)
        expected=np.array([float(case[j]['cubic']) for j in (0,32,64,128,192)])
        assert np.allclose(1/(1+np.exp(-direct)),expected,atol=1e-8)

predictions=read('predictions_dense_oof.csv')
assert len(predictions)==72*193
assert all(manual[r['rollout']]['y'] in ('0','1') for r in predictions)
for name in {r['rollout'] for r in predictions}:
    assert sorted(int(r['j']) for r in predictions if r['rollout']==name)==list(range(193))
old=read('../decoder_v1/predictions_oof.csv')
by_key={(r['rollout'],int(r['j'])):r for r in predictions}
maximum=0.
for r in old:
    new=by_key[r['rollout'],int(r['j'])]
    for kind in ('constant','linear','diagonal','quadratic'):
        maximum=max(maximum,abs(float(new[kind])-float(r[kind])))
assert maximum<1e-4
summary=json.loads((OUT/'summary.json').read_text())
assert summary['maximum_probability_difference_between_initializations']<1e-3
traces=read('intercept_convergence.csv')
for fold in (1,2,3):
    for mode in ('initialized','fixed'):
        for beta in (0.,.25,.5,.75,1.):
            group=[r for r in traces if int(r['fold'])==fold and r['mode']==mode and float(r['initial'])==beta]
            assert group
            assert float(group[-1]['objective'])<=float(group[0]['objective'])
            if mode=='fixed':assert all(float(r['beta'])==beta for r in group)
curves=read('case_intercept_curves_every_token.csv')
for r in curves:
    if r['mode']=='fixed' and int(r['j'])==0:
        expected=1/(1+np.exp(-float(r['initial_or_fixed_beta'])))
        assert abs(float(r['probability'])-expected)<2e-6
notebook=json.loads((ROOT/'notebooks/glossary/04_ajuste_trayectorias_constante_cubico.ipynb').read_text(encoding='utf-8'))
for cell in notebook['cells']:
    if cell['cell_type']=='code':
        source=cell['source']
        compile(''.join(source) if isinstance(source,list) else source,'notebook','exec')
print('OK: every-token horizons, Xi axes, X exclusion, cubic symmetry/factorials/probabilities, intercept constraints, convergence and notebook.')
print('Maximum reproduction difference versus v1:',maximum)
