"""CPU-only cumulative-state decoder with rollout-level nested validation."""
from pathlib import Path
from collections import Counter
import csv
import hashlib
import json
import platform
import sys

import numpy as np
import psutil
import sklearn
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             brier_score_loss, log_loss, roc_auc_score)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import PolynomialFeatures
from threadpoolctl import threadpool_limits

STEPS = (0, 32, 64, 128, 192)
CS = (0.001, 0.01, 0.1, 1.0)
KINDS = ('constant', 'linear', 'diagonal', 'quadratic')
SEED = 20261025


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def write_csv(path, rows):
    with Path(path).open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_context(root):
    data = root / 'data/glossary_2026-10-23'
    manual = json.loads((root / 'analysis/glossary/revision_manual.json').read_text(encoding='utf-8'))
    protocol = json.loads((data / 'protocol.json').read_text(encoding='utf-8'))
    records, labels, fingerprints = [], [], {}
    for i in range(1, 101):
        name = f'r{i:04d}'
        path = data / name / 'result.json'
        r = json.loads(path.read_text(encoding='utf-8'))
        digest = sha(path)
        assert manual[name]['result_sha256'] == digest
        assert manual[name]['y'] in ('0', '1', 'X')
        n_content = r['valid_positions'] - int(r['finish_reason'] == 'eos')
        assert n_content >= max(STEPS), f'{name} ends before the fixed analysis horizon.'
        records.append(dict(r, rollout=name))
        labels.append(manual[name]['y'])
        fingerprints[name] = digest
    assert manual['r0001']['y'] == '1' and manual['r0026']['y'] == '0'
    assert len({r['prompt_h_sha256'] for r in records}) == 1
    return data, protocol, records, np.array(labels), fingerprints


def cumulative_states(root, out):
    out.mkdir(parents=True, exist_ok=True)
    data, protocol, records, labels, fingerprints = load_context(root)
    manifest = {'version': 'cumulative-prefix-v1', 'steps': list(STEPS),
                'protocol_sha256': sha(data / 'protocol.json'), 'response_hashes': fingerprints}
    cache, metadata = out / 'a_lj.npy', out / 'a_lj_manifest.json'
    if cache.exists() and metadata.exists() and json.loads(metadata.read_text()) == manifest:
        print('Reusing verified-prefix metadata cache.', flush=True)
    else:
        import shutil
        if psutil.virtual_memory().available < 2 * 1024**3 or shutil.disk_usage(out).free < 1024**3:
            raise RuntimeError('Need at least 2 GiB available RAM and 1 GiB free disk.')
        prompt = np.load(data / 'prompt_h.npy', mmap_mode='r', allow_pickle=False)
        assert prompt.shape == (len(protocol['input_ids']), 36, 4096)
        prompt_sum = prompt.sum(axis=0, dtype=np.float64)
        t_prompt = len(prompt)
        temporary = out / 'a_lj.partial.npy'
        result = np.lib.format.open_memmap(temporary, mode='w+', dtype=np.float32,
                                           shape=(100, len(STEPS), 36, 4096))
        for b, r in enumerate(records):
            h = np.load(data / r['rollout'] / 'generated_h.npy', mmap_mode='r', allow_pickle=False)
            assert h.shape == (768, 36, 4096) and h.dtype == np.float16
            total, previous = prompt_sum.copy(), 0
            for k, j in enumerate(STEPS):
                if j > previous:
                    total += h[previous:j].sum(axis=0, dtype=np.float64)
                result[b, k] = total / (t_prompt + j)
                assert np.isfinite(result[b, k]).all()
                previous = j
            del h
            if (b + 1) % 10 == 0:
                result.flush()
                print(f'Cumulative states: {b + 1}/100; RSS {psutil.Process().memory_info().rss/1024**3:.2f} GiB', flush=True)
        result.flush()
        del result
        temporary.replace(cache)
        dump(metadata, manifest)
    raw = np.load(cache, mmap_mode='r', allow_pickle=False)
    assert raw.shape == (100, len(STEPS), 36, 4096)
    assert np.array_equal(raw[:, 0], np.broadcast_to(raw[0, 0], raw[:, 0].shape))
    dump(out / 'protocol_analysis.json', {
        'steps': list(STEPS), 'K_per_layer': 1, 'layers': list(range(36)),
        'coordinate_order': 'Xi[i,j,ell]; flatten fixed-j in i,ell order; K=1 here',
        'label_source': 'Nahuel manual 0/1; X excluded from supervised fitting',
        'labels': dict(Counter(labels.tolist())), 'seed': SEED,
        'outer_folds': 3, 'inner_folds': 3, 'C_grid': list(CS),
        'models': list(KINDS), 'time_parameters': 'same b0,g,H at every j',
        'PCA': 'one component per layer; fit using only training rollouts and fixed prefixes',
        'standardization': 'training-only mean/std of PCA coordinates, before polynomial expansion',
        'primary_horizon': 192, 'original_arrays_changed': False,
        'python': platform.python_version(), 'numpy': np.__version__, 'sklearn': sklearn.__version__,
        'code_sha256': sha(Path(__file__)),
    })
    return raw, labels, records


def fit_encoder(raw, training):
    centers, directions, variance = [], [], []
    for layer in range(36):
        x = np.asarray(raw[training, :, layer, :]).reshape(-1, 4096)
        pca = PCA(n_components=1, svd_solver='randomized', iterated_power=2, random_state=SEED)
        pca.fit(x)
        centers.append(pca.mean_)
        directions.append(pca.components_[0])
        variance.append(pca.explained_variance_ratio_[0])
    enc = {'centers': np.asarray(centers), 'directions': np.asarray(directions),
           'variance_ratio': np.asarray(variance)}
    x = project(raw, training, enc).reshape(-1, 36)
    enc['score_mean'] = x.mean(axis=0)
    enc['score_scale'] = x.std(axis=0)
    enc['score_scale'][enc['score_scale'] < 1e-12] = 1.0
    return enc


def project(raw, indices, enc):
    x = np.empty((len(indices), len(STEPS), 36), dtype=np.float64)
    for layer in range(36):
        vectors = np.asarray(raw[indices, :, layer, :])
        x[:, :, layer] = (vectors - enc['centers'][layer]) @ enc['directions'][layer]
    return x


def standardize(x, enc):
    return ((x - enc['score_mean']) / enc['score_scale']).reshape(-1, 36)


def features(z, kind):
    if kind == 'linear':
        return z
    if kind == 'diagonal':
        return np.concatenate([z, z*z], axis=1)
    return PolynomialFeatures(degree=2, include_bias=False).fit_transform(z)


def train(z, y, kind, c):
    model = LogisticRegression(C=c, penalty='l2', solver='liblinear', max_iter=3000,
                               random_state=SEED, intercept_scaling=100)
    model.fit(features(z, kind), y)
    return model


def prepare_inner(raw, labels, training, seed):
    prepared = []
    splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    y = labels[training].astype(int)
    for train_idx, valid_idx in splitter.split(training, y):
        tr, va = training[train_idx], training[valid_idx]
        enc = fit_encoder(raw, tr)
        prepared.append((standardize(project(raw, tr, enc), enc), np.repeat(labels[tr].astype(int), len(STEPS)),
                         standardize(project(raw, va, enc), enc), np.repeat(labels[va].astype(int), len(STEPS))))
    return prepared


def tune(prepared, kind):
    losses = []
    for c in CS:
        observed, predicted = [], []
        for ztr, ytr, zva, yva in prepared:
            model = train(ztr, ytr, kind, c)
            observed.extend(yva.tolist())
            predicted.extend(model.predict_proba(features(zva, kind))[:, 1].tolist())
        losses.append(float(log_loss(observed, predicted, labels=[0, 1])))
    index = int(np.argmin(losses))
    return CS[index], dict(zip(map(str, CS), losses))


def coefficients(model, kind, enc):
    d = 36
    c = model.coef_[0]
    g, h = c[:d].copy(), np.zeros((d, d))
    if kind == 'diagonal':
        np.fill_diagonal(h, 2*c[d:])
    elif kind == 'quadratic':
        index = d
        for i in range(d):
            for k in range(i, d):
                h[i, k] = h[k, i] = c[index] * (2 if i == k else 1)
                index += 1
        assert index == len(c)
    b = float(model.intercept_[0])
    inv_scale = 1 / enc['score_scale']
    offset = -enc['score_mean'] * inv_scale
    raw_h = h * inv_scale[:, None] * inv_scale[None, :]
    raw_g = inv_scale * (g + h @ offset)
    raw_b = b + g @ offset + 0.5 * offset @ h @ offset
    return dict(b0_standardized=np.array(b), g_standardized=g, H_standardized=h,
                b0=np.array(raw_b), g=raw_g, H=raw_h)


def evaluate(raw, labels, records, out):
    classified = np.flatnonzero(labels != 'X')
    y = labels[classified].astype(int)
    predictions = {kind: np.full((len(classified), len(STEPS)), np.nan) for kind in KINDS}
    outer_assignment = np.full(len(classified), -1)
    folds = []
    splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    with threadpool_limits(limits=2):
        for fold, (tr_local, te_local) in enumerate(splitter.split(classified, y)):
            training, testing = classified[tr_local], classified[te_local]
            print(f'Outer fold {fold+1}/3: {len(training)} training, {len(testing)} test rollouts', flush=True)
            prepared = prepare_inner(raw, labels, training, SEED + fold + 1)
            enc = fit_encoder(raw, training)
            ztr = standardize(project(raw, training, enc), enc)
            zte = standardize(project(raw, testing, enc), enc)
            ytr = np.repeat(labels[training].astype(int), len(STEPS))
            predictions['constant'][te_local] = float(ytr.mean())
            outer_assignment[te_local] = fold + 1
            fold_record = {'fold': fold+1, 'training': [records[i]['rollout'] for i in training],
                           'test': [records[i]['rollout'] for i in testing], 'selection': {}}
            for kind in KINDS[1:]:
                c, losses = tune(prepared, kind)
                model = train(ztr, ytr, kind, c)
                predictions[kind][te_local] = model.predict_proba(features(zte, kind))[:, 1].reshape(-1, len(STEPS))
                fold_record['selection'][kind] = {'C': c, 'inner_log_loss': losses}
            folds.append(fold_record)
        assert all(np.isfinite(p).all() for p in predictions.values())
        metric_rows = []
        for kind, probabilities in predictions.items():
            for time_idx, j in enumerate(STEPS):
                p = probabilities[:, time_idx]
                metric_rows.append(dict(model=kind, j=j, n=len(y),
                    AUROC=float(roc_auc_score(y, p)), Brier=float(brier_score_loss(y, p)),
                    log_loss=float(log_loss(y, p, labels=[0, 1])),
                    accuracy=float(accuracy_score(y, p >= .5)),
                    balanced_accuracy=float(balanced_accuracy_score(y, p >= .5))))
        write_csv(out / 'metrics_oof.csv', metric_rows)
        oof_rows = []
        for local, idx in enumerate(classified):
            for t, j in enumerate(STEPS):
                oof_rows.append(dict(rollout=records[idx]['rollout'], y=int(y[local]), j=j,
                    fold=int(outer_assignment[local]), **{kind: float(predictions[kind][local, t]) for kind in KINDS}))
        write_csv(out / 'predictions_oof.csv', oof_rows)
        print('Fitting final models on all 72 classified rollouts; these are not held-out scores.', flush=True)
        prepared = prepare_inner(raw, labels, classified, SEED + 100)
        enc = fit_encoder(raw, classified)
        x_all = project(raw, np.arange(100), enc)
        z_all = standardize(x_all, enc)
        ztr = standardize(x_all[classified], enc)
        ytr = np.repeat(y, len(STEPS))
        final = {'constant': np.full((100, len(STEPS)), float(y.mean()))}
        parameter_sets, selections = {}, {}
        for kind in KINDS[1:]:
            c, losses = tune(prepared, kind)
            model = train(ztr, ytr, kind, c)
            params = coefficients(model, kind, enc)
            xflat = x_all.reshape(-1, 36)
            logits = params['b0'] + xflat @ params['g'] + 0.5*np.einsum('ni,ij,nj->n', xflat, params['H'], xflat)
            assert np.allclose(logits, model.decision_function(features(z_all, kind)), atol=1e-7)
            p = model.predict_proba(features(z_all, kind))[:, 1].reshape(100, len(STEPS))
            assert np.max(p[:, 0]) - np.min(p[:, 0]) < 1e-12
            final[kind], parameter_sets[kind] = p, params
            selections[kind] = {'C': c, 'inner_log_loss': losses}
            np.savez_compressed(out / f'decoder_{kind}.npz', **params)
        np.savez_compressed(out / 'encoder_PCA.npz', **enc)
        # K=1: convert [rollout,j,ell] into [rollout,i,j,ell].
        np.save(out / 'Xi_final.npy', x_all[:, None, :, :], allow_pickle=False)
        write_csv(out / 'predictions_final_in_sample_or_X.csv', [dict(
            rollout=r['rollout'], label=labels[b], j=j,
            usage='X exploratory' if labels[b] == 'X' else 'in-sample; not evaluation',
            **{kind: float(final[kind][b, t]) for kind in KINDS})
            for b, r in enumerate(records) for t, j in enumerate(STEPS)])
        dump(out / 'folds_and_hyperparameters.json', {'outer': folds, 'final': selections})
        summary = {'n_classified': len(classified), 'n_X_excluded': int(sum(labels == 'X')),
            'steps': list(STEPS), 'K': 1, 'L': 36, 'd': 36,
            'quadratic_parameters_including_intercept': 703,
            'same_parameters_at_all_steps': True,
            'primary_metrics_j192': [r for r in metric_rows if r['j'] == 192],
            'final_shared_initial_probabilities': {kind: float(p[0, 0]) for kind, p in final.items()},
            'note': 'All evaluation is out-of-fold by rollout. Final fitted coefficients use all classified data.'}
        dump(out / 'summary.json', summary)
    return dict(metrics=metric_rows, predictions=predictions, classified=classified, y=y,
                final=final, parameters=parameter_sets, encoder=enc, summary=summary,
                records=records, labels=labels)


def figures(result, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                         'figure.dpi': 140, 'savefig.dpi': 180, 'font.family': 'DejaVu Sans'})
    paths = []
    colors = {'constant': '#72787b', 'linear': '#147d92', 'diagonal': '#a77c15', 'quadratic': '#b94152'}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for axis, name in zip(axes, ('r0001', 'r0026')):
        idx = next(i for i, r in enumerate(result['records']) if r['rollout'] == name)
        local = int(np.flatnonzero(result['classified'] == idx)[0])
        for kind in KINDS:
            axis.plot(STEPS, result['predictions'][kind][local], 'o-', color=colors[kind], label=kind)
        axis.set(title=f'{name} | manual Y={result["labels"][idx]}', xlabel='j: tokens generados', ylim=(-.03, 1.03))
        axis.grid(alpha=.2)
    axes[0].set_ylabel('Probabilidad de etiqueta RH (fuera de entrenamiento)')
    axes[1].legend(fontsize=8)
    fig.suptitle('Dos casos de referencia: predicciones out-of-fold')
    fig.tight_layout()
    path = out / '01_reference_cases_oof.png'; fig.savefig(path); plt.close(fig); paths.append(path)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for kind in KINDS:
        selected = [r for r in result['metrics'] if r['model'] == kind]
        for axis, metric in zip(axes[:2], ('AUROC', 'log_loss')):
            axis.plot(STEPS, [r[metric] for r in selected], 'o-', label=kind, color=colors[kind])
            axis.set(xlabel='j: tokens generados', ylabel=metric)
            axis.grid(alpha=.2)
    axes[0].axhline(.5, color='#aaa', linestyle=':', linewidth=1)
    rng = np.random.default_rng(SEED)
    t = STEPS.index(192)
    for label, color in ((0, '#147d92'), (1, '#b94152')):
        mask = result['y'] == label
        p = result['predictions']['quadratic'][mask, t]
        axes[2].scatter(np.full(sum(mask), label)+rng.uniform(-.09,.09,sum(mask)), p,
                        s=24, alpha=.8, color=color)
    axes[2].set(xticks=[0,1], xticklabels=['No RH', 'RH'], ylabel='P(RH) del decoder cuadrático',
                title='j=192: cada punto es un rollout', ylim=(-.03,1.03))
    axes[2].axhline(.5, color='#aaa', linestyle=':', linewidth=1)
    axes[1].legend(fontsize=8)
    fig.suptitle('Validación anidada por rollout | 72 respuestas, 28 X excluidas')
    fig.tight_layout()
    path = out / '02_evaluation_oof.png'; fig.savefig(path); plt.close(fig); paths.append(path)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    h = result['parameters']['quadratic']['H_standardized']
    limit = max(float(np.abs(h).max()), 1e-12)
    image = axes[0].imshow(h, cmap='RdBu_r', vmin=-limit, vmax=limit, origin='lower', aspect='equal')
    axes[0].set(title='H del modelo final | coordenadas estandarizadas', xlabel='Capa m (i=0)', ylabel='Capa l (i=0)')
    fig.colorbar(image, ax=axes[0], shrink=.8)
    axes[1].plot(range(36), result['encoder']['variance_ratio'], 'o-', color='#147d92')
    axes[1].set(title='PCA final: varianza retenida por capa', xlabel='Capa l', ylabel='Fracción de varianza (K=1)', ylim=(0,1))
    axes[1].grid(alpha=.2)
    fig.tight_layout()
    path = out / '03_final_H_and_encoder.png'; fig.savefig(path); plt.close(fig); paths.append(path)
    return paths
