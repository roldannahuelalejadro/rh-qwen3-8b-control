"""Dense trajectories, intercept sensitivity, and nested cubic comparison."""
import itertools
import json
import math
from pathlib import Path
import time

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.preprocessing import PolynomialFeatures
from threadpoolctl import threadpool_limits

from glossary_decoder import (STEPS, CS, SEED, KINDS, coefficients, dump, features,
                              fit_encoder, load_context, prepare_inner, project,
                              standardize, train, write_csv)

CONSTANTS = (0., .25, .5, .75, 1.)


def cubic_features(z):
    return PolynomialFeatures(3, include_bias=False).fit_transform(z)


def fit_cubic(z, y, c):
    model = LogisticRegression(C=c, penalty='l2', solver='liblinear', max_iter=3000,
                               random_state=SEED, intercept_scaling=100)
    model.fit(cubic_features(z), y)
    return model


def tune_cubic(prepared):
    losses = []
    for c in CS:
        observed, predictions = [], []
        for ztr, ytr, zva, yva in prepared:
            model = fit_cubic(ztr, ytr, c)
            observed.extend(yva)
            predictions.extend(model.predict_proba(cubic_features(zva))[:, 1])
        losses.append(float(log_loss(observed, predictions, labels=[0, 1])))
    return CS[int(np.argmin(losses))], dict(zip(map(str, CS), losses))


def fit_intercept_test(f, y, c, initial, fixed=False):
    # f = phi(z) - phi(z_prompt): beta is exactly the pre-generation logit.
    alpha = 1 / (c * len(y))
    trace = []
    def objective(theta):
        beta, w = (initial, theta) if fixed else (theta[0], theta[1:])
        scores = beta + f @ w
        residual = expit(scores) - y
        loss = np.mean(np.logaddexp(0, scores) - y*scores) + .5*alpha*(w @ w)
        gradient = f.T @ residual / len(y) + alpha*w
        if not fixed:
            gradient = np.r_[np.mean(residual), gradient]
        return float(loss), gradient
    def record(theta):
        loss, gradient = objective(theta)
        trace.append({'iteration': len(trace), 'objective': loss,
                      'max_abs_gradient': float(np.max(np.abs(gradient))),
                      'beta': float(initial if fixed else theta[0])})
    theta0 = np.zeros(f.shape[1] if fixed else f.shape[1]+1)
    if not fixed:
        theta0[0] = initial
    record(theta0)
    started = time.perf_counter()
    result = minimize(objective, theta0, jac=True, method='L-BFGS-B', callback=record,
                      options={'maxiter': 2000, 'gtol': 1e-8, 'ftol': 1e-13, 'maxls': 40})
    if not result.success:
        raise RuntimeError(f'Intercept optimization failed: {result.message}')
    beta, w = (initial, result.x) if fixed else (float(result.x[0]), result.x[1:])
    return dict(beta=float(beta), w=w, trace=trace, iterations=int(result.nit),
                objective=float(result.fun), elapsed=time.perf_counter()-started)


def dense_project(data, record, enc, horizon):
    prompt = np.load(data / 'prompt_h.npy', mmap_mode='r', allow_pickle=False)
    h = np.load(data / record['rollout'] / 'generated_h.npy', mmap_mode='r', allow_pickle=False)
    horizon = min(horizon, record['valid_positions'])
    assert horizon <= record['valid_positions']
    directions = enc['directions'].astype(np.float64)
    prompt_projection = np.einsum('ld,ld->l', prompt.sum(axis=0, dtype=np.float64), directions)
    center_projection = np.einsum('ld,ld->l', enc['centers'], directions)
    token_projection = np.empty((horizon, 36), dtype=np.float64)
    for start in range(0, horizon, 32):
        block = np.asarray(h[start:min(start+32,horizon)], dtype=np.float64)
        token_projection[start:start+len(block)] = np.einsum('tld,ld->tl', block, directions, optimize=True)
    sums = np.vstack((np.zeros((1,36)), np.cumsum(token_projection, axis=0)))
    x = (prompt_projection + sums) / (len(prompt) + np.arange(horizon+1))[:, None] - center_projection
    return x


def cubic_tensor(model):
    d, coefs = 36, model.coef_[0]
    g, h, tensor = coefs[:d].copy(), np.zeros((d,d)), np.zeros((d,d,d))
    index = d
    for i in range(d):
        for k in range(i,d):
            h[i,k] = h[k,i] = coefs[index] * (2 if i == k else 1)
            index += 1
    for i in range(d):
        for k in range(i,d):
            for m in range(k,d):
                positions = set(itertools.permutations((i,k,m)))
                value = coefs[index] * 6 / len(positions)
                for entry in positions:
                    tensor[entry] = value
                index += 1
    assert index == len(coefs) == 9138
    return {'b0_standardized': np.array(model.intercept_[0]), 'g_standardized': g,
            'H_standardized': h, 'T_standardized': tensor,
            'polynomial_coefficients': coefs,
            'powers': PolynomialFeatures(3,include_bias=False).fit(np.zeros((1,d))).powers_}


def run(root, out):
    out.mkdir(parents=True, exist_ok=True)
    v1 = out.parent / 'decoder_v1'
    data, protocol, records, labels, fingerprints = load_context(root)
    old_manifest = json.loads((v1 / 'a_lj_manifest.json').read_text())
    assert old_manifest['response_hashes'] == fingerprints
    raw = np.load(v1 / 'a_lj.npy', mmap_mode='r', allow_pickle=False)
    frozen = json.loads((v1 / 'folds_and_hyperparameters.json').read_text())
    indices = np.flatnonzero(labels != 'X')
    id_index = {r['rollout']: i for i,r in enumerate(records)}
    y = labels[indices].astype(int)
    row_index = {idx: k for k,idx in enumerate(indices)}
    dense = {kind:np.full((72,193), np.nan) for kind in (*KINDS,'cubic')}
    sensitivity = {(mode,b):np.full((72,len(STEPS)),np.nan)
                   for mode in ('initialized','fixed') for b in CONSTANTS}
    traces, fits, selections, case_curves, discrepancies = [], [], [], [], []
    with threadpool_limits(limits=2):
        for fold in frozen['outer']:
            print(f"Extension fold {fold['fold']}/3", flush=True)
            training = np.array([id_index[k] for k in fold['training']])
            testing = np.array([id_index[k] for k in fold['test']])
            assert not set(training) & set(testing)
            assert all(labels[i]!='X' for i in np.r_[training,testing])
            enc = fit_encoder(raw,training)
            xtr, xte = project(raw,training,enc), project(raw,testing,enc)
            ztr, zte = standardize(xtr,enc), standardize(xte,enc)
            ytr = np.repeat(labels[training].astype(int),len(STEPS))
            models = {}
            for kind in KINDS[1:]:
                models[kind] = train(ztr,ytr,kind,fold['selection'][kind]['C'])
                params = coefficients(models[kind],kind,enc)
                np.savez_compressed(out/f'fold{fold["fold"]}_{kind}.npz',**params)
            np.savez_compressed(out/f'fold{fold["fold"]}_encoder.npz',**enc)
            prepared = prepare_inner(raw,labels,training,SEED+fold['fold'])
            c, losses = tune_cubic(prepared)
            models['cubic'] = fit_cubic(ztr,ytr,c)
            np.savez_compressed(out/f'fold{fold["fold"]}_cubic.npz',**cubic_tensor(models['cubic']))
            selections.append({'fold':fold['fold'],'C':c,'inner_log_loss':losses})
            print(f'  Cubic C={c}; intercept diagnostics', flush=True)

            ftr, fte = features(ztr,'quadratic'), features(zte,'quadratic')
            z0 = standardize(xtr[:1,:1],enc)
            f0 = features(z0,'quadratic')[0]
            ftr, fte = ftr-f0, fte-f0
            cquad = fold['selection']['quadratic']['C']
            for mode in ('initialized','fixed'):
                for initial in CONSTANTS:
                    fit = fit_intercept_test(ftr,ytr,cquad,initial,mode=='fixed')
                    loc = [row_index[i] for i in testing]
                    sensitivity[(mode,initial)][loc] = expit(fit['beta']+fte@fit['w']).reshape(-1,len(STEPS))
                    fits.append({'fold':fold['fold'],'mode':mode,'initial_or_fixed_beta':initial,
                        'fitted_beta':fit['beta'],'fitted_initial_probability':float(expit(fit['beta'])),
                        'iterations':fit['iterations'],'objective':fit['objective'],
                        'elapsed_seconds':fit['elapsed'],'C':cquad})
                    traces.extend(dict(fold=fold['fold'],mode=mode,initial=initial,**row) for row in fit['trace'])
                    np.savez_compressed(out/f'fold{fold["fold"]}_{mode}_{initial:g}.npz',
                                        beta=fit['beta'],w=fit['w'],feature_origin=f0)

            for counter, idx in enumerate(testing):
                r, local = records[idx], row_index[idx]
                horizon = r['valid_positions'] if r['rollout'] in ('r0001','r0026') else 192
                x = dense_project(data,r,enc,horizon)
                compare = x[np.array(STEPS)]-project(raw,np.array([idx]),enc)[0]
                discrepancies.append({'rollout':r['rollout'],'max_coordinate_roundoff':float(np.abs(compare).max())})
                z = (x-enc['score_mean'])/enc['score_scale']
                probabilities = {'constant':np.full(len(x),float(ytr.mean()))}
                for kind in (*KINDS[1:],'cubic'):
                    f = cubic_features(z) if kind=='cubic' else features(z,kind)
                    probabilities[kind] = models[kind].predict_proba(f)[:,1]
                for kind, p in probabilities.items():
                    dense[kind][local] = p[:193]
                if r['rollout'] in ('r0001','r0026'):
                    np.savez_compressed(out/f'{r["rollout"]}_Xi_dense.npz',Xi=x[None,:,:],
                                        j=np.arange(len(x)),fold=fold['fold'])
                    case_curves.extend(dict(rollout=r['rollout'],fold=fold['fold'],y=int(labels[idx]),
                        j=j,extrapolation=j>192,final_token=(j==r['valid_positions']),
                        **{kind:float(p[j]) for kind,p in probabilities.items()}) for j in range(len(x)))
                if (counter+1)%8==0:
                    print(f'  Dense curves {counter+1}/{len(testing)}',flush=True)

        metrics=[]
        for kind,prediction in dense.items():
            assert np.isfinite(prediction).all()
            for j in range(193):
                p=prediction[:,j]
                metrics.append({'model':kind,'j':j,'AUROC':float(roc_auc_score(y,p)),
                    'Brier':float(brier_score_loss(y,p)),'log_loss':float(log_loss(y,p,labels=[0,1])),
                    'balanced_accuracy':float(balanced_accuracy_score(y,p>=.5))})
        intercept_metrics=[]
        for (mode,initial),prediction in sensitivity.items():
            for t,j in enumerate(STEPS):
                p=prediction[:,t]
                intercept_metrics.append({'mode':mode,'initial_or_fixed_beta':initial,'j':j,
                    'AUROC':float(roc_auc_score(y,p)),'Brier':float(brier_score_loss(y,p)),
                    'log_loss':float(log_loss(y,p,labels=[0,1]))})
        max_init_difference=max(float(np.abs(sensitivity[('initialized',b)]-
                                            sensitivity[('initialized',0.)]).max()) for b in CONSTANTS)
        write_csv(out/'metrics_dense_oof.csv',metrics)
        write_csv(out/'metrics_intercept_oof.csv',intercept_metrics)
        write_csv(out/'intercept_optimization.csv',fits)
        write_csv(out/'intercept_convergence.csv',traces)
        write_csv(out/'case_curves_every_token_oof.csv',case_curves)
        write_csv(out/'coordinate_roundoff.csv',discrepancies)
        write_csv(out/'predictions_dense_oof.csv',[dict(rollout=records[idx]['rollout'],y=int(y[k]),j=j,
            **{kind:float(p[k,j]) for kind,p in dense.items()}) for k,idx in enumerate(indices) for j in range(193)])
        dump(out/'cubic_selection.json',selections)
        summary={'primary_j192':[r for r in metrics if r['j']==192],
            'intercept_j192':[r for r in intercept_metrics if r['j']==192],
            'maximum_probability_difference_between_initializations':max_init_difference,
            'maximum_coordinate_roundoff':max(r['max_coordinate_roundoff'] for r in discrepancies),
            'quadratic_parameters':703,'cubic_parameters':9139,
            'dense_prediction_steps':list(range(193)),'training_steps':list(STEPS),
            'full_case_lengths_including_j0':{name:sum(r['rollout']==name for r in case_curves) for name in ('r0001','r0026')},
            'intercept_coordinate_origin':'shared prompt; beta=s(j=0); original polynomial weight penalty retained',
            'constant_grid':list(CONSTANTS),'fixed_beta_probabilities':[float(expit(b)) for b in CONSTANTS],
            'cubic_is_followup_not_independent_confirmation':True,
            'scope':'one prompt, same-model same-template rollout generalization only'}
        dump(out/'summary.json',summary)
        dump(out/'protocol.json',{'original_folds':frozen,'seed':SEED,
            'beta_objective':'mean(logaddexp(0,beta+F w)-y*(beta+F w)) + ||w||^2/(2*C*n)',
            'beta_features':'F=phi(z)-phi(z_prompt); beta is initial logit; not a probability',
            'beta_optimizer':'L-BFGS-B; maxiter=2000, gtol=1e-8, ftol=1e-13; intercept unpenalized',
            'beta_hyperparameters':'quadratic C fixed to each original training-only nested selection',
            'cubic_selection':'nested same 3 outer/3 inner folds and original C grid',
            'original_solver':'liblinear L2 logistic regression, max_iter=3000, intercept_scaling=100, default tol=1e-4',
            'original_intercept_penalty':'synthetic intercept feature: small penalty b_z^2/(2*C*100^2) in sum-loss convention',
            'original_loss':'sum(logaddexp(0,s)-y*s) + ||w||^2/(2*C) + b_z^2/(2*C*100^2)',
            'C_grid':list(CS),'X_excluded':28,'classified':72,
            'coordinate_order':'Xi[i,j,ell]; K=1; dense Xi shape [1,N+1,36]',
            'notes':'Dense inference does not refit at each j. Case curves after j192 are horizon extrapolation.'})
    result=dict(metrics=metrics,intercept_metrics=intercept_metrics,fits=fits,traces=traces,
                cases=case_curves,summary=summary)
    result['intercept_case_curves']=case_intercept_curves(out)
    return result


def case_intercept_curves(out):
    rows=[]
    for name in ('r0001','r0026'):
        with np.load(out/f'{name}_Xi_dense.npz',allow_pickle=False) as coordinates:
            x=coordinates['Xi'][0]
            fold=int(coordinates['fold'])
        with np.load(out/f'fold{fold}_encoder.npz',allow_pickle=False) as enc:
            z=(x-enc['score_mean'])/enc['score_scale']
        f=features(z,'quadratic')
        for mode in ('initialized','fixed'):
            for initial in CONSTANTS:
                with np.load(out/f'fold{fold}_{mode}_{initial:g}.npz',allow_pickle=False) as fit:
                    p=expit(fit['beta']+(f-fit['feature_origin'])@fit['w'])
                rows.extend(dict(rollout=name,mode=mode,initial_or_fixed_beta=initial,
                    j=j,probability=float(value),extrapolation=j>192) for j,value in enumerate(p))
    write_csv(out/'case_intercept_curves_every_token.csv',rows)
    sensitivity=[]
    for name in ('r0001','r0026'):
        selected=[r for r in rows if r['rollout']==name and r['mode']=='fixed']
        for j in (0,32,64,128,192,max(r['j'] for r in selected)):
            values=[r['probability'] for r in selected if r['j']==j]
            sensitivity.append(dict(rollout=name,j=j,probability_range=max(values)-min(values)))
    write_csv(out/'fixed_intercept_sensitivity_by_token.csv',sensitivity)
    return rows


def figures(result,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                         'savefig.dpi':180,'figure.dpi':130})
    paths=[]
    colors={'constant':'#717a7d','linear':'#147d92','diagonal':'#a77913','quadratic':'#bc4454','cubic':'#5b5594'}
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),sharey=True)
    for ax,name in zip(axes,('r0001','r0026')):
        rows=[r for r in result['cases'] if r['rollout']==name]
        for kind in (*KINDS,'cubic'):
            for extra,style in ((False,'-'),(True,'--')):
                part=[r for r in rows if r['extrapolation']==extra or (extra and r['j']==192)]
                ax.plot([r['j'] for r in part],[r[kind] for r in part],style,
                        color=colors[kind],label=kind if not extra else None,linewidth=1.5)
        ax.axvspan(192,rows[-1]['j'],color='#eef0f0',zorder=-1)
        ax.axvline(192,color='#92999b',linestyle=':',linewidth=1)
        ax.set(title=f'{name} | Y={rows[0]["y"]} | '+('LOT' if name=='r0001' else 'EOS'),
               xlabel='j: todos los tokens generados',ylim=(-.03,1.03))
        ax.grid(alpha=.2)
    axes[0].set_ylabel('P(etiqueta RH), rollout fuera del ajuste')
    axes[1].legend(fontsize=8)
    fig.suptitle('Trayectorias token a token | zona gris: más allá del horizonte de entrenamiento')
    fig.tight_layout();p=out/'01_dense_reference_cases.png';fig.savefig(p);plt.close(fig);paths.append(p)

    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for kind in (*KINDS,'cubic'):
        rows=[r for r in result['metrics'] if r['model']==kind]
        for ax,key in zip(axes,('AUROC','log_loss')):
            ax.plot([r['j'] for r in rows],[r[key] for r in rows],color=colors[kind],label=kind)
            ax.set(xlabel='j: cada token, misma cohorte de 72 rollouts',ylabel=key)
            ax.grid(alpha=.2)
    axes[0].axhline(.5,color='#aaa',linestyle=':',linewidth=1)
    axes[1].legend(fontsize=8);fig.tight_layout()
    p=out/'02_dense_metrics_and_cubic.png';fig.savefig(p);plt.close(fig);paths.append(p)

    fig,axes=plt.subplots(1,3,figsize=(13,4),sharey=True)
    palette=plt.get_cmap('viridis')
    for ax,fold in zip(axes,(1,2,3)):
        subset=[r for r in result['traces'] if r['mode']=='initialized' and r['fold']==fold]
        best=min(r['objective'] for r in subset)
        for k,b in enumerate(CONSTANTS):
            rows=[r for r in subset if r['initial']==b]
            ax.semilogy([r['iteration'] for r in rows],[max(r['objective']-best,1e-14) for r in rows],
                        color=palette(k/4),label=f'beta inicial={b:g}')
        ax.set(title=f'Fold {fold}',xlabel='Iteración L-BFGS-B');ax.grid(alpha=.2)
    axes[0].set_ylabel('Objetivo regularizado menos mínimo alcanzado')
    axes[-1].legend(fontsize=7);fig.tight_layout()
    p=out/'03_intercept_initialization_convergence.png';fig.savefig(p);plt.close(fig);paths.append(p)

    fig,axes=plt.subplots(1,2,figsize=(10,4))
    rows=[r for r in result['intercept_metrics'] if r['j']==192]
    for ax,key in zip(axes,('AUROC','log_loss')):
        for mode,color in (('initialized','#147d92'),('fixed','#bc4454')):
            part=[r for r in rows if r['mode']==mode]
            ax.plot([r['initial_or_fixed_beta'] for r in part],[r[key] for r in part],
                    'o-',color=color,label='Inicial y luego libre' if mode=='initialized' else 'Constante fija')
        ax.set(xlabel='beta: score inicial, no probabilidad',ylabel=key,title='Evaluación fuera de muestra, j=192');ax.grid(alpha=.2)
    axes[-1].legend(fontsize=8);fig.tight_layout()
    p=out/'04_fixed_vs_learned_intercept.png';fig.savefig(p);plt.close(fig);paths.append(p)

    fig,axes=plt.subplots(1,2,figsize=(12,4),sharey=True)
    for ax,name in zip(axes,('r0001','r0026')):
        subset=[r for r in result['intercept_case_curves'] if r['rollout']==name and r['mode']=='fixed']
        for k,b in enumerate(CONSTANTS):
            part=[r for r in subset if r['initial_or_fixed_beta']==b]
            ax.plot([r['j'] for r in part],[r['probability'] for r in part],
                    color=palette(k/4),label=f'beta fija={b:g}')
        ax.axvspan(192,max(r['j'] for r in subset),color='#eef0f0',zorder=-1)
        ax.set(title=name,xlabel='j: todos los tokens generados',ylim=(-.03,1.03));ax.grid(alpha=.2)
    axes[0].set_ylabel('P(etiqueta RH), rollout fuera del ajuste')
    axes[1].legend(fontsize=8)
    fig.suptitle('Constante fija: se reajustan g,H para cada beta | gris: extrapolación temporal')
    fig.tight_layout();p=out/'05_fixed_intercept_case_trajectories.png';fig.savefig(p);plt.close(fig);paths.append(p)
    return paths
