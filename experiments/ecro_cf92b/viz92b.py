"""WL92B reviewer visuals. python viz92b.py <R> <R86> <R92A> <out>
Error / difference maps share ONE absolute scale per figure (sqrt map, colour bar in radiance units)."""
import hashlib, json, os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', 'ecro_attr92'))
import analyze92 as AN  # noqa: E402
import analyze92b as AB  # noqa: E402
Y = lambda x: x.mean(-1)
COL = {'ecro': '#7a52a8', 'sum': '#93a1a1', 'own': '#2aa198'}
CFS = AB.CFS; MODELS = AB.MODELS


def cbar(fig, im, ax, vmax, label):
    cb = fig.colorbar(im, ax=ax, fraction=0.015)
    tk = np.array([-1, -0.25, 0, 0.25, 1.0])
    cb.set_ticks(np.sign(tk) * np.sqrt(np.abs(tk))); cb.set_ticklabels([f'{v * vmax:.2g}' for v in tk]); cb.set_label(label)


def main():
    R, R86, R92A, out = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
    os.makedirs(out, exist_ok=True)
    F = json.load(open(os.path.join(R, 'results', 'final92b.json')))
    made = []
    w = 0.8 / len(MODELS)
    # ---- fig1 / fig2: counterfactual E_H by case
    for fn, cases in (('fig1_counterfactuals_E2_regionC.png', [('E2_T1', 'C'), ('E2_T0', 'C'), ('E4_T1', 'C')]), ('fig2_counterfactuals_E1_LSPOT.png', [('E1_T1_LSPOT', 'stationary_all'), ('E1_T1_LSPOT', 'C')])):
        cases = [c for c in cases if c[0] in F['cases']]
        if not cases:
            continue
        fig, ax = plt.subplots(1, len(cases), figsize=(6.5 * len(cases), 4.2), squeeze=False)
        for i, (tag, rk) in enumerate(cases):
            o = F['cases'][tag]
            for j, n in enumerate(MODELS):
                v = [o['models'][n]['E_H'].get(cf, {}).get(rk, {}).get('E_H', np.nan) for cf in CFS]
                ax[0, i].bar(np.arange(len(CFS)) + j * w, v, w, color=COL[n.split('_')[0]], alpha=0.55 + 0.45 * (j % 2), label=n)
            fl = o['floor'].get(rk, {}).get('E_H')
            if fl is not None:
                ax[0, i].axhline(fl, color='k', ls=':', lw=0.8, label='path-traced table')
            if rk == 'C':
                ax[0, i].axhline(0.06, color='r', ls='--', lw=0.8)
            ax[0, i].set_xticks(np.arange(len(CFS)) + 0.4); ax[0, i].set_xticklabels(CFS, fontsize=7, rotation=20)
            ax[0, i].set_title(f'{tag} {rk}: E_H per counterfactual' + (' (non-admitted diagnostic)' if tag.startswith('E4') else ''), fontsize=8); ax[0, i].grid(alpha=0.25, axis='y')
        ax[0, 0].legend(fontsize=6, ncol=2)
        fig.tight_layout(); fig.savefig(os.path.join(out, fn), dpi=130); plt.close(fig); made.append(fn)
    # ---- fig3: local matrices across configurations
    L = F.get('local')
    if L:
        cfgs = ['E2_T0', 'E2_T1', 'E1_T1', 'E4_T0', 'E4_T1']
        fig, ax = plt.subplots(1, 2, figsize=(13, 4))
        for j, n in enumerate(MODELS):
            ax[0].bar(np.arange(len(cfgs)) + j * w, [L['metrics'][f'{n}:{c}']['panel']['peak'] for c in cfgs], w, color=COL[n.split('_')[0]], alpha=0.55 + 0.45 * (j % 2), label=n)
        ax[0].set_xticks(np.arange(len(cfgs)) + 0.4); ax[0].set_xticklabels(cfgs, fontsize=8); ax[0].set_title('panel a c^T: mirror-bin flux retention per configuration (frozen ER = 1)', fontsize=9)
        ax[0].legend(fontsize=6, ncol=3); ax[0].grid(alpha=0.25, axis='y')
        mats = ['panel', 'floor', 'sphere', 'walls']
        for j, n in enumerate(MODELS):
            ax[1].bar(np.arange(len(mats)) + j * w, [L['geometry_variation'][n]['E2_T0->E2_T1'][m] for m in mats], w, color=COL[n.split('_')[0]], alpha=0.55 + 0.45 * (j % 2))
        ax[1].set_xticks(np.arange(len(mats)) + 0.4); ax[1].set_xticklabels(mats, fontsize=8)
        ax[1].set_title('relative change of learned a c^T on unchanged surfaces, E2 T0 -> T1 (box moved)', fontsize=9); ax[1].grid(alpha=0.25, axis='y')
        fig.tight_layout(); fn = 'fig3_local_matrix_configurations.png'; fig.savefig(os.path.join(out, fn), dpi=130); plt.close(fig); made.append(fn)
    # ---- fig4: sharp-reader maps E2 T1 (original and two single substitutions for each OWN seed)
    tag = 'E2_T1'
    if tag in F['cases']:
        M = AN.mask_of(R, R86, tag); rf = AB.ref(R, R86, R92A, tag)
        h = Y(rf['h'].mean(0))
        m = M['C'] | M['obj_panel']; ys, xs = np.nonzero(m)
        sl = (slice(max(ys.min() - 20, 0), ys.max() + 20), slice(max(xs.min() - 20, 0), xs.max() + 20))
        cols = [(n, cf) for n in ('sum_s8601', 'own_s8601', 'own_s8602') for cf in ('ORIG', 'X_M12', 'X_L3')]
        imgs = []
        for n, cf in cols:
            x = AN.hl(R, f'X92B_{cf}_{n}_{tag}')
            if x is not None:
                imgs.append((f'{n} {cf}', Y(x['hl'].mean(0))))
        if imgs:
            emax = max(float(np.abs(x - h)[sl].max()) for _, x in imgs)
            fig, ax = plt.subplots(3, 3, figsize=(10, 9))
            for k, (lab, x) in enumerate(imgs):
                a_ = ax.ravel()[k]
                d = (x - h)[sl]
                im = a_.imshow(np.sign(d) * np.sqrt(np.abs(d) / emax), cmap='RdBu_r', vmin=-1, vmax=1); a_.set_title(lab, fontsize=8); a_.set_xticks([]); a_.set_yticks([])
            cbar(fig, im, ax, emax, 'signed error (sqrt scale)')
            fig.suptitle('E2 T1 sharp readers: signed higher-order error, original vs exact local response (X_M12) vs exact global (X_L3); shared scale', fontsize=9)
            fn = 'fig4_E2T1_sharp_error_maps.png'; fig.savefig(os.path.join(out, fn), dpi=130, bbox_inches='tight'); plt.close(fig); made.append(fn)
    # ---- fig5: level vs directional split of the reading (region C)
    rows = []
    for tag in ('E2_T1', 'E2_T0', 'E4_T1'):
        if tag in F['cases']:
            for n in MODELS:
                r = F['cases'][tag]['models'][n]['reading'].get('C', {})
                if 'single_vs_L2' in r:
                    rows.append((f'{tag} {n}', r['single_vs_L2']['level'], r['single_vs_L2']['directional'], r['global_vs_L3']['signed']))
    if rows:
        fig, ax = plt.subplots(figsize=(9, 0.32 * len(rows) + 1.5))
        yy = np.arange(len(rows))[::-1]
        ax.barh(yy - 0.27, [r[1] for r in rows], 0.27, color='#d6392a', label='single term: level (patch flux)')
        ax.barh(yy, [r[2] for r in rows], 0.27, color='#e6a02a', label='single term: directional (bin placement)')
        ax.barh(yy + 0.27, [r[3] for r in rows], 0.27, color='#2a78d6', label='global term vs L3+')
        ax.axvline(0, color='k', lw=0.6); ax.set_yticks(yy); ax.set_yticklabels([r[0] for r in rows], fontsize=7); ax.legend(fontsize=7); ax.grid(alpha=0.25, axis='x')
        ax.set_title('region C reading error (fractions of the true higher-order reading)', fontsize=9)
        fig.tight_layout(); fn = 'fig5_level_vs_directional.png'; fig.savefig(os.path.join(out, fn), dpi=130); plt.close(fig); made.append(fn)
    # ---- fig6 / fig7: E5 (only if admitted)
    E5 = F.get('e5', {})
    if E5.get('chosen') is not None and E5.get('models'):
        e = f"E5k{E5['chosen']}"
        Ms = E5['models']
        fig, ax = plt.subplots(1, 3, figsize=(16, 4))
        for i, (t, k) in enumerate((('T1', 'C'), ('T0', 'C'), ('T1', 'stationary_all'))):
            ax[i].bar(range(len(MODELS)), [Ms[n]['fidelity'][t][k]['E_H'] for n in MODELS], color=[COL[n.split('_')[0]] for n in MODELS])
            for key, ls in (('ORACLE', ':'), ('X_EXP', '-.')):
                c = E5.get('controls', {}).get(f'{key}_{t}', {}).get(k)
                if c:
                    ax[i].axhline(c['E_H'], color='k', ls=ls, lw=0.8, label=key)
            ax[i].set_xticks(range(len(MODELS))); ax[i].set_xticklabels(MODELS, fontsize=7, rotation=20); ax[i].set_title(f'held-out {e} {t} {k} E_H', fontsize=9); ax[i].grid(alpha=0.25, axis='y')
        ax[0].legend(fontsize=7)
        fig.tight_layout(); fn = 'fig6_E5_heldout_EH.png'; fig.savefig(os.path.join(out, fn), dpi=130); plt.close(fig); made.append(fn)
        Mk = AN.mask_of(R, R86, f'{e}_T1')
        r0, r1 = AB.ref(R, R86, R92A, f'{e}_T0'), AB.ref(R, R86, R92A, f'{e}_T1')
        h1 = Y(r1['h'].mean(0)); dref = h1 - Y(r0['h'].mean(0))
        show = ['sum_s8601', 'own_s8601', 'sum_s8602', 'own_s8602']
        err = [(n, Y(AN.hl(R, f'I92B_{n}_{e}_T1')['hl'].mean(0)) - h1) for n in show]
        dif = [(n, Y(AN.hl(R, f'I92B_{n}_{e}_T1')['hl'].mean(0)) - Y(AN.hl(R, f'I92B_{n}_{e}_T0')['hl'].mean(0))) for n in show]
        em = max(float(np.abs(x).max()) for _, x in err); dm = max([float(np.abs(dref).max())] + [float(np.abs(x).max()) for _, x in dif])
        fig, ax = plt.subplots(2, 5, figsize=(17, 6.5))
        ax[0, 0].axis('off')
        for j, (n, x) in enumerate(err):
            im0 = ax[0, j + 1].imshow(np.sign(x) * np.sqrt(np.abs(x) / em), cmap='RdBu_r', vmin=-1, vmax=1); ax[0, j + 1].set_title(f'{n}: T1 error', fontsize=8)
            ax[0, j + 1].contour(Mk['C'], levels=[0.5], colors='k', linewidths=0.4)
        for j, (n, x) in enumerate([('REF-ER', dref)] + dif):
            im1 = ax[1, j].imshow(np.sign(x) * np.sqrt(np.abs(x) / dm), cmap='RdBu_r', vmin=-1, vmax=1); ax[1, j].set_title(f'{n}: T1 - T0', fontsize=8)
        for a_ in ax.ravel():
            a_.set_xticks([]); a_.set_yticks([])
        cbar(fig, im0, ax[0, :], em, 'error (sqrt)'); cbar(fig, im1, ax[1, :], dm, 'change (sqrt)')
        fig.suptitle(f'held-out {e}: higher-order error at T1 (contour: region C) and signed change; scales shared per row', fontsize=9)
        fn = 'fig7_E5_maps.png'; fig.savefig(os.path.join(out, fn), dpi=130, bbox_inches='tight'); plt.close(fig); made.append(fn)
    man = {'worklog': '92B', 'source_root': R, 'files': {f: hashlib.sha256(open(os.path.join(out, f), 'rb').read()).hexdigest() for f in made},
           'provenance': 'experiments/ecro_cf92b/viz92b.py from results/ecro_cf92b (final92b.json, renders) and the WL86 / WL92A / WL92B REF-ER renders and masks'}
    json.dump(man, open(os.path.join(out, 'manifest.json'), 'w'), indent=1)
    print('VIZ', made, flush=True)


if __name__ == '__main__':
    main()
