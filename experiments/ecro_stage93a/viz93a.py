"""WL93A reviewer visuals. python viz93a.py <R> <R86> <R92A> <R92B> <out>
Error / difference maps share ONE absolute scale per figure (sqrt map, colour bar in radiance units)."""
import glob, hashlib, json, os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
for d in ('ecro86', 'ecro_attr92'):
    sys.path.insert(0, os.path.join(HERE, '..', d))
import analyze92 as AN  # noqa: E402
import analyze86 as A  # noqa: E402
P = json.load(open(os.path.join(HERE, 'protocol', 'ecro_stage93a.json')))
SEEDS = P['task_b']['training']['seeds']
BN = list(P['target']['budgets'])
Y = lambda x: x.mean(-1)
COL = {'T0': '#93a1a1', 'TQ': '#2aa198', 'own92a': '#586e75', 'enc': '#b58900'}


def cbar(fig, im, ax, vmax, label):
    cb = fig.colorbar(im, ax=ax, fraction=0.015)
    tk = np.array([-1, -0.25, 0, 0.25, 1.0])
    cb.set_ticks(np.sign(tk) * np.sqrt(np.abs(tk))); cb.set_ticklabels([f'{v * vmax:.2g}' for v in tk]); cb.set_label(label)


def main():
    R, R86, R92A, R92B, out = sys.argv[1:6]
    os.makedirs(out, exist_ok=True)
    F = json.load(open(os.path.join(R, 'results', 'final93a.json')))
    TA = F['taskA']; made = []
    # ---- fig1: target disagreement per budget and discrepancies vs B5
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.3))
    w = 0.27
    for j, g in enumerate(('train_rep', 'val', 'sharp')):
        G = TA['groups'][g]
        ax[0].bar(np.arange(len(BN)) + (j - 1) * w, [G['noise'][b]['N_single_rel_l2'] for b in BN], w, label=g)
    ax[0].axhline(0.10, color='r', ls='--', lw=0.8, label='train-budget rule 0.10')
    ax[0].set_xticks(range(len(BN))); ax[0].set_xticklabels([f"{b}\nP{P['target']['budgets'][b]['P']} Mq{P['target']['budgets'][b]['Mq']}" for b in BN], fontsize=7)
    ax[0].set_yscale('log'); ax[0].legend(fontsize=7); ax[0].grid(alpha=0.25, axis='y')
    ax[0].set_title('(1) independent target disagreement: single-realization rel L2 (panel)', fontsize=9)
    names = ['enc', 'own_s8601', 'own_s8602']
    for j, g in enumerate(('train_rep', 'val', 'sharp')):
        D = TA['groups'][g]['discrepancy']
        v = [D[n]['M'] for n in names]; ci = [D[n]['boot']['ci95'] for n in names]
        ax[1].bar(np.arange(3) + (j - 1) * w, v, w, yerr=np.array([[x - c[0] for x, c in zip(v, ci)], [c[1] - x for x, c in zip(v, ci)]]) if g != 'sharp' else None, label=g, error_kw={'lw': 0.6})
    ax[1].set_xticks(range(3)); ax[1].set_xticklabels(['(2) frozen-ER via encoding', '(3) OWN s8601', '(3) OWN s8602'], fontsize=8)
    ax[1].legend(fontsize=7); ax[1].grid(alpha=0.25, axis='y'); ax[1].set_title('debiased rel L2 vs independent B5 reference (panel; 95 % geometry bootstrap)', fontsize=9)
    pc = TA['per_cfg']
    xs = [pc[k]['N_B0'] for k in pc if pc[k]['kind'] == 'val']; ys = [pc[k]['M']['enc'] for k in pc if pc[k]['kind'] == 'val']
    ax[2].scatter(xs, ys, s=12, color='#2a78d6', label='validation geometries')
    for k in pc:
        if pc[k]['kind'] == 'val' and (pc[k]['N_B0'] > 0.3 or pc[k]['M']['enc'] > 0.8):
            ax[2].annotate(pc[k]['id'][:6], (pc[k]['N_B0'], pc[k]['M']['enc']), fontsize=7)
    ax[2].set_xscale('log'); ax[2].set_xlabel('B0 (WL93 budget) single-realization noise'); ax[2].set_ylabel('M(enc) vs B5')
    ax[2].grid(alpha=0.25); ax[2].set_title('per geometry: WL93-budget target noise vs encoding discrepancy', fontsize=9)
    fig.tight_layout(); fn = 'fig1_target_reliability.png'; fig.savefig(os.path.join(out, fn), dpi=130); plt.close(fig); made.append(fn)
    # ---- fig2: Task B
    E = F.get('evalB')
    if E:
        fig, ax = plt.subplots(1, 3, figsize=(17, 4.3))
        V = E['groups']['val']
        for k, s in enumerate(SEEDS):
            for j, n in enumerate(['own92a', 'T0', 'TQ']):
                key = f'own_s{s}' if n == 'own92a' else f'{n}_s{s}'
                ax[0].bar(j + 0.4 * k - 0.2, V[key]['M'], 0.38, color=COL[n], alpha=0.6 + 0.4 * k, yerr=V[key]['boot']['se'], error_kw={'lw': 0.6})
        ax[0].set_xticks(range(3)); ax[0].set_xticklabels(['WL92A OWN (init)', 'T0', 'TQ'], fontsize=8)
        ax[0].set_title('validation source stage: M vs independent B5 (left s8601, right s8602)', fontsize=9); ax[0].grid(alpha=0.25, axis='y')
        C = F.get('criteria', {})
        if C.get('per_seed'):
            for k, s in enumerate(SEEDS):
                p = C['per_seed'][str(s)] if str(s) in C['per_seed'] else C['per_seed'][s]
                ax[1].errorbar(k, p['dM'], yerr=[[p['dM'] - p['ci95'][0]], [p['ci95'][1] - p['dM']]], fmt='o', color='k')
                ax[1].hlines(-p['threshold'], k - 0.3, k + 0.3, color='r', ls='--', lw=0.8)
            ax[1].axhline(0, color='k', lw=0.5); ax[1].set_xticks(range(len(SEEDS))); ax[1].set_xticklabels([f's{s}' for s in SEEDS])
            ax[1].set_title('dM = M(TQ) - M(T0), validation (95 % paired bootstrap; red: predeclared signal threshold)', fontsize=8); ax[1].grid(alpha=0.25)
        for n in ('T0', 'TQ'):
            for k, s in enumerate(SEEDS):
                fn_ = os.path.join(R, 'results', 'models', f'{n}_s{s}.json')
                if os.path.exists(fn_):
                    lg = [r for r in json.load(open(fn_))['log'] if 'val' in r]
                    ax[2].plot([r['step'] for r in lg], [r['val']['Bq_vs_trainval_q_panel_rel_l1'] for r in lg], color=COL[n], ls='-' if k == 0 else '--', label=f'{n} s{s}')
        ax[2].set_title('validation panel Bq vs independent training-validation target during training', fontsize=8); ax[2].legend(fontsize=7); ax[2].grid(alpha=0.25)
        fig.tight_layout(); fn = 'fig2_training_smoke.png'; fig.savefig(os.path.join(out, fn), dpi=130); plt.close(fig); made.append(fn)
    # ---- fig3: rendered smoke E2 T1 maps (diagnostic)
    tag = 'E2_T1'
    rf = None
    for root in (R, R92B, R92A, R86):
        if glob.glob(os.path.join(root, 'results', 'renders', f'REF_ER_{tag}_r*.npz')):
            rf = A.load(root, f'REF_ER_{tag}'); break
    show = [f'{n}_s{SEEDS[0]}' for n in ('own92a', 'T0', 'TQ')] + [f'{n}_s{SEEDS[1]}' for n in ('own92a', 'T0', 'TQ')]
    imgs = [(n, AN.hl(R, f'I93A_{n}_{tag}')) for n in show]
    imgs = [(n, Y(x['hl'].mean(0))) for n, x in imgs if x is not None]
    if rf is not None and imgs:
        M = AN.mask_of(R, R86, tag); h = Y(rf['h'].mean(0))
        m = M['C'] | M['obj_panel']; ys, xs = np.nonzero(m)
        sl = (slice(max(ys.min() - 20, 0), ys.max() + 20), slice(max(xs.min() - 20, 0), xs.max() + 20))
        emax = max(float(np.abs(x - h)[sl].max()) for _, x in imgs)
        fig, ax = plt.subplots(2, 3, figsize=(10, 7))
        for k, (n, x) in enumerate(imgs):
            a_ = ax.ravel()[k]; d = (x - h)[sl]
            eh = F['renders'].get(tag, {}).get(n, {}).get('C', {}).get('E_H')
            im = a_.imshow(np.sign(d) * np.sqrt(np.abs(d) / emax), cmap='RdBu_r', vmin=-1, vmax=1)
            a_.set_title(f'{n}  region C E_H {eh:.4f}' if eh is not None else n, fontsize=8); a_.set_xticks([]); a_.set_yticks([])
        cbar(fig, im, ax, emax, 'signed higher-order error (sqrt)')
        fig.suptitle('E2 T1 sharp readers (diagnostic smoke render, 2 x 4,096 spp): shared scale', fontsize=9)
        fn = 'fig3_E2T1_render_smoke.png'; fig.savefig(os.path.join(out, fn), dpi=130, bbox_inches='tight'); plt.close(fig); made.append(fn)
    man = {'worklog': '93A', 'source_root': R, 'files': {f: hashlib.sha256(open(os.path.join(out, f), 'rb').read()).hexdigest() for f in made},
           'provenance': 'experiments/ecro_stage93a/viz93a.py from results/ecro_stage93a (final93a.json, taskA.json, evalB.json, model logs, renders) and the WL86 / WL92B REF-ER renders and masks'}
    json.dump(man, open(os.path.join(out, 'manifest.json'), 'w'), indent=1)
    print('VIZ', made, flush=True)


if __name__ == '__main__':
    main()
