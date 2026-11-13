"""WL93 reviewer visuals. python viz93.py <R> <R86> <R92A> <R92B> <out>
Error / difference maps share ONE absolute scale per figure (sqrt map, colour bar in radiance units)."""
import glob, hashlib, json, os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
for d in ('ecro86', 'ecro_attr92'):
    sys.path.insert(0, os.path.join(HERE, '..', d))
import analyze92 as AN  # noqa: E402
import analyze86 as A  # noqa: E402
P = json.load(open(os.path.join(HERE, 'protocol', 'ecro_stage93.json')))
ARMS = list(P['arms']); SEEDS = P['training']['seeds']
Y = lambda x: x.mean(-1)
COL = {'T0': '#93a1a1', 'TQ': '#2aa198', 'U0': '#b58900', 'UQ': '#d33682', 'C0': '#6c71c4', 'CQ': '#268bd2', 'own92a': '#586e75', 'sum92a': '#cccccc'}


def cbar(fig, im, ax, vmax, label):
    cb = fig.colorbar(im, ax=ax, fraction=0.015)
    tk = np.array([-1, -0.25, 0, 0.25, 1.0])
    cb.set_ticks(np.sign(tk) * np.sqrt(np.abs(tk))); cb.set_ticklabels([f'{v * vmax:.2g}' for v in tk]); cb.set_label(label)


def main():
    R, R86, R92A, R92B, out = sys.argv[1:6]
    os.makedirs(out, exist_ok=True)
    F = json.load(open(os.path.join(R, 'results', 'final93.json')))
    made = []

    def ref(tag):
        for root in (R, R92B, R92A, R86):
            if glob.glob(os.path.join(root, 'results', 'renders', f'REF_ER_{tag}_r*.npz')):
                return A.load(root, f'REF_ER_{tag}')
    C = F.get('conditions', {})
    names = ARMS + ['own92a']
    # ---- fig1: region C E_H per arm and seed on E6 / E5 / E2, T1 and T0
    cases = [(cn, t) for cn in ('E6', 'E5', 'E2') if cn in C for t in ('T1', 'T0')]
    if cases:
        fig, ax = plt.subplots(1, len(cases), figsize=(3.6 * len(cases), 4), squeeze=False)
        for i, (cn, t) in enumerate(cases):
            for j, a in enumerate(names):
                for k, s in enumerate(SEEDS):
                    m = C[cn]['models'].get(f'{a}_s{s}')
                    if m and 'C' in m['fidelity'][t]:
                        x = m['fidelity'][t]['C']
                        ax[0, i].bar(j + 0.4 * k - 0.2, x['E_H'], 0.38, yerr=x['noise_bound'], color=COL[a], alpha=0.6 + 0.4 * k, error_kw={'lw': 0.6})
            for key, ls in (('ORACLE', ':'), ('X_EXP', '-.')):
                c = C[cn].get('controls', {}).get(f'{key}_{t}', {}).get('C')
                if c:
                    ax[0, i].axhline(c['E_H'], color='k', ls=ls, lw=0.8, label=key)
            ax[0, i].set_xticks(range(len(names))); ax[0, i].set_xticklabels(names, fontsize=7, rotation=30)
            lab = {'E6': 'held-out E6', 'E5': 'E5 (inspected)', 'E2': 'E2 (inspected)'}[cn]
            ax[0, i].set_title(f'{lab} {t}: region C E_H (left s{SEEDS[0]}, right s{SEEDS[1]})', fontsize=7); ax[0, i].grid(alpha=0.25, axis='y')
        ax[0, 0].legend(fontsize=6)
        fig.tight_layout(); fn = 'fig1_regionC_EH.png'; fig.savefig(os.path.join(out, fn), dpi=130); plt.close(fig); made.append(fn)
    # ---- fig2: source-stage error (Bq vs q, panel) and local matrices
    fig, ax = plt.subplots(1, 3, figsize=(16, 4))
    for i, cn in enumerate(('E2', 'E6')):
        if cn not in C:
            continue
        for j, a in enumerate(ARMS + ['own92a']):
            for k, s in enumerate(SEEDS):
                d = (C[cn]['models'].get(f'{a}_s{s}') or {}).get('diag', {}).get('T1') or {}
                if 'Bq_vs_q_panel_rel_l1' in d:
                    ax[i].bar(j + 0.4 * k - 0.2, d['Bq_vs_q_panel_rel_l1'], 0.38, color=COL[a], alpha=0.6 + 0.4 * k)
        idn = F.get('ident') or {}
        if cn == 'E2' and idn.get('E2'):
            ax[i].axhline(idn['E2']['E2_T1']['irreducible_panel_rel_l1'], color='k', ls=':', lw=0.8, label='exact frozen-ER through deployed encoding')
            ax[i].legend(fontsize=6)
        ax[i].set_xticks(range(len(ARMS) + 1)); ax[i].set_xticklabels(ARMS + ['own92a(=init)'], fontsize=7, rotation=20)
        ax[i].set_title(f'{cn} T1: source-stage table Bq vs physical q, panel patches (rel L1)', fontsize=8); ax[i].grid(alpha=0.25, axis='y')
    for j, a in enumerate(ARMS):
        for k, s in enumerate(SEEDS):
            m = (C.get('E2', {}).get('models', {}).get(f'{a}_s{s}') or {}).get('diag', {}).get('T1') or {}
            if m:
                ax[2].bar(j + 0.4 * k - 0.2, m['source_matrix_panel']['peak'], 0.18, color=COL[a], alpha=0.9)
                ax[2].bar(j + 0.4 * k - 0.02, m['receiver_matrix_panel']['peak'], 0.18, color=COL[a], alpha=0.45, hatch='//')
    ax[2].axhline(1.0, color='k', lw=0.6)
    ax[2].set_xticks(range(len(ARMS))); ax[2].set_xticklabels(ARMS, fontsize=7)
    ax[2].set_title('E2 T1 panel mirror-bin retention: source (solid) / receiver (hatched) factor (frozen ER = 1)', fontsize=8); ax[2].grid(alpha=0.25, axis='y')
    fig.tight_layout(); fn = 'fig2_source_stage_and_matrices.png'; fig.savefig(os.path.join(out, fn), dpi=130); plt.close(fig); made.append(fn)
    # ---- fig3: held-out E6 maps (T1 error and T1 - T0 change)
    ch = F.get('e6', {}).get('chosen')
    if ch is not None and 'E6' in C:
        e = f'E6k{ch}'
        Mk = AN.mask_of(R, R86, f'{e}_T1')
        r0, r1 = ref(f'{e}_T0'), ref(f'{e}_T1')
        h1 = Y(r1['h'].mean(0)); dref = h1 - Y(r0['h'].mean(0))
        show = [f'{a}_s{SEEDS[0]}' for a in ('T0', 'TQ', 'U0', 'UQ', 'CQ')]
        ld = lambda n, t: AN.hl(R, f'I93_{n}_{e}_{t}')
        show = [n for n in show if ld(n, 'T1') is not None and ld(n, 'T0') is not None]
        if show:
            err = [(n, Y(ld(n, 'T1')['hl'].mean(0)) - h1) for n in show]
            dif = [(n, Y(ld(n, 'T1')['hl'].mean(0)) - Y(ld(n, 'T0')['hl'].mean(0))) for n in show]
            em = max(float(np.abs(x).max()) for _, x in err); dm = max([float(np.abs(dref).max())] + [float(np.abs(x).max()) for _, x in dif])
            nc = len(show) + 1
            fig, ax = plt.subplots(2, nc, figsize=(3.4 * nc, 6.5))
            ax[0, 0].imshow(np.clip(h1 / max(np.quantile(h1, 0.995), 1e-9), 0, 1) ** 0.5, cmap='gray'); ax[0, 0].contour(Mk['C'], levels=[0.5], colors='r', linewidths=0.4)
            ax[0, 0].set_title('REF-ER higher-order T1 (region C red)', fontsize=8)
            for j, (n, x) in enumerate(err):
                im0 = ax[0, j + 1].imshow(np.sign(x) * np.sqrt(np.abs(x) / em), cmap='RdBu_r', vmin=-1, vmax=1); ax[0, j + 1].set_title(f'{n}: T1 error', fontsize=8)
                ax[0, j + 1].contour(Mk['C'], levels=[0.5], colors='k', linewidths=0.4)
            for j, (n, x) in enumerate([('REF-ER', dref)] + dif):
                im1 = ax[1, j].imshow(np.sign(x) * np.sqrt(np.abs(x) / dm), cmap='RdBu_r', vmin=-1, vmax=1); ax[1, j].set_title(f'{n}: T1 - T0', fontsize=8)
            for a_ in ax.ravel():
                a_.set_xticks([]); a_.set_yticks([])
            cbar(fig, im0, ax[0, 1:], em, 'error (sqrt)'); cbar(fig, im1, ax[1, :], dm, 'change (sqrt)')
            fig.suptitle(f'held-out {e} (seed {SEEDS[0]}): higher-order error at T1 and signed change; scales shared per row', fontsize=9)
            fn = 'fig3_E6_maps.png'; fig.savefig(os.path.join(out, fn), dpi=130, bbox_inches='tight'); plt.close(fig); made.append(fn)
    # ---- fig4: training curves
    curves = {}
    for a in ARMS:
        for s in SEEDS:
            fn_ = os.path.join(R, 'results', 'models', f'{a}_s{s}.json')
            if os.path.exists(fn_):
                curves[f'{a}_s{s}'] = json.load(open(fn_))['log']
    if curves:
        fig, ax = plt.subplots(1, 3, figsize=(16, 4))
        for n, lg in curves.items():
            a = n.split('_')[0]; st = [r['step'] for r in lg]
            for i, k in enumerate(('L_sum', 'L_single', 'L_source_stage')):
                ax[i].plot(st, [r[k] for r in lg], color=COL[a], ls='-' if n.endswith(str(SEEDS[0])) else '--', lw=0.9, label=n)
        for i, k in enumerate(('L_sum', 'L_single', 'L_source_stage (monitored for every arm)')):
            ax[i].set_yscale('log'); ax[i].set_title(k, fontsize=9); ax[i].grid(alpha=0.25)
        ax[0].legend(fontsize=6, ncol=2)
        fig.tight_layout(); fn = 'fig4_training_curves.png'; fig.savefig(os.path.join(out, fn), dpi=130); plt.close(fig); made.append(fn)
    man = {'worklog': '93', 'source_root': R, 'files': {f: hashlib.sha256(open(os.path.join(out, f), 'rb').read()).hexdigest() for f in made},
           'provenance': 'experiments/ecro_stage93/viz93.py from results/ecro_stage93 (final93.json, renders, model logs) and the WL86 / WL92A / WL92B / WL93 REF-ER renders and masks'}
    json.dump(man, open(os.path.join(out, 'manifest.json'), 'w'), indent=1)
    print('VIZ', made, flush=True)


if __name__ == '__main__':
    main()
