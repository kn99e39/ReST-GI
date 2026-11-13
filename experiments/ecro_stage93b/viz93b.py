"""WL93B reviewer visuals. python viz93b.py <R> <R86> <R92A> <R92B> <out>
Error / difference maps share ONE absolute scale per figure (sqrt map, colour bar in radiance units); images share one exposure."""
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
SEEDS = [8601, 8602]
ARMS = ['T0', 'TQ', 'U0', 'UQ', 'C0', 'CQ']
COL = {'T0': '#93a1a1', 'TQ': '#2aa198', 'U0': '#b58900', 'UQ': '#d33682', 'C0': '#6c71c4', 'CQ': '#268bd2', 'own': '#586e75'}
Y = lambda x: x.mean(-1)


def cbar(fig, im, ax, vmax, label):
    cb = fig.colorbar(im, ax=ax, fraction=0.015)
    tk = np.array([-1, -0.25, 0, 0.25, 1.0])
    cb.set_ticks(np.sign(tk) * np.sqrt(np.abs(tk))); cb.set_ticklabels([f'{v * vmax:.2g}' for v in tk]); cb.set_label(label)


def main():
    R, R86, R92A, R92B, out = sys.argv[1:6]
    os.makedirs(out, exist_ok=True)
    F = json.load(open(os.path.join(R, 'results', 'final93b.json')))
    acc = json.load(open(os.path.join(R, 'results', 'account.json')))
    made = []

    def ref(tag):
        for root in (R, R92B, R92A, R86):
            if glob.glob(os.path.join(root, 'results', 'renders', f'REF_ER_{tag}_r*.npz')):
                return A.load(root, f'REF_ER_{tag}')
    # ---- fig1: stage / integrated accounting on validation
    V = acc['groups']['val']
    keys = [('SRC', 'source stage vs B5 (rel L1, panel)'), ('RECV_Q', 'receiver use on the physical first flight vs t2'), ('SINGLE', 'single flight vs t2'), ('WHOLE', 'whole table vs t')]
    fig, ax = plt.subplots(1, 4, figsize=(19, 4))
    names = ['own'] + ARMS
    for i, (k, t) in enumerate(keys):
        for j, a in enumerate(names):
            for q, s in enumerate(SEEDS):
                n = f'{a}_s{s}'
                if n in V:
                    ax[i].bar(j + 0.4 * q - 0.2, V[n][k], 0.38, color=COL[a], alpha=0.6 + 0.4 * q)
            if a in ('T0', 'TQ'):
                for q, s in enumerate(SEEDS):
                    n = f'R_{a}_s{s}'
                    if n in V:
                        ax[i].plot(j + 0.4 * q - 0.2, V[n][k], 'kx', ms=5)
        if k == 'RECV_Q':
            ax[i].axhline(V['own_s8601']['RECV_Q_floor'], color='k', ls=':', lw=0.8, label='exact frozen-ER local (floor)')
            ax[i].legend(fontsize=7)
        lo = min(V[n][k] for n in V); hi = max(V[n][k] for n in V)
        ax[i].set_ylim(lo - 0.15 * (hi - lo), hi + 0.15 * (hi - lo))
        ax[i].set_xticks(range(len(names))); ax[i].set_xticklabels(['WL92A own'] + ARMS, fontsize=8); ax[i].grid(alpha=0.25, axis='y')
        ax[i].set_title(f'{t}\n(left s8601, right s8602; x: retrained R_T0 / R_TQ)', fontsize=8)
    fig.suptitle('validation (55 geometries x 4 lights, g00010 included): stage and integrated relative L1', fontsize=9)
    fig.tight_layout(); fn = 'fig1_stage_accounting.png'; fig.savefig(os.path.join(out, fn), dpi=130); plt.close(fig); made.append(fn)
    # ---- fig2: per-geometry differences (UQ - TQ, UQ - CQ) for SRC and WHOLE
    C = json.load(open(os.path.join(R, 'results', 'configs.json')))['configs']
    vidx = [i for i, c in enumerate(C) if c['kind'] == 'val']
    rel = lambda r, k: (r['src']['l1'] / r['src']['sq']) if k == 'SRC' else r[k][0] / r[k][1]
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.4))
    for i, (k, lab) in enumerate((('SRC', 'source stage'), ('whole_l1', 'whole table'))):
        for q, s in enumerate(SEEDS):
            for x, y, mk in (('UQ', 'TQ', 'o'), ('UQ', 'CQ', '^'), ('U0', 'T0', 's')):
                d = [rel(acc['per_cfg'][f'{x}_s{s}'][j], k) - rel(acc['per_cfg'][f'{y}_s{s}'][j], k) for j in vidx]
                ax[i].scatter(np.arange(len(vidx)) + 0.15 * q, d, s=10, marker=mk, color=['#d33682', '#268bd2', '#b58900'][['UQ-TQ', 'UQ-CQ', 'U0-T0'].index(f'{x}-{y}')], alpha=0.5 + 0.4 * q,
                              label=f'{x} - {y} s{s}')
        g10 = [k_ for k_, j in enumerate(vidx) if C[j].get('file') == 'g00010.npz']
        for g in g10:
            ax[i].axvline(g, color='r', ls=':', lw=0.8)
        ax[i].axhline(0, color='k', lw=0.6); ax[i].grid(alpha=0.25)
        ax[i].set_title(f'per validation geometry: difference of {lab} relative L1 (red line: g00010)', fontsize=9); ax[i].set_xlabel('validation geometry (sorted)')
    ax[0].legend(fontsize=6, ncol=3)
    fig.tight_layout(); fn = 'fig2_per_geometry_differences.png'; fig.savefig(os.path.join(out, fn), dpi=130); plt.close(fig); made.append(fn)
    # ---- fig3 / fig4: E2 T1 sharp readers (monochrome) and the LPAT pattern diagnostic
    for tag, fn, title, closeup in (('E2_T1', 'fig3_E2T1_sharp_maps.png', 'E2 T1 sharp readers', False), ('E2_T1_LPAT', 'fig4_pattern_LPAT.png', 'LPAT 2 x 2 checker (diagnostic; cone class unseen in training)', False),
                                    ('E2_T1_LPAT', 'fig4b_pattern_LPAT_closeup.png', 'LPAT close-up of region C (sharp light -> panel -> receiver readers)', True)):
        rf = ref(tag)
        try:
            M = AN.mask_of(R, R86, tag)
        except FileNotFoundError:
            continue
        if rf is None:
            continue
        h = Y(rf['h'].mean(0))
        m = M['C'] if closeup else (M['C'] | M.get('obj_panel', M['C']))
        if not m.any():
            continue
        ys, xs = np.nonzero(m); sl = (slice(max(ys.min() - 20, 0), ys.max() + 20), slice(max(xs.min() - 20, 0), xs.max() + 20))
        show = [f'{a}_s{s}' for s in SEEDS for a in ('T0', 'TQ', 'UQ', 'CQ')]
        imgs = [(n, AN.hl(R, f'I93B_{n}_{tag}')) for n in show]
        imgs = [(n, Y(x['hl'].mean(0))) for n, x in imgs if x is not None]
        if not imgs:
            continue
        vmax = float(np.quantile(h[sl], 0.995)); emax = max(float(np.abs(x - h)[sl].max()) for _, x in imgs)
        fig, ax = plt.subplots(4, 5, figsize=(17, 13))
        for q, s in enumerate(SEEDS):
            ax[2 * q, 0].imshow(np.clip(h[sl] / vmax, 0, 1) ** 0.5, cmap='gray'); ax[2 * q, 0].set_title('reference higher-order', fontsize=8)
            ax[2 * q + 1, 0].axis('off')
            for j, a in enumerate(('T0', 'TQ', 'UQ', 'CQ')):
                x = dict(imgs).get(f'{a}_s{s}')
                if x is None:
                    continue
                eh = F['renders'].get(tag, {}).get(f'{a}_s{s}', {}).get('C', {}).get('E_H')
                ax[2 * q, j + 1].imshow(np.clip(x[sl] / vmax, 0, 1) ** 0.5, cmap='gray'); ax[2 * q, j + 1].set_title(f'{a} s{s}' + (f'  C E_H {eh:.4f}' if eh is not None else ''), fontsize=8)
                d = (x - h)[sl]
                im = ax[2 * q + 1, j + 1].imshow(np.sign(d) * np.sqrt(np.abs(d) / emax), cmap='RdBu_r', vmin=-1, vmax=1); ax[2 * q + 1, j + 1].set_title(f'{a} s{s} - reference', fontsize=8)
                ax[2 * q + 1, j + 1].contour(M['C'][sl], levels=[0.5], colors='k', linewidths=0.4)
        for a_ in ax.ravel():
            a_.set_xticks([]); a_.set_yticks([])
        cbar(fig, im, ax, emax, 'signed error (sqrt)')
        fig.suptitle(f'{title}: identical camera / exposure (sqrt tone map, shared white point) and one shared error scale; contour: region C', fontsize=9)
        fig.savefig(os.path.join(out, fn), dpi=120, bbox_inches='tight'); plt.close(fig); made.append(fn)
    # ---- fig5: operators
    O = F.get('operators') or {}
    if O:
        fig, ax = plt.subplots(1, 2, figsize=(13, 4))
        for j, a in enumerate(['own'] + ARMS):
            for q, s in enumerate(SEEDS):
                o = O.get(f'{a}_s{s}')
                if o:
                    ax[0].bar(j + 0.4 * q - 0.3, o['source_panel']['peak'], 0.18, color=COL[a], alpha=0.95)
                    ax[0].bar(j + 0.4 * q - 0.1, o['receiver_panel']['peak'], 0.18, color=COL[a], alpha=0.45, hatch='//')
                    ax[1].bar(j + 0.4 * q - 0.2, o['source_minus_receiver_rel'], 0.38, color=COL[a], alpha=0.6 + 0.4 * q)
        for a_ in ax:
            a_.set_xticks(range(len(ARMS) + 1)); a_.set_xticklabels(['WL92A own'] + ARMS, fontsize=8); a_.grid(alpha=0.25, axis='y')
        ax[0].axhline(1, color='k', lw=0.6)
        ax[0].set_title('E2 T1 panel mirror-bin retention: source (solid) / receiver (hatched) operator; frozen ER = 1', fontsize=8)
        ax[1].set_title('relative source - receiver operator difference on the (intrinsically one) panel material', fontsize=8)
        fig.tight_layout(); fn = 'fig5_operators.png'; fig.savefig(os.path.join(out, fn), dpi=130); plt.close(fig); made.append(fn)
    man = {'worklog': '93B', 'source_root': R, 'files': {f: hashlib.sha256(open(os.path.join(out, f), 'rb').read()).hexdigest() for f in made},
           'provenance': 'experiments/ecro_stage93b/viz93b.py from results/ecro_stage93b (final93b.json, account.json, renders) and the WL86 / WL92B / WL93B REF-ER renders and masks'}
    json.dump(man, open(os.path.join(out, 'manifest.json'), 'w'), indent=1)
    print('VIZ', made, flush=True)


if __name__ == '__main__':
    main()
