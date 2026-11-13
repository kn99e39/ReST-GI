"""WL93C reviewer visuals. python viz93c.py <R> <R86> <R92A> <R92B> <out>
Error maps share ONE absolute scale per figure (sqrt map, colour bar in radiance units); images share one exposure."""
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
COL = {'own': '#586e75', 'T0': '#93a1a1', 'TQ': '#2aa198', 'TR': '#b58900', 'TQR': '#d33682', 'ORA': '#268bd2', 'ORA_SRC': '#6c71c4', 'ORA_RCV': '#859900', 'ORA_PHYS': '#cb4b16'}
Y = lambda x: x.mean(-1)


def cbar(fig, im, ax, vmax, label):
    cb = fig.colorbar(im, ax=ax, fraction=0.015)
    tk = np.array([-1, -0.25, 0, 0.25, 1.0])
    cb.set_ticks(np.sign(tk) * np.sqrt(np.abs(tk))); cb.set_ticklabels([f'{v * vmax:.2g}' for v in tk]); cb.set_label(label)


def bars(ax, G, names, k, title, floor=None):
    for j, a in enumerate(names):
        for q, s in enumerate(SEEDS):
            n = f'{a}_s{s}'
            if n in G and G[n].get(k) is not None:
                ax.bar(j + 0.4 * q - 0.2, G[n][k], 0.38, color=COL[a], alpha=0.6 + 0.4 * q)
    vals = [G[f'{a}_s{s}'][k] for a in names for s in SEEDS if f'{a}_s{s}' in G and G[f'{a}_s{s}'].get(k) is not None]
    if vals:
        lo, hi = min(vals), max(vals); ax.set_ylim(lo - 0.15 * (hi - lo + 1e-9), hi + 0.15 * (hi - lo + 1e-9))
    if floor is not None:
        ax.axhline(floor, color='k', ls=':', lw=0.8, label='exact frozen-ER floor'); ax.legend(fontsize=6)
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, fontsize=7, rotation=20); ax.grid(alpha=0.25, axis='y'); ax.set_title(title, fontsize=8)


def main():
    R, R86, R92A, R92B, out = sys.argv[1:6]
    os.makedirs(out, exist_ok=True)
    F = json.load(open(os.path.join(R, 'results', 'final93c.json')))
    made = []

    def ref(tag):
        for root in (R, R92B, R92A, R86):
            if glob.glob(os.path.join(root, 'results', 'renders', f'REF_ER_{tag}_r*.npz')):
                return A.load(root, f'REF_ER_{tag}')
    # ---- fig1: Task A oracle accounting
    O = F.get('oracle_groups')
    if O:
        V = O['val']; names = ['own', 'T0', 'TQ', 'ORA', 'ORA_SRC', 'ORA_RCV', 'ORA_PHYS']
        fig, ax = plt.subplots(1, 5, figsize=(22, 4))
        bars(ax[0], V, names, 'SRC', 'source stage vs B5 (ORA_PHYS = the reference itself)')
        bars(ax[1], V, names, 'RECV_Q', 'receiver use on the physical first flight vs t2', floor=V['ORA_s8601']['RECV_Q_floor'])
        bars(ax[2], V, names, 'SINGLE', 'single flight vs t2')
        bars(ax[3], V, names, 'WHOLE', 'whole table vs t')
        for j, a in enumerate(names):
            for q, s in enumerate(SEEDS):
                n = f'{a}_s{s}'
                ax[4].bar(j + 0.4 * q - 0.3, V[n]['SINGLE_SIGNED'], 0.18, color=COL[a], alpha=0.95)
                ax[4].bar(j + 0.4 * q - 0.1, V[n]['GLOBAL_SIGNED'], 0.18, color=COL[a], alpha=0.45, hatch='//')
        ax[4].axhline(0, color='k', lw=0.6); ax[4].set_xticks(range(len(names))); ax[4].set_xticklabels(names, fontsize=7, rotation=20); ax[4].grid(alpha=0.25, axis='y')
        ax[4].set_title('signed level: single vs t2 (solid) / frozen global vs t - t2 (hatched)', fontsize=8)
        fig.suptitle('Task A (validation, 55 x 4, g00010 included): exact physical local operator at both uses + frozen global (ORA) vs learned tied arms; left s8601, right s8602', fontsize=9)
        fig.tight_layout(); fn = 'fig1_oracle_accounting.png'; fig.savefig(os.path.join(out, fn), dpi=130); plt.close(fig); made.append(fn)
    # ---- fig2: Task B accounting
    B = F.get('account_groups')
    if B:
        V = B['val']; names = ['own', 'T0', 'TQ', 'TR', 'TQR', 'ORA']
        fig, ax = plt.subplots(1, 4, figsize=(19, 4))
        bars(ax[0], V, names, 'SRC', 'source stage vs B5'); bars(ax[1], V, names, 'RECV_Q', 'receiver use on the physical first flight vs t2', floor=V['ORA_s8601']['RECV_Q_floor'])
        bars(ax[2], V, names, 'SINGLE', 'single flight vs t2'); bars(ax[3], V, names, 'WHOLE', 'whole table vs t')
        fig.suptitle('Task B (validation): tied arms under stage supervision; left s8601, right s8602', fontsize=9)
        fig.tight_layout(); fn = 'fig2_training_accounting.png'; fig.savefig(os.path.join(out, fn), dpi=130); plt.close(fig); made.append(fn)
    # ---- fig3: E2 T1 / E5 T1 sharp maps
    for tag in ('E2_T1', 'E5k995_T1'):
        rf = ref(tag); M = AN.mask_of(R, R86, tag)
        if rf is None:
            continue
        h = Y(rf['h'].mean(0)); m = M['C'] | M.get('obj_panel', M['C'])
        ys, xs = np.nonzero(m); sl = (slice(max(ys.min() - 20, 0), ys.max() + 20), slice(max(xs.min() - 20, 0), xs.max() + 20))
        show = [a for a in ('T0', 'TQ', 'TR', 'TQR', 'ORA') if AN.hl(R, f'I93C_{a}_s{SEEDS[0]}_{tag}') is not None]
        imgs = {(a, s): Y(AN.hl(R, f'I93C_{a}_s{s}_{tag}')['hl'].mean(0)) for a in show for s in SEEDS}
        emax = max(float(np.abs(x - h)[sl].max()) for x in imgs.values())
        fig, ax = plt.subplots(2, len(show), figsize=(3.4 * len(show), 6.5), squeeze=False)
        for q, s in enumerate(SEEDS):
            for j, a in enumerate(show):
                d = (imgs[(a, s)] - h)[sl]
                eh = F['renders'].get(tag, {}).get(f'{a}_s{s}', {}).get('C', {}).get('E_H')
                im = ax[q, j].imshow(np.sign(d) * np.sqrt(np.abs(d) / emax), cmap='RdBu_r', vmin=-1, vmax=1)
                ax[q, j].contour(M['C'][sl], levels=[0.5], colors='k', linewidths=0.4)
                ax[q, j].set_title(f'{a} s{s}' + (f'  C E_H {eh:.4f}' if eh is not None else ''), fontsize=8); ax[q, j].set_xticks([]); ax[q, j].set_yticks([])
        cbar(fig, im, ax, emax, 'signed higher-order error (sqrt)')
        fig.suptitle(f'{tag}: signed higher-order error vs REF-ER, one shared scale (ORA = exact local operator at both uses + frozen global; diagnostic)', fontsize=9)
        fn = f'fig3_{tag}_sharp_maps.png'; fig.savefig(os.path.join(out, fn), dpi=120, bbox_inches='tight'); plt.close(fig); made.append(fn)
    man = {'worklog': '93C', 'source_root': R, 'files': {f: hashlib.sha256(open(os.path.join(out, f), 'rb').read()).hexdigest() for f in made},
           'provenance': 'experiments/ecro_stage93c/viz93c.py from results/ecro_stage93c (final93c.json, oracle / account accounting, renders) and the WL86 / WL92B REF-ER renders and masks'}
    json.dump(man, open(os.path.join(out, 'manifest.json'), 'w'), indent=1)
    print('VIZ', made, flush=True)


if __name__ == '__main__':
    main()
