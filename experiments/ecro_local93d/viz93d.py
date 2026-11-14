"""WL93D reviewer visuals. python viz93d.py <R> <R86> <R92A> <R92B> <out>
Error maps share ONE absolute scale per figure (sqrt map, colour bar in radiance units)."""
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
FAMS = ['own', 'T0']; SEEDS = [8601, 8602]
ARMS = ['C0', 'C1_src', 'C1_rcv', 'C1_both', 'C2_src', 'C2_rcv', 'C2_both']
COLA = {'C0': '#586e75', 'C1_src': '#b58900', 'C1_rcv': '#cb4b16', 'C1_both': '#dc322f', 'C2_src': '#6c71c4', 'C2_rcv': '#268bd2', 'C2_both': '#2aa198'}
Y = lambda x: x.mean(-1)


def cbar(fig, im, ax, vmax, label):
    cb = fig.colorbar(im, ax=ax, fraction=0.015)
    tk = np.array([-1, -0.25, 0, 0.25, 1.0])
    cb.set_ticks(np.sign(tk) * np.sqrt(np.abs(tk))); cb.set_ticklabels([f'{v * vmax:.2g}' for v in tk]); cb.set_label(label)


def ps(d, s):
    return d.get(str(s), d.get(s))


def main():
    R, R86, R92A, R92B, out = sys.argv[1:6]
    os.makedirs(out, exist_ok=True)
    F = json.load(open(os.path.join(R, 'results', 'final93d.json')))
    made = []
    FN = F['functional']
    if FN:
        mats = ['S1', 'S2', 'B', 'LAM']; probes = ['canonical', 'src', 'rcv']
        fig, ax = plt.subplots(len(FAMS), 4, figsize=(20, 4.2 * len(FAMS)), squeeze=False)
        for r, fam in enumerate(FAMS):
            f = FN['families'][fam]
            for k, (key, lab) in enumerate((('E_phys', 'error vs M64 (E_phys)'), ('D_geo', 'geometry drift (D_geo)'), ('D_seed', 'seed drift (D_seed)'), ('S_geo', 'geometry share of squared error (S_geo)'))):
                w = 0.12
                for j, p in enumerate(probes):
                    for q, s in enumerate(SEEDS):
                        v = [(f[m][p]['D_seed'] if key == 'D_seed' else ps(f[m][p]['per_seed'], s)[key]) for m in mats]
                        ax[r, k].bar(np.arange(len(mats)) + (2 * j + q - 2.5) * w, v, w, color=['#93a1a1', '#2aa198', '#d33682'][j], alpha=0.55 + 0.45 * q, label=f'{p} s{s}' if k == 0 else None)
                ax[r, k].set_xticks(range(len(mats))); ax[r, k].set_xticklabels(mats); ax[r, k].grid(alpha=0.25, axis='y')
                ax[r, k].set_title(f'{fam}: {lab}', fontsize=9)
                if key == 'D_geo':
                    ax[r, k].axhline(0.10, color='r', ls='--', lw=0.7)
                if key == 'S_geo':
                    ax[r, k].axhline(0.25, color='r', ls='--', lw=0.7)
                if key == 'E_phys':
                    ax[r, k].axhline(0.30, color='r', ls=':', lw=0.7)
            ax[r, 0].legend(fontsize=6, ncol=3)
        fig.suptitle('Task A: functional analysis of frozen learned local operators (probe bases: canonical one-hot, source-use / receiver-use material bases; red: predeclared thresholds)', fontsize=9)
        fig.tight_layout(); fn = 'fig1_functional.png'; fig.savefig(os.path.join(out, fn), dpi=120); plt.close(fig); made.append(fn)
        fig, ax = plt.subplots(1, 2, figsize=(14, 4))
        labs = []; ein, eout, din, dout = [], [], [], []
        for fam in FAMS:
            for key, v in FN['families'][fam]['subspace'].items():
                labs.append(f'{fam} {key}'); ein.append(v['e_in']); eout.append(v['e_out']); din.append(v['drift_in']); dout.append(v['drift_out'])
        x = np.arange(len(labs))
        ax[0].bar(x - 0.2, ein, 0.4, label='in-span error'); ax[0].bar(x + 0.2, eout, 0.4, label='out-of-span error')
        ax[1].bar(x - 0.2, din, 0.4, label='in-span geometry drift'); ax[1].bar(x + 0.2, dout, 0.4, label='out-of-span geometry drift')
        for a_ in ax:
            a_.set_xticks(x); a_.set_xticklabels(labs, rotation=70, fontsize=6); a_.legend(fontsize=7); a_.grid(alpha=0.25, axis='y')
        ax[0].set_title('Task C: (A - M) on evaluation incident fields, split by the material-level training span (relative to |M x|)', fontsize=8)
        ax[1].set_title('Task C: (A(g) - A(g\')) on evaluation fields, split by span', fontsize=8)
        fig.tight_layout(); fn = 'fig2_subspace.png'; fig.savefig(os.path.join(out, fn), dpi=120); plt.close(fig); made.append(fn)
    S = F.get('account_summary')
    if S:
        fig, ax = plt.subplots(1, 2, figsize=(15, 4))
        for k, key in enumerate(('SINGLE_L1', 'WHOLE_L1')):
            for j, arm in enumerate(ARMS):
                for q, (fam, s) in enumerate([(f, s) for f in FAMS for s in SEEDS]):
                    v = S.get(f'{fam}_s{s}:{arm}', {}).get('val', {}).get(key)
                    if v is not None:
                        ax[k].bar(q + 0.11 * (j - 3), v, 0.11, color=COLA[arm], label=arm if q == 0 else None)
            ax[k].set_xticks(range(4)); ax[k].set_xticklabels([f'{f} s{s}' for f in FAMS for s in SEEDS]); ax[k].grid(alpha=0.25, axis='y')
            vals = [S[x]['val'][key] for x in S if 'val' in S[x]]; ax[k].set_ylim(min(vals) * 0.97, max(vals) * 1.03)
            ax[k].set_title(f'Task B validation (55, g00010 incl.): {key.lower()} per arm', fontsize=9)
        ax[0].legend(fontsize=7, ncol=4)
        fig.tight_layout(); fn = 'fig3_interventions_tables.png'; fig.savefig(os.path.join(out, fn), dpi=120); plt.close(fig); made.append(fn)
    RD = F.get('renders') or {}
    t1 = [t for t in RD if t.endswith('_T1')]
    if t1:
        fig, ax = plt.subplots(1, len(t1), figsize=(6 * len(t1), 4), squeeze=False)
        for i, tag in enumerate(t1):
            for j, arm in enumerate(ARMS):
                for q, (fam, s) in enumerate([(f, s) for f in FAMS for s in SEEDS]):
                    m = RD[tag].get(f'{fam}_s{s}:{arm}', {}).get('C')
                    if m:
                        ax[0, i].bar(q + 0.11 * (j - 3), m['E_H'], 0.11, color=COLA[arm], yerr=m['noise_bound'], error_kw={'lw': 0.5}, label=arm if q == 0 else None)
            ax[0, i].axhline(0.06, color='r', ls='--', lw=0.7); ax[0, i].set_xticks(range(4)); ax[0, i].set_xticklabels([f'{f} s{s}' for f in FAMS for s in SEEDS], fontsize=8)
            ax[0, i].set_title(f'{tag}: region C E_H per arm (rule 0.06)', fontsize=9); ax[0, i].grid(alpha=0.25, axis='y')
        ax[0, 0].legend(fontsize=6, ncol=4)
        fig.tight_layout(); fn = 'fig4_interventions_renders.png'; fig.savefig(os.path.join(out, fn), dpi=120); plt.close(fig); made.append(fn)
        for tag in t1:
            rf = None
            for root in (R, R92B, R92A, R86):
                if glob.glob(os.path.join(root, 'results', 'renders', f'REF_ER_{tag}_r*.npz')):
                    rf = A.load(root, f'REF_ER_{tag}'); break
            M = AN.mask_of(R, R86, tag); h = Y(rf['h'].mean(0))
            m = M['C'] | M.get('obj_panel', M['C']); ys, xs = np.nonzero(m)
            sl = (slice(max(ys.min() - 20, 0), ys.max() + 20), slice(max(xs.min() - 20, 0), xs.max() + 20))
            show = ['C0', 'C1_both', 'C2_src', 'C2_rcv', 'C2_both']
            imgs = {(s, a): AN.hl(R, f'I93D_own_s{s}_{a}_{tag}') for s in SEEDS for a in show}
            imgs = {k: Y(v['hl'].mean(0)) for k, v in imgs.items() if v is not None}
            if not imgs:
                continue
            emax = max(float(np.abs(x - h)[sl].max()) for x in imgs.values())
            fig, ax = plt.subplots(2, len(show), figsize=(3.4 * len(show), 6.5), squeeze=False)
            for q, s in enumerate(SEEDS):
                for j, a in enumerate(show):
                    if (s, a) not in imgs:
                        continue
                    d = (imgs[(s, a)] - h)[sl]
                    eh = RD[tag].get(f'own_s{s}:{a}', {}).get('C', {}).get('E_H')
                    im = ax[q, j].imshow(np.sign(d) * np.sqrt(np.abs(d) / emax), cmap='RdBu_r', vmin=-1, vmax=1)
                    ax[q, j].contour(M['C'][sl], levels=[0.5], colors='k', linewidths=0.4)
                    ax[q, j].set_title(f'own s{s} {a}' + (f'  C {eh:.4f}' if eh is not None else ''), fontsize=8); ax[q, j].set_xticks([]); ax[q, j].set_yticks([])
            cbar(fig, im, ax, emax, 'signed higher-order error (sqrt)')
            fig.suptitle(f'{tag}: deployed (C0), partner-geometry operator (C1_both) and frozen M64 substitutions (C2) - signed error vs REF-ER, one shared scale', fontsize=9)
            fn = f'fig5_{tag}_maps.png'; fig.savefig(os.path.join(out, fn), dpi=120, bbox_inches='tight'); plt.close(fig); made.append(fn)
    man = {'worklog': '93D', 'source_root': R, 'files': {f: hashlib.sha256(open(os.path.join(out, f), 'rb').read()).hexdigest() for f in made},
           'provenance': 'experiments/ecro_local93d/viz93d.py from results/ecro_local93d (final93d.json, functional / account accounting, renders) and the WL86 / WL92B / WL93D REF-ER renders and masks'}
    json.dump(man, open(os.path.join(out, 'manifest.json'), 'w'), indent=1)
    print('VIZ', made, flush=True)


if __name__ == '__main__':
    main()
