"""WL87 reviewer visuals (no new metric). Error maps use ONE absolute scale per figure (sqrt map, real-unit colour bar).

  python viz87.py <R87> <R86> <out_dir>
Produces what the stage results allow, writes manifest.json (sha256, provenance) into out_dir.
"""
import glob, hashlib, json, math, os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'ecro86'))
import analyze86 as A  # noqa: E402
C = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#6250d6', '#e34948', '#52514e', '#0b0b0b', '#8c8b86', '#b3d1f2']
plt.rcParams.update({'font.size': 8, 'axes.edgecolor': '#52514e', 'axes.grid': True, 'grid.color': '#e4e3df', 'grid.linewidth': 0.5})
tm = lambda x: np.clip(x / (1 + x), 0, 1) ** (1 / 2.2)


def hl(root, pat):
    fns = sorted(glob.glob(os.path.join(root, 'results', 'renders', pat + '_r*.npz')))
    return np.mean([np.load(f)['hl'].astype(np.float64) for f in fns], 0) if fns else None


def main():
    R87, R86, out = sys.argv[1], sys.argv[2], sys.argv[3]
    os.makedirs(out, exist_ok=True)
    files = []
    s1 = json.load(open(os.path.join(R87, 'results', 'stage1_87.json')))
    s2p = os.path.join(R87, 'results', 'stage2_87.json'); s2 = json.load(open(s2p)) if os.path.exists(s2p) else None
    s3p = os.path.join(R87, 'results', 'stage3_87.json'); s3 = json.load(open(s3p)) if os.path.exists(s3p) else None
    ec = os.path.join(R87, 'results', 'econ', 'econ87.json'); econ = json.load(open(ec)) if os.path.exists(ec) else None
    E = [e for e in ('E1', 'E2', 'E3') if e in s1]

    def save(fig, name):
        fig.savefig(os.path.join(out, name), dpi=100); plt.close(fig); files.append(name)

    # 1. higher-order error maps at T1, shared absolute scale over all edits / arms
    arms = [('ORACLE', 'oracle table (diagnostic)'), ('ECRO_ecro_s8601', 'ECRO s8601 (learned)'), ('EXP', 'EXP (explicit, deployed state)'),
            ('RAYSEND', 'C2-RAYSEND'), ('C3', 'C3 sender-side')]
    maps = {}
    for e in E:
        root = R87 if e == 'E3' else R86
        ref = A.load(root, f'REF_ER_{e}_T1')
        h = ref['h'].mean(0)
        src = {'ORACLE': hl(R86, f'ORA_{e}_T1') if e != 'E3' else hl(R87, f'ORA87_{e}_T1'),
               'ECRO_ecro_s8601': hl(R86, f'EC_ecro_s8601_ecro_{e}_T1') if e != 'E3' else hl(R87, f'EC87_ecro_s8601_ecro_{e}_T1')}
        for a in ('EXP', 'RAYSEND', 'C3'):
            src[a] = hl(R87, f'X87_{a}_{e}_T1')
        maps[e] = {a: (v - h).mean(-1) for a, v in src.items() if v is not None}
        maps[e]['_ref'] = ref['tot'].mean(0)
    vmax = max(np.abs(m).max() for e in maps for k, m in maps[e].items() if not k.startswith('_'))
    vmax = float(np.percentile(np.concatenate([np.abs(m).ravel() for e in maps for k, m in maps[e].items() if not k.startswith('_')]), 99.9))
    fig, axs = plt.subplots(len(E), len(arms) + 1, figsize=(3.4 * (len(arms) + 1), 3.3 * len(E)))
    axs = np.atleast_2d(axs)
    for r, e in enumerate(E):
        axs[r, 0].imshow(tm(maps[e]['_ref'])); axs[r, 0].set_title(f'{e} T1 REF-ER (65,536 spp)'); axs[r, 0].axis('off')
        for c, (a, lab) in enumerate(arms):
            ax = axs[r, c + 1]
            if a in maps[e]:
                d = maps[e][a]
                im = ax.imshow(np.sign(d) * np.sqrt(np.abs(d)), cmap='RdBu_r', vmin=-math.sqrt(vmax), vmax=math.sqrt(vmax))
                eh = s1[e]['fidelity']['T1'].get(a, {})
                ax.set_title(f'{lab}\nE_H stat {eh.get("stationary_all", {}).get("E_H", float("nan")):.4f}  C {eh.get("C", {}).get("E_H", float("nan")):.4f}')
            ax.axis('off')
    fig.colorbar(im, ax=axs, fraction=0.015, label=f'signed sqrt(hl - h_ref), shared over the figure; |x| clipped at p99.9 = {vmax:.3g} (radiance)')
    fig.suptitle('Higher-order term error at T1 (blue: deficit, red: excess); same LOW estimator and seeds for every arm')
    save(fig, 'higher_order_error_T1.png')

    # 2. E_H by region class and arm (T1), with WL86 thresholds
    regs = ['stationary_all', 'A', 'C', 'D', 'obj_panel', 'obj_sphere']
    show = ['ORACLE', 'ECRO_ecro_s8601', 'ECRO_ecro_s8602', 'EXP', 'BINQ', 'ENC', 'RAYSEND', 'C3']
    fig, axs = plt.subplots(1, len(E), figsize=(6 * len(E), 3.6), squeeze=False)
    for i, e in enumerate(E):
        ax = axs[0, i]
        T = s1[e]['fidelity']['T1']
        x = np.arange(len(regs)); wdt = 0.8 / len(show)
        for j, a in enumerate(show):
            if a in T:
                ax.bar(x + j * wdt, [T[a].get(k, {}).get('E_H', np.nan) for k in regs], wdt, color=C[j], label=a)
        ax.axhline(0.06, color='k', lw=0.8, ls='--'); ax.axhline(0.03, color='k', lw=0.6, ls=':')
        ax.set_xticks(x + 0.4); ax.set_xticklabels(regs, rotation=20); ax.set_title(f'{e} T1: E_H (WL86 thresholds 0.03 stationary, 0.06 regions)')
        ax.set_ylabel('E_H (share of total radiance)')
    axs[0, 0].legend(fontsize=6, ncol=2)
    fig.tight_layout(); save(fig, 'eh_by_region_T1.png')

    # 3. realizations: region-C signed error (EXP vs ECRO through each state), deployed and pooled
    if s2:
        fig, axs = plt.subplots(1, 2 * len(E), figsize=(3.2 * 2 * len(E), 3.4), squeeze=False)
        k = 0
        for e in E:
            if e not in s2['per_edit']:
                continue
            pe = s2['per_edit'][e]
            for t in ('T0', 'T1'):
                ax = axs[0, k]; k += 1
                for j, a in enumerate(('EXP', 'ECRO')):
                    rows = pe[a]['rows']
                    v = [r[t]['C']['signed'] for r in rows[1:]]
                    ax.scatter(np.full(len(v), j) + np.linspace(-0.15, 0.15, len(v)), v, s=8, color=C[j], label=f'{a} (16 realizations)')
                    ax.scatter([j], [rows[0][t]['C']['signed']], marker='*', s=80, color='k')
                ax.scatter([2], [pe['POOL'][t]['C']['signed']], marker='D', s=30, color=C[2], label='EXP pooled 8,192 rays (diagnostic)')
                ax.scatter([3], [s1[e]['signed'][t]['C3']['C']], marker='s', s=30, color=C[3], label='C3 sender-side')
                ax.axhline(0, color='k', lw=0.6)
                ax.set_xticks([0, 1, 2, 3]); ax.set_xticklabels(['EXP', 'ECRO', 'POOL', 'C3'])
                ax.set_title(f'{e} {t} region C signed error\n(star: deployed state)'); ax.set_ylabel('sum(hl - h) / sum tot')
        axs[0, 0].legend(fontsize=6)
        fig.tight_layout(); save(fig, 'realizations_region_C.png')
        # family reading per realization
        fam = s2['family']
        fig, axs = plt.subplots(1, len(fam), figsize=(4 * len(fam), 3.2), squeeze=False)
        for i, (key, f) in enumerate(fam.items()):
            ax = axs[0, i]
            for j, nm in enumerate(('BIN_read_over_truth', 'RAY_read_over_truth')):
                v = [r[nm] for r in f['rows'][1:]]
                ax.scatter(np.full(len(v), j) + np.linspace(-0.15, 0.15, len(v)), v, s=8, color=C[j])
                ax.scatter([j], [f['rows'][0][nm]], marker='*', s=80, color='k')
            ax.axhline(1, color='k', lw=0.6)
            ax.set_xticks([0, 1]); ax.set_xticklabels(['BIN (sender table)', 'RAY (exact sender)'])
            ax.set_title(f'{key}: light->panel->Lambert family\nread / path-traced truth (star: deployed)')
        fig.tight_layout(); save(fig, 'family_reading_realizations.png')

    # 4. economics: error vs edit-to-first-frame wall time
    if econ:
        ed = [e for e in econ['edits']]
        fig, axs = plt.subplots(2, len(ed), figsize=(5 * len(ed), 7), squeeze=False)
        for i, e in enumerate(ed):
            lm = econ[e]['ladder_median']
            fx = s3[e]['fixed_wall_s'] if s3 else {}
            for row, reg in enumerate(('stationary_all', 'C')):
                ax = axs[row, i]
                for j, a in enumerate(lm):
                    xs = [fx.get(a, 0) + m['wall_s'] for m in lm[a]]
                    ax.loglog(xs, [m['err_' + reg] for m in lm[a]], marker='o', ms=2.5, color=C[j % len(C)], label=a, lw=1)
                for q in (0.10, 0.05, 0.03):
                    ax.axhline(q, color='k', lw=0.5, ls=':')
                ax.set_xlabel('edit-to-first-frame wall time (s): fixed per-edit cost + frame at the ladder spp')
                ax.set_ylabel(f'relative L1 vs REF-ER ({reg})'); ax.set_title(f'{e} T1, {reg}')
        axs[0, 0].legend(fontsize=6, ncol=2)
        fig.tight_layout(); save(fig, 'economics_frontier.png')

    # 5. per-order field error of EXP and C3 vs per-order path-traced truth
    fig, axs = plt.subplots(1, len(E), figsize=(4.5 * len(E), 3.2), squeeze=False)
    for i, e in enumerate(E):
        ax = axs[0, i]
        od = s1[e]['orders'].get('T1')
        if not od:
            continue
        for j, a in enumerate(('EXP', 'C3')):
            ax.plot([2, 3, 4, 5], [x['rel_l1'] for x in od[a]], marker='o', color=C[j], label=f'{a} rel L1')
            ax.plot([2, 3, 4, 5], [x['flux_ratio'] - 1 for x in od[a]], marker='x', ls='--', color=C[j], label=f'{a} flux ratio - 1')
        ax.plot([2, 3, 4, 5], [x['truth_noise'] for x in od['EXP']], color='k', lw=0.8, label='truth half-split noise')
        ax.set_xlabel('scattering events at the table patch'); ax.set_title(f'{e} T1 per-order table error'); ax.legend(fontsize=6)
    fig.tight_layout(); save(fig, 'orders_field_error.png')

    man = {'worklog': 87, 'source': 'experiments/gm1_explicit87/viz87.py', 'inputs': ['results/gm1_explicit87 (stage1_87.json, stage2_87.json, stage3_87.json, econ87.json)',
                                                                                         'server renders ~/rest_gm187/results/renders, ~/rest_ecro86/results/renders'],
           'files': {f: hashlib.sha256(open(os.path.join(out, f), 'rb').read()).hexdigest() for f in files}}
    json.dump(man, open(os.path.join(out, 'manifest.json'), 'w'), indent=1)
    print('viz', files)


if __name__ == '__main__':
    main()
