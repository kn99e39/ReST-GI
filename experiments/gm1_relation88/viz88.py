"""WL88 reviewer visuals (no new metric). Error maps share ONE absolute scale per figure (sqrt map, real-unit colour bar).

  python viz88.py <R88> <R86> <out_dir>
Produces what the stage results allow and writes manifest.json (sha256, provenance) into out_dir.
"""
import glob, hashlib, json, os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'ecro86'))
import analyze86 as A  # noqa: E402
C = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#6250d6', '#e34948', '#52514e', '#0b0b0b', '#8c8b86', '#b3d1f2']
plt.rcParams.update({'font.size': 8, 'axes.edgecolor': '#52514e', 'axes.grid': True, 'grid.color': '#e4e3df', 'grid.linewidth': 0.5})
tm = lambda x: np.clip(0.3 * x / (1 + 0.3 * x), 0, 1) ** (1 / 2.2)


def hl(root, pat):
    fns = sorted(glob.glob(os.path.join(root, 'results', 'renders', pat + '_r*.npz')))
    return np.mean([np.load(f)['hl'].astype(np.float64) for f in fns], 0) if fns else None


def jl(fn):
    return json.load(open(fn)) if os.path.exists(fn) else None


def main():
    R, R86, out = sys.argv[1], sys.argv[2], sys.argv[3]
    os.makedirs(out, exist_ok=True)
    files = []
    s1 = jl(os.path.join(R, 'results', 'stage1_88.json'))
    s2 = jl(os.path.join(R, 'results', 'stage2_88.json'))
    s3 = jl(os.path.join(R, 'results', 'stage3_88.json'))
    ec = jl(os.path.join(R, 'results', 'econ', 'econ88.json'))

    def save(fig, name):
        fig.savefig(os.path.join(out, name), dpi=100); plt.close(fig); files.append(name)

    # 1. workload: references and geometry-only masks
    try:
        r0 = A.load(R, 'REF_ER_W1_T0'); r1 = A.load(R, 'REF_ER_W1_T1')
        mz = np.load(os.path.join(R, 'results', 'masks', 'W1.npz'))
        fig, ax = plt.subplots(2, 4, figsize=(14, 7))
        ax[0, 0].imshow(tm(r0['tot'].mean(0))); ax[0, 0].set_title('W1 T0 REF-ER')
        ax[0, 1].imshow(tm(r1['tot'].mean(0))); ax[0, 1].set_title('W1 T1 REF-ER')
        ax[0, 2].imshow(tm(4 * r1['H2'].mean(0))); ax[0, 2].set_title('T1 family H2 (>= 2 S1 vertices) x4')
        ax[0, 3].imshow(tm(4 * (r1['H1G'] + r1['H2G']).mean(0))); ax[0, 3].set_title('T1 G-involving families (H1G + H2G) x4')
        for i, k in enumerate(['C_ss', 'R_mirror', 'D', 'A']):
            ax[1, i].imshow(mz[k], cmap='gray'); ax[1, i].set_title(f'mask {k} ({int(mz[k].sum())} px)')
        for a in ax.ravel():
            a.axis('off')
        fig.suptitle('WL88 workload W1 (glossy S1 slab G rotates): references and geometry-only region masks')
        fig.tight_layout(); save(fig, 'fig1_workload.png')
    except Exception as ex:                                      # noqa: BLE001
        print('fig1 skipped', ex)
    # 2. higher-order error maps at T1 (shared absolute scale per configuration row set)
    arms = [('ORACLE', 'oracle table'), ('R0', 'R0 = WL87 EXP'), ('R1', 'R1 = SAM (candidate)'), ('R1M', 'R1M matched rays'), ('C3', 'C3 source-dependent')]
    rows = []
    for cfg, root in (('W1_T1', R), ('E2_T1', R86)):
        try:
            ref = A.load(root, f'REF_ER_{cfg}')
        except AssertionError:
            continue
        h = ref['h'].mean(0)
        m = {a: hl(R, f'X88_{a}_{cfg}') for a, _ in arms}
        rows.append((cfg, ref['tot'].mean(0), {a: (v - h).mean(-1) for a, v in m.items() if v is not None}))
    if rows:
        allv = np.concatenate([np.abs(v).ravel() for _, _, d in rows for v in d.values()])
        vmax = float(np.percentile(allv, 99.9))
        fig, axs = plt.subplots(len(rows), len(arms) + 1, figsize=(3.3 * (len(arms) + 1), 3.3 * len(rows)))
        axs = np.atleast_2d(axs)
        for r, (cfg, tot, d) in enumerate(rows):
            axs[r, 0].imshow(tm(tot)); axs[r, 0].set_title(f'{cfg} REF-ER'); axs[r, 0].axis('off')
            for c, (a, lab) in enumerate(arms):
                ax = axs[r, c + 1]; ax.axis('off')
                if a in d:
                    im = ax.imshow(np.sqrt(np.abs(d[a])), cmap='magma', vmin=0, vmax=np.sqrt(vmax)); ax.set_title(f'{lab}\n|hl - h|')
        cb = fig.colorbar(im, ax=axs.ravel().tolist(), shrink=0.6)
        tk_ = np.linspace(0, np.sqrt(vmax), 5); cb.set_ticks(tk_); cb.set_ticklabels([f'{t * t:.3g}' for t in tk_])
        cb.set_label('|hl - h_REF| (radiance, sqrt scale, shared)')
        save(fig, 'fig2_error_maps_T1.png')
    # 3. realizations
    if s2:
        real = s2['realizations']
        fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
        for i, (cfg, k) in enumerate([('W1_T1', 'C_ss'), ('W1_T1', 'R_mirror'), ('E2_T1', 'C')]):
            if cfg not in real or k not in real[cfg]['R0']:
                continue
            for j, a in enumerate(('R0', 'R1')):
                v = real[cfg][a][k]
                ax[i].errorbar([j], [v['E_H_mean']], yerr=[[v['E_H_mean'] - v['E_H_min']], [v['E_H_max'] - v['E_H_mean']]], fmt='o', color=C[j], capsize=4)
                ax[i].text(j + 0.08, v['E_H_mean'], f"{v['E_H_mean']:.3f}\nsd {v['E_H_sd']:.3f}", fontsize=7)
            ax[i].axhline(0.06, color=C[5], lw=0.8, ls='--'); ax[i].set_xticks([0, 1]); ax[i].set_xticklabels(['R0 (16)', 'R1 (16)'])
            ax[i].set_title(f'{cfg} {k}: E_H over realizations (mean, range)'); ax[i].set_xlim(-0.5, 1.8)
        fig.tight_layout(); save(fig, 'fig3_realizations.png')
    # 4. Task A factors
    if s1:
        fac = s1['taskA_factors']
        items = [(cfg, k) for cfg in fac for k in fac[cfg] if (cfg == 'E2_T1' and k == 'C') or (cfg == 'W1_T1' and k in ('C_ss', 'R_mirror'))]
        if items:
            fig, ax = plt.subplots(1, len(items), figsize=(5 * len(items), 4), squeeze=False)
            for i, (cfg, k) in enumerate(items):
                d = fac[cfg][k]
                labs = list(d)
                ax[0, i].barh(range(len(labs)), [d[l]['d_signed'] for l in labs], xerr=[d[l]['noise'] for l in labs], color=C[0])
                ax[0, i].set_yticks(range(len(labs))); ax[0, i].set_yticklabels([l.split(' (')[0] for l in labs], fontsize=7)
                ax[0, i].axvline(0, color='k', lw=0.6); ax[0, i].set_title(f'{cfg} {k}: signed-error change per factor')
            fig.tight_layout(); save(fig, 'fig4_taskA_factors.png')
    # 5. lifecycle cost, ladders, scaling
    if s3 and ec:
        fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))
        life = s3['lifecycle_per_edit']
        arms_ = [a for a in ('R0', 'R1', 'R1M', 'C3', 'TF64', 'TF256', 'TF1024', 'PT') if a in life]
        keys = sorted({k for a in arms_ for k in life[a] if k != 'total_per_edit'})
        bottom = np.zeros(len(arms_))
        for i, k in enumerate(keys):
            v = np.array([life[a].get(k, 0.0) for a in arms_]) * 1000
            ax[0].bar(range(len(arms_)), v, bottom=bottom, color=C[i % len(C)], label=k); bottom += v
        ax[0].set_xticks(range(len(arms_))); ax[0].set_xticklabels(arms_); ax[0].set_ylabel('ms per edit (warm median)')
        ax[0].legend(fontsize=6); ax[0].set_title('per-edit lifecycle (before rendering)')
        for i, a in enumerate(ec['ladder_median']):
            m = ec['ladder_median'][a]
            ax[1].loglog([r['wall_s'] for r in m], [r['err_C_ss'] for r in m], '-o', ms=3, color=C[i % len(C)], label=a)
        ax[1].axhline(0.06, color='k', lw=0.6, ls='--'); ax[1].set_xlabel('render wall s'); ax[1].set_ylabel('rel L1 in C_ss (vs REF-ER)'); ax[1].legend(fontsize=6)
        ax[1].set_title('W1 T1 quality ladder, sharp region C_ss')
        for i, (k, v) in enumerate(s3['scaling'].items()):
            ax[2].loglog(v['N'], [r['R1_build_s'] for r in v['rows']], '-o', color=C[i], label=f'R1 build ({k})')
            ax[2].loglog(v['N'], [r['R0_state_s'] for r in v['rows']], '--s', color=C[i], label=f'R0 state ({k})')
            ax[2].loglog(v['N'], [r['R1_apply_J4_s'] for r in v['rows']], ':^', color=C[i], label=f'R1 apply J4 ({k})')
        ax[2].set_xlabel('patches N'); ax[2].set_ylabel('s'); ax[2].legend(fontsize=6); ax[2].set_title('scaling (cost only)')
        fig.tight_layout(); save(fig, 'fig5_costs_ladder_scaling.png')
    man = {'worklog': 88, 'source_root': R, 'files': {f: hashlib.sha256(open(os.path.join(out, f), 'rb').read()).hexdigest() for f in files},
           'provenance': 'experiments/gm1_relation88/viz88.py from results/gm1_relation88 (renders, stage JSON)'}
    json.dump(man, open(os.path.join(out, 'manifest.json'), 'w'), indent=1)
    print('VIZ', files, flush=True)


if __name__ == '__main__':
    main()
