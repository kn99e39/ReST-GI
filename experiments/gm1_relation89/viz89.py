"""WL89 reviewer visuals (no new metric). Error / change maps share ONE absolute scale per figure (sqrt or signed-sqrt, real-unit colour bar).

  python viz89.py <R89> <out_dir>
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


def hl(R, pat):
    fns = sorted(glob.glob(os.path.join(R, 'results', 'renders', pat + '_r*.npz')))
    return np.mean([np.load(f)['hl'].astype(np.float64) for f in fns], 0) if fns else None


def jl(fn):
    return json.load(open(fn)) if os.path.exists(fn) else None


def contour(ax, m, col='#1baf7a'):
    ax.contour(m.astype(float), levels=[0.5], colors=[col], linewidths=0.6)


def main():
    R, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    files = []
    s2 = jl(os.path.join(R, 'results', 'stage2_89.json'))
    s3 = jl(os.path.join(R, 'results', 'stage3_89.json'))
    ec = jl(os.path.join(R, 'results', 'econ', 'econ89.json'))

    def save(fig, name):
        fig.savefig(os.path.join(out, name), dpi=100); plt.close(fig); files.append(name)

    # 0. qualification (every candidate tried)
    cands = [os.path.basename(f)[3:-5] for f in sorted(glob.glob(os.path.join(R, 'results', 'wq_*.json')))]
    if cands:
        fig, ax = plt.subplots(len(cands), 5, figsize=(17, 3.4 * len(cands)), squeeze=False)
        for i, c in enumerate(cands):
            q0, q1 = A.load(R, f'REF_ERQ_{c}_T0'), A.load(R, f'REF_ERQ_{c}_T1')
            mz = np.load(os.path.join(R, 'results', 'masks', f'M_{c}.npz'))
            wq = jl(os.path.join(R, 'results', f'wq_{c}.json'))
            fails = [k.split('_')[0] for k, v in wq['qualification'].items() if not v['PASS']]
            dH2 = np.abs((q1['H2'] - q0['H2']).mean(0)).mean(-1); dG = np.abs((q1['H1G'] + q1['H2G'] - q0['H1G'] - q0['H2G']).mean(0)).mean(-1)
            vm = float(np.percentile(np.concatenate([dH2.ravel(), dG.ravel()]), 99.9))
            ax[i, 0].imshow(tm(q1['tot'].mean(0))); ax[i, 0].set_title(f'{c} T1 REF-ER (qualification partition)')
            ax[i, 1].imshow(np.sqrt(dH2), cmap='magma', vmin=0, vmax=np.sqrt(vm)); ax[i, 1].set_title('|T1 - T0| of H2 (sqrt, shared in row)')
            ax[i, 2].imshow(np.sqrt(dG), cmap='magma', vmin=0, vmax=np.sqrt(vm)); ax[i, 2].set_title('|T1 - T0| of H1G + H2G')
            ax[i, 3].imshow(mz['C_ss'] * 1.0 + mz['R_mirror'] * 2.0, cmap='viridis'); ax[i, 3].set_title('C_ss (1) / R_mirror (2)')
            ax[i, 4].imshow(mz['D'], cmap='gray'); ax[i, 4].set_title(f"D; ADMITTED={wq['ADMITTED']} fails {','.join(fails) or '-'}")
            for a in ax[i]:
                a.axis('off')
        fig.suptitle('WL89 workload qualification (independent partition, 8 x 8192 spp per T)')
        fig.tight_layout(); save(fig, 'fig0_workload_qualification.png')
    if not os.path.exists(os.path.join(R, 'results', 'masks', 'W1.npz')) or not glob.glob(os.path.join(R, 'results', 'renders', 'X89_R0_W1_T1_r*.npz')):
        return finish(out, files, R)
    mz = np.load(os.path.join(R, 'results', 'masks', 'W1.npz'))
    sharp = mz['C_ss'] | mz['R_mirror']
    r0, r1 = A.load(R, 'REF_ER_W1_T0'), A.load(R, 'REF_ER_W1_T1')
    arms = [('ORACLE', 'oracle table'), ('R0', 'R0 receiver (WL87 EXP)'), ('C3UR', 'C3UR sender, reused'), ('R1', 'R1 SAM'), ('R1M', 'R1M matched rays'), ('C3', 'C3 source-dependent')]
    # 1. sharp-region error maps at T1 (shared absolute scale)
    h1 = r1['h'].mean(0)
    E = {a: np.abs(hl(R, f'X89_{a}_W1_T1') - h1).mean(-1) for a, _ in arms if hl(R, f'X89_{a}_W1_T1') is not None}
    vmax = float(np.percentile(np.concatenate([v[sharp] for v in E.values()]), 99.5))
    fig, ax = plt.subplots(2, 4, figsize=(15, 7.6))
    ax = ax.ravel()
    ax[0].imshow(tm(r1['tot'].mean(0))); contour(ax[0], mz['C_ss']); contour(ax[0], mz['R_mirror'], '#eb6834'); ax[0].set_title('W1 T1 REF-ER; C_ss green, R_mirror orange')
    for i, (a, lab) in enumerate(arms):
        if a in E:
            im = ax[i + 1].imshow(np.sqrt(E[a]), cmap='magma', vmin=0, vmax=np.sqrt(vmax)); contour(ax[i + 1], sharp, '#9fd3ff')
            ax[i + 1].set_title(f'{lab}\n|hl - h|, sharp-region mean {E[a][sharp].mean():.4f}')
    for a in ax:
        a.axis('off')
    cb = fig.colorbar(im, ax=list(ax), shrink=0.6); t_ = np.linspace(0, np.sqrt(vmax), 5); cb.set_ticks(t_); cb.set_ticklabels([f'{x * x:.3g}' for x in t_])
    cb.set_label('|hl - h_REF| radiance (sqrt scale, shared)')
    save(fig, 'fig1_sharp_error_maps_T1.png')
    # 2. signed change maps (T1 - T0) of the higher-order radiance
    dref = (r1['h'].mean(0) - r0['h'].mean(0)).mean(-1)
    D = {'REF-ER': dref}
    for a in ('R0', 'C3UR', 'R1'):
        x0, x1 = hl(R, f'X89_{a}_W1_T0'), hl(R, f'X89_{a}_W1_T1')
        if x0 is not None:
            D[a] = (x1 - x0).mean(-1)
    vm = float(np.percentile(np.abs(np.concatenate([v.ravel() for v in D.values()])), 99.5))
    ss = lambda x: np.sign(x) * np.sqrt(np.abs(x))
    fig, ax = plt.subplots(2, 4, figsize=(15, 7.6))
    for i, (k, v) in enumerate(D.items()):
        im = ax[0, i].imshow(ss(v), cmap='RdBu_r', vmin=-np.sqrt(vm), vmax=np.sqrt(vm)); ax[0, i].set_title(f'{k}: T1 - T0 of h' if k == 'REF-ER' else f'{k}: T1 - T0 of hl')
        if k != 'REF-ER':
            ax[1, i].imshow(ss(v - dref), cmap='RdBu_r', vmin=-np.sqrt(vm), vmax=np.sqrt(vm)); ax[1, i].set_title(f'{k}: change error'); contour(ax[1, i], sharp, '#1baf7a')
    ax[1, 0].imshow(mz['C_ss'] * 1.0 + mz['R_mirror'] * 2.0, cmap='viridis'); ax[1, 0].set_title('regions C_ss (1) / R_mirror (2)')
    for a in ax.ravel():
        a.axis('off')
    cb = fig.colorbar(im, ax=list(ax.ravel()), shrink=0.6); t_ = np.linspace(-np.sqrt(vm), np.sqrt(vm), 5); cb.set_ticks(t_)
    cb.set_ticklabels([f'{np.sign(x) * x * x:.3g}' for x in t_]); cb.set_label('signed radiance change (signed-sqrt, shared)')
    save(fig, 'fig2_signed_change_maps.png')
    # 3. relation coverage / missing important energy projected to the image (patch of the first camera hit)
    rz = os.path.join(R, 'results', 'tables', 'R_W1_T1.npz')
    if os.path.exists(rz):
        z = np.load(rz)
        pid = mz['pid']; ok = mz['frame'] & (pid >= 0)
        tot = z['imp_energy_per_receiver']
        fig, ax = plt.subplots(1, 4, figsize=(17, 4.2))
        ax[0].imshow(np.where(ok, tot[np.clip(pid, 0, len(tot) - 1)], 0) ** 0.5, cmap='viridis'); ax[0].set_title('important peaked-sender irradiance per receiver patch (sqrt)')
        mx = max(float(z[f'miss_mean_{a}'].max()) for a in ('R0', 'C3UR', 'R1'))
        for i, a in enumerate(('R0', 'C3UR', 'R1')):
            mm = z[f'miss_mean_{a}']
            frac = mm.sum() / max(tot.sum(), 1e-30)
            im = ax[i + 1].imshow(np.where(ok, mm[np.clip(pid, 0, len(mm) - 1)], 0) ** 0.5, cmap='magma', vmin=0, vmax=mx ** 0.5)
            ax[i + 1].set_title(f'{a}: missed important energy (mean over 17 draws)\nfraction {frac:.3f}')
        for a in ax:
            a.axis('off')
        fig.colorbar(im, ax=list(ax[1:]), shrink=0.7).set_label('missed irradiance per receiver patch (sqrt, shared)')
        save(fig, 'fig3_relation_coverage_missing_energy.png')
    # 4. rendered T0 / T1 comparisons (REF LOW e1 + e2 plus each arm's table term)
    fig, ax = plt.subplots(2, 4, figsize=(15, 7.6))
    for r, (t, ref) in enumerate((('T0', r0), ('T1', r1))):
        low = ref['e1'].mean(0) + ref['e2'].mean(0)
        ax[r, 0].imshow(tm(ref['tot'].mean(0))); ax[r, 0].set_title(f'{t} REF-ER')
        for i, a in enumerate(('R0', 'C3UR', 'R1')):
            x = hl(R, f'X89_{a}_W1_{t}')
            if x is not None:
                ax[r, i + 1].imshow(tm(low + x)); ax[r, i + 1].set_title(f'{t} REF LOW + {a} table term')
    for a in ax.ravel():
        a.axis('off')
    save(fig, 'fig4_rendered_T0_T1.png')
    # 5. realizations
    if s2:
        real = s2['realizations']
        fig, ax = plt.subplots(1, 4, figsize=(16, 3.8))
        for i, (cfg, k) in enumerate([('W1_T1', 'C_ss'), ('W1_T1', 'R_mirror'), ('W1_T0', 'C_ss'), ('W1_T0', 'R_mirror')]):
            if cfg not in real:
                continue
            for j, a in enumerate(('R0', 'C3UR', 'R1')):
                if k in real[cfg][a]:
                    v = real[cfg][a][k]
                    ax[i].errorbar([j], [v['E_H_mean']], yerr=[[v['E_H_mean'] - v['E_H_min']], [v['E_H_max'] - v['E_H_mean']]], fmt='o', color=C[j], capsize=4)
                    ax[i].text(j + 0.08, v['E_H_mean'], f"{v['E_H_mean']:.3f}\nsd {v['E_H_sd']:.3f}", fontsize=7)
            ax[i].axhline(0.06, color=C[5], lw=0.8, ls='--'); ax[i].set_xticks([0, 1, 2]); ax[i].set_xticklabels(['R0', 'C3UR', 'R1']); ax[i].set_xlim(-0.5, 2.8)
            ax[i].set_title(f'{cfg} {k}: E_H over 16 realizations')
        fig.tight_layout(); save(fig, 'fig5_realizations.png')
    # 6. cost-quality
    if s3 and ec:
        fig, ax = plt.subplots(1, 4, figsize=(20, 4.4))
        life = s3['lifecycle_per_edit']
        an = [a for a in ('R0', 'C3UR', 'R1', 'R1M', 'C3', 'TF256', 'TF1024', 'PT') if a in life]
        keys = sorted({k for a in an for k in life[a] if k != 'total_per_edit'})
        bottom = np.zeros(len(an))
        for i, k in enumerate(keys):
            v = np.array([life[a].get(k, 0.0) for a in an]) * 1000
            ax[0].bar(range(len(an)), v, bottom=bottom, color=C[i % len(C)], label=k); bottom += v
        ax[0].set_xticks(range(len(an))); ax[0].set_xticklabels(an); ax[0].set_ylabel('ms per edit (warm median, before rendering)'); ax[0].legend(fontsize=6)
        ax[0].set_title('per-edit lifecycle')
        for j, key in enumerate(('err_C_ss', 'err_R_mirror')):
            for i, a in enumerate(ec['ladder_median']):
                m = ec['ladder_median'][a]
                fixed = life[a]['total_per_edit'] if a in life else 0.0
                ax[1 + j].loglog([fixed + r['wall_s'] for r in m], [r[key] for r in m], '-o', ms=3, color=C[i % len(C)], label=a)
            ax[1 + j].axhline(0.06, color='k', lw=0.6, ls='--'); ax[1 + j].set_xlabel('edit-to-first-frame wall s'); ax[1 + j].set_ylabel(f'rel L1 {key[4:]} vs REF-ER')
            ax[1 + j].legend(fontsize=6); ax[1 + j].set_title(f'quality vs edit-to-first-frame, {key[4:]}')
        for i, (k, v) in enumerate(s3['scaling'].items()):
            for a, mk in (('C3UR_build_s', 'o'), ('R0_state_s', 's'), ('R1_build_s', '^'), ('C3UR_update_s', 'x')):
                ax[3].loglog(v['N'], [r[a] for r in v['rows']], '-' + mk, color=C[i * 4 % len(C) + ('C3UR_build_s', 'R0_state_s', 'R1_build_s', 'C3UR_update_s').index(a)],
                             label=f'{a} ({k})')
        ax[3].set_xlabel('patches N'); ax[3].set_ylabel('s'); ax[3].legend(fontsize=5); ax[3].set_title('construction / update scaling')
        fig.tight_layout(); save(fig, 'fig6_cost_quality_scaling.png')
    finish(out, files, R)


def finish(out, files, R):
    man = {'worklog': 89, 'source_root': R, 'files': {f: hashlib.sha256(open(os.path.join(out, f), 'rb').read()).hexdigest() for f in files},
           'provenance': 'experiments/gm1_relation89/viz89.py from results/gm1_relation89 (renders, stage JSON)'}
    json.dump(man, open(os.path.join(out, 'manifest.json'), 'w'), indent=1)
    print('VIZ', files, flush=True)


if __name__ == '__main__':
    main()
