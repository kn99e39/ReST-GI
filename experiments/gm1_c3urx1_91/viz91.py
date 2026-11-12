"""WL91 reviewer visuals (no new metric). Error / change maps share ONE absolute scale per figure (sqrt / signed-sqrt, real-unit colour bar).
  python viz91.py <R91> <R89> <out_dir>   -> figures + manifest.json (sha256)
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
ss = lambda x: np.sign(x) * np.sqrt(np.abs(x))


def hl(R, tag):
    fns = sorted(glob.glob(os.path.join(R, 'results', 'renders', tag + '_r*.npz')))
    return np.mean([np.load(f)['hl'].astype(np.float64) for f in fns], 0) if fns else None


def mname(tag):
    p = tag.split('_')
    return p[0] + ('_' + p[2] if len(p) > 2 else '')


def main():
    R, R89, out = sys.argv[1], sys.argv[2], sys.argv[3]
    os.makedirs(out, exist_ok=True)
    files = []
    fin = json.load(open(os.path.join(R, 'results', 'final91.json')))
    tags = list(fin['conditions'])

    def save(fig, name):
        fig.savefig(os.path.join(out, name), dpi=100); plt.close(fig); files.append(name)
    data = {}
    for t in tags:
        ref = A.load(R89, f'REF_ER_{t}'); mz = np.load(os.path.join(R89, 'results', 'masks', mname(t) + '.npz'))
        data[t] = (ref, mz)
    arms = [('ORACLE', 'X91_ORACLE_{t}', 'oracle table'), ('R0', 'X91_R0_{t}_s86001', 'R0 (deployed)'), ('C3UR', 'X91_C3UR_{t}_s86001', 'C3UR (deployed)'),
            ('C3URX1', 'X91_C3URX1_{t}_s86001', 'C3URX1 (deployed)'), ('C3URX1_POOL', 'X91P_C3URX1_POOL_{t}', 'C3URX1 pooled 16')]
    # 1. sharp-region error maps on the reproduction and held-out conditions (one shared scale)
    show = [t for t in ('W1_T1', 'W2_T1', 'W1_T1_LMIX89', 'W1_T1_LSPOT89') if t in data]
    E = {t: {a: np.abs(hl(R, pat.format(t=t)) - data[t][0]['h'].mean(0)).mean(-1) for a, pat, _ in arms if hl(R, pat.format(t=t)) is not None} for t in show}
    vmax = float(np.percentile(np.concatenate([v[data[t][1]['C_ss'] | data[t][1]['R_mirror']] for t in show for v in E[t].values()]), 99.5))
    fig, ax = plt.subplots(len(show), len(arms) + 1, figsize=(3.1 * (len(arms) + 1), 3.1 * len(show)), squeeze=False)
    for r, t in enumerate(show):
        ref, mz = data[t]
        sharp = mz['C_ss'] | mz['R_mirror']
        ax[r, 0].imshow(tm(ref['tot'].mean(0))); ax[r, 0].contour(mz['C_ss'].astype(float), levels=[0.5], colors=['#1baf7a'], linewidths=0.6)
        ax[r, 0].contour(mz['R_mirror'].astype(float), levels=[0.5], colors=['#eb6834'], linewidths=0.6); ax[r, 0].set_title(f'{t} REF-ER', fontsize=7)
        for c, (a, _, lab) in enumerate(arms):
            if a in E[t]:
                im = ax[r, c + 1].imshow(np.sqrt(E[t][a]), cmap='magma', vmin=0, vmax=np.sqrt(vmax)); ax[r, c + 1].contour(sharp.astype(float), levels=[0.5], colors=['#9fd3ff'], linewidths=0.5)
                rm = mz['R_mirror']
                ax[r, c + 1].set_title(f'{lab}\nR_mirror mean {E[t][a][rm].mean():.4f}' if rm.sum() > 30 else lab, fontsize=7)
    for a in ax.ravel():
        a.axis('off')
    cb = fig.colorbar(im, ax=list(ax.ravel()), shrink=0.5); t_ = np.linspace(0, np.sqrt(vmax), 5); cb.set_ticks(t_); cb.set_ticklabels([f'{x * x:.3g}' for x in t_])
    cb.set_label('|hl - h_REF| radiance (sqrt scale, shared)')
    save(fig, 'fig1_sharp_error_maps.png')
    # 2. rendered comparisons W1 / W2 (reference LOW + table term)
    rs = [t for t in ('W1_T0', 'W1_T1', 'W2_T0', 'W2_T1') if t in data]
    fig, ax = plt.subplots(len(rs), 4, figsize=(13, 3.3 * len(rs)), squeeze=False)
    for r, t in enumerate(rs):
        ref = data[t][0]; low = ref['e1'].mean(0) + ref['e2'].mean(0)
        ax[r, 0].imshow(tm(ref['tot'].mean(0))); ax[r, 0].set_title(f'{t} REF-ER')
        for c, a in enumerate(('R0', 'C3UR', 'C3URX1')):
            x = hl(R, f'X91_{a}_{t}_s86001')
            if x is not None:
                ax[r, c + 1].imshow(tm(low + x)); ax[r, c + 1].set_title(f'{t} REF LOW + {a} table term')
    for a in ax.ravel():
        a.axis('off')
    save(fig, 'fig2_rendered_W1_W2.png')
    # 3. signed change maps (T1 - T0) for the held-out edit and the reproduction edit
    for e in ('W2', 'W1'):
        if f'{e}_T0' not in data or f'{e}_T1' not in data:
            continue
        r0, mz = data[f'{e}_T0']; r1, _ = data[f'{e}_T1']
        dref = (r1['h'].mean(0) - r0['h'].mean(0)).mean(-1)
        D = {'REF-ER': dref}
        for a in ('R0', 'C3UR', 'C3URX1'):
            D[a] = (hl(R, f'X91_{a}_{e}_T1_s86001') - hl(R, f'X91_{a}_{e}_T0_s86001')).mean(-1)
        vm = float(np.percentile(np.abs(np.concatenate([v.ravel() for v in D.values()])), 99.5))
        fig, ax = plt.subplots(2, 4, figsize=(15, 7.6))
        for i, (k, v) in enumerate(D.items()):
            im = ax[0, i].imshow(ss(v), cmap='RdBu_r', vmin=-np.sqrt(vm), vmax=np.sqrt(vm)); ax[0, i].set_title(f'{e} {k}: T1 - T0')
            if k != 'REF-ER':
                ax[1, i].imshow(ss(v - dref), cmap='RdBu_r', vmin=-np.sqrt(vm), vmax=np.sqrt(vm)); ax[1, i].set_title(f'{k}: change error')
                ax[1, i].contour((mz['C_ss'] | mz['R_mirror']).astype(float), levels=[0.5], colors=['#1baf7a'], linewidths=0.5)
        ax[1, 0].imshow(mz['C_ss'] * 1.0 + mz['R_mirror'] * 2.0, cmap='viridis'); ax[1, 0].set_title('C_ss (1) / R_mirror (2)')
        for a in ax.ravel():
            a.axis('off')
        cb = fig.colorbar(im, ax=list(ax.ravel()), shrink=0.6); t_ = np.linspace(-np.sqrt(vm), np.sqrt(vm), 5); cb.set_ticks(t_)
        cb.set_ticklabels([f'{np.sign(x) * x * x:.3g}' for x in t_]); cb.set_label('signed radiance change (signed-sqrt, shared)')
        save(fig, f'fig3_signed_change_{e}.png')
    # 4. realizations per condition
    fig, ax = plt.subplots(1, len(tags), figsize=(3.6 * len(tags), 3.8), squeeze=False)
    for i, t in enumerate(tags):
        s = fin['conditions'][t]['realizations']; dep = fin['conditions'][t]['deployed']
        k = 'R_mirror' if fin['conditions'][t]['pixels'].get('R_mirror', 0) >= 1000 else 'C_ss'
        for j, a in enumerate(('R0', 'C3UR', 'C3URX1')):
            v = s[a][k]
            ax[0, i].errorbar([j], [v['E_H_mean']], yerr=[[v['E_H_mean'] - v['E_H_min']], [v['E_H_max'] - v['E_H_mean']]], fmt='o', color=C[j], capsize=4)
            ax[0, i].text(j + 0.08, v['E_H_mean'], f"{v['E_H_mean']:.3f}", fontsize=7)
        for nm, col in (('ORACLE', C[6]), ('C3UR_POOL', C[1]), ('C3URX1_POOL', C[2])):
            if nm in dep and k in dep[nm]:
                ax[0, i].axhline(dep[nm][k]['E_H'], color=col, lw=0.8, ls=':')
        ax[0, i].axhline(0.06, color=C[5], lw=0.8, ls='--'); ax[0, i].set_xticks(range(3)); ax[0, i].set_xticklabels(['R0', 'C3UR', 'C3URX1']); ax[0, i].set_xlim(-0.5, 2.8)
        ax[0, i].set_title(f'{t}\n{k} E_H, 16 draws (dotted: oracle / pooled)', fontsize=7)
    fig.tight_layout(); save(fig, 'fig4_realizations_per_condition.png')
    # 5. source-dependent maps: C3URX1 error minus C3UR error under the held-out sources
    sr = [t for t in ('W1_T1_LMIX89', 'W1_T1_LSPOT89') if t in data]
    if sr:
        Dm = {}
        for t in sr:
            h = data[t][0]['h'].mean(0)
            ex_ = np.abs(hl(R, f'X91_C3URX1_{t}_s86001') - h).mean(-1); ec_ = np.abs(hl(R, f'X91_C3UR_{t}_s86001') - h).mean(-1)
            Dm[t] = (ec_, ex_, ex_ - ec_)
        vm = float(np.percentile(np.concatenate([np.abs(x).ravel() for v in Dm.values() for x in v]), 99.7))
        fig, ax = plt.subplots(len(sr), 4, figsize=(14, 3.6 * len(sr)), squeeze=False)
        for r, t in enumerate(sr):
            ax[r, 0].imshow(tm(data[t][0]['tot'].mean(0))); ax[r, 0].set_title(f'{t} REF-ER')
            ax[r, 1].imshow(np.sqrt(Dm[t][0]), cmap='magma', vmin=0, vmax=np.sqrt(vm)); ax[r, 1].set_title('C3UR |error|')
            ax[r, 2].imshow(np.sqrt(Dm[t][1]), cmap='magma', vmin=0, vmax=np.sqrt(vm)); ax[r, 2].set_title('C3URX1 |error|')
            im = ax[r, 3].imshow(ss(Dm[t][2]), cmap='RdBu_r', vmin=-np.sqrt(vm), vmax=np.sqrt(vm)); ax[r, 3].set_title('C3URX1 |error| - C3UR |error| (red: X1 worse)')
        for a in ax.ravel():
            a.axis('off')
        cb = fig.colorbar(im, ax=list(ax.ravel()), shrink=0.6); t_ = np.linspace(-np.sqrt(vm), np.sqrt(vm), 5); cb.set_ticks(t_)
        cb.set_ticklabels([f'{np.sign(x) * x * x:.3g}' for x in t_]); cb.set_label('radiance error (sqrt / signed-sqrt, shared)')
        save(fig, 'fig5_source_dependent_maps.png')
    # 6. cost-quality curves and lifecycle / memory breakdown
    ec = fin.get('economics')
    lp = os.path.join(R, 'results', 'econ', 'ladder91.json')
    if ec and os.path.exists(lp):
        lad = json.load(open(lp))['configs']
        fig, ax = plt.subplots(1, len(lad), figsize=(5.6 * len(lad), 4.2), squeeze=False)
        fixed = {a: ec['lifecycle_ms'][a]['edit_without_render'] / 1e3 for a in ec['lifecycle_ms']}
        for i, (t, z) in enumerate(lad.items()):
            lm = z['ladder_median']
            key = 'err_R_mirror' if 'err_R_mirror' in lm['PT'][0] else 'err_C_ss'
            for arm, m in lm.items():
                base = arm.split('_s')[0]
                col = {'PT': C[6], 'R0': C[0], 'C3UR': C[1], 'C3URX1': C[2]}[base]
                ax[0, i].loglog([fixed[base] + r['wall_s'] for r in m], [r[key] for r in m], '-o', ms=2.5, color=col, lw=1.4 if arm.endswith('86001') or arm == 'PT' else 0.6,
                                label=base if arm.endswith('86001') or arm == 'PT' else None)
            for lv in (0.1, 0.06, 0.04):
                ax[0, i].axhline(lv, color='k', lw=0.4, ls='--')
            ax[0, i].set_xlabel('geometry-edit to frame, wall s'); ax[0, i].set_ylabel(f'rel L1 {key[4:]} vs REF-ER'); ax[0, i].legend(fontsize=7)
            ax[0, i].set_title(f'{t}: quality vs edit-to-frame (thin lines: other draws)', fontsize=8)
        fig.tight_layout(); save(fig, 'fig6_cost_quality_curves.png')
        L = ec['lifecycle_ms']; an = [a for a in ('C3URX1', 'C3UR', 'R0', 'PT') if a in L]
        fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))
        for j, (pk, title) in enumerate((('edit_parts_ms', 'geometry edit -> 64-spp frame (end to end)'), ('source_parts_ms', 'source change -> 64-spp frame (end to end)'))):
            keys = sorted({k for a in an for k in L[a][pk]})
            bottom = np.zeros(len(an))
            for i, k in enumerate(keys):
                v = np.array([L[a][pk].get(k, 0.0) for a in an])
                ax[j].bar(range(len(an)), v, bottom=bottom, color=C[i % len(C)], label=k); bottom += v
            ax[j].set_xticks(range(len(an))); ax[j].set_xticklabels(an); ax[j].set_ylabel('ms (warm median)'); ax[j].legend(fontsize=6); ax[j].set_title(title)
        mem = {a: L[a]['bytes'] for a in an if L[a].get('bytes')}
        keys = sorted({k for v in mem.values() for k in v})
        bottom = np.zeros(len(mem))
        for i, k in enumerate(keys):
            v = np.array([mem[a].get(k, 0) / 1e6 for a in mem])
            ax[2].bar(range(len(mem)), v, bottom=bottom, color=C[i % len(C)], label=k); bottom += v
        ax[2].set_xticks(range(len(mem))); ax[2].set_xticklabels(list(mem)); ax[2].set_ylabel('MB'); ax[2].legend(fontsize=6); ax[2].set_title('state and scratch buffers')
        fig.tight_layout(); save(fig, 'fig7_lifecycle_memory_breakdown.png')
    man = {'worklog': 91, 'source_root': R, 'files': {f: hashlib.sha256(open(os.path.join(out, f), 'rb').read()).hexdigest() for f in files},
           'provenance': 'experiments/gm1_c3urx1_91/viz91.py from results/gm1_c3urx1_91 (renders, final91.json, ladder91.json) and the WL89 references / masks'}
    json.dump(man, open(os.path.join(out, 'manifest.json'), 'w'), indent=1)
    print('VIZ', files, flush=True)


if __name__ == '__main__':
    main()
