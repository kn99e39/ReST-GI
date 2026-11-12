"""WL90 reviewer visuals (no new metric). Error maps share ONE absolute scale per figure (sqrt / signed-sqrt, real-unit colour bar).
  python viz90.py <R90> <R89> <out_dir>   -> figures + manifest.json (sha256)
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


def hl(R, tag):
    fns = sorted(glob.glob(os.path.join(R, 'results', 'renders', tag + '_r*.npz')))
    return np.mean([np.load(f)['hl'].astype(np.float64) for f in fns], 0) if fns else None


def main():
    R, R89, out = sys.argv[1], sys.argv[2], sys.argv[3]
    os.makedirs(out, exist_ok=True)
    files = []
    fin = json.load(open(os.path.join(R, 'results', 'final90.json')))

    def save(fig, name):
        fig.savefig(os.path.join(out, name), dpi=100); plt.close(fig); files.append(name)
    cfgs = list(fin['configs'])
    # 1. error maps: where the intermediate projection loses the sharp transport (primary configuration)
    cfg = 'W1_T1' if 'W1_T1' in cfgs else cfgs[0]
    mz = np.load(os.path.join(R89, 'results', 'masks', cfg.split('_')[0] + '.npz'))
    sharp = mz['C_ss'] | mz['R_mirror']
    ref = A.load(R89, f'REF_ER_{cfg}')
    h = ref['h'].mean(0)
    arms = [('ORACLE', 'oracle table'), ('axsx', 'X(exact, exact) = M4\nno intermediate projection'), ('M1', 'M1: first gather exact'), ('M0', 'M0: table recurrence\n(converged R0)'),
            ('a1sx', 'X(bin, exact point)\nangular projection only'), ('axs1', 'X(exact dir, patch)\nspatial projection only'), ('a1s1', 'X(bin, patch)\nprojection once')]
    E = {a: np.abs(hl(R, f'X90_{a}_{cfg}') - h).mean(-1) for a, _ in arms if hl(R, f'X90_{a}_{cfg}') is not None}
    vmax = float(np.percentile(np.concatenate([v[sharp] for v in E.values()]), 99.5))
    fig, ax = plt.subplots(2, 4, figsize=(15.5, 7.8)); ax = ax.ravel()
    ax[0].imshow(tm(ref['tot'].mean(0))); ax[0].contour(mz['C_ss'].astype(float), levels=[0.5], colors=['#1baf7a'], linewidths=0.6)
    ax[0].contour(mz['R_mirror'].astype(float), levels=[0.5], colors=['#eb6834'], linewidths=0.6); ax[0].set_title(f'{cfg} REF-ER; C_ss green, R_mirror orange')
    for i, (a, lab) in enumerate(arms):
        if a in E:
            im = ax[i + 1].imshow(np.sqrt(E[a]), cmap='magma', vmin=0, vmax=np.sqrt(vmax)); ax[i + 1].contour(sharp.astype(float), levels=[0.5], colors=['#9fd3ff'], linewidths=0.5)
            ax[i + 1].set_title(f'{lab}\nR_mirror mean {E[a][mz["R_mirror"]].mean():.4f}', fontsize=7)
    for a in ax:
        a.axis('off')
    cb = fig.colorbar(im, ax=list(ax), shrink=0.6); t_ = np.linspace(0, np.sqrt(vmax), 5); cb.set_ticks(t_); cb.set_ticklabels([f'{x * x:.3g}' for x in t_])
    cb.set_label('|hl - h_REF| radiance (sqrt scale, shared)')
    save(fig, 'fig1_error_maps_projection_factors.png')
    # 2. signed maps of the projection loss M0 - X(exact) and its factors (shared signed-sqrt scale), full frame (sharp and broad)
    ee = hl(R, f'X90_axsx_{cfg}')
    D = {'M0 - EE (all intermediate projections)': hl(R, f'X90_M0_{cfg}') - ee, 'X(bin, patch) - EE (projection once)': hl(R, f'X90_a1s1_{cfg}') - ee,
         'angular only - EE': hl(R, f'X90_a1sx_{cfg}') - ee, 'spatial only - EE': hl(R, f'X90_axs1_{cfg}') - ee, 'M1 - EE (orders >= 2 projected)': hl(R, f'X90_M1_{cfg}') - ee,
         'EE - ORACLE (not projection)': ee - hl(R, f'X90_ORACLE_{cfg}')}
    D = {k: v.mean(-1) for k, v in D.items()}
    vm = float(np.percentile(np.abs(np.concatenate([v.ravel() for v in D.values()])), 99.7))
    ss = lambda x: np.sign(x) * np.sqrt(np.abs(x))
    fig, ax = plt.subplots(2, 3, figsize=(13, 8)); ax = ax.ravel()
    for i, (k, v) in enumerate(D.items()):
        im = ax[i].imshow(ss(v), cmap='RdBu_r', vmin=-np.sqrt(vm), vmax=np.sqrt(vm)); ax[i].set_title(k, fontsize=8); ax[i].axis('off')
        ax[i].contour(sharp.astype(float), levels=[0.5], colors=['#1baf7a'], linewidths=0.5)
    cb = fig.colorbar(im, ax=list(ax), shrink=0.6); t_ = np.linspace(-np.sqrt(vm), np.sqrt(vm), 5); cb.set_ticks(t_)
    cb.set_ticklabels([f'{np.sign(x) * x * x:.3g}' for x in t_]); cb.set_label('signed table-term difference (signed-sqrt, shared)')
    save(fig, 'fig2_signed_projection_loss_maps.png')
    # 3. decomposition bars per configuration (R_mirror and C_ss E_H)
    fig, ax = plt.subplots(2, len(cfgs), figsize=(6.2 * len(cfgs), 8), squeeze=False)
    order = ['ORACLE', 'axsx', 'M3', 'M2', 'M1', 'M0', 'a16sx', 'a4sx', 'a1sx', 'axs16', 'axs4', 'axs1', 'a16s16', 'a4s4', 'a1s1', 'P1only', 'P2plus']
    lab = {'ORACLE': 'ORACLE', 'axsx': 'EE = M4', 'M0': 'M0 (table)', 'a1sx': 'ang bin', 'a4sx': 'ang 4', 'a16sx': 'ang 16', 'axs1': 'spa patch', 'axs4': 'spa 4', 'axs16': 'spa 16',
           'a1s1': 'both 1', 'a4s4': 'both 4', 'a16s16': 'both 16'}
    for j, c in enumerate(cfgs):
        st = fin['configs'][c]['stats']
        for r_, k in enumerate(('R_mirror', 'C_ss')):
            names = [n for n in order if n in st and k in st[n]]
            v = [st[n][k]['E_H'] for n in names]
            col = [C[6] if n == 'ORACLE' else C[2] if n == 'axsx' else C[0] if n.startswith('M') else C[1] if n.startswith('a') and 'sx' in n else C[3] if n.startswith('ax') else C[4]
                   for n in names]
            ax[r_, j].bar(range(len(names)), v, color=col)
            ax[r_, j].set_xticks(range(len(names))); ax[r_, j].set_xticklabels([lab.get(n, n) for n in names], rotation=60, ha='right', fontsize=7)
            ax[r_, j].set_title(f'{c} {k}: E_H (4,096 spp, common random numbers)'); ax[r_, j].axhline(st['ORACLE'][k]['E_H'], color='k', lw=0.6, ls='--')
    fig.tight_layout(); save(fig, 'fig3_gap_decomposition.png')
    # 4. per-order field error by patch class (table recurrence vs exact gather vs truth noise)
    fo = fin['configs'][cfg]['field_orders']
    cls = ['panel', 'G_or_boxA', 'floor', 'walls']
    fig, ax = plt.subplots(1, len(cls), figsize=(4.2 * len(cls), 3.8))
    for i, c in enumerate(cls):
        x = np.arange(2, 6)
        for nm, lb, col in (('O_M0', 'table recurrence M0', C[0]), ('O_a1s1', 'projection once', C[4]), ('O_a1sx', 'angular only', C[1]), ('O_axs1', 'spatial only', C[3]),
                            ('O_axsx', 'exact gather', C[2])):
            ax[i].plot(x, [fo[nm][j][c]['rel_l1'] if c in fo[nm][j] else np.nan for j in range(4)], '-o', ms=3, color=col, label=lb)
        ax[i].plot(x, [fo['O_M0'][j][c]['truth_noise'] if c in fo['O_M0'][j] else np.nan for j in range(4)], 'k--', lw=0.8, label='truth noise')
        ax[i].set_title(f'{cfg} {c}: field rel L1 per table order'); ax[i].set_xlabel('table order (events at the receiver)'); ax[i].set_xticks(x)
        if i == 0:
            ax[i].legend(fontsize=6)
    fig.tight_layout(); save(fig, 'fig4_per_order_field_error.png')
    # 5. feasibility probe: realizations and measured cost
    fz = fin.get('feasibility', {})
    if fz:
        fc = list(fz)
        fig, ax = plt.subplots(1, len(fc) + 1, figsize=(5.5 * (len(fc) + 1), 4))
        for i, c in enumerate(fc):
            s = fz[c]['realizations']
            for j, a in enumerate(('R0', 'R0X1', 'C3UR', 'C3URX1')):
                if 'R_mirror' in s.get(a, {}):
                    v = s[a]['R_mirror']
                    ax[i].errorbar([j], [v['E_H_mean']], yerr=[[v['E_H_mean'] - v['E_H_min']], [v['E_H_max'] - v['E_H_mean']]], fmt='o', color=C[j], capsize=4)
                    ax[i].text(j + 0.08, v['E_H_mean'], f"{v['E_H_mean']:.3f}", fontsize=7)
            ax[i].axhline(0.06, color=C[5], lw=0.8, ls='--'); ax[i].set_xticks(range(4)); ax[i].set_xticklabels(['R0', 'R0X1', 'C3UR', 'C3URX1']); ax[i].set_xlim(-0.5, 3.8)
            ax[i].set_title(f'{c} R_mirror: E_H over the realizations (mean, range)')
        if 'costs' in fin:
            cst = fin['costs']
            names = ['R0', 'R0X1', 'C3UR', 'C3URX1']
            a_ = ax[-1]
            a_.bar(np.arange(4) - 0.2, [cst['per_source'][n] * 1000 for n in names], 0.4, color=C[0], label='source-only update')
            a_.bar(np.arange(4) + 0.2, [cst['per_edit_excluding_geometry'][n] * 1000 for n in names], 0.4, color=C[1], label='per edit (excl. geometry update)')
            a_.set_xticks(range(4)); a_.set_xticklabels(names); a_.set_ylabel('ms (warm median)'); a_.legend(fontsize=7); a_.set_title('measured cost of the first-gather-exact probe')
        fig.tight_layout(); save(fig, 'fig5_feasibility_probe.png')
    man = {'worklog': 90, 'source_root': R, 'files': {f: hashlib.sha256(open(os.path.join(out, f), 'rb').read()).hexdigest() for f in files},
           'provenance': 'experiments/gm1_gap90/viz90.py from results/gm1_gap90 (renders, final90.json) and the WL89 references / masks'}
    json.dump(man, open(os.path.join(out, 'manifest.json'), 'w'), indent=1)
    print('VIZ', files, flush=True)


if __name__ == '__main__':
    main()
