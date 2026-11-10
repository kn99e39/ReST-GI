"""WL83 reviewer visuals (no new metric): the evaluation data, fitted / learned parameters, renders and timings of the batch.

  python viz83.py <root> <w81> [--dev]
Selection (predeclared, by the learned E seed-8301 ranking on test83_iid, never by ER): best / median / worst test83_iid, median / worst
test83_hard. Error maps: absolute, sqrt scale, one shared scale per output kind (render error maps).
"""
import json, math, os, sys
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', 'fnto_compiled82'))
import viz82 as V82  # noqa: E402  (albedo_curve, cl_eval; WL82 palette)
import closure82 as CL  # noqa: E402
import closure83  # noqa: E402,F401  (robust refraction patch, see closure83.py)
import data83 as D83  # noqa: E402
DA = V82.DA

P83 = json.load(open(os.path.join(HERE, 'protocol', 'compiled83.json')))
DFS = int(os.environ.get('W83_DF_STEPS', '12000'))   # dev smoke only overrides
dev_t = torch.device('cuda')
INK = V82.INK
S0 = P83['compiler']['seeds'][0]
SHOW = [('DF_E', 'E direct fit', '#eb6834', '-'), ('L_E', 'E learned', '#eb6834', '--'), ('DF_ER', 'ER direct fit', '#1baf7a', '-'),
        ('L_ER', 'ER learned', '#1baf7a', '--'), ('DF_C0', 'C0 direct fit', '#2a78d6', ':')]


def params(R, s):
    ps = {}
    for t, _, _, _ in SHOW:
        kind, arm = t.split('_')
        fn = os.path.join(R, 'fit', f'fit_{s}_{arm}_cert_metric_{DFS}_generic.npz' if kind == 'DF' else f'fit_{s}_L_{arm}_s{S0}.npz')
        with torch.no_grad():
            ps[t] = (arm, CL.decode(arm, 'cert', torch.from_numpy(np.load(fn)['z']).to(dev_t)))
    return ps


def main():
    root, w81 = sys.argv[1], sys.argv[2]
    R = os.path.join(root, 'results')
    out = os.path.join(R, 'viz'); os.makedirs(out, exist_ok=True)
    man = {'files': [], 'selection_rule': __doc__.split('\n')[3:5]}
    lanes = {s: np.load(os.path.join(R, 'eval', f'lanes83_{s}.npz')) for s in ['test83_iid', 'test83_hard', 'test83_block']}
    # 1. DF vs learned per-material E_L1 (transfer), test83_iid and hard
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.2))
    for ax, s in zip(axs, ['test83_iid', 'test83_hard', 'test83_block']):
        for arm, col in [('E', '#eb6834'), ('ER', '#1baf7a'), ('C0', '#2a78d6')]:
            x = lanes[s][f'DF_{arm}__el1'].mean((1, 2)); y = lanes[s][f'L_{arm}_s{S0}__el1'].mean((1, 2))
            ax.scatter(x, y, s=6, color=col, alpha=0.6, label=f'{arm}: median DF {np.median(x):.3f} / learned {np.median(y):.3f}')
        m = 0.6
        ax.plot([0, m], [0, m], color=INK, lw=0.8); ax.set_xlim(0, m); ax.set_ylim(0, m)
        ax.set_xlabel('per-material mean E_L1, direct fit'); ax.set_ylabel(f'learned (seed {S0})'); ax.set_title(s, fontsize=9); ax.legend(fontsize=6, frameon=False)
    fig.tight_layout(); fn = os.path.join(out, 'transfer_scatter.png'); fig.savefig(fn, dpi=110); plt.close(fig); man['files'].append(fn)
    # 2. training curves
    fig, ax = plt.subplots(figsize=(7, 4))
    for arm, col in [('C0', '#2a78d6'), ('E', '#eb6834'), ('ER', '#1baf7a')]:
        for sd, ls in zip(P83['compiler']['seeds'], ['-', '--']):
            j = json.load(open(os.path.join(R, 'compiler', f'train_{arm}_s{sd}.json')))
            ax.plot([l['step'] for l in j['log']], [l['val_loss_mean'] for l in j['log']], color=col, ls=ls, label=f'{arm} s{sd} (selected step {j["selected_step"]})')
    ax.set_yscale('log'); ax.set_xlabel('step'); ax.set_ylabel('validation metric objective (mean over val83 materials)'); ax.legend(fontsize=6, frameon=False)
    fig.tight_layout(); fn = os.path.join(out, 'training_curves.png'); fig.savefig(fn, dpi=110); plt.close(fig); man['files'].append(fn)
    # 3. albedo vs theta and normalised in-plane slices for the predeclared materials
    picks = {}
    for s, keys in [('test83_iid', ['best', 'median', 'worst']), ('test83_hard', ['median', 'worst'])]:
        el = lanes[s][f'L_E_s{S0}__el1'].mean((1, 2)); o = np.argsort(el)
        idx = {'best': o[0], 'median': o[len(o) // 2], 'worst': o[-1]}
        for k in keys:
            picks[f'{s}:{k}'] = (s, int(idx[k]))
    man['materials'] = {k: {'set': s, 'index': i} for k, (s, i) in picks.items()}
    thd = np.array([0, 10, 20, 30, 40, 50, 60, 70, 75, 80, 84, 87, 89])
    fig, axs = plt.subplots(1, len(picks), figsize=(3.3 * len(picks), 3.2))
    figs, axss = plt.subplots(len(picks), 3, figsize=(13, 2.6 * len(picks)))
    t = np.radians(np.linspace(-89.5, 89.5, 359))
    for i, (name, (s, k)) in enumerate(picks.items()):
        d = D83.descs(os.path.join(R, 'data'), w81, s); ps = params(R, s)
        ref, cl = V82.albedo_curve(d, k, thd, ps, P83['seeds']['viz'] + 13 * i)
        ax = axs[i]
        ax.plot(thd, ref.mean(1), color=INK, lw=2, marker='o', ms=3, label='reference')
        for tg, lab, col, ls in SHOW:
            ax.plot(thd, cl[tg].mean(1), color=col, ls=ls, lw=1.2, label=lab)
        ax.set_title(f'{name} (#{k})', fontsize=8); ax.set_xlabel('theta_i (deg)')
        if i == 0:
            ax.set_ylabel('directional albedo (channel mean)'); ax.legend(fontsize=6, frameon=False)
        ths = [20, 60, 80]
        refa, cla = V82.albedo_curve(d, k, ths, ps, P83['seeds']['viz'] + 17 * i + 5)
        for c, th in enumerate(ths):
            wi = np.repeat([[math.sin(math.radians(th)), 0, math.cos(math.radians(th))]], len(t), 0)
            wo = np.stack([np.sin(t), np.zeros_like(t), np.cos(t)], 1)
            fA, fB = DA.ref_eval(d, np.full(len(t), k), wi, wo, 4096, P83['seeds']['viz'] + 1000 + 10 * i + c, 0)
            a = axss[i, c]
            a.semilogy(np.degrees(t), np.maximum((0.5 * (fA + fB) / np.maximum(refa[c], 1e-12)).mean(1), 1e-6), color=INK, lw=2, label='reference')
            for tg, lab, col, ls in SHOW:
                v = V82.cl_eval(ps[tg], np.full(len(t), k), wi, wo)
                a.semilogy(np.degrees(t), np.maximum((v / np.maximum(cla[tg][c], 1e-12)).mean(1), 1e-6), color=col, ls=ls, lw=1.1, label=lab)
            a.axvline(-th, color='#c3c2b7', lw=0.6); a.set_title(f'{name} (#{k}) theta_i={th}', fontsize=8)
            if i == 0 and c == 0:
                a.legend(fontsize=6, frameon=False)
    fig.suptitle('WL83: directional albedo vs incidence (same 4096 mixture samples per point for every curve)', fontsize=9)
    fig.tight_layout(); fn = os.path.join(out, 'albedo_vs_theta.png'); fig.savefig(fn, dpi=110); plt.close(fig); man['files'].append(fn)
    figs.suptitle('WL83: in-plane normalised outgoing flux p(wo | wi) = f cos / A(wi), channel mean', fontsize=9)
    figs.tight_layout(); fn = os.path.join(out, 'inplane_shape.png'); figs.savefig(fn, dpi=100); plt.close(figs); man['files'].append(fn)
    # 4. renders + error maps (one shared error scale)
    rn = json.load(open(os.path.join(R, 'render', 'render83.json')))
    arms = rn['arms']; spp = rn['spp_test']
    imgs = [r['image'] for r in rn['rows']]
    emax = 0.0
    for im in imgs:
        hq = np.load(os.path.join(R, 'render', f'{im}_refhq.npy'))
        for a in arms:
            emax = max(emax, float(np.abs(np.load(os.path.join(R, 'render', f'{im}_{a}{spp}.npy')) - hq).mean(2).max()))
    tm = lambda x: np.clip(x / (1 + x), 0, 1) ** (1 / 2.2)
    for half, sub in enumerate([imgs[: (len(imgs) + 1) // 2], imgs[(len(imgs) + 1) // 2:]]):
        if not sub:
            continue
        fig, axs = plt.subplots(len(sub), 2 + 2 * len(arms), figsize=(2.0 * (2 + 2 * len(arms)), 2.1 * len(sub)), squeeze=False)
        for r, im in enumerate(sub):
            hq = np.load(os.path.join(R, 'render', f'{im}_refhq.npy')); r64 = np.load(os.path.join(R, 'render', f'{im}_ref{spp}.npy'))
            axs[r, 0].imshow(tm(hq)); axs[r, 1].imshow(tm(r64))
            for c, a in enumerate(arms):
                x = np.load(os.path.join(R, 'render', f'{im}_{a}{spp}.npy'))
                axs[r, 2 + c].imshow(tm(x))
                e = axs[r, 2 + len(arms) + c].imshow(np.sqrt(np.abs(x - hq).mean(2)), cmap='viridis', vmin=0, vmax=math.sqrt(emax))
                if r == 0:
                    axs[r, 2 + c].set_title(f'{a} {spp} spp', fontsize=7); axs[r, 2 + len(arms) + c].set_title(f'|{a} - REF_HQ|', fontsize=7)
            if r == 0:
                axs[r, 0].set_title('REF_HQ', fontsize=7); axs[r, 1].set_title(f'REF {spp} spp', fontsize=7)
            axs[r, 0].text(-0.05, 0.5, im, transform=axs[r, 0].transAxes, rotation=90, va='center', ha='right', fontsize=6)
            for a_ in axs[r]:
                a_.axis('off')
        fig.colorbar(e, ax=axs[:, -1], fraction=0.02, label=f'sqrt(mean_c |err|), shared scale, max {emax:.3g}')
        fig.suptitle('WL83 renders (Mitsuba path, max depth 4, point light; tone map x/(1+x), gamma 2.2)', fontsize=9)
        fn = os.path.join(out, f'renders_{half + 1}.png'); fig.savefig(fn, dpi=95); plt.close(fig); man['files'].append(fn)
    man['render_err_max'] = emax
    # 5. GPU time per arm / op (largest io size, median over materials)
    be = json.load(open(os.path.join(R, 'bench', 'bench83.json')))
    nmax = max(r['n'] for r in be['rows'] if r['mode'] == 'io')
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.6))
    for ax, mode in zip(axs, ['io', 'fused16']):
        n = nmax if mode == 'io' else max(r['n'] for r in be['rows'] if r['mode'] == 'fused16')
        ops = ['eval', 'pdf', 'eval_pdf', 'sample']
        for j, (a, col) in enumerate([('REF_1', INK), ('C0', '#2a78d6'), ('E', '#eb6834'), ('ER', '#1baf7a')]):
            v = [np.median([r['gpu_ms_median'] for r in be['rows'] if r['arm'] == a and r['mode'] == mode and r['n'] == n and r['op'] == op] or [np.nan]) for op in ops]
            ax.bar(np.arange(4) + (j - 1.5) * 0.2, v, width=0.18, color=col, label=a)
        ax.set_xticks(range(4)); ax.set_xticklabels(ops); ax.set_yscale('log'); ax.set_ylabel('GPU ms (kernel history, median over materials)')
        ax.set_title(f'{mode}, n = {n} (REF_1 = one walk; no pdf)', fontsize=8); ax.legend(fontsize=7, frameon=False)
    fig.tight_layout(); fn = os.path.join(out, 'gpu_time.png'); fig.savefig(fn, dpi=110); plt.close(fig); man['files'].append(fn)
    json.dump(man, open(os.path.join(out, 'viz_manifest.json'), 'w'), indent=1, default=str)


if __name__ == '__main__':
    main()
    os._exit(0)
