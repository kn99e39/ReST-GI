"""WL82 reviewer visuals (no new metric): the data the WL82 metrics were computed on, for predeclared materials (protocol visuals).

  python viz82.py <root> <w81data> [--dev]
Arms shown: reference, C0_cert_metric (control), E_cert_metric, ER_cert_metric, C0_unb_shape (shape-focused R = 8).
Selection by the C0 control ranking only (protocol visuals.selection_rule). Error maps: absolute |closure - reference|, sqrt scale, one
colour scale shared by every error map of the same kind in the batch (highlight grids; sphere images).
"""
import json, math, os, sys
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import closure82 as CL  # noqa: E402
sys.path.insert(0, os.path.join(HERE, '..', 'fnto_compiled81'))
sys.path.insert(0, os.path.join(HERE, '..', 'fnto_compiled80c'))
import data81 as DA  # noqa: E402
import eval80c as EV  # noqa: E402
import data82 as D82  # noqa: E402

P82 = json.load(open(os.path.join(HERE, 'protocol', 'compiled82.json')))
dev_t = torch.device('cuda')
DEV = '--dev' in sys.argv
INK = '#0b0b0b'
SHOW = [('C0_cert_metric', 'C0 control', '#2a78d6', '-'), ('E_cert_metric', 'E (energy)', '#eb6834', '-'),
        ('ER_cert_metric', 'ER (energy + refraction)', '#1baf7a', '-'), ('C0_unb_shape', 'R8 shape-focused fit', '#eda100', '--')]
plt.rcParams.update({'font.size': 8, 'axes.edgecolor': '#52514e', 'axes.labelcolor': INK, 'xtick.color': '#52514e', 'ytick.color': '#52514e',
                     'axes.grid': True, 'grid.color': '#e4e3df', 'grid.linewidth': 0.5})


def load_params(R, setname, steps):
    ps = {}
    for t, _, _, _ in SHOW:
        arm, mode, _ = t.split('_')
        fn = os.path.join(R, 'fit', f'fit_{setname}_{t}_{steps[t]}_generic.npz')
        with torch.no_grad():
            ps[t] = (arm, CL.decode(arm, mode, torch.from_numpy(np.load(fn)['z']).to(dev_t)))
    return ps


def cl_eval(ap, m, wi, wo):
    arm, p = ap
    with torch.no_grad():
        return CL.eval_closure(arm, CL.gather(p, torch.as_tensor(np.asarray(m), device=dev_t).long()), torch.as_tensor(wi, dtype=torch.float32, device=dev_t),
                               torch.as_tensor(wo, dtype=torch.float32, device=dev_t)).double().cpu().numpy()


def albedo_curve(d, k, ths, ap_list, seed, nsa=4096, walks=1024):
    rng = np.random.default_rng(seed)
    wi1 = DA.sph(np.radians(ths), np.zeros(len(ths)))
    WI = np.repeat(wi1, nsa, 0); M = np.full(len(WI), k)
    wo, q, use = DA.mixture(d[M], WI, rng)
    fA, fB = DA.ref_eval(d, M, WI, wo, walks, seed, 0)
    w = (use / q)[:, None]
    ref = ((0.5 * (fA + fB)) * w).reshape(len(ths), nsa, 3).mean(1)
    return ref, {t: (cl_eval(ap, M, WI, wo) * w).reshape(len(ths), nsa, 3).mean(1) for t, ap in ap_list.items()}


def figure_set(R, out, setname, d, picks, ps, seed0, manifest, errmax):
    # 1. in-plane slices (eval) and normalised outgoing-flux slices p = eval / A
    ths = [20, 60, 80]
    t = np.radians(np.linspace(-89.5, 89.5, 359))
    curves = {}
    for name, k in picks.items():
        ref_a, cl_a = albedo_curve(d, k, ths, ps, seed0 + 7 * k)
        curves[name] = (ref_a, cl_a)
    for kind in ['eval', 'shape']:
        fig, axs = plt.subplots(len(picks), 3, figsize=(13, 2.6 * len(picks)), squeeze=False)
        for r, (name, k) in enumerate(picks.items()):
            ref_a, cl_a = curves[name]
            for c, th in enumerate(ths):
                wi = np.repeat([[math.sin(math.radians(th)), 0, math.cos(math.radians(th))]], len(t), 0)
                wo = np.stack([np.sin(t), np.zeros_like(t), np.cos(t)], 1)
                fA, fB = DA.ref_eval(d, np.full(len(t), k), wi, wo, 4096, seed0 + 1000 + 10 * r + c, 0)
                ref = 0.5 * (fA + fB)
                ax = axs[r, c]
                nr = ref.mean(1) if kind == 'eval' else (ref / np.maximum(ref_a[c], 1e-12)).mean(1)
                ax.semilogy(np.degrees(t), np.maximum(nr, 1e-6), color=INK, lw=2, label='reference')
                for tg, lab, col, ls in SHOW:
                    v = cl_eval(ps[tg], np.full(len(t), k), wi, wo)
                    nv = v.mean(1) if kind == 'eval' else (v / np.maximum(cl_a[tg][c], 1e-12)).mean(1)
                    ax.semilogy(np.degrees(t), np.maximum(nv, 1e-6), color=col, lw=1.2, ls=ls, label=lab)
                ax.axvline(-th, color='#c3c2b7', lw=0.6)
                ax.set_title(f'{setname} {name} (#{k})  theta_i = {th} deg', fontsize=8)
                ax.set_xlabel('signed theta_o in the incidence plane (deg; negative = mirror side)')
                if r == 0 and c == 0:
                    ax.legend(fontsize=7, frameon=False)
        ttl = 'eval = f cos(theta_o), channel mean' if kind == 'eval' else 'normalised outgoing flux p(wo | wi) = f cos / A(wi), channel mean (energy removed)'
        fig.suptitle(f'WL82 {setname}: in-plane {ttl}; fits = WL82 metric objective (shape-focused fit: shape objective)', fontsize=9)
        fig.tight_layout()
        fn = os.path.join(out, f'{setname}_inplane_{kind}.png'); fig.savefig(fn, dpi=110); plt.close(fig); manifest['files'].append(fn)
    # 2. directional albedo vs incidence
    thd = np.array([0, 10, 20, 30, 40, 50, 60, 70, 75, 80, 84, 87, 89])
    fig, axs = plt.subplots(1, len(picks), figsize=(3.2 * len(picks), 3), squeeze=False)
    for i, (name, k) in enumerate(picks.items()):
        ref_a, cl_a = albedo_curve(d, k, thd, ps, seed0 + 3 * k + 1)
        ax = axs[0, i]
        ax.plot(thd, ref_a.mean(1), color=INK, lw=2, marker='o', ms=3, label='reference')
        for tg, lab, col, ls in SHOW:
            ax.plot(thd, cl_a[tg].mean(1), color=col, lw=1.2, ls=ls, marker='.', label=lab)
        ax.set_title(f'{name} (#{k})', fontsize=8); ax.set_xlabel('theta_i (deg)')
        if i == 0:
            ax.set_ylabel('directional albedo (channel mean)'); ax.legend(fontsize=6, frameon=False)
    fig.suptitle(f'WL82 {setname}: directional albedo vs incidence (same 4096 mixture samples per point for every curve; shape fit has free energy)', fontsize=9)
    fig.tight_layout()
    fn = os.path.join(out, f'{setname}_albedo_vs_theta.png'); fig.savefig(fn, dpi=110); plt.close(fig); manifest['files'].append(fn)
    # 3. highlight grids at 50 deg + absolute error maps (shared error scale: errmax)
    hl = DA.A['highlight']; n = hl['grid']
    cols = ['reference'] + [s[0] for s in SHOW[:3]]
    fig, axs = plt.subplots(len(picks), 7, figsize=(19, 2.6 * len(picks)), squeeze=False)
    for r, (name, k) in enumerate(picks.items()):
        wi = DA.sph(np.radians([50.0]), np.zeros(1))
        smax = np.clip(hl['extent_alpha_mult'] * d[[k], 0], hl['extent_min'], hl['extent_max'])
        wo, sx, sy = EV.grid_dirs(wi, smax, n)
        woc = wo[0].copy(); woc[:, 2] = np.maximum(woc[:, 2], 1e-4); woc /= np.linalg.norm(woc, axis=1, keepdims=True)
        WI = np.repeat(wi, n * n, 0)
        fA, fB = DA.ref_eval(d, np.full(n * n, k), WI, woc, 1024, seed0 + 2000 + r, 0)
        imgs = {'reference': (0.5 * (fA + fB)).mean(1).reshape(n, n)}
        for tg in cols[1:]:
            imgs[tg] = cl_eval(ps[tg], np.full(n * n, k), WI, woc).mean(1).reshape(n, n)
        vmax = max(v.max() for v in imgs.values())
        for c, tg in enumerate(cols):
            im = axs[r, c].imshow(np.sqrt(np.maximum(imgs[tg], 0)).T, origin='lower', cmap='magma', vmin=0, vmax=math.sqrt(vmax),
                                  extent=[-smax[0], smax[0], -smax[0], smax[0]])
            axs[r, c].set_title(f'{name}: {tg}', fontsize=7); axs[r, c].grid(False)
        plt.colorbar(im, ax=axs[r, 3], fraction=0.046, label='sqrt(eval)')
        for c, tg in enumerate(cols[1:]):
            e = np.abs(imgs[tg] - imgs['reference'])
            im = axs[r, 4 + c].imshow(np.sqrt(e).T, origin='lower', cmap='viridis', vmin=0, vmax=math.sqrt(errmax),
                                      extent=[-smax[0], smax[0], -smax[0], smax[0]])
            axs[r, 4 + c].set_title(f'|{tg} - ref|', fontsize=7); axs[r, 4 + c].grid(False)
        plt.colorbar(im, ax=axs[r, 6], fraction=0.046, label=f'sqrt|err| (shared scale, max {errmax:.3g})')
        manifest.setdefault('highlight_err_max_observed', 0.0)
        manifest['highlight_err_max_observed'] = max(manifest['highlight_err_max_observed'],
                                                     max(float(np.abs(imgs[tg] - imgs['reference']).max()) for tg in cols[1:]))
    fig.suptitle(f'WL82 {setname} highlight grids, theta_i 50 deg (half-vector slope coordinates); values sqrt-scaled per row; errors on one batch-wide scale', fontsize=9)
    fig.tight_layout()
    fn = os.path.join(out, f'{setname}_highlight_grids.png'); fig.savefig(fn, dpi=100); plt.close(fig); manifest['files'].append(fn)


def main():
    root, w81 = sys.argv[1], sys.argv[2]
    R = os.path.join(root, 'results')
    out = os.path.join(R, 'viz')
    os.makedirs(out, exist_ok=True)
    steps = {t: v['steps'] for t, v in json.load(open(os.path.join(R, 'steps.json')))['arms'].items()}
    manifest = {'selection_rule': P82['visuals']['selection_rule'], 'files': [], 'materials': {}}
    ERRMAX = P82['visuals'].get('highlight_err_scale')
    lanes = {s: np.load(os.path.join(R, 'eval', f'lanes82_{s}.npz')) for s in ['gate', 'test_iid', 'real']}
    picks_all = {}
    for s in ['gate', 'test_iid']:
        d = D82.descs(w81, s)
        el = lanes[s][f'C0_cert_metric_{steps["C0_cert_metric"]}_generic__el1'].mean((1, 2))
        o = np.argsort(el)
        pk = {'best_C0': int(o[0]), 'median_C0': int(o[len(o) // 2]), 'worst_C0': int(o[-1])}
        if s == 'gate':
            pk = {'wl81_worst_139': 139, 'wl81_sharp_base_222': 222, **pk} if len(d) > 222 else pk
        else:
            import eval82 as E82
            cm = np.nonzero(E82.cluster_mask(d))[0]
            if len(cm):
                pk['cluster_worst_C0'] = int(cm[np.argmax(el[cm])])
        picks_all[s] = pk
        manifest['materials'][s] = {k: {'index': v, 'descriptor': d[v].tolist(), 'C0_E_L1_mean': float(el[v])} for k, v in pk.items()}
    # batch-wide error scale for highlight maps: first pass = max over all shown cases (computed inside figure_set on the fly is circular,
    # so the protocol scale is used if given, else a two-pass run)
    for pass_ in ([0] if ERRMAX else [0, 1]):
        errmax = ERRMAX or (manifest.get('highlight_err_max_observed') if pass_ == 1 else 1.0)
        mf = {'files': [], 'highlight_err_max_observed': 0.0} if pass_ == 0 and not ERRMAX else manifest
        for s in ['gate', 'test_iid', 'real']:
            d = D82.descs(w81, s)
            ps = load_params(R, s, steps)
            pk = picks_all.get(s) or {P82['real_materials']['list'][i]['name']: i for i in range(len(d))}
            figure_set(R, out, s, d, pk, ps, P82['seeds']['viz'] + 100 * ['gate', 'test_iid', 'real'].index(s), mf, errmax)
        if pass_ == 0 and not ERRMAX:
            manifest['highlight_err_max_observed'] = mf['highlight_err_max_observed']
    # 4. per-material distributions (test_iid)
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.2))
    for ax, key, lab in zip(axs, ['el1', 'eshape', 'ealb'], ['E_L1', 'E_shape', 'E_alb']):
        bins = np.linspace(0, 0.8 if key != 'ealb' else 0.5, 51)
        for tg, nm_, col, ls in SHOW:
            v = lanes['test_iid'][f'{tg}_{steps[tg]}_generic__{key}'].mean((1, 2))
            ax.hist(v, bins=bins, histtype='step', lw=1.4, color=col, ls=ls, label=f'{nm_} (median {np.median(v):.3f})')
        ax.axvline({'el1': 0.10, 'eshape': 0.10, 'ealb': 0.03}[key], color=INK, ls=':', lw=1, label='WL81 median threshold')
        ax.set_xlabel(f'per-material mean {lab} (test_iid)'); ax.legend(fontsize=6, frameon=False)
    axs[0].set_ylabel('materials')
    fig.tight_layout()
    fn = os.path.join(out, 'test_iid_distributions.png'); fig.savefig(fn, dpi=110); plt.close(fig); manifest['files'].append(fn)
    # 5. decomposition bars (medians over (material, wi) groups), base vs oracle, gate and test_iid
    fig, axs = plt.subplots(1, 2, figsize=(13, 3.6))
    tags = ['C0_cert_metric', 'C0_unb_metric', 'C0_unb_shape', 'E_cert_metric', 'E_unb_metric', 'Einc_unb_metric', 'ER_cert_metric', 'ER_unb_metric', 'ER_unb_shape']
    for ax, s in zip(axs, ['gate', 'test_iid']):
        ev = json.load(open(os.path.join(R, 'eval', f'eval82_{s}.json')))['arms']
        x = np.arange(len(tags))
        vals = [[ev[f'{t}_{steps[t]}_generic'][k1][k2]['median'] if k1 != 'shape' else ev[f'{t}_{steps[t]}_generic']['shape']['median']
                 for t in tags] for k1, k2 in [('summary', 'E_L1'), ('shape', None), ('summary', 'E_alb'), ('oracle', 'E_L1')]]
        for j, (lab, col) in enumerate(zip(['E_L1', 'E_shape', 'E_alb', 'oracle E_L1'], ['#2a78d6', '#eb6834', '#1baf7a', '#6250d6'])):
            ax.bar(x + (j - 1.5) * 0.2, vals[j], width=0.18, color=col, label=lab)
        ax.axhline(0.10, color=INK, ls=':', lw=1); ax.axhline(0.03, color=INK, ls='--', lw=0.6)
        ax.set_xticks(x); ax.set_xticklabels(tags, rotation=30, ha='right', fontsize=6); ax.set_title(f'{s}: medians over (material, wi)', fontsize=8)
        ax.legend(fontsize=6, frameon=False)
    fig.tight_layout()
    fn = os.path.join(out, 'decomposition_bars.png'); fig.savefig(fn, dpi=110); plt.close(fig); manifest['files'].append(fn)
    # 6. real-material spheres (direct lighting at the surface, no GI) + absolute error maps on one shared scale
    S = np.load(os.path.join(R, 'data', 'sphere82_real.npz'))
    mask, wiS, woS = S['mask'], S['wi'], S['wo']
    ref = 0.5 * (S['img'][:, :, 0] + S['img'][:, :, 1])          # [nm, nL, npx, 3]
    d = D82.real_desc(); ps = load_params(R, 'real', steps)
    nmR, nL = ref.shape[0], ref.shape[1]
    cl = {tg: np.zeros_like(ref) for tg, _, _, _ in SHOW[:3]}
    for tg in cl:
        for j in range(nL):
            ok = (woS[j][:, 2] > 1e-4) & (wiS[:, 2] > 1e-4)
            for m in range(nmR):
                cl[tg][m, j][ok] = cl_eval(ps[tg], np.full(ok.sum(), m), wiS[ok], woS[j][ok])
    emax = max(float(np.abs(cl[tg] - ref).max()) for tg in cl)
    res = mask.shape[0]

    def img(v):
        o = np.zeros((res, res, 3)); o[mask] = v
        return o
    for j in range(nL):
        fig, axs = plt.subplots(nmR, 7, figsize=(15, 2.2 * nmR), squeeze=False)
        for m in range(nmR):
            vis = [ref[m, j]] + [cl[tg][m, j] for tg in cl]
            for c, v in enumerate(vis):
                axs[m, c].imshow(np.clip(img(v) / (1 + img(v)), 0, 1) ** (1 / 2.2)); axs[m, c].axis('off')
                axs[m, c].set_title(['reference', *cl][c] if m == 0 else '', fontsize=7)
            for c, tg in enumerate(cl):
                e = np.abs(img(cl[tg][m, j]) - img(ref[m, j])).mean(2)
                im = axs[m, 4 + c].imshow(np.sqrt(e), cmap='viridis', vmin=0, vmax=math.sqrt(emax)); axs[m, 4 + c].axis('off')
                axs[m, 4 + c].set_title(f'|{tg} - ref|' if m == 0 else '', fontsize=7)
            axs[m, 0].text(-0.1, 0.5, P82['real_materials']['list'][m]['name'], transform=axs[m, 0].transAxes, rotation=90, va='center', ha='right', fontsize=6)
        fig.colorbar(im, ax=axs[:, 6], fraction=0.03, label=f'sqrt(mean_c |err|), shared scale, max {emax:.3g}')
        fig.suptitle(f'WL82 real materials, light {j + 1}: eval(view, light) per pixel (direct lighting at the surface; tone map x/(1+x), gamma 2.2)', fontsize=9)
        fn = os.path.join(out, f'real_spheres_light{j + 1}.png'); fig.savefig(fn, dpi=100); plt.close(fig); manifest['files'].append(fn)
    manifest['sphere_err_max'] = emax
    json.dump(manifest, open(os.path.join(out, 'viz_manifest.json'), 'w'), indent=1)
    print(json.dumps(manifest['materials'], indent=1))


if __name__ == '__main__':
    main()
    os._exit(0)
