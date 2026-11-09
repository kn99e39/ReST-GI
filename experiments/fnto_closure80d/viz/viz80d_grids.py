"""WL80C/80D reviewer figures (post hoc visual export, no new metric):
  1. highlight slope grids used by the R3 metric: reference / control / candidate, abs-error (shared
     sqrt scale over all outputs) and log2 ratio (fixed +-2), 8 materials x 4 theta
  2. directional albedo vs theta (to 89.999 deg): reference (MC) / control / candidate (exact VNDF)
  3. sampler histograms of one mixture closure: WL80C defective vs WL80D corrected vs pdf

  python viz80d_grids.py <wl80c_smoke_dir> <wl80d_results_dir> <out_dir>
"""
import json, math, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..')); sys.path.insert(0, os.path.join(HERE, '..', '..', 'fnto_compiled80c'))
import ref80c as R
mi = R.mi_init()
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import closure80c as C
import energy80d as E
import eval80c as EV
import closure80c_dr as CD80C
import closure80d_dr as CD80D

dev = torch.device('cuda')


def main():
    c80, r80d, out = sys.argv[1:4]
    os.makedirs(out, exist_ok=True)
    d = np.load(os.path.join(c80, 'materials.npz'))['test_iid']
    zP = torch.from_numpy(np.load(os.path.join(c80, 'fit.npz'))['zP']).to(dev)
    cand = np.load(os.path.join(r80d, 'candidate.npz'))
    pP = C.decode(zP, 'passive')
    pC0 = E.decode_candidate(torch.from_numpy(cand['zC']).to(dev))
    pC = E.scaled(pC0, torch.from_numpy(cand['M']).to(dev))
    arms = {'control (WL80C)': pP, 'candidate (WL80D)': pC}
    # ---------- 1. highlight grids (same definition as eval80c.highlight_eval)
    hl = EV.P['angular']['highlight']; n = hl['grid']
    ths = hl['theta_deg']
    # reproduce WL80C eval-set phis by regenerating sets (same seed path as run80d.regen_sets)
    import run80d as RD
    _, evs, _ = RD.regen_sets(d)
    phis = evs['phis'][:, 0]
    bsdf, nb = R.textured_principled(d)
    ax_, ay_ = EV.principled_alphas(d)
    grids = {}
    for k in range(len(d)):
        for t in ths:
            wi = R.sph(np.radians(t), phis[k])[None]
            smax = np.clip(hl['extent_alpha_mult'] * max(ax_[k], ay_[k]), hl['extent_min'], hl['extent_max'])
            wo, sx, sy = EV.grid_dirs(wi, np.array([smax]), n)
            ok = wo[0, :, 2] > 1e-4
            woc = wo[0].copy(); woc[:, 2] = np.maximum(woc[:, 2], 1e-4); woc /= np.linalg.norm(woc, axis=1, keepdims=True)
            fr, _ = R.ref_eval_pdf(bsdf, nb, np.full(n * n, k), np.repeat(wi, n * n, 0), woc)
            g = {'reference': np.where(ok, fr.mean(1), np.nan)}
            for a, p in arms.items():
                with torch.no_grad():
                    pr = C.eval_closure(C.gather(p, torch.full((n * n,), k, device=dev)),
                                        torch.from_numpy(np.repeat(wi, n * n, 0).astype(np.float32)).to(dev),
                                        torch.from_numpy(woc.astype(np.float32)).to(dev)).mean(1).cpu().numpy()
                g[a] = np.where(ok, pr, np.nan)
            grids[(k, t)] = {kk: v.reshape(n, n) for kk, v in g.items()}
            grids[(k, t)]['smax'] = smax
    np.savez_compressed(os.path.join(out, 'highlight_grids.npz'),
                        **{f'm{k}_t{t}_{a.split()[0]}': v for (k, t), g in grids.items() for a, v in g.items() if a != 'smax'})
    errmax = max(np.nanmax(np.abs(g[a] - g['reference'])) for g in grids.values() for a in arms)
    for k in range(len(d)):
        fig, axs = plt.subplots(7, len(ths), figsize=(3.1 * len(ths), 19), constrained_layout=True)
        for j, t in enumerate(ths):
            g = grids[(k, t)]; s = g['smax']
            vmax = np.nanmax([g['reference'], g['control (WL80C)'], g['candidate (WL80D)']])
            vmin = max(vmax * 1e-3, 1e-6)
            ext = [-s, s, -s, s]
            for i, a in enumerate(['reference', 'control (WL80C)', 'candidate (WL80D)']):
                im = axs[i, j].imshow(np.log10(np.clip(g[a].T, vmin, None)), origin='lower', extent=ext, cmap='magma',
                                      vmin=np.log10(vmin), vmax=np.log10(vmax))
                axs[i, j].set_title(f'{a}\ntheta_i={t} deg', fontsize=8)
            fig.colorbar(im, ax=axs[0:3, j], shrink=0.6, label='log10 eval (f cos)')
            for i, a in enumerate(['control (WL80C)', 'candidate (WL80D)']):
                im = axs[3 + i, j].imshow(np.sqrt(np.abs(g[a] - g['reference']).T), origin='lower', extent=ext,
                                          cmap='inferno', vmin=0, vmax=math.sqrt(errmax))
                axs[3 + i, j].set_title(f'|{a.split()[0]} - ref| (sqrt, shared)', fontsize=8)
            cb = fig.colorbar(im, ax=axs[3:5, j], shrink=0.6)
            tk = np.linspace(0, math.sqrt(errmax), 5); cb.set_ticks(tk); cb.set_ticklabels([f'{x * x:.3g}' for x in tk])
            for i, a in enumerate(['control (WL80C)', 'candidate (WL80D)']):
                im = axs[5 + i, j].imshow(np.log2(np.clip(g[a], 1e-12, None) / np.clip(g['reference'], 1e-12, None)).T,
                                          origin='lower', extent=ext, cmap='RdBu_r', vmin=-2, vmax=2)
                axs[5 + i, j].set_title(f'log2({a.split()[0]} / ref)', fontsize=8)
            fig.colorbar(im, ax=axs[5:7, j], shrink=0.6, label='log2 ratio')
        for a_ in axs.ravel():
            a_.set_xlabel('half-vector slope x (tangent)', fontsize=7); a_.set_ylabel('slope y', fontsize=7)
        dd = d[k]
        fig.suptitle(f'material {k}: base {dd[0]:.2f}/{dd[1]:.2f}/{dd[2]:.2f} metallic {dd[3]:.2f} rough {dd[4]:.2f} aniso {dd[5]:.2f} '
                     f'clearcoat {dd[6]:.2f} gloss {dd[7]:.2f} sheen {dd[8]:.2f}  (abs-error scale max {errmax:.3g})', fontsize=9)
        fig.savefig(os.path.join(out, f'highlight_grid_material{k}.png'), dpi=80)
        plt.close(fig)
    # ---------- 2. albedo vs theta
    th = np.concatenate([np.linspace(0, 85, 35), np.linspace(85.5, 89.5, 9), [89.8, 89.9, 89.95, 89.99, 89.999]])
    u = E.strat_u(128, 256, 80470, dev)
    fig, axs = plt.subplots(2, 4, figsize=(18, 8), constrained_layout=True)
    alb = {}
    for k in range(len(d)):
        wi = E.wi_grid(th, [phis[k] * 180 / math.pi % 90], dev)
        res = {}
        for a, p in arms.items():
            A = E.lobe_albedo(p['au'][k:k + 1], p['av'][k:k + 1], p['f0'][k:k + 1], wi, u)
            res[a] = E.mixture_albedo(p['lam'][k:k + 1], A)[0].cpu().numpy()   # [3, W]
        rng = np.random.default_rng(80480 + k)
        ns = 8192
        wiN = np.repeat(wi.cpu().numpy().astype(np.float64), ns, 0)
        wo, q, f, use = R.mixture_dirs(bsdf, nb, np.full(len(wiN), k), wiN, rng)
        res['reference'] = (np.where(use[:, None], f / q[:, None], 0).reshape(len(th), ns, 3).mean(1)).T
        alb[k] = res
        ax = axs.ravel()[k]
        for a, st in [('reference', 'k-'), ('control (WL80C)', 'b--'), ('candidate (WL80D)', 'r-')]:
            ax.plot(th, res[a].max(0), st, label=f'{a} (max over RGB)')
        ax.axhline(1.0, color='gray', lw=0.8)
        ax.set_xlabel('theta_i (deg)'); ax.set_ylabel('directional albedo'); ax.set_title(f'material {k}')
        ax.axvline(89.9, color='orange', lw=0.8, ls=':', label='compile grid end 89.9')
        ax.legend(fontsize=7)
    fig.savefig(os.path.join(out, 'albedo_vs_theta.png'), dpi=80); plt.close(fig)
    np.savez_compressed(os.path.join(out, 'albedo_vs_theta.npz'), theta=th, **{f'm{k}_{a.split()[0]}': v for k, r in alb.items() for a, v in r.items()})
    # zoom on grazing
    fig, axs = plt.subplots(2, 4, figsize=(18, 8), constrained_layout=True)
    for k in range(len(d)):
        ax = axs.ravel()[k]; sel = th >= 85
        for a, st in [('reference', 'k-'), ('control (WL80C)', 'b--'), ('candidate (WL80D)', 'r-')]:
            ax.plot(90 - th[sel], alb[k][a].max(0)[sel], st, marker='.', label=a)
        ax.set_xscale('log'); ax.invert_xaxis(); ax.axhline(1.0, color='gray', lw=0.8)
        ax.set_xlabel('90 - theta_i (deg, log)'); ax.set_title(f'material {k} (grazing zoom)'); ax.legend(fontsize=7)
    fig.savefig(os.path.join(out, 'albedo_grazing_zoom.png'), dpi=80); plt.close(fig)
    # ---------- 3. sampler histograms: defective (WL80C) vs corrected (WL80D) vs pdf
    s = np.zeros(53, np.float32); s[24:38] = 0.3; s[38:45] = 1.0
    s[[0, 8, 16]] = 0.5; s[[1, 9, 17]] = 0.5; s[45] = 0.5; s[46] = 0.5; s[24] = 0.15; s[31] = 0.15
    wi1 = R.sph(np.radians(45), 0.0)
    N = 1 << 21
    rng = np.random.default_rng(80490)
    u1 = rng.random(N).astype(np.float32); u2 = rng.random((N, 2)).astype(np.float32)
    V3 = lambda a: mi.Vector3f(mi.Float(a[:, 0].astype(np.float32)), mi.Float(a[:, 1].astype(np.float32)), mi.Float(a[:, 2].astype(np.float32)))
    P2 = mi.Point2f(mi.Float(u2[:, 0]), mi.Float(u2[:, 1]))
    wiN = np.repeat(wi1[None], N, 0)
    hists = {}
    nb_t, nb_p = 45, 90
    for name, mod in [('WL80C sampler (defective)', CD80C), ('WL80D sampler (corrected)', CD80D)]:
        cl = mod.DrClosure(s)
        wo, p_, w_, v_ = cl.sample(V3(wiN), mi.Float(u1), P2)
        wo = np.stack([np.asarray(wo[i]) for i in range(3)], 1); v_ = np.asarray(v_)
        tt = np.degrees(np.arccos(np.clip(wo[v_, 2], -1, 1))); pp = np.degrees(np.arctan2(wo[v_, 1], wo[v_, 0]))
        hists[name], _, _ = np.histogram2d(tt, pp, bins=[nb_t, nb_p], range=[[0, 90], [-180, 180]])
        hists[name] /= N
    tg = (np.arange(nb_t * 6) + .5) * 90 / (nb_t * 6); pg = -180 + (np.arange(nb_p * 6) + .5) * 360 / (nb_p * 6)
    T, PH = np.meshgrid(np.radians(tg), np.radians(pg), indexing='ij')
    wq = R.sph(T.ravel(), PH.ravel())
    pq = np.asarray(CD80D.DrClosure(s).pdf(V3(np.repeat(wi1[None], len(wq), 0)), V3(wq)), np.float64)
    dA = np.sin(T.ravel()) * np.radians(90 / (nb_t * 6)) * np.radians(360 / (nb_p * 6))
    hists['expected from pdf()'] = (pq * dA).reshape(nb_t, 6, nb_p, 6).sum((1, 3))
    vmax = max(h.max() for h in hists.values())
    fig, axs = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
    for ax, (name, h) in zip(axs, hists.items()):
        im = ax.imshow(np.sqrt(h), origin='lower', extent=[-180, 180, 0, 90], aspect='auto', cmap='viridis', vmin=0, vmax=math.sqrt(vmax))
        ax.set_title(f'{name}\nmass in upper hemisphere = {h.sum():.3f}', fontsize=9); ax.set_xlabel('phi_o (deg)'); ax.set_ylabel('theta_o (deg)')
    cb = fig.colorbar(im, ax=axs, shrink=0.8); tk = np.linspace(0, math.sqrt(vmax), 5); cb.set_ticks(tk); cb.set_ticklabels([f'{x * x:.2g}' for x in tk])
    cb.set_label('probability per bin (sqrt display)')
    fig.suptitle('Mixture closure 0.5 Lambert + 0.5 GGX(alpha 0.15), wi theta 45 deg: sampled directions vs pdf()', fontsize=10)
    fig.savefig(os.path.join(out, 'sampler_fix_histograms.png'), dpi=80); plt.close(fig)
    json.dump({'abs_error_scale_max': float(errmax), 'grid_theta_deg': ths, 'phis_rad': phis.tolist()},
              open(os.path.join(out, 'viz_meta.json'), 'w'), indent=1)
    print('done', errmax, flush=True)


if __name__ == '__main__':
    main()
    sys.stdout.flush()
    os._exit(0)
