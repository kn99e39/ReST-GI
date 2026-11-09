"""WL81 reviewer visuals (no new metric): the data the Stage C metrics were computed on, for selected gate materials.

  python viz81.py <data> <fitC> <evalC> <out>

1. In-plane lobe slices: reference (4096-walk mean, the evaluation target) vs control / candidate / unbounded direct fits, eval(wi, wo)
   along the incidence plane, theta_i in {20, 60, 80} deg, for 6 gate materials (median / worst / best E_L1 of the candidate and
   3 family-stratum examples). Log scale.
2. Directional albedo vs theta_i (fresh 4096-walk reference, closures by quadrature) for the same materials.
3. Highlight grids at theta_i 50 deg: reference (1024 walks) vs candidate, shared colour scale per material.
4. Per-material E_L1 histograms per arm (from the frozen lane files).
"""
import json, math, os, sys
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import closure81 as K
C = K.install()
sys.path.insert(0, os.path.join(HERE, '..', 'fnto_compiled80c'))
import eval80c as EV  # noqa: E402
import eval81 as EV81  # noqa: E402
import data81 as DA  # noqa: E402

dev_t = torch.device('cuda')
ARMS = ['control', 'candidate', 'unbounded']
COL = {'reference': 'k', 'control': 'tab:blue', 'candidate': 'tab:red', 'unbounded': 'tab:green'}


def closure_eval(p, m, wi, wo):
    with torch.no_grad():
        return K.eval_closure(C.gather(p, torch.from_numpy(np.asarray(m, np.int64)).to(dev_t)),
                              torch.from_numpy(wi.astype(np.float32)).to(dev_t), torch.from_numpy(wo.astype(np.float32)).to(dev_t)).double().cpu().numpy()


def main():
    data, fitC, evalC, out = sys.argv[1:5]
    os.makedirs(out, exist_ok=True)
    d = np.load(os.path.join(data, 'materials.npz'))['gate']
    ps = {a: EV81.compile_runtime(a, torch.from_numpy(np.load(os.path.join(fitC, f'direct_{a}_gate.npz'))['z']).to(dev_t))[0] for a in ARMS}
    lane = {a: np.load(os.path.join(evalC, f'lane_{a}_gate.npz'))['el1'] for a in ARMS}
    mm = lane['candidate'].mean((1, 2))
    order = np.argsort(mm)
    pick = {'best': int(order[0]), 'median': int(order[len(order) // 2]), 'worst': int(order[-1]),
            'sharp_base': int(np.argmin(d[:, 9])), 'thick_white': int(np.argmax(d[:, 2] * d[:, 3:6].mean(1))), 'thin_sharp_coat': int(np.argmin(d[:, 0] + d[:, 2]))}
    manifest = {'materials': {k: {'index': v, 'descriptor': d[v].tolist(), 'E_L1_mean': {a: float(lane[a][v].mean()) for a in ARMS}} for k, v in pick.items()}, 'files': []}
    # 1. in-plane slices
    ths = [20, 60, 80]
    fig, axs = plt.subplots(len(pick), 3, figsize=(15, 3.2 * len(pick)))
    t = np.radians(np.linspace(-89.5, 89.5, 359))
    for r, (name, k) in enumerate(pick.items()):
        for c, th in enumerate(ths):
            wi = np.array([[math.sin(math.radians(th)), 0, math.cos(math.radians(th))]])
            wo = np.stack([np.sin(t), np.zeros_like(t), np.cos(t)], 1)
            WI = np.repeat(wi, len(t), 0)
            fA, fB = DA.ref_eval(d, np.full(len(t), k), WI, wo, 4096, 81990, r * 10 + c)
            ref = (0.5 * (fA + fB)).mean(1)
            ax = axs[r, c]
            ax.semilogy(np.degrees(t), np.maximum(ref, 1e-6), color=COL['reference'], lw=2, label='reference (4096 walks)')
            for a in ARMS:
                v = closure_eval(ps[a], np.full(len(t), k), WI, wo).mean(1)
                ax.semilogy(np.degrees(t), np.maximum(v, 1e-6), color=COL[a], lw=1, ls='--' if a == 'unbounded' else '-', label=a)
            ax.set_title(f'{name} (#{k}) theta_i={th}', fontsize=9)
            ax.set_xlabel('signed theta_o in the incidence plane (deg; - = mirror side)', fontsize=7)
            ax.axvline(-th, color='0.8', lw=0.5)
            if r == 0 and c == 0:
                ax.legend(fontsize=7)
    fig.suptitle('WL81 Stage C: in-plane eval = f cos(theta_o), channel mean, gate-set direct fits vs reference', fontsize=11)
    fig.tight_layout()
    fn = os.path.join(out, 'stageC_inplane_slices.png'); fig.savefig(fn, dpi=110); plt.close(fig); manifest['files'].append(fn)
    # 2. albedo vs theta
    th = np.array([0, 10, 20, 30, 40, 50, 60, 70, 75, 80, 84, 87, 89])
    fig, axs = plt.subplots(2, 3, figsize=(15, 7))
    rng = np.random.default_rng(81991)
    for i, (name, k) in enumerate(pick.items()):
        ax = axs.flat[i]
        nsa = 4096
        wi1 = DA.sph(np.radians(th), np.zeros(len(th)))
        WI = np.repeat(wi1, nsa, 0); M = np.full(len(WI), k)
        wo, q, use = DA.mixture(d[M], WI, rng)
        fA, fB = DA.ref_eval(d, M, WI, wo, 1024, 81992, i)
        w = (use / q)[:, None]
        aref = ((0.5 * (fA + fB)) * w).reshape(len(th), nsa, 3).mean(1).mean(1)
        ax.plot(th, aref, 'k-o', ms=3, lw=2, label='reference')
        for a in ARMS:
            v = (closure_eval(ps[a], M, WI, wo) * w).reshape(len(th), nsa, 3).mean(1).mean(1)
            ax.plot(th, v, color=COL[a], marker='.', ls='--' if a == 'unbounded' else '-', label=a)
        ax.set_title(f'{name} (#{k})', fontsize=9); ax.set_xlabel('theta_i (deg)'); ax.set_ylabel('directional albedo (channel mean)')
        if i == 0:
            ax.legend(fontsize=7)
    fig.suptitle('WL81 Stage C: directional albedo vs incidence (same 4096 mixture samples per point for every curve)', fontsize=11)
    fig.tight_layout()
    fn = os.path.join(out, 'stageC_albedo_vs_theta.png'); fig.savefig(fn, dpi=110); plt.close(fig); manifest['files'].append(fn)
    # 3. highlight grids at 50 deg (reference vs candidate), shared colour scale per material
    hl = DA.A['highlight']; n = hl['grid']
    fig, axs = plt.subplots(len(pick), 3, figsize=(10, 3.2 * len(pick)))
    for r, (name, k) in enumerate(pick.items()):
        wi = DA.sph(np.radians([50.0]), np.zeros(1))
        smax = np.clip(hl['extent_alpha_mult'] * d[[k], 0], hl['extent_min'], hl['extent_max'])
        wo, sx, sy = EV.grid_dirs(wi, smax, n)
        woc = wo[0].copy(); woc[:, 2] = np.maximum(woc[:, 2], 1e-4); woc /= np.linalg.norm(woc, axis=1, keepdims=True)
        WI = np.repeat(wi, n * n, 0)
        fA, fB = DA.ref_eval(d, np.full(n * n, k), WI, woc, 1024, 81993, r)
        ref = (0.5 * (fA + fB)).mean(1).reshape(n, n)
        can = closure_eval(ps['candidate'], np.full(n * n, k), WI, woc).mean(1).reshape(n, n)
        vmax = max(ref.max(), can.max())
        for c, (img, ttl) in enumerate([(ref, 'reference'), (can, 'candidate'), (np.abs(can - ref), '|candidate - reference|')]):
            im = axs[r, c].imshow(np.sqrt(np.maximum(img, 0)).T, origin='lower', cmap='magma', vmin=0, vmax=math.sqrt(vmax),
                                  extent=[-smax[0], smax[0], -smax[0], smax[0]])
            axs[r, c].set_title(f'{name}: {ttl}', fontsize=8)
            plt.colorbar(im, ax=axs[r, c], fraction=0.046, label='sqrt(eval)')
    fig.suptitle('WL81 Stage C highlight grids, theta_i 50 deg (half-vector slope coordinates; sqrt scale shared per row)', fontsize=10)
    fig.tight_layout()
    fn = os.path.join(out, 'stageC_highlight_grids.png'); fig.savefig(fn, dpi=110); plt.close(fig); manifest['files'].append(fn)
    # 4. per-material E_L1 histograms
    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.linspace(0, 1.0, 51)
    for a in ARMS:
        ax.hist(lane[a].mean((1, 2)), bins=bins, histtype='step', lw=1.5, color=COL[a], label=f'{a} (median {np.median(lane[a]):.3f})')
    ax.axvline(0.10, color='k', ls=':', label='R1 median threshold 0.10')
    ax.set_xlabel('per-material mean E_L1 (gate set)'); ax.set_ylabel('materials'); ax.legend(fontsize=8)
    fig.tight_layout()
    fn = os.path.join(out, 'stageC_EL1_hist.png'); fig.savefig(fn, dpi=110); plt.close(fig); manifest['files'].append(fn)
    json.dump(manifest, open(os.path.join(out, 'viz_manifest.json'), 'w'), indent=1)
    print(json.dumps(manifest['materials'], indent=1))


if __name__ == '__main__':
    main()
    os._exit(0)
