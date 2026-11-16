"""WL98 D2 scratch (design input only, not evidence of record): angular representation capacity of the frozen S1 ER
once-scattered lobe (scene86 panel material, mats86.npz row 1) at several incidence angles.

Compares, by the number of stored values per (patch, incidence):
  (a) fixed equal-solid-angle piecewise-constant bins NZ x NPHI, solid-angle-mean (sa, M64 convention) and
      cosine-weighted (cos, flux-conserving) convention;
  (b) adaptive quadtree on the (z, phi) equal-area square, flux-conserving leaves (sparse multi-resolution, type B);
  (c) lobe mixture: fixed cosine (Lambert) component + K Gaussians in the gnomonic tangent plane centred at the
      lobe peak, fitted by weighted EM on a coarser grid, evaluated on the fine grid, energy matched (type C;
      about 6 values per Gaussian + 6 shared).
Target g(w_o) = f(w_i, w_o) cos(theta_o) (flux density per unit solid angle); readout radiance L = g / cos(theta_o).
Metrics: rel L1 of g (delivery), rel L2 of L (readout), flux ratio, peak-radiance ratio; RGB-averaged.
Uses the WL97 Track A numpy ER port read-only (experiments/audit97/A/er_np.py). numpy only, CPU.
These are capacity (best-fit, per-incidence oracle) numbers: they say nothing about learnability.
"""
import sys, os, math, json, heapq
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'experiments', 'audit97', 'A'))
import er_np as E  # noqa: E402

S1 = np.load(os.path.join(ROOT, 'experiments', 'ecro86', 'data', 'mats86.npz'))['states'][1]


def grid(nzf, npf):
    zc = (np.arange(nzf) + 0.5) / nzf
    pc = (np.arange(npf) + 0.5) / npf * 2 * math.pi - math.pi
    Z, PH = np.meshgrid(zc, pc, indexing='ij')
    R = np.sqrt(np.maximum(1 - Z * Z, 0))
    return np.stack([R * np.cos(PH), R * np.sin(PH), Z], -1), Z, (1.0 / nzf) * (2 * math.pi / npf)


NZF, NPF = 512, 2048
WO, COS, DW = grid(NZF, NPF)
WOc, COSc, DWc = grid(128, 512)


def metrics(gh, g):
    L, Lh = g / COS[..., None], gh / COS[..., None]
    l1 = (np.abs(gh - g).sum((0, 1)) / g.sum((0, 1)))
    l2 = np.sqrt(((Lh - L) ** 2).sum((0, 1)) / (L ** 2).sum((0, 1)))
    fl = gh.sum((0, 1)) / g.sum((0, 1))
    pk = Lh.reshape(-1, 3).max(0) / L.reshape(-1, 3).max(0)
    return dict(relL1=float(l1.mean()), relL2_rad=float(l2.mean()), flux=float(fl.mean()), peak=float(pk.mean()))


def bins(g, nz, nphi, conv):
    bz, bp = NZF // nz, NPF // nphi
    c = COS.reshape(nz, bz, nphi, bp, 1)
    if conv == 'sa':      # solid-angle mean of radiance, re-emitted with the exact cosine (M64 / qtable convention)
        L = (g / COS[..., None]).reshape(nz, bz, nphi, bp, 3).mean((1, 3), keepdims=True)
    else:                 # cosine-weighted mean: flux conserving per bin
        L = g.reshape(nz, bz, nphi, bp, 3).sum((1, 3), keepdims=True) / c.sum((1, 3), keepdims=True)
    return (L * c).reshape(NZF, NPF, 3)


def quadtree(g, budgets, base=(4, 16)):
    """Flux-conserving leaves; greedily split the leaf with the largest L1 error into 4 children."""
    gh = np.zeros_like(g)
    out, heap = {}, []

    def leaf(z0, z1, p0, p1):
        blk, c = g[z0:z1, p0:p1], COS[z0:z1, p0:p1, None]
        L = blk.sum((0, 1)) / c.sum((0, 1))
        gh[z0:z1, p0:p1] = L * c
        err = float(np.abs(gh[z0:z1, p0:p1] - blk).sum())
        if (z1 - z0) > 1 and (p1 - p0) > 1:
            heapq.heappush(heap, (-err, z0, z1, p0, p1))
    nz, nphi = base
    bz, bp = NZF // nz, NPF // nphi
    for i in range(nz):
        for j in range(nphi):
            leaf(i * bz, (i + 1) * bz, j * bp, (j + 1) * bp)
    M = nz * nphi
    for b in sorted(budgets):
        while M + 3 <= b and heap:
            _, z0, z1, p0, p1 = heapq.heappop(heap)
            zm, pm = (z0 + z1) // 2, (p0 + p1) // 2
            for a in ((z0, zm), (zm, z1)):
                for q in ((p0, pm), (pm, p1)):
                    leaf(a[0], a[1], q[0], q[1])
            M += 3
        out[b] = dict(leaves=M, **metrics(gh, g))
    return out


def frame(c):
    t = np.cross(c, [0, 0, 1.0]) if abs(c[2]) < 0.99 else np.array([1.0, 0, 0])
    t = t / np.linalg.norm(t)
    return t, np.cross(c, t)


def comp_dens(W, c, t, b, mu, S, nu=None):
    dc = W @ c
    ok = dc > 0.15
    y = np.stack([(W @ t) / np.maximum(dc, 1e-9), (W @ b) / np.maximum(dc, 1e-9)], -1)
    J = np.where(ok, 1.0 / np.maximum(dc, 1e-9) ** 3, 0.0)
    d = y - mu
    Si = np.linalg.inv(S)
    q = np.einsum('ni,ij,nj->n', d, Si, d)
    if nu is None:
        ker = np.exp(-0.5 * np.minimum(q, 700))
    else:
        ker = (1.0 + q / nu) ** (-(nu + 2) / 2.0)
    return np.where(ok, ker / (2 * math.pi * math.sqrt(np.linalg.det(S))) * J, 0.0), y, ok, q


def fit_set(g, rng, core_mass=0.995, cap=100000, nrest=30000):
    """Importance subset of the FINE grid for EM: the smallest point set holding core_mass of the flux (capped, reweighted)
    plus a reweighted uniform subsample of the rest, so that sub-cell lobes are resolved."""
    gs = g.sum(-1).ravel() * DW
    order = np.argsort(-gs)
    cum = np.cumsum(gs[order]) / gs.sum()
    nc = int(np.searchsorted(cum, core_mass)) + 1
    core, rest = order[:nc], order[nc:]
    wc = gs[core].copy()
    if core.size > cap:
        sel = rng.choice(core.size, cap, replace=False)
        core, wc = core[sel], wc[sel] * (nc / cap)
    sub = rng.choice(rest, min(nrest, rest.size), replace=False) if rest.size else rest
    ws = gs[sub] * (rest.size / max(sub.size, 1))
    idx = np.concatenate([core, sub]); w = np.concatenate([wc, ws])
    return idx, w / w.sum()


def mixture(gc, g, K, rng, iters=120, restarts=3, nu=None):
    """nu=None: Gaussian components; nu=2: bivariate Student-t components (a GGX NDF is exactly a nu=2 bivariate t in the
    gnomonic plane of the half vector, so this is the heavy-tailed, microfacet-shaped lobe family)."""
    idx, w = fit_set(g, rng)
    Wall = WO.reshape(-1, 3)
    W = Wall[idx]
    c = Wall[np.argmax((g.sum(-1) / COS).ravel())]
    t, b = frame(c)
    plam = W[:, 2] / math.pi
    _, y, ok, _ = comp_dens(W, c, t, b, np.zeros(2), np.eye(2))
    best = None
    for r in range(restarts):
        idx = rng.choice(np.where(ok)[0], size=K, p=w[ok] / w[ok].sum())
        mu = y[idx].copy()
        S = np.array([np.eye(2) * (2e-4 * (1 + 4 * r)) ** 2 * 50] * K)
        pi = np.full(K + 1, 1.0 / (K + 1))
        ll = -np.inf
        for it in range(iters):
            dens = np.zeros((W.shape[0], K + 1)); dens[:, 0] = plam
            qs = np.zeros((W.shape[0], K))
            for k in range(K):
                dens[:, k + 1], _, _, qs[:, k] = comp_dens(W, c, t, b, mu[k], S[k], nu)
            num = dens * pi
            tot = num.sum(1) + 1e-300
            ll = float((w * np.log(tot)).sum())
            rr = num / tot[:, None] * w[:, None]
            pi = rr.sum(0) / rr.sum()
            for k in range(K):
                wk = rr[:, k + 1] * ok
                uk = wk if nu is None else wk * (nu + 2.0) / (nu + qs[:, k])
                sw = wk.sum() + 1e-300
                mu[k] = (uk[:, None] * y).sum(0) / (uk.sum() + 1e-300)
                d = y - mu[k]
                S[k] = (uk[:, None, None] * np.einsum('ni,nj->nij', d, d)).sum(0) / sw + np.eye(2) * 1e-8
        if best is None or ll > best[0]:
            best = (ll, pi.copy(), mu.copy(), S.copy())
    _, pi, mu, S = best
    Wf = WO.reshape(-1, 3)
    dens = pi[0] * Wf[:, 2] / math.pi
    for k in range(K):
        dens = dens + pi[k + 1] * comp_dens(Wf, c, t, b, mu[k], S[k], nu)[0]
    hemi_mass = float((dens * DW).sum())       # mixture mass on the upper hemisphere (<= 1): the rest is truncated
    shape = dens.reshape(NZF, NPF) / max(hemi_mass, 1e-30)
    gh = shape[..., None] * (g.sum((0, 1)) * DW)[None, None, :]   # energy matched per channel (the learned 'mass')
    return gh, hemi_mass


def main():
    rng = np.random.default_rng(98002)
    res = {}
    for ci in [0.025, 0.05, 0.10, 0.25, 0.40, 0.80]:
        wi = np.array([math.sqrt(1 - ci * ci), 0.0, ci])
        g = E.eval_cos(S1, wi[None, None, :], WO)
        gc = None
        L = g.sum(-1) / COS
        hw = float((L > 0.5 * L.max()).sum() * DW)
        row = {'albedo_rgb': (g.sum((0, 1)) * DW).tolist(), 'half_max_solid_angle_sr': hw}
        for nz, nphi in [(4, 16), (8, 32), (16, 64), (32, 128), (64, 256)]:
            for conv in ('sa', 'cos'):
                row[f'bins_{nz}x{nphi}_{conv}'] = dict(values=nz * nphi, **metrics(bins(g, nz, nphi, conv), g))
        row['quadtree'] = {str(k): v for k, v in quadtree(g, [64, 128, 256, 512, 1024, 2048, 4096]).items()}
        for K in [1, 2, 4]:
            gh, hm = mixture(gc, g, K, rng)
            row[f'mix_K{K}'] = dict(values=6 * K + 6, hemi_mass=hm, **metrics(gh, g))
            gh, hm = mixture(gc, g, K, rng, nu=2.0)
            row[f'tmix_K{K}'] = dict(values=6 * K + 6, hemi_mass=hm, **metrics(gh, g))
        res[f'cos_i={ci}'] = row
        print(ci, 'half-max sr %.5f' % hw,
              '| 4x16 sa L1 %.3f flux %.3f | 4x16 cos L1 %.3f | 8x32 cos L1 %.3f | 32x128 cos L1 %.3f L2 %.3f'
              ' | qt1024 L1 %.3f L2 %.3f | mix1 L1 %.3f | mix2 L1 %.3f L2 %.3f pk %.2f | mix4 L1 %.3f L2 %.3f'
              ' | t1 L1 %.3f L2 %.3f pk %.2f | t2 L1 %.3f L2 %.3f | t4 L1 %.3f L2 %.3f hm %.3f | qt256 L1 %.3f qt4096 L1 %.3f' % (
                  row['bins_4x16_sa']['relL1'], row['bins_4x16_sa']['flux'], row['bins_4x16_cos']['relL1'],
                  row['bins_8x32_cos']['relL1'],
                  row['bins_32x128_cos']['relL1'], row['bins_32x128_cos']['relL2_rad'],
                  row['quadtree']['1024']['relL1'], row['quadtree']['1024']['relL2_rad'],
                  row['mix_K1']['relL1'],
                  row['mix_K2']['relL1'], row['mix_K2']['relL2_rad'], row['mix_K2']['peak'],
                  row['mix_K4']['relL1'], row['mix_K4']['relL2_rad'],
                  row['tmix_K1']['relL1'], row['tmix_K1']['relL2_rad'], row['tmix_K1']['peak'],
                  row['tmix_K2']['relL1'], row['tmix_K2']['relL2_rad'],
                  row['tmix_K4']['relL1'], row['tmix_K4']['relL2_rad'], row['tmix_K4']['hemi_mass'],
                  row['quadtree']['256']['relL1'], row['quadtree']['4096']['relL1']), flush=True)
    out = os.path.join(ROOT, 'results', 'wl98_calc', 'D2', 'angular_capacity.json')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'w', newline='\n') as fh:
        json.dump(res, fh, indent=1)
    print('wrote', out)


if __name__ == '__main__':
    main()
