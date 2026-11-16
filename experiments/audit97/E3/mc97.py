"""WL97 audit track E3: Monte Carlo self-check of the area-form quadrature in SOLID-ANGLE form (the estimator structure of the path-traced
reference: uniform point on the box-A patch, cosine-distributed direction, analytic hit on the panel rectangle, exact once-scattered ER
radiance at the hit point). It shares er97 / geo97 with quad97 but none of the area-form geometry terms (no cos cos / r^2, no dA_s grid,
no receiver grid), so it tests the Jacobian, the visibility masks and the quadrature convergence of quad97 'exact'. Visibility: box A is
convex and nothing else lies between box A, the panel and the light (val97_dr.py T_VIS).

  exitance_j = rho_A * pi * E_{x ~ U(patch), d ~ cos / pi}[ L(hit(x, d) -> x) ],  rho_A = 0.5

  python mc97.py <CODE> <OUT> <log2 samples per batch> <batches> [seed]
"""
import json, math, os, sys, time
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import er97  # noqa: E402
import geo97  # noqa: E402
import quad97 as Q  # noqa: E402

DT, DEV, T = Q.DT, Q.DEV, Q.T


def main():
    CODE, OUT, lg, nb = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
    seed = int(sys.argv[5]) if len(sys.argv) > 5 else 97001
    fx, sc, st, LP, I, pan, faces = Q.setup(CODE)
    s1 = er97.ER(st[1], DEV)
    It = T(I); LPt = T(LP)
    n = 1 << lg
    g = torch.Generator(device=DEV); g.manual_seed(seed)
    xc = float(pan['c'][0]); cy, cz = float(pan['c'][1]), float(pan['c'][2]); hy, hz = float(pan['h'][0]), float(pan['h'][1])
    bm = np.zeros((54, nb, 3)); hits = np.zeros(54)
    t0 = time.time()
    with torch.no_grad():
        for f in faces:
            O, e1, e2, area = geo97.patch_rects(f)
            t1 = T(f['a1']); t2 = T(f['a2']); nf = T(f['n'])
            for j in range(9):
                pj = f['base'] - geo97.BOXA0 + j
                for b in range(nb):
                    u = torch.rand(4, n, dtype=DT, device=DEV, generator=g)
                    X = T(O[j])[None] + u[0][:, None] * T(e1)[None] + u[1][:, None] * T(e2)[None]
                    r = torch.sqrt(u[2]); ph = 2 * math.pi * u[3]
                    d = (r * torch.cos(ph))[:, None] * t1[None] + (r * torch.sin(ph))[:, None] * t2[None] + torch.sqrt(1 - u[2])[:, None] * nf[None]
                    ok = d[:, 0] < -1e-12
                    tt = (xc - X[:, 0]) / torch.where(ok, d[:, 0], -torch.ones_like(d[:, 0]))
                    ok &= tt > 0
                    H = X + tt[:, None] * d
                    ok &= ((H[:, 1] - cy).abs() <= hy) & ((H[:, 2] - cz).abs() <= hz)
                    ii = torch.nonzero(ok).reshape(-1)
                    if ii.numel() == 0:
                        continue
                    Hh = H[ii]; dd = d[ii]
                    (lx, ly, lz), d2 = Q.light_state(Hh, LP, I)
                    fe = s1.eval((-dd[:, 1], -dd[:, 2], torch.clamp(-dd[:, 0], min=1e-12)), (lx, ly, lz))
                    for c in range(3):
                        bm[pj, b, c] = float((fe[c] / d2).sum()) * float(It[c]) * 0.5 * math.pi / n
                    hits[pj] += ii.numel()
    ex3 = bm.mean(1)                                            # [54, 3]
    tot_b = bm.sum((0, 2))                                      # [nb] batch totals
    se_patch = bm.sum(2).std(1, ddof=1) / math.sqrt(nb)
    res = {'log2_samples_per_batch': lg, 'batches': nb, 'seed': seed, 'seconds': time.time() - t0, 'total': float(ex3.sum()),
           'total_se_batch_means': float(tot_b.std(ddof=1) / math.sqrt(nb)), 'hit_fraction': (hits / (n * nb)).tolist()}
    np.savez_compressed(os.path.join(OUT, f'mc_l{lg}b{nb}s{seed}.npz'), ex3=ex3, se_patch=se_patch, batch_totals=tot_b, meta=json.dumps(res))
    print('MC', json.dumps({k: v for k, v in res.items() if k != 'hit_fraction'}), flush=True)


if __name__ == '__main__':
    main()
