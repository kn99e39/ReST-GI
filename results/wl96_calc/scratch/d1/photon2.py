"""Sender-directed lobe-importance quadrature: per panel cell, N_q directions drawn from the cell's exact outgoing-flux pmf on a fine
direction grid (ideal lobe sampling, systematic = stratified resampling), equal-flux photons, traced to box A. Bias + spread over seeds."""
import json, math, os
os.environ.setdefault('HP', '0.004'); os.environ.setdefault('NR', '4')
import numpy as np
import deliv96 as D
import photon96 as P
from multiprocessing import Pool
NQ = [16, 64, 256]; SEEDS = 8

def work(args):
    pts, pl, area = args
    wl, ph = D.incident(pts, pl); wloc = D.loc(wl)
    dep = {(q, s): np.zeros(54) for q in NQ for s in range(SEEDS)}
    for j in range(len(pts)):
        if ph[j] <= 0: continue
        fl = (D.er.eval(P.VL, np.broadcast_to(wloc[j], P.VL.shape)) * ph[j]).sum(1) * P.VL[:, 2] * P.DW * area
        tot = fl.sum(); cdf = np.cumsum(fl) / tot
        rng = np.random.default_rng(1000 + j)
        for q in NQ:
            for s in range(SEEDS):
                u = (np.arange(q) + rng.random()) / q                      # systematic (stratified) resampling
                k = np.minimum(np.searchsorted(cdf, u), len(cdf) - 1)
                ok, p, fa = P.hit_boxA(np.broadcast_to(pts[j] + D.N_ * 1e-4, (q, 3)), P.VW[k])
                if ok.any():
                    np.add.at(dep[(q, s)], P.patch_of(p[ok], fa[ok]) - 1130, tot / q)
    return dep

if __name__ == '__main__':
    out = {}
    area_p = np.array([np.linalg.norm(D.G.PE1[q]) * np.linalg.norm(D.G.PE2[q]) for q in D.BOXA])
    for hc in (0.05, 0.025):
        n1, n2 = int(round(1.2 / hc)), int(round(1.4 / hc))
        gy = D.y0 + (np.arange(n1) + 0.5) * hc; gz = D.z0 + (np.arange(n2) + 0.5) * hc
        YY, ZZ = np.meshgrid(gy, gz, indexing='ij')
        pts = np.stack([np.full(YY.size, -1.5), YY.ravel(), ZZ.ravel()], 1)
        for ln, pl in D.LIGHTS.items():
            ch = np.array_split(np.arange(len(pts)), 32)
            with Pool(16) as po:
                parts = po.map(work, [(pts[c], pl, hc * hc) for c in ch])
            ref = np.asarray(json.load(open('deliv96_hp40_nr4.json'))['res'][ln]['REF'])
            for q in NQ:
                exs = [0.5 * sum(pp[(q, s)] for pp in parts) * 3 / 3 / area_p for s in range(SEEDS)]
                # channel sum: fl summed channels already; exitance = rho * E summed over channels
                l1 = [float(np.abs(e - ref).sum() / ref.sum()) for e in exs]
                m = np.mean(exs, 0)
                key = f'{ln}|hc{int(hc*1000)}|nq{q}'
                out[key] = {'photons_per_light': int(len(pts) * q), 'per_draw_rel_l1_median': float(np.median(l1)), 'per_draw_rel_l1_max': float(np.max(l1)),
                            'mean_of_draws_rel_l1': float(np.abs(m - ref).sum() / ref.sum()), 'ratio_mean': float(m.sum() / ref.sum())}
                print(key, out[key], flush=True)
    json.dump(out, open('photon2.json', 'w'), indent=1)
