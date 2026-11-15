import json, math, time
import numpy as np
from ck2_surrogate import *
z = np.load('est.npz'); Et = z['truth'].sum(1)
q = np.load('q.npy')
def draw(rng, value):
    us = rng.random((4, 512))
    E = np.zeros(9); nh = np.zeros(9)
    k = np.arange(512); bn = k // KAP; iz, ip = bn // NPHI, bn % NPHI
    zz = (iz + us[2]) / NZ; ph = (ip + us[3]) * 2 * math.pi / NPHI - math.pi; r = np.sqrt(1 - zz * zz)
    w = r[:, None] * np.cos(ph)[:, None] * fe1 + r[:, None] * np.sin(ph)[:, None] * fe2 + zz[:, None] * fn
    for p in range(9):
        iu, iv = p % 3, p // 3
        u = -0.45 + (iu + us[0]) * 0.3; v = -0.45 + (iv + us[1]) * 0.3
        o = fc + u[:, None] * fe1 + v[:, None] * fe2
        ok, y = hit_panel(o, w)
        if ok.any():
            val = value(y[ok], -w[ok]).sum(1)
            E[p] = (val * zz[ok] * DOM / KAP).sum(); nh[p] = ok.sum()
    return E, nh
def stats(Ed):
    rl = np.abs(Ed - Et).sum(1) / Et.sum(); fr = Ed.sum(1) / Et.sum()
    pool = Ed.mean(0)
    return {'pooled_ratio': float(pool.sum() / Et.sum()), 'pooled_rel_l1': float(np.abs(pool - Et).sum() / Et.sum()),
            'per_draw_rel_l1_median': float(np.median(rl)), 'p90': float(np.quantile(rl, .9)), 'worst': float(rl.max()),
            'per_draw_flux_ratio_p10_p90': [float(np.quantile(fr, .1)), float(np.quantile(fr, .9))]}
out = {}
D = 64
for name, fn_ in (('X1', lambda y, v: L1(y, v)),
                  ('X1c', lambda y, v: L1(np.stack([np.full(len(y), PX), PY0 + (np.floor((y[:, 1] - PY0) / .2) + .5) * .2, PZ0 + (np.floor((y[:, 2] - PZ0) / .2) + .5) * .2], -1), v)),
                  ('binned_q', lambda y, v: q[patch_of(y)[0], panel_bin(v)])):
    rng = np.random.default_rng(960000)
    Ed = []; NH = []
    for d in range(D):
        e, nh = draw(rng, fn_); Ed.append(e); NH.append(nh)
    out[name] = stats(np.array(Ed))
    out[name]['rays_hitting_panel_per_patch_mean'] = float(np.mean(NH))
# ray-level contribution concentration for X1: fraction of rays carrying 50 % / 90 % of the patch estimate (pooled)
rng = np.random.default_rng(1)
us = rng.random((4, 512 * 64))
k = np.arange(512 * 64) % 512; bn = k // KAP; iz, ip = bn // NPHI, bn % NPHI
zz = (iz + us[2]) / NZ; ph = (ip + us[3]) * 2 * math.pi / NPHI - math.pi; r = np.sqrt(1 - zz * zz)
w = r[:, None] * np.cos(ph)[:, None] * fe1 + r[:, None] * np.sin(ph)[:, None] * fe2 + zz[:, None] * fn
conc = []
for p in range(9):
    iu, iv = p % 3, p // 3
    o = fc + (-0.45 + (iu + us[0]) * 0.3)[:, None] * fe1 + (-0.45 + (iv + us[1]) * 0.3)[:, None] * fe2
    ok, y = hit_panel(o, w)
    c = np.zeros(len(w)); c[ok] = L1(y[ok], -w[ok]).sum(1) * zz[ok]
    cs = np.sort(c)[::-1]; cum = np.cumsum(cs) / cs.sum()
    conc.append([float((np.searchsorted(cum, .5) + 1) / len(c)), float((np.searchsorted(cum, .9) + 1) / len(c))])
out['X1_ray_fraction_carrying_50_90pct'] = {'median_over_patches': np.median(np.array(conc), 0).tolist(), 'per_patch': conc}
print(json.dumps(out, indent=1))
