"""scratch 3b: L1-optimised scale of the moment-matched Gaussian covariance (1-D search over a width multiplier, energy-preserving or
L1-optimal amplitude), per ER component; reports the best achievable L1 for this restricted Gaussian family."""
import json, numpy as np, lobe_fit as LF
WO = LF.WO
def gfit_scaled(f):
    if f.sum() <= 0: return np.zeros_like(f)
    pk = WO[np.argmax(f)]
    t = np.cross(pk, [0, 0, 1.0]); t = t/np.linalg.norm(t) if np.linalg.norm(t) > 1e-6 else np.array([1.0, 0, 0]); b = np.cross(pk, t)
    u = np.stack([WO @ t, WO @ b], 1); front = WO @ pk > 0
    w = f*front; m = (u*w[:, None]).sum(0)/w.sum(); du = u-m
    C = (du[:, :, None]*du[:, None, :]*w[:, None, None]).sum(0)/w.sum(); Ci = np.linalg.inv(C+1e-12*np.eye(2))
    q = np.einsum('mi,ij,mj->m', du, Ci, du)
    best = None
    for s in np.geomspace(0.02, 1.5, 40):
        g = np.exp(-0.5*q/s**2)*front
        for amp in np.geomspace(0.2, 2.0, 25):
            gg = g*(f.max()*amp/g.max())
            e = np.abs(gg-f).sum()
            if best is None or e < best[0]: best = (e, gg)
    return best[1]
res = {}
for th in (10, 45, 75, 85, 88):
    tt = np.radians(th); wi = np.array([np.sin(tt), 0, np.cos(tt)])
    comps = LF.comp_evals(wi); tot = comps.sum(0)
    per = comps[0] + sum(gfit_scaled(c) for c in comps[1:])
    one = comps[0] + gfit_scaled(tot-comps[0])
    res[th] = dict(one_l1=float(np.abs(one-tot).sum()/tot.sum()), per7_l1=float(np.abs(per-tot).sum()/tot.sum()),
                   per7_energy=float(per.sum()/tot.sum()), per7_peak=float(per.max()/tot.max()))
    print(th, {k: round(v, 3) for k, v in res[th].items()})
json.dump(res, open('lobe_fit2.json', 'w'), indent=1)
