"""WL96 D4 scratch 3: how many moment-matched anisotropic Gaussian lobes (in the tangent plane at the lobe peak) represent the S1 ER response
f(wi, .) cos to a single incident direction; L1 error over the hemisphere (luminance), by incidence. Lobes grouped by GGX component
(each ER GGX lobe term fitted separately, i.e. 'per-component' capacity), or the whole closure as 1 lobe. No optimiser (numpy only)."""
import json, os, sys
import numpy as np
H = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, H)
import er_np as ER
st = np.load('c:/Projects/ReST-GI/experiments/ecro86/data/mats86.npz')['states'][1]
P = ER.unpack(st)
n = 900; z = (np.arange(n) + .5) / n; ph = (np.arange(2 * n) + .5) / (2 * n) * 2 * np.pi
Z, PH = np.meshgrid(z, ph); r = np.sqrt(1 - Z ** 2)
WO = np.stack([r * np.cos(PH), r * np.sin(PH), Z], -1).reshape(-1, 3); dO = 2 * np.pi / (n * 2 * n)


def comp_evals(wi):
    """per-component eval (lum) [C, M]: Lambert, outer k0..k2, inner k3..k6 (with s(mu) factors)."""
    W = np.broadcast_to(wi, WO.shape)
    co = WO[:, 2:3]
    out = []
    lamL = P['lam'].sum(0)
    sI = ER.s_interp(P['s'], W[:, 2]) * ER.s_interp(P['s'], WO[:, 2])          # [M,3]
    out.append((sI * P['lam'][:, 0][None] * co / np.pi).sum(1))
    fo = ER.lobe_f(P['au'][[0, 1, 2]], P['av'][[0, 1, 2]], P['f0'][[0, 1, 2]], W, WO)
    for j in range(3):
        out.append(fo[:, j] * co[:, 0] * P['lam'][:, 1 + j].sum())
    fi = ER.lobe_f(P['au'][[3, 4, 5, 6]], P['av'][[3, 4, 5, 6]], P['f0'][[3, 4, 5, 6]], ER.refr(W, P['eta']), ER.refr(WO, P['eta'])) / P['eta'] ** 2
    for j in range(4):
        out.append((sI * (fi[:, j:j + 1] * co) * P['lam'][:, 4 + j][None]).sum(1))
    return np.array(out)


def gauss_fit(f):
    """moment-matched 2D Gaussian in the tangent plane at the peak direction; returns the fitted function on WO."""
    if f.sum() <= 0:
        return np.zeros_like(f)
    pk = WO[np.argmax(f)]
    t = np.cross(pk, [0, 0, 1.0]); t = t / np.linalg.norm(t) if np.linalg.norm(t) > 1e-6 else np.array([1.0, 0, 0]); b = np.cross(pk, t)
    u = np.stack([WO @ t, WO @ b], 1); front = WO @ pk > 0
    wgt = f * front
    m = (u * wgt[:, None]).sum(0) / wgt.sum()
    du = u - m
    C = (du[:, :, None] * du[:, None, :] * wgt[:, None, None]).sum(0) / wgt.sum()
    Ci = np.linalg.inv(C + 1e-12 * np.eye(2))
    g = np.exp(-0.5 * np.einsum('mi,ij,mj->m', du, Ci, du)) * front
    return g * (f.sum() / g.sum())


res = {}
for th in (10, 45, 75, 80, 85, 88):
    t = np.radians(th); wi = np.array([np.sin(t), 0, np.cos(t)])
    comps = comp_evals(wi)
    tot = comps.sum(0)
    one = gauss_fit(tot - comps[0]) + comps[0]
    per = comps[0] + sum(gauss_fit(c) for c in comps[1:])
    big = np.argsort([-(c.sum()) for c in comps[1:]])[:2] + 1
    two = comps[0] + sum(gauss_fit(comps[j]) for j in big) + sum(comps[j] for j in range(1, 8) if j not in big) * 0
    e = lambda a: float(np.abs(a - tot).sum() / tot.sum())
    pk = lambda a: float(a.max() / tot.max())
    res[th] = dict(one_lobe_l1=e(one), one_lobe_peak=pk(one), per_component_7_l1=e(per), per_component_peak=pk(per),
                   two_largest_components_only_l1=e(two), energy_share_by_component=[float(c.sum() / tot.sum()) for c in comps])
json.dump(res, open(os.path.join(H, 'lobe_fit.json'), 'w'), indent=1)
for k, v in res.items():
    print(k, {a: (round(b, 3) if not isinstance(b, list) else [round(x, 3) for x in b]) for a, b in v.items()})
