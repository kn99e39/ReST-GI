"""Effective rank of the exact S1 local operator projected on nz x nphi equal-solid-angle bins (incident-bin centre x outgoing-bin mean)."""
import json, math, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import toy96 as T
res = {}
for nz, nphi, nd in ((4, 16, 6), (8, 32, 4), (16, 64, 3)):
    NB = nz * nphi
    b = (np.arange(nd) + 0.5) / nd
    zc = (np.arange(nz) + 0.5) / nz; pc = (np.arange(nphi) + 0.5) / nphi * 2 * math.pi - math.pi
    Zc, Pc = [m.ravel() for m in np.meshgrid(zc, pc, indexing='ij')]
    win = np.stack([np.sqrt(1 - Zc ** 2) * np.cos(Pc), np.sqrt(1 - Zc ** 2) * np.sin(Pc), Zc], -1)          # [NB,3] incident centres
    zs = ((np.arange(nz)[:, None] + b[None]) / nz); ps = ((np.arange(nphi)[:, None] + b[None]) / nphi) * 2 * math.pi - math.pi
    Zg = np.broadcast_to(zs[:, None, :, None], (nz, nphi, nd, nd)); Pg = np.broadcast_to(ps[None, :, None, :], (nz, nphi, nd, nd))
    wo = np.stack([np.sqrt(1 - Zg ** 2) * np.cos(Pg), np.sqrt(1 - Zg ** 2) * np.sin(Pg), Zg], -1).reshape(NB, nd * nd, 3)
    M = np.zeros((NB, NB, 3))
    for i in range(NB):
        # outgoing radiance per unit incident irradiance from direction win[i]: eval(wo, wi) / cos(wi) (eval contains cos of 2nd arg)
        f = T.er_eval(wo, np.broadcast_to(win[i], wo.shape)) / win[i, 2]
        M[i] = f.mean(1)
    # flux-weighted: incident irradiance unit per bin, outgoing flux weight W_T = z dOmega
    W = Zc * 2 * math.pi / NB
    out = {}
    for c in range(3):
        A = M[..., c] * W[None, :]
        s = np.linalg.svd(A, compute_uv=False)
        e = np.cumsum(s ** 2) / (s ** 2).sum()
        out[f'c{c}'] = {f'rank_{int(p*100)}': int(np.searchsorted(e, p) + 1) for p in (0.9, 0.99, 0.999)}
        # best rank-r Frobenius rel error in operator (flux-weighted)
        out[f'c{c}']['rel_fro_err_r16'] = float(np.sqrt(max(0, 1 - e[15]))); out[f'c{c}']['rel_fro_err_r32'] = float(np.sqrt(max(0, 1 - e[31])))
    res[f'{nz}x{nphi}'] = out
    print(nz, nphi, json.dumps(out), flush=True)
json.dump(res, open(os.path.join(T.OUT, 'res_rankM.json'), 'w'), indent=1)
