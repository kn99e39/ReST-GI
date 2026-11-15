import sys, time, json, math
import numpy as np
from ck2_surrogate import *
h = 0.005
pts = recv_points(4)
ny = int(round((PY1 - PY0) / h)); nz = int(round((PZ1 - PZ0) / h))
gy = PY0 + (np.arange(ny) + .5) * h; gz = PZ0 + (np.arange(nz) + .5) * h
Yg, Zg = np.meshgrid(gy, gz, indexing='ij')
yp = np.stack([np.full(Yg.size, PX), Yg.ravel(), Zg.ravel()], -1)
pid, iu, iv = patch_of(yp)
def area_est(value_fn):
    E = np.zeros(pts.shape[:-1] + (3,))
    for i in range(pts.shape[0]):
        for j in range(pts.shape[1]):
            x = pts[i, j]; d = x - yp; d2 = (d * d).sum(-1); v = d / np.sqrt(d2)[:, None]
            cy = v[:, 0]; cx = -(v @ fn); ok = (cy > 0) & (cx > 0)
            val = value_fn(yp[ok], v[ok], np.nonzero(ok)[0])
            E[i, j] = (val * (cy[ok] * cx[ok] / d2[ok] * h * h)[:, None]).sum(0)
    return E.mean(1)          # [9, 3]
res = {}
Et = area_est(lambda y, v, ix: L1(y, v)); res['truth'] = Et
def subcell(s):
    cs = 0.2 / s
    def f(y, v, ix):
        c = y.copy()
        c[:, 1] = PY0 + (np.floor((y[:, 1] - PY0) / cs) + .5) * cs; c[:, 2] = PZ0 + (np.floor((y[:, 2] - PZ0) / cs) + .5) * cs
        return L1(c, v)
    return f
for s in (1, 2, 4, 8):
    res[f'cell_centre_s{s}'] = area_est(subcell(s))
# physical bin table q (patch-average, bin-average radiance), 6x6 positions x 16x16 directions per bin
t0 = time.time()
q = np.zeros((42, 64, 3))
gpos = (np.arange(6) + .5) / 6
for n in range(42):
    a, b = n % NU, n // NU
    P = np.array([[PX, PY0 + (a + u) * .2, PZ0 + (b + w) * .2] for u in gpos for w in gpos])
    for bn in range(64):
        iz, ip = bn // NPHI, bn % NPHI
        A = (np.arange(16) + .5) / 16; Aa, Bb = np.meshgrid(A, A, indexing='ij')
        z = (iz + Aa.ravel()) / NZ; ph = (ip + Bb.ravel()) * 2 * math.pi / NPHI - math.pi; r = np.sqrt(1 - z * z)
        loc = np.stack([r * np.cos(ph), r * np.sin(ph), z], -1)          # panel frame (t=Y, b=Z, n=X)
        w = np.stack([loc[:, 2], loc[:, 0], loc[:, 1]], -1)
        Y = np.repeat(P, len(w), 0); Wd = np.tile(w, (len(P), 1))
        q[n, bn] = L1(Y, Wd).mean(0)
res['t_q_s'] = time.time() - t0
def qread(y, v, ix):
    return q[pid[ix], panel_bin(v)]
res['binned_q'] = area_est(qread)
# outgoing sub-bin refinement of q (a x a sub-bins per bin) is a separate factor; skip.
def summ(E):
    tot = Et.sum(1); e = E.sum(1)
    return {'flux_ratio': float(e.sum() / tot.sum()), 'rel_l1': float(np.abs(e - tot).sum() / tot.sum()), 'corr': float(np.corrcoef(e, tot)[0, 1])}
out = {k: summ(v) for k, v in res.items() if isinstance(v, np.ndarray)}
out['t_q_s'] = res['t_q_s']
np.save('q.npy', q); np.savez('est.npz', **{k: v for k, v in res.items() if isinstance(v, np.ndarray)})
print(json.dumps(out, indent=1))
