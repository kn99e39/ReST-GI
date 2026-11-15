import json, math
import numpy as np
from ck2_surrogate import *
Et = np.load('est.npz')['truth'].sum(1)
out = {}
for cs in (0.1, 0.05, 0.025, 0.0125):
    for m in (2, 4, 8):
        ny = int(round((PY1 - PY0) / cs)); nz = int(round((PZ1 - PZ0) / cs))
        gy = PY0 + (np.arange(ny) + .5) * cs; gz = PZ0 + (np.arange(nz) + .5) * cs
        Yg, Zg = np.meshgrid(gy, gz, indexing='ij')
        yp = np.stack([np.full(Yg.size, PX), Yg.ravel(), Zg.ravel()], -1)
        pts = recv_points(m)
        E = np.zeros(9)
        for i in range(9):
            for j in range(pts.shape[1]):
                x = pts[i, j]; d = x - yp; d2 = (d * d).sum(-1); v = d / np.sqrt(d2)[:, None]
                cy = v[:, 0]; cx = -(v @ fn); ok = (cy > 0) & (cx > 0)
                E[i] += (L1(yp[ok], v[ok]).sum(1) * cy[ok] * cx[ok] / d2[ok] * cs * cs).sum() / pts.shape[1]
        out[f'cell{cs}_m{m}'] = {'evals_per_light_face': int(yp.shape[0] * 9 * m * m), 'flux_ratio': float(E.sum() / Et.sum()),
                                 'rel_l1': float(np.abs(E - Et).sum() / Et.sum())}
print(json.dumps(out, indent=1))
