import sys, time, json, math
import numpy as np
from ck2_surrogate import *
def area_truth(pts, h):
    ny = int(round((PY1 - PY0) / h)); nz = int(round((PZ1 - PZ0) / h))
    gy = PY0 + (np.arange(ny) + .5) * h; gz = PZ0 + (np.arange(nz) + .5) * h
    Y, Z = np.meshgrid(gy, gz, indexing='ij')
    yp = np.stack([np.full(Y.size, PX), Y.ravel(), Z.ravel()], -1)
    E = np.zeros(pts.shape[:-1] + (3,))
    for i in range(pts.shape[0]):
        for j in range(pts.shape[1]):
            x = pts[i, j]; d = x - yp; d2 = (d * d).sum(-1); v = d / np.sqrt(d2)[:, None]
            cy = v[:, 0]; cx = -(v @ fn)
            ok = (cy > 0) & (cx > 0)
            val = L1(yp[ok], v[ok])
            E[i, j] = (val * (cy[ok] * cx[ok] / d2[ok] * h * h)[:, None]).sum(0)
    return E
if __name__ == '__main__':
    m = int(sys.argv[1]); h = float(sys.argv[2])
    t0 = time.time()
    pts = recv_points(m)
    E = area_truth(pts, h)
    np.save(f'Ea_m{m}_h{h}.npy', E)
    print(json.dumps({'m': m, 'h': h, 'patch_irr_sum': E.mean(1).sum(1).round(4).tolist(), 'total': float(E.mean(1).sum()), 's': time.time() - t0}))
