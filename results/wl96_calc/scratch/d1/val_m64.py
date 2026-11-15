import math, numpy as np, sys
from er_np import ER
st = np.load('C:/Projects/ReST-GI/experiments/ecro86/data/mats86.npz')['states']
er = ER(st[1])
A4 = np.load('C:/Projects/ReST-GI/results/ecro_attr95a/ops95a.npz')['A4'].astype(np.float64)
NZ, NP = 4, 16; SUB = 8
def dirs(bins, uz, up):
    z = (bins[:, None] // NP + uz) / NZ; phi = (bins[:, None] % NP + up) * (2*math.pi/NP) - math.pi
    r = np.sqrt(np.maximum(1 - z*z, 0)); return np.stack([r*np.cos(phi), r*np.sin(phi), z], -1)
rng = np.random.default_rng(0)
# outgoing: stratified 48x48 per bin
NO = 48
g = (np.arange(NO) + 0.5) / NO
uo_z, uo_p = np.meshgrid(g, g, indexing='ij'); uo_z = uo_z.ravel(); uo_p = uo_p.ravel()
ci = (np.arange(SUB) + 0.5) / SUB
cz, cp = np.meshgrid(ci, ci, indexing='ij'); cz = cz.ravel(); cp = cp.ravel()
out = {}
for m in [12, 13, 14, 15, 3, 20, 40, 60]:
    wi = dirs(np.array([m]), cz, cp)[0]            # 64 incident cell centres
    M = np.zeros((64, 3))
    for d in range(64):
        vo = dirs(np.array([d]), uo_z, uo_p)[0]     # outgoing samples
        f = er.eval(vo[None, :, :], wi[:, None, :]) / wi[:, None, 2:3]
        M[d] = f.mean((0, 1))
    A = A4[m]
    l1 = np.abs(M - A).sum() / A.sum(); pk = A.sum(1).argmax()
    out[m] = (round(float(l1), 4), int(pk), int(M.sum(1).argmax()), round(float(M[pk].sum() / A[pk].sum()), 4), round(float(M.sum() / A.sum()), 4))
    print(m, out[m], flush=True)
