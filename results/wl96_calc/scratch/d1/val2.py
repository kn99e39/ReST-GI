import math, numpy as np
from er_np import ER
st = np.load('C:/Projects/ReST-GI/experiments/ecro86/data/mats86.npz')['states']
er = ER(st[1])
A4 = np.load('C:/Projects/ReST-GI/results/ecro_attr95a/ops95a.npz')['A4'].astype(np.float64)
NZ, NP = 4, 16; SUB = 8
def dirs(b, uz, up):
    z = (b // NP + uz) / NZ; phi = (b % NP + up) * (2*math.pi/NP) - math.pi
    r = np.sqrt(np.maximum(1 - z*z, 0)); return np.stack([r*np.cos(phi), r*np.sin(phi), z], -1)
m = 13
ci = (np.arange(SUB) + 0.5) / SUB; cz, cp = [a.ravel() for a in np.meshgrid(ci, ci, indexing='ij')]
wi = dirs(m, cz, cp)
for NO in (48, 128):
    g = (np.arange(NO) + 0.5) / NO; uz, up = [a.ravel() for a in np.meshgrid(g, g, indexing='ij')]
    M = np.zeros((64, 3))
    for d in range(64):
        vo = dirs(d, uz, up)
        M[d] = (er.eval(vo[None], wi[:, None]) / wi[:, None, 2:3]).mean((0, 1))
    zb = ((np.arange(64)//16)+0.5)/4
    alb_me = (M.sum(1)/3 * zb).sum()*2*math.pi/64; alb_A = (A4[m].sum(1)/3*zb).sum()*2*math.pi/64
    print(NO, 'ring sums me/A4', [round(M[16*i:16*i+16].sum()/A4[m,16*i:16*i+16].sum(),3) for i in range(4)], 'albedo', round(alb_me,4), round(alb_A,4))
