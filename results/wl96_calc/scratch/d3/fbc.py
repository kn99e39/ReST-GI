"""Candidate B (fine binned sender carrier) flight: source-independent sender-anchored relation with r rays per (cell, bin);
table = physical L1 mean per (10 cm cell, 16 x 64 bin) under the light; deposit flux into box-A patches."""
import json, math, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import toy96 as T
import toy96b as TB
import photon as PH
SUB, NZ, NPHI = 2, 16, 64
NB = NZ * NPHI; nu, nv = 6 * SUB, 7 * SUB


def full_table(light, ns=4, nd=3):
    a = (np.arange(ns) + 0.5) / ns; bb = (np.arange(nd) + 0.5) / nd
    A_, B_ = [m.ravel() for m in np.meshgrid(a, a, indexing='ij')]
    Zb, Pb = [m.ravel() for m in np.meshgrid(bb, bb, indexing='ij')]
    iz = np.arange(NB) // NPHI; ip = np.arange(NB) % NPHI
    z = (iz[:, None] + Zb[None]) / NZ; ph = (ip[:, None] + Pb[None]) / NPHI * 2 * math.pi - math.pi
    s_ = np.sqrt(1 - z * z)
    wl = np.stack([s_ * np.cos(ph), s_ * np.sin(ph), z], -1)
    ww = wl[..., 0:1] * T.PT_ + wl[..., 1:2] * T.PB_ + wl[..., 2:3] * T.PN_          # [NB, nd2, 3]
    Tab = np.zeros((nu, nv, NB, 3))
    for i in range(nu):
        for j in range(nv):
            Y = T.ORG + ((i + A_) * T.W_P / nu)[:, None] * T.PT_ + ((j + B_) * T.H_P / nv)[:, None] * T.PB_
            Lr = TB.L1s(np.broadcast_to(Y[None, None], (NB, ww.shape[1], len(Y), 3)), np.broadcast_to(ww[:, :, None], (NB, ww.shape[1], len(Y), 3)), light)
            Tab[i, j] = Lr.mean((1, 2))
    return Tab


def flight(Tab, r, seed):
    rng = np.random.default_rng(seed)
    ci, cj, b = [m.ravel() for m in np.meshgrid(np.arange(nu), np.arange(nv), np.arange(NB), indexing='ij')]
    n = len(b)
    E = np.zeros((54, 3)); Acell = (T.W_P / nu) * (T.H_P / nv); Om = 2 * math.pi / NB
    for k in range(r):
        u = rng.random((4, n))
        y = T.ORG + ((ci + u[0]) * T.W_P / nu)[:, None] * T.PT_ + ((cj + u[1]) * T.H_P / nv)[:, None] * T.PB_
        iz, ip = b // NPHI, b % NPHI
        z = (iz + u[2]) / NZ; ph = (ip + u[3]) / NPHI * 2 * math.pi - math.pi
        s_ = np.sqrt(1 - z * z)
        v = (s_ * np.cos(ph))[:, None] * T.PT_ + (s_ * np.sin(ph))[:, None] * T.PB_ + z[:, None] * T.PN_
        pid = PH.hit_box(y + v * 1e-6, v)
        g = pid >= 0
        w = Tab[ci[g], cj[g], b[g]] * (Acell * Om * z[g] / r)[:, None]
        for ch in range(3):
            E[:, ch] += np.bincount(pid[g], weights=w[:, ch], minlength=54)
    return E / PH.AREA[:, None], int(n * r)


if __name__ == '__main__':
    li = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    LIGHTS = [T.LIGHT0, (np.array([-0.9, -0.5, 2.4]), np.array([30.0, 30.0, 30.0]))]
    light = LIGHTS[li]
    Ex = np.load(os.path.join(T.OUT, f'E_exact_L{li}.npy'))
    t0 = time.time(); Tab = full_table(light); res = {'table_s': time.time() - t0, 'table_floats': int(Tab.size)}
    for r in (1, 4, 16):
        ms = []
        for d in range(8):
            E, nr = flight(Tab, r, 9600 + 10 * r + d)
            ms.append(T.metrics(E, Ex))
        rl = [m['rel_l1'] for m in ms]
        res[f'r{r}'] = {'rays_per_facet': nr, 'rel_l1_median': float(np.median(rl)), 'rel_l1_worst': float(max(rl)),
                        'ratio_range': [float(min(m['ratio'] for m in ms)), float(max(m['ratio'] for m in ms))]}
        print(r, json.dumps(res[f'r{r}']), flush=True)
    json.dump(res, open(os.path.join(T.OUT, f'res_fbc_L{li}.json'), 'w'), indent=1)
