"""WL96 design calc (CPU, scratch): deterministic area-form quadrature of exactly-2-event S1 -> box A delivery on CK2 and two extra
lights, for carrier arms (REF exact, X1c, node-interpolated incident carrier, binned-incident exact-outgoing, M64 binned-outgoing).
No relation sampling: every arm is the expectation of an unbiased receiver-gather estimator of that arm's sender radiance."""
import json, math, os, sys, time
import numpy as np
from multiprocessing import Pool

sys.path.insert(0, 'C:/Projects/ReST-GI/experiments/ecro86')
import scene86 as S  # noqa
from er_np import ER  # noqa

R = 'C:/Projects/ReST-GI'
ST = np.load(f'{R}/experiments/ecro86/data/mats86.npz')['states']
er = ER(ST[1])
A4 = np.load(f'{R}/results/ecro_attr95a/ops95a.npz')['A4'].astype(np.float64)
CK = json.load(open(f'{R}/experiments/ecro_local93d/addendum_checker/scene_ck2.json'))
G = S.Geom(CK['geom'])
S1 = np.arange(1088, 1130); BOXA = np.arange(1130, 1184)
HP = float(os.environ.get('HP', '0.004'))      # panel grid spacing
NR = int(os.environ.get('NR', '4'))            # receiver points per patch side
LIGHTS = {'CK2': [-1.45, -0.6, 1.7], 'L2': [-0.9, -0.9, 1.0], 'L3': [-0.8, -0.3, 2.5]}
I0 = 30.0
pc = np.array([-1.5, 0.3, 1.0]); T_, B_, N_ = np.array([0, 1., 0]), np.array([0, 0, 1.]), np.array([1., 0, 0])
y0, z0 = 0.3 - 0.6, 1.0 - 0.7                   # panel lower corner (y, z)
ny, nz = int(round(1.2 / HP)), int(round(1.4 / HP))
gy = y0 + (np.arange(ny) + 0.5) * HP; gz = z0 + (np.arange(nz) + 0.5) * HP
YY, ZZ = np.meshgrid(gy, gz, indexing='ij')
PY = np.stack([np.full(YY.size, -1.5), YY.ravel(), ZZ.ravel()], 1)          # panel points [P, 3]
PATCH = (np.minimum(((PY[:, 1] - y0) / 0.2).astype(int), 5) + 6 * np.minimum(((PY[:, 2] - z0) / 0.2).astype(int), 6))
PCEN = np.stack([np.full(42, -1.5), y0 + 0.1 + 0.2 * (np.arange(42) % 6), z0 + 0.1 + 0.2 * (np.arange(42) // 6)], 1)
assert np.allclose(PCEN, G.PC[S1])


def loc(w):
    return np.stack([w @ T_, w @ B_, w @ N_], -1)


def sphere_hit(o, e, c, r):
    d = e - o; f = o - c
    a = (d * d).sum(-1); b = 2 * (f * d).sum(-1); cc = (f * f).sum(-1) - r * r
    disc = b * b - 4 * a * cc
    sq = np.sqrt(np.maximum(disc, 0))
    t1 = (-b - sq) / (2 * a); t2 = (-b + sq) / (2 * a)
    return (disc > 0) & (((t1 > 0) & (t1 < 1)) | ((t2 > 0) & (t2 < 1)))


def light_vis(p, pl):
    e = np.broadcast_to(np.asarray(pl), p.shape)
    o = p + N_ * 1e-4
    v = ~(S.seg_obb(o, e, G.obb('A', 0)) | S.seg_obb(o, e, G.obb('B', 0)) | sphere_hit(o, e, G.SC, G.SR))
    return v


def incident(p, pl):
    """exact incident direction (world) and Phi = I V / d^2 at points p."""
    dl = np.asarray(pl) - p; d2 = (dl * dl).sum(-1); w = dl / np.sqrt(d2)[:, None]
    V = light_vis(p, pl) & (w @ N_ > 1e-6)
    return w, I0 * V / d2


def bin_centroid(wl, nzb, npb):
    z = np.clip(wl[:, 2], 0, 1); phi = np.arctan2(wl[:, 1], wl[:, 0])
    iz = np.clip(np.floor(z * nzb), 0, nzb - 1); ip = np.clip(np.floor((phi + math.pi) * npb / (2 * math.pi)), 0, npb - 1)
    zc = (iz + 0.5) / nzb; pcn = (ip + 0.5) * 2 * math.pi / npb - math.pi
    r = np.sqrt(1 - zc * zc)
    return np.stack([r * np.cos(pcn), r * np.sin(pcn), zc], -1)


def node_interp(h, pl):
    """bilinear interpolation of (Phi, w) from a node grid of spacing h -> (w_unit, Phi) at every panel point."""
    ky, kz = int(round(1.2 / h)), int(round(1.4 / h))
    ny_, nz_ = np.arange(ky + 1) * h + y0, np.arange(kz + 1) * h + z0
    NY, NZ_ = np.meshgrid(ny_, nz_, indexing='ij')
    nd = np.stack([np.full(NY.size, -1.5), NY.ravel(), NZ_.ravel()], 1)
    w, ph = incident(nd, pl)
    w = w.reshape(ky + 1, kz + 1, 3); ph = ph.reshape(ky + 1, kz + 1)
    fy = (PY[:, 1] - y0) / h; fz = (PY[:, 2] - z0) / h
    iy = np.minimum(fy.astype(int), ky - 1); iz = np.minimum(fz.astype(int), kz - 1)
    ay = fy - iy; az = fz - iz
    wts = [((1 - ay) * (1 - az), iy, iz), (ay * (1 - az), iy + 1, iz), ((1 - ay) * az, iy, iz + 1), (ay * az, iy + 1, iz + 1)]
    W = sum(c[:, None] * (ph[a, b][:, None] * w[a, b]) for c, a, b in wts)   # Phi-weighted direction moment (source-linear)
    P_ = sum(c * ph[a, b] for c, a, b in wts)
    wn = W / np.maximum(np.linalg.norm(W, axis=1, keepdims=True), 1e-30)
    return wn, P_


def receivers():
    pts, nrm, pid = [], [], []
    g = (np.arange(NR) + 0.5) / NR
    for p in BOXA:
        for a in g:
            for b in g:
                pts.append(G.PO[p] + a * G.PE1[p] + b * G.PE2[p]); nrm.append(G.PN[p]); pid.append(p)
    return np.array(pts), np.array(nrm), np.array(pid)


def bin64(wloc):
    z = np.clip(wloc[..., 2], 0, 1); phi = np.arctan2(wloc[..., 1], wloc[..., 0])
    iz = np.clip(np.floor(z * 4), 0, 3).astype(int); ip = np.clip(np.floor((phi + math.pi) * 16 / (2 * math.pi)), 0, 15).astype(int)
    return iz * 16 + ip


ARMS = None


def setup_arms(pl):
    wl, ph = incident(PY, pl)
    wloc = loc(wl); E = ph * wloc[:, 2]                      # irradiance
    arms = {'REF': (wloc, ph)}
    wc, phc = incident(PCEN, pl)
    arms['X1c'] = (loc(wc)[PATCH], phc[PATCH])
    for h in (0.2, 0.1, 0.05, 0.025):
        wn, pn = node_interp(h, pl)
        arms[f'NODE{int(h * 1000)}mm'] = (loc(wn), pn)
    for nzb, npb in ((4, 16), (8, 32), (16, 64), (32, 128)):
        wcb = bin_centroid(wloc, nzb, npb)
        arms[f'BIC{nzb * npb}'] = (wcb, E / np.maximum(wcb[:, 2], 1e-9))     # f(w_c, v) * E  (eval / cos_c * E)
    # M64 binned outgoing: Bq[p, bo] = sum_m s_pool[p, m] M64[m, bo]
    b = bin64(wloc)
    s = np.zeros((42, 64, 3))
    np.add.at(s, (PATCH, b), E[:, None] * np.ones(3))
    cnt = np.bincount(PATCH, minlength=42)[:, None, None]
    s /= cnt
    Bq = np.einsum('nmc,mdc->ndc', s, A4)
    return arms, Bq


def work(args):
    xs, ns, pl = args
    arms, Bq = setup_arms(pl)
    out = {k: np.zeros((len(xs), 3)) for k in list(arms) + ['M64det']}
    for i, (x, n) in enumerate(zip(xs, ns)):
        d = x - PY; r2 = (d * d).sum(1); v = d / np.sqrt(r2)[:, None]
        cy = v @ N_; cx = -(v @ n)
        ok = (cy > 1e-6) & (cx > 1e-6)
        if not ok.any():
            continue
        g = (cy * cx / r2 * HP * HP)[ok]
        vl = loc(v[ok])
        for k, (wi, ph) in arms.items():
            L = er.eval(vl, wi[ok]) * ph[ok][:, None]
            out[k][i] = (L * g[:, None]).sum(0)
        L = Bq[PATCH[ok], bin64(vl)]
        out['M64det'][i] = (L * g[:, None]).sum(0)
    return out


def main():
    xs, ns, pid = receivers()
    res = {}
    for ln, pl in LIGHTS.items():
        t0 = time.time()
        ch = np.array_split(np.arange(len(xs)), 32)
        with Pool(16) as po:
            parts = po.map(work, [(xs[c], ns[c], pl) for c in ch])
        arms = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
        # per box-A patch exitance (rho = 0.5, channels summed)
        ex = {k: 0.5 * np.bincount(pid - 1130, weights=v.sum(1), minlength=54) / NR ** 2 for k, v in arms.items()}
        res[ln] = {k: v.tolist() for k, v in ex.items()}
        print(ln, 'done', round(time.time() - t0, 1), 's', flush=True)
    json.dump({'HP': HP, 'NR': NR, 'res': res}, open(f'deliv96_hp{int(HP * 1e4)}_nr{NR}.json', 'w'))


if __name__ == '__main__':
    main()
