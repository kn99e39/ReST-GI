"""Sender-directed deterministic lobe quadrature (per light): panel cells x fine outgoing grid, keep the directions carrying 99 % of each
cell's outgoing flux ('photons'), trace to box A (OBB slab test), deposit per patch. Bias / ray-count check (scratch, WL96)."""
import json, math, os, sys
os.environ.setdefault('HP', '0.004'); os.environ.setdefault('NR', '4')
import numpy as np
import deliv96 as D
from multiprocessing import Pool
NZO, NPO = int(os.environ.get('NZO', 96)), int(os.environ.get('NPO', 384))
i = np.arange(NZO * NPO); zc = (i // NPO + 0.5) / NZO; pc_ = (i % NPO + 0.5) * 2 * math.pi / NPO - math.pi
r = np.sqrt(1 - zc * zc); VL = np.stack([r * np.cos(pc_), r * np.sin(pc_), zc], -1); DW = 2 * math.pi / len(VL)
VW = VL[:, 0:1] * D.T_ + VL[:, 1:2] * D.B_ + VL[:, 2:3] * D.N_          # world directions
c, ax, h = D.G.obb('A', 0)

def hit_boxA(o, d):
    lo = (o - c) @ ax.T; dd = d @ ax.T
    dd = np.where(np.abs(dd) < 1e-12, 1e-12, dd)
    t0 = (-h - lo) / dd; t1 = (h - lo) / dd
    tmin = np.minimum(t0, t1).max(1); tmax = np.maximum(t0, t1).min(1)
    ok = (tmax >= tmin) & (tmin > 0)
    p = lo + tmin[:, None] * dd                                  # local hit point
    fa = np.argmin(np.abs(np.abs(p) - h), 1)                     # face axis
    return ok, p, fa

def patch_of(p, fa):
    """box-A patch id from local hit point (scene86 box_faces ordering)."""
    pairs = {(0, 1): (1, 2), (0, 0): (2, 1), (1, 1): (2, 0), (1, 0): (0, 2), (2, 1): (0, 1), (2, 0): (1, 0)}
    out = np.zeros(len(p), int)
    for k in range(len(p)):
        a = fa[k]; pos = int(p[k, a] > 0); i1, i2 = pairs[(a, pos)]
        u = (p[k, i1] + h[i1]) / (2 * h[i1]); v = (p[k, i2] + h[i2]) / (2 * h[i2])
        iu = min(int(u * 3), 2); iv = min(int(v * 3), 2)
        out[k] = 1130 + 9 * (2 * a + pos) + iu + 3 * iv
    return out

def work(args):
    pts, pl, carrier, AREA = args
    wl, ph = D.incident(pts, pl)
    if carrier == 'node50':
        # node-interpolated carrier (5 cm) evaluated at pts: reuse D.node_interp machinery on a point list
        save = D.PY; D.PY = pts; wn, pn = D.node_interp(0.05, pl); D.PY = save; wl, ph = wn, pn
    wloc = D.loc(wl)
    dep = np.zeros((54, 3)); nkept = 0; nbox = 0
    for j in range(len(pts)):
        if ph[j] <= 0: continue
        L = D.er.eval(VL, np.broadcast_to(wloc[j], VL.shape)) * ph[j]            # radiance toward each direction
        fl = L * (VL[:, 2:3] * DW * AREA)                                        # flux per direction
        fs = fl.sum(1); o = np.argsort(-fs); cs = np.cumsum(fs[o])
        k = o[:np.searchsorted(cs, 0.99 * cs[-1]) + 1]; nkept += len(k)
        ok, p, fa = hit_boxA(np.broadcast_to(pts[j] + D.N_ * 1e-4, (len(k), 3)), VW[k])
        if ok.any():
            pid = patch_of(p[ok], fa[ok]); nbox += int(ok.sum())
            np.add.at(dep, pid - 1130, fl[k][ok])
    return dep, nkept, nbox

if __name__ == '__main__':
    out = {}
    for hc, sub in ((0.05, 1), (0.025, 1)):
        n1, n2 = int(round(1.2 / hc)) * sub, int(round(1.4 / hc)) * sub
        gy = D.y0 + (np.arange(n1) + 0.5) * hc / sub; gz = D.z0 + (np.arange(n2) + 0.5) * hc / sub
        YY, ZZ = np.meshgrid(gy, gz, indexing='ij')
        pts = np.stack([np.full(YY.size, -1.5), YY.ravel(), ZZ.ravel()], 1)
        AREA = (hc / sub) ** 2

        for ln, pl in D.LIGHTS.items():
            for carrier in ('exact', 'node50'):
                ch = np.array_split(np.arange(len(pts)), 32)
                with Pool(16) as po:
                    parts = po.map(work, [(pts[cc], pl, carrier, AREA) for cc in ch])
                dep = sum(p[0] for p in parts); nk = sum(p[1] for p in parts); nb = sum(p[2] for p in parts)
                area = np.array([np.linalg.norm(D.G.PE1[q]) * np.linalg.norm(D.G.PE2[q]) for q in D.BOXA])
                ex = 0.5 * dep.sum(1) / area
                ref = np.asarray(json.load(open('deliv96_hp40_nr4.json'))['res'][ln]['REF'])
                key = f'{ln}|hc{int(hc * 1000)}|{carrier}'
                out[key] = {'ratio': float(ex.sum() / ref.sum()), 'rel_l1': float(np.abs(ex - ref).sum() / ref.sum()), 'cells': len(pts),
                            'photons_kept_99pct': int(nk), 'photons_on_boxA': int(nb)}
                print(key, out[key], flush=True)
    json.dump(out, open('photon96.json', 'w'), indent=1)
