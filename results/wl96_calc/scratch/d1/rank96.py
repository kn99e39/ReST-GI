"""Capacity of a source-independent, material-level linear S1 operator with fine incident cells (16 x 64) and continuous-ish outgoing
(64 x 256 cells): truncated-SVD rank R -> CK2 / L2 / L3 delivery error (scratch, WL96)."""
import json, math, os
os.environ.setdefault('HP', '0.004'); os.environ.setdefault('NR', '4')
import numpy as np
import deliv96 as D
from multiprocessing import Pool

def cells(nz, npp):
    i = np.arange(nz * npp); z = (i // npp + 0.5) / nz; ph = (i % npp + 0.5) * 2 * math.pi / npp - math.pi
    r = np.sqrt(1 - z * z); return np.stack([r * np.cos(ph), r * np.sin(ph), z], -1)
def cidx(w, nz, npp):
    z = np.clip(w[..., 2], 0, 1); ph = np.arctan2(w[..., 1], w[..., 0])
    return (np.clip(np.floor(z * nz), 0, nz - 1) * npp + np.clip(np.floor((ph + math.pi) * npp / (2 * math.pi)), 0, npp - 1)).astype(int)
NZI, NPI, NZO, NPO = 16, 64, 64, 256
wi_c, wo_c = cells(NZI, NPI), cells(NZO, NPO)
Kf = None
RANKS = [8, 16, 32, 64, 128, 256, 0]

def build():
    K = np.zeros((len(wi_c), len(wo_c), 3))
    for a in range(0, len(wi_c), 32):
        K[a:a + 32] = D.er.eval(wo_c[None], wi_c[a:a + 32, None]) / wi_c[a:a + 32, None, 2:3]   # f(w_in, v)
    return K

def work(args):
    idx, pl, Ks = args
    xs, ns, pid = D.receivers()
    wl, ph = D.incident(D.PY, pl); wloc = D.loc(wl); E = ph * wloc[:, 2]
    mi = cidx(wloc, NZI, NPI)
    out = {r: np.zeros((len(idx), 3)) for r in Ks}
    for j, i in enumerate(idx):
        x, n = xs[i], ns[i]
        d = x - D.PY; r2 = (d * d).sum(1); v = d / np.sqrt(r2)[:, None]
        cy = v @ D.N_; cx = -(v @ n); ok = (cy > 1e-6) & (cx > 1e-6)
        if not ok.any(): continue
        g = (cy * cx / r2 * D.HP ** 2)[ok]; vo = cidx(D.loc(v[ok]), NZO, NPO)
        for r, Kr in Ks.items():
            out[r][j] = ((Kr[mi[ok], vo] * E[ok][:, None]) * g[:, None]).sum(0)
    return out

if __name__ == '__main__':
    K = build()
    zi, zo = wi_c[:, 2], wo_c[:, 2]
    Ks = {}; info = {}
    for R in RANKS:
        Kr = np.zeros_like(K)
        for c in range(3):
            X = K[..., c] * zi[:, None] * zo[None, :]
            if R == 0:
                Kr[..., c] = K[..., c]; continue
            U, s, Vt = np.linalg.svd(X, full_matrices=False)
            Kr[..., c] = ((U[:, :R] * s[:R]) @ Vt[:R]) / zi[:, None] / zo[None, :]
            if c == 0 and R == 8:
                e = s ** 2 / (s ** 2).sum(); ce = np.cumsum(e)
                info['rank_energy_0.9_0.99_0.999'] = [int(np.searchsorted(ce, q) + 1) for q in (0.9, 0.99, 0.999)]
                info['flux_l1_rank'] = {}
            if c == 0:
                info.setdefault('flux_l1_rank', {})[R] = float(np.abs(Kr[..., 0] - K[..., 0]).__mul__(zi[:, None] * zo[None, :]).sum() / (K[..., 0] * zi[:, None] * zo[None, :]).sum())
                info.setdefault('neg_mass_frac', {})[R] = float(np.minimum(Kr[..., 0], 0).__mul__(zi[:, None] * zo[None, :]).sum() / -(K[..., 0] * zi[:, None] * zo[None, :]).sum())
        Ks[R] = Kr.astype(np.float32)
    print(json.dumps(info), flush=True)
    xs, ns, pid = D.receivers()
    res = {}
    for ln, pl in D.LIGHTS.items():
        ch = np.array_split(np.arange(len(xs)), 32)
        with Pool(8) as po:
            parts = po.map(work, [(c, pl, Ks) for c in ch])
        ex = {}
        for R in RANKS:
            a = np.concatenate([p[R] for p in parts])
            ex[R] = 0.5 * np.bincount(pid - 1130, weights=a.sum(1), minlength=54) / D.NR ** 2
        ref = np.asarray(json.load(open('deliv96_hp40_nr4.json'))['res'][ln]['REF'])
        res[ln] = {str(R): {'ratio': float(v.sum() / ref.sum()), 'rel_l1': float(np.abs(v - ref).sum() / ref.sum())} for R, v in ex.items()}
        print(ln, json.dumps(res[ln]), flush=True)
    json.dump({'info': info, 'res': res}, open('rank96.json', 'w'), indent=1)
