"""Critic re-check (scratch): does d1's continuous-kernel rank result depend on the SVD row weighting?
Same kernel / delivery code as d1 rank96.py; only the weighting of the truncated SVD changes:
  'flux'   = d1's X = K * z_in * z_out (reproduce)
  'rownorm'= each incident row scaled to unit outgoing flux (no grazing down-weighting)
  'ck2w'   = rows weighted by CK2's actual incident-cell irradiance histogram (light-conditioned operator)
"""
import json, math, os, sys
W96 = r'C:\Users\dna10\AppData\Local\Temp\claude\462fa2b2dbb71cdd\46d65e15-e083-4057-875f-9282fb68aa5e\scratchpad\wl96'
sys.path.insert(0, W96)
os.environ.setdefault('HP', '0.01'); os.environ.setdefault('NR', '4')
os.chdir(W96)
import numpy as np
import deliv96 as D
from multiprocessing import Pool
OUT = os.path.join(W96, 'critic_phys')
NZI, NPI, NZO, NPO = 16, 64, 64, 256
RANKS = [16, 32, 64, 128]


def cells(nz, npp):
    i = np.arange(nz * npp); z = (i // npp + 0.5) / nz; ph = (i % npp + 0.5) * 2 * math.pi / npp - math.pi
    r = np.sqrt(1 - z * z); return np.stack([r * np.cos(ph), r * np.sin(ph), z], -1)


def cidx(w, nz, npp):
    z = np.clip(w[..., 2], 0, 1); ph = np.arctan2(w[..., 1], w[..., 0])
    return (np.clip(np.floor(z * nz), 0, nz - 1) * npp + np.clip(np.floor((ph + math.pi) * npp / (2 * math.pi)), 0, npp - 1)).astype(int)


wi_c, wo_c = cells(NZI, NPI), cells(NZO, NPO)
_K = {}


def init(path):
    _K['K'] = np.load(path, mmap_mode='r')


def work(args):
    idx, pl = args
    Kr = _K['K']
    xs, ns, pid = D.receivers()
    wl, ph = D.incident(D.PY, pl); wloc = D.loc(wl); E = ph * wloc[:, 2]
    mi = cidx(wloc, NZI, NPI)
    out = np.zeros((len(idx), 3))
    for j, i in enumerate(idx):
        x, n = xs[i], ns[i]
        d = x - D.PY; r2 = (d * d).sum(1); v = d / np.sqrt(r2)[:, None]
        cy = v @ D.N_; cx = -(v @ n); ok = (cy > 1e-6) & (cx > 1e-6)
        if not ok.any(): continue
        g = (cy * cx / r2 * D.HP ** 2)[ok]; vo = cidx(D.loc(v[ok]), NZO, NPO)
        out[j] = ((np.asarray(Kr[mi[ok], vo]) * E[ok][:, None]) * g[:, None]).sum(0)
    return out


def deliver(path, pl):
    xs, ns, pid = D.receivers()
    ch = np.array_split(np.arange(len(xs)), 16)
    with Pool(8, initializer=init, initargs=(path,)) as po:
        parts = po.map(work, [(c, pl) for c in ch])
    a = np.concatenate(parts)
    return 0.5 * np.bincount(pid - 1130, weights=a.sum(1), minlength=54) / D.NR ** 2


if __name__ == '__main__':
    K = np.zeros((len(wi_c), len(wo_c), 3))
    for a in range(0, len(wi_c), 32):
        K[a:a + 32] = D.er.eval(wo_c[None], wi_c[a:a + 32, None]) / wi_c[a:a + 32, None, 2:3]
    zi, zo = wi_c[:, 2], wo_c[:, 2]
    # CK2 incident-cell irradiance histogram on the panel
    wl, ph = D.incident(D.PY, D.LIGHTS['CK2']); wloc = D.loc(wl); E = ph * wloc[:, 2]
    hist = np.bincount(cidx(wloc, NZI, NPI), weights=E, minlength=len(wi_c))
    info = {'ck2_cells_with_99pct_irr': int(np.searchsorted(np.cumsum(np.sort(hist)[::-1]) / hist.sum(), 0.99) + 1)}
    refs = json.load(open(os.path.join(W96, 'deliv96_hp40_nr4.json')))['res']
    res = {}
    CFG = [('flux', 64, ('CK2',)), ('rownorm', 32, ('CK2',)), ('rownorm', 64, ('CK2', 'L2', 'L3')), ('rownorm', 128, ('CK2',)),
           ('ck2w', 16, ('CK2', 'L2', 'L3')), ('ck2w', 64, ('CK2',)), ('full', 0, ('CK2',))]
    for wname, R, lts in CFG:
        if True:
            Kr = np.zeros_like(K)
            for c in range(3):
                if wname == 'full':
                    Kr[..., c] = K[..., c]; continue
                X = K[..., c] * zo[None, :]
                if wname == 'flux':
                    rw = zi.copy()
                elif wname == 'rownorm':
                    rw = 1.0 / np.maximum(X.sum(1), 1e-12)
                else:
                    rw = hist / hist.max() + 1e-6
                Xw = X * rw[:, None]
                U, s, Vt = np.linalg.svd(Xw, full_matrices=False)
                Kr[..., c] = ((U[:, :R] * s[:R]) @ Vt[:R]) / rw[:, None] / zo[None, :]
            path = os.path.join(OUT, 'Ktmp.npy'); np.save(path, Kr.astype(np.float32))
            for ln in lts:
                v = deliver(path, D.LIGHTS[ln]); ref = np.asarray(refs[ln]['REF'])
                res.setdefault(wname, {}).setdefault(ln, {})[R] = [float(v.sum() / ref.sum()), float(np.abs(v - ref).sum() / ref.sum())]
            print(wname, R, json.dumps({ln: res[wname][ln][R] for ln in lts}), flush=True)
    json.dump({'info': info, 'res': res}, open(os.path.join(OUT, 'rankw.json'), 'w'), indent=1)
    os.remove(os.path.join(OUT, 'Ktmp.npy'))
