"""Predicted single-draw variance of the receiver-anchored 512-ray gather (8 rays / equal-solid-angle bin, random origin over the patch)
for the exact per-ray S1 radiance (X1) and the binned M64 table, from the deterministic panel quadrature (scratch, WL96)."""
import json, math, os, sys
import numpy as np
os.environ.setdefault('HP', '0.004'); os.environ.setdefault('NR', '4')
import deliv96 as D
from multiprocessing import Pool
DOM = 2 * math.pi / 64

def work(args):
    idx, pl = args
    arms, Bq = D.setup_arms(pl)
    xs, ns, pid = D.receivers()
    res = []
    for i in idx:
        x, n, p = xs[i], ns[i], pid[i]
        t, b = D.G.PT[p], D.G.PB[p]
        d = x - D.PY; r2 = (d * d).sum(1); v = d / np.sqrt(r2)[:, None]
        cy = v @ D.N_; cx = -(v @ n); ok = (cy > 1e-6) & (cx > 1e-6)
        out = {}
        if ok.any():
            dw = (cy / r2 * D.HP ** 2)[ok]                     # solid angle of each panel sample seen from x
            w = -v[ok]; wl = np.stack([w @ t, w @ b, w @ n], -1)
            rb = D.bin64(wl)
            vl = D.loc(v[ok])
            for k in ('REF', 'NODE200mm', 'BIC1024'):
                wi, ph = arms[k]
                g = (D.er.eval(vl, wi[ok]) * ph[ok][:, None]).sum(1) * cx[ok]      # channel-summed L cos_x
                out[k] = (np.bincount(rb, g * dw, 64), np.bincount(rb, DOM * g * g * dw, 64))
            g = Bq[D.PATCH[ok], D.bin64(vl)].sum(1) * cx[ok]
            out['M64'] = (np.bincount(rb, g * dw, 64), np.bincount(rb, DOM * g * g * dw, 64))
            gr = (D.er.eval(vl, arms['REF'][0][ok]) * arms['REF'][1][ok][:, None]).sum(1) * cx[ok]
            o = np.argsort(-gr); c = np.cumsum((gr * dw)[o]); tot = c[-1]
            out['omega'] = [float(dw[o][:np.searchsorted(c, q * tot) + 1].sum()) for q in (0.5, 0.9)]
        res.append((p, out))
    return res

if __name__ == '__main__':
    xs, ns, pid = D.receivers()
    allr = {}
    for ln, pl in D.LIGHTS.items():
        ch = np.array_split(np.arange(len(xs)), 32)
        with Pool(16) as po:
            parts = sum(po.map(work, [(c, pl) for c in ch]), [])
        arms = ['REF', 'NODE200mm', 'BIC1024', 'M64']
        EY = {k: np.zeros((54, 64)) for k in arms}; EY2 = {k: np.zeros((54, 64)) for k in arms}; om = np.zeros((54, 2)); cnt = np.zeros(54)
        for p, o in parts:
            j = p - 1130; cnt[j] += 1
            for k in arms:
                if k in o:
                    EY[k][j] += o[k][0]; EY2[k][j] += o[k][1]
            if 'omega' in o:
                om[j] += o['omega']
        res = {}
        rng = np.random.default_rng(0)
        for k in arms:
            m1 = EY[k] / cnt[:, None]; m2 = EY2[k] / cnt[:, None]
            mean_p = m1.sum(1)                                  # patch E (channel-summed)
            var_p = ((m2 - m1 ** 2) / 8).sum(1)                 # 8 rays per bin
            tot = mean_p.sum()
            # Gaussian-approx expected per-draw abs rel L1 on box A, and draws needed for pooled SE-L1 <= 0.059 (ref half-noise)
            el1 = float((np.sqrt(2 / math.pi) * np.sqrt(var_p)).sum() / tot)
            res[k] = {'rel_se_boxA_total': float(np.sqrt(var_p.sum()) / tot), 'expected_abs_rel_l1_single_draw_gauss': el1,
                      'draws_for_l1_0.059': float((el1 / 0.059) ** 2), 'draws_for_l1_0.02': float((el1 / 0.02) ** 2)}
        omm = om / cnt[:, None]
        E = (EY['REF'] / cnt[:, None]).sum(1); top = np.argsort(-E)[:9]
        res['omega50_90_sr_top9_mean'] = omm[top].mean(0).tolist()
        res['expected_rays_in_omega50_90_per_draw_top9'] = (omm[top].mean(0) * 8 / DOM).tolist()
        allr[ln] = res
        print(ln, json.dumps(res, indent=0), flush=True)
    json.dump(allr, open('var96.json', 'w'), indent=1)
