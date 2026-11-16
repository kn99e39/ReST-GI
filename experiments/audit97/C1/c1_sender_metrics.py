"""WL97 audit, track C1: independent WL94D sender metrics from the raw dump of c1_sender_dump.py (numpy only).

  python c1_sender_metrics.py <SND_DIR> <R94D> <out.json>
Recomputes, per model (own / D / T / R / P x 2 seeds), on CK2 and pooled over the 24 validation items:
  field rel L1 / energy ratio / peak ratio of Bl vs Bm on the S1 panel, under coded (8 x 8) / true (4 x 16) / no angular weight
  delivery rel L1 / ratio through the exact receiver for receiver sets: all non-panel, box A, box B, box A + B (= s94.py 1130:1238), panel face
  S1 local operator vs M64 in MATCHED norms: pooled flux-L1, pooled flux-Frobenius (per colour), mass, operator-level peak retention
  reference operators in the same norms: zero, SVD-16 (Eckart-Young), SVD-16 clipped at 0, NMF-16 (Frobenius multiplicative updates)
  CK2 single-flight delivered exitance vs the physical e2_p field reference (flight 'floor'): learned and M64 sender, both weights
"""
import json, os, sys
import numpy as np

SND, R, OUTFN = sys.argv[1:4]
NB = 64
DOM = 2 * np.pi / NB
W = {'coded': ((np.arange(NB) // 8) + 0.5) / 8 * DOM, 'true': ((np.arange(NB) // 16) + 0.5) / 4 * DOM, 'none': np.ones(NB)}
ZB = ((np.arange(NB) // 16) + 0.5) / 4
PANEL = np.arange(1088, 1130)
SETS = {'receivers': np.setdiff1d(np.arange(1270), PANEL), 'boxA': np.arange(1130, 1184), 'boxB': np.arange(1184, 1238),
        'boxAB': np.arange(1130, 1238), 'panel_face': np.arange(1130, 1139)}


def field(Bl, Bm):
    Bl, Bm = Bl.astype(np.float64), Bm.astype(np.float64)
    o = {}
    for wn, w in W.items():
        ww = w[None, None, :, None]
        o[wn] = [float((np.abs(Bl - Bm) * ww).sum()), float((Bm * ww).sum()), float((Bl * ww).sum())]
    pk = Bm.argmax(2)[:, :, None, :]
    o['peak'] = [float(np.take_along_axis(Bl, pk, 2).sum()), float(np.take_along_axis(Bm, pk, 2).sum())]
    return o


def pearson(a, b):
    a, b = a - a.mean(), b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else None


def add(acc, x):
    for k, v in x.items():
        acc[k] = [p + q for p, q in zip(acc[k], v)] if k in acc else list(v)


def fin_field(a):
    o = {wn: {'rel_l1': a[wn][0] / a[wn][1], 'energy_ratio': a[wn][2] / a[wn][1]} for wn in W}
    o['peak_ratio'] = a['peak'][0] / a['peak'][1]
    return o


def delivery(z, pre):
    o = {}
    for wn in ('coded', 'true'):
        el, em = z[f'{pre}|el_{wn}'], z[f'{pre}|em_{wn}']
        for sn, idx in SETS.items():
            o[f'{wn}|{sn}'] = [float(np.abs(el - em)[:, idx].sum()), float(em[:, idx].sum()), float(el[:, idx].sum())]
    return o


def fin_del(a):
    return {k: {'rel_l1': v[0] / v[1] if v[1] > 0 else None, 'ratio': v[2] / v[1] if v[1] > 0 else None} for k, v in a.items()}


def fin_op(ops):
    s = lambda k: sum(o[k] for o in ops)
    fn = np.sum([o['fro_num_c'] for o in ops], 0); fd = np.sum([o['fro_den_c'] for o in ops], 0)
    return {'flux_l1': s('l1_num') / s('l1_den'), 'mass': s('mass_num') / s('l1_den'), 'flux_fro_pooled_c': np.sqrt(fn / fd).tolist(),
            'flux_fro_patchmean_c': np.mean([o['fro_rel_c_patchmean'] for o in ops], 0).tolist(), 'op_peak_retention': s('peak_num') / s('peak_den')}


# ---------------------------------------------------------------------------------------- reference operators in the matched norms
M1 = np.load(os.path.join(SND, 'M64_S1.npy')).astype(np.float64)            # [bi, bo, 3]
fw = ZB[:, None] * ZB[None, :]


def opm(Ah):
    """metrics of an approximation Ah [bi, bo, 3] of M1 in the WL94D pooled flux-L1 norm and the WL92 per-colour flux-Frobenius norm."""
    X, T = Ah * fw[..., None], M1 * fw[..., None]
    pk = T.argmax(1)
    return {'flux_l1': float(np.abs(X - T).sum() / T.sum()), 'mass': float(X.sum() / T.sum()),
            'flux_fro_c': [float(np.linalg.norm(X[..., c] - T[..., c]) / np.linalg.norm(T[..., c])) for c in range(3)],
            'op_peak_retention': float(np.take_along_axis(X, pk[:, None, :], 1).sum() / T.max(1).sum()), 'min': float(Ah.min())}


def nmf(V, r, iters=4000, seed=0):
    rs = np.random.default_rng(seed)
    Wm = rs.random((V.shape[0], r)) * np.sqrt(V.mean() / r) + 1e-6; H = rs.random((r, V.shape[1])) * np.sqrt(V.mean() / r) + 1e-6
    for _ in range(iters):
        H *= (Wm.T @ V) / (Wm.T @ Wm @ H + 1e-30)
        Wm *= (V @ H.T) / (Wm @ H @ H.T + 1e-30)
    return Wm @ H


refops = {'zero': opm(np.zeros_like(M1))}
for r in (8, 16, 32):
    Ah = np.zeros_like(M1); Ac = np.zeros_like(M1)
    for c in range(3):
        X = M1[..., c] * fw
        U, s, Vt = np.linalg.svd(X)
        Xr = (U[:, :r] * s[:r]) @ Vt[:r]
        Ah[..., c] = Xr / fw; Ac[..., c] = np.maximum(Xr, 0) / fw
    refops[f'svd{r}'] = opm(Ah); refops[f'svd{r}_clip0'] = opm(Ac)
An = np.zeros_like(M1)
for c in range(3):
    best = None
    for sd in range(3):
        Xn = nmf(M1[..., c] * fw, 16, seed=sd)
        e = np.linalg.norm(Xn - M1[..., c] * fw)
        if best is None or e < best[0]:
            best = (e, Xn)
    An[..., c] = best[1] / fw
refops['nmf16_fro'] = opm(An)
# directional albedo of M64 S1 per incident bin under both weights (energy accounting)
refops['M64_S1_albedo'] = {wn: {'max': float((M1 * W[wn][None, :, None]).sum(1).max()), 'mean': float((M1 * W[wn][None, :, None]).sum(1).mean())} for wn in ('coded', 'true')}

# ---------------------------------------------------------------------------------------- per model
meta = json.load(open(os.path.join(SND, 'sender_dump_meta.json')))
zf = np.load(os.path.join(R, 'results', 'fields', 'CKF_none.npz'))
out = {'reference_operators': refops, 'models': {}, 'identity_vs_stored': {}}
for tag, m in meta.items():
    z = np.load(os.path.join(SND, f'SND_{tag}.npz'))
    o = {}
    o['ck2'] = {'field': fin_field(field(z['ck2|Bl'], z['ck2|Bm'])), 'delivery': fin_del(delivery(z, 'ck2')), 'operator': fin_op([m['op']['ck2']])}
    fa, da = {}, {}
    nv = len(m['op']['val'])
    for i in range(nv):
        add(fa, field(z[f'val{i:02d}|Bl'], z[f'val{i:02d}|Bm'])); add(da, delivery(z, f'val{i:02d}'))
    o['val'] = {'field': fin_field(fa), 'delivery': fin_del(da), 'operator': fin_op(m['op']['val'])}
    # CK2 delivered single flight (exact receiver) vs the physical e2_p reference
    fl = {}
    for wn in ('coded', 'true'):
        w = W[wn]
        ea, eb = np.einsum('nbc,b->n', zf['e2_p_a'].astype(np.float64), w), np.einsum('nbc,b->n', zf['e2_p_b'].astype(np.float64), w)
        er = 0.5 * (ea + eb)
        for sn, key in (('learned_sender', 'ck2|Dl'), ('M64_sender', 'ck2|Dm')):
            ex = np.einsum('nbc,b->n', z[key].astype(np.float64), w)
            for rn in ('receivers', 'boxA', 'boxB', 'boxAB', 'panel_face'):
                idx = SETS[rn]
                fl[f'{wn}|{sn}|{rn}'] = {'ratio': float(ex[idx].sum() / er[idx].sum()), 'corr': pearson(ex[idx], er[idx]), 'corr_halves': pearson(ea[idx], eb[idx]),
                                         'rel_l1': float(np.abs(ex[idx] - er[idx]).sum() / er[idx].sum()),
                                         'half_noise_rel_l1': float(0.5 * np.abs(ea[idx] - eb[idx]).sum() / er[idx].sum()), 'n': int(len(idx))}
    o['ck2_flight_vs_e2p'] = fl
    out['models'][tag] = o
    # identity with the stored WL94D sender JSON (coded weights, box = 1130:1238)
    k, sd = tag.split('_s')
    sj = os.path.join(R, 'results', 'sender', f'SN_{k}_s{sd}.json')
    if os.path.exists(sj):
        st = json.load(open(sj)); d = {}
        for sp in ('val', 'ck2'):
            mine = {'rel_l1_flux': o[sp]['field']['coded']['rel_l1'], 'energy_ratio': o[sp]['field']['coded']['energy_ratio'], 'peak_ratio': o[sp]['field']['peak_ratio'],
                    'delivery_rel_l1': o[sp]['delivery']['coded|receivers']['rel_l1'], 'delivery_ratio': o[sp]['delivery']['coded|receivers']['ratio'],
                    'delivery_box_rel_l1': o[sp]['delivery']['coded|boxAB']['rel_l1'], 'operator_rel_l1': o[sp]['operator']['flux_l1']}
            for kk, v in mine.items():
                s0 = st[sp][kk] if kk != 'operator_rel_l1' else st[sp][kk]['train_bins']
                d[f'{sp}|{kk}'] = {'stored': s0, 'mine': v, 'abs_diff': abs(s0 - v)}
        out['identity_vs_stored'][tag] = d
        print(tag, 'max |stored - mine| (coded, boxAB):', max(x['abs_diff'] for x in d.values()), flush=True)
json.dump(out, open(OUTFN, 'w'), indent=1)
print('WROTE', OUTFN)
