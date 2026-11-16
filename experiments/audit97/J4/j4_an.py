"""WL97 audit, track J4: analysis of j4_attr.py outputs (numpy only).
Relative to the mask total of the stored path-traced reference (e1 + e2 + h, mean of 4 reps), per mask:
  signed (+- SE over reps), L1 of the mean difference, 16x16-block L1 (+ noise realization from the rep half-split),
  debiased relative L2 from the cross product of the two rep halves (unbiased for the structured part).
Usage: python j4_an.py <J4 outdir> <ORIG renders> <masks94.npz> <REFA dir> <out.json>
"""
import glob, json, math, os, sys
import numpy as np

J4D, ORIG, MASKS, REFA, OUT = sys.argv[1:6]
fs = sorted(glob.glob(os.path.join(J4D, 'J4_s*.npz')))
Z = [np.load(f) for f in fs]
R = len(Z)
g = lambda k: np.stack([z[k].astype(np.float64) for z in Z])
ref = {k: np.stack([np.load(os.path.join(ORIG, f'REF_ER_CK_none_CK2_T1_r{r}.npz'))[k].astype(np.float64) for r in range(4)]) for k in ('e1', 'e2', 'h')}
tot = (ref['e1'] + ref['e2'] + ref['h']).mean(0)
mk = np.load(MASKS); sid = mk['sid']
MS = {'ROI_T': mk['ROI_T'], 'DIFF': mk['DIFF'], 'WHOLE': np.ones_like(mk['ROI_T']), 'PANEL': mk['panel'], 'BOXA_PIX': sid == 7}
seeds = [97700 + r for r in range(4)]
ld = lambda k, s: np.load(os.path.join(REFA, f'REFA_k{k}_s{s}.npz'))['h'].astype(np.float64)
h3 = np.stack([ld(3, s) for s in seeds]); h6 = np.stack([ld(6, s) for s in seeds])

A2, A5, B2, B5, C2, C5, D2, D5 = [g(k) for k in ('A2', 'A5', 'B2', 'B5', 'C2', 'C5', 'D2', 'D5')]
T2, T3, T3a, T3b, T2a, T2b, T3o, T3p = [g(k) for k in ('T2', 'T3', 'T3a', 'T3b', 'T2a', 'T2b', 'T3o', 'T3p')]


def blockL1(img, m, b=16):
    x = img * m[..., None]
    H, W = m.shape
    return float(np.abs(x.reshape(H // b, b, W // b, b, 3).sum((1, 3))).sum())


def st(d, m, extra_var=None):
    """d [R, H, W, 3] = per-rep difference images (paired within a rep)."""
    den = tot[m].sum(); t2 = float((tot[m] ** 2).sum())
    k = d.shape[0]
    dm = d.mean(0)
    ha, hb = d[:k // 2].mean(0), d[k // 2:].mean(0)
    s = np.array([x[m].sum() for x in d]) / den
    o = {'signed': float(s.mean()), 'signed_se': float(s.std(ddof=1) / math.sqrt(k)) if k > 1 else None,
         'L1': float(np.abs(dm[m]).sum() / den),
         'L1_block16': blockL1(dm, m) / den, 'L1_block16_noise': blockL1(0.5 * (ha - hb), m) / den,
         'L1_block8': blockL1(dm, m, 8) / den, 'L1_block8_noise': blockL1(0.5 * (ha - hb), m, 8) / den,
         'L1_block64': blockL1(dm, m, 64) / den, 'L1_block64_noise': blockL1(0.5 * (ha - hb), m, 64) / den}
    cp = float((ha[m] * hb[m]).sum())
    o['l1_equiv_of_debiased_l2 (gaussian, x rms/mean of tot)'] = math.copysign(math.sqrt(abs(cp) / m.sum()) * math.sqrt(2 / math.pi) / (den / m.sum()), cp)
    o['rel_l2_debiased_vs_tot'] = math.copysign(math.sqrt(abs(cp) / t2), cp)
    return o


def st_ext(x, y, m):
    """x [R, ...] own arm vs independent external reference y [4, ...] (not paired)."""
    den = tot[m].sum(); t2 = float((tot[m] ** 2).sum())
    d = x.mean(0) - y.mean(0)
    kx, ky = x.shape[0], y.shape[0]
    nz = 0.5 * (x[:kx // 2].mean(0) - x[kx // 2:].mean(0)) - 0.5 * (y[:ky // 2].mean(0) - y[ky // 2:].mean(0))
    sx = np.array([a[m].sum() for a in x]) / den; sy = np.array([a[m].sum() for a in y]) / den
    cp = float(((x[:kx // 2].mean(0) - y[:ky // 2].mean(0))[m] * (x[kx // 2:].mean(0) - y[ky // 2:].mean(0))[m]).sum())
    return {'signed': float(sx.mean() - sy.mean()), 'signed_se': float(math.sqrt(sx.var(ddof=1) / kx + sy.var(ddof=1) / ky)) if kx > 1 else None,
            'ratio': float(sx.mean() / sy.mean()), 'L1': float(np.abs(d[m]).sum() / den),
            'L1_block16': blockL1(d, m) / den, 'L1_block16_noise': blockL1(nz, m) / den,
            'rel_l2_debiased_vs_tot': math.copysign(math.sqrt(abs(cp) / t2), cp)}


O = {'n_reps': R, 'spp_per_rep': int(Z[0]['spp']), 'files': [os.path.basename(f) for f in fs], 'masks': {}}
for mn, m in MS.items():
    den = tot[m].sum()
    q = {'pixels': int(m.sum()), 'rms_over_mean_of_tot': float(math.sqrt((tot[m] ** 2).mean()) / tot[m].mean()),
         'shares_of_mask_total': {'A2 (3 camera events)': float(A2.mean(0)[m].sum() / den), 'A5-A2 (4..6)': float((A5 - A2).mean(0)[m].sum() / den),
                                  'T3o read': float(T3o.mean(0)[m].sum() / den), 'T3p read': float(T3p.mean(0)[m].sum() / den)},
         'sanity_A2_vs_REFA_h3': st_ext(A2, h3, m), 'sanity_A5_vs_REFA_h6': st_ext(A5, h6, m),
         'sanity_tailA_vs_REFA_h6_minus_h3': st_ext(A5 - A2, h6 - h3, m)}
    for part, (a, b_, c, d_, t) in {'e2_content (camera 3 events)': (A2, B2, C2, D2, T2),
                                   'tail (camera 4..6 events)': (A5 - A2, B5 - B2, C5 - C2, D5 - D2, T3),
                                   'all (camera 3..6 events)': (A5, B5, C5, D5, T2 + T3)}.items():
        q[part] = {'D-A discretisation (patch x bin), no table noise': st(d_ - a, m),
                   'B-A spatial only (random point in patch, exact direction)': st(b_ - a, m),
                   'C-A angular only (exact point, random direction in bin)': st(c - a, m),
                   'D-B-C+A interaction': st(d_ - b_ - c + a, m),
                   'T-D stored table minus its expectation (table noise + inconsistency)': st(t - d_, m),
                   'T-A stored table read minus exact continuation (total readout error)': st(t - a, m)}
    q['table half noise 0.5*(T3a-T3b)'] = st(0.5 * (T3a - T3b), m)
    q['table half noise 0.5*(T2a-T2b)'] = st(0.5 * (T2a - T2b), m)
    q['T_vs_REFA: T2 vs h3'] = st_ext(T2, h3, m); q['T_vs_REFA: T3 vs h6-h3'] = st_ext(T3, h6 - h3, m)
    q['T_vs_stored_REF_h: T2+T3'] = st_ext(T2 + T3, ref['h'], m)
    q['probe_fail_weight_share'] = {a: float(g('fail_' + a).mean(0)[m].sum() / max(g('wsum_' + a).mean(0)[m].sum(), 1e-30)) for a in 'ABCD'}
    O['masks'][mn] = q
json.dump(O, open(OUT, 'w'), indent=1)
sh = lambda v: ' '.join(f'{a.split(" ")[0]}={b:+.5f}' for a, b in v.items() if b is not None)
for mn in MS:
    m = O['masks'][mn]
    print('=====', mn, m['pixels'], 'rms/mean', round(m['rms_over_mean_of_tot'], 3), json.dumps({k: round(v, 4) for k, v in m['shares_of_mask_total'].items()}))
    for k in ('sanity_A2_vs_REFA_h3', 'sanity_tailA_vs_REFA_h6_minus_h3', 'T_vs_REFA: T2 vs h3', 'T_vs_REFA: T3 vs h6-h3', 'T_vs_stored_REF_h: T2+T3',
              'table half noise 0.5*(T2a-T2b)', 'table half noise 0.5*(T3a-T3b)'):
        print(k[:34].ljust(34), sh(m[k]))
    print('fail', json.dumps({k: round(v, 6) for k, v in m['probe_fail_weight_share'].items()}))
    for part in ('e2_content (camera 3 events)', 'tail (camera 4..6 events)', 'all (camera 3..6 events)'):
        for k, v in m[part].items():
            print(part[:4], '|', k[:22].ljust(22), '|', sh(v))
