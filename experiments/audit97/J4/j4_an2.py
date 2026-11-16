"""WL97 audit, track J4: second analysis (numpy only) - uses the INDEPENDENT audit reference tables (track E4, 4 new field halves) read on
the same camera paths (j4_tread.py) to separate, without any noise model,
  * table estimation error of the CKF reference (Tc - Tn; per-table variance from the audit half split),
  * the part of 'table read minus exact continuation' that is common to two independent table references
    = cross product <Tc - A, Tn - A> (table noise drops out; equals the discretisation error D - A if D is the table expectation),
  * any systematic difference between a table and the probe arm D: cross product <Tc - D, Tn - D> (zero if D is the table expectation).
Usage: python j4_an2.py <J4 outdir> <ORIG renders> <masks94.npz> <out.json>
"""
import glob, json, math, os, sys
import numpy as np

J4D, ORIG, MASKS, OUT = sys.argv[1:5]
fa = sorted(glob.glob(os.path.join(J4D, 'J4_s*.npz')))
ft = [f.replace('J4_s', 'J4T_s') for f in fa]
ZA = [np.load(f) for f in fa]; ZT = [np.load(f) for f in ft]
R = len(ZA)
ga = lambda k: np.stack([z[k].astype(np.float64) for z in ZA])
gt = lambda k: np.stack([z[k].astype(np.float64) for z in ZT])
ref = {k: np.stack([np.load(os.path.join(ORIG, f'REF_ER_CK_none_CK2_T1_r{r}.npz'))[k].astype(np.float64) for r in range(4)]) for k in ('e1', 'e2', 'h')}
tot = (ref['e1'] + ref['e2'] + ref['h']).mean(0)
mk = np.load(MASKS); sid = mk['sid']
MS = {'ROI_T': mk['ROI_T'], 'DIFF': mk['DIFF'], 'WHOLE': np.ones_like(mk['ROI_T']), 'PANEL': mk['panel'], 'BOXA_PIX': sid == 7}
A2, A5, D2, D5 = ga('A2'), ga('A5'), ga('D2'), ga('D5')
T2c, T3c, T2n, T3n = gt('T2c'), gt('T3c'), gt('T2n'), gt('T3n')
T2n01, T2n23, T3n01, T3n23 = gt('T2n01'), gt('T2n23'), gt('T3n01'), gt('T3n23')
O = {'n_reps': R, 'spp_per_rep': int(ZA[0]['spp']),
     'same_lanes_check_max_abs(T2c - T2 of j4_attr)': float(np.abs(T2c - ga('T2')).max()), 'masks': {}}


def blockL1(img, m, b=16):
    x = img * m[..., None]
    H, W = m.shape
    return float(np.abs(x.reshape(H // b, b, W // b, b, 3).sum((1, 3))).sum())


def cross(x, y, m):
    """rel L2 (vs rms of tot) from the cross product of x (first half of the reps) and y (second half) and vice versa (independent MC)."""
    t2 = float((tot[m] ** 2).sum()); k = x.shape[0]
    cp = 0.5 * (float((x[:k // 2].mean(0)[m] * y[k // 2:].mean(0)[m]).sum()) + float((x[k // 2:].mean(0)[m] * y[:k // 2].mean(0)[m]).sum()))
    return math.copysign(math.sqrt(abs(cp) / t2), cp)


def det(x, m):
    """deterministic (paired) difference image x [R, ...]: signed, L1, block-16 L1, rel L2."""
    den = tot[m].sum(); t2 = float((tot[m] ** 2).sum()); d = x.mean(0)
    return {'signed': float(d[m].sum() / den), 'L1': float(np.abs(d[m]).sum() / den), 'L1_block16': blockL1(d, m) / den,
            'rel_l2': math.sqrt(float((d[m] ** 2).sum()) / t2)}


def sg(x, m):
    den = tot[m].sum(); s = np.array([a[m].sum() for a in x]) / den
    return {'signed': float(s.mean()), 'se': float(s.std(ddof=1) / math.sqrt(len(s)))}


for mn, m in MS.items():
    q = {}
    for part, (a, d, tc, tn, n01, n23) in {'e2_content (camera 3 events)': (A2, D2, T2c, T2n, T2n01, T2n23),
                                          'tail (camera 4..6 events)': (A5 - A2, D5 - D2, T3c, T3n, T3n01, T3n23)}.items():
        vn = det(0.5 * (n01 - n23), m)                          # noise of the audit 4-half mean (one realization)
        q[part] = {
            'Tc - Tn (two independent table references, paired camera paths)': det(tc - tn, m),
            'audit table noise of the 4-half mean, 0.5 (n01 - n23)': vn,
            'implied CKF table noise rel_l2 = sqrt(|Tc-Tn|^2 - noise_n^2)': math.sqrt(max(det(tc - tn, m)['rel_l2'] ** 2 - vn['rel_l2'] ** 2, 0.0)),
            'Tc - A signed': sg(tc - a, m), 'Tn - A signed': sg(tn - a, m), 'Tc - D signed': sg(tc - d, m), 'Tn - D signed': sg(tn - d, m),
            'D - A signed': sg(d - a, m),
            'debiased rel_l2: D - A (cross halves)': cross(d - a, d - a, m),
            'debiased rel_l2: Tc - A': cross(tc - a, tc - a, m), 'debiased rel_l2: Tn - A': cross(tn - a, tn - a, m),
            'table-noise-free rel_l2 common to both tables vs truth: <Tc - A, Tn - A>': cross(tc - a, tn - a, m),
            'debiased rel_l2: Tc - D': cross(tc - d, tc - d, m), 'debiased rel_l2: Tn - D': cross(tn - d, tn - d, m),
            'systematic table-vs-D rel_l2: <Tc - D, Tn - D>': cross(tc - d, tn - d, m)}
    O['masks'][mn] = q
json.dump(O, open(OUT, 'w'), indent=1)
print('same lanes', O['same_lanes_check_max_abs(T2c - T2 of j4_attr)'], 'reps', R)
for mn in ('ROI_T', 'PANEL', 'WHOLE', 'BOXA_PIX'):
    for part, v in O['masks'][mn].items():
        print('==', mn, part)
        for k, x in v.items():
            print('  ', k[:74].ljust(74), json.dumps({a: round(b, 5) for a, b in x.items()}) if isinstance(x, dict) else round(x, 5))
