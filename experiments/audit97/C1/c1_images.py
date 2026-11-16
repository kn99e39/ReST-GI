"""WL97 audit, track C1: independent recomputation of the WL94..WL94D image metrics (E_H on ROI_T / DIFF) from the stored render replicates,
with a statistical characterisation the original 'noise_bound' does not give. numpy only; reads ~/rest_x94d/results/{renders,pre} read-only.

  python c1_images.py <R> <out.json>
Definitions (protocol ecro_mat94 / analyze86 header): E_H = sum_m |hl - h| / sum_m tot,  signed = sum_m (hl - h) / sum_m tot,
tot = e1 + e2 + h of the reference (replicate mean), channel-wise sums over the pixels of the mask m.
Added by the audit:
  nb_coded   the original bound: (0.5 sum|x_a - x_b| + 0.5 sum|h_a - h_b|) / den (first vs second half of the replicates)
  se_mc      Monte-Carlo standard error of E_H: pixels are independent estimators, Var(|z|) <= Var(z)  =>  se <= sqrt(sum var_x + var_h) / den
  floor      E|noise| inflation bound of the L1 statistic: sum sqrt(2/pi (var_x + var_h)) / den  (attained only where the true error is 0)
  half1/2    the statistic on two disjoint data halves (arm replicate 0 + reference replicates 0,1 | arm replicate 1 + reference 2,3)
Pairwise (same seed, variant none, ROI_T): d = E_H(A) - E_H(B), its two disjoint-half values, and se_d <= sqrt(sum var_xA + var_xB + 4 var_h)/den.
"""
import glob, json, os, re, sys
import numpy as np

R, OUTFN = sys.argv[1:3]
RD = os.path.join(R, 'results', 'renders')
VAR = ['none', 'U07', 'U03', 'bx75', 'bx38', 'bx38i']
TAG = 'CK2_T1'
M = np.load(os.path.join(R, 'results', 'pre', 'masks94.npz'))
REG = {'ROI_T': M['ROI_T'], 'DIFF': M['DIFF']}


def stack(pat, key):
    fns = sorted(glob.glob(os.path.join(RD, pat)))
    return np.stack([np.load(f)[key].astype(np.float64) for f in fns]) if fns else None


refs = {}
for v in VAR:
    h = stack(f'REF_ER_CK_{v}_{TAG}_r*.npz', 'h')
    tot = stack(f'REF_ER_CK_{v}_{TAG}_r*.npz', 'e1') + stack(f'REF_ER_CK_{v}_{TAG}_r*.npz', 'e2') + h
    refs[v] = {'h': h, 'tot': tot.mean(0), 'hm': h.mean(0), 'var_h': h.var(0, ddof=1) / h.shape[0]}
    assert h.shape[0] == 4

fams = sorted({re.match(r'CK_(?:%s)_(.+)_%s_r\d+\.npz' % ('|'.join(sorted(VAR, key=len, reverse=True)), TAG), os.path.basename(f)).group(1)
               for f in glob.glob(os.path.join(RD, f'CK_*_{TAG}_r*.npz'))})
print(len(fams), 'render families', flush=True)


def stat(x, ref, m):
    """x [k, H, W, 3] arm replicates."""
    k = x.shape[0]; h = ref['h']; n = h.shape[0]
    den = ref['tot'][m].sum()
    xm = x.mean(0)
    var_x = x.var(0, ddof=1) / k
    e = float(np.abs(xm[m] - ref['hm'][m]).sum() / den)
    o = {'E_H': e, 'signed': float((xm[m] - ref['hm'][m]).sum() / den), 'H_share': float(ref['hm'][m].sum() / den), 'pixels': int(m.sum()),
         'nb_coded': float((0.5 * np.abs(x[:k // 2].mean(0)[m] - x[k // 2:].mean(0)[m]).sum() + 0.5 * np.abs(h[:n // 2].mean(0)[m] - h[n // 2:].mean(0)[m]).sum()) / den),
         'se_mc': float(np.sqrt((var_x[m] + ref['var_h'][m]).sum()) / den),
         'se_signed': float(np.sqrt((var_x[m] + ref['var_h'][m]).sum()) / den),
         'floor': float(np.sqrt(2 / np.pi * (var_x[m] + ref['var_h'][m])).sum() / den),
         'half1': float(np.abs(x[0][m] - h[:2].mean(0)[m]).sum() / den), 'half2': float(np.abs(x[1][m] - h[2:].mean(0)[m]).sum() / den),
         'replicates': int(k)}
    return o, var_x


out = {'families': fams, 'E_H': {}, 'pairs': {}}
keep = {}
for fam in fams:
    for v in VAR:
        x = stack(f'CK_{v}_{fam}_{TAG}_r*.npz', 'hl')
        if x is None:
            continue
        for rn, m in REG.items():
            o, var_x = stat(x, refs[v], m)
            out['E_H'][f'{fam}|{v}|{rn}'] = o
            if v == 'none' and rn == 'ROI_T':
                keep[fam] = (x, var_x)
    print(fam, 'none ROI_T', {k: round(val, 5) for k, val in out['E_H'][f'{fam}|none|ROI_T'].items() if k in ('E_H', 'signed', 'nb_coded', 'se_mc', 'floor', 'half1', 'half2')}, flush=True)

m = REG['ROI_T']; ref = refs['none']; den = ref['tot'][m].sum(); h = ref['h']
for sd in ('s8601', 's8602'):
    fs = [f for f in keep if f'_{sd}_' in f]
    for a in fs:
        for b in fs:
            if a >= b:
                continue
            (xa, va), (xb, vb) = keep[a], keep[b]
            ea = lambda x, hh: np.abs(x[m] - hh[m]).sum() / den
            d = ea(xa.mean(0), ref['hm']) - ea(xb.mean(0), ref['hm'])
            d1 = ea(xa[0], h[:2].mean(0)) - ea(xb[0], h[:2].mean(0)); d2 = ea(xa[1], h[2:].mean(0)) - ea(xb[1], h[2:].mean(0))
            out['pairs'][f'{a}|{b}'] = {'d': float(d), 'd_half1': float(d1), 'd_half2': float(d2),
                                        'se_d_bound': float(np.sqrt((va[m] + vb[m] + 4 * ref['var_h'][m]).sum()) / den)}
json.dump(out, open(OUTFN, 'w'), indent=1)
print('WROTE', OUTFN)
