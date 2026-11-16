"""WL97 audit, track J4: split of the camera-vertex-2 patch-discretisation error (D - A, j4_attr.py arms) and of the stored-table read
error (T - A) by whether the camera's FIRST vertex lies on the sharp panel (masks94 'panel') or not, inside ROI_T.  numpy only.
Usage: python j4_split.py <J4 outdir> <ORIG renders> <masks94.npz>
"""
import glob, os, sys
import numpy as np
J4D, ORIG, MASKS = sys.argv[1:4]
Z = [np.load(f) for f in sorted(glob.glob(os.path.join(J4D, 'J4_s*.npz')))]
g = lambda k: np.stack([z[k].astype(np.float64) for z in Z])
ref = [np.load(os.path.join(ORIG, f'REF_ER_CK_none_CK2_T1_r{r}.npz')) for r in range(4)]
tot = np.mean([r['e1'].astype(np.float64) + r['e2'] + r['h'] for r in ref], 0)
mk = np.load(MASKS); roi = mk['ROI_T'].astype(bool); pan = mk['panel'].astype(bool)
A2, A5, D2, D5, T2, T3 = [g(k) for k in ('A2', 'A5', 'D2', 'D5', 'T2', 'T3')]
k = A2.shape[0]
def dbl2(d, m):                       # debiased rel L2 (cross product of rep halves), rel to ||tot||_2 over the ROI_T mask
    a, b = d[:k // 2].mean(0)[m], d[k // 2:].mean(0)[m]
    return float(np.sign((a * b).sum()) * np.sqrt(abs((a * b).sum())) / np.sqrt((tot[roi] ** 2).sum()))
def l1b(d, m, bs=16):
    x = d.mean(0) * m[..., None]; H, W = m.shape
    return float(np.abs(x.reshape(H // bs, bs, W // bs, bs, 3).sum((1, 3))).sum() / tot[roi].sum())
for nm, m in (('ROI_T', roi), ('ROI_T & panel', roi & pan), ('ROI_T \ panel', roi & ~pan)):
    sh = tot[m].sum() / tot[roi].sum()
    print(f'{nm:16s} pix {int(m.sum()):6d} share_of_ROI_total {sh:.3f}')
    for lab, d in (('tail D-A', (D5 - D2) - (A5 - A2)), ('tail T-A', T3 - (A5 - A2)), ('e2 D-A', D2 - A2), ('e2 T-A', T2 - A2)):
        print(f'   {lab:9s} debiasedL2/ROI {dbl2(d, m):+.5f}  blockL1_16/ROI {l1b(d, m):.5f}  signed/ROI {d.mean(0)[m].sum() / tot[roi].sum():+.5f}')
