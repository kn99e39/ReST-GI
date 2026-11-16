"""WL97 audit, track E1, add-on (local, numpy only): what does "rank 16" imply for the CK2 once-scattered S1 sender?
Unconstrained weighted truncated SVD of the stored M64 S1 operator (ops95a.npz 'A4', [incident bin, outgoing bin, channel]) per colour
channel, for several incident-bin weightings; each truncated operator is applied to the deployed CK2 source encoding s16 on S1 and compared
with M64(s16) using the corrected W_T weights. No learning, no non-negativity (Eckart-Young optimum in its own weighted Frobenius norm:
an indication of what the rank alone permits, not an admissible operator).
  python e1_rank.py <repo_root>
"""
import json, math, os, sys
import numpy as np
root = sys.argv[1]
NB = 64; DOM = 2 * math.pi / NB
ZC = ((np.arange(NB) // 16) + 0.5) / 4; WT = ZC * DOM
M = np.load(os.path.join(root, 'results/ecro_attr95a/ops95a.npz'))['A4'].astype(np.float64)
z = np.load(os.path.join(root, 'results/audit97/E1/sender_e1.npz'))
s = z['s16_S1'].astype(np.float64)
tr = json.load(open(os.path.join(root, 'results/audit97/E1/train.json')))['train']
w_train = np.array(tr['direct_irradiance_share_by_incident_bin'])
ck2 = s.sum((0, 2)); ck2 = ck2 / ck2.sum()


def fm(X, Y):
    Wd = WT[None, :, None]; pk = Y.argmax(1)[:, None, :]
    return [round(float(v), 4) for v in ((np.abs(X - Y) * Wd).sum() / (Y * Wd).sum(), (X * Wd).sum() / (Y * Wd).sum(),
                                         np.take_along_axis(X, pk, 1).sum() / np.take_along_axis(Y, pk, 1).sum())]


def trunc(wi, r):
    A = np.zeros_like(M)
    si = np.sqrt(np.maximum(wi, 1e-12)); so = np.sqrt(WT)
    for c in range(3):
        U, S, Vt = np.linalg.svd(si[:, None] * M[:, :, c] * so[None, :], full_matrices=False)
        A[:, :, c] = ((U[:, :r] * S[:r]) @ Vt[:r]) / si[:, None] / so[None, :]
    return A


ref = np.einsum('nmc,mdc->ndc', s, M)
out = {'ck2_incident_bins': np.nonzero(ck2 > 0)[0].tolist(), 'ck2_incident_share': ck2[ck2 > 0].tolist(),
       'train_direct_irradiance_share_on_ck2_bins': float(w_train[ck2 > 0].sum()), 'columns': ['rel_l1', 'energy', 'peak', 'min_entry', 'albedo_max']}
ring0x = lambda k: ZC * np.where(np.arange(NB) < 16, k, 1.0)
for wn, wi in (('flux_zc', ZC), ('uniform', np.ones(NB)), ('train_direct_irradiance', w_train), ('flux_ring0_x10', ring0x(10.0)),
               ('flux_ring0_x100', ring0x(100.0)), ('ck2_bins_only', ck2 + 1e-9)):
    out[wn] = {}
    for r in (4, 8, 16, 32, 48):
        A = trunc(wi, r)
        out[wn][r] = fm(np.einsum('nmc,mdc->ndc', s, A), ref) + [round(float(A.min()), 4), round(float((A * WT[None, :, None]).sum(1).max()), 4)]
json.dump(out, open(os.path.join(root, 'results/audit97/E1/rank.json'), 'w'), indent=1)
for k, v in out.items():
    print(k, json.dumps(v))
