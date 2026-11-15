"""WL96 CPU-only algebra: support-restricted exactness of rank-limited operators on ring-0 inputs, and the ring allocation of the WL95A bound
operators (inputs: results/ecro_attr95a/ops95a.npz = A1 NMF16, A2 L1-NMF16, A3 rank-32, A4 M64 S1; CK2 support weights from ck2_support_E.npy)."""
import os
import numpy as np
H = os.path.dirname(os.path.abspath(__file__))
z = np.load(os.path.join(H, '..', 'ecro_attr95a', 'ops95a.npz'))
M = z['A4'].astype(np.float64)                      # [in 64, out 64, 3]
ZB = ((np.arange(64) // 16) + 0.5) / 4; WT = ZB * 2 * np.pi / 64; ZZ = ZB[:, None] * ZB[None, :]
E = np.load(os.path.join(H, 'ck2_support_E.npy'))
ring0 = np.arange(16); sup = np.nonzero(E)[0]
out = {}
# (1) M P_ring0: rank and an exact non-negative rank-16 factorisation (c = one-hot on ring-0 inputs, a = M rows)
P = np.zeros((64, 64)); P[ring0, ring0] = 1
for c in range(3):
    Mr = P @ M[..., c]
    out[f'rank_MP_ring0_c{c}'] = int(np.linalg.matrix_rank(Mr))
C = np.zeros((64, 16)); C[ring0, np.arange(16)] = 1          # c: input bin -> channel (non-negative)
Afac = np.stack([C @ M[ring0, :, c] for c in range(3)], -1)   # a c^T reproduces M on ring-0 rows
rng = np.random.default_rng(96000)
s = np.zeros((1000, 64, 3)); s[:, ring0] = rng.random((1000, 16, 3))
err = np.abs(np.einsum('kmc,mdc->kdc', s, Afac) - np.einsum('kmc,mdc->kdc', s, M)).max() / np.einsum('kmc,mdc->kdc', s, M).max()
out['ring0_nonneg_rank16_exact_max_rel'] = float(err)
out['ck2_support_bins'] = sup.tolist(); out['ck2_support_rank_needed'] = int(len(sup))
# (2) where the WL95A bound operators spend their error: row-wise flux error per incident ring, and on the CK2 support (E-weighted)
def rows(A):
    num = (np.abs(A - M) * ZZ[..., None]).sum((1, 2)); den = (M * ZZ[..., None]).sum((1, 2))
    r = {f'ring{i}_op_l1': float(num[16 * i:16 * i + 16].sum() / den[16 * i:16 * i + 16].sum()) for i in range(4)}
    Bs = np.einsum('m,mdc->dc', E, A); Bm = np.einsum('m,mdc->dc', E, M)
    pk = Bm.argmax(0)
    r['ck2_support_field_rel_l1'] = float((np.abs(Bs - Bm) * WT[:, None]).sum() / (Bm * WT[:, None]).sum())
    r['ck2_support_energy'] = float((Bs * WT[:, None]).sum() / (Bm * WT[:, None]).sum())
    r['ck2_support_peak'] = float(sum(Bs[pk[c], c] for c in range(3)) / sum(Bm[pk[c], c] for c in range(3)))
    return r
for k in ('A1', 'A2', 'A3'):
    out[k] = rows(z[k].astype(np.float64))
out['row_weight_share_by_ring_in_flux_norm'] = [float((M[16 * i:16 * i + 16] * ZZ[16 * i:16 * i + 16, :, None]).sum() / (M * ZZ[..., None]).sum()) for i in range(4)]
import json
json.dump(out, open(os.path.join(H, 'ring_alloc.json'), 'w'), indent=1)
print(json.dumps(out, indent=1))
