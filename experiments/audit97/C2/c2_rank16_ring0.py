"""WL97 audit C2: rank-16 ring-0 counterexample (WL96 results/wl96_calc/ring_alloc.py) re-derived independently, and what it does / does not imply.
numpy only.  scene86.Geom is imported ONLY for the patch geometry (PO / PE1 / PE2 / frames); all algebra is local.
Run from the repository root."""
import json, math, os, sys
import numpy as np

sys.path.insert(0, 'experiments/ecro86')
import scene86 as S  # noqa: E402  (geometry only)

z = np.load('results/ecro_attr95a/ops95a.npz')
M = z['A4'].astype(np.float64)                      # [in, out, 3]
NB = 64
ZB = ((np.arange(NB) // 16) + 0.5) / 4
WT = ZB * 2 * math.pi / NB
ZZ = ZB[:, None] * ZB[None, :]
out = {}

# ---- 1. CK2 incident support on the S1 panel: dense sampling of the whole panel (not only the 16 deployed source points), unshadowed
sc = json.load(open('experiments/ecro_local93d/addendum_checker/scene_ck2.json'))
G = S.Geom(sc['geom'])
S1 = np.arange(1088, 1130)
L = np.array(sc['lights'][0]['pos'], float)
rg = np.random.default_rng(9702)
E = np.zeros(NB)
zmin, zmax = 1e9, -1e9
for p in S1:
    u = rg.random((4096, 2))
    x = G.PO[p][None] + u[:, :1] * G.PE1[p][None] + u[:, 1:] * G.PE2[p][None]
    d = L[None] - x
    dist = np.linalg.norm(d, axis=1)
    w = d / dist[:, None]
    zc = w @ G.PN[p]
    ok = zc > 1e-6
    iz = np.minimum((zc * 4).astype(int), 3)
    ip = np.minimum(((np.arctan2(w @ G.PB[p], w @ G.PT[p]) + math.pi) * 16 / (2 * math.pi)).astype(int), 15)
    b = iz * 16 + ip
    np.add.at(E, b[ok], (zc / dist ** 2)[ok])
    zmin, zmax = min(zmin, zc.min()), max(zmax, zc.max())
E_audit = E / E.sum()
E_orig = np.load('results/wl96_calc/ck2_support_E.npy')
out['support'] = {'bins_dense': np.nonzero(E)[0].tolist(), 'bins_orig_16pts': np.nonzero(E_orig)[0].tolist(), 'cos_incidence_min_max': [float(zmin), float(zmax)],
                  'share_dense': {int(b): float(E_audit[b]) for b in np.nonzero(E)[0]}, 'share_orig': {int(b): float(E_orig[b] / E_orig.sum()) for b in np.nonzero(E_orig)[0]},
                  'ring0_upper_cos': 0.25, 'incidence_deg_from_normal_min_max': [float(np.degrees(np.arccos(zmax))), float(np.degrees(np.arccos(zmin)))]}

# ---- 2. rank facts
sv = [np.linalg.svd(M[..., c], compute_uv=False) for c in range(3)]
out['rank_M_full'] = [int(np.linalg.matrix_rank(M[..., c])) for c in range(3)]
out['rank_M_ring0_rows'] = [int(np.linalg.matrix_rank(M[:16, :, c])) for c in range(3)]
sup = np.nonzero(E)[0]
out['rank_M_support_rows'] = [int(np.linalg.matrix_rank(M[sup][:, :, c])) for c in range(3)]
out['sv_energy_rank16_full'] = [float((s[:16] ** 2).sum() / (s ** 2).sum()) for s in sv]
out['min_nonneg_M'] = float(M.min())

# ---- 3. the counterexample: c = selector on ring-0 inputs, a = M rows -> exact on ring-0-supported s, zero response elsewhere
C = np.zeros((NB, 16)); C[np.arange(16), np.arange(16)] = 1.0
A_r0 = np.stack([C @ M[:16, :, c] for c in range(3)], -1)        # [in, out, 3]
rg = np.random.default_rng(9703)
s = np.zeros((1000, NB, 3)); s[:, :16] = rg.random((1000, 16, 3))
ref = np.einsum('kmc,mdc->kdc', s, M)
out['counterexample'] = {'max_rel_err_ring0_inputs': float(np.abs(np.einsum('kmc,mdc->kdc', s, A_r0) - ref).max() / ref.max()), 'nonneg': bool(A_r0.min() >= 0),
                         'rank': [int(np.linalg.matrix_rank(A_r0[..., c])) for c in range(3)],
                         'albedo_max': float((A_r0 * WT[None, :, None]).sum(1).max()),
                         # the SAME operator on inputs supported on the other rings: it returns zero (relative error 1)
                         'rel_err_on_ring_r_inputs': {}}
for r in range(4):
    s2 = np.zeros((200, NB, 3)); s2[:, 16 * r:16 * r + 16] = rg.random((200, 16, 3))
    y = np.einsum('kmc,mdc->kdc', s2, M); y2 = np.einsum('kmc,mdc->kdc', s2, A_r0)
    out['counterexample']['rel_err_on_ring_r_inputs'][f'ring{r}'] = float((np.abs(y2 - y) * WT[None, :, None]).sum() / (y * WT[None, :, None]).sum())
# operator-level flux L1 of the counterexample against M over ALL rows (the WL95A op metric): what the exact ring-0 copy costs globally
out['counterexample']['op_flux_l1_all_rows'] = float((np.abs(A_r0 - M) * ZZ[..., None]).sum() / (M * ZZ[..., None]).sum())


def cfield(A, Ew):
    Bs = np.einsum('m,mdc->dc', Ew, A); Bm = np.einsum('m,mdc->dc', Ew, M)
    pk = Bm.argmax(0)
    return {'rel_l1': float((np.abs(Bs - Bm) * WT[:, None]).sum() / (Bm * WT[:, None]).sum()), 'energy': float((Bs * WT[:, None]).sum() / (Bm * WT[:, None]).sum()),
            'peak': float(sum(Bs[pk[c], c] for c in range(3)) / sum(Bm[pk[c], c] for c in range(3)))}


# ---- 4. the WL95A bound operators on the CK2 support (recomputing ring_alloc.json with my own code and the dense support)
out['bound_ops'] = {}
for k in ('A1', 'A2', 'A3'):
    A = z[k].astype(np.float64)
    num = (np.abs(A - M) * ZZ[..., None]).sum((1, 2)); den = (M * ZZ[..., None]).sum((1, 2))
    out['bound_ops'][k] = {'ring_op_l1': [float(num[16 * i:16 * i + 16].sum() / den[16 * i:16 * i + 16].sum()) for i in range(4)],
                           'ck2_orig_support': cfield(A, E_orig), 'ck2_dense_support': cfield(A, E),
                           'rank': [int(np.linalg.matrix_rank(A[..., c], tol=1e-9 * np.linalg.norm(A[..., c]))) for c in range(3)],
                           'op_flux_l1_all': float(num.sum() / den.sum())}
out['row_weight_share_by_ring'] = [float((M[16 * i:16 * i + 16] * ZZ[16 * i:16 * i + 16, :, None]).sum() / (M * ZZ[..., None]).sum()) for i in range(4)]

# ---- 5. is there ONE light-independent rank-16 operator that is good on every ring?  Signed truncated SVD under different row weightings
#         (signed SVD is a lower bound on the weighted-Frobenius error of any rank-16 operator, non-negative or not, in that norm).
def wsvd16(rowscale, k=16):
    A = np.zeros_like(M)
    for c in range(3):
        X = rowscale[:, None] * M[..., c] * np.sqrt(ZB)[None, :]
        U, s_, Vt = np.linalg.svd(X, full_matrices=False)
        A[..., c] = ((U[:, :k] * s_[:k]) @ Vt[:k]) / rowscale[:, None] / np.sqrt(ZB)[None, :]
    return A


def ring_err(A):
    num = (np.abs(A - M) * ZZ[..., None]).sum((1, 2)); den = (M * ZZ[..., None]).sum((1, 2))
    return [float(num[16 * i:16 * i + 16].sum() / den[16 * i:16 * i + 16].sum()) for i in range(4)]


rowflux = (M * WT[None, :, None]).sum((1, 2))             # outgoing flux per unit irradiance in bin m (sum of channels)
out['svd16'] = {}
for nm, rs_ in (('flux_norm_rows_zb', np.sqrt(ZB)), ('equal_rows', np.ones(NB)), ('unit_out_flux_rows', 1.0 / rowflux),
                ('ring0_x10', np.where(np.arange(NB) < 16, 10.0, 1.0)), ('ring0_x100', np.where(np.arange(NB) < 16, 100.0, 1.0))):
    A = wsvd16(rs_)
    out['svd16'][nm] = {'ring_op_l1': ring_err(A), 'ck2_dense_support': cfield(A, E), 'min_entry': float(A.min())}
for k in (16, 24, 32, 48):
    A = wsvd16(1.0 / rowflux, k)
    out['svd16'][f'unit_out_flux_rows_rank{k}'] = {'ring_op_l1': ring_err(A), 'ck2_dense_support': cfield(A, E)}
# per-ring-0-row numerical rank spectrum: how many singular values the 16 ring-0 rows need on their own
for c in range(3):
    s0 = np.linalg.svd(M[:16, :, c] * np.sqrt(ZB)[None], compute_uv=False)
    out[f'ring0_rows_sv_energy_c{c}'] = [float((s0[:k] ** 2).sum() / (s0 ** 2).sum()) for k in (1, 2, 4, 8, 12, 16)]
json.dump(out, open('results/audit97/C2/rank16_ring0.json', 'w'), indent=1)
print(json.dumps(out, indent=1))
orig = json.load(open('results/wl96_calc/ring_alloc.json'))
print('ORIG ring_alloc A1', orig['A1'], '\nORIG shares', orig['row_weight_share_by_ring_in_flux_norm'])
