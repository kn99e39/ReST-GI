"""WL97 audit, phase 2, track J1 (local, numpy only): constructive NON-NEGATIVE rank-16 check.

j1_rank16_pareto.py gives the exact attainable frontier for SIGNED rank-16 operators (a lower bound for non-negative ones).
This script gives an explicit admissible (non-negative, rank <= 16 per colour) operator that is exact on the CK2 incident
support and asks what that costs on the training incidence distribution:

  H(k_ded) = [rows of M for the k_ded dedicated incident bins, one component each]  +  NMF_{16-k_ded}(remaining rows, training weights)

  k_ded = 0   -> plain training-weighted NMF16 (same recipe as phase-1 track B 'train' fit; reproduces its numbers)
  k_ded = 4   -> exact on the CK2 support (bins 12-15), 12 components for the other 60 incident bins
  k_ded = 16  -> exact on all of ring 0, nothing left (the phase-1 F0/C2 'ring-0 exact' counterexample)

Metrics are those of experiments/audit97/B/rank16_allocation.py (so the rows are directly comparable with B-06) plus rel L2.
NMF: Lee-Seung multiplicative updates, row-weighted Frobenius on the flux matrix X = M z_m z_d, 3 restarts, 4,000 iterations.
"""
import json, os
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
z = np.load(os.path.join(ROOT, 'results', 'ecro_attr95a', 'ops95a.npz'))
M = z['A4'].astype(np.float64)
cov = json.load(open(os.path.join(ROOT, 'results', 'audit97', 'B', 'learned_s1_albedo.json')))
p_tr = np.array(cov['train_P92A']['panel_irradiance_share_per_incident_bin'], np.float64); p_tr /= p_tr.sum()
p_ck = np.array(cov['CK2']['panel_irradiance_share_per_incident_bin'], np.float64); p_ck /= p_ck.sum()
NB = 64
ZB = ((np.arange(NB) // 16) + 0.5) / 4
WT = ZB * 2 * np.pi / NB
ZZ = ZB[:, None] * ZB[None, :]
sup = np.nonzero(p_ck > 1e-12)[0]


def nmf(X, w, k, iters=4000, seed=0):
    sw = np.sqrt(np.maximum(w, 1e-12))[:, None]
    Y = X * sw
    rng = np.random.default_rng(seed)
    C = rng.random((X.shape[0], k)) * np.sqrt(Y.mean() / k) + 1e-9
    A = rng.random((X.shape[1], k)) * np.sqrt(Y.mean() / k) + 1e-9
    for _ in range(iters):
        C *= (Y @ A) / (C @ (A.T @ A) + 1e-30)
        A *= (Y.T @ C) / (A @ (C.T @ C) + 1e-30)
    return (C / sw) @ A.T, float(((Y - C @ A.T) ** 2).sum() / (Y ** 2).sum())


def hybrid(ded, k=16):
    """non-negative operator: dedicated exact rows `ded` + NMF_{k-len(ded)} of the remaining rows under training weights."""
    ded = np.asarray(ded, int); rest = np.setdiff1d(np.arange(NB), ded)
    A_ = np.zeros_like(M)
    for c in range(3):
        X = M[..., c] * ZZ
        Xh = np.zeros_like(X)
        Xh[ded] = X[ded]
        kk = k - len(ded)
        if kk > 0:
            best = None
            for sd in range(3):
                Y, o = nmf(X[rest], p_tr[rest], kk, seed=97100 + 10 * c + sd)
                if best is None or o < best[1]:
                    best = (Y, o)
            Xh[rest] = best[0]
        A_[..., c] = Xh / ZZ
    return A_


def metrics(A):
    num = (np.abs(A - M) * ZZ[..., None]).sum((1, 2)); den = (M * ZZ[..., None]).sum((1, 2))
    r = {f'ring{i}_op_l1': float(num[16 * i:16 * i + 16].sum() / den[16 * i:16 * i + 16].sum()) for i in range(4)}
    r['op_flux_l1_all'] = float(num.sum() / den.sum())

    def field(E):
        Bs = np.einsum('m,mdc->dc', E, A); Bm = np.einsum('m,mdc->dc', E, M); pk = Bm.argmax(0)
        return {'field_rel_l1': float((np.abs(Bs - Bm) * WT[:, None]).sum() / (Bm * WT[:, None]).sum()),
                'energy': float((Bs * WT[:, None]).sum() / (Bm * WT[:, None]).sum()),
                'peak': float(sum(Bs[pk[c], c] for c in range(3)) / sum(Bm[pk[c], c] for c in range(3)))}
    r['ck2'] = field(p_ck)
    row_l1 = (np.abs(A - M) * WT[None, :, None]).sum((1, 2)); row_m = (M * WT[None, :, None]).sum((1, 2))
    r['train_weighted_row_rel_l1'] = float((p_tr * row_l1).sum() / (p_tr * row_m).sum())
    r['not_ck2_train_weighted_row_rel_l1'] = float((np.delete(p_tr * row_l1, sup)).sum() / (np.delete(p_tr * row_m, sup)).sum())
    D2 = (((A - M) * ZB[None, :, None]) ** 2).sum((1, 2)); N2 = ((M * ZB[None, :, None]) ** 2).sum((1, 2))
    r['train_weighted_rel_l2_per_unit_irradiance'] = float(np.sqrt((p_tr * D2).sum() / (p_tr * N2).sum()))
    r['ck2_weighted_rel_l2_per_unit_irradiance'] = float(np.sqrt((p_ck * D2).sum() / (p_ck * N2).sum()))
    alb = (A * WT[None, :, None]).sum(1)
    r['albedo_max'] = float(alb.max()); r['min_entry'] = float(A.min())
    r['rank_c1'] = int(np.linalg.matrix_rank(A[..., 1] * ZZ, tol=1e-9 * (M[..., 1] * ZZ).max()))
    return r


res = {'target': 'results/ecro_attr95a/ops95a.npz A4 (M64, S1)', 'ck2_support_bins': [int(v) for v in sup],
       'train_share_on_ck2_bins': float(p_tr[sup].sum()), 'arms': {}}
cases = {'nmf16_train (k_ded=0)': [], 'ck2_exact + nmf12_train (k_ded=4)': list(sup), 'ring0_exact (k_ded=16)': list(range(16)),
         'nmf12_train_only (rank 12, reference)': None}
for nm, ded in cases.items():
    if ded is None:
        A_ = hybrid([], k=12)
    else:
        A_ = hybrid(ded)
    res['arms'][nm] = metrics(A_)
for nm in ('A1', 'A2', 'A3'):
    res['arms'][f'WL95A_{nm}'] = metrics(z[nm].astype(np.float64))
res['arms']['zero'] = metrics(np.zeros_like(M))
out = os.path.join(ROOT, 'results', 'audit97', 'J1', 'j1_rank16_nonneg_construct.json')
json.dump(res, open(out, 'w', newline='\n'), indent=1)
for nm, r in res['arms'].items():
    print(f"{nm:40s} ck2 L1/en/pk {r['ck2']['field_rel_l1']:.3f}/{r['ck2']['energy']:.3f}/{r['ck2']['peak']:.3f}  train rowL1 {r['train_weighted_row_rel_l1']:.3f} "
          f"(not-ck2 {r['not_ck2_train_weighted_row_rel_l1']:.3f})  train L2 {r['train_weighted_rel_l2_per_unit_irradiance']:.3f}  opL1 {r['op_flux_l1_all']:.3f} "
          f"rings {[round(r[f'ring{i}_op_l1'], 2) for i in range(4)]}  alb {r['albedo_max']:.3f} min {r['min_entry']:.1e} rank {r['rank_c1']}")
