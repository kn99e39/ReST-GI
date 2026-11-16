"""WL97 audit track B: is 'rank-16 cannot represent the CK2 S1 lobe' a capacity statement or a fit-distribution (coverage) statement?

  python rank16_allocation.py <ops95a.npz> <learned_s1_albedo.json> <out_json>

Independent numpy calculation. Target: the WL95A A4 operator (M64 restricted to the S1 panel material; [in 64, out 64, 3]) read from
results/ecro_attr95a/ops95a.npz. For each incident-row weighting w (training panel-irradiance share per incident bin, measured by this
audit from the 928 x 4 stored WL92A training source encodings; the CK2 share; mixtures) a per-channel NON-NEGATIVE rank-16 factorisation
of the flux matrix X[m, d] = M[m, d] z_m z_d is fitted by row-weighted multiplicative updates (Lee-Seung, Frobenius), 3 restarts.
Reported per fit: CK2 field metrics (the WL95/96 definitions: field rel L1 / energy / peak of sum_m E_m A[m, d] with W_T), per-ring
operator flux L1, training-weighted field proxy and max directional albedo. Also the trivially exact rank-4 CK2-support operator.
Nothing here is a trained ECRO model; it bounds what a light-independent non-negative rank-16 S1 operator can do under each allocation.
"""
import json, sys
import numpy as np

z = np.load(sys.argv[1]); cov = json.load(open(sys.argv[2])); outfn = sys.argv[3]
M = z['A4'].astype(np.float64)
NB = 64
ZB = ((np.arange(NB) // 16) + 0.5) / 4
WT = ZB * 2 * np.pi / NB
ZZ = ZB[:, None] * ZB[None, :]
p_tr = np.array(cov['train_P92A']['panel_irradiance_share_per_incident_bin'], np.float64); p_tr /= p_tr.sum()
p_ck = np.array(cov['CK2']['panel_irradiance_share_per_incident_bin'], np.float64); p_ck /= p_ck.sum()
sup = np.nonzero(p_ck)[0]


def nmf(X, w, k=16, iters=4000, seed=0):
    sw = np.sqrt(np.maximum(w, 1e-12))[:, None]
    Y = X * sw
    rng = np.random.default_rng(seed)
    C = rng.random((X.shape[0], k)) * np.sqrt(Y.mean() / k) + 1e-9
    A = rng.random((X.shape[1], k)) * np.sqrt(Y.mean() / k) + 1e-9
    for _ in range(iters):
        C *= (Y @ A) / (C @ (A.T @ A) + 1e-30)
        A *= (Y.T @ C) / (A @ (C.T @ C) + 1e-30)
    return (C / sw) @ A.T, float(((Y - C @ A.T) ** 2).sum() / (Y ** 2).sum())


def fit(w, k=16):
    A_ = np.zeros_like(M); obj = []
    for c in range(3):
        X = M[..., c] * ZZ
        best = None
        for sd in range(3):
            Xh, o = nmf(X, w, k, seed=97000 + 10 * c + sd)
            if best is None or o < best[1]:
                best = (Xh, o)
        A_[..., c] = best[0] / ZZ; obj.append(best[1])
    return A_, obj


def metrics(A):
    num = (np.abs(A - M) * ZZ[..., None]).sum((1, 2)); den = (M * ZZ[..., None]).sum((1, 2))
    r = {f'ring{i}_op_l1': float(num[16 * i:16 * i + 16].sum() / den[16 * i:16 * i + 16].sum()) for i in range(4)}

    def field(E):
        Bs = np.einsum('m,mdc->dc', E, A); Bm = np.einsum('m,mdc->dc', E, M); pk = Bm.argmax(0)
        return {'field_rel_l1': float((np.abs(Bs - Bm) * WT[:, None]).sum() / (Bm * WT[:, None]).sum()),
                'energy': float((Bs * WT[:, None]).sum() / (Bm * WT[:, None]).sum()),
                'peak': float(sum(Bs[pk[c], c] for c in range(3)) / sum(Bm[pk[c], c] for c in range(3)))}
    r['ck2'] = field(p_ck)
    # training proxy: expected single-bin field error under the training incidence distribution (rows weighted by p_tr)
    row_l1 = (np.abs(A - M) * WT[None, :, None]).sum((1, 2)); row_m = (M * WT[None, :, None]).sum((1, 2))
    r['train_weighted_row_rel_l1'] = float((p_tr * row_l1).sum() / (p_tr * row_m).sum())
    r['not_ck2_train_weighted_row_rel_l1'] = float((np.delete(p_tr * row_l1, sup)).sum() / (np.delete(p_tr * row_m, sup)).sum())
    alb = (A * WT[None, :, None]).sum(1)
    r['albedo_max'] = float(alb.max()); r['min_entry'] = float(A.min())
    return r


out = {'target': 'ops95a.npz A4 (M64, S1)', 'M_albedo_max': float((M * WT[None, :, None]).sum(1).max()),
       'train_share_ring': [float(p_tr[16 * i:16 * i + 16].sum()) for i in range(4)], 'train_share_ck2_bins': float(p_tr[sup].sum()),
       'ck2_bins': sup.tolist(), 'reference_arms': {k: metrics(z[k].astype(np.float64)) for k in ('A1', 'A2', 'A3')}, 'fits': {}}
weights = {'train': p_tr, 'uniform_bins': np.full(NB, 1 / NB), 'ck2_only': p_ck + 1e-9}
for lam in (0.02, 0.05, 0.10, 0.25, 0.50):
    weights[f'mix_ck2_{lam:.2f}'] = (1 - lam) * p_tr + lam * p_ck
for nm, w in weights.items():
    A_, obj = fit(w)
    out['fits'][nm] = {'ck2_weight_share': float(w[sup].sum() / w.sum()), 'ring0_weight_share': float(w[:16].sum() / w.sum()), 'objective': obj, **metrics(A_)}
    print(nm, json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in out['fits'][nm].items() if k != 'objective'}), flush=True)
# trivially exact rank-4 operator on the CK2 support
A4x = np.zeros_like(M); A4x[sup] = M[sup]
out['rank4_ck2_support_exact'] = metrics(A4x)
json.dump(out, open(outfn, 'w'), indent=1)
print('A1', json.dumps(out['reference_arms']['A1'])); print('A2', json.dumps(out['reference_arms']['A2']))
