"""WL97 audit, phase 2, track J1 (local, numpy only): decisive check for the rank-16 "capacity vs allocation" conflict.

Target: the stored S1 binned material operator M64 (results/ecro_attr95a/ops95a.npz, A4; [in bin, out bin, rgb]).
Family: ANY light-independent linear operator of rank <= 16 per colour in the 64 x 64 bin basis (signed entries allowed, so the
result is a LOWER bound for every non-negative rank-16 operator, whatever its fitting or training procedure).

Two error functionals per colour (squared, relative, flux-weighted over outgoing bins: X[m, d] = M[m, d] z_m z_d as in WL92/WL95):
  f_ck(A) = sum_m p_ck[m] ||X_m - A_m||^2 / sum_m p_ck[m] ||X_m||^2     incidence weights = CK2 panel irradiance share per incident bin
  f_tr(A) = sum_m p_tr[m] ||X_m - A_m||^2 / sum_m p_tr[m] ||X_m||^2     incidence weights = WL92A training panel irradiance share
(p_* from results/audit97/B/learned_s1_albedo.json; rows of X carry z_m, so p/z_m^2-free "per unit irradiance" variants are also given.)

For lam in [0, 1] the row-weighted Eckart-Young optimum of lam f_ck + (1 - lam) f_tr is EXACT:
  phi(lam) = min_{rank<=16} lam f_ck + (1 - lam) f_tr,
so every rank-16 operator satisfies lam f_ck(A) + (1 - lam) f_tr(A) >= phi(lam) for all lam (supporting lines of the attainable set),
and the optimiser itself is an attainable (signed) point.  The script reports the attainable frontier points and, for a required
CK2 error level, the rigorous lower bound on the training-weighted error.

Also: singular spectrum / rank needed, per-ring row-restricted ranks, the same frontier for rank 32 and 48, and the bound arms
A1/A2/A3 placed on the same axes.
"""
import json, os
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
z = np.load(os.path.join(ROOT, 'results', 'ecro_attr95a', 'ops95a.npz'))
M = z['A4'].astype(np.float64)                                   # [64, 64, 3]
cov = json.load(open(os.path.join(ROOT, 'results', 'audit97', 'B', 'learned_s1_albedo.json')))
p_tr = np.array(cov['train_P92A']['panel_irradiance_share_per_incident_bin'], np.float64); p_tr /= p_tr.sum()
p_ck = np.array(cov['CK2']['panel_irradiance_share_per_incident_bin'], np.float64); p_ck /= p_ck.sum()
NB = 64
ZB = ((np.arange(NB) // 16) + 0.5) / 4
ring = np.arange(NB) // 16
sup = np.nonzero(p_ck > 1e-12)[0]


def Xof(A, c, unit):
    """flux matrix; unit=True drops the incident z_m factor (error per unit irradiance in each incident bin)."""
    return A[:, :, c] * ZB[None, :] * (1.0 if unit else ZB[:, None])


def f(A, c, p, unit):
    X, Y = Xof(M, c, unit), Xof(A, c, unit)
    return float((p * ((X - Y) ** 2).sum(1)).sum() / (p * (X ** 2).sum(1)).sum())


def ey(c, w, k, unit):
    """row-weighted Eckart-Young: argmin_{rank<=k} sum_m w_m ||X_m - A_m||^2 ; returns operator in M units."""
    X = Xof(M, c, unit)
    sw = np.sqrt(np.maximum(w, 0))
    nz = sw > 0
    U, s, Vt = np.linalg.svd(sw[nz, None] * X[nz], full_matrices=False)
    Y = np.zeros_like(X)
    Y[nz] = ((U[:, :k] * s[:k]) @ Vt[:k]) / sw[nz, None]
    # rows with zero weight: best completion inside the fitted row space (projection), cannot increase the weighted objective
    P = Vt[:k].T @ Vt[:k]
    Y[~nz] = X[~nz] @ P
    return Y


def frontier(k, unit, lams):
    pts = []
    for lam in lams:
        num = {'ck': 0.0, 'tr': 0.0}; den = {'ck': 0.0, 'tr': 0.0}; phi = []
        for c in range(3):
            X = Xof(M, c, unit)
            nck, ntr = (p_ck * (X ** 2).sum(1)).sum(), (p_tr * (X ** 2).sum(1)).sum()
            w = lam * p_ck / nck + (1 - lam) * p_tr / ntr
            Y = ey(c, w, k, unit)
            eck, etr = (p_ck * ((X - Y) ** 2).sum(1)).sum(), (p_tr * ((X - Y) ** 2).sum(1)).sum()
            num['ck'] += eck; num['tr'] += etr; den['ck'] += nck; den['tr'] += ntr
            phi.append(float(lam * eck / nck + (1 - lam) * etr / ntr))
        pts.append({'lam': float(lam), 'rel_l2_ck2': float(np.sqrt(num['ck'] / den['ck'])), 'rel_l2_train': float(np.sqrt(num['tr'] / den['tr'])),
                    'phi_rgb': phi})
    return pts


def bound_train_given_ck(pts, eps):
    """rigorous per-colour lower bound on f_tr for any rank-k operator with f_ck <= eps^2 (per colour), from the supporting lines."""
    out = []
    for c in range(3):
        best = 0.0
        for p in pts:
            lam = p['lam']
            if lam >= 1.0:
                continue
            best = max(best, (p['phi_rgb'][c] - lam * eps ** 2) / (1 - lam))
        out.append(float(np.sqrt(max(best, 0.0))))
    return out


lams = [0.0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999, 1.0]
res = {'target': 'results/ecro_attr95a/ops95a.npz A4 (M64, S1)', 'ck2_support_bins': [int(v) for v in sup],
       'train_share_on_ck2_bins': float(p_tr[sup].sum()), 'train_share_by_ring': [float(p_tr[ring == r].sum()) for r in range(4)]}

# 1. spectrum / rank needed (uniform rows, flux form), whole matrix and per incident ring
spec = {}
for c in range(3):
    X = Xof(M, c, False)
    s = np.linalg.svd(X, compute_uv=False)
    tail = np.sqrt(np.maximum(np.cumsum((s ** 2)[::-1])[::-1], 0) / (s ** 2).sum())          # tail[k] = rel Fro error of best rank-k
    d = {'rel_fro_rank': {str(k): float(tail[k]) if k < NB else 0.0 for k in (4, 8, 16, 24, 32, 48)},
         'rank_for_10pct': int(np.argmax(tail <= 0.10)), 'rank_for_5pct': int(np.argmax(tail <= 0.05)),
         'numerical_rank_1e-8': int((s > 1e-8 * s[0]).sum())}
    for r in range(4):
        sr = np.linalg.svd(Xof(M, c, True)[ring == r], compute_uv=False)
        tr_ = np.sqrt(np.maximum(np.cumsum((sr ** 2)[::-1])[::-1], 0) / (sr ** 2).sum())
        d[f'ring{r}_rows_rel_fro_rank4_8_12'] = [float(tr_[4]), float(tr_[8]), float(tr_[12])]
    ss = np.linalg.svd(Xof(M, c, True)[sup], compute_uv=False)
    d['ck2_support_rows_numerical_rank'] = int((ss > 1e-8 * ss[0]).sum())
    spec[f'c{c}'] = d
res['spectrum'] = spec

# 2. attainable frontier (signed rank-k, exact) and rigorous bounds
for unit in (False, True):
    tag = 'per_unit_irradiance' if unit else 'flux_form'
    res[tag] = {}
    for k in (16, 32, 48):
        pts = frontier(k, unit, lams)
        res[tag][f'rank{k}'] = {'frontier': [{kk: (v if kk != 'phi_rgb' else [round(x, 6) for x in v]) for kk, v in p.items()} for p in pts],
                                'bound_rel_l2_train_rgb_given_ck2_le': {str(e): bound_train_given_ck(pts, e) for e in (0.05, 0.1, 0.2, 0.3, 0.5)},
                                'bound_rel_l2_ck2_rgb_given_train_le': None}
        # symmetric bound: CK2 error given a training-error budget
        sym = {}
        for e in (0.2, 0.3, 0.4, 0.5):
            out = []
            for c in range(3):
                best = 0.0
                for p in pts:
                    lam = p['lam']
                    if lam <= 0.0:
                        continue
                    best = max(best, (p['phi_rgb'][c] - (1 - lam) * e ** 2) / lam)
                out.append(float(np.sqrt(max(best, 0.0))))
            sym[str(e)] = out
        res[tag][f'rank{k}']['bound_rel_l2_ck2_rgb_given_train_le'] = sym
    # 3. the WL95A bound arms on the same axes
    arms = {}
    for nm in ('A1', 'A2', 'A3'):
        Aop = z[nm].astype(np.float64)
        nck = sum((p_ck * (Xof(M, c, unit) ** 2).sum(1)).sum() for c in range(3)); ntr = sum((p_tr * (Xof(M, c, unit) ** 2).sum(1)).sum() for c in range(3))
        eck = sum((p_ck * ((Xof(M, c, unit) - Xof(Aop, c, unit)) ** 2).sum(1)).sum() for c in range(3))
        etr = sum((p_tr * ((Xof(M, c, unit) - Xof(Aop, c, unit)) ** 2).sum(1)).sum() for c in range(3))
        arms[nm] = {'rel_l2_ck2': float(np.sqrt(eck / nck)), 'rel_l2_train': float(np.sqrt(etr / ntr)),
                    'rank_c0': int(np.linalg.matrix_rank(Aop[:, :, 0], tol=1e-8 * np.abs(Aop).max()))}
    zero = {'rel_l2_ck2': 1.0, 'rel_l2_train': 1.0}
    res[tag]['arms'] = arms; res[tag]['zero_operator'] = zero

out = os.path.join(ROOT, 'results', 'audit97', 'J1', 'j1_rank16_pareto.json')
os.makedirs(os.path.dirname(out), exist_ok=True)
json.dump(res, open(out, 'w', newline='\n'), indent=1)
print('spectrum c1', json.dumps(spec['c1']))
for tag in ('flux_form', 'per_unit_irradiance'):
    for k in (16, 32, 48):
        r = res[tag][f'rank{k}']
        print(tag, 'rank', k)
        print('  frontier (lam, ck2, train):', [(p['lam'], round(p['rel_l2_ck2'], 3), round(p['rel_l2_train'], 3)) for p in r['frontier']])
        print('  train >= given ck2 <=:', {e: [round(x, 3) for x in v] for e, v in r['bound_rel_l2_train_rgb_given_ck2_le'].items()})
        print('  ck2 >= given train <=:', {e: [round(x, 3) for x in v] for e, v in r['bound_rel_l2_ck2_rgb_given_train_le'].items()})
    print(tag, 'arms', json.dumps(res[tag]['arms']))
