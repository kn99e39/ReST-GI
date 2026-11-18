"""WL101H estimator tests on synthetic data with known statistical relationships (run before any evidence).
  python tests_h.py [gbuffer_npz coverage_npz]   -> writes nothing; exits non-zero on failure.
"""
import sys

import numpy as np
import torch

import stats_h as S

torch.manual_seed(0)
rng = np.random.default_rng(101800)
fails = []


def check(name, ok, detail=''):
    print(('PASS ' if ok else 'FAIL ') + name + (' ' + detail if detail else ''), flush=True)
    if not ok:
        fails.append(name)


def T(x):
    return torch.as_tensor(x, dtype=torch.float64)


# 1. gram with unit weights == numpy ddof=1 variance / covariance averaged over elements
R, M = 16, 300
X = rng.normal(size=(R, M)); Y = 0.3 * X + rng.normal(size=(R, M))
G = S.gram([T(X), T(Y)]).numpy()
vx = np.var(X, 0, ddof=1).mean(); vy = np.var(Y, 0, ddof=1).mean()
cxy = (((X - X.mean(0)) * (Y - Y.mean(0))).sum(0) / (R - 1)).mean()
check('gram_equals_numpy', np.allclose(G, [[vx, cxy], [cxy, vy]], rtol=1e-12, atol=1e-14))
Wu = torch.ones(3, R, dtype=torch.float64)
check('gram_unit_weights', np.allclose(S.gram([T(X), T(Y)], Wu).numpy(), G[None], rtol=1e-12))

# 2. known moments: L = Z1 + 0.5 Z2, H = 0.8 Z2 + 0.3 Z3 -> Var L 1.25, Var H 0.73, Cov 0.4 ; identity exact
R, M = 64, 200000
Z = rng.normal(size=(3, R, M))
L = Z[0] + 0.5 * Z[1]; H = 0.8 * Z[1] + 0.3 * Z[2]; Tt = L + H
G = S.gram([T(L), T(H), T(Tt)]).numpy()
check('known_var_L', abs(G[0, 0] - 1.25) < 0.01, f'{G[0, 0]:.4f}')
check('known_var_H', abs(G[1, 1] - 0.73) < 0.01, f'{G[1, 1]:.4f}')
check('known_cov_LH', abs(G[0, 1] - 0.40) < 0.01, f'{G[0, 1]:.4f}')
ident = G[2, 2] - (G[0, 0] + G[1, 1] + 2 * G[0, 1])
check('variance_identity_exact', abs(ident) < 1e-12 * G[2, 2], f'{ident:.2e}')
# negative covariance retained with its sign
Hn = -0.8 * Z[1] + 0.3 * Z[2]
check('signed_negative_cov', abs(S.gram([T(L), T(Hn)]).numpy()[0, 1] + 0.40) < 0.01)

# 3. pick-freeze stages: X = aJ + bP + cJP + dS + ePS ; V_J = a^2, V_P = b^2 + c^2, V_S = d^2 + e^2 ; H has no S terms
R, M = 64, 100000
J = rng.normal(size=(R, M)); P1 = rng.normal(size=(R, M)); P2 = rng.normal(size=(R, M))
S1, S2, S3 = (rng.normal(size=(R, M)) for _ in range(3))
coefL = (0.7, 0.5, 0.4, 0.9, 0.3); coefH = (0.6, 0.8, 0.2, 0.0, 0.0)


def f(cf, j, p, s):
    a, b, c, d, e = cf
    return a * j + b * p + c * j * p + d * s + e * p * s


A = [T(f(coefL, J, P1, S1)), T(f(coefH, J, P1, S1))]
Bv = [T(f(coefL, J, P1, S2)), T(f(coefH, J, P1, S2))]
C = [T(f(coefL, J, P2, S3)), T(f(coefH, J, P2, S3))]
D = S.pf_decompose(A, Bv, C)
truth = {'J': np.array([[0.49, 0.42], [0.42, 0.36]]), 'P': np.array([[0.25 + 0.16, 0.4 + 0.08], [0.48, 0.64 + 0.04]]),
         'S': np.array([[0.81 + 0.09, 0.0], [0.0, 0.0]])}
for k, tv in truth.items():
    est = D[k].numpy()
    check(f'pick_freeze_{k}', np.allclose(est, tv, atol=0.02), np.array2string(est, precision=3))
check('pick_freeze_high_suffix_exact_zero', float(D['S'][1, 1]) == 0.0 and float(D['S'][0, 1]) == 0.0)
check('pick_freeze_sum', np.allclose((D['J'] + D['P'] + D['S']).numpy(), D['tot'].numpy(), rtol=1e-12))

# 4. stable squared bias cross product: unbiased for mean b^2 despite large independent noise
M = 400000
mu = rng.uniform(0, 2, size=(2, M)); b = rng.normal(0, 0.05, size=(2, M))
noise = 0.2
g1 = mu + b + rng.normal(0, noise, (2, M)); g2 = mu + b + rng.normal(0, noise, (2, M))
r1 = mu + rng.normal(0, noise, (2, M)); r2 = mu + rng.normal(0, noise, (2, M))
bs = S.stable_bias_sq(T(g1), T(g2), T(r1), T(r2)).numpy()
check('stable_bias_sq', abs(bs[0, 0] - (b[0] ** 2).mean()) < 3e-4, f'{bs[0, 0]:.5f} vs {(b[0] ** 2).mean():.5f}')
check('stable_bias_cross_near_zero', abs(bs[0, 1] - (b[0] * b[1]).mean()) < 3e-4)
rn = S.ref_mean_noise(T(r1), T(r2)).numpy()
check('ref_mean_noise', abs(rn[0] - noise ** 2 / 2) < 2e-4, f'{rn[0]:.5f}')

# 5. bootstrap: unit weights reproduce the point estimate; percentile CI coverage of the ROI variance
for (Rb, Mb) in ((64, 40), (64, 2000)):
    cover = cover_share = 0
    nexp = 300
    for e in range(nexp):
        Z3 = rng.normal(size=(2, Rb, Mb))
        Lb, Hb = T(1.5 * Z3[0]), T(0.6 * Z3[0] + 0.8 * Z3[1])                 # Var L 2.25, Var H 1.0, Cov 0.9, Var T 5.05
        Wb = S.boot_weights(Rb, 400, 1000 + e, 'cpu')
        gd = S.gram([Lb, Hb], Wb).numpy(); gp = S.gram([Lb, Hb]).numpy()
        c = S.ci(gd[:, 0, 0], gp[0, 0])
        cover += c['lo'] <= 2.25 <= c['hi']
        share = lambda g: 2 * g[..., 0, 1] / (g[..., 0, 0] + g[..., 1, 1] + 2 * g[..., 0, 1])
        c = S.ci(share(gd), share(gp))
        cover_share += c['lo'] <= 1.8 / 5.05 <= c['hi']
    check(f'bootstrap_coverage_var_R{Rb}_M{Mb}', 0.90 <= cover / nexp <= 0.99, f'{cover / nexp:.3f}')
    check(f'bootstrap_coverage_cov_share_R{Rb}_M{Mb}', 0.90 <= cover_share / nexp <= 0.99, f'{cover_share / nexp:.3f}')
check('z_quantile', abs(S.z_two_sided(0.05) - 1.959964) < 1e-5)
check('log_slope', abs(S.log_slope([1, 4, 16], [8, 2, 0.5]) + 1) < 1e-12)

# 6. optional: frozen mask counts from the gbuffer and coverage files
if len(sys.argv) > 2:
    import common_h as CH
    cov = dict(np.load(sys.argv[2]))
    m = CH.all_masks(sys.argv[1], cov)
    exp = CH.P['rois']['expected_counts']
    got = {k: int(m[k].sum()) for k in exp}
    check('mask_counts', got == exp, str(got))
    check('mask_disjoint', not np.any(m['near'] & m['flank']) and not np.any(m['control'] & m['contact_shadow'])
          and not np.any(m['sss_interior'] & m['silhouette']) and not np.any(m['control'] & m['silhouette']))

print('FAILED: ' + ', '.join(fails) if fails else 'ALL TESTS PASSED')
sys.exit(1 if fails else 0)
