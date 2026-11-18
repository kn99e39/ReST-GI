"""WL104B focused tests (run before evidence). Exit 1 on failure.
  <venv-cu128 python> tests104b.py
Covers: grid / solid-angle normalisation, direction conventions (mirror, local frames), angular integration of a known response,
closure runtime equivalence and the declared ER reciprocity on random directions, layered-reference isotropy (convention check),
U-statistic decomposition identity and its estimator, energy additivity and the energy / shape split algebra.
"""
import sys

import numpy as np

import h104b as H
import stats104b as SB

sys.path.insert(0, str(H.HERE.parent / 'ecro_wl84_104a'))
import stats104a as SA  # noqa: E402

fails = []
rng = np.random.default_rng(104301)


def check(name, ok, detail=''):
    print(('PASS ' if ok else 'FAIL ') + name + (' ' + str(detail) if detail else ''), flush=True)
    if not ok:
        fails.append(name)


# 1. solid angles and integration
d, dw, _, _ = H.hemi_grid(1.0, 2.0)
check('hemisphere_solid_angle', abs(dw.sum() - 2 * np.pi) < 1e-9, dw.sum())
lam = 0.73 / np.pi * d[:, 2]                                   # Lambert eval = rho / pi cos
check('lambert_albedo_on_grid', abs((lam * dw).sum() - 0.73) < 2e-4, (lam * dw).sum())
c = H.mirror(H.sph(np.radians(40), 0.3))
pd, pw, rdeg, psi = H.polar_grid(c, 15, 0.25, 96)
check('polar_cap_area', abs(pw.sum() - 2 * np.pi * (1 - np.cos(np.radians(15)))) < 1e-12)
check('polar_dirs_within_cap', np.degrees(np.arccos(np.clip(pd @ (c / np.linalg.norm(c)), -1, 1))).max() <= 15 + 1e-9)
wi = H.sph(np.radians(40), 0.3); m = H.mirror(wi)
check('mirror_convention', abs(m @ np.array([0, 0, 1.0]) - wi[2]) < 1e-12 and np.allclose(m[:2], -wi[:2]))

# 2. closure runtime equivalence and declared reciprocity on random upper-hemisphere directions
u = rng.random((20000, 4))
W1 = H.sph(np.arccos(u[:, 0]), 2 * np.pi * u[:, 1]); W2 = H.sph(np.arccos(u[:, 2]), 2 * np.pi * u[:, 3])
mat = H.material('test83_hard_1')
for arm in ('ER_L', 'ER_DF'):
    t32 = H.er_eval_torch(mat[arm]['p32'], W1.astype(np.float32), W2.astype(np.float32))
    drv = H.er_eval_dr(mat[arm]['state'], W1, W2)
    check(f'runtime_equivalence_{arm}', np.abs(t32 - drv).max() / np.abs(t32).max() <= 1e-5, np.abs(t32 - drv).max() / np.abs(t32).max())
    f12 = H.er_eval_torch(mat[arm]['p'], W1, W2) / W2[:, 2:3]; f21 = H.er_eval_torch(mat[arm]['p'], W2, W1) / W1[:, 2:3]
    ok = (W1[:, 2] > 0.05) & (W2[:, 2] > 0.05)
    check(f'er_reciprocity_{arm}', np.abs(f12 - f21)[ok].max() / np.abs(f12[ok]).max() <= 1e-4)

# 3. layered reference: isotropy under a common azimuthal rotation (convention check, within walk noise)
n = 4000
wi = H.sph(np.radians(55) * np.ones(n), np.zeros(n)); wo = H.sph(np.arccos(rng.uniform(0.2, 1, n)), rng.uniform(0, 2 * np.pi, n))
rot = lambda v, a: np.stack([np.cos(a) * v[:, 0] - np.sin(a) * v[:, 1], np.sin(a) * v[:, 0] + np.cos(a) * v[:, 1], v[:, 2]], 1)
a1, b1 = H.lay_eval(mat['desc'], wi, wo, 512, 11)
a2, b2 = H.lay_eval(mat['desc'], rot(wi, 1.1), rot(wo, 1.1), 512, 12)
r1, r2 = H.lum(0.5 * (a1 + b1)).sum(), H.lum(0.5 * (a2 + b2)).sum()
se = np.sqrt(((H.lum(a1 - b1) / 2) ** 2).sum() + ((H.lum(a2 - b2) / 2) ** 2).sum())
check('layered_isotropy', abs(r1 - r2) <= 4 * se, (r1, r2, se))

# 4. U-statistic: decomposition identity, agreement with stats104a, unbiasedness
R, M = 24, 5000
mu = rng.normal(0, 0.05, (3, M))
X = mu[:, None, :] + rng.normal(0, 0.3, (3, R, M))
tot = X.sum(0)
pt_t, _ = SB.ms_loo(tot, tot)
parts = sum((1 if i == j else 2) * SB.ms_loo(X[i], X[j])[0] for i in range(3) for j in range(i, 3))
check('ms_decomposition_identity', abs(pt_t - parts) <= 1e-12 * max(abs(pt_t), 1), pt_t - parts)
check('ms_matches_stats104a', abs(pt_t - SA.stable_ms_jk(tot)['point']) < 1e-12 and abs(SB.jk_ci(*SB.ms_loo(tot, tot))['se'] - SA.stable_ms_jk(tot)['se']) < 1e-12)
check('ms_unbiased', abs(pt_t - (mu.sum(0) ** 2).mean()) < 0.003, (pt_t, (mu.sum(0) ** 2).mean()))

# 5. energy additivity and energy / shape split algebra
e = rng.normal(size=(R, 3))
check('energy_additivity', abs(SA.ratio_jk(e.sum(1), np.full(R, 10.0))['point'] - sum(SA.ratio_jk(e[:, j], np.full(R, 10.0))['point'] for j in range(3))) < 1e-12)
pl, px, ra = rng.random(100), rng.random(100), rng.uniform(0.9, 1.1, 100)
en = (ra - 1) * pl; sh = (px - pl) - en
check('energy_shape_split_sums', np.allclose(en + sh, px - pl))
print('FAILED ' + ', '.join(fails) if fails else 'ALL TESTS PASSED', flush=True)
import os
os._exit(1 if fails else 0)
