"""WL104A focused tests (synthetic estimator tests + ROI / metric arithmetic). Exit 1 on failure.
  python tests104a.py
"""
import sys

import numpy as np

import stats104a as S

rng = np.random.default_rng(104001)
fails = []


def check(name, ok, detail=''):
    print(('PASS ' if ok else 'FAIL ') + name + (' ' + str(detail) if detail else ''), flush=True)
    if not ok:
        fails.append(name)


# 1. stable mean square: unbiased for mean(mu^2) despite noise much larger than the signal, paired and A/B forms
M, R = 20000, 32
mu = rng.normal(0, 0.05, M)
cover = 0
for e in range(200):
    D = mu + rng.normal(0, 0.5, (R, M))
    s = S.stable_ms_jk(D)
    cover += s['lo'] <= (mu ** 2).mean() <= s['hi']
check('stable_ms_coverage_paired (jackknife of a U-statistic: conservative allowed)', 0.90 <= cover / 200 <= 1.0, cover / 200)
D = mu + rng.normal(0, 0.5, (R, M))
check('stable_ms_unbiased_vs_naive', abs(S.stable_ms_jk(D)['point'] - (mu ** 2).mean()) < 0.0015 and (D.mean(0) ** 2).mean() > 2 * (mu ** 2).mean(),
      (S.stable_ms_jk(D)['point'], (mu ** 2).mean()))
# A/B: A = mu + table noise tA + path noise, B = mu + independent table noise tB + path noise: the AB form removes the table bias
tA, tB = rng.normal(0, 0.1, M), rng.normal(0, 0.1, M)
A = mu + tA + rng.normal(0, 0.3, (R, M)); Bm = mu + tB + rng.normal(0, 0.3, (R, M))
check('stable_ms_AB_removes_shared_table_noise', abs(S.stable_ms_jk(A, Bm)['point'] - (mu ** 2).mean()) < 0.002 and S.stable_ms_jk(A)['point'] > (mu ** 2).mean() + 0.005,
      (S.stable_ms_jk(A, Bm)['point'], S.stable_ms_jk(A)['point'], (mu ** 2).mean()))
# 2. ratio jackknife coverage
cover = 0
for e in range(400):
    den = rng.normal(100, 5, R); num = 0.02 * den + rng.normal(0, 1.0, R)
    c = S.ratio_jk(num, den)
    cover += c['lo'] <= 0.02 <= c['hi']
check('ratio_jk_coverage', 0.91 <= cover / 400 <= 0.985, cover / 400)
# 3. bounds of |x|
check('abs_bounds', S.abs_bounds({'lo': -0.01, 'hi': 0.02}) == (0.0, 0.02) and S.abs_bounds({'lo': -0.03, 'hi': -0.01}) == (0.01, 0.03))
# 4. rel_rms monotone transform, negative S floors to 0
r = S.rel_rms({'point': -1e-6, 'lo': -2e-6, 'hi': 4e-6}, 1e-2)
check('rel_rms_floor', r['point'] == 0.0 and r['lo'] == 0.0 and abs(r['hi'] - 0.02) < 1e-12)
# 5. energy bias arithmetic: HIGH difference over TOTAL, sign kept
num = np.array([-1.0, -1.2, -0.8]); den = np.array([100.0, 100.0, 100.0])
check('energy_sign', abs(S.ratio_jk(num, den)['point'] + 0.01) < 1e-12)
print('FAILED ' + ', '.join(fails) if fails else 'ALL TESTS PASSED')
sys.exit(1 if fails else 0)
