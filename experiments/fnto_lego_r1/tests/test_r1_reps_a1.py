"""Amendment 1 picker: represented area equals the ball area, representatives lie in the ball, the estimator is unbiased."""
import sys, types
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree

code = (Path(__file__).resolve().parents[1] / "r1_reps_a1.py").read_text(encoding="utf-8").replace("import r1_common as C", "")
m = types.ModuleType("m"); exec(code, m.__dict__)
rng = np.random.default_rng(0)
src = rng.random((3000, 3)) * 0.1; area = rng.random(3000) + 0.1; ax = rng.integers(0, 6, 3000); q = rng.random((50, 3)) * 0.1
rep, A, nb = m.near_exact_ori(q, src, area, ax, 0.05, 96, rng)
t = cKDTree(src); balls = t.query_ball_point(q, 0.05); ok = True
for i, b in enumerate(balls):
    ok &= abs(A[i].sum() - area[b].sum()) < 1e-4 * area[b].sum(); ok &= set(rep[i][rep[i] >= 0].tolist()) <= set(b)
f = rng.random(3000); ests = np.array([(lambda r: (r[1] * f[r[0]]).sum(1))(m.near_exact_ori(q[:5], src, area, ax, 0.05, 96, np.random.default_rng(s))) for s in range(400)])
ex = np.array([(area[b] * f[b]).sum() for b in balls[:5]]); bias = np.abs(ests.mean(0) / ex - 1); se = ests.std(0) / np.sqrt(400) / ex
print("ball sizes", nb[:5], "area/ball ok", ok, "rel bias", bias.round(4), "3 se", (3 * se).round(4))
assert ok and np.all(bias < 3 * se + 1e-3), "FAIL"
print("PASS")
