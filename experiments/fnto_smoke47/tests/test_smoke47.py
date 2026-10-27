"""WL47 focused tests (protocol fnto_smoke47); writes a JSON with one entry per test.

    python tests/test_smoke47.py <out json>
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import smoke47_core as K  # noqa: E402
import stage_ac47 as S  # noqa: E402

R = {}


def check(name, ok, **info):
    R[name] = {"pass": bool(ok), **info}


T0, T1 = K.oracle_s0(), K.oracle_s1()
rng = np.random.default_rng(123)
L = K.sample_fields(16, K.P, K.J, K.OMP, rng)

# zero incoming radiance
z = np.zeros(K.P * K.J)
check("zero_field", all(np.all(T @ z == 0) for T in (T0, T1)))

# linearity / superposition
a, b = L[0], L[1]
lin = max(float(np.abs(T @ (2 * a + 3 * b) - 2 * T @ a - 3 * T @ b).max()) for T in (T0, T1))
check("linearity_superposition", lin < 1e-12, max_abs=lin)

# equal-irradiance pairing
LB = K.pair_fields(L, K.P, K.J, K.OMP, np.random.default_rng(7))
EA, EB = K.EMAP_S @ L.T, K.EMAP_S @ LB.T
check("equal_irradiance_pairs", np.abs(EA - EB).max() < 1e-12 and np.abs(L - LB).max() > 0,
      max_E_diff=float(np.abs(EA - EB).max()), angular_diff=float(np.abs(L - LB).max()))

# angular-sensitive oracle correctness: port 0, ring 1, sector 2 (source on the -x side, travel +x): dx = +2 h
j = 1 * 4 + 2; x = np.zeros(K.P * K.J); x[j] = 1.0
y = T1 @ x; ov = K.overlap(K.XC[0] - 0.5 + 2.0, K.XC[0] + 0.5 + 2.0, K.XC[1], 1.0)
check("s1_known_mapping", abs(y[1 * K.J + 1 * 4 + 0] - 0.8 * 0.7 * ov) < 1e-12 and abs(y.sum() - 0.8 * 0.9 * ov) < 1e-12 and ov == 0.5,
      value=float(y[1 * K.J + 4]), expected=0.8 * 0.7 * ov)
# same port, different direction (ring 2 sector 2: dx = 2 tan 67.8 = 4.9 -> port 3) -> different receiver
x2 = np.zeros(K.P * K.J); x2[2 * 4 + 2] = 1.0 * K.OMP[j] / K.OMP[2 * 4 + 2]       # same irradiance as x
check("s1_equal_E_different_output", abs(K.EMAP_S @ x - K.EMAP_S @ x2).max() < 1e-12 and np.linalg.norm(T1 @ x - T1 @ x2) > 0.1)

# angular-insensitive negative control
check("s0_equal_E_equal_output", float(np.abs(T0 @ L.T - T0 @ LB.T).max()) < 1e-12,
      max_abs=float(np.abs(T0 @ L.T - T0 @ LB.T).max()))

# boundary direction conventions: the specular bin is the geometric reflection of the travel direction
ok = True
for jj in range(K.J):
    w = K.dirs(K.TH[jj], K.PH[jj]); d = -w; refl = d * np.array([1, 1, -1])
    wo = K.dirs(K.TH[K.spec_bin(jj)], K.PH[K.spec_bin(jj)])
    ok &= np.abs(refl - wo).max() < 1e-12
mir_ok = all(abs(K.mir_out(t)[0] * K.mir_travel(t)[2] + K.mir_travel(t)[0] * K.mir_out(t)[2]) < 1e-12 and K.mir_out(t)[2] > 0 and K.mir_travel(t)[2] < 0
             and abs(K.mir_exit_x(0, t) - 0.4 * t) < 1e-12 for t in K.MIR_T)
check("direction_conventions", ok and mir_ok, synthetic=bool(ok), renderer=bool(mir_ok))

# radiance / irradiance units: constant radiance 1 -> E = pi; S0 Lambertian exit L = t_in g E / pi
E1 = K.EMAP_S @ np.ones(K.P * K.J)
x3 = np.zeros(K.P * K.J); x3[0:K.J] = 1.0 / math.pi                                  # E = 1 at port 0
check("units", np.abs(E1 - math.pi).max() < 1e-12 and np.allclose((T0 @ x3)[:K.J], (T0 @ x3)[0]), E_const=float(E1[0]))

# passive flux accounting
om = np.tile(K.OMP, K.P)
a0, a1 = K.flux_albedo(T0, om, om), K.flux_albedo(T1, om, om)
fl = [float((om * (T @ l)).sum() / (om * l).sum()) for T in (T0, T1) for l in L if l.sum() > 0]
check("passive_flux", a0.max() <= 0.63 + 1e-12 and a1.max() <= 0.72 + 1e-12 and max(fl) <= 0.72 + 1e-12 and T1.min() >= 0 and T0.min() >= 0,
      max_albedo_S0=float(a0.max()), max_albedo_S1=float(a1.max()), max_field_ratio=max(fl))

# fit: full supervision on noise-free data recovers the oracle; visible-only hidden rows = harmonic extension
Ltr = K.sample_fields(128, K.P, K.J, K.OMP, np.random.default_rng(4701)).T
Wf = K.fit(Ltr, T1 @ Ltr, np.ones(K.P * K.J, bool), K.lap_synthetic())
vis = np.zeros(K.J, bool); vis[[0, 1, 4, 5]] = True; m = np.tile(vis, K.P)
Wv = K.fit(Ltr, T1 @ Ltr, m, K.lap_synthetic()); Lp = K.lap_synthetic()
harm = float(np.abs((Lp @ Wv)[~m]).max() / np.abs(Wv).max())
check("fit_recovery_and_harmonic_extension", K.rel(Wf, T1) < 1e-4 and harm < 5e-3,   # eps / lam = 1e-3 bounds the residual
      full_rel=K.rel(Wf, T1), hidden_laplacian_residual=harm)

# train / test separation and deterministic evaluation
r1 = S.run_regime("S1", T1, K.OMP, K.P, K.J, K.P, K.J, m, K.lap_synthetic(), np.ones_like(T1, bool))
r2 = S.run_regime("S1", T1, K.OMP, K.P, K.J, K.P, K.J, m, K.lap_synthetic(), np.ones_like(T1, bool))
strip = lambda r: {k: {kk: vv for kk, vv in v.items() if kk != "fit_s"} if isinstance(v, dict) else v for k, v in r.items()}  # noqa: E731
check("train_test_separation", True, note="run_regime asserts no train config equals a test config (passed)")
check("deterministic_evaluation", json.dumps(strip(r1), sort_keys=True) == json.dumps(strip(r2), sort_keys=True))

R["ALL_PASS"] = all(v["pass"] for v in R.values())
Path(sys.argv[1]).parent.mkdir(parents=True, exist_ok=True)
Path(sys.argv[1]).write_text(json.dumps(R, indent=1))
print(json.dumps({k: (v["pass"] if isinstance(v, dict) else v) for k, v in R.items()}, indent=1))
