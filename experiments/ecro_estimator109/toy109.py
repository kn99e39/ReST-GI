"""WL109 Stage C (part 1): exact discrete model of the ECRO pixel estimator and of C1 (CPU, numpy).

Model of one pixel:  J -> first interaction state j (prob q_j);  P -> continuation outcome d ~ p(d | j) drawn by inverse CDF of
one uniform (the same function serves the main and the inner continuation);  S -> second-walk noise s in {-1, +1}.
    T(j, d) = f(j, d) / p(d | j)  (throughput),  v(j, d) in {0, 1} (visibility / escape),  a(j, d) = table address
    E1 = e1[j] + g1[j, d]   (sphere-like first hits carry P dependence through the walk NEE)
    E2 = T v (dir[a] (1 + sig s) + ball[j])
    H  = T v B[a]
R0:  Y = E1 + E2 + H (shared j, d, s).  C1(M):  Y = E1 + E2 + (1/M) sum_m T(j, d'_m) v B[a(j, d'_m)] with iid d'_m | j.
Exact expectations and variances follow by enumeration; the simulated estimator is compared against them.
"""
import math

import numpy as np

CASES = ("constant", "piecewise", "directional", "visibility", "contrast", "zero", "positive_cov", "negative_cov")


class Model:
    def __init__(self, case, seed):
        rs = np.random.default_rng(seed)
        nj, nd = 6, 12
        self.case, self.nj, self.nd = case, nj, nd
        self.q = np.full(nj, 1.0 / nj)
        p = rs.uniform(0.2, 1.0, (nj, nd)); self.p = p / p.sum(1, keepdims=True)
        f = rs.uniform(0.2, 1.0, (nj, nd)); self.f = f / f.sum(1, keepdims=True) * rs.uniform(0.5, 0.9, (nj, 1))
        if case in ("positive_cov", "negative_cov"):
            # perfect importance sampling (constant throughput per j) isolates the table/direct-light covariance sign
            self.f = self.p * rs.uniform(0.5, 0.9, (nj, 1))
        self.T = self.f / self.p
        self.v = np.ones((nj, nd))
        n_patch, n_bin = 5, 4
        self.addr = rs.integers(0, n_patch, (nj, nd)) * n_bin + rs.integers(0, n_bin, (nj, nd))
        na = n_patch * n_bin
        bin_dir = np.tile(np.linspace(0.1, 1.0, n_bin), n_patch)
        if case == "constant":
            B = np.full(na, 0.7)
        elif case == "piecewise":
            B = np.repeat(rs.uniform(0.0, 2.0, n_patch), n_bin)
        elif case == "directional":
            B = 0.3 + 1.5 * bin_dir
        elif case == "visibility":
            B = rs.uniform(0.2, 1.5, na)
            self.v[: nj // 2, : nd // 2] = 0.0                 # an occluder edge seen by half of the first-hit states
        elif case == "contrast":
            B = np.where(np.arange(na) // n_bin == 0, 1000.0, 1.0)
        elif case == "zero":
            B = np.zeros(na)
        else:
            B = rs.uniform(0.1, 2.0, na)
        self.B = B
        direct = rs.uniform(0.0, 1.0, na)
        if case == "positive_cov":
            direct = 2.0 * B
        elif case == "negative_cov":
            direct = 2.0 * (B.max() - B)
        self.dir = direct
        self.sig = 0.6
        self.e1 = rs.uniform(0.0, 1.0, nj)
        self.g1 = rs.uniform(-0.2, 0.2, (nj, nd)) * (np.arange(nj)[:, None] % 2)
        self.ball = rs.uniform(0.0, 0.3, nj)
        self.cdf = np.cumsum(self.p, 1)
        self.cdf[:, -1] = 1.0

    # ---- exact moments over (j, d, s)
    def exact(self, M):
        q, p, T, v, a = self.q, self.p, self.T, self.v, self.addr
        H = T * v * self.B[a]
        E1 = self.e1[:, None] + self.g1
        L_s = {s: E1 + T * v * (self.dir[a] * (1 + self.sig * s) + self.ball[:, None]) for s in (-1, 1)}
        w = q[:, None] * p
        EH = float((w * H).sum())
        EL = float(sum(0.5 * (w * L).sum() for L in L_s.values()))
        h = (p * H).sum(1)                                     # E[H | j]
        Lj = sum(0.5 * (p * L).sum(1) for L in L_s.values())   # E[L | j]
        var_L = float(sum(0.5 * (w * (L - EL) ** 2).sum() for L in L_s.values()))
        var_H = float((w * (H - EH) ** 2).sum())
        cov_LH = float(sum(0.5 * (w * (L - EL) * (H - EH)).sum() for L in L_s.values()))
        var_h = float((q * (h - EH) ** 2).sum())
        evar_H = var_H - var_h
        cov_J = float((q * (Lj - EL) * (h - EH)).sum())
        var_r0 = var_L + var_H + 2 * cov_LH
        var_c1 = var_L + var_h + evar_H / M + 2 * cov_J
        return {"E_TOTAL": EL + EH, "E_H": EH, "var_L": var_L, "var_H": var_H, "two_cov_LH": 2 * cov_LH, "var_R0": var_r0,
                "var_EH_given_J": var_h, "two_cov_J": 2 * cov_J, "var_C1_H": var_h + evar_H / M, "var_C1": var_c1}

    # ---- simulated estimators (n samples per pixel estimate, K independent replicates)
    def _cont(self, j, u):
        return (u[..., None] > self.cdf[j]).sum(-1).clip(0, self.nd - 1)

    def simulate(self, M, n, K, seed):
        rs = np.random.default_rng(seed)
        j = rs.choice(self.nj, size=(K, n), p=self.q)
        d = self._cont(j, rs.random((K, n)))
        s = rs.choice([-1.0, 1.0], size=(K, n))
        T, v, a = self.T[j, d], self.v[j, d], self.addr[j, d]
        E1 = self.e1[j] + self.g1[j, d]
        L = E1 + T * v * (self.dir[a] * (1 + self.sig * s) + self.ball[j])
        H0 = T * v * self.B[a]
        hc = np.zeros_like(H0)
        for _ in range(M):
            dm = self._cont(j, rs.random((K, n)))          # fresh, independent stream; same continuation function
            hc += self.T[j, dm] * self.v[j, dm] * self.B[self.addr[j, dm]]
        hc /= M
        return {"R0": (L + H0).mean(1), "C1": (L + hc).mean(1), "H0": H0.mean(1), "HC": hc.mean(1), "L": L.mean(1)}


def run(cfg, seeds):
    M_list, n, K = cfg["M"], cfg["n"], cfg["K"]
    z_crit, var_z = cfg["mean_z_max"], cfg["var_ratio_z_max"]
    rows, ok = [], True
    for case in CASES:
        for si, seed in enumerate(seeds):
            mdl = Model(case, seed)
            for M in M_list:
                ex = mdl.exact(M)
                sim = mdl.simulate(M, n, K, seed * 7 + M)
                row = {"case": case, "seed": int(seed), "M": M, "exact": ex}
                for arm, var_key in (("R0", "var_R0"), ("C1", "var_C1")):
                    x = sim[arm]
                    mean, sd = float(x.mean()), float(x.std(ddof=1))
                    z = (mean - ex["E_TOTAL"]) / (sd / math.sqrt(K)) if sd > 0 else (0.0 if abs(mean - ex["E_TOTAL"]) < 1e-12 else math.inf)
                    var_exact = ex[var_key] / n
                    ratio = float(x.var(ddof=1) / var_exact) if var_exact > 0 else (1.0 if x.var() == 0 else math.inf)
                    # normal interval of the variance ratio from the sample kurtosis
                    if var_exact > 0:
                        c = x - mean
                        kurt = float((c ** 4).mean() / (c ** 2).mean() ** 2)
                        se = ratio * math.sqrt(max(kurt - 1 + 2 / (K - 1), 1e-12) / K)
                    else:
                        se = 0.0
                    lo, hi = ratio - var_z * se, ratio + var_z * se
                    passed = abs(z) <= z_crit and lo <= 1.0 <= hi
                    row[arm] = {"mean": mean, "z": float(z), "var_ratio_to_exact": ratio, "var_ratio_interval": [lo, hi], "pass": bool(passed)}
                    ok &= passed
                hz = (sim["HC"].mean() - ex["E_H"]) / (sim["HC"].std(ddof=1) / math.sqrt(K) + 1e-300)
                row["C1_H_mean_z"] = float(hz) if case != "zero" else 0.0
                row["predicted_C1_over_R0"] = ex["var_C1"] / ex["var_R0"] if ex["var_R0"] > 0 else None
                if case == "zero":
                    zero_ok = bool(np.all(sim["HC"] == 0) and np.all(sim["H0"] == 0) and np.allclose(sim["C1"], sim["R0"]))
                    row["zero_exact"] = zero_ok
                    ok &= zero_ok
                rows.append(row)
    # the predeclared known-negative case must be a real negative: exact C1(M=1) variance above R0
    neg = [r for r in rows if r["case"] == "negative_cov" and r["M"] == 1]
    neg_ok = all(r["exact"]["var_C1"] > r["exact"]["var_R0"] and r["C1"]["var_ratio_to_exact"] > 0 for r in neg)
    neg_emp = all(r["C1"]["var_ratio_to_exact"] * r["exact"]["var_C1"] > r["R0"]["var_ratio_to_exact"] * r["exact"]["var_R0"] for r in neg)
    pos = [r for r in rows if r["case"] == "positive_cov" and r["M"] == 1]
    pos_ok = all(r["exact"]["var_C1"] < r["exact"]["var_R0"] for r in pos)
    ok &= neg_ok and neg_emp and pos_ok
    return {"pass": bool(ok), "negative_case_exact_worse": bool(neg_ok), "negative_case_empirical_worse": bool(neg_emp),
            "positive_case_exact_better": bool(pos_ok), "rows": rows}
