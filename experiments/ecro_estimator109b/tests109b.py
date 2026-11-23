"""WL109B focused CPU tests (no GPU): seed independence, estimand preservation, XJ / XJL algebra, inversion, lam, grouping,
calibration classifier, synthetic generator moments.

CUDA_VISIBLE_DEVICES= python tests109b.py
"""
import math
import os
from pathlib import Path
import sys

import numpy as np

assert os.environ.get("CUDA_VISIBLE_DEVICES", None) == ""
import torch  # noqa: E402

import common109b as CB  # noqa: E402
import calib109b as CL  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ecro_estimator109a"))
import common109a as CA  # noqa: E402
import tests109a as TA  # noqa: E402
import stats109 as ST  # noqa: E402


def b_seed_set(smoke):
    s = {}
    s["null"] = [CB.seed_null(v, si, k, smoke) for v in range(6) for si in range(3) for k in range(4096)]
    s["paired"] = [CB.seed_paired(v, si, r, smoke) for v in range(6) for si in range(3) for r in range(1024)]
    s["replication"] = [CB.seed_replication(v, si, r, smoke) for v in range(6) for si in range(3) for r in range(1024)]
    s["grid"] = [CB.seed_grid(v, gi, r, smoke) for v in range(6) for gi in range(18) for r in range(64)]
    s["timing"] = [CB.seed_timing(v, slot, i, smoke) for v in range(6) for slot in range(22) for i in range(100)]
    s["dyn"] = [CB.seed_dynamic(slot, r, smoke) for slot in range(70) for r in range(1024)]
    s["dref"] = [CB.seed_dynamic_reference(b, smoke) for b in range(16)]
    s["mech"] = [CB.seed_mech(slot, r, smoke) for slot in range(10) for r in range(1024)]
    return s


def a_seed_set():
    out = set()
    for smoke in (False, True):
        out |= {CA.seed_null(v, si, k, smoke) for v in range(6) for si in range(3) for k in range(2048)}
        out |= {CA.seed_paired(v, si, r, smoke) for v in range(6) for si in range(3) for r in range(1024)}
        out |= {CA.seed_grid(v, gi, r, smoke) for v in range(6) for gi in range(18) for r in range(64)}
        out |= {CA.seed_timing(v, slot, i, smoke) for v in range(6) for slot in range(22) for i in range(100)}
        out |= {CA.seed_dynamic(slot, r, smoke) for slot in range(70) for r in range(1024)}
        out |= {CA.seed_dynamic_reference(b, smoke) for b in range(16)}
        out |= {CA.seed_mech(slot, r, smoke) for slot in range(10) for r in range(1024)}
    return out


def test_seed_families():
    fams = {}
    for smoke in (False, True):
        for k, v in b_seed_set(smoke).items():
            fams[k + ("s" if smoke else "e")] = v
    sets = {k: set(v) for k, v in fams.items()}
    for k, v in fams.items():
        assert len(sets[k]) == len(v), k
    names = list(sets)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            assert not (sets[a] & sets[b]), (a, b)
    allb = set().union(*sets.values())
    assert not (allb & TA.wl109_seed_set()), "WL109B seeds overlap WL109"
    assert not (allb & a_seed_set()), "WL109B seeds overlap WL109A"
    dev = set(range(77_000_001, 77_200_000))                       # WL109B development study seeds
    assert not (allb & dev)
    ev = set().union(*[sets[k] for k in sets if k.endswith("e")])
    assert min(ev) > 26_999_999 + 10000, "evidence seeds above every WL101-WL109A evidence family"
    lv = {CB.launch_seed(x) for x in list(allb)[:200000]}
    assert len(lv) == min(200000, len(allb))


def test_original_q_unchanged():
    TA.test_q_identity_with_frozen_stats()


def test_xj_bruteforce_and_jackknife():
    rs = np.random.default_rng(11)
    for G, X in ((16, 30), (8, 7), (32, 5)):
        d = rs.standard_normal((G, X)) + 0.3
        scale = 0.9
        U, se2 = CB.xj_core(d, scale)
        brute = sum(float(d[g] @ d[h]) for g in range(G) for h in range(G) if g != h) / (G * (G - 1) * X * scale)
        assert abs(U - brute) <= 1e-12 * max(1, abs(brute))
        loo = []
        for g in range(G):
            e = np.delete(d, g, 0)
            loo.append(sum(float(e[i] @ e[j]) for i in range(G - 1) for j in range(G - 1) if i != j) / ((G - 1) * (G - 2) * X * scale))
        ps = G * U - (G - 1) * np.asarray(loo)
        assert abs(ps.mean() - U) <= 1e-10 * max(1, abs(U))
        assert abs(ps.var(ddof=1) / G - se2) <= 1e-10 * max(1, se2)


def test_xj_unbiased_same_estimand_as_original():
    """Both the original Q and U estimate mean(mu^2) / scale; Monte Carlo check with known mu."""
    rs = np.random.default_rng(12)
    R, G, Pp, scale = 128, 16, 40, 2.0
    mu = rs.standard_normal((3, Pp)) * 0.2
    q_true = float((mu ** 2).mean() / scale)
    u, q9 = [], []
    for _ in range(3000):
        D = mu[None] + rs.standard_normal((R, 3, Pp))
        gm = D.reshape(G, R // G, -1).mean(1)
        u.append(CB.xj_core(gm, scale)[0])
        S1 = D.reshape(8, R // 8, 3, Pp).sum(1); S2 = (D.reshape(8, R // 8, 3, Pp) ** 2).sum(1)
        q9.append(CA.q_summary(CA.q_from_group_sums(S1, S2, R // 8, scale))["point"])
    u, q9 = np.asarray(u), np.asarray(q9)
    assert abs(u.mean() - q_true) < 4 * u.std() / math.sqrt(len(u)), (u.mean(), q_true)
    assert abs(q9.mean() - q_true) < 4 * q9.std() / math.sqrt(len(q9)), (q9.mean(), q_true)


def test_xjl_inversion():
    rs = np.random.default_rng(13)
    for _ in range(200):
        G, X = 16, 60
        d = rs.standard_normal((G, X)) * rs.uniform(0.1, 2) + rs.uniform(-0.5, 0.5)
        scale = rs.uniform(0.2, 3); lam = rs.uniform(0.01, 5); R = G * 32
        r = CB.xjl_bounds(d, scale, lam, R)
        U, se2 = CB.xj_core(d, scale)
        b = 4 * lam / (R * X * scale)
        tu = CB.t_quant(1 - CB.INF["alpha_upper_one_sided"], G - 1)
        tl = CB.t_quant(1 - CB.INF["alpha_lower_one_sided_family"], G - 1)
        assert r["upper"] >= 0 and r["lower_family"] >= 0
        assert r["lower_family"] <= max(U, 0) + 1e-12 <= r["upper"] + 2e-12 or r["upper"] == 0
        if r["upper"] > 0:
            assert abs(U - (r["upper"] - tu * math.sqrt(se2 + b * r["upper"]))) <= 1e-9 * max(1, abs(U))
        if r["lower_family"] > 0:
            assert abs(U - (r["lower_family"] + tl * math.sqrt(se2 + b * r["lower_family"]))) <= 1e-9 * max(1, abs(U))
        # b -> 0 recovers the plain jackknife-t upper bound
        r0 = CB.xjl_bounds(d, scale, 0.0, R)
        assert abs(r0["upper"] - max(0.0, U + tu * math.sqrt(se2))) <= 1e-9 * max(1, abs(U)) or U + tu * math.sqrt(se2) < 0


def test_lam_and_grouping():
    rs = np.random.default_rng(14)
    for k, X in ((32, 10), (32, 200)):
        m = rs.standard_normal((k, X)) @ np.diag(rs.uniform(0.5, 2, X))
        lam = CB.lam_max(m, 16)
        c = np.cov(m.T, ddof=1)
        ref = float(np.linalg.eigvalsh(np.atleast_2d(c))[-1]) * 16
        assert abs(lam - ref) <= 1e-9 * ref
    SG1 = rs.standard_normal((32, 3, 7))
    gm = CB.groups_from_subgroups(SG1, 16, 16)
    assert np.allclose(gm, SG1.reshape(16, 2, 3, 7).sum(1) / 32)


def test_coverage_class():
    assert CB.coverage_class(40, 2000, 0.05)[0] == "PASS"
    assert CB.coverage_class(130, 2000, 0.05)[0] == "FAIL"
    assert CB.coverage_class(95, 2000, 0.05)[0] == "BORDERLINE"
    assert CB.coverage_class(32, 600, 0.05)[0] == "BORDERLINE"            # 0.053 from 600 trials is not a failure


def test_synthetic_moments():
    rs = np.random.default_rng(15)
    for fam in CB.P["calibration"]["synthetic"]["families"]:
        Pp = 30
        xy = rs.uniform(0, 10, (Pp, 2))
        sm = CL._smooth_matrix(xy, 1.5, 4.0) if fam == "correlated" else None
        D, var = CL.synth_noise(fam, np.random.default_rng(16), 4000, Pp, sm)
        assert D.shape == (4000, 3, Pp)
        z = D.mean(0) / np.sqrt(var / 4000)
        assert np.abs(z).max() < 6, (fam, np.abs(z).max())
        if fam in ("gauss_iid", "mc_like_4", "asym_mc4"):
            assert abs(D.var(0).mean() / var.mean() - 1) < 0.1, fam


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name, flush=True)
    print("ALL TESTS PASS")
