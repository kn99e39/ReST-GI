"""WL109A focused CPU tests (no GPU): seeds, frozen-statistic identity, injection identity, accumulators, synthetic moments.

CUDA_VISIBLE_DEVICES= python tests109a.py
"""
import math
import os

import numpy as np

assert os.environ.get("CUDA_VISIBLE_DEVICES", None) == ""
import torch  # noqa: E402

import common109a as CA  # noqa: E402
import common109 as C9  # noqa: E402
import stats109 as ST  # noqa: E402
import calib109a as CB  # noqa: E402
from analyze109a import comp_maps  # noqa: E402


def wl109_seed_set():
    s = set()
    P9 = C9.P
    for smoke in (False, True):
        for v in range(6):
            for spp in C9.NOMINAL:
                for r in range(P9["final"]["escalated_replicates"]):
                    s.add(C9.seed_paired(v, spp, r, smoke))
            for spp in C9.GRID:
                for r in range(P9["final"]["replicates"]):
                    s.add(C9.seed_grid(v, spp, r, smoke))
        for d in range(4):
            for m in range(3):
                for r in range(P9["development"]["replicates"]):
                    s.add(C9.seed_dev(d, m, r, smoke))
        for t in range(44):
            for c in range(125):
                for r in range(64):
                    s.add(C9.seed_synth(t, c, r, smoke))
        for v in range(20):
            for slot in range(100):
                for i in range(100):
                    s.add(C9.seed_timing(v, slot, i, smoke))
        for slot in range(100):
            for r in range(64):
                s.add(C9.seed_dynamic(slot, r, smoke))
        for b in range(16):
            s.add(C9.seed_dynamic_reference(b, smoke))
    return s


def test_seed_families():
    fam = {}
    for smoke in (False, True):
        t = "s" if smoke else "e"
        a = fam.setdefault("null" + t, [])
        for v in range(6):
            for si in range(3):
                a += [CA.seed_null(v, si, k, smoke) for k in range(2048)]
        a = fam.setdefault("paired" + t, [])
        for v in range(6):
            for si in range(3):
                a += [CA.seed_paired(v, si, r, smoke) for r in range(1024)]
        a = fam.setdefault("grid" + t, [])
        for v in range(6):
            for gi in range(18):
                a += [CA.seed_grid(v, gi, r, smoke) for r in range(64)]
        a = fam.setdefault("timing" + t, [])
        for v in range(6):
            for slot in range(22):
                a += [CA.seed_timing(v, slot, i, smoke) for i in range(100)]
        a = fam.setdefault("dyn" + t, [])
        for slot in range(70):
            a += [CA.seed_dynamic(slot, r, smoke) for r in range(1024)]
        fam.setdefault("dref" + t, []).extend(CA.seed_dynamic_reference(b, smoke) for b in range(16))
        a = fam.setdefault("mech" + t, [])
        for slot in range(10):
            a += [CA.seed_mech(slot, r, smoke) for r in range(1024)]
    sets = {k: set(v) for k, v in fam.items()}
    for k, v in fam.items():
        assert len(sets[k]) == len(v), k
    names = list(sets)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            assert not (sets[a] & sets[b]), (a, b)
    w9 = wl109_seed_set()
    allv = set().union(*sets.values())
    assert not (allv & w9), "WL109A seeds overlap WL109"
    assert min(allv) > max(x for x in w9 if x < 50000000), "evidence seeds above every WL101-WL109 evidence family"
    # renderer stream values distinct as well
    lv = {C9.launch_seed(x) for x in list(allv)[:200000]}
    assert len(lv) == min(200000, len(allv))


def test_q_identity_with_frozen_stats():
    rs = np.random.default_rng(1)
    for R, G, Pp in ((32, 8, 50), (512, 8, 37), (64, 16, 11)):
        D = rs.standard_t(3, size=(R, 3, Pp)) + 0.05 * rs.standard_normal((1, 3, Pp))
        scale = 0.7
        mask = np.ones(Pp, bool)
        ref = ST.spatial_bias_power(torch.as_tensor(D), mask, scale, G)
        n = R // G
        S1 = D.reshape(G, n, 3, Pp).sum(1); S2 = (D.reshape(G, n, 3, Pp) ** 2).sum(1)
        q = CA.q_summary(CA.q_from_group_sums(S1, S2, n, scale))
        assert abs(q["point"] - ref["point"]) <= 1e-12 * max(1, abs(ref["point"])), (q, ref)
        assert abs(q["se"] - ref["se"]) <= 1e-12 * max(1, ref["se"])
        assert abs(q["upper95"] - ref["upper95"]) <= 1e-12 * max(1, abs(ref["upper95"]))
        # subgroup-of-16 sums merged into groups equal direct group sums
        if R % 16 == 0 and (R // 16) % G == 0:
            SG1 = D.reshape(R // 16, 16, 3, Pp).sum(1)
            assert np.allclose(SG1.reshape(G, -1, 3, Pp).sum(1), S1, rtol=1e-12, atol=1e-12)


def test_injection_identity():
    rs = np.random.default_rng(2)
    R, G, Pp = 64, 8, 40
    n = R // G
    D = rs.standard_normal((R, 3, Pp))
    scale = 1.3
    X = D.reshape(G, n, -1)
    dbar, s2 = CB.group_stats(X)
    Q0 = CB.qg0(dbar, s2, n, scale)
    b = CA.bias_pattern("uniform", Pp, {"noise_sd": np.ones(Pp), "xy": np.zeros((Pp, 2))}, rs).reshape(-1)
    for q in (0.005, 0.02):
        beta = math.sqrt(q * scale) * b
        Xb = X + beta[None, None, :]
        db2, s22 = CB.group_stats(Xb)
        direct = CB.qg0(db2, s22, n, scale)
        formula = Q0 + 2 * math.sqrt(q / scale) * (dbar @ b / b.size) + q
        assert np.allclose(direct, formula, rtol=1e-10, atol=1e-12)
    for kind in ("noise_proportional", "localized", "signed_smooth"):
        bb = CA.bias_pattern(kind, Pp, {"noise_sd": rs.uniform(0.5, 2, Pp), "xy": rs.uniform(0, 30, (Pp, 2))}, rs)
        assert abs((bb ** 2).mean() - 1) < 1e-12 and bb.shape == (3, Pp)


def test_components_from_sums():
    rs = np.random.default_rng(3)
    R, Pp = 40, 25
    L = rs.standard_normal((R, 3, Pp)); H = 0.5 * L + rs.standard_normal((R, 3, Pp)); Cc = 0.1 * H + rs.standard_normal((R, 3, Pp))
    g = {"L": L.sum(0), "H": H.sum(0), "C": Cc.sum(0), "LL": (L * L).sum(0), "HH": (H * H).sum(0), "CC": (Cc * Cc).sum(0),
         "LH": (L * H).sum(0), "LC": (L * Cc).sum(0), "HC": (H * Cc).sum(0)}
    cm = comp_maps(g, R)
    cov = lambda a, b: ((a - a.mean(0)) * (b - b.mean(0))).sum(0) / (R - 1)
    assert np.allclose(cm["var_T_R0"], (L + H).var(0, ddof=1)) and np.allclose(cm["var_T_C1"], (L + Cc).var(0, ddof=1))
    assert np.allclose(cm["two_cov_L_HC"], 2 * cov(L, Cc)) and np.allclose(cm["cov_H_HC"], cov(H, Cc))


def test_mc_like_moments():
    rs = np.random.default_rng(4)
    for k in (4, 32):
        x = CB._mc_like(rs, (400000,), k)
        p, s = 0.3, 1.5
        mean = p * math.exp(s * s / 2)
        assert abs(x.mean() - mean) < 6 * math.sqrt(CB._mc_var(k) / x.size), (k, x.mean(), mean)
        assert 0.8 < x.var() / CB._mc_var(k) < 1.25, (k, x.var() / CB._mc_var(k))


def test_states():
    lim = 0.01
    mk = lambda up, lo: {"upper95": up, "lower95": lo}
    assert CA.q_state({"4": mk(0.005, -0.001), "16": mk(0.009, 0.0)}, lim) == "EQUIVALENT"
    assert CA.q_state({"4": mk(0.02, 0.011), "16": mk(0.005, 0.0)}, lim) == "BIASED"
    assert CA.q_state({"4": mk(0.012, 0.002)}, lim) == "UNRESOLVED"
    rs = np.random.default_rng(5)
    d = {4: rs.normal(0, 1e-4, 512), 16: rs.normal(0, 1e-4, 512)}
    assert CA.b1_state(d, 1.0)["state"] == "EQUIVALENT"
    d = {4: rs.normal(0.02, 1e-4, 512)}
    assert CA.b1_state(d, 1.0)["state"] == "BIASED"


def test_t_coverage_gauss():
    """Q's 8-group t bound has about nominal one-sided coverage on Gaussian noise (sanity of the calibration harness)."""
    rs = np.random.default_rng(6)
    G, n, Pp, scale = 8, 16, 200, 0.5
    non = 0
    reps = 1500
    for _ in range(reps):
        X = rs.standard_normal((G, n, Pp))
        dbar, s2 = CB.group_stats(X)
        Q0 = CB.qg0(dbar, s2, n, scale)
        pt, se, up, lo = CB.summaries(Q0[None, :])
        non += int(up[0] < 0)
    assert 0.005 < non / reps < 0.05, non / reps


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name, flush=True)
    print("ALL TESTS PASS")
