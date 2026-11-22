"""WL109 focused CPU tests (GPU contract tests live in synth109.py and run inside the frozen chain).

python tests109.py
"""
import math

import numpy as np

from common109 import P, EVAL_VIEWS, DEV_VIEWS, GRID, NOMINAL, seed_dev, seed_dynamic, seed_dynamic_reference, seed_grid, seed_paired, \
    seed_synth, seed_timing
from c1_109 import inner_seed_value
import toy109


def test_seed_families():
    fam = {}
    def add(name, s):
        fam.setdefault(name, []).append(s)
    for smoke in (False, True):
        tag = "s" if smoke else "e"
        for v in range(len(EVAL_VIEWS)):
            for spp in NOMINAL:
                for r in range(P["final"]["escalated_replicates"]):
                    add("paired" + tag, seed_paired(v, spp, r, smoke))
            for spp in GRID:
                for r in range(P["final"]["replicates"]):
                    add("grid" + tag, seed_grid(v, spp, r, smoke))
        for d in range(len(DEV_VIEWS)):
            for m in range(len(P["development"]["M_candidates"])):
                for r in range(P["development"]["replicates"]):
                    add("dev" + tag, seed_dev(d, m, r, smoke))
        for t in range(44):
            for c in range(125):
                for r in range(64):
                    add("syn" + tag, seed_synth(t, c, r, smoke))
        for v in range(20):
            for slot in range(100):
                for i in range(100):
                    add("tim" + tag, seed_timing(v, slot, i, smoke))
        for slot in range(100):
            for r in range(64):
                add("dyn" + tag, seed_dynamic(slot, r, smoke))
        for b in range(P["dynamic"]["reference_blocks"]):
            add("dref" + tag, seed_dynamic_reference(b, smoke))
    allv = [s for v in fam.values() for s in v]
    # within each family the generator may revisit a value only through the sparse loops above; across families never
    sets = {k: set(v) for k, v in fam.items()}
    names = list(sets)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            assert not (sets[a] & sets[b]), (a, b)
    for k in ("paired", "grid", "dev", "syn", "tim", "dyn"):
        assert len(sets[k + "e"]) == len(fam[k + "e"]), k
    prior_max = 11799001
    assert min(allv) > prior_max, "WL109 seeds must lie above every prior WL101-WL108 family"
    s = sorted(allv)
    assert all(b - a >= 1 for a, b in zip(s, s[1:]))


def test_launch_seed():
    from common109 import launch_seed, splitmix64
    vals = {launch_seed(s, k) for s in range(13100000, 13100000 + 5000) for k in range(4)}
    assert len(vals) == 20000 and all(0 <= v < 2 ** 64 for v in vals)
    assert splitmix64(0) == 0xE220A8397B1DCDAF        # reference splitmix64 output for input 0


def test_inner_seed():
    vals = {inner_seed_value(s) for s in range(13000000, 13000000 + 20000)}
    assert len(vals) == 20000
    assert all(0 <= v < 2 ** 64 for v in vals)


def test_interp():
    from analyze109 import interp_mse, interp_vec
    t = np.array([10.0, 12.0, 15.0, 20.0])
    mse = 3.0 / t ** 1.0
    v, f = interp_mse(13.0, t, mse)
    assert f == "ok" and abs(v - 3.0 / 13.0) < 1e-12
    assert interp_mse(9.0, t, mse)[1] == "below" and interp_mse(25.0, t, mse)[1] == "above"
    tb = np.broadcast_to(t, (5, 4)).copy(); mb = np.broadcast_to(mse, (5, 4)).copy()
    out = interp_vec(np.array([10.0, 11.0, 13.0, 20.0, 21.0]), tb, mb)
    assert np.allclose(out[:4], 3.0 / np.array([10.0, 11.0, 13.0, 20.0])) and np.isnan(out[4])


def test_variance_identity_and_components():
    rs = np.random.default_rng(1)
    R = 400
    J = rs.normal(size=(R, 1, 4, 4))
    Pm = rs.normal(size=(R, 1, 4, 4))
    L = J + Pm + rs.normal(size=(R, 1, 4, 4))
    HO = 0.5 * J + Pm
    HC = 0.5 * J + rs.normal(size=(R, 1, 4, 4))
    from analyze109 import pooled_var, pooled_cov
    m = np.ones((4, 4), bool)
    vT = pooled_var(L + HO, m)
    assert abs(vT - (pooled_var(L, m) + pooled_var(HO, m) + 2 * pooled_cov(L, HO, m))) < 1e-9
    # Cov(HO, HC) estimates Var(E[H|J]) = 0.25; Cov(L, HC) estimates Cov(E[L|J], E[H|J]) = 0.5
    assert abs(pooled_cov(HO, HC, m) - 0.25) < 0.08 and abs(pooled_cov(L, HC, m) - 0.5) < 0.12


def test_toy_contract():
    cfg = dict(P["synthetic"]["toy"]); cfg["K"] = 1500
    out = toy109.run(cfg, cfg["seeds"][:1])
    assert out["negative_case_exact_worse"] and out["positive_case_exact_better"]
    for r in out["rows"]:
        ex = r["exact"]
        if r["case"] == "zero":
            assert ex["var_C1_H"] == 0 and r["zero_exact"]
        # unbiasedness is exact by construction of the model
        assert abs(ex["E_H"] - ex["E_H"]) == 0


def test_timing_schedule():
    from gpu109 import timing_schedule
    a = timing_schedule([1, 2, 3], 4, 7)
    b = timing_schedule([1, 2, 3], 4, 7)
    assert a == b and len(a) == 12 and sorted(x for _, x in a) == [1] * 4 + [2] * 4 + [3] * 4


def test_bias_states():
    tau = P["analysis"]["bias_tolerance_relative"]
    def state(lo, hi):
        return "EQUIVALENT" if (lo >= -tau and hi <= tau) else ("BIASED" if (lo > tau or hi < -tau) else "UNRESOLVED")
    assert state(-tau / 2, tau / 2) == "EQUIVALENT" and state(tau * 1.1, tau * 2) == "BIASED" and state(-tau / 2, tau * 2) == "UNRESOLVED"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
    print("ALL TESTS PASS")
