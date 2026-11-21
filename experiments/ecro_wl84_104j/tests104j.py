"""WL104J Stage 2 focused tests (gate before primary fitting). Exits non-zero on any failure.

Usage: python tests104j.py <data_dir> <out_json>
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

import common104j as CM
import fit104j as FT

GF, CL, IO = CM.GF, CM.CL, CM.IO
DEV = CM.DEV
TOL64 = 1e-12


def rel(a, b):
    return float(np.max(np.abs(a - b)) / max(np.max(np.abs(b)), 1e-300))


def rand_dirs(rng, n):
    u = rng.random((n, 2))
    z = np.sqrt(u[:, 0]) * 0.999 + 1e-4
    r = np.sqrt(1 - z * z)
    ph = 2 * np.pi * u[:, 1]
    return np.stack([r * np.cos(ph), r * np.sin(ph), z], -1)


def main():
    data_dir, out = Path(sys.argv[1]), Path(sys.argv[2])
    rows, raw, compiled, desc, ref, wl104i, seen = CM.inputs()
    n = len(rows)
    rng = np.random.default_rng(CM.P["seeds"]["test_directions"])
    res = {}
    # test lanes: random directions in every probe slot (incident fixed per probe as in the reference)
    L = 256
    inc = CM.incident_directions()
    wi = np.broadcast_to(inc[None, :, None, :], (n, CM.NP, L, 3)).copy()
    wo = rand_dirs(rng, n * CM.NP * L).reshape(n, CM.NP, L, 3)
    # ---- T1 original response identity (R0): ER-v1-GF arm A vs original 64-parameter decode and compiled runtime
    yR0 = GF.from_original(raw["R0"])
    FA = CM.predict("A", {"y": yR0}, wi, wo, CM.PROBE_GROUP)
    with torch.no_grad():
        po = CL.decode("ER", "cert", torch.as_tensor(raw["R0"], dtype=torch.float64, device=DEV))
    Fo = np.empty_like(FA)
    with torch.no_grad():
        for m in range(n):
            q = {k: v[m:m + 1].expand(CM.NP * L, *v.shape[1:]) for k, v in po.items()}
            Fo[m] = CL.eval_closure("ER", q, torch.as_tensor(wi[m].reshape(-1, 3), device=DEV),
                                    torch.as_tensor(wo[m].reshape(-1, 3), device=DEV)).cpu().numpy().reshape(CM.NP, L, 3)
    Fd = np.stack([CM.CI.drjit_eval(compiled["R0"][m], wi[m].reshape(-1, 3), wo[m].reshape(-1, 3)).reshape(CM.NP, L, 3)
                   for m in range(n)])
    res["T1_A_vs_original_float64"] = rel(FA, Fo)
    res["T1_A_vs_compiled_runtime"] = rel(FA, Fd)
    ok = res["T1_A_vs_original_float64"] <= TOL64 and res["T1_A_vs_compiled_runtime"] <= 2e-6
    # ---- T2 A in B: section of shared states reproduces A (historical, WL104I, random, extreme)
    cases = {"R0": yR0, "WL104D_A": GF.from_original(raw["A"]), "WL104D_C": GF.from_original(raw["C"]),
             "ER_L": GF.from_original(raw["ER_L"]), "WL104I_B1_generic": wl104i["B1_generic"]}
    yr = np.repeat(GF.from_original(CM.CI.generic(n)), 1, 0) + rng.standard_normal((n, 58))
    cases["random"] = yr
    ye = yr.copy(); ye[:, GF.SL_KNOT] = rng.choice([-8.0, 8.0], (n, 15)); ye[:, :21] = rng.choice([-8.0, 8.0], (n, 21))
    cases["extreme"] = ye
    t2 = {}
    for k, y in cases.items():
        Fa = CM.predict("A", {"y": y}, wi, wo, CM.PROBE_GROUP)
        Fb = CM.predict("B", FT.local_from_A(y), wi, wo, CM.PROBE_GROUP)
        t2[k] = rel(Fb, Fa)
    res["T2_B_section_reproduces_A"] = t2
    ok &= max(t2.values()) <= TOL64
    # ---- T3 B in C: identical decode with shared shapes; group-local shapes act only on their group
    loc = FT.local_from_A(yR0)
    Fb = CM.predict("B", loc, wi, wo, CM.PROBE_GROUP)
    Fc = CM.predict("C", {**loc}, wi, wo, CM.PROBE_GROUP)
    res["T3_C_with_B_state_equals_B"] = rel(Fc, Fb)
    loc2 = {k: v.copy() for k, v in loc.items()}
    g = 4
    loc2["shape_raw"][:, g] += 0.3
    loc2["amp"][:, 7] -= 0.3
    F2 = CM.predict("C", loc2, wi, wo, CM.PROBE_GROUP)
    changed = np.abs(F2 - Fb).max(axis=(0, 2, 3)) > 0
    expect = np.isin(CM.PROBE_GROUP, [g, 7])
    res["T3_group_locality_changed_probes"] = np.nonzero(changed)[0].tolist()
    ok &= res["T3_C_with_B_state_equals_B"] == 0.0 and bool(np.all(changed == expect))
    # ---- T4 nonnegativity / budget / bounds for random local states
    locr = {k: v.copy() for k, v in loc.items()}
    locr["amp"] = locr["amp"] + 2.0 * rng.standard_normal(locr["amp"].shape)
    locr["shape_raw"] = locr["shape_raw"] + 2.0 * rng.standard_normal(locr["shape_raw"].shape)
    with torch.no_grad():
        p = CM.lane_params("C", locr)
    lam = p["lam"].cpu().numpy()
    T = CM.terms("C", locr, wi[:, :, :64], wo[:, :, :64], CM.PROBE_GROUP)
    Fn = CM.predict("C", locr, wi[:, :, :64], wo[:, :, :64], CM.PROBE_GROUP)
    tsum = T["O0"] + T["O1"] + T["O2"] + T["SL"] + T["SI0"] + T["SI1"] + T["SI2"] + T["SI3"]
    res["T4_min_term"] = float(min(v.min() for v in T.values()))
    res["T4_max_budget"] = float(lam.sum(-1).max())
    res["T4_terms_sum_vs_native"] = rel(tsum, Fn)
    res["T4_f0_range"] = [float(p["f0"].min()), float(p["f0"].max())]
    ok &= res["T4_min_term"] >= 0 and res["T4_max_budget"] <= 1 + 1e-12 and res["T4_terms_sum_vs_native"] <= TOL64
    # ---- T5 cosine convention / coordinates: a Lambert-only local state gives s(mu_i) s(mu_o) cos_o / pi
    lamb = {k: v.copy() for k, v in loc.items()}
    lamb["amp"] = np.full_like(lamb["amp"], -60.0)
    lamb["amp"][..., 0] = 60.0                                        # lambda_L -> 1
    FL = CM.predict("B", lamb, wi[:, :, :64], wo[:, :, :64], CM.PROBE_GROUP)
    with torch.no_grad():
        sn = torch.as_tensor(lamb["s_n"], device=DEV)
        exp = np.empty_like(FL)
        for m in range(n):
            a = torch.as_tensor(wi[m, :, :64].reshape(-1, 3), device=DEV)
            b = torch.as_tensor(wo[m, :, :64].reshape(-1, 3), device=DEV)
            s = sn[m:m + 1].expand(len(a), 3, 5)
            v = CL.s_interp(s, a[:, 2]) * CL.s_interp(s, b[:, 2]) * (b[:, 2:3] / np.pi)
            exp[m] = v.cpu().numpy().reshape(CM.NP, 64, 3)
    res["T5_lambert_cosine_convention"] = rel(FL, exp)
    ok &= res["T5_lambert_cosine_convention"] <= 1e-12
    # ---- data-based tests (fit set)
    dj = IO.read(data_dir / "data.json")
    with np.load(data_dir / "fit_set.npz") as z:
        d = {k: z[k] for k in z.files}
    res["T9_isolation"] = dj["isolation"]
    res["T9_data_qualified"] = dj["qualified"]
    ok &= bool(dj["qualified"]["all"]) or (dj["smoke"] and dj["qualified"]["isolation"])
    # ---- T6 masking + T7 deterministic replay (3 steps, real data, R = 1)
    data = FT.FitData(d, n, 1)
    init = FT.local_from_A(yR0)
    t6 = {}
    for arm in ("B", "C", "C_noF"):
        a1, _ = FT.train(data, arm, init, 3)
        a2, _ = FT.train(data, arm, init, 3)
        t6[f"{arm}_replay_max_abs"] = float(max(np.abs(a1[k] - a2[k]).max() for k in a1))
        frozen = [k for k in init if k not in FT.TRAINABLE[arm]]
        t6[f"{arm}_frozen_changed"] = float(max(np.abs(a1[k] - init[k]).max() for k in frozen))
        if arm == "C_noF":
            t6["C_noF_f0_changed"] = float(np.abs(a1["shape_raw"][..., FT.F0_COLS] - init["shape_raw"][..., FT.F0_COLS]).max())
            t6["C_noF_alpha_changed"] = float(np.abs(np.delete(a1["shape_raw"] - init["shape_raw"], FT.F0_COLS, -1)).max())
    aA1, _ = FT.train(data, "A", {"y": yR0}, 3)
    aA2, _ = FT.train(data, "A", {"y": yR0}, 3)
    t6["A_replay_max_abs"] = float(np.abs(aA1["y"] - aA2["y"]).max())
    res["T6_T7_masking_replay"] = t6
    ok &= all(v == 0 for k, v in t6.items() if "replay" in k or "frozen" in k or "f0_changed" in k) and t6["C_noF_alpha_changed"] > 0
    # ---- T8 gradient consistency (float64 objective vs central finite differences)
    data64 = FT.FitData(d, n, 1)
    for k in ("wi", "wo", "ref", "w"):
        setattr(data64, k, getattr(data64, k).double())
    base = {kk: torch.as_tensor(v, dtype=torch.float64, device=DEV) for kk, v in init.items()}
    t8 = {}
    for arm, key, idx in (("B", "amp", (0, 3, 1, 2)), ("C", "shape_raw", (2, 5, 4)), ("C", "amp", (8, 9, 0, 5))):
        x = base[key].clone().requires_grad_(True)
        st = {**base, key: x}
        p = CM.lane_params(arm, st, torch.float64)
        pp = CM.to_lanes(CM.per_probe(p, arm, n, CM.PROBE_GROUP), data64.L)
        pred = CL.eval_closure("ER", pp, data64.wi, data64.wo)
        obj = FT.F.group_terms(pred, data64.ref, data64.w, data64.L)["metric"].sum()
        gauto = torch.autograd.grad(obj, x)[0][idx].item()
        h = 1e-6
        vals = []
        for sgn in (1, -1):
            xx = base[key].clone(); xx[idx] += sgn * h
            with torch.no_grad():
                p = CM.lane_params(arm, {**base, key: xx}, torch.float64)
                pp = CM.to_lanes(CM.per_probe(p, arm, n, CM.PROBE_GROUP), data64.L)
                pred = CL.eval_closure("ER", pp, data64.wi, data64.wo)
                vals.append(FT.F.group_terms(pred, data64.ref, data64.w, data64.L)["metric"].sum().item())
        gfd = (vals[0] - vals[1]) / (2 * h)
        t8[f"{arm}_{key}_{idx}"] = {"autograd": gauto, "fd": gfd, "rel": abs(gauto - gfd) / max(abs(gfd), 1e-12)}
    res["T8_gradient"] = t8
    ok &= all(v["rel"] <= 1e-5 for v in t8.values())
    # ---- T10 synthetic truths: S1 representable by B, S2 by C, both not by a shared A state
    t10 = {}
    for ctl in ("S1", "S2"):
        tr = FT.synthetic_truth(ctl, yR0)
        Ft = CM.predict(tr["arm"], tr["state"], wi, wo, CM.PROBE_GROUP)
        t10[f"{ctl}_differs_from_shared"] = rel(Ft, FA)
        if ctl == "S1":
            t10["S1_amp_only_changed"] = bool(np.array_equal(tr["state"]["shape_raw"], FT.local_from_A(yR0)["shape_raw"]))
    res["T10_synthetic"] = t10
    ok &= all(v > 1e-3 for k, v in t10.items() if "differs" in k) and t10["S1_amp_only_changed"]
    res["passed"] = bool(ok)
    res["commit"] = CM.commit()
    res["inputs_sha256"] = seen
    CM.write(out, res)
    print(json.dumps({k: v for k, v in res.items() if k not in ("inputs_sha256",)}, default=float, indent=1)[:4000], flush=True)
    if not ok:
        raise IO.Stop("FOCUSED_TESTS_FAILED")


if __name__ == "__main__":
    main()
