"""WL104K Stage E: independent angular evaluation of A / T / P, contrasts, transfer, synthetic controls, convergence, Jacobian rank,
runtime / storage, gate profiles and the angular decision labels.

Usage: python eval104k.py <run_root> <out_dir>
  run_root contains fit_primary/, fit_extended/, synth_S0/, synth_ST/, synth_SP/, tests.json, tests_fitted.json.
"""
import argparse
import time
from pathlib import Path

import numpy as np
import torch

import common104k as CM

GF, CL, IO, J = CM.GF, CM.CL, CM.IO, CM.J
LUMA = CM.LUMA
REGIONS = {"core": (0, 3), "shoulder": (3, 12), "tail": (12, 181)}


def metrics(pred, ref, rad=None):
    out = dict(CM.wl82_terms(pred, ref["mean"], ref["w"]))
    if "a" in ref:
        for h in ("a", "b"):
            out[f"el1_{h}"] = CM.wl82_terms(pred, ref[h], ref["w"])["el1"]
    if rad is not None:
        pl, rl = (pred @ LUMA) * ref["w"], (ref["mean"] @ LUMA) * ref["w"]
        tot = np.maximum(rl.sum(-1), 1e-20)
        for k, (lo, hi) in REGIONS.items():
            m = (rad >= lo) & (rad < hi)
            den = np.maximum((rl * m).sum(-1), 1e-20)
            out[f"{k}_signed"] = ((pl - rl) * m).sum(-1) / den
            out[f"{k}_l1"] = (np.abs(pl - rl) * m).sum(-1) / den
            out[f"{k}_l1_of_total"] = (np.abs(pl - rl) * m).sum(-1) / tot
    return out


def load_states(path, R, n):
    with np.load(path) as z:
        raw = {k: z[k] for k in z.files}
    out = {}
    for arm in CM.ARMS:
        if f"{arm}_y" not in raw:
            continue
        st = {k: raw[f"{arm}_{k}"] for k in ("y", "g") if f"{arm}_{k}" in raw}
        out[arm] = [{k: v.reshape(R, n, *v.shape[1:])[r] for k, v in st.items()} for r in range(R)]
    truth = {k[6:]: v for k, v in raw.items() if k.startswith("truth_")}
    return out, truth


def contrast(EX, EY, mats, rng, nb=10000):
    sx, sy = EX[:, mats].sum((1, 2)), EY[:, mats].sum((1, 2))
    per = sx / sy - 1
    idx = rng.integers(0, len(mats), (nb, len(mats)))
    bx = EX[:, mats].sum(-1)[:, idx].sum(-1)
    by = EY[:, mats].sum(-1)[:, idx].sum(-1)
    med = np.median(bx / by - 1, 0)
    return {"per_start": per.tolist(), "median": float(np.median(per)), "lo": float(np.quantile(med, .025)),
            "hi": float(np.quantile(med, .975))}


def jac_rank(arm, st, ev, m, n_lanes=64):
    """Numerical Jacobian of the reference-lane response w.r.t. all trainable parameters of one material (float64)."""
    y = torch.as_tensor(st["y"][m:m + 1], dtype=torch.float64, device=CM.DEV)
    g = torch.as_tensor(st["g"][m:m + 1], dtype=torch.float64, device=CM.DEV) if arm != "A" else None
    wi = torch.as_tensor(ev["wi"][m, :, :n_lanes].reshape(-1, 3), dtype=torch.float64, device=CM.DEV)
    wo = torch.as_tensor(ev["wo"][m, :, :n_lanes].reshape(-1, 3), dtype=torch.float64, device=CM.DEV)
    sizes = [58] + ([g.numel()] if g is not None else [])

    def f(x):
        yy = x[:58].reshape(1, 58)
        p, _ = CM.decode("A", {"y": yy}, torch.float64)
        q = {k: v.expand(wi.shape[0], *v.shape[1:]) for k, v in p.items()}
        if g is None:
            return CL.eval_closure("ER", q, wi, wo).reshape(-1)
        gg = x[58:].reshape(1, *g.shape[1:]).expand(wi.shape[0], *g.shape[1:])
        return CM.eval_sdm(q, gg, wi, wo).reshape(-1)
    x0 = torch.cat([y.reshape(-1)] + ([g.reshape(-1)] if g is not None else []))
    cols = []
    for i in range(x0.numel()):                                              # forward mode, one column per parameter
        v = torch.zeros_like(x0); v[i] = 1.0
        cols.append(torch.func.jvp(f, (x0,), (v,))[1])
    Jm = torch.stack(cols, 1).cpu().numpy()
    s = np.linalg.svd(Jm, compute_uv=False)
    r = s / s[0]
    return {"params": int(sum(sizes)), "rank_1e-10": int((r > 1e-10).sum()), "rank_1e-6": int((r > 1e-6).sum()),
            "rank_1e-3": int((r > 1e-3).sum())}


def timing(arm, st, n_lanes=1 << 22, reps=5):
    """Torch float32 evaluation time per call (one material, n_lanes lanes)."""
    rng = np.random.default_rng(0)
    d = rng.standard_normal((2, n_lanes, 3)); d[..., 2] = np.abs(d[..., 2]); d /= np.linalg.norm(d, axis=-1, keepdims=True)
    wi = torch.as_tensor(d[0], dtype=torch.float32, device=CM.DEV); wo = torch.as_tensor(d[1], dtype=torch.float32, device=CM.DEV)
    with torch.no_grad():
        p, g = CM.decode(arm, {k: v[:1] for k, v in st.items()}, torch.float32)
        q = {k: v.expand(n_lanes, *v.shape[1:]) for k, v in p.items()}
        gg = None if g is None else g.expand(n_lanes, *g.shape[1:])
        CM.lane_eval(arm, q, gg, wi, wo); torch.cuda.synchronize()
        t = []
        for _ in range(reps):
            t0 = time.perf_counter(); CM.lane_eval(arm, q, gg, wi, wo); torch.cuda.synchronize(); t.append(time.perf_counter() - t0)
    return float(np.median(t))


def dr_timing(arm, st, n_lanes=1 << 22, reps=5):
    mi, dr = CM.K.mi, CM.K.dr
    comp = GF.compile_states(st["y"][:1])
    comp = comp.cpu().numpy() if torch.is_tensor(comp) else np.asarray(comp)
    cl = CM.CD.DrClosure83("ER", comp[0]) if arm == "A" else CM.DrSDM(comp[0], st["g"][0])
    rng = np.random.default_rng(0)
    d = rng.standard_normal((2, n_lanes, 3)); d[..., 2] = np.abs(d[..., 2]); d /= np.linalg.norm(d, axis=-1, keepdims=True)
    V = lambda a: mi.Vector3f(*[mi.Float(np.ascontiguousarray(a[:, k], np.float32)) for k in range(3)])
    wi, wo = V(d[0]), V(d[1])
    t = []
    for _ in range(reps + 1):
        t0 = time.perf_counter()
        f = cl.eval(wi, wo, mi.Bool(True)); dr.eval(f); dr.sync_thread()
        t.append(time.perf_counter() - t0)
    return float(np.median(t[1:]))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root", type=Path)
    ap.add_argument("out", type=Path)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    t0 = time.monotonic()
    rows, raw, _, _, ref, seen = CM.inputs()
    n = len(rows)
    ref_sha = CM.sha(IO.ANGULAR)
    cohorts = {c: [i for i, r in enumerate(rows) if r["cohort"] == c] for c in ("development", "held_out")}
    ev, peak = CM.eval_reference(ref, n)
    ev["rad"] = CM.mirror_radius(ev["wi"], ev["wo"])
    fit = CM.load_set("fit_set", n)
    t1 = CM.load_set("polar_transfer_set", n)
    lo = np.repeat(np.arange(CM.G - 1), 2)
    t2 = CM.load_set("azimuth_transfer_set", n)
    fj = IO.read(args.root / "fit_primary/fit.json")
    if CM.sha(args.root / "fit_primary/states.npz") != fj["states_sha256"]:
        raise IO.Stop("STATES_HASH primary")
    R = len(fj["start_keys"])
    prim, _ = load_states(args.root / "fit_primary/states.npz", R, n)
    per = {}
    for arm, lst in prim.items():
        per[arm] = []
        for st in lst:
            e = {"eval": metrics(CM.predict(arm, st, ev["wi"], ev["wo"]), ev, ev["rad"]),
                 "fit": metrics(CM.predict(arm, st, fit["wi"], fit["wo"]), fit),
                 "polar_transfer": metrics(CM.predict(arm, st, t1["wi"], t1["wo"]), t1),
                 "azimuth_transfer": metrics(CM.predict(arm, st, t2["wi"], t2["wo"]), t2)}
            e["eval"]["peak_signed"] = (CM.predict(arm, st, peak["wi"], peak["wo"]) @ LUMA).max(-1) / np.maximum(peak["ref"].max(-1), 1e-20) - 1
            per[arm].append(e)
        print("evaluated", arm, flush=True)
    arr = lambda arm, part, key: np.stack([per[arm][r][part][key] for r in range(R)])
    # reproduction of WL104J A
    with np.load(CM.J_STATES) as z:
        yJ = z["A_y"]
    repro = {"max_abs_state_diff_vs_WL104J_A": float(np.abs(np.stack([s["y"] for s in prim["A"]]).reshape(-1, 58) - yJ).max())}
    # ---------------- contrasts
    rng = np.random.default_rng(CM.P["seeds"]["bootstrap"])
    pairs = {"T-A": ("T", "A"), "P-T": ("P", "T"), "P-A": ("P", "A")}
    con = {}
    for name, (x, y) in pairs.items():
        c = {}
        for part, key in (("eval", "el1"), ("eval", "el1_a"), ("eval", "el1_b"), ("eval", "eshape"), ("eval", "ealb"), ("fit", "el1"),
                          ("polar_transfer", "el1"), ("azimuth_transfer", "el1")):
            c[f"{part}:{key}"] = {coh: contrast(arr(x, part, key), arr(y, part, key), ms, rng) for coh, ms in cohorts.items()}
        con[name] = c
    # concealed regression checks (P vs A, median start)
    elA, elP = np.median(arr("A", "eval", "el1"), 0), np.median(arr("P", "eval", "el1"), 0)
    mat_ratio = elP.mean(1) / elA.mean(1) - 1
    grp_ratio = {coh: (elP[ms].reshape(len(ms), CM.G, 2).sum((0, 2)) / elA[ms].reshape(len(ms), CM.G, 2).sum((0, 2)) - 1).tolist()
                 for coh, ms in cohorts.items()}
    regress = {"material_relative_change": mat_ratio.tolist(), "group_relative_change": grp_ratio,
               "concealed_regression": bool((mat_ratio > 0.05).any() or any(np.array(v).max() > 0.05 for v in grp_ratio.values()))}
    noise = {}
    for coh, ms in cohorts.items():
        nz = np.abs(arr("A", "eval", "el1_a") - arr("A", "eval", "el1_b"))[:, ms] / 2
        noise[coh] = {"ratio": float(nz.mean() / arr("A", "eval", "el1")[:, ms].mean())}
        noise[coh]["resolves"] = noise[coh]["ratio"] <= 0.0125
    # ---------------- convergence
    conv = {}
    for arm in CM.ARMS:
        lc = fj[arm]["last_2000_relative_change"]
        conv[arm] = {"fraction_units_le_0.5pct": float((np.array(lc) <= 0.005).mean()) if lc is not None else 0.0,
                     "max": float(np.max(lc)) if lc is not None else None}
    xj = IO.read(args.root / "fit_extended/fit.json")
    g0 = fj["start_keys"].index("generic")
    obj = {arm: np.array(fj[arm]["final_objective"])[g0] for arm in CM.ARMS}
    ext = {arm: np.array(xj[arm]["final_objective"])[0] for arm in CM.ARMS}
    tc = {}
    for nm, (x, y) in pairs.items():
        a12, a24 = obj[x].sum() / obj[y].sum() - 1, ext[x].sum() / ext[y].sum() - 1
        tc[nm] = {"12k": float(a12), "24k": float(a24), "abs_change_pp": float(abs(a24 - a12) * 100)}
    xs, _ = load_states(args.root / "fit_extended/states.npz", 1, n)
    ext_eval = {arm: float(CM.wl82_terms(CM.predict(arm, xs[arm][0], ev["wi"], ev["wo"]), ev["mean"], ev["w"])["el1"].mean()) for arm in xs}
    resolved = all(v["fraction_units_le_0.5pct"] >= 0.9 for v in conv.values()) and all(v["abs_change_pp"] <= 3 for v in tc.values())
    convergence = {"last_2000": conv, "extended_training_contrast": tc, "extended_heldout_el1_mean": ext_eval, "resolved": bool(resolved)}
    # ---------------- synthetic controls
    syn = {}
    for ctl in ("S0", "ST", "SP"):
        d = args.root / f"synth_{ctl}"
        sj = IO.read(d / "fit.json")
        if CM.sha(d / "states.npz") != sj["states_sha256"]:
            raise IO.Stop(f"STATES_HASH {ctl}")
        ss, truth = load_states(d / "states.npz", 1, n)
        tgt = CM.predict(sj["truth_arm"], truth, ev["wi"], ev["wo"])
        e = {arm: CM.wl82_terms(CM.predict(arm, ss[arm][0], ev["wi"], ev["wo"]), tgt, ev["w"])["el1"] for arm in ss}
        m = {arm: float(v.mean()) for arm, v in e.items()}
        s = {"heldout_el1_mean": m, "per_material": {arm: v.mean(1).tolist() for arm, v in e.items()},
             "training_objective": {arm: sj[arm]["final_objective"] for arm in ss},
             "truth_objective_on_noisy_target": sj["truth_objective_on_noisy_target"],
             "gate_stats": {arm: sj[arm].get("gate_stats") for arm in ss}}
        if ctl == "S0":
            s["pass"] = bool(abs(m["T"] - m["A"]) <= 1e-3 and abs(m["P"] - m["A"]) <= 1e-3)
        elif ctl == "ST":
            s["pass"] = bool(m["T"] <= 0.5 * m["A"] and m["P"] <= 0.5 * m["A"])
        else:
            s["pass"] = bool(m["P"] <= 0.5 * m["T"])
        syn[ctl] = s
    tests = IO.read(args.root / "tests.json")
    tfit = IO.read(args.root / "tests_fitted.json")
    # ---------------- labels
    coh = list(cohorts)
    qual = {"baseline_identity": tests["passed_parts"]["baseline_identity"],
            "reciprocity": tests["passed_parts"]["reciprocity_passivity"] and all(
                v["reciprocity_interior"]["f_rel"] <= 1e-12 and v["reciprocity_interior"]["cos_identity_rel"] <= 1e-12 for v in tfit["fitted"].values()),
            "passivity": tests["passed_parts"]["reciprocity_passivity"] and all(
                v["envelope"]["min_F"] >= 0 and v["envelope"]["max_excess_over_base_rel"] <= 1e-15 and v["mc_albedo"]["max_albedo_minus_4se"] <= 1
                for v in tfit["fitted"].values()),
            "gradients_trainability_parity": tests["passed_parts"]["gradients"] and tests["passed_parts"]["trainability"] and tests["passed_parts"]["drjit_parity"],
            "synthetic_controls": all(v["pass"] for v in syn.values()) and tests["passed_parts"]["synthetic_representability"],
            "reference_resolves": all(v["resolves"] for v in noise.values())}
    qualified = all(qual.values()) and resolved
    cpa, cpt = con["P-A"], con["P-T"]
    gain_pass = all(cpa["eval:el1"][c]["median"] <= -0.10 and cpa["eval:el1"][c]["hi"] < 0 and cpa["eval:el1_a"][c]["median"] < 0 and
                    cpa["eval:el1_b"][c]["median"] < 0 and sum(x < 0 for x in cpa["eval:el1"][c]["per_start"]) >= 3 for c in coh) \
        and not regress["concealed_regression"]
    gain = "PASS" if (gain_pass and qualified) else ("FAIL" if qualified else "INCONCLUSIVE")
    pc_sup = all(cpt["eval:el1"][c]["median"] <= -0.05 and cpt["eval:el1"][c]["hi"] < 0 and cpt["eval:el1_a"][c]["median"] < 0 and
                 cpt["eval:el1_b"][c]["median"] < 0 and sum(x < 0 for x in cpt["eval:el1"][c]["per_start"]) >= 3 for c in coh)
    per_comp = "SUPPORTED" if (pc_sup and qualified) else ("NOT_SUPPORTED" if qualified else "INCONCLUSIVE")
    tparts = [cpa[f"{p}:el1"][c] for p in ("polar_transfer", "azimuth_transfer") for c in coh]
    if not qualified:
        transfer = "INCONCLUSIVE"
    elif all(t["median"] < 0 and t["hi"] < 0 for t in tparts):
        transfer = "PASS"
    elif any(t["median"] >= 0 for t in tparts):
        transfer = "FAIL"
    else:
        transfer = "INCONCLUSIVE"
    labels = {"qualification": qual, "optimization_resolved": bool(resolved), "angular_fidelity_gain": gain,
              "per_component_benefit": per_comp, "heldout_directional_transfer": transfer,
              "scene_stage_eligible": bool(gain == "PASS")}
    # ---------------- rank, runtime, storage
    best = {arm: np.argmin(np.array(fj[arm]["final_objective"]), 0) for arm in CM.ARMS}
    bst = {arm: {k: np.stack([prim[arm][best[arm][m]][k][m] for m in range(n)]) for k in prim[arm][0]} for arm in CM.ARMS}
    rank = {arm: [jac_rank(arm, bst[arm], ev, m) for m in (0, 2, 4, 8)] for arm in CM.ARMS}
    rank_id = {arm: [jac_rank(arm, {"y": bst["A"]["y"], "g": CM.identity_gates(n, arm)}, ev, m) for m in (0, 2)] for arm in ("T", "P")}
    cost = {arm: {"torch_float32_seconds_per_4M_lanes": timing(arm, bst[arm]), "drjit_seconds_per_4M_lanes": dr_timing(arm, bst[arm]),
                  "state_floats": 69 + 15 * CM.NJ[arm], "added_bytes_float32": 4 * 15 * CM.NJ[arm],
                  "fit_seconds_4starts": fj[arm]["seconds"], "fit_peak_allocated_bytes": fj[arm]["peak_allocated_bytes"]} for arm in CM.ARMS}
    # ---------------- figure data
    lanes = {"ref_mean": ev["mean"], "w": ev["w"], "wi": ev["wi"], "wo": ev["wo"], "peak_wi": peak["wi"], "peak_wo": peak["wo"],
             "peak_ref": peak["ref"]}
    for arm in CM.ARMS:
        lanes[f"pred_{arm}"] = CM.predict(arm, bst[arm], ev["wi"], ev["wo"])
        lanes[f"peak_{arm}"] = CM.predict(arm, bst[arm], peak["wi"], peak["wo"]) @ LUMA
    np.savez_compressed(args.out / "lanes.npz", **{k: np.asarray(v, np.float32) for k, v in lanes.items()})
    gates = {arm: [prim[arm][r]["g"].tolist() for r in range(R)] for arm in ("T", "P")}
    # component energy share (best start) to show which gates act on non-negligible components
    share = {}
    for arm in ("A", "P"):
        Cg = CM.comp_responses(arm, bst[arm], ev["wi"][:, :, :256], ev["wo"][:, :, :256])
        en = ((Cg @ LUMA) * ev["w"][:, :, :256, None]).sum(2)
        share[arm] = (en / np.maximum(en.sum(-1, keepdims=True), 1e-300)).tolist()
    tables = {arm: {k: np.median(arr(arm, "eval", k), 0).tolist() for k in per[arm][0]["eval"]} for arm in CM.ARMS}
    for arm in CM.ARMS:
        tables[arm]["el1_per_start"] = arr(arm, "eval", "el1").tolist()
        tables[arm]["fit_el1"] = np.median(arr(arm, "fit", "el1"), 0).tolist()
        tables[arm]["polar_transfer_el1"] = np.median(arr(arm, "polar_transfer", "el1"), 0).tolist()
        tables[arm]["azimuth_transfer_el1"] = np.median(arr(arm, "azimuth_transfer", "el1"), 0).tolist()
    if CM.sha(IO.ANGULAR) != ref_sha:
        raise IO.Stop("EVALUATION_REFERENCE_CHANGED")
    res = {"materials": rows, "cohorts": cohorts, "probe_polar": CM.POLAR.tolist(), "start_keys": fj["start_keys"], "contrasts": con,
           "regression": regress, "reference_resolution": noise, "convergence": convergence, "synthetic": syn, "labels": labels,
           "reproduction": repro, "tables": tables, "best_start": {k: v.tolist() for k, v in best.items()}, "jacobian_rank": rank,
           "jacobian_rank_at_identity": rank_id, "cost": cost, "gates": gates, "component_share": share,
           "fitted_contracts_passed": tfit["passed"], "commit": CM.commit(), "protocol_sha256": CM.sha(CM.HERE / "protocol.json"),
           "inputs_sha256": seen, "lanes_sha256": CM.sha(args.out / "lanes.npz"), "seconds": time.monotonic() - t0}
    CM.write(args.out / "eval.json", res)
    print("eval done", labels["angular_fidelity_gain"], labels["per_component_benefit"], labels["heldout_directional_transfer"], flush=True)


if __name__ == "__main__":
    main()
