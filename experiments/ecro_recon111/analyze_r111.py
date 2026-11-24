"""WL111 frozen analysis: equal-spp and equal-E2E comparisons, routes A / B per regime, absolute contract C, dynamic
safety D1-D6, reference qualification and the verdict mapping of the protocol. numpy only (reads the evidence JSON).

python analyze_r111.py ROOT OUT_JSON
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

import common_r111 as C

D = C.P["decision"]
ARMS_SINGLE = ("R0", "R3", "S0", "S1")
REAL = (0, 1)
DROIS = C.DECISION_ROIS
BOOT = C.P["seeds"]["bootstrap"]
DRAWS = 10000


def med(v):
    return float(np.median(np.asarray(v, np.float64)))


# ------------------------------------------------------------------------------------------------ timing
def load_timing(root):
    t = {}
    for name in ("static", "camera_Q2", "continuous_Q4"):
        p = root / "timing" / f"{name}.json"
        if p.exists():
            t[name] = C.read(p)["times_ms"]
    return t


def e2e(t, regime, arm, spp):
    """Median E2E and the raw samples; candidate arms of both realizations share the S0 / ST0 timing (same network)."""
    a = {"S1": "S0", "ST1": "ST0"}.get(arm, arm)
    v = t.get(regime, {}).get(f"{a}_{spp}")
    return (med(v), np.asarray(v, np.float64)) if v else (None, None)


# ------------------------------------------------------------------------------------------------ static quality tables
def static_tables(root):
    """-> data[view][arm][spp][frame_label][roi] = dict(mse list over replicates, var_ref, ref2), flicker, reset checks."""
    data, flick, resets = {}, {}, []
    for f in sorted((root / "static").glob("*_r*.json")):
        r = C.read(f)
        v, spp = r["view"], r["spp"]
        for key, met in r["metrics"].items():
            arm, fr = key.rsplit("_f", 1)
            fr = int(fr)
            label = f"f{fr}"
            if arm in ARMS_SINGLE:
                label = "single"                 # frames 1 and 64 of a single-frame arm are iid draws
            for roi, m in met.items():
                d = data.setdefault(v, {}).setdefault(arm, {}).setdefault(spp, {}).setdefault(label, {}).setdefault(
                    roi, {"mse": [], "bias": [], "var_ref": m["var_ref"], "ref2": m["ref2"], "grad_ratio": [], "grad_error": []})
                d["mse"].append(m["mse"])
                d["bias"].append(m["bias"])
                if "grad_ratio" in m:
                    d["grad_ratio"].append(m["grad_ratio"])
                    d["grad_error"].append(m["grad_error"])
        for arm, val in r["flicker"].items():
            if val is not None:
                flick.setdefault(v, {}).setdefault(arm, {}).setdefault(spp, []).append(val)
        resets += list(r["reset_check"].values())
    return data, flick, resets


def cell(data, v, arm, spp, label, roi):
    try:
        return data[v][arm][spp][label][roi]
    except KeyError:
        return None


def rmse_of(c, idx=None):
    mse = np.asarray(c["mse"], np.float64)
    if idx is not None:
        mse = mse[idx]
    return math.sqrt(max(mse.mean() - c["var_ref"], 1e-30) / c["ref2"])


def gm_ratio(data, views, num, den, roi, rng=None):
    """GM over views of RMSE(num) / RMSE(den); num / den = (arm, spp, label). With rng: one bootstrap draw (views, then
    replicates within view, independently per arm)."""
    vs = list(views)
    if rng is not None:
        vs = [vs[i] for i in rng.integers(0, len(vs), len(vs))]
    logs = []
    for v in vs:
        a = cell(data, v, *num, roi)
        b = cell(data, v, *den, roi)
        if a is None or b is None:
            continue
        ia = rng.integers(0, len(a["mse"]), len(a["mse"])) if rng is not None else None
        ib = rng.integers(0, len(b["mse"]), len(b["mse"])) if rng is not None else None
        logs.append(math.log(rmse_of(a, ia) / rmse_of(b, ib)))
    return math.exp(np.mean(logs)) if logs else float("nan")


def ratio_ci(data, views, num, den, roi, seed_):
    pt = gm_ratio(data, views, num, den, roi)
    rng = np.random.default_rng(seed_)
    bs = np.array([gm_ratio(data, views, num, den, roi, rng) for _ in range(DRAWS // 10)])
    return pt, C.ci95(bs)


def mean_bias(data, views, arm, spp, label, roi):
    vals = [np.mean(cell(data, v, arm, spp, label, roi)["bias"]) for v in views if cell(data, v, arm, spp, label, roi)]
    return float(np.mean(vals)) if vals else None


def mean_grad(data, views, arm, spp, label, roi="silhouette"):
    vals = [np.mean(cell(data, v, arm, spp, label, roi)["grad_ratio"]) for v in views if cell(data, v, arm, spp, label, roi)]
    return float(np.mean(vals)) if vals else None


# ------------------------------------------------------------------------------------------------ routes
def time_ratio_ci(ta, tb, seed_):
    rng = np.random.default_rng(seed_)
    bs = [np.median(rng.choice(ta, len(ta))) / np.median(rng.choice(tb, len(tb))) for _ in range(2000)]
    return float(np.median(ta) / np.median(tb)), C.ci95(bs)


def routes(data, views, t, regime_t, cand, cand_label, anchor, grid, flicker=None, seed_=0):
    """Routes A and B for one realization's candidate arm (cand, cand_label) against R3 at `anchor` (label 'single')."""
    den = ("R3", anchor, "single")
    r3t, r3s = e2e(t, regime_t, "R3", anchor)
    out = {"anchor": anchor, "candidate": cand, "regime_timing": regime_t, "R3_e2e_ms": r3t, "points": {}}
    if r3t is None:
        out["A"] = out["B"] = {"pass": None, "reason": "R3 anchor timing missing"}
        return out
    pts = []
    for spp in grid:
        ct, cs = e2e(t, regime_t, cand, spp)
        if ct is None or cell(data, views[0], cand, spp, cand_label, "primary") is None:
            continue
        num = (cand, spp, cand_label)
        r, ci = ratio_ci(data, views, num, den, "primary", seed_ + spp)
        rois = {roi: ratio_ci(data, views, num, den, roi, seed_ + 100 + spp) for roi in DROIS}
        tr, tci = time_ratio_ci(cs, r3s, seed_ + 200 + spp)
        pts.append({"spp": spp, "e2e_ms": ct, "rmse_ratio": r, "ci95": ci, "rois": {k: {"ratio": v[0], "ci95": v[1]} for k, v in rois.items()},
                    "time_ratio": tr, "time_ci95": tci})
    out["points"] = pts
    roi_ok = lambda p: all(p["rois"][k]["ratio"] <= 1.05 and p["rois"][k]["ci95"][1] <= 1.10 for k in DROIS if p["rois"][k]["ratio"] == p["rois"][k]["ratio"])
    art_ok = lambda p: True if flicker is None else flicker(p["spp"])
    elig = [p for p in pts if p["ci95"][1] <= 1.05]
    if elig:
        c = min(elig, key=lambda p: p["e2e_ms"])
        okA = c["time_ratio"] <= 0.85 and c["time_ci95"][1] < 1.0 and roi_ok(c) and art_ok(c)
        out["A"] = {"pass": bool(okA), "point": c["spp"], "time_ratio": c["time_ratio"], "rmse_ratio": c["rmse_ratio"],
                    "roi_ok": roi_ok(c), "artifact_ok": art_ok(c)}
    else:
        out["A"] = {"pass": False, "reason": "no candidate point with primary RMSE ratio upper95 <= 1.05"}
    within = [p for p in pts if p["e2e_ms"] <= r3t]
    if within:
        c = max(within, key=lambda p: p["spp"])
        Dp = 1 - c["rmse_ratio"]
        lo = 1 - c["ci95"][1]
        okB = Dp >= 0.10 and lo > 0 and roi_ok(c) and art_ok(c)
        out["B"] = {"pass": bool(okB), "point": c["spp"], "D": Dp, "D_lower95": lo, "roi_ok": roi_ok(c), "artifact_ok": art_ok(c)}
    else:
        out["B"] = {"pass": None, "reason": "no candidate point within the R3 anchor E2E (not evaluable)"}
    return out


# ------------------------------------------------------------------------------------------------ sequences
def seq_tables(root, name):
    """-> per arm per spp: per replicate list of per-frame (mse, var_ref, ref2) for each ROI; plus raw records."""
    tab, recs = {}, []
    for f in sorted((root / "seq").glob(f"{name}_spp*_r*.json")):
        r = C.read(f)
        recs.append(r)
        for k, row in r["frames"].items():
            for arm, met in row.get("metrics", {}).items():
                for roi, m in met.items():
                    tab.setdefault(arm, {}).setdefault(r["spp"], {}).setdefault(roi, {}).setdefault(r["rep"], []).append(
                        (m["mse"], m["var_ref"], m["ref2"], m["bias"], m.get("grad_ratio")))
    return tab, recs


def seq_rmse(tab, arm, spp, roi, rng=None):
    reps = tab.get(arm, {}).get(spp, {}).get(roi)
    if not reps:
        return None
    keys = sorted(reps)
    if rng is not None:
        keys = [keys[i] for i in rng.integers(0, len(keys), len(keys))]
    rows = []
    for k in keys:
        fr = reps[k]
        if rng is not None:
            fr = [fr[i] for i in rng.integers(0, len(fr), len(fr))]
        rows += fr
    a = np.asarray([x[:3] for x in rows], np.float64)
    return math.sqrt(max((a[:, 0] - a[:, 1]).mean(), 1e-30) / a[:, 2].mean())


def seq_ratio(tab, num, den, roi, seed_):
    pt = seq_rmse(tab, *num, roi) / seq_rmse(tab, *den, roi)
    rng = np.random.default_rng(seed_)
    bs = []
    for _ in range(1000):
        a, b = seq_rmse(tab, *num, roi, rng), seq_rmse(tab, *den, roi, rng)
        bs.append(a / b)
    return pt, C.ci95(bs)


def seq_routes(tab, t, regime_t, cand, anchor, grid, seed_):
    r3t, r3s = e2e(t, regime_t, "R3", anchor)
    out = {"anchor": anchor, "candidate": cand, "R3_e2e_ms": r3t, "points": []}
    if r3t is None or seq_rmse(tab, "R3", anchor, "primary") is None:
        out["A"] = out["B"] = {"pass": None, "reason": "missing R3 anchor"}
        return out
    for spp in grid:
        ct, cs = e2e(t, regime_t, cand, spp)
        if ct is None or seq_rmse(tab, cand, spp, "primary") is None:
            continue
        r, ci = seq_ratio(tab, (cand, spp), ("R3", anchor), "primary", seed_ + spp)
        rois = {roi: seq_ratio(tab, (cand, spp), ("R3", anchor), roi, seed_ + 50 + spp) for roi in DROIS
                if seq_rmse(tab, cand, spp, roi) is not None}
        tr, tci = time_ratio_ci(cs, r3s, seed_ + 300 + spp)
        out["points"].append({"spp": spp, "e2e_ms": ct, "rmse_ratio": r, "ci95": ci, "time_ratio": tr, "time_ci95": tci,
                              "rois": {k: {"ratio": v[0], "ci95": v[1]} for k, v in rois.items()}})
    roi_ok = lambda p: all(v["ratio"] <= 1.05 and v["ci95"][1] <= 1.10 for v in p["rois"].values())
    elig = [p for p in out["points"] if p["ci95"][1] <= 1.05]
    if elig:
        c = min(elig, key=lambda p: p["e2e_ms"])
        out["A"] = {"pass": bool(c["time_ratio"] <= 0.85 and c["time_ci95"][1] < 1.0 and roi_ok(c)), "point": c["spp"],
                    "time_ratio": c["time_ratio"], "rmse_ratio": c["rmse_ratio"], "roi_ok": roi_ok(c)}
    else:
        out["A"] = {"pass": False, "reason": "no candidate point with primary RMSE ratio upper95 <= 1.05"}
    within = [p for p in out["points"] if p["e2e_ms"] <= r3t]
    if within:
        c = max(within, key=lambda p: p["spp"])
        Dp, lo = 1 - c["rmse_ratio"], 1 - c["ci95"][1]
        out["B"] = {"pass": bool(Dp >= 0.10 and lo > 0 and roi_ok(c)), "point": c["spp"], "D": Dp, "D_lower95": lo, "roi_ok": roi_ok(c)}
    else:
        out["B"] = {"pass": None, "reason": "no candidate point within the R3 anchor E2E (not evaluable)"}
    return out


def dynamic_safety(root, t):
    res = {}
    # D1 / D4 (post-edit)
    d1, eq = [], []
    for name in ("Q3", "Q4", "Q5"):
        _, recs = seq_tables(root, name)
        for r in recs:
            for k, row in r["frames"].items():
                for ri in REAL:
                    d1.append(row[f"ST{ri}_age_max"] <= row["since_edit"] + 1)
                    if row.get(f"ST{ri}_eq_S") is not None:
                        eq.append(row[f"ST{ri}_eq_S"])
    res["D1_no_stale_history"] = {"pass": bool(d1) and all(d1) and (max(eq) <= 1e-6 if eq else False),
                                  "frames_checked": len(d1), "max_post_edit_rel_l1": max(eq) if eq else None}
    # D2
    _, recs = seq_tables(root, "Q2")
    dis = ret = 0
    for r in recs:
        for row in r["frames"].values():
            for ri in REAL:
                dis += row.get(f"ST{ri}_disoccluded", 0)
                ret += row.get(f"ST{ri}_disoccluded_retained", 0)
    res["D2_disocclusion"] = {"pass": dis > 0 and ret / dis <= 0.005, "disoccluded": dis, "retained": ret,
                              "fraction": ret / dis if dis else None}
    # D3 / D6
    tab2, recs2 = seq_tables(root, "Q2")
    d3 = {}
    for ri in REAL:
        tce_st, tce_s, slopes = [], [], []
        for r in recs2:
            for pair, row in r["temporal"].items():
                if f"ST{ri}" in row:
                    tce_st.append(row[f"ST{ri}"]["tce_raw_mse"])
                    tce_s.append(row[f"S{ri}"]["tce_raw_mse"])
                    if row[f"ST{ri}"]["slope"] is not None:
                        slopes.append(row[f"ST{ri}"]["slope"])
        if tce_st:
            ratio = math.sqrt(max(np.mean(tce_st), 1e-30) / max(np.mean(tce_s), 1e-30))
            sl = float(np.mean(slopes)) if slopes else None
            d3[f"r{ri}"] = {"tce_ratio_ST_over_S": ratio, "slope_ST": sl,
                            "pass": ratio <= 1.05 and sl is not None and 0.9 <= sl <= 1.1}
    res["D3_no_attenuation"] = {"pass": bool(d3) and all(v["pass"] for v in d3.values()), **d3}
    d6 = {}
    for ri in REAL:
        row = {}
        for spp in C.P["sequences"]["candidate_spp_grid_dynamic"]:
            for roi in ("primary", "silhouette", "sss_interior"):
                a, b = seq_rmse(tab2, f"ST{ri}", spp, roi), seq_rmse(tab2, f"S{ri}", spp, roi)
                if a and b:
                    row[f"{spp}_{roi}"] = a / b
        d6[f"r{ri}"] = {"ratios": row, "pass": bool(row) and all(v <= 1.05 for v in row.values())}
    res["D6_camera_motion_quality"] = {"pass": bool(d6) and all(v["pass"] for v in d6.values()), **d6}
    # D4 (static forced reset)
    _, _, resets = static_tables(root)
    res["D4_reset_regression"] = {"pass": bool(resets) and max(resets) <= 1e-6 and res["D1_no_stale_history"]["max_post_edit_rel_l1"] is not None
                                  and res["D1_no_stale_history"]["max_post_edit_rel_l1"] <= 1e-6,
                                  "max_static_reset_rel_l1": max(resets) if resets else None}
    # D5
    d5 = {}
    for spp in C.P["sequences"]["candidate_spp_grid_dynamic"]:
        a, _ = e2e(t, "continuous_Q4", "ST0", spp)
        b, _ = e2e(t, "continuous_Q4", "S0", spp)
        if a and b:
            d5[str(spp)] = {"ST": a, "S": b, "pass": a <= 1.05 * b + 0.2}
    res["D5_latency_continuous"] = {"pass": bool(d5) and all(v["pass"] for v in d5.values()), **d5}
    res["all_pass"] = all(v["pass"] for k, v in res.items() if k.startswith("D"))
    return res


# ------------------------------------------------------------------------------------------------ main
def main(root, out):
    root = Path(root)
    t = load_timing(root)
    data, flick, resets = static_tables(root)
    sealed = C.P["views"]["sealed"]["names"]
    hist = C.P["views"]["historical"]
    grid = C.P["static_protocol"]["spp_grid"]
    dgrid = C.P["sequences"]["candidate_spp_grid_dynamic"]
    warm = f"f{C.P['static_protocol']['recorded_frames'][-1]}"
    res = {"protocol_sha256": C.sha(C.PROTOCOL), "regimes": {}, "equal_spp": {}}

    def flicker_ok(ri, anchor):
        def f(spp):
            a = [np.mean(flick[v][f"ST{ri}"][spp]) for v in sealed if spp in flick.get(v, {}).get(f"ST{ri}", {})]
            b = [np.mean(flick[v]["R3"][anchor]) for v in sealed if anchor in flick.get(v, {}).get("R3", {})]
            return bool(a and b and np.mean(a) <= 1.05 * np.mean(b))
        return f

    for anchor in (4, 16, 32):
        for ri in REAL:
            key = f"anchor{anchor}_r{ri}"
            res["regimes"].setdefault("static_warm", {})[key] = routes(data, sealed, t, "static", f"ST{ri}", warm, anchor, grid,
                                                                       flicker_ok(ri, anchor), BOOT + 10 * anchor + ri)
            res["regimes"].setdefault("cold_reset", {})[key] = routes(data, sealed, t, "static", f"S{ri}", "single", anchor, grid,
                                                                      None, BOOT + 1000 + 10 * anchor + ri)
            res["regimes"].setdefault("static_warm_historical", {})[key] = routes(data, hist, t, "static", f"ST{ri}", warm, anchor,
                                                                                  grid, None, BOOT + 2000 + 10 * anchor + ri)
    tab4, _ = seq_tables(root, "Q4")
    tab2, _ = seq_tables(root, "Q2")
    for ri in REAL:
        res["regimes"].setdefault("continuous_edit", {})[f"anchor4_r{ri}"] = seq_routes(tab4, t, "continuous_Q4", f"ST{ri}", 4, dgrid, BOOT + 3000 + ri)
        res["regimes"].setdefault("camera_motion", {})[f"anchor4_r{ri}"] = seq_routes(tab2, t, "camera_Q2", f"ST{ri}", 4, dgrid, BOOT + 4000 + ri)
    # equal-spp tables (sealed and historical, cold / warm, every arm)
    for vs_name, vs in (("sealed", sealed), ("historical", hist)):
        tbl = {}
        for spp in grid:
            row = {}
            for arm, label in (("R0", "single"), ("R3", "single"), ("S0", "single"), ("S1", "single"), ("ST0", warm), ("ST1", warm),
                               ("ST0", "f1"), ("SVGF", warm), ("SVGF", "f1")):
                vals = [rmse_of(cell(data, v, arm, spp, label, "primary")) for v in vs if cell(data, v, arm, spp, label, "primary")]
                if vals:
                    row[f"{arm}_{label}"] = {"rmse_mean_over_views": float(np.mean(vals)),
                                             "bias_primary": mean_bias(data, vs, arm, spp, label, "primary"),
                                             "bias_silhouette": mean_bias(data, vs, arm, spp, label, "silhouette"),
                                             "grad_silhouette": mean_grad(data, vs, arm, spp, label)}
            tbl[str(spp)] = row
        res["equal_spp"][vs_name] = tbl
    # absolute contract C at routed (or 4-spp) points, sealed views
    cres = {}
    for ri in REAL:
        for regime, label, arm in (("static_warm", warm, f"ST{ri}"), ("cold_reset", "single", f"S{ri}")):
            rr = res["regimes"][regime][f"anchor4_r{ri}"]
            p = next((rr[x]["point"] for x in ("A", "B") if rr.get(x, {}).get("pass")), 4)
            c = {"point": p}
            c["bias"] = {roi: mean_bias(data, sealed, arm, p, label, roi) for roi in C.BIAS_ROIS}
            c["silhouette_grad_ratio"] = mean_grad(data, sealed, arm, p, label)
            c["boundary_vs_R3"] = {roi: ratio_ci(data, sealed, (arm, p, label), ("R3", 4, "single"), roi, BOOT + 5000 + ri)
                                   for roi in ("silhouette", "material_boundary")}
            c["pass"] = (all(b is not None and abs(b) <= 0.01 for b in c["bias"].values())
                         and c["silhouette_grad_ratio"] is not None and 0.8 <= c["silhouette_grad_ratio"] <= 1.2
                         and all(v[1][1] <= 1.05 for v in c["boundary_vs_R3"].values()))
            cres[f"{regime}_r{ri}"] = c
        # Q4
        rr = res["regimes"]["continuous_edit"][f"anchor4_r{ri}"]
        p = next((rr[x]["point"] for x in ("A", "B") if rr.get(x, {}).get("pass")), 4)
        reps = tab4.get(f"ST{ri}", {}).get(p, {})
        b = {roi: float(np.mean([x[3] for rep in reps.get(roi, {}).values() for x in rep])) if reps.get(roi) else None for roi in C.BIAS_ROIS}
        g = [x[4] for rep in reps.get("silhouette", {}).values() for x in rep if x[4] is not None]
        cres[f"continuous_edit_r{ri}"] = {"point": p, "bias": b, "silhouette_grad_ratio": float(np.mean(g)) if g else None,
                                          "pass": all(v is not None and abs(v) <= 0.01 for v in b.values())
                                          and bool(g) and 0.8 <= float(np.mean(g)) <= 1.2}
    res["absolute_contract_C"] = {"pass": all(v["pass"] for v in cres.values()), **cres}
    res["dynamic_safety_D"] = dynamic_safety(root, t)
    # reference qualification (sealed views and sequences) against the R3 anchor RMSE
    refq = C.read(root / "refs" / "refs.json") if (root / "refs" / "refs.json").exists() else {}
    r3rm = {v: rmse_of(cell(data, v, "R3", 4, "single", "primary")) for v in sealed if cell(data, v, "R3", 4, "single", "primary")}
    res["reference_qualification"] = {v: {"ref_se": refq.get(v, {}).get("primary"), "R3_4spp_rmse": r3rm.get(v),
                                          "ratio": (refq.get(v, {}).get("primary") or float("nan")) / r3rm[v] if v in r3rm else None}
                                      for v in sealed}
    res["reference_qualification"]["pass_10pct_of_R3"] = all((x["ratio"] or 1) <= 0.10 for k, x in res["reference_qualification"].items()
                                                             if isinstance(x, dict))
    # verdict
    def passes(regime, ri):
        r = res["regimes"][regime][f"anchor4_r{ri}"]
        return bool(r.get("A", {}).get("pass") or r.get("B", {}).get("pass"))
    qual = C.read(root / "refs" / "qualification.json") if (root / "refs" / "qualification.json").exists() else {}
    res["sequence_qualification"] = qual
    q4ok = bool(qual.get("Q4", {}).get("qualifies"))
    sw = all(passes("static_warm", ri) for ri in REAL)
    cr = all(passes("cold_reset", ri) for ri in REAL)
    ce = q4ok and all(passes("continuous_edit", ri) for ri in REAL)     # a non-qualifying Q4 cannot support a dynamic claim
    dsafe = res["dynamic_safety_D"]["all_pass"]
    cpass = res["absolute_contract_C"]["pass"]
    if not res["reference_qualification"]["pass_10pct_of_R3"]:
        verdict = "INCONCLUSIVE"
    elif sw and cr and ce and dsafe:
        verdict = "QUALITY_PERFORMANCE_SUPPORTED" if cpass else "FRONTIER_GAIN_BUT_QUALITY_UNQUALIFIED"
    elif sw and dsafe:
        verdict = "STATIC_ONLY_SUPPORTED"
    else:
        verdict = "NOT_SUPPORTED"
    res["verdict_inputs"] = {"q4_qualifies": q4ok, "static_warm": sw, "cold_reset": cr, "continuous_edit": ce, "dynamic_safety": dsafe, "absolute_contract": cpass}
    res["verdict"] = verdict
    C.write(Path(out), res)
    print("VERDICT", verdict, res["verdict_inputs"], flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
