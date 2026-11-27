"""WL111A frozen final analysis (protocol/recon111a.json "decision"). numpy only; reads OUT/final.

  integrity            replicate counts, seed uniqueness, graph == eager, devices
  qualification        static reference precision per decision view; edited-state qualification
  routes               WL111 analyze_r111.routes unchanged (anchor R3 at 4 spp) per regime and realization:
                       static (decision set K1-K6 + N1-N6, static timing), edited (qualifying states, edit timing),
                       historical (V0 / H1-H5, static timing; reported)
  mechanism            S111A_ri vs S111_ri at 4 spp on the decision set (GM primary RMSE ratio, bootstrap 95 %)
  absolute_contract_C  at the routed static operating point (else 4 spp), per realization
  tables               equal-spp, per view, per ROI (incl. contact), LOW / HIGH, bias / variance decomposition, T_PHYSICAL
  verdict              protocol mapping

python analyzeF_r111a.py [OUT_DIR]      -> OUT_DIR/analysis.json (default OUT/final)
"""
import math
import sys
from pathlib import Path

import numpy as np

import common_r111a as A
import common_r111 as C
import analyze_r111 as AZ

P = C.read(A.HERE / "protocol" / "recon111a.json")
GRID = P["quality_protocol"]["spp_grid"]
REAL = (0, 1)
ARMS = ("R0", "R3", "S111_r0", "S111_r1", "S111A_r0", "S111A_r1")
TIMING_ARM = {"R0": "R0", "R3": "R3", "S111_r0": "S111", "S111_r1": "S111", "S111A_r0": "S111A", "S111A_r1": "S111A"}
ROIS = ("primary", "sss_interior", "silhouette", "material_boundary", "control", "contact")
BOOT = 17990111


def load(root):
    recs = {}
    for f in sorted((root / "quality").glob("*_spp*.json")):
        r = C.read(f)
        recs[(r["key"], r["spp"])] = r
    return recs


def tables(recs, target="total"):
    """-> data[key][arm][spp]['single'][roi] in analyze_r111's layout (TOTAL vs T_ECRO, or vs T_PHYSICAL)."""
    data = {}
    for (key, spp), r in recs.items():
        for rep in r["reps"]:
            for arm in ARMS:
                met = rep[arm].get(target)
                if met is None:
                    continue
                for roi, m in met.items():
                    d = data.setdefault(key, {}).setdefault(arm, {}).setdefault(spp, {}).setdefault("single", {}).setdefault(
                        roi, {"mse": [], "bias": [], "var_ref": m["var_ref"], "ref2": m["ref2"], "grad_ratio": [], "grad_error": []})
                    d["mse"].append(m["mse"])
                    d["bias"].append(m["bias"])
                    if "grad_ratio" in m:
                        d["grad_ratio"].append(m["grad_ratio"])
                        d["grad_error"].append(m["grad_error"])
    return data


def timing(root):
    t = {}
    for name in ("static", "edit"):
        p = root / "timing" / f"{name}.json"
        if not p.exists():
            continue
        raw = C.read(p)["times_ms"]
        t[name] = {}
        for arm, ta in TIMING_ARM.items():
            for spp in GRID:
                if f"{ta}_{spp}" in raw:
                    t[name][f"{arm}_{spp}"] = raw[f"{ta}_{spp}"]
                    if f"{ta}_{spp}_update_part" in raw:
                        t[name][f"{arm}_{spp}_update_part"] = raw[f"{ta}_{spp}_update_part"]
    return t


def rm(data, v, arm, spp, roi="primary"):
    c = AZ.cell(data, v, arm, spp, "single", roi)
    return AZ.rmse_of(c) if c else None


def integrity(recs):
    seeds, ge, n, dev = [], {}, {}, set()
    for (key, spp), r in recs.items():
        n[f"{key}_{spp}"] = len(r["reps"])
        dev.add(r["device"])
        for rep in r["reps"]:
            seeds.append(rep["seed"])
            for a, x in rep.get("graph_vs_eager_rel_l1", {}).items():
                ge[a] = max(ge.get(a, 0.0), x)
    return {"items": len(recs), "replicates_min": min(n.values()) if n else 0, "seeds_unique": len(seeds) == len(set(seeds)),
            "n_seeds": len(seeds), "graph_vs_eager_max_rel_l1": ge, "devices": sorted(dev),
            "graph_eager_ok": all(x <= 1e-3 for x in ge.values())}


def static_qualification(recs, data, views):
    out = {}
    for v in views:
        r4 = recs.get((v, 4))
        if r4 is None:
            out[v] = {"qualifies": False, "reason": "missing"}
            continue
        se = r4["ref_precision"]["primary"]
        r3 = rm(data, v, "R3", 4)
        mins = min(x for a in ARMS if a != "R0" for s in GRID if (x := rm(data, v, a, s)) is not None)
        out[v] = {"ref_se": se, "R3_4spp_rmse": r3, "min_decision_rmse": mins, "ratio_R3": se / r3, "ratio_min": se / mins,
                  "qualifies": bool(se <= 0.10 * r3 and se <= 0.30 * mins)}
    return out


def edited_qualification(root, data):
    st = C.read(root / "states.json")
    out = {}
    for k, s in st.items():
        r3 = rm(data, k, "R3", 4)
        se = s["ref_precision"]["primary"]
        prec = r3 is not None and se <= 0.10 * r3
        chg = s["total_change"] >= 0.01 and s["total_change"] >= 10 * s["noise_total"]
        out[k] = {**{x: s[x] for x in ("total_change", "indirect_change", "noise_total", "pixels", "reference_key")},
                  "ref_se": se, "R3_4spp_rmse": r3, "precision_ok": bool(prec), "change_ok": bool(chg), "qualifies": bool(prec and chg)}
    return out


def equal_spp(data, views, recs):
    tab = {}
    for spp in GRID:
        row = {}
        for arm in ARMS:
            if not all(AZ.cell(data, v, arm, spp, "single", "primary") for v in views):
                continue
            e = {"rmse_mean_over_views": float(np.mean([rm(data, v, arm, spp) for v in views])),
                 "bias": {roi: AZ.mean_bias(data, views, arm, spp, "single", roi) for roi in ROIS},
                 "grad_silhouette": AZ.mean_grad(data, views, arm, spp, "single"),
                 "grad_material_boundary": AZ.mean_grad(data, views, arm, spp, "single", "material_boundary")}
            if arm != "R3":
                e["gm_ratio_vs_R3_same_spp"] = {roi: AZ.gm_ratio(data, views, (arm, spp, "single"), ("R3", spp, "single"), roi) for roi in ROIS}
                e["gm_ratio_vs_R3_4spp"] = {roi: AZ.gm_ratio(data, views, (arm, spp, "single"), ("R3", 4, "single"), roi) for roi in ROIS}
            row[arm] = e
        tab[str(spp)] = row
    return tab


def per_view(data, views):
    return {v: {str(spp): {arm: {roi: rm(data, v, arm, spp, roi) for roi in ROIS} for arm in ARMS} for spp in GRID} for v in views}


def components(recs, views):
    """GM over views of LOW / HIGH relative RMSE (debiased only where the reference has component variance) / R3 at the
    same spp, plus the mean component share of each arm's TOTAL squared error."""
    out = {}
    for spp in GRID:
        row = {}
        for arm in ARMS:
            e = {}
            for c in ("low", "high"):
                for roi in ("primary", "silhouette", "material_boundary", "sss_interior"):
                    lr = []
                    for v in views:
                        r = recs.get((v, spp))
                        if r is None or roi not in r["reps"][0][arm][c]:
                            continue
                        f = lambda a: math.sqrt(max(np.mean([x[a][c][roi]["mse"] for x in r["reps"]]) - (r["reps"][0][a][c][roi]["var_ref"] or 0.0), 1e-30)
                                                / r["reps"][0][a][c][roi]["ref2"])
                        lr.append(math.log(f(arm) / f("R3")))
                    if lr:
                        e[f"{c}_{roi}_gm_ratio_vs_R3"] = float(math.exp(np.mean(lr)))
            row[arm] = e
        out[str(spp)] = row
    return out


def decomposition(recs, views, spp=4):
    out = {}
    for arm in ARMS:
        e = {}
        for c in ("total", "low", "high"):
            for roi in ROIS:
                xs = [recs[(v, spp)]["decomposition"][arm][c].get(roi) for v in views if (v, spp) in recs]
                xs = [x for x in xs if x and x.get("rel_rmse") is not None]
                if xs:
                    e[f"{c}_{roi}"] = {"rel_rmse": float(np.mean([x["rel_rmse"] for x in xs])),
                                       "rel_bias_rms": float(np.mean([x["rel_bias_rms"] for x in xs])),
                                       "rel_std": float(np.mean([x["rel_std"] for x in xs])),
                                       "bias_share": float(np.mean([x["bias_share"] for x in xs if x["bias_share"] is not None]))}
        out[arm] = e
    return out


def physical(recs):
    data = tables(recs, "physical")
    out = {}
    for key in data:
        r = recs[(key, 4)]
        out[key] = {"phys_minus_ecro_rel": r["physical"]["phys_minus_ecro"], "phys_minus_ecro_signed": r["physical"]["phys_minus_ecro_signed"],
                    "phys_ref_precision": r["physical"]["ref_precision"],
                    "vs_T_PHYSICAL_4spp": {arm: {roi: rm(data, key, arm, 4, roi) for roi in ROIS} for arm in ARMS},
                    "vs_T_PHYSICAL_bias_4spp": {arm: {roi: float(np.mean(AZ.cell(data, key, arm, 4, "single", roi)["bias"]))
                                                      for roi in ROIS if AZ.cell(data, key, arm, 4, "single", roi)} for arm in ARMS}}
    return out


def timing_summary(root, t):
    out = {}
    for name in ("static", "edit"):
        if name in t:
            out[name] = {k: AZ.med(v) for k, v in t[name].items() if not k.endswith("update_part")}
            out[name + "_update_part"] = {k: AZ.med(v) for k, v in t[name].items() if k.endswith("update_part")}
    for name in ("components", "futility"):
        p = root / "timing" / f"{name}.json"
        if p.exists():
            out[name] = C.read(p)
            if name == "components":
                out[name] = {k: out[name]["components"][k] for k in ("median_ms", "torch_memory", "params")} | {
                    "conditions": out[name]["conditions"]}
    for name in ("static", "edit"):
        p = root / "timing" / f"{name}.json"
        if p.exists():
            out[f"{name}_conditions"] = C.read(p)["conditions"]
    return out


def contract_C(data, views, routes_static, ri):
    arm = f"S111A_r{ri}"
    rr = routes_static[arm]
    p = next((rr[x]["point"] for x in ("A", "B") if rr.get(x, {}).get("pass")), 4)
    c = {"point": p, "bias": {roi: AZ.mean_bias(data, views, arm, p, "single", roi) for roi in C.BIAS_ROIS},
         "bias_contact_descriptive": AZ.mean_bias(data, views, arm, p, "single", "contact"),
         "silhouette_grad_ratio": AZ.mean_grad(data, views, arm, p, "single"),
         "boundary_vs_R3": {roi: AZ.ratio_ci(data, views, (arm, p, "single"), ("R3", 4, "single"), roi, BOOT + 500 + ri)
                            for roi in ("silhouette", "material_boundary")}}
    c["pass"] = bool(all(b is not None and abs(b) <= 0.01 for b in c["bias"].values())
                     and c["silhouette_grad_ratio"] is not None and 0.8 <= c["silhouette_grad_ratio"] <= 1.2
                     and all(v[1][1] <= 1.05 for v in c["boundary_vs_R3"].values()))
    return c


def main(root):
    root = Path(root)
    recs = load(root)
    data = tables(recs)
    t = timing(root)
    V = P["views"]
    dec = V["sealed_wl111"] + V["sealed_new"]
    hist = V["historical"]
    res = {"protocol_sha256": C.sha(A.HERE / "protocol" / "recon111a.json"), "integrity": integrity(recs)}
    res["static_qualification"] = static_qualification(recs, data, dec)
    res["historical_reference_precision"] = static_qualification(recs, data, hist)
    eq = edited_qualification(root, data)
    res["edited_qualification"] = eq
    qual_states = [k for k, v in eq.items() if v["qualifies"]]
    dec_ok = all(v["qualifies"] for v in res["static_qualification"].values())
    # routes (anchor R3 at 4 spp), every candidate arm; S111 reported for comparison
    res["routes"] = {}
    for regime, views, treg, off in (("static", dec, "static", 0), ("historical", hist, "static", 100), ("edited", qual_states, "edit", 200)):
        if not views or treg not in t:
            res["routes"][regime] = {"evaluable": False, "views": views}
            continue
        res["routes"][regime] = {"views": views}
        for k, arm in enumerate(("S111A_r0", "S111A_r1", "S111_r0", "S111_r1")):
            res["routes"][regime][arm] = AZ.routes(data, views, t, treg, arm, "single", 4, GRID, None, BOOT + off + 10 * k)
    # mechanism
    res["mechanism"] = {}
    for ri in REAL:
        m = {}
        for spp in GRID:
            m[str(spp)] = {roi: AZ.ratio_ci(data, dec, (f"S111A_r{ri}", spp, "single"), (f"S111_r{ri}", spp, "single"), roi,
                                            BOOT + 1000 + 10 * spp + ri) for roi in ("primary",) + C.DECISION_ROIS + ("control", "contact")}
        m["pass_4spp"] = bool(m["4"]["primary"][1][1] < 1.0)
        res["mechanism"][f"r{ri}"] = m
    res["mechanism_edited"] = {f"r{ri}": AZ.ratio_ci(data, qual_states, (f"S111A_r{ri}", 4, "single"), (f"S111_r{ri}", 4, "single"),
                                                     "primary", BOOT + 1500 + ri) if qual_states else None for ri in REAL}
    # equal-spp anchors at 1 / 2 spp (R3 out-of-distribution diagnostics)
    res["ood_anchor_diagnostics"] = {f"spp{s}": {arm: AZ.ratio_ci(data, dec, (arm, s, "single"), ("R3", s, "single"), "primary", BOOT + 2000 + s)
                                                 for arm in ("S111_r0", "S111_r1", "S111A_r0", "S111A_r1")} for s in (1, 2)}
    res["contract_C"] = {f"r{ri}": contract_C(data, dec, res["routes"]["static"], ri) for ri in REAL} if "S111A_r0" in res["routes"]["static"] else None
    res["equal_spp"] = {"decision": equal_spp(data, dec, recs), "historical": equal_spp(data, hist, recs)}
    if qual_states:
        res["equal_spp"]["edited_qualifying"] = equal_spp(data, qual_states, recs)
    res["equal_spp"]["edited_all"] = equal_spp(data, list(eq), recs)
    res["per_view"] = per_view(data, dec + hist + list(eq))
    res["components"] = {"decision": components(recs, dec), "historical": components(recs, hist)}
    res["decomposition_4spp"] = {"decision": decomposition(recs, dec), "historical": decomposition(recs, hist),
                                 "edited_all": decomposition(recs, list(eq))}
    res["physical"] = physical(recs)
    res["timing"] = timing_summary(root, t)

    def passes(regime, ri):
        r = res["routes"].get(regime, {}).get(f"S111A_r{ri}")
        return bool(r and (r.get("A", {}).get("pass") or r.get("B", {}).get("pass")))

    route_static = all(passes("static", ri) for ri in REAL)
    mech = all(res["mechanism"][f"r{ri}"]["pass_4spp"] for ri in REAL)
    cpass = bool(res["contract_C"] and all(res["contract_C"][f"r{ri}"]["pass"] for ri in REAL))
    integ = res["integrity"]["graph_eager_ok"] and res["integrity"]["seeds_unique"]
    excl = all(C.read(root / "timing" / f"{n}.json")["exclusivity"] for n in ("static", "edit")) if t else False
    if not dec_ok or not integ or not excl:
        verdict = "INCONCLUSIVE"
    elif route_static:
        verdict = "QUALITY_PERFORMANCE_SUPPORTED" if cpass else "FRONTIER_GAIN_BUT_QUALITY_UNQUALIFIED"
    elif mech:
        verdict = "PARTIAL_MECHANISM_SUPPORT"
    else:
        verdict = "NOT_SUPPORTED"
    edited = {"qualifying_states": qual_states, "claim_allowed": len(qual_states) >= 4,
              "route_both": all(passes("edited", ri) for ri in REAL) if len(qual_states) >= 4 else None}
    res["verdict_inputs"] = {"decision_set_qualified": dec_ok, "integrity": integ, "timing_exclusivity_recorded": excl,
                             "static_route_both": route_static, "mechanism_both": mech, "contract_C_both": cpass, "edited": edited}
    res["verdict"] = verdict
    C.write(root / "analysis.json", res)
    print("VERDICT", verdict, res["verdict_inputs"], flush=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else A.OUT / "final")
