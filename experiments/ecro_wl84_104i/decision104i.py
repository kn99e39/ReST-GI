"""WL104I decision table from the frozen rules (protocol decision_rules). Reads outputs only.

Usage: python decision104i.py <primary_dir>
"""
import sys
from pathlib import Path

import numpy as np

import common104i as CM

IO = CM.IO
MARGIN = 0.05
SCENES = ("test83_iid_1_L2", "test83_hard_1_L1", "test83_iid_159_L2")


def main():
    root = Path(sys.argv[1])
    eq = IO.read(root / "equivalence/equivalence.json")
    idt = IO.read(root / "ident/ident.json")["results"]
    fit = IO.read(root / "fit/fit.json")
    ang = IO.read(root / "angular/angular.json")
    scenes = {s: IO.read(root / "scenes" / s / "analysis.json") for s in SCENES}
    nat = IO.read(root / "native/native_audit.json")
    d = {}
    d["response_family_equivalence"] = "PASS" if eq["passed"] else "FAIL"
    on = [v["original"]["exact_nulls"] for v in idt.values()]
    cn = [v["candidate"]["exact_nulls"] for v in idt.values()]
    gv = max(max(v["original"]["gauge_vector_response_relative"]) for v in idt.values())
    gauge_ok = all(x == 6 for x in on) and all(x == 0 for x in cn) and eq["worst"]["orbit_infinitesimal"] <= 1e-9 \
        and eq["worst"]["orbit_finite"] <= 1e-9
    d["exact_gauge_removal"] = "PASS" if gauge_ok else "FAIL"
    tang = min(v["tangent"]["min_principal_cosine"] for v in idt.values())
    d["structural_identifiability_improvement"] = "PASS" if gauge_ok and tang >= 1 - 1e-9 else "FAIL"
    prim = {}
    for c in ("development", "held_out"):
        prim[f"angular_el1_{c}"] = ang["summary"][c]["all"]["el1"]
    for s, a in scenes.items():
        for ev in ("e1", "e2", "total"):
            prim[f"scene_{s}_{ev}"] = a["primary_fixed"][ev]
    if all(v["hi"] <= MARGIN for v in prim.values()):
        d["training_fidelity_preservation"] = "PASS"
    elif any(v["lo"] > MARGIN for v in prim.values()):
        d["training_fidelity_preservation"] = "FAIL"
    else:
        d["training_fidelity_preservation"] = "INCONCLUSIVE"
    local = {f"{c}|{b}": ang["summary"][c][b]["el1"] for c in ("development", "held_out") for b in ang["summary"][c]
             if isinstance(ang["summary"][c][b], dict) and "el1" in ang["summary"][c][b] and ang["summary"][c][b]["el1"].get("local_regression_flag")}
    # practical benefit
    sp = {c: ang["summary"][c]["response_spread"]["median_ratio_B_over_A"] for c in ("development", "held_out")}
    keys = fit["start_keys"]
    idx = [keys.index(str(s)) for s in CM.SEEDS]
    fa = np.asarray(fit["arms"]["A1"]["final_objective"])[idx]       # [seeds, materials]
    fb = np.asarray(fit["arms"]["B1"]["final_objective"])[idx]
    ratio_m = np.median(fb, 0) / np.median(fa, 0)
    cost = fit["arms"]["B1"]["seconds"] / fit["arms"]["A1"]["seconds"]
    benefits, regressions = [], []
    if all(v <= 0.8 for v in sp.values()):
        benefits.append("stability")
    if any(v >= 1.25 for v in sp.values()):
        regressions.append("stability")
    if np.median(ratio_m) <= 0.99 and (ratio_m < 1).sum() >= 6:
        benefits.append("optimization")
    if np.median(ratio_m) >= 1.01 and (ratio_m > 1).sum() >= 6:
        regressions.append("optimization")
    if cost <= 0.9:
        benefits.append("cost")
    if cost >= 1.1:
        regressions.append("cost")
    d["practical_benefit"] = "PASS" if benefits and not regressions else ("INCONCLUSIVE" if benefits and regressions else "FAIL")
    d["native_proposal_qualification"] = nat["label"]
    core = (d["response_family_equivalence"], d["exact_gauge_removal"], d["training_fidelity_preservation"])
    d["candidate_viability"] = "SUPPORTED" if all(x == "PASS" for x in core) else ("NOT SUPPORTED" if "FAIL" in core else "INCONCLUSIVE")
    out = {"decision": d, "primary_metrics": prim, "local_regression_flags": local, "practical": {
        "spread_ratio_B_over_A": sp, "final_objective_ratio_per_material": ratio_m.tolist(),
        "final_objective_ratio_median": float(np.median(ratio_m)), "fit_wall_time_ratio": cost,
        "benefits": benefits, "regressions": regressions},
        "exact_nulls": {"original": on, "candidate": cn}, "gauge_vector_response_max": gv, "tangent_min_cosine": tang,
        "note": "Diagnostic table under the frozen rules. Viability does not mean adoption; adoption is a human decision."}
    IO.write(root / "decision.json", out)
    print(d, flush=True)


if __name__ == "__main__":
    main()
