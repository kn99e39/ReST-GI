"""WL104K-C decision (protocol decision): collects Stage A, contracts, fit decision, independent test and scene analyses into
decision.json with the separate conclusions required by the brief. No threshold is set here that is not in protocol.json.

  python decision104kc.py <root>
"""
import sys
from pathlib import Path

import numpy as np

import common104kc as C

NEW = list(C.P.get("seeds", {}).get("scene_seed_bases", {}))
REGRESSION_SCENES = ("test83_iid_1_L2", "test83_hard_1_L1", "test83_iid_159_L2", "test83_hard_53_L2", "test83_block_186_L2")
NA = "NOT_AVAILABLE"


def rd(p):
    return C.read(p) if Path(p).exists() else None


def scene_labels(root):
    out = {}
    for s in list(NEW) + list(REGRESSION_SCENES):
        a = rd(root / "scenes" / s / "analysis104kc.json")
        if a is None:
            out[s] = None
            continue
        out[s] = {"identity_passed": a["identity_passed"], "new_independent": a["new_independent"],
                  "contrasts": {k: {ev: {q: v[ev].get(q) for q in ("point", "lo", "hi", "label", "x_rms", "o0_rms")}
                                    for ev in ("e1", "e2", "total")} for k, v in a.get("contrasts", {}).items()},
                  "absolute_objects_rms": a.get("absolute_objects_rms")}
    return out


def scene_verdict(sc, arm="CR"):
    qualified_new = [s for s in NEW if sc.get(s) and sc[s]["identity_passed"]]
    labs = {s: (sc[s]["contrasts"].get(arm, {}).get("total", {}).get("label") if sc.get(s) else None) for s in sc}
    regressed = [s for s, l in labs.items() if l == "REGRESSED"]
    improved_new = [s for s in qualified_new if labs.get(s) == "IMPROVED"]
    resolved_new = [s for s in qualified_new if labs.get(s) in ("IMPROVED", "NEUTRAL", "REGRESSED")]
    if regressed:
        v = "SCENE_NOT_SUPPORTED"
    elif not resolved_new:
        v = "SCENE_INCONCLUSIVE"
    elif improved_new:
        v = "SCENE_SUPPORTED_IMPROVED"
    else:
        v = "SCENE_PRESERVED"
    return {"verdict": v, "total_labels": labs, "regressed": regressed, "improved_new_independent": improved_new,
            "qualified_new_independent": qualified_new}


def main():
    root = Path(sys.argv[1])
    sa = rd(root / "stageA/stageA.json")
    con = rd(root / "contracts/fitted.json")
    fd = rd(root / "fit_decision/fit_decision.json")
    test = rd(root / "test_eval/test.json")
    data = rd(root / "data/data.json")
    sc = scene_labels(root)
    rules = C.P["decision"]
    out = {"protocol_sha256": C.sha(C.PROTOCOL), "commit": C.commit()}
    # 1 numerical qualification
    out["numerical_qualification"] = {"verdict": sa["verdict"] if sa else NA,
                                      "explanation": sa.get("explanation") if sa else None,
                                      "numerical_null_test": test.get("numerical_null") if test else None}
    qualified = bool(sa and sa["qualified"] and con and con.get("passed") and data and data["qualified"]["all"])
    out["qualified"] = qualified
    # 2 optimization adequacy and 3 objective feasibility (training data)
    if fd:
        cr = fd["training"]["CR"]
        frac_feas_final = cr["feasible_fraction_per_checkpoint"][-1]
        conv = cr["F_last_two_rel_change_median"]
        out["optimization_adequacy"] = {
            "CR_feasible_fraction_last_checkpoint": frac_feas_final, "CR_F_last_two_rel_change_median": conv,
            "CR_kkt_projected_gradient_median": cr["kkt_projected_gradient_median"], "CR_final_rho_max": cr["final_rho_max"],
            "CR_units_moved_from_warm_start": cr["units_moved_from_warm_start"],
            "converged": bool(frac_feas_final >= rules["optimization"]["feasible_fraction_min"]
                              and conv is not None and conv <= rules["optimization"]["F_change_max"]),
            "controls": {a: fd["training"][a].get("criterion_last_two_rel_change_median") for a in ("O0C", "E2W")}}
        frac = cr["units_F_le_0.9"] / cr["units"]
        out["objective_feasibility"] = {"units_core_reduced_ge_10pct_feasible": cr["units_F_le_0.9"], "units": cr["units"],
                                        "fraction": frac, "F_returned_median": cr["F_returned_median"],
                                        "verdict": ("FEASIBLE_IMPROVEMENT" if frac >= rules["objective_feasibility"]["fraction_min"]
                                                    else "LIMITED_FEASIBLE_IMPROVEMENT"),
                                        "training_terms_rel_change_median": cr["training_terms_rel_change_median"]}
        out["synthetic_feasibility"] = fd["truth_feasibility"]
    # 4 synthetic representation adequacy, 5 independent angular fidelity, 7 stability
    if test:
        s = test["synthetic"]["controls"]
        out["synthetic_adequacy"] = {"SYN_RECOVERY_CR_S": s["SYN_RECOVERY_CR_S"], "SYN_RECOVERY_CR_S_vs_O0C_S": s["SYN_RECOVERY_CR_S_vs_O0C_S"],
                                     "SYN_RECOVERY_CR_SD": s["SYN_RECOVERY_CR_SD"], "CR_S_ratio": s["CR_S"]["ratio"],
                                     "CR_SD_ratio": s["CR_SD"]["ratio"], "data_CR_SD_vs_CR_S_e_prof": s["data_CR_SD_vs_CR_S_e_prof"],
                                     "pooled_median": test["synthetic"]["pooled_median"], "instability": test["synthetic"]["instability"]}
        lab = test["labels"]
        out["independent_angular"] = {a: {k: v for k, v in lab[a].items() if k != "material_el1_change"} for a in lab}
        for a in out["independent_angular"]:
            if not qualified:
                out["independent_angular"][a]["label"] = "UNQUALIFIED"
        out["across_start_stability"] = {a: {"STABILITY_GAIN": lab[a]["STABILITY_GAIN"],
                                             "instability_change": test["contrasts"][a]["instability"]["all"]["median"]}
                                         for a in ("CR", "O0C", "E2W", "E2_B")}
    # 6 scene fidelity
    out["scenes"] = sc
    out["scene_fidelity"] = {a: scene_verdict(sc, a) for a in ("CR", "O0C", "E2_B")}
    # completion answers (evidence statements; no recommendation)
    ang = out.get("independent_angular", {}).get("CR", {}).get("label", NA)
    scv = out["scene_fidelity"]["CR"]["verdict"]
    if not qualified:
        q1 = "UNQUALIFIED"
    elif ang == "QUALIFIED_IMPROVEMENT" and scv in ("SCENE_SUPPORTED_IMPROVED", "SCENE_PRESERVED"):
        q1 = "YES_UNDER_TESTED_CONDITIONS"
    elif ang.startswith("QUALIFIED_IMPROVEMENT") and scv == "SCENE_INCONCLUSIVE":
        q1 = "ANGULAR_YES_SCENE_INCONCLUSIVE"
    elif ang in ("TRADE_OFF", "REGRESSION") or scv == "SCENE_NOT_SUPPORTED":
        q1 = "NO_UNDER_TESTED_CONDITIONS"
    else:
        q1 = "NOT_DEMONSTRATED"
    out["completion"] = {"Q1_improved_peak_with_preserved_broad_albedo_scene": q1, "angular_label_CR": ang, "scene_verdict_CR": scv}
    C.write(root / "decision.json", out)
    print("decision", {"qualified": qualified, "Q1": q1, "angular": ang, "scene": scv}, flush=True)


if __name__ == "__main__":
    main()
