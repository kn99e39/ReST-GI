"""WL104K frozen decision table (protocol.json) from eval.json and, if run, the scene analyses.

Usage: python decision104k.py <run_root>
"""
import sys
from pathlib import Path

import common104k as CM

SCENES = ("test83_iid_1_L2", "test83_hard_1_L1", "test83_iid_159_L2")


def main():
    root = Path(sys.argv[1])
    E = CM.read(root / "eval/eval.json")
    L = E["labels"]
    q = L["qualification"]
    scene = "NOT_RUN"
    scene_detail = None
    if L["scene_stage_eligible"]:
        an = {}
        for s in SCENES:
            p = root / "scenes" / s / "analysis.json"
            an[s] = CM.read(p) if p.exists() else None
        if any(v is None or not v["identity_passed"] for v in an.values()):
            scene = "INCONCLUSIVE"
        else:
            tot = {s: v["primary"]["P-A"]["total"] for s, v in an.items()}
            improved = sum(t["hi"] < 0 for t in tot.values())
            regress = any(t["lo"] > 0.02 for t in tot.values())
            scene = "SUPPORTED" if (improved >= 2 and not regress) else ("NOT_SUPPORTED" if (improved == 0 or regress) else "INCONCLUSIVE")
            scene_detail = {s: {"point": t["point"], "lo": t["lo"], "hi": t["hi"]} for s, t in tot.items()}
    contracts_ok = q["baseline_identity"] and q["reciprocity"] and q["passivity"] and q["synthetic_controls"]
    if contracts_ok and L["angular_fidelity_gain"] == "PASS" and L["heldout_directional_transfer"] == "PASS" and scene == "SUPPORTED":
        viability = "SUPPORTED"
    elif contracts_ok and L["optimization_resolved"] and (L["angular_fidelity_gain"] == "FAIL" or scene == "NOT_SUPPORTED"):
        viability = "NOT_SUPPORTED"
    else:
        viability = "INCONCLUSIVE"
    pf = lambda b: "PASS" if b else "FAIL"
    table = {"baseline_identity": pf(q["baseline_identity"]), "reciprocity": pf(q["reciprocity"]), "passivity": pf(q["passivity"]),
             "synthetic_controls": pf(q["synthetic_controls"]), "angular_fidelity_gain": L["angular_fidelity_gain"],
             "per_component_benefit_beyond_shared_gating": L["per_component_benefit"],
             "heldout_directional_transfer": L["heldout_directional_transfer"], "scene_utility": scene,
             "runtime_feasibility": "MEASURED" if E.get("cost") else "UNQUALIFIED", "candidate_architecture_viability": viability}
    out = {"decision_table": table, "optimization_resolved": L["optimization_resolved"], "qualification": q,
           "scene_total_P_minus_A": scene_detail, "medians": {k: {c: v["eval:el1"][c]["median"] for c in E["cohorts"]}
                                                              for k, v in E["contrasts"].items()},
           "commit": CM.commit(), "eval_sha256": CM.sha(root / "eval/eval.json")}
    CM.write(root / "decision.json", out)
    print("decision", table, flush=True)


if __name__ == "__main__":
    main()
