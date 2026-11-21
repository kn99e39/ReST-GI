"""WL104J frozen decision rules (protocol.json "verdict") applied to eval.json.

Usage: python decision104j.py <eval_dir>
"""
import sys
from pathlib import Path

import common104j as CM


def main():
    d = Path(sys.argv[1])
    E = CM.read(d / "eval.json")
    con = E["contrasts"]
    syn = E["synthetic"]
    coh = list(E["cohorts"])
    labels = {k: con[k]["label"] for k in ("B-A", "C-B", "C-A", "C_noF-B", "C-C_noF")}
    controls = {k: syn[k]["pass"] for k in ("S0a", "S1", "S2")}
    controls_ok = all(controls.values())
    optim_ok = E["convergence"]["resolved"]
    start_stable = {k: all(E["start_stability"][k][c] >= 3 for c in coh) for k in ("B-A", "C-B", "C-A")}
    overfit = {k: any(con[k]["overfit"][c]["overfit_dominated"] for c in coh) for k in ("B-A", "C-B", "C-A")}
    slack_ok = {k: all(E["slack"][c][k]["separable"] for c in coh) for k in ("B-A", "C-B", "C-A")}
    ref_ok = all(v["resolves"] for v in E["reference_resolution"].values())
    ca = labels["C-A"]["label"]
    reasons = []
    if not controls_ok:
        reasons.append("CONTROL_FAILED")
    if not optim_ok:
        reasons.append("OPTIMIZATION_UNRESOLVED")
    if not ref_ok:
        reasons.append("REFERENCE_INSUFFICIENT")
    if ca in ("SUBSTANTIAL", "MEASURABLE"):
        if not start_stable["C-A"]:
            reasons.append("START_UNSTABLE")
        if overfit["C-A"]:
            reasons.append("OVERFIT_DOMINATED")
        if not slack_ok["C-A"]:
            reasons.append("SLACK_INSEPARABLE")
    # cohort disagreement: SUBSTANTIAL-size effect in exactly one cohort
    c_ca = con["C-A"]["eval:el1"]
    sub_coh = [c for c in coh if c_ca[c]["median"] <= -0.20 and c_ca[c]["hi"] <= -0.10]
    if ca != "SUBSTANTIAL" and len(sub_coh) == 1:
        reasons.append("COHORT_DISAGREEMENT")
    if reasons:
        verdict = "INCONCLUSIVE"
    elif ca == "SUBSTANTIAL":
        verdict = "SUPPORTED"
    else:
        verdict = "NOT_SUPPORTED"
    attribution = None
    if ca in ("SUBSTANTIAL", "MEASURABLE"):
        sh = {}
        for c in coh:
            gA = E["slack"][c]["B-A"]["physical_gain"]
            gC = E["slack"][c]["C-A"]["physical_gain"]
            sh[c] = gA / gC if gC > 0 else None
        if all(v is not None and v >= 2 / 3 for v in sh.values()):
            attribution = "AMPLITUDE_PRIMARY"
        elif all(v is not None and v <= 1 / 3 for v in sh.values()):
            attribution = "LOBE_PARAMETER_PRIMARY"
        else:
            attribution = "MIXED"
        attribution = {"label": attribution, "amplitude_share": sh}
    out = {"verdict": verdict, "reasons": reasons, "effect_labels": labels, "synthetic_controls": controls,
           "optimization_resolved": optim_ok, "start_stable": start_stable, "overfit_dominated": overfit,
           "slack_separable": slack_ok, "reference_resolves": ref_ok,
           "residual_large": {c: E["residual_C_over_A"][c] >= 0.5 for c in coh}, "residual_C_over_A": E["residual_C_over_A"],
           "transfer": {k: con[k]["transfer_label"] for k in ("B-A", "C-B", "C-A")}, "attribution": attribution,
           "medians": {k: {c: con[k]["eval:el1"][c]["median"] for c in coh} for k in ("B-A", "C-B", "C-A", "C_noF-B", "C-C_noF")},
           "commit": CM.commit(), "eval_sha256": CM.sha(d / "eval.json")}
    CM.write(d / "decision.json", out)
    print("decision", verdict, reasons, {k: v["label"] for k, v in labels.items()}, attribution, flush=True)


if __name__ == "__main__":
    main()
