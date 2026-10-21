"""Worklog-37 outcome evaluation of protocol b4_v1 (b4_v1.1) from the final evaluation record (any Python 3, stdlib only).

    python b4_verdict.py --res <results/rna_selective_reconstruction_lego> --out <results/evaluation/37>

Applies the predeclared outcome_rules of protocol/b4_v1.json to eval/eval.json (held-out T1, PU composition primary) and
the owned arms' train.json, and writes verdict.json to <res>/eval and <out>. Everything under "attribution" and
"post_hoc" is descriptive analysis written after the result was seen; it does not change the verdict.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


def db(mse_ref, mse):
    return 10 * math.log10(mse_ref / mse)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    res, out = Path(a.res), Path(a.out)
    proto = json.loads((Path(__file__).parent / "protocol/b4_v1.json").read_text())
    ev = json.loads((res / "eval/eval.json").read_text())
    orc = json.loads((res / "oracle/oracle.json").read_text())
    tr = {d: json.loads((res / d / "train.json").read_text()) for d in ("SO_T0", "A_full", "B_sel")}
    T1, T0 = ev["sets"]["heldout_T1"]["rows"], ev["sets"]["heldout_T0"]["rows"]
    D = ev["D_star"]; nu = len(T1["A_full-PU"]["per_unit_px"])
    full, sel, frz, rna1, rna0 = T1["A_full-PU"], T1["B_sel-PU"], T1["SO_T0-PU"], T1["RNA_T1"], T0["RNA_T0"]
    reg = lambda row, r: row["regions"][r]["mse"]  # noqa: E731

    q = {"RNA_T0_view_mean_psnr": rna0["view_mean_psnr"], "RNA_T1_view_mean_psnr": rna1["view_mean_psnr"], "threshold_psnr": 32.44,
         "A_full_minus_RNA_T1_db": full["view_mean_psnr"] - rna1["view_mean_psnr"], "A_full_logl1_ratio_vs_RNA_T1": full["logl1"] / rna1["logl1"]}
    q["pass"] = q["RNA_T0_view_mean_psnr"] >= 32.44 and q["RNA_T1_view_mean_psnr"] >= 32.44 and q["A_full_minus_RNA_T1_db"] >= -0.5 \
        and q["A_full_logl1_ratio_vs_RNA_T1"] <= 1.10
    e_frz, e_full, e_sel = reg(frz, "A_changed"), reg(full, "A_changed"), reg(sel, "A_changed")
    disc = {"frozen_changed_mse": e_frz, "A_full_changed_mse": e_full, "ratio": e_frz / e_full}; disc["pass"] = disc["ratio"] >= 1.25
    infl = ev["sets"]["heldout_T1"]["influence"]["B_sel:SO_T0"]
    R = (e_frz - e_sel) / (e_frz - e_full)
    pos = {
        "view_mean_gap_db": {"value": sel["view_mean_psnr"] - full["view_mean_psnr"], "threshold": -0.26},
        "changed_region_recovery_R": {"value": R, "threshold": 0.8, "B_sel_changed_mse": e_sel},
        "Vstar_owned_mse_ratio": {"value": reg(sel, "Vstar_owned") / reg(full, "Vstar_owned"), "threshold": 1.10},
        "Vstar_parameters_bit_identical": {"value": bool(tr["B_sel"]["frozen_units_bit_identical"]) and tr["B_sel"]["units_changed"] == D},
        "output_change_inside_predicted_support": {"value": bool(infl["changed_within_predicted"]) and infl["outside_changed"] == 0
                                                   and infl["hard_outside_changed"] == 0, "changed_px": infl["changed"], "predicted_px": infl["predicted"]},
    }
    pos["view_mean_gap_db"]["pass"] = pos["view_mean_gap_db"]["value"] >= -0.26
    pos["changed_region_recovery_R"]["pass"] = R >= 0.8
    pos["Vstar_owned_mse_ratio"]["pass"] = pos["Vstar_owned_mse_ratio"]["value"] <= 1.10
    for k in ("Vstar_parameters_bit_identical", "output_change_inside_predicted_support"):
        pos[k]["pass"] = pos[k]["value"]
    positive = all(v["pass"] for v in pos.values())
    negative = pos["view_mean_gap_db"]["value"] < -1.0 or R < 0.5
    degenerate = len(D) >= 15
    if not q["pass"]:
        case = "FULL_RECOMPUTE_FAILURE"
    elif not disc["pass"]:
        case = "INCONCLUSIVE_NON_DISCRIMINATIVE_EDIT"
    elif positive:
        case = "POSITIVE"
    elif negative:
        case = "NEGATIVE"
    else:
        case = "CONDITIONAL"

    # descriptive attribution (post-result)
    px = full["per_unit_px"]; tot = sum(px); a_u, b_u, f_u = full["per_unit_mse"], sel["per_unit_mse"], frz["per_unit_mse"]
    gap = sum((b_u[i] - a_u[i]) * px[i] for i in range(nu))
    units = [{"unit": i, "set": "D*" if i in D else "V*", "px": px[i], "RNA_T1": rna1["per_unit_mse"][i], "SO_T0_on_T0": T0["SO_T0-PU"]["per_unit_mse"][i],
              "frozen": f_u[i], "A_full": a_u[i], "B_sel": b_u[i], "B_sel_over_A_full": b_u[i] / a_u[i], "share_of_pooled_gap": (b_u[i] - a_u[i]) * px[i] / gap,
              "oracle_rms_change": orc["units"][i]["rms_change"], "oracle_dirty_at_2tau": orc["units"][i]["dirty_at_2tau"]} for i in range(nu)]
    pooled_full, pooled_sel = reg(full, "all"), reg(sel, "all")
    u3 = (pooled_sel * tot - (b_u[3] - f_u[3]) * px[3]) / tot
    tf, ts = tr["A_full"]["seconds"]["total"], tr["B_sel"]["seconds"]["total"]
    pp = tr["B_sel"]["parameters_per_unit"]
    rec = {
        "protocol": "b4_v1.1 (b4_v1 + amendment 1; execution amendments 2-4)", "worklog": 37,
        "evaluation_regime": "held-out T1, 16 views, 1024^2, reference = mean of 2 x 2048 spp; display transform (x/(1+x))^(1/2.2); owned arms under PU (h 4.5 mm)",
        "D_star": D, "V_star": [i for i in range(nu) if i not in D], "degenerate_selectivity": degenerate,
        "qualification": q, "discriminative_edit": disc, "positive_criteria": pos, "negative": negative, "case": case,
        "arms_view_mean_psnr_T1": {n: T1[n]["view_mean_psnr"] for n in ("RNA_T1", "A_full-PU", "B_sel-PU", "SO_T0-PU", "A_full", "B_sel", "SO_T0")},
        "static_T0": {"RNA_T0": rna0["view_mean_psnr"], "SO_T0-PU": T0["SO_T0-PU"]["view_mean_psnr"]},
        "recompute_cost": {"A_full_s": tf, "B_sel_s": ts, "ratio": ts / tf, "trained_parameters_B_sel": sum(pp[i] for i in D), "parameters_total": sum(pp),
                           "trained_fraction": sum(pp[i] for i in D) / sum(pp), "machine": tr["A_full"]["gpu"],
                           "note": "both arms ran uninterrupted, alone on the GPU, in chain v6 (amendment 3); wall-clock includes data setup"},
        "attribution": {"note": "descriptive, written after the result; single seed per arm",
                        "pooled_gap_db": db(pooled_full, pooled_sel),
                        "gap_share": {"D*_owned": sum(u["share_of_pooled_gap"] for u in units if u["set"] == "D*"),
                                      "V*_owned": sum(u["share_of_pooled_gap"] for u in units if u["set"] == "V*")},
                        "units": units},
        "post_hoc": {"note": "counterfactual computed after the result; not a protocol outcome",
                     "pooled_gap_db_if_unit3_kept_frozen": db(pooled_full, u3)},
        "sources": {p: hashlib.sha256((res / p).read_bytes()).hexdigest()[:16] for p in ("eval/eval.json", "oracle/oracle.json", "A_full/train.json", "B_sel/train.json", "SO_T0/train.json")},
    }
    for d in (res / "eval", out):
        d.mkdir(parents=True, exist_ok=True); (d / "verdict.json").write_text(json.dumps(rec, indent=1))
    print(case, json.dumps({k: (v["value"], v["pass"]) for k, v in pos.items()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
