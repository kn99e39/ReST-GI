"""Worklog-38 outcome of protocol b4_seedrep_v1 from the server evaluation record (any Python 3, stdlib only).

    python b4_seedrep_verdict.py --res <results/rna_selective_reconstruction_lego> --out <results/evaluation/38>

Inputs: <res>/seedrep_server/... (eval.json and train.json files copied from LabServer63) and the worklog-37 PC records
(<res>/eval/eval.json, <res>/B_sel/train.json) for seed 0. Writes seedrep_verdict.json to <res>/eval and <out>.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

D_STAR = [0, 2, 3, 6, 7, 8, 13, 16]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    res, out = Path(a.res), Path(a.out)
    srv = res / "seedrep_server/repo/results/rna_selective_reconstruction_lego"
    ev_path = srv / "eval_seedrep/eval.json"
    S = json.loads(ev_path.read_text())["sets"]["heldout_T1"]; rows, infl = S["rows"], S["influence"]
    pc = json.loads((res / "eval/eval.json").read_text())["sets"]["heldout_T1"]["rows"]
    mse = lambda r, k: r["regions"][k]["mse"]  # noqa: E731
    fr = rows["SO_T0-PU"]
    check = {n: rows[n]["view_mean_psnr"] - pc[n]["view_mean_psnr"] for n in ("SO_T0-PU", "A_full-PU", "B_sel-PU")}
    seeds = []
    for s, (an, bn, tj) in {0: ("A_full", "B_sel", res / "B_sel/train.json"),
                             1: ("A_full_s1", "B_sel_s1", srv / "B_sel_s1/train.json"),
                             2: ("A_full_s2", "B_sel_s2", srv / "B_sel_s2/train.json")}.items():
        A, B = rows[an + "-PU"], rows[bn + "-PU"]; t = json.loads(tj.read_text()); i = infl[f"{bn}:SO_T0"]
        seeds.append({
            "seed": s, "trained_on": "RTX 5080 PC (worklog 37)" if s == 0 else "LabServer63 RTX 3080 Ti (concurrent)",
            "A_full_view_mean": A["view_mean_psnr"], "B_sel_view_mean": B["view_mean_psnr"], "g": B["view_mean_psnr"] - A["view_mean_psnr"],
            "R": (mse(fr, "A_changed") - mse(B, "A_changed")) / (mse(fr, "A_changed") - mse(A, "A_changed")),
            "Vstar_owned_ratio": mse(B, "Vstar_owned") / mse(A, "Vstar_owned"),
            "Vstar_bit_identical": bool(t["frozen_units_bit_identical"]) and t["units_changed"] == D_STAR,
            "influence_inside_support": bool(i["changed_within_predicted"]) and i["outside_changed"] == 0 and i["hard_outside_changed"] == 0,
            "unit3_B_over_A": B["per_unit_mse"][3] / A["per_unit_mse"][3], "unit3_B_over_frozen": B["per_unit_mse"][3] / fr["per_unit_mse"][3],
            "per_unit_B_over_A": [b / x for b, x in zip(B["per_unit_mse"], A["per_unit_mse"])],
        })
    g = [x["g"] for x in seeds]; mean_g = sum(g) / len(g)
    if mean_g >= -0.26:
        case = "WITHIN_MARGIN"
    elif all(x < -0.26 for x in g):
        case = "SYSTEMATIC_SHORTFALL"
    else:
        case = "UNRESOLVED"
    worse = sum(x["unit3_B_over_A"] > 1 for x in seeds)
    unit3 = "systematic" if worse == 3 else ("not_systematic" if worse <= 1 else "mixed")
    flags = [x["seed"] for x in seeds if x["g"] < -1.0 or x["R"] < 0.5 or not x["Vstar_bit_identical"] or not x["influence_inside_support"]]
    va = [x["A_full_view_mean"] for x in seeds]; vb = [x["B_sel_view_mean"] for x in seeds]
    rec = {
        "protocol": "b4_seedrep_v1", "worklog": 38,
        "evaluation_regime": "held-out T1, 16 views, 1024^2, 4096-spp PC references; PU h 4.5 mm; all seeds evaluated on LabServer63 in one b4_eval call",
        "cross_machine_eval_check": {"delta_view_mean_server_minus_pc": check, "tolerance": 0.01, "pass": all(abs(v) <= 0.01 for v in check.values())},
        "seeds": seeds, "mean_g": mean_g, "g_range": [min(g), max(g)], "case": case,
        "unit3": {"seeds_B_worse_than_A": worse, "classification": unit3}, "flags": flags,
        "seed_spread_view_mean": {"A_full": max(va) - min(va), "B_sel": max(vb) - min(vb)},
        "frozen_view_mean": fr["view_mean_psnr"],
        "note": "Supplementary to worklog 37; does not change its predeclared CONDITIONAL verdict. Wall-clock of the concurrent server runs is not a cost measurement.",
        "sources": {p.relative_to(res).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()[:16] for p in
                    (ev_path, res / "eval/eval.json", res / "B_sel/train.json", srv / "B_sel_s1/train.json", srv / "B_sel_s2/train.json",
                     srv / "A_full_s1/train.json", srv / "A_full_s2/train.json")},
    }
    for d in (res / "eval", out):
        d.mkdir(parents=True, exist_ok=True); (d / "seedrep_verdict.json").write_text(json.dumps(rec, indent=1))
    print(case, "mean g %.3f" % mean_g, "unit3", unit3, "flags", flags, "eval check", rec["cross_machine_eval_check"]["pass"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
