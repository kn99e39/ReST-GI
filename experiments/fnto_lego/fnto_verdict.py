"""FNTO Batch 6 verdict under the predeclared protocol fnto_v1 (numbers only from stored records; no new rendering).

    python fnto_verdict.py <fnto eval dir> -> <dir>/verdict.json

Inputs: FNTO eval.json (this batch), worklog-37 eval.json (RNA_T0/RNA_T1 and Surface-Owned arms), worklog-39
diag_material.json / diag_litshadow_T0.json (RNA per material / lit-shadow on held-out T0, same references),
results/fnto_lego/contracts.json (synthetic contracts A-F).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

import fnto_common as C

REFLECTIVE = ["Gray", "Black", "Light_brown", "Sand", "Brown"]
GLASS = ["transparent", "Red_Glass"]


def pooled(mats, names):
    px = sum(mats[n]["px"] for n in names if mats[n]["px"])
    return sum(mats[n]["mse"] * mats[n]["px"] for n in names if mats[n]["px"]) / max(px, 1), px


def main(d):
    d = Path(d)
    ev = json.loads((d / "eval.json").read_text())
    proto = json.loads((C.HERE / "protocol/fnto_v1.json").read_text())
    b4 = json.loads((C.B4 / "eval/eval.json").read_text())["sets"]
    rna_mat = json.loads((C.SPT / "eval/diag_material.json").read_text())["heldout_T0"]["RNA_T0"]
    rna_ls = json.loads((C.SPT / "eval/diag_litshadow_T0.json").read_text())
    con = json.loads((C.OUT / "contracts.json").read_text())
    rna_t0 = b4["heldout_T0"]["rows"]["RNA_T0"]["view_mean_psnr"]; rna_t1 = b4["heldout_T1"]["rows"]["RNA_T1"]["view_mean_psnr"]
    e_rna_t1 = b4["heldout_T1"]["rows"]["RNA_T1"]["regions"]["A_changed"]["mse"]
    e_afull = b4["heldout_T1"]["rows"]["A_full-PU"]["regions"]["A_changed"]["mse"]
    T0 = ev["sets"]["heldout_T0"]["S_T0"]; T1 = ev["sets"]["heldout_T1"]
    upd, stale = T1["E_upd"], T1["D_stale"]
    S = T0["view_mean_psnr"] - rna_t0
    static = "credible" if S >= -1.0 else ("weak" if S >= -2.0 else "fail")
    m = T0["materials"]
    y_ratio = m["Yellow"]["mse"] / rna_mat["Yellow"]["mse"]
    r_f, _ = pooled(m, REFLECTIVE); r_r, _ = pooled(rna_mat, REFLECTIVE); refl_ratio = r_f / r_r
    g_f, g_px = pooled(m, GLASS); g_r, _ = pooled(rna_mat, GLASS); glass_ratio = g_f / g_r
    sh_ratio = T0["shadow_mse"] / rna_ls["rna_sh"]["mse"]; lit_ratio = T0["lit_mse"] / rna_ls["rna_lit"]["mse"]
    mat_screen = all(x <= 3.0 for x in (y_ratio, refl_ratio, sh_ratio))
    Es, Eu = stale["regions"]["A_changed"]["mse"], upd["regions"]["A_changed"]["mse"]
    F = Es / Eu
    R_rna = (Es - Eu) / (Es - e_rna_t1) if Es > e_rna_t1 else None
    R_afull = (Es - Eu) / (Es - e_afull) if Es > e_afull else None
    W = upd["view_mean_psnr"] - rna_t1
    pv = ev["per_view"]
    diverged = any(r["diverged"] for s_ in pv.values() for arm in s_.values() for r in arm)
    unconverged = [(s_, a_, r["v"]) for s_, arms in pv.items() for a_, rs in arms.items() for r in rs if r["resid"] >= 1e-3]
    params_ok = bool(ev["kernel_parameters_bit_identical"])
    contracts_ok = all(v["pass"] for k, v in con.items() if isinstance(v, dict))
    info_path_fail = [k for k in ("B_nonlocal_internal", "C_transmission", "D_directional_sharp") if not con.get(k, {}).get("pass", False)]
    supported = (static == "credible" and mat_screen and F >= 1.5 and R_rna is not None and R_rna >= 0.7 and W >= -1.0
                 and params_ok and not diverged and not unconverged and contracts_ok)
    not_supported = (static == "fail" or F < 1.1 or (R_rna is not None and R_rna < 0.3) or diverged or bool(info_path_fail))
    case = "SUPPORTED" if supported else ("NOT SUPPORTED" if not_supported else "CONDITIONAL")
    out = {"protocol": proto["protocol"], "case": case,
           "static": {"S_T0_view_mean": T0["view_mean_psnr"], "RNA_T0": rna_t0, "S": S, "class": static},
           "material_screen_T0": {"yellow_mse_ratio_vs_RNA": y_ratio, "reflective_pooled_ratio": refl_ratio, "shadowed_ratio": sh_ratio,
                                  "lit_ratio": lit_ratio, "glass_pooled_ratio_descriptive": glass_ratio, "glass_px": g_px,
                                  "glass_flag_gt5": glass_ratio > 5, "pass_all_le_3": mat_screen},
           "edit": {"E_stale_A_changed": Es, "E_upd_A_changed": Eu, "F": F, "R_vs_RNA_T1": R_rna, "R_vs_A_full_PU": R_afull,
                    "E_RNA_T1": e_rna_t1, "E_A_full_PU": e_afull, "W_vs_RNA_T1_dB": W, "E_upd_view_mean": upd["view_mean_psnr"],
                    "D_stale_view_mean": stale["view_mean_psnr"], "E_full_view_mean": T1.get("E_full", {}).get("view_mean_psnr")},
           "kernel_parameters_bit_identical": params_ok, "diverged_any_view": diverged, "unconverged": unconverged,
           "contracts_all_pass": contracts_ok, "information_path_contract_failures": info_path_fail,
           "rules": proto["outcome_rules"]}
    (d / "verdict.json").write_text(json.dumps(out, indent=1, default=float)); print(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main(sys.argv[1])
