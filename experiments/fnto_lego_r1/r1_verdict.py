"""WL42 verdict under protocol fnto_r1 (numbers only from stored records; written BEFORE any r1 training or held-out result).

    python r1_verdict.py [--root results/fnto_lego_r1] -> <root>/verdict.json

Gate inputs (all under <root> unless noted):
  G1  gate1/gate1.json                      G2a anchors.json (support reach)        G2b gate2/quadrature.json, quadrature_a1.json
  G2c gate2/coverage.json                   G2d K_*/train.json (held-out probes), eval_K_*/eval.json (K_only | first_r1)
  G3a render_checks/render_checks.json      G3b gate3/light_table_radiometry.json
  G4a relations.json + eval (E_upd vs E_relfull)   G4b tests/solver_detection.json   G4c K_*/rho.json
  G5  eval_K_*/eval.json + worklog-37 eval.json (RNA_T0 / RNA_T1, C.B4/eval/eval.json; historical, unchanged)
Kernel used for Gate 5 = the first of K_r1, K_r1P whose rho_T0 < 1 (protocol conditional_variant_r1P). Verdict rules are
copied from protocol/fnto_r1.json verdict_rules; nothing here is tuned after seeing results.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

import r1_common as C

RNA_T0_HIST = 33.569  # protocol G5 'static' constant; cross-checked against the worklog-37 eval.json below


def jl(p, default=None):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else default


def fin(x):
    return x is not None and isinstance(x, (int, float)) and math.isfinite(x)


def main(root):
    root = Path(root); proto = jl(C.HERE1 / "protocol/fnto_r1.json")
    b4 = jl(C.B4 / "eval/eval.json")["sets"]
    rna_t0 = b4["heldout_T0"]["rows"]["RNA_T0"]["view_mean_psnr"]; e_rna_t1 = b4["heldout_T1"]["rows"]["RNA_T1"]["regions"]["A_changed"]["mse"]
    assert abs(rna_t0 - RNA_T0_HIST) < 5e-3, (rna_t0, RNA_T0_HIST)
    gates = {}
    # ---- G1 (measured at declaration) ----
    g1 = jl(root / "gate1/gate1.json"); a, b = g1["r1"]["G1a"], g1["r1"]["G1b"]
    gates["G1"] = {"origin_within_0p2mm": a["frac_origin_within_0p2mm_of_surface"], "inside_face": a["frac_first_hit_plus_n_inside_face"],
                   "agreement": b["port_vs_pixel_visibility_agreement"], "irradiance_ratio": b["pooled_port_over_pixel_irradiance"],
                   "criteria_pass": {"origin": a["frac_origin_within_0p2mm_of_surface"] >= 0.99, "inside_face": a["frac_first_hit_plus_n_inside_face"] < 0.01,
                                     "agreement": b["port_vs_pixel_visibility_agreement"] >= 0.97, "irradiance": 0.95 <= b["pooled_port_over_pixel_irradiance"] <= 1.05},
                   "pass": bool(g1["r1"]["pass"]), "note": proto["verdict_rules"]["G1_note"]}
    # ---- G2 ----
    sup = jl(root / "anchors.json")["support"]["0"]
    g2a = sup["band_reach_q10_q50_q90_m"][1] >= 0.98 * C.R_LOCAL and sup["mid_reach_q10_q50_q90_m"][1] >= 0.98 * C.R_MID
    q0, q1 = jl(root / "gate2/quadrature.json"), jl(root / "gate2/quadrature_a1.json")
    cov = jl(root / "gate2/coverage.json"); outside = cov["r1"]["r1_coarse"]["indirect_one_bounce_energy_outside_full_span"]
    gates["G2"] = {"a_reach_pass": bool(g2a), "a_reach_q50_m": [sup["band_reach_q10_q50_q90_m"][1], sup["mid_reach_q10_q50_q90_m"][1]],
                   "b_quadrature": {"a0": {k: q0[k]["median_rel_err"] for k in ("band", "mid")}, "a1": {k: q1[k]["median_rel_err"] for k in ("band", "mid")},
                                    "pass": bool(q1["band"]["pass"] and q1["mid"]["pass"])},
                   "c_coverage": {"indirect_outside_full_span": outside, "effective_rank_90_99_999": cov["r1"]["r1_coarse"]["effective_rank_90_99_999"],
                                  "batch6_common_outside_full_span": cov["batch6_common"]["indirect_one_bounce_energy_outside_full_span"], "pass": bool(outside <= 0.10)}}
    # ---- G3 ----
    rc = jl(root / "render_checks/render_checks.json")["clamp"]; g3b = jl(root / "gate3/light_table_radiometry.json")
    lin = abs(rc["superposition_none"]["rel_signed_energy"])
    gates["G3"] = {"a_linearity_abs_signed": lin, "a_pass": bool(lin <= 0.01), "a_clamped_signed": rc["superposition_generator"]["rel_signed_energy"],
                   "b_max_rel_dev": g3b["max_rel_dev_from_mean"], "b_pass": bool(g3b["pass"]), "b_per_family_median": {k: v["median"] for k, v in g3b["per_family"].items()}}
    gates["G3"]["pass"] = gates["G3"]["a_pass"] and gates["G3"]["b_pass"]
    # ---- G4b (solver detection) and G4a part 1 (incremental relation update) ----
    det = jl(root / "tests/solver_detection.json"); rel = jl(root / "relations.json")["incremental_T0_to_T1"]
    gates["G4"] = {"b_detection_pass": bool(all(v["ok"] for v in det.values())),
                   "a_incremental": {"mismatch_vs_full": rel["mismatch_vs_full"], "mismatch_same_distance": rel["mismatch_same_distance"],
                                     "mismatch_same_material_and_component": rel["mismatch_same_material_and_component"], "rays_selected": rel["rays_selected"],
                                     "rays_total": rel["rays_total"], "update_seconds": rel["select_s"] + rel["trace_map_s"]}}
    # ---- G4c: which kernel passes; G2d held-out probe fidelity ----
    kernels = {}
    for k in ("K_r1", "K_r1P"):
        tr, rho = jl(root / k / "train.json"), jl(root / k / "rho.json")
        if tr is None:
            continue
        ev_final = [e for e in tr["evals"] if e.get("step") == "final"]
        kernels[k] = {"trained": True, "steps": tr["args"]["steps"], "passivity": tr["args"]["passivity"], "digest": tr.get("digest"),
                      "train_hours": tr["train_s"] / 3600, "peak_mem_gib": tr.get("peak_mem_gib"), "params": tr["params_A"] + tr["params_B"],
                      "holdout_probe_psnr": ev_final[0]["psnr"] if ev_final else None, "uniform_albedo": tr.get("uniform_albedo"),
                      "rho_T0": rho["T0"] if rho else None, "rho_T1_inc": rho["T1_inc"] if rho else None,
                      "G4c_pass": bool(rho is not None and fin(rho["T0"]) and rho["T0"] < 1)}
    gate5_kernel = next((k for k in ("K_r1", "K_r1P") if kernels.get(k, {}).get("G4c_pass")), None)
    g4c_both_fail = all(not kernels.get(k, {}).get("G4c_pass", False) for k in ("K_r1", "K_r1P")) and "K_r1" in kernels and \
        (("K_r1P" in kernels) or not (kernels["K_r1"]["rho_T0"] is not None and kernels["K_r1"]["rho_T0"] >= 1))
    # if K_r1 failed G4c and K_r1P was never trained, G4 is incomplete, not failed twice
    gates["G4"]["c"] = {k: {"rho_T0": v["rho_T0"], "rho_T1_inc": v["rho_T1_inc"], "pass": v["G4c_pass"]} for k, v in kernels.items()}
    gates["G4"]["c_pass_any"] = gate5_kernel is not None; gates["G4"]["c_fail_both"] = bool(g4c_both_fail and gate5_kernel is None)
    out = {"protocol": proto["protocol"], "amendments": [a_["id"] for a_ in proto.get("amendments", [])], "gates": gates, "kernels": kernels,
           "gate5_kernel": gate5_kernel, "rules": proto["verdict_rules"]}
    # ---- G5 on the Gate-5 kernel ----
    ev = jl(root / f"eval_{gate5_kernel}/eval.json") if gate5_kernel else None
    g5 = None
    if ev is not None:
        T0, T1 = ev["sets"]["heldout_T0"], ev["sets"]["heldout_T1"]
        s_hist = T0["S_T0|hist"]["view_mean_psnr"]; S = s_hist - rna_t0 if fin(s_hist) else float("nan")
        static = "credible" if S >= -1.0 else ("weak" if S >= -2.0 else "fail")             # NaN (unmeasurable) -> fail, as fnto_v1
        Es, Eu = T1["D_stale|hist"]["regions"]["A_changed"]["mse"], T1["E_upd|hist"]["regions"]["A_changed"]["mse"]
        F = Es / Eu if (fin(Es) and fin(Eu) and Eu > 0) else float("nan")
        R_rna = (Es - Eu) / (Es - e_rna_t1) if (fin(Es) and fin(Eu) and Es > e_rna_t1) else None
        sc = ev["signed_change"].get("hist", {}).get("A_changed", {}).get("E_upd", {})
        corr, slope = sc.get("corr"), sc.get("slope")
        k_only = T0.get("K_only|first_r1", {}).get("view_mean_psnr")
        probe = kernels[gate5_kernel]["holdout_probe_psnr"]; probe_all = probe["all"] if probe else None
        g2d = bool(fin(probe_all) and probe_all >= 28 and fin(k_only) and k_only >= 28)
        relfull_gap = abs(T1["E_upd|hist"]["view_mean_psnr"] - T1["E_relfull|hist"]["view_mean_psnr"]) if "E_relfull|hist" in T1 else None
        g4a = bool(rel["mismatch_same_material_and_component"] == 0 and fin(relfull_gap) and relfull_gap <= 0.05)
        pv = ev["per_view"]; statuses = {(s_, a_, r["v"]): r["status"] for s_, arms in pv.items() for a_, rs in arms.items() for r in rs if a_ != "K_only"}
        n_div = sum(v == "diverged" for v in statuses.values()); n_unc = sum(v == "unconverged" for v in statuses.values()); n_all = len(statuses)
        all_conv = n_all > 0 and all(v == "converged" for v in statuses.values())
        gates["G2"]["d_kernel_fidelity"] = {"holdout_probe_psnr_all": probe_all, "K_only_vs_first_r1_T0_view_mean": k_only, "pass": g2d}
        gates["G4"]["a_relfull_gap_dB"] = relfull_gap; gates["G4"]["a_pass"] = g4a
        gates["G4"]["pass"] = bool(g4a and gates["G4"]["b_detection_pass"] and kernels[gate5_kernel]["G4c_pass"])
        g5 = {"kernel": gate5_kernel, "S_T0_hist_view_mean": s_hist, "RNA_T0_hist": rna_t0, "S_dB": S, "static_class": static,
              "S_T0_r1_view_mean": T0.get("S_T0|r1", {}).get("view_mean_psnr"),
              "E_stale_A_changed_hist": Es, "E_upd_A_changed_hist": Eu, "E_RNA_T1_A_changed_hist": e_rna_t1, "F": F, "R_RNA": R_rna,
              "signed_change_hist_A_changed_E_upd": sc, "signed_change_r1_A_changed_E_upd": ev["signed_change"].get("r1", {}).get("A_changed", {}).get("E_upd"),
              "signed_change_hist_stationary_E_upd": ev["signed_change"].get("hist", {}).get("stationary", {}).get("E_upd"),
              "solves": {"n": n_all, "diverged": n_div, "unconverged": n_unc, "all_converged": all_conv},
              "kernel_parameters_bit_identical": bool(ev.get("kernel_parameters_bit_identical")),
              "spectral_radius_eval": ev.get("spectral_radius"),
              "arms_view_mean_psnr": {s_: {k: r["view_mean_psnr"] for k, r in rows.items()} for s_, rows in ev["sets"].items()}}
        supported = (g2d and gates["G3"]["pass"] and gates["G4"]["pass"] and all_conv and static == "credible" and fin(F) and F >= 1.5
                     and R_rna is not None and R_rna >= 0.7 and fin(corr) and corr >= 0.5 and fin(slope) and 0.5 <= slope <= 1.5)
        ns_reasons = [r for r, c in (("G4c failed for both K_r1 and K_r1P", gates["G4"]["c_fail_both"]), ("a held-out view solve diverged", n_div > 0),
                                     ("static representability fail / unmeasurable", static == "fail"), ("F < 1.1", fin(F) and F < 1.1),
                                     ("corr < 0.2", fin(corr) and corr < 0.2)) if c]
        not_supported = bool(ns_reasons)
        out["verdict"] = "SUPPORTED" if supported else ("NOT SUPPORTED" if not_supported else "LIMITED")
        out["not_supported_reasons"] = ns_reasons
        out["supported_conditions"] = {"G2d": g2d, "G3": gates["G3"]["pass"], "G4": gates["G4"]["pass"], "all_converged": all_conv, "S_credible": static == "credible",
                                       "F_ge_1.5": fin(F) and F >= 1.5, "R_RNA_ge_0.7": R_rna is not None and R_rna >= 0.7,
                                       "corr_ge_0.5": fin(corr) and corr >= 0.5, "slope_in_0.5_1.5": fin(slope) and 0.5 <= slope <= 1.5}
    elif gates["G4"]["c_fail_both"]:
        out["verdict"] = "NOT SUPPORTED"; out["not_supported_reasons"] = ["G4c failed for both K_r1 and K_r1P (stopped at Gate 4)"]
    else:
        out["verdict"] = "INCOMPLETE"; out["not_supported_reasons"] = []
        out["missing"] = "no Gate-5 kernel with rho_T0 < 1 has an eval.json yet" if gate5_kernel is None else f"eval_{gate5_kernel}/eval.json missing"
    out["G5"] = g5
    out["G1_and_G2_failures_carried"] = [k for k, v in (("G1", gates["G1"]["pass"]), ("G2b", gates["G2"]["b_quadrature"]["pass"]), ("G2c", gates["G2"]["c_coverage"]["pass"])) if not v]
    (root / "verdict.json").write_text(json.dumps(out, indent=1, default=float)); print(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--root", default=str(C.OUT1)); a = ap.parse_args(); main(a.root)
