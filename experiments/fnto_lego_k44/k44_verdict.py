"""WL44 decision under protocol fnto_k44 (rules fixed in protocol/fnto_k44.json before any candidate result).

    python k44_verdict.py        -> results/fnto_lego_k44/verdict_k44.json
Inputs: results/fnto_lego_k44/C_decomp_{K_r1P,K_r1P_s1,K_k44P}.json (k44_decomp), results/fnto_lego_r1/eval_{...}/eval.json
"""
import json

import k44_common as K

R1 = K.C.OUT1


def get(tag):
    return json.loads((K.OUT / f"C_decomp_{tag}.json").read_text())


def lego(d, s):
    return d[s]["Konly_dom:lego"]["pooled_psnr"]


def main():
    proto = json.loads((K.HERE / "protocol/fnto_k44.json").read_text())
    b, s1, c = get("K_r1P"), get("K_r1P_s1"), get("K_k44P")
    spread = abs(lego(s1, "heldout_T0") - lego(b, "heldout_T0"))
    thr = max(0.3, 2 * spread)
    dT0 = lego(c, "heldout_T0") - lego(b, "heldout_T0"); dT1 = lego(c, "heldout_T1") - lego(b, "heldout_T1")
    lum_b = b["heldout_T0"]["opaque_shadow_mean_pred_lum"]["value"]; lum_c = c["heldout_T0"]["opaque_shadow_mean_pred_lum"]["value"]
    lum_red = 1 - lum_c / lum_b
    improved = dT0 >= thr and dT1 > 0 and lum_red >= 0.30
    decision = "IMPROVED" if improved else ("MECHANISM_ONLY" if lum_red >= 0.30 else "NO_IMPROVEMENT")
    rows = {}
    for tag, d in (("K_r1P", b), ("K_r1P_s1", s1), ("K_k44P", c)):
        ev = json.loads((R1 / f"eval_{tag}/eval.json").read_text())
        rows[tag] = {s: {"Konly_lego": lego(d, s), "Konly_box": d[s]["Konly_dom:box"]["pooled_psnr"], "Konly_all": d[s]["Konly:K"]["pooled_psnr"],
                         "Konly_opaque_shadow": d[s]["Konly_opaque_shadow"]["pooled_psnr"], "opaque_shadow_mean_pred_lum": d[s]["opaque_shadow_mean_pred_lum"]["value"],
                         "yellow_shadow": d[s].get("Kmat:Yellow:shadow", {}).get("pooled_psnr"), "yellow_lit": d[s].get("Kmat:Yellow:lit", {}).get("pooled_psnr"),
                         "full_pred_vs_full_r1": d[s].get("swap:full_pred", {}).get("pooled_psnr"),
                         "indirect_all": {k: d[s]["ind:all"][k] for k in ("pred_over_ref", "corr", "explained_fraction")} if "ind:all" in d[s] else None}
                     for s in ("heldout_T0", "heldout_T1")}
        rows[tag]["eval"] = {"rho_T0": ev["spectral_radius"]["T0"]["rho_estimate"],
                             "views_mean": {f"{s}:{k}": r["view_mean_psnr"] for s, rr in ev["sets"].items() for k, r in rr.items()},
                             "diverged": sum(r["status"] == "diverged" for s_, arms in ev["per_view"].items() for a_, rs in arms.items() for r in rs),
                             "signed_change_hist_A_changed_E_upd": ev["signed_change"].get("hist", {}).get("A_changed", {}).get("E_upd"),
                             "params_identical": ev.get("kernel_parameters_bit_identical")}
    out = {"protocol": proto["protocol"], "decision": decision, "seed_spread_T0_lego_dB": spread, "threshold_dB": thr,
           "delta_T0_lego_dB": dT0, "delta_T1_lego_dB": dT1, "opaque_shadow_lum_reduction": lum_red, "rows": rows, "rules": proto["decision_rules"]}
    K.save("verdict_k44.json", out)


if __name__ == "__main__":
    main()
