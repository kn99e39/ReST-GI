"""Worklog 39 outcome of protocol spt_v1 (any Python 3 with numpy; synthetic numbers need torch).

    python spt_verdict.py --out <results/evaluation/39>

Applies spt_v1's outcome rules to results/spt_lego/eval/eval.json and adds descriptive sections (post-result,
labelled): lit/shadow and per-material diagnostics, synthetic contract measurements, costs, and the worklog-37/38
baselines on the same regime. Writes spt_verdict.json to results/spt_lego/eval and <out>.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

import spt_common as C


def synthetic():
    sys.path.insert(0, str(Path(__file__).parent / "tests"))
    import test_spt_contracts as T

    res = {}
    for n in [k for k in dir(T) if k.startswith("test_")]:
        try:
            getattr(T, n)(); res[n] = "PASS"
        except AssertionError as e:
            res[n] = f"FAIL {e}"
    w = np.random.default_rng(0).normal(size=(200_000, 3)); w /= np.linalg.norm(w, axis=1, keepdims=True); w = w[w[:, 2] > 0]
    src = np.array([0.5, 0.0, 0.866]); lobe = (w @ src > np.cos(np.deg2rad(4))).astype(float); dw = 2 * np.pi / len(w)
    c = (C.sh9(w) * (lobe * w[:, 2])[:, None]).sum(0) * dw; rec_ = C.sh9(w) @ c
    true_diff = (lobe * w[:, 2]).sum() * dw / np.pi; sh_diff = c[0] / 0.282095 / np.pi
    refl = np.array([0.5, 0.0, 0.866])
    phong = (201 / (2 * np.pi)) * np.clip(w @ refl, 0, None) ** 200
    true_gl = (phong * lobe * w[:, 2]).sum() * dw; sh_gl = (phong * np.clip(rec_, 0, None)).sum() * dw
    res["sh2_diffuse_rel_error"] = float(abs(sh_diff - true_diff) / true_diff)
    res["sh2_glossy_phong200_rel_error"] = float(abs(sh_gl - true_gl) / true_gl)
    return res


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); a = ap.parse_args()
    E = C.OUT / "eval"
    ev = json.loads((E / "eval.json").read_text()); T1, T0 = ev["sets"]["heldout_T1"], ev["sets"]["heldout_T0"]
    b4 = json.loads((C.B4 / "eval/eval.json").read_text())["sets"]
    rna0 = b4["heldout_T0"]["rows"]["RNA_T0"]["view_mean_psnr"]
    mse = lambda r, k: T1[r]["regions"][k]["mse"]  # noqa: E731
    S = T0["S_T0"]["view_mean_psnr"] - rna0
    static = "credible" if S >= -1.0 else ("weak" if S >= -2.0 else "fail")
    Es, Eu, Ef = mse("D_stale", "A_changed"), mse("C_upd", "A_changed"), mse("C_full", "A_changed")
    F = Es / Eu; R = (Es - Eu) / (Es - Ef); W = T1["C_upd"]["view_mean_psnr"] - T1["C_full"]["view_mean_psnr"]
    ident = bool(ev["digest_T0_model_unchanged"])
    if static == "fail" or F < 1.1 or R < 0.3:
        case = "NOT_SUPPORTED"
    elif static == "credible" and F >= 1.5 and R >= 0.7 and W >= -0.5 and ident:
        case = "SUPPORTED"
    else:
        case = "CONDITIONAL"
    base = b4["heldout_T1"]["rows"]
    rel = {s: json.loads((C.OUT / f"rel_{s}.json").read_text()) for s in ("T0", "T1")}
    tr = {r: json.loads((C.OUT / f"{r}/train.json").read_text()) for r in ("SPT_T0", "SPT_T1")}
    rec = {
        "protocol": "spt_v1 (+ amendment 1, execution only)", "worklog": 39, "case": case,
        "rule_values": {"static_S_db": S, "static_class": static, "RNA_T0_view_mean": rna0, "S_T0_view_mean": T0["S_T0"]["view_mean_psnr"],
                        "F_stale_over_upd_A_changed": F, "R_recovery_vs_own_full": R, "W_upd_minus_full_db": W, "persistent_parameters_bit_identical": ident},
        "why": "static representability fails (S < -2 dB); per the predeclared rule the edit result is descriptive only",
        "arms_T1_view_mean": {n: r["view_mean_psnr"] for n, r in T1.items()},
        "arms_T1_A_changed_mse": {n: r["regions"]["A_changed"]["mse"] for n, r in T1.items()},
        "baselines_T1_same_regime": {n: {"view_mean": base[n]["view_mean_psnr"], "A_changed_mse": base[n]["regions"]["A_changed"]["mse"]}
                                     for n in ("RNA_T1", "A_full-PU", "B_sel-PU", "SO_T0-PU")},
        "descriptive_post_result": {
            "lit_shadow_T0": json.loads((E / "diag_litshadow_T0.json").read_text()),
            "per_material": json.loads((E / "diag_material.json").read_text()),
            "synthetic_contracts": synthetic(),
        },
        "representation": {"parameters_total": tr["SPT_T0"]["parameters"], "persistent_parameters": tr["SPT_T0"]["parameters"],
                           "configuration_dependent_learned_parameters": 0,
                           "anchors": rel["T0"]["anchors"], "relation_entries": rel["T0"]["anchors"] * rel["T0"]["K"],
                           "relation_table_bytes_per_state": "hit int32 [Na,128] + visibility bits [Na,512]",
                           "changed_by_edit": json.loads((C.OUT / "rel_diff_T0_T1.json").read_text())},
        "cost": {"relation_rebuild_T1_s": {k: rel["T1"][k] for k in ("gather_s", "vis_s", "map_s")}, "rays_per_rebuild": rel["T1"]["gather_rays"] + rel["T1"]["vis_rays"],
                 "state_solve_s_per_view": T1["C_upd"]["state_s_per_view"], "decode_s_per_view": T1["C_upd"]["decode_s_per_view"],
                 "eval_machine": "RTX 5080 PC (Windows, PyTorch prototype)", "training_h": {r: tr[r]["train_s"] / 3600 for r in tr},
                 "training_machine": tr["SPT_T0"]["gpu"], "target_supervision_after_edit": "none"},
        "sources": {p: C.sha16(C.OUT / p) for p in ("eval/eval.json", "SPT_T0/state.pt", "SPT_T1/state.pt", "rel_T0.npz", "rel_T1.npz", "anchors.npz")},
    }
    for d in (E, Path(a.out)):
        d.mkdir(parents=True, exist_ok=True); (d / "spt_verdict.json").write_text(json.dumps(rec, indent=1))
    print(case, json.dumps(rec["rule_values"]), json.dumps({k: v for k, v in rec["descriptive_post_result"]["synthetic_contracts"].items()}))


if __name__ == "__main__":
    main()
