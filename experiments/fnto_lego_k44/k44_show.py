"""Print a compact view of a WL44 decomposition json (C_decomp_<tag>.json)."""
import json
import sys

d = json.load(open(sys.argv[1]))
for s in ("heldout_T0", "heldout_T1"):
    o = d[s]; print("=====", s)
    for k, v in o.items():
        if k.startswith(("Konly:", "swap:")):
            print(f"{k:40s} {v['pooled_psnr']:.2f}")
    print("-- K-only per material x lit/shadow")
    rows = sorted([(k, v) for k, v in o.items() if k.startswith("Kmat:") and v["px"]], key=lambda kv: -kv[1]["share_of_K_se"])
    for k, v in rows:
        print(f"{k:28s} px {v['px']:8d} share {v['share_of_K_se']:.3f} psnr {v['pooled_psnr']:.1f} p/r {v['pred_over_ref_lin']:.2f} "
              f"ref {v['mean_ref_lin']:.4f} loc {v['mean_loc']:.4f} glob {v['mean_glob']:.4f} bm {v['mean_band_mid']:.4f} bias {v['mean_bias_disp_lum']:+.4f}")
    print("-- indirect component (lum, linear)")
    for k, v in o.items():
        if k.startswith("ind:"):
            print(f"{k:18s} px {v['px']:8d} pred {v['mean_pred_lum']:.4f} ref {v['mean_ref_lum']:.4f} ratio {v['pred_over_ref']:.2f} "
                  f"corr {v['corr']:.3f} expl {v['explained_fraction']:.3f} rmse {v['rmse_lum']:.4f} rmsref {v['rms_ref_lum']:.4f}")
for k, v in d.get("edit_split_lum_linear", {}).items():
    print(f"edit {k:18s} " + " ".join(f"{a} {b:.3f}" for a, b in v.items() if a != "px"))
