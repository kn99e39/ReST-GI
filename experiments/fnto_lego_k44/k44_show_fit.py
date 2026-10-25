"""Print compact views of WL44 A_fit_<tag>.json and B_quad_<tag>.json."""
import json
import sys

d = json.load(open(sys.argv[1]))
for s in ("train", "holdout"):
    o = d[s]; a = o["all"]
    print("=====", s, f"all {a['psnr']:.2f}", {k: round(a[k], 2) for k in ("psnr_loc_only", "psnr_no_glob", "psnr_no_band", "psnr_no_mid")})
    for k, v in o.items():
        if k.startswith("family:"):
            print(f"   {k:20s} {v['psnr']:.2f}")
    rows = sorted([(k, v) for k, v in o.items() if k.startswith("mat:")], key=lambda kv: -kv[1]["share_se"])
    for k, v in rows[:14]:
        print(f"   {k:24s} px {v['px']:9d} share {v['share_se']:.3f} psnr {v['psnr']:.1f} p/r {v['pred_over_ref']:.2f} ref {v['mean_ref']:.4f} "
              f"parts {[round(x, 4) for x in v['mean_parts_loc_glob_band_mid']]} noglob {v['psnr_no_glob']:.1f} noband {v['psnr_no_band']:.1f} nomid {v['psnr_no_mid']:.1f}")
if len(sys.argv) > 2:
    q = json.load(open(sys.argv[2]))
    print("===== quadrature", q["cost"])
    for v, dd in q["psnr"].items():
        print(f"   {v:16s} all {dd['all']['psnr']:.2f}  " + " ".join(f"{k}:{x['psnr']:.1f}" for k, x in dd.items() if k in ("Yellow:lit", "Yellow:shadow", "Gray:shadow", "Sand:shadow")))
    for k, v in q["term_change"].items():
        print(f"   {k:20s} mean|rel| {v['mean_abs_rel_change']:.3f} signed {v['signed_rel_change']:+.3f}")
