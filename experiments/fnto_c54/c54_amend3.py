"""WL54 amendment 3 analysis (rules fixed in the protocol before the renders).
    P53_MI_VARIANT=cuda_ad_rgb python c54_amend3.py vol <out json> <cycles_hi_vol dir> <wl53 results>
    P53_MI_VARIANT=cuda_ad_rgb python c54_amend3.py lam <out json> <cycles_hi dir> <stageB dir> <stageC dir> <fixture,...>
vol: the WL53 gap8 VOLUME scene, camera obot, at 1 mm (WL53's stored renders, 2 seeds), 0.5 mm and 0.25 mm (seed 5499): V1 port signal of every
     group and the bias of the stored WL53 prediction (part_s1, nee) and of this batch's C1 (= C0 for this medium scene) against each.
lam: Lambert fixtures re-rendered at 0.25 mm: amended-coverage group biases of C0, G16, F16, C1 and the sensitivity arms against the 0.25 mm
     reference, plus the reference-validity flag of each group (|F16| <= 0.03 + 3 SE).
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "phys_ref53"))
import c54_amend1 as AM  # noqa: E402
import c54_common as CC  # noqa: E402
import c54_fixtures as CF  # noqa: E402
import c54_stageB as SB  # noqa: E402
import p53_compose as CP  # noqa: E402
import p53_fixtures as FX  # noqa: E402
import p53_oracle as OR  # noqa: E402
import p53_report as RP  # noqa: E402


def ports(ss, g, imgs):
    pid = CP.classify(ss, CC.blender_ortho_rays(g, SB.V1_OFF)); ms = [CP.port_means(pid, im.reshape(-1, 3), ss.Ptot) for im in imgs]
    cnt = ms[0][1]; cov = cnt / np.maximum(AM.expected_pixels(ss, g), 1e-9)
    return np.stack([m[0] for m in ms]), cnt, cov


def vol(out, vdir, R):
    scene = "gap8"; sd = FX.scene_def(scene); ss = OR.scene_set(sd); groups = RP.port_groups(scene); key = "obot__nee"
    pred = np.load(R / "oracle" / "pred" / f"{scene}__part_s1.npz")[key]; refs = {}
    g1 = json.load(open(R / "cycles" / f"ccam_{scene}.json"))["obot"]
    refs["1mm_wl53"] = (g1, [np.load(f) for f in sorted((R / "cycles").glob(f"cimg_{scene}_obot_*.npy"))])
    geo = json.load(open(vdir / f"ccam_{scene}.json"))
    for px in ("0.5", "0.25"):
        refs[f"{px}mm"] = (geo[f"obot@{px}"], [np.load(vdir / f"cimg_{scene}_obot_px{px}_s5499.npy")])
    res = {}
    for tag, (g, imgs) in refs.items():
        ref, cnt, cov = ports(ss, g, imgs); ok = (cnt >= SB.MINPIX) & (cov >= AM.COV); rm = ref.mean(0); r = {}
        for gn in np.unique(groups[ok]):
            m = ok & (groups == gn)
            if m.sum() < 5:
                continue
            r[gn] = dict(n=int(m.sum()), signal=rm[m].sum(0).tolist(), wl53_pred_bias=((pred[m] - rm[m]).sum(0) / rm[m].sum(0)).tolist(), cov_median=float(np.median(cov[m])))
        res[tag] = r
    g = "slab_lo:-y"; s5, s25 = np.array(res["0.5mm"][g]["signal"]), np.array(res["0.25mm"][g]["signal"])
    res["rule"] = dict(group=g, signal_change_0p5_to_0p25=((s25 - s5) / s5).tolist(), pixel_effect=bool(np.max(np.abs((s25 - s5) / s5)) >= 0.05),
                       bias_1mm=res["1mm_wl53"][g]["wl53_pred_bias"], bias_0p5=res["0.5mm"][g]["wl53_pred_bias"], bias_0p25=res["0.25mm"][g]["wl53_pred_bias"])
    CC.dump(res, out); print(json.dumps(res["rule"], indent=1))
    for tag in ("1mm_wl53", "0.5mm", "0.25mm"):
        for gn, v in res[tag].items():
            print(tag, gn, v["n"], "cov", round(v["cov_median"], 2), "bias R/G/B", [round(100 * x, 1) for x in v["wl53_pred_bias"]])


def lam(out, hdir, bdir, sdir, fixtures):
    res = {}
    for name in fixtures:
        sd = CF.scene_def(name); groups = CF.coarse_labels(name); ss = CC.sceneset_of(sd); arr = dict(np.load(bdir / f"{name}_coarse_arrays.npz"))
        arr.update({k: v for k, v in np.load(sdir / f"{name}_casp_arrays.npz").items() if k.startswith("C1")})
        if not (hdir / f"ccam_{name}.json").exists():
            continue
        geo = json.load(open(hdir / f"ccam_{name}.json")); res[name] = {}
        for key, g in geo.items():
            cam, px = key.split("@"); files = sorted(hdir.glob(f"cimg_{name}_{cam}_px{px}_s*.npy")); imgs = [np.load(f) for f in files]
            ref, cnt, cov = ports(ss, g, imgs); rm = ref[:, :, 0].mean(0)
            se = ref[:, :, 0].std(0, ddof=1) / np.sqrt(len(imgs)) if len(imgs) > 1 else np.zeros_like(rm)
            ok = (cnt >= SB.MINPIX) & (cov >= AM.COV); r = {}
            for arm in ("C0prod", "G16", "F16", "C1", "C1_c1", "C1_c4", "C1_b4"):
                r[arm] = {gn: dict(n=v["n"], share=v["share"], bias=v["bias"], se=v["se"], ruled=v["ruled"]) for gn, v in SB.group_stats(arr[arm], rm, ok, groups, se).items()}
            for gn in r["F16"]:
                r["F16"][gn]["ref_validated"] = bool(abs(r["F16"][gn]["bias"]) <= 0.03 + 3 * r["F16"][gn]["se"])
            res[name][cam] = dict(seeds=len(imgs), px_mm=float(px), groups=r)
            for gn, v in r["C0prod"].items():
                if v["ruled"]:
                    print(f"{name:10s} {cam:9s} {gn:12s} n={v['n']:3d} C0={100 * v['bias']:+6.1f}% F16={100 * r['F16'][gn]['bias']:+5.1f}% C1={100 * r['C1'][gn]['bias']:+5.1f}% c4={100 * r['C1_c4'][gn]['bias']:+5.1f}% valid={r['F16'][gn]['ref_validated']}")
    CC.dump(res, out)


if __name__ == "__main__":
    if sys.argv[1] == "vol":
        vol(Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4]))
    else:
        lam(Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4]), Path(sys.argv[5]), sys.argv[6].split(","))
