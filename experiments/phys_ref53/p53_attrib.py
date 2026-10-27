"""WL53 POST HOC attribution diagnostics (labelled post hoc in the worklog; no rule or threshold changes).

    P53_MI_VARIANT=llvm_ad_rgb python p53_attrib.py <results dir>
(1) Comparison-unit check: the rules average Cycles pixels whose centre AND four corner rays hit the same port ("interior"
    pixels). On faces with a steep within-port gradient this drops the port's edge rows, while the oracle predicts the
    whole-port average. Here the reference is re-averaged with centre-ray classification only (pixel centres sample the
    port uniformly) and every ruled group is re-compared.
(2) Composition: bias of the K-only arm (order0: no C coupling) next to the composed arm for the scenes that have one.
"""
import json
import sys
from pathlib import Path

import numpy as np

import p53_compose as CP
import p53_fixtures as FX
import p53_render as RD
import p53_report as RP


def main():
    R = Path(sys.argv[1]); rd, od, cd = R / "render", R / "oracle", R / "cycles"; out = {}
    z = np.load(od / "g2_pred.npz")
    cen = {}
    for gj in sorted(cd.glob("ccam_*.json")):
        tag = gj.stem[5:]
        if tag not in ("iso_L1", "iso_L2", "iso_L3", "gap4", "gap8", "floor", "lblock", "lblock_pose1", "bend30", "bend90", "dent"):
            continue
        geo = json.load(open(gj)); sd = FX.scene_def(tag); ss = RD.sceneset_of(sd); cen[tag] = {}
        for cam, g in geo.items():
            pid = CP.classify(ss, RD.blender_ortho_rays(g, offsets=((0.5, 0.5),)))
            ms = [CP.port_means(pid, np.load(f).reshape(-1, 3), ss.Ptot) for f in sorted(cd.glob(f"cimg_{tag}_{cam}_*.npy"))]
            cen[tag][cam] = (np.stack([m[0] for m in ms]), ms[0][1])
    # (1a) isolated slab: TR exact vs reference, interior vs centre classification
    g2 = {}
    for name in ("iso_L1", "iso_L2", "iso_L3"):
        sd = FX.scene_def(name); grp = RP.port_groups(name); inter = RP.load_ports(rd / f"cports_{name}.npz"); g2[name] = {}
        for cam in sd["cams"]:
            k = ("top", "bot", "otop", "obot").index(RP.dirkey(sd["cams"][cam]["dir"]))
            a = RP.compare(inter[cam][0], z[f"{name}__L_exact"][k], inter[cam][1], grp, 0.015)
            b = RP.compare(cen[name][cam][0], z[f"{name}__L_exact"][k], np.where(inter[cam][1] >= RP.MINPIX, cen[name][cam][1], 0), grp, 0.015)
            g2[name][cam] = {gname: dict(interior_bias=a[gname]["bias"], centre_bias=b[gname]["bias"], se=a[gname]["se"], ruled=a[gname]["ruled"])
                             for gname in a if any(a[gname]["ruled"])}
    out["tr_exact_interior_vs_centre"] = g2
    # (1b) compositions
    comp = {}
    arms = [("gap4", "part_s1", "nee"), ("gap8", "part_s1", "nee"), ("floor", "part_s1", "nee"), ("lblock", "part_s1", "nee"),
            ("lblock_pose1", "reuse_pose0", "nee"), ("lblock_pose1", "recomputed", "nee"), ("gap4", "order0", "nee"), ("gap8", "order0", "nee"),
            ("floor", "order0", "nee"), ("bend30", "current", "nee"), ("bend90", "current", "nee"), ("dent", "current", "nee"),
            ("bend30", "stale_rest", "bin"), ("bend90", "stale_rest", "bin"), ("dent", "stale_rest", "bin"), ("bend30", "current", "bin"),
            ("bend90", "current", "bin"), ("dent", "current", "bin"), ("gap4", "part_s1", "bin"), ("floor", "part_s1", "bin"), ("lblock", "part_s1", "bin")]
    for scene, arm, mode in arms:
        p = od / "pred" / f"{scene}__{arm}.npz"
        if not p.exists():
            continue
        P = np.load(p); sd = FX.scene_def(scene); grp = RP.port_groups(scene); inter = RP.load_ports(rd / f"cports_{scene}.npz"); r = {}
        for cam in sd["cams"]:
            key = f"{RP.dirkey(sd['cams'][cam]['dir'])}__{mode}"
            if key not in P.files:
                continue
            a = RP.compare(inter[cam][0], P[key], inter[cam][1], grp, 0.03)
            b = RP.compare(cen[scene][cam][0], P[key], np.where(inter[cam][1] >= RP.MINPIX, cen[scene][cam][1], 0), grp, 0.03)
            r[cam] = {g: dict(interior_bias=a[g]["bias"], centre_bias=b[g]["bias"], se=a[g]["se"], share=a[g]["share"], pass_interior=a[g]["pass_"], pass_centre=b[g]["pass_"])
                      for g in a if any(a[g]["ruled"])}
        comp[f"{scene}__{arm}__{mode}"] = r
    out["composition_interior_vs_centre"] = comp
    json.dump(out, open(R / "attrib_posthoc.json", "w"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    print("written", R / "attrib_posthoc.json")


if __name__ == "__main__":
    main()
