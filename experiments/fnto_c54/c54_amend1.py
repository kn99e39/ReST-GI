"""WL54 amendment 1 analysis (post hoc, labelled): comparison tables under the declared (>= 30 pixels) and the amended (>= 90 % of the expected
pixel count) coverage rules, partially visible groups listed separately, and the Cycles pixel-size diagnostic.
    P53_MI_VARIANT=cuda_ad_rgb python c54_amend1.py tables <out json> <cycles dir> <stageB dir> <stageC dir> <fixture,...>
    P53_MI_VARIANT=cuda_ad_rgb python c54_amend1.py res    <out json> <cycles_res dir> <stageB dir>
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "phys_ref53"))
import c54_common as CC  # noqa: E402
import c54_fixtures as CF  # noqa: E402
import c54_stageB as SB  # noqa: E402
import p53_compose as CP  # noqa: E402

COV = 0.9


def expected_pixels(ss, geo):
    c = -np.asarray(geo["fwd"], float); c /= np.linalg.norm(c); px = geo["width"] / geo["res"]
    ct = torch.tensor(c, dtype=torch.float32, device="cuda")
    proj = torch.zeros(ss.Ptot, device="cuda").index_add_(0, ss.tri_port, ss.area * (ss.n * ct).sum(1).clamp(min=0))
    return proj.cpu().numpy() / px ** 2


def tables(out, cdir, bdir, sdir, fixtures):
    res = {}
    for name in fixtures:
        sd = CF.scene_def(name); groups = CF.coarse_labels(name); ss = CC.sceneset_of(sd); arr = dict(np.load(bdir / f"{name}_coarse_arrays.npz"))
        if (sdir / f"{name}_casp_arrays.npz").exists():
            arr.update({k: v for k, v in np.load(sdir / f"{name}_casp_arrays.npz").items() if k.startswith("C1")})
        geo, p1 = SB.cycles_ports(cdir, name, sd, ss, SB.V1_OFF); res[name] = {}
        arms = [a for a in ("C0prod", "C0inf", "F4", "F8", "F16", "G16", "C1", "C1_c1", "C1_c4", "C1_b4") if a in arr]
        for cam, g in geo.items():
            ref, cnt = p1[cam]; refm = ref.mean(0); se = ref.std(0, ddof=1) / np.sqrt(len(ref)); exp = expected_pixels(ss, g)
            cov = np.where(exp > 0, cnt / np.maximum(exp, 1e-9), 0.0)
            ok_decl = cnt >= SB.MINPIX; ok_amend = ok_decl & (cov >= COV); partial = ok_decl & (cov < COV)
            rec = dict(n_declared=int(ok_decl.sum()), n_amended=int(ok_amend.sum()), n_partial=int(partial.sum()),
                       partial_groups={gn: dict(n=int(((groups == gn) & partial).sum()), cov_median=float(np.median(cov[(groups == gn) & partial]))) for gn in np.unique(groups[partial])})
            for tag, ok in (("declared", ok_decl), ("amended", ok_amend), ("partial", partial)):
                if not ok.any():
                    continue
                rec[tag] = {arm: {k: v for k, v in SB.group_stats(arr[arm], refm, ok, groups, se).items()} for arm in arms}
            res[name][cam] = rec
    CC.dump(res, out)
    for name in res:
        for cam, rec in res[name].items():
            for tag in ("amended", "partial"):
                if tag not in rec:
                    continue
                for gn, v in rec[tag]["C0prod"].items():
                    if not v["ruled"] and tag == "amended":
                        continue
                    f = rec[tag]["F16"].get(gn, {}).get("bias")
                    c1 = rec[tag].get("C1", {}).get(gn, {}).get("bias")
                    print(f"{name:10s} {cam:9s} {tag:8s} {gn:12s} n={v['n']:3d} C0={100 * v['bias']:+6.1f}%  F16={100 * f:+6.1f}%" + (f"  C1={100 * c1:+6.1f}%" if c1 is not None else ""))


def resolution(out, rdir, bdir):
    name = "gap4"; sd = CF.scene_def(name); groups = CF.coarse_labels(name); ss = CC.sceneset_of(sd); arr = np.load(bdir / f"{name}_coarse_arrays.npz"); geo = json.load(open(rdir / f"ccam_{name}.json")); res = {}
    for key, g in geo.items():
        cam, px = key.split("@"); pid = CP.classify(ss, CC.blender_ortho_rays(g, SB.V1_OFF)); img = np.load(rdir / f"cimg_{name}_{cam}_px{px}.npy").reshape(-1, 3)
        m, cnt = CP.port_means(pid, img, ss.Ptot); m = m[:, 0]; ok = cnt >= 0.9 * expected_pixels(ss, g)
        gname = "slab_lo:top" if cam == "gapdown" else "slab_hi:bot"; sel = ok & (groups == gname)
        res[key] = dict(group=gname, n=int(sel.sum()), **{arm: float((arr[arm][sel] - m[sel]).sum() / m[sel].sum()) for arm in ("C0prod", "F16", "G16")}, signal=float(m[sel].sum()))
        print(key, json.dumps(res[key]))
    for cam, gname in (("gapdown", "slab_lo:top"), ("gapup", "slab_hi:bot")):
        e5, e125 = abs(res[f"{cam}@0.5"]["F16"]), abs(res[f"{cam}@0.125"]["F16"])
        res[f"{cam}_rule"] = dict(F16_bias_0p5=res[f"{cam}@0.5"]["F16"], F16_bias_0p125=res[f"{cam}@0.125"]["F16"], moves_toward_F16=bool(e125 <= 0.5 * e5), applies=bool(e5 > 0.03))
    CC.dump(res, out)


if __name__ == "__main__":
    if sys.argv[1] == "tables":
        tables(Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4]), Path(sys.argv[5]), sys.argv[6].split(","))
    else:
        resolution(Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4]))
