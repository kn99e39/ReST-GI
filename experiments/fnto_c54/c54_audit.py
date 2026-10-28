"""WL54 Stage A: independent audit of the WL53 composition evidence (protocol fnto_c54).

    P53_MI_VARIANT=cuda_ad_rgb python c54_audit.py <stage> <wl53 results dir> <out dir>
stages
  replay  recompute the G3 composition tables from the stored predictions and stored Cycles port averages; compare to
          report.json (claims A, D). No new render, no new C.
  unit    re-extract the Cycles port averages from the stored Cycles images with two comparison units
            V0  WL53 'interior' unit: centre and four corner rays hit the same port
            V1  centre-ray unit: the pixel centre ray (the Cycles image is point-sampled: GAUSSIAN filter 0.01 px)
          and compare BOTH against the same stored WL53 predictions (claims B, D).
  rerun   run the production composition (WL53 code, same seeds) again into <out>/oracle and diff the predictions against
          the stored ones (claim A, determinism).
"""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("P53_MI_VARIANT", "cuda_ad_rgb")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "phys_ref53"))
import numpy as np  # noqa: E402

import p53_compose as CP  # noqa: E402
import p53_fixtures as FX  # noqa: E402
import c54_common as CC  # noqa: E402
import p53_report as RP  # noqa: E402

V0_OFF = ((0.5, 0.5), (0.1, 0.1), (0.9, 0.1), (0.1, 0.9), (0.9, 0.9))
V1_OFF = ((0.5, 0.5),)


jclean = CC.jclean


def stage_replay(R, out):
    rep = json.load(open(R / "report.json")); od, rd = R / "oracle" / "pred", R / "render"
    res = {}; worst = 0.0
    for scene in ("gap4", "gap8", "floor"):
        mit = RP.load_ports(rd / f"cports_{scene}.npz"); grp = RP.port_groups(scene); sd = FX.scene_def(scene)
        P1 = np.load(od / f"{scene}__part_s1.npz"); P2 = np.load(od / f"{scene}__part_s2.npz"); stored = rep["G3"][scene]["part_s1__nee"]["cams"]
        for cam, (ref2, cnt) in mit.items():
            key = f"{RP.dirkey(sd['cams'][cam]['dir'])}__nee"
            if key not in P1.files:
                continue
            new = RP.compare(ref2, P1[key], cnt, grp, 0.03, pred2=np.stack([P1[key], P2[key]]))
            for g, v in new.items():
                d = float(np.max(np.abs(np.array(v["bias"]) - np.array(stored[cam][g]["bias"]))))
                worst = max(worst, d)
                res[f"{scene}:{cam}:{g}"] = dict(bias=v["bias"], stored=stored[cam][g]["bias"], ruled=v["ruled"][0], max_abs_diff=d)
    res["_worst_abs_diff_vs_report"] = worst
    json.dump(jclean(res), open(out / "replay.json", "w"), indent=1)
    print("replay worst |bias - stored| =", worst, " groups:", len(res) - 1)


ARMS = [("floor_lam", "floor_lam", "lambert", "bin"), ("gap8", "gap8", "part_s1", "nee"), ("gap4", "gap4", "part_s1", "nee"), ("floor", "floor", "part_s1", "nee")]


def stage_unit(R, out):
    pdir, cdir = R / "oracle" / "pred", R / "cycles"; res = {}
    for tag, scene, arm, mode in ARMS:
        sd = FX.scene_def(scene); ss = CC.sceneset_of(sd); grp = RP.port_groups(scene); geo = json.load(open(cdir / f"ccam_{tag}.json"))
        P = np.load(pdir / f"{tag}__{arm}.npz"); stored = np.load(R / "render" / f"cports_{tag}.npz")
        res[tag] = {}
        for cam, g in geo.items():
            key = f"{RP.dirkey(sd['cams'][cam]['dir'])}__{mode}"
            if key not in P.files:
                continue
            files = sorted(cdir.glob(f"cimg_{tag}_{cam}_*.npy")); imgs = [np.load(f).reshape(-1, 3) for f in files]; rec = {}
            for name, off in (("V0", V0_OFF), ("V1", V1_OFF)):
                pid = CP.classify(ss, CC.blender_ortho_rays(g, off))
                means = [CP.port_means(pid, im, ss.Ptot) for im in imgs]
                ref2 = np.stack([m[0] for m in means]); cnt = means[0][1]
                if name == "V0":
                    rec["V0_matches_stored_cports"] = float(np.max(np.abs(ref2 - stored[f"{cam}__mean"])))
                rec[name] = RP.compare(ref2, P[key], cnt, grp, 0.03)
                rec[f"{name}_pixels"] = int(cnt.sum())
            res[tag][cam] = rec
        print("unit", tag, flush=True)
    json.dump(jclean(res), open(out / "unit.json", "w"), indent=1)
    # compact table of the WL53-ruled groups under both units
    rows = []
    for tag in res:
        for cam, rec in res[tag].items():
            for gname, v in rec["V0"].items():
                if not any(v["ruled"]):
                    continue
                w = rec["V1"].get(gname)
                rows.append((tag, cam, gname, v["n_ports"], v["share"][0], v["bias"], w["bias"] if w else None))
    with open(out / "unit_table.txt", "w") as f:
        for r in rows:
            line = f"{r[0]:10s} {r[1]:9s} {r[2]:14s} n={r[3]:3d} share={r[4]:.2f}  V0={[round(100 * x, 1) for x in r[5]]}  V1={[round(100 * x, 1) for x in r[6]] if r[6] else None}"
            f.write(line + "\n"); print(line)


def stage_rerun(R, out):
    import torch
    import p53_oracle as OR
    o = out / "oracle"; o.mkdir(parents=True, exist_ok=True)
    t = o / "T"
    if not t.exists():
        os.symlink(R / "oracle" / "T", t)
    only = [a for a in sys.argv[4:]] or ["gap8"]
    if "floor_lam" in only:
        OR.stage_lamfloor(o, OR.CFG); only = [a for a in only if a != "floor_lam"]
    if only:
        OR.stage_compose(o, OR.CFG, only=only)
    diffs = {}
    for f in sorted((o / "pred").glob("*.npz")):
        old = R / "oracle" / "pred" / f.name
        if not old.exists():
            continue
        a, b = np.load(f), np.load(old)
        diffs[f.stem] = {k: float(np.max(np.abs(a[k] - b[k])) / max(1e-30, float(np.max(np.abs(b[k]))))) for k in a.files if k in b.files}
    json.dump(jclean(diffs), open(out / "rerun_diff.json", "w"), indent=1)
    print(json.dumps(jclean(diffs), indent=1)[:3000])


if __name__ == "__main__":
    st, R, out = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3]); out.mkdir(parents=True, exist_ok=True)
    {"replay": stage_replay, "unit": stage_unit, "rerun": stage_rerun}[st](R, out)
