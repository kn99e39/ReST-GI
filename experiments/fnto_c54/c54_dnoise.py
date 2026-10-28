"""WL54: Monte-Carlo noise of the paired stale-vs-current K differential. D = pred(stale K) - pred(current K) is computed with BOTH stored rest-pose
oracle seeds (slab_s1 5311, slab_s2 5312) as the stale K, same C, same sun term; the difference of the two D estimates is the noise of D from the
oracle operator alone (the current K is a single seed, its noise adds, so this is a lower bound).
    P53_MI_VARIANT=cuda_ad_rgb python c54_dnoise.py <out json> <wl53 results> <scene,...>"""
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "phys_ref53"))
import c54_casp as CA  # noqa: E402
import c54_common as CC  # noqa: E402
import c54_stageB as SB  # noqa: E402
import c54_stageE as SE  # noqa: E402
import p53_fixtures as FX  # noqa: E402
import p53_oracle as OR  # noqa: E402
import p53_report as RP  # noqa: E402

out = Path(sys.argv[1]); R = Path(sys.argv[2]); res = {}
for scene in sys.argv[3].split(","):
    cfg = SE.SCENES[scene]; sd = FX.scene_def(scene); groups = RP.port_groups(scene); ss0 = OR.scene_set(sd); odir = R / "oracle"
    Tc = OR.load_T(odir, cfg["T"]); Tm = {ci: Tc["T"] for ci, c in enumerate(ss0.comps) if c.kind == "medium"}; KCm = {ci: Tc["KC"] for ci in Tm}
    sun = SE.prepare_sun(sd, ss0, cfg["cams"]); lev, nu, _ = CA.assign_levels(ss0); off = 0; lbc = []
    for ci, comp in enumerate(ss0.comps):
        lbc.append(lev[off:off + comp.P] if comp.kind == "lambert" else np.ones(comp.P, int)); off += comp.P
    ov, par = SE.comps_with_levels(sd, lbc); SE.scene_name_global = scene
    cur, _ = SE.run_arm(sd, ov, par, sun, Tm, KCm, cfg["cams"], cfg["kci"], ("bin",), (5810, 5811), label="cur")
    D = {}
    for tag in ("slab_s1", "slab_s2"):
        S = OR.load_T(odir, tag); st, _ = SE.run_arm(sd, ov, par, sun, {ci: S["T"] for ci in Tm}, None, cfg["cams"], cfg["kci"], ("bin",), (5810, 5811), label="stale_" + tag); D[tag] = (st, S["T"].shape); del S; torch.cuda.empty_cache()
    res[scene] = {}
    for cam in ("top", "otop", "bot"):
        key = f"{cam}__bin"; d1 = D["slab_s1"][0][key] - cur[key]; d2 = D["slab_s2"][0][key] - cur[key]; res[scene][cam] = {}
        gsel = np.unique(groups)
        for gn in gsel:
            m = (groups == gn)
            if m.sum() < 5:
                continue
            ref = cur[key][m].sum(0)
            res[scene][cam][gn] = dict(D_s1=(d1[m].sum(0) / np.maximum(ref, 1e-30)).tolist(), D_s2=(d2[m].sum(0) / np.maximum(ref, 1e-30)).tolist(),
                                       noise=((d1[m].sum(0) - d2[m].sum(0)) / np.maximum(ref, 1e-30)).tolist())
    print("done", scene, flush=True)
CC.dump(res, out)
