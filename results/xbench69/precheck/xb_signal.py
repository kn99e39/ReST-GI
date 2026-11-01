"""Track B gate pre-check (read-only): T0 floor -> T1 floor_c4 change of the WL55 Cycles references on exterior views."""
import json, os, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, os.getcwd())
import k55_common as KCm
import k55_fixtures as KF

CYC = Path.home() / "rest_wl55" / "out" / "cycles_cuda"
res = {}
ref = {}
for scene in ("floor", "floor_c4"):
    sd = KF.scene_def(scene); geo = json.load(open(CYC / f"ccam_{scene}.json")); ss = KCm.classify_sceneset(sd)
    lab = KCm.labels_of(sd); ref[scene] = {}
    for cam in ("top", "otop"):
        files = sorted(CYC.glob(f"cimg_{scene}_{cam}_s*.npy"))
        u = KCm.cycles_ports(ss, geo[cam], [np.load(f)[..., :3] for f in files])["amended"]
        ref[scene][cam] = (u, lab)
for cam in ("top", "otop"):
    (u0, lab), (u1, _) = ref["floor"][cam], ref["floor_c4"][cam]
    both = u0["sel"] & u1["sel"]
    d = u1["mean"] - u0["mean"]; sed = np.hypot(u0["se"], u1["se"])
    out = {}
    for g in np.unique(lab[both]):
        m = both & (lab == g); s0 = u0["mean"][m].sum(0); s1 = u1["mean"][m].sum(0)
        se = np.sqrt((sed[m] ** 2).sum(0))
        z = np.abs(d[m]) / np.maximum(sed[m], 1e-30)
        rel = np.abs(d[m]) / np.maximum(np.abs(u0["mean"][m]), 1e-30)
        out[str(g)] = dict(n=int(m.sum()), group_rel_change=((s1 - s0) / s0).tolist(), group_rel_se=(se / s0).tolist(),
                           frac_ports_z3=(z > 3).mean(0).tolist(), frac_ports_rel_gt_3pct_and_z3=((z > 3) & (rel > 0.03)).mean(0).tolist(),
                           abs_change_share=(np.abs(d[m]).sum(0) / s0).tolist(), max_rel=rel.max(0).tolist(),
                           p90_rel=np.percentile(rel, 90, axis=0).tolist())
    res[cam] = dict(n_sel_T0=int(u0["sel"].sum()), n_sel_T1=int(u1["sel"].sum()), n_both=int(both.sum()), groups=out)
json.dump(res, open(sys.argv[1], "w"), indent=1)
print(json.dumps(res, indent=1))
