"""WL54 temporal change tracking (Lambert, exact K): T0 = floor_lam, T1 = rot_lam (the box is rotated 17 deg and shifted (3.7, -2.1) mm; the coarse ports of
the box and the floor keep their ids, so port changes are material-consistent). Per camera and group: change of the port radiance T1 - T0 of each arm vs the Cycles
change (centre-ray unit, amended coverage in BOTH poses), regression slope and correlation over ports, and the pooled change error.
    P53_MI_VARIANT=cuda_ad_rgb python c54_change.py <out json> <cycles dir> <stageB dir> <stageC dir>"""
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

out, cdir, bdir, sdir = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])
A0, A1 = "floor_lam", "rot_lam"; groups = CF.coarse_labels(A0); assert (groups == CF.coarse_labels(A1)).all()
arr = {n: dict(np.load(bdir / f"{n}_coarse_arrays.npz")) | {k: v for k, v in np.load(sdir / f"{n}_casp_arrays.npz").items() if k.startswith("C1")} for n in (A0, A1)}
res = {}
for cam in ("underup", "underdown", "top", "otop"):
    data = {}
    for n in (A0, A1):
        sd = CF.scene_def(n); ss = CC.sceneset_of(sd); geo, p1 = SB.cycles_ports(cdir, n, sd, ss, SB.V1_OFF); ref, cnt = p1[cam]
        cov = cnt / np.maximum(AM.expected_pixels(ss, geo[cam]), 1e-9); data[n] = (ref.mean(0), (cnt >= SB.MINPIX) & (cov >= AM.COV))
    ok = data[A0][1] & data[A1][1]; dref = data[A1][0] - data[A0][0]; res[cam] = {}
    for arm in ("C0prod", "C1", "C1_c4", "F16", "G16"):
        dy = arr[A1][arm] - arr[A0][arm]; r = {}
        for g in np.unique(groups[ok]):
            m = ok & (groups == g)
            if m.sum() < 5 or np.abs(dref[m]).sum() < 1e-6 * np.abs(data[A0][0][m]).sum():
                continue
            x, y = dref[m], dy[m]; slope = float((x * y).sum() / (x * x).sum()); corr = float(np.corrcoef(x, y)[0, 1]) if x.std() > 0 and y.std() > 0 else None
            r[g] = dict(n=int(m.sum()), slope=slope, corr=corr, pooled_change_err=float((y - x).sum() / np.abs(x).sum()), ref_change_rel_to_signal=float(x.sum() / data[A0][0][m].sum()))
        res[cam][arm] = r
CC.dump(res, out)
for cam in res:
    for arm in ("C0prod", "C1", "F16"):
        for g, v in res[cam][arm].items():
            print(cam, arm, g, {k: (round(x, 3) if isinstance(x, float) else x) for k, x in v.items()})
