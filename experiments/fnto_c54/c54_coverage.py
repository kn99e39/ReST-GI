"""WL54 amendment-1 diagnostic: sensitivity of the group biases to the per-port pixel-coverage threshold. The protocol declared >= 30 pixels per
port (copied from WL53, where pixels were 1 mm and a port had ~100 pixels); at 0.5 mm that is 7.5 % of a port. Reports the biases for
coverage >= 30 px (declared), >= 30 % and >= 90 % of the port's full pixel count, per arm.
    python c54_coverage.py <out json> <cycles dir> <stageB dir> <fixture,...>
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "phys_ref53"))
import c54_common as CC  # noqa: E402
import c54_fixtures as CF  # noqa: E402
import c54_stageB as SB  # noqa: E402
import p53_compose as CP  # noqa: E402


def main():
    out = Path(sys.argv[1]); cdir, bdir = Path(sys.argv[2]), Path(sys.argv[3]); res = {}
    for name in sys.argv[4].split(","):
        sd = CF.scene_def(name); groups = CF.coarse_labels(name); ss = CC.sceneset_of(sd); arr = np.load(bdir / f"{name}_coarse_arrays.npz")
        geo, p1 = SB.cycles_ports(cdir, name, sd, ss, SB.V1_OFF); res[name] = {}
        for cam, g in geo.items():
            ref, cnt = p1[cam]; refm = ref.mean(0); se = ref.std(0, ddof=1) / np.sqrt(len(ref))
            px_full = {}
            # full-port pixel count: pixels per unit area = 1 / px^2; a port of area A_p seen with projected-area factor c has A_p c / px^2 pixels
            res[name][cam] = {}
            for thr_name, rule in (("px30", lambda c: c >= 30), ("cov30", lambda c: c >= 0.3 * c.max() if False else None), ("cov90", None)):
                pass
            # full count per port = max over the group's ports of the same face orientation is not general; use the count seen when the port is fully inside the view:
            # estimate from the group's median of the top-quartile counts
            full = np.zeros(len(groups))
            for gname in np.unique(groups):
                m = (groups == gname) & (cnt > 0)
                if m.any():
                    full[groups == gname] = np.percentile(cnt[m], 90)
            for thr_name, frac in (("px30", None), ("cov30", 0.3), ("cov90", 0.9)):
                ok = (cnt >= 30) if frac is None else (cnt >= frac * np.maximum(full, 1))
                if not ok.any():
                    continue
                r = {}
                for arm in ("C0prod", "F16", "G16"):
                    r[arm] = {k: dict(n=v["n"], bias=v["bias"], se=v["se"]) for k, v in SB.group_stats(arr[arm], refm, ok, groups, se).items() if v["ruled"]}
                res[name][cam][thr_name] = r
    CC.dump(res, out)
    for name in res:
        for cam in res[name]:
            for thr, r in res[name][cam].items():
                for g in r["C0prod"]:
                    print(f"{name:10s} {cam:9s} {thr:6s} {g:12s} n={r['C0prod'][g]['n']:3d} C0={100*r['C0prod'][g]['bias']:+6.1f}%  F16={100*r['F16'][g]['bias']:+6.1f}%  G16={100*r['G16'][g]['bias']:+6.1f}%")


if __name__ == "__main__":
    main()
