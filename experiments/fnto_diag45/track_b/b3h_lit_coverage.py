"""WL45 Track B3h (amendment 6, descriptive, CPU) - source-side excitation of K's nonlocal parts in training: in how many
of the 1,000 training probe images (prep2_train, the K_k44P training set) is each Lego port directly lit by at least
one of the image's lights (port_vis > 0 for some j < nlights)? Reported for Yellow ports overall and for the port
classes of B3g on held-out T0 views 6 / 15 (visible in the view / observed in training only / never observed), area
weighted: fraction never lit, median lit-image count.

    python b3h_lit_coverage.py <out dir>
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "fnto_lego_r1")); sys.path.insert(0, str(HERE.parents[1] / "fnto_lego"))
import r1_common as C  # noqa: E402


def main(out):
    out = Path(out); An = np.load(C.OUT1 / "anchors.npz"); N = len(An["obj"]); nA = int(An["n_A"]); Ar = An["area"].astype(np.float64)
    pr = C.OUT1 / "stagea/prep2_train"; pv = np.load(pr / "port_vis.npy", mmap_mode="r"); nl = np.load(pr / "nlights.npy")
    lit = np.zeros(N, np.int64)
    for i in range(pv.shape[0]):
        lit += (np.asarray(pv[i, :max(int(nl[i]), 1)]).max(0) > 0)
    pp = np.load(pr / "px_port.npy", mmap_mode="r"); pd = np.load(pr / "px_dom.npy", mmap_mode="r"); cnt = np.zeros(N, np.int64)
    for s in range(0, len(pp), 1 << 24):
        a, d = np.asarray(pp[s:s + (1 << 24)]), np.asarray(pd[s:s + (1 << 24)]); cnt[:nA] += np.bincount(a[(d == 0) & (a >= 0) & (a < nA)], minlength=nA)
    yp = (An["mat"] == 0) & (An["obj"] == 0); lego = An["obj"] == 0
    st = lambda m: {"area_share_of_yellow": float(Ar[m].sum() / Ar[yp].sum()), "area_never_lit": float(Ar[m & (lit == 0)].sum() / max(Ar[m].sum(), 1e-30)),  # noqa: E731
                    "area_lit_le_5_images": float(Ar[m & (lit <= 5)].sum() / max(Ar[m].sum(), 1e-30)), "median_lit_images": float(np.median(lit[m])) if m.any() else None}
    rec = {"images": int(pv.shape[0]), "yellow_all": st(yp), "yellow_observed": st(yp & (cnt > 0)), "yellow_never_observed": st(yp & (cnt == 0)),
           "other_lego_area_never_lit": float(Ar[lego & ~yp & (lit == 0)].sum() / Ar[lego & ~yp].sum())}
    px = np.load(C.OUT1 / "eval_px/heldout_T0.npz")
    for v in (6, 15):
        val = px[f"v{v}_valid"].astype(bool); visp = np.zeros(N, bool); visp[px[f"v{v}_port"][val].astype(np.int64)] = True
        rec[f"view{v}"] = {nm: st(yp & m) for nm, m in (("visible", visp), ("observed", ~visp & (cnt > 0)), ("never", ~visp & (cnt == 0)))}
    (out / "b3h_lit_coverage.json").write_text(json.dumps(rec, indent=1)); print(json.dumps(rec, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
