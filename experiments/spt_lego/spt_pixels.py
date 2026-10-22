"""SPT-1 pixel -> anchor interpolation for an RNA-format dataset (any venv with h5py + scipy).

    python spt_pixels.py <dataset name> <state> <owner cache suffix>
    e.g. python spt_pixels.py train_T0 T0 9e9dce083ac9

Pixels: alpha > 0 (the upstream RNA mask), in H5 order. Object from the B4 owner cache (label 16 = box B). Canonical
position = world - offset(state, object). Up to N_INT same-object anchors with normal . anchor normal > 0.5, nearest
first, weights 1/(d + 2 mm) normalised; if none is compatible, the nearest anchor alone. Geometry only: no radiance.
Writes results/spt_lego/pixels/<dataset>.{idx,w,obj,start}.npy
"""

from __future__ import annotations

import json
import sys
import time

import h5py
import numpy as np
from scipy.spatial import cKDTree

import spt_common as C

N_INT = 4


def main(name, state, key):
    A = np.load(C.OUT / "anchors.npz"); pos, nrm, obj = A["pos"].astype(np.float64), A["nrm"].astype(np.float64), A["obj"]
    trees = {o: (cKDTree(pos[obj == o]), np.flatnonzero(obj == o)) for o in (0, 1)}
    offs = C.offsets(state)
    f = h5py.File(C.B4 / f"data/{name}.h5", "r"); own = C.B4 / f"data/{name}.h5.owners_b4_{key}"
    n_img = f["alpha"].shape[0]
    IDX, W, OBJ, start = [], [], [], [0]
    t0 = time.time(); fallback = 0; dmax = []
    for i in range(n_img):
        a = np.nan_to_num(f["alpha"][i, :, 0].astype(np.float32)); m = a > 0
        p = f["position"][i][m].astype(np.float64); n = f["normal"][i][m].astype(np.float64); c = f["camera_dir"][i][m].astype(np.float64)
        n = np.where(((n * c).sum(1) < 0)[:, None], -n, n)
        lab = np.load(own / f"{i:04d}.npz")["lab"]; assert len(lab) == len(p), (i, len(lab), len(p))
        o = (lab == C.B_UNIT).astype(np.int64)
        idx = np.full((len(p), N_INT), -1, np.int64); w = np.zeros((len(p), N_INT))
        for ob in (0, 1):
            mm = o == ob
            if not mm.any():
                continue
            tree, gidx = trees[ob]
            d, ii = tree.query(p[mm] - offs[ob], k=8, workers=-1)
            cand = gidx[ii]
            ok = (nrm[cand] * n[mm][:, None, :]).sum(-1) > 0.5
            rank = np.where(ok, np.arange(8)[None], 99)
            sel = np.argsort(rank, 1, kind="stable")[:, :N_INT]
            okk = np.take_along_axis(ok, sel, 1); dd = np.take_along_axis(d, sel, 1); cc = np.take_along_axis(cand, sel, 1)
            none = ~okk.any(1); fallback += int(none.sum())
            okk[none, 0] = True
            ww = np.where(okk, 1.0 / (dd + 2e-3), 0.0); ww /= ww.sum(1, keepdims=True)
            idx[mm] = np.where(okk, cc, -1); w[mm] = ww; dmax.append(dd[:, 0])
        IDX.append(idx.astype(np.int32)); W.append(w.astype(np.float16)); OBJ.append(o.astype(np.int8)); start.append(start[-1] + len(p))
    out = C.OUT / "pixels"; out.mkdir(parents=True, exist_ok=True)
    np.save(out / f"{name}.idx.npy", np.concatenate(IDX)); np.save(out / f"{name}.w.npy", np.concatenate(W))
    np.save(out / f"{name}.obj.npy", np.concatenate(OBJ)); np.save(out / f"{name}.start.npy", np.asarray(start, np.int64))
    dmax = np.concatenate(dmax)
    rec = {"dataset": name, "state": state, "images": n_img, "pixels": int(start[-1]), "fallback_nearest_only": fallback,
           "nearest_anchor_dist_q50_q99_m": [float(np.quantile(dmax, 0.5)), float(np.quantile(dmax, 0.99))], "seconds": time.time() - t0,
           "anchors_sha16": C.sha16(C.OUT / "anchors.npz")}
    (out / f"{name}.json").write_text(json.dumps(rec, indent=1)); print(rec)


if __name__ == "__main__":
    main(*sys.argv[1:4])
