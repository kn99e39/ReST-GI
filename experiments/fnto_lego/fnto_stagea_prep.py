"""FNTO Stage A preprocessing (8dna26 .venv-cu128: numpy, scipy, Mitsuba).

    python fnto_stagea_prep.py <split> [<split> ...]       # train / holdout / superpose

Per Stage A image (fnto_stagea_render.py output), from geometry only (no radiance is read except the target `col`):
  pixels : domain (A / B by the image's box placement), canonical position (domain frame), camera-facing shading normal,
           camera direction, target radiance, sun visibility, nearest normal-compatible port of the same domain, material;
  ports  : incident sunlight of every port, E(s) = E_SUN max(0, n_s . w_L) V(s), V traced in the image's own geometry
           (Lego, box at its offset or absent, the shadow-only masks), and the hemispherical bin of w_L in s's frame.
The incident field of a Stage A image is fully known (one delta sun times traced visibility); a model trained on these
pairs (incident field -> first-intrinsic-event radiance) never sees free-space interreflection or the held-out T1 edit.
Two-sun linearity probes (split superpose, part 2) store each sun's incident field separately (portE / portE2, ...):
the model is applied to their SUM, never to a fitted combination.
Writes results/fnto_lego/stagea/prep_<split>/{px_*.npy, start.npy, portE[2].npy, portbin[2].npy, suns[2].npy, meta.json}.
"""

from __future__ import annotations

import json
import sys
import time

import numpy as np
from scipy.spatial import cKDTree

import fnto_common as C
import fnto_relations as R

D = C.OUT / "stagea"


def nearest_ports(A, dom, canon, n, trees):
    out = np.full(len(canon), -1, np.int64)
    for o in (0, 1):
        m = dom == o
        if not m.any():
            continue
        tree, g = trees[o]
        _, ii = tree.query(canon[m], k=8, workers=-1)
        cand = g[ii]
        ok = (A["nrm"][cand] * n[m][:, None, :]).sum(-1) > 0.5
        first = np.where(ok.any(1), np.argmax(ok, 1), 0)
        out[m] = cand[np.arange(len(cand)), first]
    return out


def det_frame_np(n):
    return C.frames(n.astype(np.float64), np.zeros(len(n)))


def main(split):
    A = R.load_anchors()
    Np = len(A["obj"])
    cfgs = {c["i"]: c for c in json.loads((D / split / "configs.json").read_text())}
    ids = sorted(cfgs)
    od = D / f"prep_{split}"; od.mkdir(parents=True, exist_ok=True)
    trees = {o: (cKDTree(A["pos"][A["obj"] == o]), np.flatnonzero(A["obj"] == o)) for o in (0, 1)}
    bV = np.load(C.BOX_T0)["V"]; blo, bhi = bV.min(0), bV.max(0)
    t_, b_, n_ = det_frame_np(A["nrm"])
    sizes = []
    for i in ids:
        z = np.load(D / split / f"img_{i:05d}.npz", mmap_mode="r"); sizes.append(len(z["pix"]))
    start = np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64); P = int(start[-1])
    mm = lambda k, dt, w: np.lib.format.open_memmap(od / f"px_{k}.npy", "w+", dt, (P, w) if w > 1 else (P,))  # noqa: E731
    pos, nrm, cam, col = mm("pos", np.float32, 3), mm("nrm", np.float16, 3), mm("cam", np.float16, 3), mm("col", np.float32, 3)
    vis, dom, port, pix = mm("vis", np.uint8, 1), mm("dom", np.uint8, 1), mm("port", np.int32, 1), mm("pix", np.int32, 1)
    two = any(cfgs[i].get("sun2") is not None for i in ids)
    names = ("", "2") if two else ("",)
    portE = [np.lib.format.open_memmap(od / f"portE{x}.npy", "w+", np.float16, (len(ids), Np)) for x in names]
    portbin = [np.lib.format.open_memmap(od / f"portbin{x}.npy", "w+", np.uint8, (len(ids), Np)) for x in names]
    suns = [np.zeros((len(ids), 3), np.float32) for _ in names]; t0 = time.time(); trace_s = 0.0
    for j, i in enumerate(ids):
        cfg = cfgs[i]; z = np.load(D / split / f"img_{i:05d}.npz")
        s, e = start[j], start[j + 1]
        p = z["pos"].astype(np.float64); n = z["nrm"].astype(np.float32); c = z["cam"].astype(np.float32)
        n = np.where(((n * c).sum(1) < 0)[:, None], -n, n)
        bo = None if cfg["box_offset"] is None else np.asarray(cfg["box_offset"], np.float64)
        d = np.zeros(len(p), np.uint8)
        if bo is not None:
            d[np.all((p >= blo + bo - 1e-3) & (p <= bhi + bo + 1e-3), 1)] = 1
        canon = p - np.where(d[:, None] == 1, bo if bo is not None else 0, 0)
        pos[s:e] = canon; nrm[s:e] = n; cam[s:e] = c; col[s:e] = z["col"]; vis[s:e] = z["vis"]; dom[s:e] = d; pix[s:e] = z["pix"]
        port[s:e] = nearest_ports(A, d, canon, n, trees)
        t = time.time()
        geo = R.Geo(np.zeros(3), bo, masks=cfg["masks"])
        present = np.flatnonzero((A["obj"] == 0) | (bo is not None))
        for slot, key in ((0, "sun"), (1, "sun2")):
            if cfg.get(key) is None:
                continue
            w = np.asarray(cfg[key], np.float64); w /= np.linalg.norm(w)
            cosv = np.clip(A["nrm"][present] @ w, 0, None)
            lit = present[cosv > 0]
            V = np.zeros(Np, bool)
            if len(lit):
                V[lit] = R.sun_visible(A, geo, lit, w)
            E = np.zeros(Np, np.float32); E[present] = C.E_SUN * cosv * V[present]
            portE[slot][j] = E.astype(np.float16)
            portbin[slot][j] = C.hemi_bin_local(np.stack([t_ @ w, b_ @ w, n_ @ w], 1)).astype(np.uint8)
            suns[slot][j] = w
        trace_s += time.time() - t
        if j % 50 == 0:
            print(f"{split} {j}/{len(ids)} {time.time() - t0:.0f}s trace {trace_s:.0f}s", flush=True)
    for a in (pos, nrm, cam, col, vis, dom, port, pix, *portE, *portbin):
        a.flush()
    np.save(od / "start.npy", start); np.save(od / "ids.npy", np.asarray(ids))
    for x, sn in zip(names, suns):
        np.save(od / f"suns{x}.npy", sn)
    meta = {"split": split, "images": len(ids), "pixels": P, "pixels_B": int(np.asarray(dom).sum()), "ports": Np,
            "anchors_sha16": C.sha16(C.OUT / "anchors.npz"), "seconds": time.time() - t0, "trace_s": trace_s}
    (od / "meta.json").write_text(json.dumps(meta, indent=1)); print(meta)


if __name__ == "__main__":
    for sp in sys.argv[1:]:
        main(sp)
