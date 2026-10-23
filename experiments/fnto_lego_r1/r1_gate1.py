"""WL42 Gate 1 - boundary-state correctness (PC, 8dna26 .venv-cu128: Mitsuba). Batch 6 ports vs r1 ports, same checks.

    python r1_gate1.py <out dir>

G1a origin: distance from the ray origin (Batch 6: centroid + 1 mm along n; r1: medoid surface sample + 0.1 mm) to the
    nearest surface along +-n, and whether the first hit along +n is the inside face of a closed solid.
G1b sun at ports vs the renderer's own per-pixel sun visibility (independent ground truth: Stage A holdout, 120 images,
    the generator's white-diffuse direct pass). Pixels map to their nearest normal-compatible port; the port's sun
    visibility is traced in the image's own geometry (masks, box offset). Batch 6: one binary ray from the port origin.
    r1: the fraction of its S_PORT surface samples that see the sun. Reported: agreement of the port value with the
    mean pixel visibility of its pixels, and the pooled port / pixel direct-irradiance ratio.
Predeclared pass (WL42, before measurement): r1 origins within 0.2 mm of a surface for >= 99 % of ports; first hit along
+n an inside face < 1 %; G1b agreement >= 0.97 and pooled irradiance ratio in [0.95, 1.05].
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

import r1_common as C
import fnto_relations as R

HO = C.OUT / "stagea/holdout"


def origins(A, r1):
    if r1:
        return A["pos"].astype(np.float64) + C.EPS1 * A["nrm"], A["nrm"].astype(np.float64)
    return A["pos"].astype(np.float64) + C.EPS * A["nrm"], A["nrm"].astype(np.float64)


def g1a(A, r1, geo):
    nA = int(A["n_A"]); sel = np.arange(0, nA, 7)
    o, n = origins(A, r1); o, n = o[sel], n[sel]
    vp, _, prim_p, sid_p, tp = geo.intersect(o, n)
    vm, _, prim_m, sid_m, tm = geo.intersect(o, -n)
    V, F = R.mesh_a()
    cr = np.cross(V[F[prim_p, 1]] - V[F[prim_p, 0]], V[F[prim_p, 2]] - V[F[prim_p, 0]]) * A["tri_sign"][prim_p][:, None]
    inside = vp & ~A["tri_two"][prim_p] & ((cr * n).sum(1) > 0)
    d = np.minimum(np.where(vp, tp, np.inf), np.where(vm, tm, np.inf))
    off = C.EPS1 if r1 else C.EPS
    return {"ports": int(len(sel)), "frac_origin_within_0p2mm_of_surface": float((np.abs(d - off) <= 2e-4).mean()) if r1 else float((d <= 2e-4 + off).mean()),
            "dist_q50_q90_q99_mm": (np.quantile(d[np.isfinite(d)], [0.5, 0.9, 0.99]) * 1e3).tolist(), "frac_first_hit_plus_n_inside_face": float(inside.mean())}


def g1b(A, r1):
    cfg = {c["i"]: c for c in json.loads((HO / "configs.json").read_text())}
    trees = {o: (cKDTree(A["pos"][A["obj"] == o]), np.flatnonzero(A["obj"] == o)) for o in (0, 1)}
    bV = np.load(C.BOX_T0)["V"]; blo, bhi = bV.min(0), bV.max(0)
    agree_num, agree_den, Eport, Epix, n_pix = 0.0, 0.0, 0.0, 0.0, 0
    for i in sorted(cfg)[::3]:                                   # 40 of the 120 holdout images
        c = cfg[i]; z = np.load(HO / f"img_{i:05d}.npz")
        p = z["pos"].astype(np.float64); nn = z["nrm"].astype(np.float32); cam = z["cam"].astype(np.float32)
        nn = np.where(((nn * cam).sum(1) < 0)[:, None], -nn, nn); vis = z["vis"].astype(bool)
        bo = None if c["box_offset"] is None else np.asarray(c["box_offset"])
        dom = np.zeros(len(p), np.int64)
        if bo is not None:
            dom[np.all((p >= blo + bo - 1e-3) & (p <= bhi + bo + 1e-3), 1)] = 1
        m = dom == 0
        w = np.asarray(c["sun"], np.float64); w /= np.linalg.norm(w)
        cp = np.clip(nn[m] @ w, 0, None); keep = cp > 0.1
        tree, g = trees[0]
        _, ii = tree.query(p[m][keep], k=8, workers=-1); cand = g[ii]
        ok = (A["nrm"][cand] * nn[m][keep][:, None, :]).sum(-1) > 0.5
        port = cand[np.arange(len(cand)), np.where(ok.any(1), np.argmax(ok, 1), 0)]
        geo = R.Geo(np.zeros(3), bo, masks=c["masks"])
        up = np.unique(port)
        if r1:
            sp = A["spos"][up].astype(np.float64); sn = A["snrm"][up].astype(np.float64); valid = np.isfinite(sp[..., 0])
            o_ = (sp + C.EPS1 * sn)[valid]; lit = (sn[valid] @ w) > 0
            vv = np.zeros(len(o_), bool)
            if lit.any():
                vv[lit] = ~geo.occluded(o_[lit], np.broadcast_to(w, (int(lit.sum()), 3)).copy())
            frac = np.zeros(valid.shape); frac[valid] = vv; Vport = frac.sum(1) / valid.sum(1)
            cosp = np.where(valid, np.clip(sn @ w, 0, None), 0).sum(1) / valid.sum(1)
        else:
            o_ = A["pos"][up].astype(np.float64) + C.EPS * A["nrm"][up]
            Vport = (~geo.occluded(o_, np.broadcast_to(w, (len(up), 3)).copy())).astype(float); cosp = np.clip(A["nrm"][up] @ w, 0, None)
        lut = {int(u): k for k, u in enumerate(up)}; k_ = np.array([lut[int(x)] for x in port])
        pv = vis[m][keep].astype(float)
        mean_pix = np.bincount(k_, pv, len(up)) / np.maximum(np.bincount(k_, minlength=len(up)), 1)
        cntp = np.bincount(k_, minlength=len(up))
        agree_num += float((cntp * (1 - np.abs(Vport - mean_pix))).sum()); agree_den += float(cntp.sum())
        Eport += float((cntp * C.E_SUN * cosp * Vport).sum()); Epix += float((C.E_SUN * cp[keep] * pv).sum()); n_pix += int(keep.sum())
    return {"images": 40, "pixels": n_pix, "port_vs_pixel_visibility_agreement": agree_num / agree_den, "pooled_port_over_pixel_irradiance": Eport / Epix}


def main(out):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    A0 = R.load_anchors(); A1 = dict(np.load(C.OUT1 / "anchors.npz"))
    geo = R.Geo(*C.offsets("T0"))
    rec = {"batch6": {"G1a": g1a(A0, False, geo), "G1b": g1b(A0, False)}, "r1": {"G1a": g1a(A1, True, geo), "G1b": g1b(A1, True)}}
    a, b = rec["r1"]["G1a"], rec["r1"]["G1b"]
    rec["r1"]["pass"] = bool(a["frac_origin_within_0p2mm_of_surface"] >= 0.99 and a["frac_first_hit_plus_n_inside_face"] < 0.01
                             and b["port_vs_pixel_visibility_agreement"] >= 0.97 and 0.95 <= b["pooled_port_over_pixel_irradiance"] <= 1.05)
    (out / "gate1.json").write_text(json.dumps(rec, indent=1)); print(json.dumps(rec, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
