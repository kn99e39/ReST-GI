"""WL42 Gate 1 diagnostics (post-measurement, descriptive): separate port-construction defects from resolution limits.

D1 interior-region disagreement: only ports whose pixels are ALL lit or ALL shadowed (away from shadow edges), the share
   of ports whose traced value disagrees (lit pixels but port < 0.5 / shadowed pixels but port >= 0.5).
D2 inside-face first hits along +n: the share whose hit triangle belongs to ANOTHER connected component (the port's
   surface lies inside another closed solid: enclosed geometry, dark in the renderer as well) vs the same component.
"""
import json, sys
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
import r1_common as C
import fnto_relations as R
import r1_gate1 as G
import r1_map as M


def interior(A, r1):
    cfg = {c["i"]: c for c in json.loads((G.HO / "configs.json").read_text())}
    tree, g = cKDTree(A["pos"][A["obj"] == 0]), np.flatnonzero(A["obj"] == 0)
    bV = np.load(C.BOX_T0)["V"]; blo, bhi = bV.min(0), bV.max(0)
    lit_bad = lit_n = dark_bad = dark_n = 0
    for i in sorted(cfg)[::3]:
        c = cfg[i]; z = np.load(G.HO / f"img_{i:05d}.npz")
        p = z["pos"].astype(np.float64); nn = z["nrm"].astype(np.float32); cam = z["cam"].astype(np.float32)
        nn = np.where(((nn * cam).sum(1) < 0)[:, None], -nn, nn); vis = z["vis"].astype(bool)
        bo = None if c["box_offset"] is None else np.asarray(c["box_offset"])
        m = np.ones(len(p), bool)
        if bo is not None:
            m &= ~np.all((p >= blo + bo - 1e-3) & (p <= bhi + bo + 1e-3), 1)
        w = np.asarray(c["sun"], np.float64); w /= np.linalg.norm(w)
        keep = m & (np.clip(nn @ w, 0, None) > 0.1)
        if keep.sum() < 100:
            continue
        geo = R.Geo(np.zeros(3), bo, masks=c["masks"])
        if r1:
            port, okp = M.pixels_to_ports(A, geo, p[keep], cam[keep].astype(np.float64)); sel_ = okp & (port >= 0)
            kidx = np.flatnonzero(keep)[sel_]; keep = np.zeros_like(keep); keep[kidx] = True; port = port[sel_]
        else:
            _, ii = tree.query(p[keep], k=8, workers=-1); cand = g[ii]
            ok = (A["nrm"][cand] * nn[keep][:, None, :]).sum(-1) > 0.5
            port = cand[np.arange(len(cand)), np.where(ok.any(1), np.argmax(ok, 1), 0)]
        up, k_ = np.unique(port, return_inverse=True)
        mp = np.bincount(k_, vis[keep], len(up)) / np.bincount(k_, minlength=len(up))
        if r1:
            sp = A["spos"][up].astype(np.float64); sn = A["snrm"][up].astype(np.float64); valid = np.isfinite(sp[..., 0])
            o_ = (sp + C.EPS1 * sn)[valid]; vv = ~geo.occluded(o_, np.broadcast_to(w, (len(o_), 3)).copy()) & ((sn[valid] @ w) > 0)
            fr = np.zeros(valid.shape); fr[valid] = vv; Vp = fr.sum(1) / valid.sum(1)
        else:
            Vp = (~geo.occluded(A["pos"][up].astype(np.float64) + C.EPS * A["nrm"][up], np.broadcast_to(w, (len(up), 3)).copy())).astype(float)
        L = mp == 1.0; D = mp == 0.0
        lit_bad += int((Vp[L] < 0.5).sum()); lit_n += int(L.sum()); dark_bad += int((Vp[D] >= 0.5).sum()); dark_n += int(D.sum())
    return {"ports_all_pixels_lit": lit_n, "of_which_port_dark": lit_bad / max(lit_n, 1), "ports_all_pixels_dark": dark_n, "of_which_port_lit": dark_bad / max(dark_n, 1)}


def enclosed(A, geo):
    nA = int(A["n_A"]); sel = np.arange(0, nA, 7)
    o = A["pos"][sel].astype(np.float64) + C.EPS1 * A["nrm"][sel]; n = A["nrm"][sel].astype(np.float64)
    vp, _, prim, sid, tp = geo.intersect(o, n)
    V, F = R.mesh_a(); comp = np.load(C.MESH_A)["comp"]
    cr = np.cross(V[F[prim, 1]] - V[F[prim, 0]], V[F[prim, 2]] - V[F[prim, 0]]) * A["tri_sign"][prim][:, None]
    inside = vp & ~A["tri_two"][prim] & ((cr * n).sum(1) > 0)
    other = inside & (comp[prim] != A["comp"][sel])
    return {"inside_first_hit": float(inside.mean()), "inside_in_other_component": float(other.mean()), "inside_same_component": float((inside & ~other).mean())}


if __name__ == "__main__":
    A0 = R.load_anchors(); A1 = dict(np.load(C.OUT1 / "anchors.npz")); geo = R.Geo(*C.offsets("T0"))
    rec = {"batch6_interior": interior(A0, False), "r1_interior_triangle_mapped": interior(A1, True), "r1_enclosed": enclosed(A1, geo)}
    Path(sys.argv[1]).mkdir(parents=True, exist_ok=True); (Path(sys.argv[1]) / "gate1_diag.json").write_text(json.dumps(rec, indent=1)); print(json.dumps(rec, indent=1))
