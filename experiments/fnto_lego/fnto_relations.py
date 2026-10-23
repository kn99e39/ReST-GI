"""FNTO editable free-space coupling geometry: relation tables and incident-sun visibility (Mitsuba cuda_ad_rgb).

    python fnto_relations.py relations T0 T1        # full rebuild per state + incremental T0 -> T1 update and equality check

Coupling C is the free-flight operator between optical-domain ports. Its geometric part is built here from the CURRENT
geometry only (no radiance, no reference image): every port casts K_GATHER fixed cosine-distributed directions in its own
rotated tangent frame; a hit is mapped to the port of the hit triangle's side that faces the ray origin (domain A: the
triangle -> port table of fnto_anchors; domain B: the analytic face grid), a miss leaves the scene (no world light).
Same-domain hits (e.g. Lego bucket -> Lego plate) are ordinary coupling entries: exposed same-asset free-space transport
is never stored in a kernel (contract §2).

Compression / reuse: ports (H_VOX clusters, 430 k) instead of pixels; K_GATHER fixed directions; and an incremental update
after a rigid edit that re-traces only rays that (a) start on a moved domain or (b) whose old segment touches the old or
new bounds of the moved domain (slab test, 1 mm margin) -- all other entries are reused. The update is checked against
a full rebuild of the new state; the only admissible differences are ties (identical hit distance) between coincident or
edge-sharing triangles that two BVHs resolve differently.
"""

from __future__ import annotations

import json
import sys
import time

import numpy as np

import fnto_common as C

_MI = None


def mi_init():
    global _MI
    if _MI is None:
        import drjit as dr
        import mitsuba as mi

        mi.set_variant("cuda_ad_rgb")
        _MI = (mi, dr)
    return _MI


_MESH = None
_TREE = {}


def cluster_key(p, n, mat):
    from fnto_anchors import cluster_key as ck

    return ck(p, n, mat)


def nearest_port(A, q, n, mat):
    """Nearest domain-A port of the same material whose normal agrees with n (fallback: nearest of that material)."""
    from scipy.spatial import cKDTree

    out = np.full(len(q), -1, np.int64)
    for mt in np.unique(mat):
        if mt not in _TREE:
            g = np.flatnonzero((A["obj"] == 0) & (A["mat"] == mt)); _TREE[mt] = (cKDTree(A["pos"][g]), g)
        tree, g = _TREE[mt]
        mm = mat == mt
        _, ii = tree.query(q[mm], k=8, workers=-1)
        cand = g[ii]
        ok = (A["nrm"][cand] * n[mm][:, None, :]).sum(-1) > 0.3
        first = np.where(ok.any(1), np.argmax(ok, 1), 0)
        out[mm] = cand[np.arange(len(cand)), first]
    return out


def mesh_a():
    global _MESH
    if _MESH is None:
        z = np.load(C.MESH_A); _MESH = (z["V"].astype(np.float64), z["F"])
    return _MESH


def load_anchors():
    z = np.load(C.OUT / "anchors.npz")
    return {k: z[k] for k in z.files}


def gather_dirs(A, idx):
    """World-frame gather directions [n, K, 3] of ports idx (translation-only edits: canonical frame == world frame)."""
    t, b, n = C.frames(A["nrm"][idx].astype(np.float64), A["rot"][idx].astype(np.float64))
    L = C.cosine_dirs_local()
    return L[None, :, 0:1] * t[:, None] + L[None, :, 1:2] * b[:, None] + L[None, :, 2:3] * n[:, None]


class Geo:
    """Mitsuba scene of a geometry state: Lego (offset a), box (offset b, or absent), optional shadow-only spheres."""

    def __init__(self, off_a, off_b, masks=(), with_box=True, with_lego=True, meshes=None):
        mi, dr = mi_init()
        self.off = (np.asarray(off_a, np.float64), None if off_b is None else np.asarray(off_b, np.float64))
        shapes = {"type": "scene"}
        m = np.load(C.MESH_A); bV = np.load(C.BOX_T0)

        def mesh(name, V, F):
            me = mi.Mesh(name, V.shape[0], F.shape[0], has_vertex_normals=False)
            p = mi.traverse(me)
            p["vertex_positions"] = mi.Float(np.ascontiguousarray(V.astype(np.float32).ravel()))
            p["faces"] = mi.UInt32(np.ascontiguousarray(F.astype(np.uint32).ravel()))
            p.update()
            return me

        self.names = []
        if meshes is not None:                                  # synthetic scenes (contract tests): [(name, V, F), ...]
            with_lego = with_box = False
            for name, V, F in meshes:
                shapes[name] = mesh(name, np.asarray(V, np.float64), np.asarray(F)); self.names.append(name)
        if with_lego:
            shapes["A"] = mesh("A", m["V"] + self.off[0], m["F"]); self.names.append("A")
        if with_box and off_b is not None:
            shapes["B"] = mesh("B", bV["V"] + self.off[1], bV["F"]); self.names.append("B")
        for k, s in enumerate(masks):
            shapes[f"M{k}"] = {"type": "sphere", "center": list(map(float, s["c"])), "radius": float(s["r"])}
            self.names.append(f"M{k}")
        self.scene = mi.load_dict(shapes)
        self.mi, self.dr = mi, dr
        self.shape_ids = [s.id() for s in self.scene.shapes()]

    def _ray(self, o, d):
        mi = self.mi
        f = lambda a: mi.Float(np.ascontiguousarray(a.astype(np.float32)))  # noqa: E731
        return mi.Ray3f(mi.Point3f(f(o[:, 0]), f(o[:, 1]), f(o[:, 2])), mi.Vector3f(f(d[:, 0]), f(d[:, 1]), f(d[:, 2])))

    def intersect(self, o, d):
        si = self.scene.ray_intersect(self._ray(o, d))
        valid = np.array(si.is_valid())
        p = np.stack([np.array(si.p.x), np.array(si.p.y), np.array(si.p.z)], 1).astype(np.float64)
        prim = np.array(si.prim_index).astype(np.int64)
        sid = np.full(len(valid), -1, np.int64)
        if valid.any():
            sh = si.shape
            for s in self.scene.shapes():                     # few shapes: compare shape pointers
                hit_k = np.array(self.dr.eq(sh, self.mi.ShapePtr(s)))
                sid[hit_k & valid] = self.names.index(s.id()) if s.id() in self.names else -2
        t = np.array(si.t).astype(np.float64)
        p = o + np.where(valid, t, 0)[:, None] * d          # float64 from (o, t, d): deterministic across scenes/BVHs
        return valid, p, prim, sid, t

    def occluded(self, o, d):
        return np.array(self.scene.ray_test(self._ray(o, d)))


def map_hits(A, geo, valid, p, prim, sid, d):
    """Hit -> global port index (-1 = miss or a non-domain shape such as a mask; -2 = the inside face of a closed solid,
    i.e. the ray origin is buried in overlapping geometry and receives no radiance along that direction)."""
    out = np.full(len(valid), -1, np.int64)
    nA = int(A["n_A"])
    if "A" in geo.names:
        m = valid & (sid == geo.names.index("A"))
        if m.any():
            t = prim[m]
            V, F = mesh_a()
            cr = np.cross(V[F[t, 1]] - V[F[t, 0]], V[F[t, 2]] - V[F[t, 0]]) * A["tri_sign"][t][:, None]
            back = (cr * -d[m]).sum(1) < 0
            flip = A["tri_two"][t] & back                                      # two-sided: the side facing the ray
            n = np.where(flip[:, None], -cr, cr)
            q = p[m] - geo.off[0]
            key = cluster_key(q, n, A["tri_mat"][t].astype(np.int64))
            ka = A["keyA"]
            j = np.clip(np.searchsorted(ka, key), 0, len(ka) - 1)
            port = np.where(ka[j] == key, j, -1).astype(np.int64)
            miss = port < 0
            if miss.any():                                                      # neighbouring cluster: nearest same-material port
                port[miss] = nearest_port(A, q[miss], n[miss], A["tri_mat"][t][miss])
            port[back & ~A["tri_two"][t]] = -2      # inside face of a closed solid: the origin is enclosed, no radiance
            out[m] = port
    if "B" in geo.names:
        m = valid & (sid == geo.names.index("B"))
        if m.any():
            q = p[m] - geo.off[1]
            bx = np.load(C.BOX_T0)["V"]; lo, hi = bx.min(0), bx.max(0); c = 0.5 * (lo + hi); h = 0.5 * (hi - lo)
            r = (q - c) / h
            ax = np.argmax(np.abs(r), 1); side = np.where(r[np.arange(len(r)), ax] > 0, 1, -1)
            rec = json.loads((C.OUT / "anchors.json").read_text())["B"]["faces"]
            port = np.full(len(q), -1, np.int64)
            for f in rec:
                mm = (ax == f["axis"]) & (side == f["side"])
                if not mm.any():
                    continue
                u_, v_ = [i for i in range(3) if i != f["axis"]]
                iu = np.clip(((q[mm, u_] - f["lo_u"]) / f["du"]).astype(np.int64), 0, f["nu"] - 1)
                iv = np.clip(((q[mm, v_] - f["lo_v"]) / f["dv"]).astype(np.int64), 0, f["nv"] - 1)
                port[mm] = nA + f["start"] + iu * f["nv"] + iv
            out[m] = port
    return out


def map_hits_nearest(A, geo, valid, p, prim, sid, d, dom_of_shape):
    """Generic hit mapping for synthetic domains: the nearest port of the hit shape's domain whose normal faces the ray
    origin (dom_of_shape: shape name -> domain id; shapes not listed are pure occluders -> -1)."""
    from scipy.spatial import cKDTree

    out = np.full(len(valid), -1, np.int64)
    for name, o in dom_of_shape.items():
        if name not in geo.names:
            continue
        m = valid & (sid == geo.names.index(name))
        if not m.any():
            continue
        g = np.flatnonzero(A["obj"] == o)
        _, ii = cKDTree(A["pos"][g]).query(p[m], k=min(8, len(g)))
        ii = np.atleast_2d(ii); cand = g[ii]
        ok = (A["nrm"][cand] * -d[m][:, None, :]).sum(-1) > 0
        first = np.where(ok.any(1), np.argmax(ok, 1), 0)
        out[m] = np.where(ok.any(1), cand[np.arange(len(cand)), first], -2)
    return out


def port_offsets(A, geo):
    off = np.zeros((len(A["obj"]), 3))
    off[A["obj"] == 0] = geo.off[0]
    if geo.off[1] is not None and len(geo.off[1]):
        off[A["obj"] == 1] = geo.off[1]
    return off


def gather(A, geo, idx=None, chunk=60000, mapper=None):
    """Relation rows of ports idx: (hit port [n, K] int32, hit distance [n, K] float32). mapper(valid, p, prim, sid, d)
    overrides the Lego/box hit mapping (synthetic contract tests)."""
    idx = np.arange(len(A["obj"])) if idx is None else idx
    present = np.ones(len(A["obj"]), bool)
    if geo.off[1] is None and mapper is None:
        present[A["obj"] == 1] = False
    off = port_offsets(A, geo)
    H = np.full((len(idx), C.K_GATHER), -1, np.int32); T = np.full((len(idx), C.K_GATHER), np.inf, np.float32)
    for s in range(0, len(idx), chunk):
        ii = idx[s:s + chunk]
        d = gather_dirs(A, ii).reshape(-1, 3)
        o = np.repeat(A["pos"][ii].astype(np.float64) + off[ii] + C.EPS * A["nrm"][ii], C.K_GATHER, 0)
        valid, p, prim, sid, t = geo.intersect(o, d)
        h = map_hits(A, geo, valid, p, prim, sid, d) if mapper is None else mapper(valid, p, prim, sid, d)
        H[s:s + chunk] = h.reshape(len(ii), C.K_GATHER); T[s:s + chunk] = np.where(valid, t, np.inf).reshape(len(ii), C.K_GATHER)
    H[~present[idx]] = -1
    return H, T


def sun_visible(A, geo, idx, w, chunk=500000):
    """Sun visibility of ports idx toward direction w [3] (or per-port [n, 3]) in the scene of geo."""
    off = port_offsets(A, geo)
    w = np.broadcast_to(np.asarray(w, np.float64), (len(idx), 3))
    out = np.zeros(len(idx), bool)
    for s in range(0, len(idx), chunk):
        ii = idx[s:s + chunk]
        o = A["pos"][ii].astype(np.float64) + off[ii] + C.EPS * A["nrm"][ii]
        out[s:s + chunk] = ~geo.occluded(o, w[s:s + chunk])
    return out


def slab_hit(o, d, t, lo, hi):
    """Does segment o + s d, s in [0, t] (t may be inf) touch the box [lo, hi]?"""
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = 1.0 / d
        t1 = (lo - o) * inv; t2 = (hi - o) * inv
    tmin = np.nanmax(np.minimum(t1, t2), 1); tmax = np.nanmin(np.maximum(t1, t2), 1)
    return (tmax >= np.maximum(tmin, 0)) & (tmin <= t)


def incremental(A, geo_new, H_old, T_old, moved_obj, old_off, new_off, margin=1e-3):
    """Re-trace only the rays a rigid move of domain `moved_obj` can change; reuse every other relation entry."""
    t0 = time.time()
    bV = np.load(C.BOX_T0)["V"] if moved_obj == 1 else np.load(C.MESH_A)["V"]
    lo, hi = bV.min(0), bV.max(0)
    boxes = [(lo + old_off - margin, hi + old_off + margin), (lo + new_off - margin, hi + new_off + margin)]
    n = len(A["obj"])
    off_old = np.zeros((n, 3)); off_old[A["obj"] == moved_obj] = old_off
    redo = np.zeros((n, C.K_GATHER), bool)
    redo[A["obj"] == moved_obj] = True
    rest = np.flatnonzero(A["obj"] != moved_obj)
    for s in range(0, len(rest), 60000):
        ii = rest[s:s + 60000]
        d = gather_dirs(A, ii).reshape(-1, 3)
        o = np.repeat(A["pos"][ii].astype(np.float64) + off_old[ii] + C.EPS * A["nrm"][ii], C.K_GATHER, 0)
        t = T_old[ii].reshape(-1).astype(np.float64)
        hit = np.zeros(len(o), bool)
        for lo_, hi_ in boxes:
            hit |= slab_hit(o, d, t, lo_, hi_)
        redo[ii] = hit.reshape(len(ii), C.K_GATHER)
    t_select = time.time() - t0
    rows = np.flatnonzero(redo.any(1))
    t1 = time.time()
    Hn, Tn = gather(A, geo_new, rows)
    H = H_old.copy(); T = T_old.copy()
    H[rows] = np.where(redo[rows], Hn, H_old[rows]); T[rows] = np.where(redo[rows], Tn, T_old[rows])
    return H, T, {"rays_selected": int(redo.sum()), "rays_total": int(redo.size), "rows_retraced": int(len(rows)),
                  "rays_retraced_incl_row_padding": int(len(rows) * C.K_GATHER), "select_s": t_select, "trace_map_s": time.time() - t1}


def cmd_relations(states):
    A = load_anchors()
    rec = {}
    for st in states:
        oa, ob = C.offsets(st)
        geo = Geo(oa, ob)
        t = time.time(); H, T = gather(A, geo); sec = time.time() - t
        np.savez_compressed(C.OUT / f"rel_{st}.npz", hit=H, t=T)
        rec[st] = {"seconds_full_build": sec, "rays": int(H.size), "miss_fraction": float((H < 0).mean()),
                   "same_domain_A_fraction": float(((H >= 0) & (H < int(A["n_A"])))[A["obj"] == 0].mean())}
        print(st, rec[st], flush=True)
    if states == ["T0", "T1"]:
        z0 = np.load(C.OUT / "rel_T0.npz"); z1 = np.load(C.OUT / "rel_T1.npz")
        geo1 = Geo(*C.offsets("T1"))
        H, T, inc = incremental(A, geo1, z0["hit"], z0["t"], 1, C.offsets("T0")[1], C.offsets("T1")[1])
        inc["equal_to_full_rebuild"] = bool((H == z1["hit"]).all())
        inc["entries_changed_T0_T1"] = int((z0["hit"] != z1["hit"]).sum())
        inc["rows_changed_T0_T1"] = int((z0["hit"] != z1["hit"]).any(1).sum())
        mm = H != z1["hit"]
        inc["mismatch_entries"] = int(mm.sum())
        # a mismatch with an identical hit distance is a tie between coincident / edge-sharing triangles that the two BVHs
        # resolve differently (same surface point, adjacent cluster): physically equivalent, not a missed change
        inc["mismatch_same_distance"] = int((np.abs(T[mm] - z1["t"][mm]) < 1e-5).sum())
        np.savez_compressed(C.OUT / "rel_T1_inc.npz", hit=H, t=T)
        rec["incremental_T0_to_T1"] = inc; print(inc)
    (C.OUT / "relations.json").write_text(json.dumps(rec, indent=1))


if __name__ == "__main__":
    if sys.argv[1] == "relations":
        cmd_relations(sys.argv[2:])
