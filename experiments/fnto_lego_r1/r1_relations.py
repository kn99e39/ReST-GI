"""WL42 r1 free-space coupling geometry (PC, Mitsuba cuda_ad_rgb).

    python r1_relations.py T0 T1          # full builds + incremental T0 -> T1 update and its comparison

Per port: K_GATHER = 128 cosine-distributed directions split over its S_PORT surface samples (128 / S per sample, each in
its own triangle frame with a per-port rotation; a port with fewer samples reuses them cyclically). Rays start EPS1 =
0.1 mm off a real surface point (Batch 6: 1 mm off a cluster centroid that could lie inside geometry, audit #1).
Hits map to ports by surface identity (r1_map: cluster key of the hit triangle's facing side; box: analytic grid);
the inside face of a closed solid = the origin is enclosed -> -2 (no radiance); miss -> -1.
Incremental update after a rigid move: re-trace the rays that start on the moved domain or whose old segment touches the
moved domain's old/new bounds; reuse the rest. Compared with a full rebuild: entries that differ are classified by hit
distance AND by whether the two ports share material and component (Batch 6 called equal-distance entries 'physically
equivalent' although they mapped to different materials, audit #10).
"""

from __future__ import annotations

import json
import sys
import time

import numpy as np

import r1_common as C
import fnto_relations as R
import r1_map as M


def load():
    z = np.load(C.OUT1 / "anchors.npz")
    A = {k: z[k] for k in z.files}
    A["faces"] = json.loads((C.OUT1 / "anchors.json").read_text())["B"]["faces"]
    return A


def ray_setup(A, idx):
    """Origins [n, K, 3] and directions [n, K, 3] of the gathers of ports idx (canonical == world for translations)."""
    K = C.K_GATHER; S = C.S_PORT; per = K // S
    sp = A["spos"][idx].astype(np.float64); sn = A["snrm"][idx].astype(np.float64)
    valid = np.isfinite(sp[..., 0]); nval = valid.sum(1)
    slot = np.arange(K) // per                                                    # sample slot of gather k
    use = slot[None] % np.maximum(nval, 1)[:, None]                               # cycle over the valid samples
    order = np.argsort(~valid, axis=1, kind="stable")                             # valid samples first
    sidx = np.take_along_axis(order, use, 1)
    P = np.take_along_axis(sp, sidx[..., None], 1); N = np.take_along_axis(sn, sidx[..., None], 1)
    rot = np.repeat(A["rot"][idx].astype(np.float64)[:, None], K, 1).reshape(-1)
    t, b, n = C.frames(N.reshape(-1, 3), rot)
    L = C.cosine_dirs_local()[np.tile(np.arange(K), len(idx))]
    d = (L[:, 0:1] * t + L[:, 1:2] * b + L[:, 2:3] * n).reshape(len(idx), K, 3)
    o = P + C.EPS1 * N
    return o, d


def map_hits(A, geo, valid, p, prim, sid, d):
    out = np.full(len(valid), -1, np.int64)
    nA = int(A["n_A"])
    if "A" in geo.names:
        m = valid & (sid == geo.names.index("A"))
        if m.any():
            port, inside = M.hits_to_ports(A, prim[m], p[m], d[m])
            port[inside] = -2
            out[m] = port
    if "B" in geo.names:
        m = valid & (sid == geo.names.index("B"))
        if m.any():
            q = p[m] - geo.off[1]
            bx = np.load(C.BOX_T0)["V"]; lo, hi = bx.min(0), bx.max(0); c = 0.5 * (lo + hi); h = 0.5 * (hi - lo)
            r = (q - c) / h; ax = np.argmax(np.abs(r), 1); side = np.where(r[np.arange(len(r)), ax] > 0, 1, -1)
            port = np.full(len(q), -1, np.int64)
            for f in A["faces"]:
                mm = (ax == f["axis"]) & (side == f["side"])
                if mm.any():
                    u_, v_ = [i for i in range(3) if i != f["axis"]]
                    iu = np.clip(((q[mm, u_] - f["lo_u"]) / f["du"]).astype(np.int64), 0, f["nu"] - 1)
                    iv = np.clip(((q[mm, v_] - f["lo_v"]) / f["dv"]).astype(np.int64), 0, f["nv"] - 1)
                    port[mm] = nA + f["start"] + iu * f["nv"] + iv
            out[m] = port
    return out


def offs(A, geo):
    o = np.zeros((len(A["obj"]), 3)); o[A["obj"] == 0] = geo.off[0]
    if geo.off[1] is not None:
        o[A["obj"] == 1] = geo.off[1]
    return o


def gather(A, geo, idx=None, chunk=40000):
    idx = np.arange(len(A["obj"])) if idx is None else idx
    off = offs(A, geo)
    H = np.full((len(idx), C.K_GATHER), -1, np.int32); T = np.full((len(idx), C.K_GATHER), np.inf, np.float32)
    for s in range(0, len(idx), chunk):
        ii = idx[s:s + chunk]
        o, d = ray_setup(A, ii); o = o + off[ii][:, None]
        o = o.reshape(-1, 3); d = d.reshape(-1, 3)
        valid, p, prim, sid, t = geo.intersect(o, d)
        H[s:s + chunk] = map_hits(A, geo, valid, p, prim, sid, d).reshape(len(ii), -1)
        T[s:s + chunk] = np.where(valid, t, np.inf).reshape(len(ii), -1)
    if geo.off[1] is None:
        H[A["obj"][idx] == 1] = -1
    return H, T


def sun_vis(A, geo, w, idx=None, chunk=400000):
    """Fraction of each port's surface samples that see the sun direction w (current geometry)."""
    idx = np.arange(len(A["obj"])) if idx is None else idx
    off = offs(A, geo)
    sp = A["spos"][idx].astype(np.float64) + off[idx][:, None]; sn = A["snrm"][idx].astype(np.float64)
    valid = np.isfinite(sp[..., 0]) & ((sn @ np.asarray(w)) > 0)
    o = (sp + C.EPS1 * sn)[valid]; vis = np.zeros(o.shape[0], bool)
    for s in range(0, len(o), chunk):
        oo = o[s:s + chunk]; vis[s:s + chunk] = ~geo.occluded(oo, np.broadcast_to(np.asarray(w, np.float64), oo.shape).copy())
    frac = np.zeros(valid.shape); frac[valid] = vis
    nsamp = np.isfinite(A["spos"][idx][..., 0]).sum(1)
    return frac.sum(1) / np.maximum(nsamp, 1), np.where(valid, np.clip(sn @ np.asarray(w), 0, None), 0).sum(1) / np.maximum(nsamp, 1)


def incremental(A, geo_new, H_old, T_old, moved_obj, old_off, new_off, margin=1e-3):
    t0 = time.time()
    bV = np.load(C.BOX_T0)["V"] if moved_obj == 1 else np.load(C.MESH_A)["V"]
    lo, hi = bV.min(0), bV.max(0)
    boxes = [(lo + old_off - margin, hi + old_off + margin), (lo + new_off - margin, hi + new_off + margin)]
    n = len(A["obj"]); off_old = np.zeros((n, 3)); off_old[A["obj"] == moved_obj] = old_off
    redo = np.zeros((n, C.K_GATHER), bool); redo[A["obj"] == moved_obj] = True
    rest = np.flatnonzero(A["obj"] != moved_obj)
    for s in range(0, len(rest), 40000):
        ii = rest[s:s + 40000]
        o, d = ray_setup(A, ii); o = (o + off_old[ii][:, None]).reshape(-1, 3); d = d.reshape(-1, 3)
        t = T_old[ii].reshape(-1).astype(np.float64); hit = np.zeros(len(o), bool)
        for lo_, hi_ in boxes:
            hit |= R.slab_hit(o, d, t, lo_, hi_)
        redo[ii] = hit.reshape(len(ii), -1)
    t_sel = time.time() - t0
    rows = np.flatnonzero(redo.any(1)); t1 = time.time()
    Hn, Tn = gather(A, geo_new, rows)
    H = H_old.copy(); T = T_old.copy()
    H[rows] = np.where(redo[rows], Hn, H_old[rows]); T[rows] = np.where(redo[rows], Tn, T_old[rows])
    return H, T, {"rays_selected": int(redo.sum()), "rows_retraced": int(len(rows)), "rays_traced_incl_rows": int(len(rows) * C.K_GATHER),
                  "rays_total": int(redo.size), "select_s": t_sel, "trace_map_s": time.time() - t1}


def main(states):
    A = load(); rec = {}
    for st in states:
        geo = R.Geo(*C.offsets(st))
        t = time.time(); H, T = gather(A, geo); sec = time.time() - t
        np.savez_compressed(C.OUT1 / f"rel_{st}.npz", hit=H, t=T)
        rec[st] = {"seconds_full_build": sec, "rays": int(H.size), "miss": float((H == -1).mean()), "enclosed": float((H == -2).mean())}
        print(st, rec[st], flush=True)
    if states == ["T0", "T1"]:
        z0 = np.load(C.OUT1 / "rel_T0.npz"); z1 = np.load(C.OUT1 / "rel_T1.npz")
        H, T, inc = incremental(A, R.Geo(*C.offsets("T1")), z0["hit"], z0["t"], 1, C.offsets("T0")[1], C.offsets("T1")[1])
        mm = H != z1["hit"]; a, b = H[mm], z1["hit"][mm]; both = (a >= 0) & (b >= 0)
        inc.update({"entries_changed_T0_T1": int((z0["hit"] != z1["hit"]).sum()), "mismatch_vs_full": int(mm.sum()),
                    "mismatch_same_distance": int((np.abs(T[mm] - z1["t"][mm]) < 1e-5).sum()),
                    "mismatch_same_material_and_component": int(((A["mat"][a[both]] == A["mat"][b[both]]) & (A["comp"][a[both]] == A["comp"][b[both]])).sum()),
                    "mismatch_one_side_blocked_or_miss": int((~both).sum())})
        np.savez_compressed(C.OUT1 / "rel_T1_inc.npz", hit=H, t=T); rec["incremental_T0_to_T1"] = inc; print(inc)
    (C.OUT1 / "relations.json").write_text(json.dumps(rec, indent=1))


if __name__ == "__main__":
    main(sys.argv[1:])
