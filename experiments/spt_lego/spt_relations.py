"""SPT-1 surface anchors and current-geometry relation tables (Windows 8DNA venv: Mitsuba cuda_ad_rgb).

    python spt_relations.py anchors                 # canonical anchors from the observed T0 training surface (once)
    python spt_relations.py relations T0 T1         # relation table per geometry state

Anchors (configuration-invariant, canonical frame = T0 placement): observed train_T0 surface points (alpha > 0,
camera-facing normals, object from the B4 owner cache: label 16 = box B), voxel-downsampled at H_VOX with the
normal's dominant signed axis as part of the key (separates the two sides of thin parts). The anchor normal is the
mean normal of its cell.

Relations of a state (built only from that state's geometry; no radiance and no reference image is read):
  * hit[a, k]: sender anchor hit by gather ray k of anchor a (-1 = miss). Rays start at x_a + EPS n_a along K fixed
    cosine-distributed local directions (per-anchor rotation, seed 0). A hit point is mapped to the nearest anchor of
    the hit object that faces the ray origin (n_s . -d > 0), else to the nearest anchor.
  * vis[a, b]: sun visibility of anchor a toward light-bin centre b (bit-packed).
Configuration-independent per-anchor gather weights sh[a, k, :] = SH9(d_ak) * pi / K are stored with the anchors.
"""

from __future__ import annotations

import json
import sys
import time

import numpy as np

import spt_common as C

H_VOX = 0.012
MAX_IMAGES = 200            # every other train_T0 image for anchor construction


def load_mesh_scene(state):
    import mitsuba as mi

    mi.set_variant("cuda_ad_rgb")
    import drjit as dr

    a_off, b_off = C.offsets(state)
    m = np.load(C.ROOT / "results/rna_surface_ownership_lego/mesh/lego_mesh.npz")
    b = np.load(C.B4 / "scene/B_box_T0.npz")

    def mesh(name, V, F):
        me = mi.Mesh(name, V.shape[0], F.shape[0], has_vertex_normals=False)
        p = mi.traverse(me)
        p["vertex_positions"] = mi.Float(np.ascontiguousarray(V.astype(np.float32).ravel()))
        p["faces"] = mi.UInt32(np.ascontiguousarray(F.astype(np.uint32).ravel()))
        p.update()
        return me

    ma, mb = mesh("A", m["V"] + a_off, m["F"]), mesh("B", b["V"] + b_off, b["F"])
    scene = mi.load_dict({"type": "scene", "A": ma, "B": mb})
    shapes = scene.shapes()
    ids = {s.id(): i for i, s in enumerate(shapes)}
    return mi, dr, scene, ids


def vec(mi, a):
    return mi.Vector3f(*[mi.Float(np.ascontiguousarray(a[:, i].astype(np.float32))) for i in range(3)])


def cmd_anchors():
    import h5py

    f = h5py.File(C.B4 / "data/train_T0.h5", "r")
    own = C.B4 / "data/train_T0.h5.owners_b4_9e9dce083ac9"
    keys, P, N, O = [], [], [], []
    t0 = time.time()
    for i in range(0, f["alpha"].shape[0], f["alpha"].shape[0] // MAX_IMAGES):
        a = np.nan_to_num(f["alpha"][i, :, 0].astype(np.float32)); m = a > 0
        p = f["position"][i][m].astype(np.float64); n = f["normal"][i][m].astype(np.float64)
        c = f["camera_dir"][i][m].astype(np.float64)
        n = np.where(((n * c).sum(1) < 0)[:, None], -n, n)
        lab = np.load(own / f"{i:04d}.npz")["lab"]
        assert len(lab) == len(p), (i, len(lab), len(p))
        o = (lab == C.B_UNIT).astype(np.int64)
        ijk = np.floor(p / H_VOX).astype(np.int64) + 4096
        ax = np.argmax(np.abs(n), 1); sg = (n[np.arange(len(n)), ax] > 0).astype(np.int64)
        keys.append((((ijk[:, 0] * 8192 + ijk[:, 1]) * 8192 + ijk[:, 2]) * 6 + ax * 2 + sg) * 2 + o)
        P.append(p.astype(np.float32)); N.append(n.astype(np.float32)); O.append(o.astype(np.int8))
    keys = np.concatenate(keys); P = np.concatenate(P); N = np.concatenate(N); O = np.concatenate(O)
    u, inv, cnt = np.unique(keys, return_inverse=True, return_counts=True)
    pos = np.stack([np.bincount(inv, P[:, j], len(u)) for j in range(3)], 1) / cnt[:, None]
    nrm = np.stack([np.bincount(inv, N[:, j], len(u)) for j in range(3)], 1)
    nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)
    obj = (u % 2).astype(np.int8)
    keep = obj == 0                                       # A: observed cells; B: analytic grid on all six faces (its
    pos, nrm, obj, cnt = pos[keep], nrm[keep], obj[keep], cnt[keep]   # underside is never observed but is a sender)
    bV = np.load(C.B4 / "scene/B_box_T0.npz")["V"]; lo, hi = bV.min(0), bV.max(0)
    bp, bn = [], []
    for ax in range(3):
        u_, v_ = [i for i in range(3) if i != ax]
        gu = np.arange(lo[u_] + H_VOX / 2, hi[u_], H_VOX); gv = np.arange(lo[v_] + H_VOX / 2, hi[v_], H_VOX)
        U, V_ = np.meshgrid(gu, gv, indexing="ij")
        for side, val in ((-1, lo[ax]), (1, hi[ax])):
            q = np.zeros((U.size, 3)); q[:, u_] = U.ravel(); q[:, v_] = V_.ravel(); q[:, ax] = val
            nn = np.zeros((U.size, 3)); nn[:, ax] = side
            bp.append(q); bn.append(nn)
    bp, bn = np.concatenate(bp), np.concatenate(bn)
    pos = np.concatenate([pos, bp]); nrm = np.concatenate([nrm, bn])
    obj = np.concatenate([obj, np.ones(len(bp), np.int8)]); cnt = np.concatenate([cnt, np.zeros(len(bp), cnt.dtype)])
    rot = np.random.default_rng(0).uniform(0, 2 * np.pi, len(pos))
    t, b, n = C.frames(nrm.astype(np.float64), rot)
    L = C.cosine_dirs_local()
    dirs = L[None, :, 0:1] * t[:, None] + L[None, :, 1:2] * b[:, None] + L[None, :, 2:3] * n[:, None]   # [Na, K, 3]
    sh = (C.sh9(dirs) * np.pi / C.K_GATHER).astype(np.float16)
    C.OUT.mkdir(parents=True, exist_ok=True)
    np.savez(C.OUT / "anchors.npz", pos=pos.astype(np.float32), nrm=nrm.astype(np.float32), obj=obj, count=cnt.astype(np.int32),
             dirs=dirs.astype(np.float32), h_vox=H_VOX, images_used=MAX_IMAGES)
    np.save(C.OUT / "anchor_sh.npy", sh)
    rec = {"anchors": int(len(pos)), "A": int((obj == 0).sum()), "B": int((obj == 1).sum()), "points": int(len(P)),
           "h_vox": H_VOX, "seconds": time.time() - t0}
    (C.OUT / "anchors.json").write_text(json.dumps(rec, indent=1)); print(rec)


def cmd_relations(states):
    from scipy.spatial import cKDTree

    A = np.load(C.OUT / "anchors.npz"); pos, nrm, obj, dirs = A["pos"].astype(np.float64), A["nrm"].astype(np.float64), A["obj"], A["dirs"]
    Na, K = dirs.shape[:2]
    trees = {o: (cKDTree(pos[obj == o]), np.flatnonzero(obj == o)) for o in (0, 1)}
    bc = C.bin_centres()
    for state in states:
        mi, dr, scene, ids = load_mesh_scene(state)
        offs = C.offsets(state)
        world = pos + np.stack(offs)[obj]
        rec = {"state": state, "anchors": int(Na), "K": int(K), "bins": C.N_BINS}
        # gather rays
        o = np.repeat(world + C.EPS * nrm, K, 0); d = dirs.reshape(-1, 3)
        dr.sync_thread(); t = time.time()
        si = scene.ray_intersect(mi.Ray3f(mi.Point3f(*[mi.Float(np.ascontiguousarray(o[:, i].astype(np.float32))) for i in range(3)]), vec(mi, d)))
        valid = np.array(si.is_valid()); hp = np.stack([np.array(si.p.x), np.array(si.p.y), np.array(si.p.z)], 1).astype(np.float64)
        bV = np.load(C.B4 / "scene/B_box_T0.npz")["V"] + offs[1]          # B is an axis-aligned box: hits on it lie in its AABB
        inB = np.all((hp >= bV.min(0) - 1e-3) & (hp <= bV.max(0) + 1e-3), 1)
        hit_obj = np.where(valid, np.where(inB, 1, 0), -1)
        dr.sync_thread(); rec["gather_rays"] = int(len(d)); rec["gather_s"] = time.time() - t
        t = time.time()
        hit = np.full(len(d), -1, np.int64)
        for ob in (0, 1):
            m = hit_obj == ob
            if not m.any():
                continue
            canon = hp[m] - offs[ob]
            tree, gidx = trees[ob]
            dd, ii = tree.query(canon, k=8, workers=-1)
            cand = gidx[ii]                                                   # [M, 8]
            facing = (nrm[cand] * -d[m][:, None, :]).sum(-1) > 0
            first = np.where(facing.any(1), np.argmax(facing, 1), 0)
            hit[m] = cand[np.arange(len(cand)), first]
            rec[f"hit_{'AB'[ob]}"] = int(m.sum()); rec[f"hit_{'AB'[ob]}_anchor_dist_q99_m"] = float(np.quantile(dd[np.arange(len(dd)), first], 0.99))
        rec["map_s"] = time.time() - t
        hit = hit.reshape(Na, K).astype(np.int32)
        # sun visibility per anchor and light bin
        dr.sync_thread(); t = time.time()
        vis = np.zeros((Na, C.N_BINS), bool)
        step = 16384
        for s in range(0, Na, step):
            w = world[s:s + step] + C.EPS * nrm[s:s + step]
            oo = np.repeat(w, C.N_BINS, 0); dd_ = np.tile(bc, (len(w), 1))
            r = mi.Ray3f(mi.Point3f(*[mi.Float(np.ascontiguousarray(oo[:, i].astype(np.float32))) for i in range(3)]), vec(mi, dd_))
            vis[s:s + step] = ~np.array(scene.ray_test(r)).reshape(len(w), C.N_BINS)
        dr.sync_thread(); rec["vis_rays"] = int(Na * C.N_BINS); rec["vis_s"] = time.time() - t
        rec["miss_fraction"] = float((hit < 0).mean())
        np.savez_compressed(C.OUT / f"rel_{state}.npz", hit=hit, vis=np.packbits(vis, axis=1))
        (C.OUT / f"rel_{state}.json").write_text(json.dumps(rec, indent=1)); print(rec)
    if len(states) == 2:
        r0, r1 = (np.load(C.OUT / f"rel_{s}.npz") for s in states)
        hc = (r0["hit"] != r1["hit"]).any(1); vc = (r0["vis"] != r1["vis"]).any(1)
        diff = {"anchors": int(Na), "hit_changed_anchors": int(hc.sum()), "vis_changed_anchors": int(vc.sum()),
                "any_changed_anchors": int((hc | vc).sum()), "any_changed_fraction": float((hc | vc).mean()),
                "hit_entries_changed": int((r0["hit"] != r1["hit"]).sum()), "hit_entries": int(r0["hit"].size)}
        (C.OUT / f"rel_diff_{states[0]}_{states[1]}.json").write_text(json.dumps(diff, indent=1)); print(diff)


if __name__ == "__main__":
    if sys.argv[1] == "anchors":
        cmd_anchors()
    else:
        cmd_relations(sys.argv[2:])
