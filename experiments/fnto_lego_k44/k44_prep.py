"""WL44 candidate preprocessing (protocol fnto_k44): r1_prep with the renderer-consistent light-visibility rule.

Identical to r1_prep.main (same pixels, identity tracing, light table, port surface samples) except:
  * visibility of a light from a pixel / port surface sample ignores occluder hits closer than T_MIN = 0.3 mm to the
    offset origin (r1: any hit; A6 validation vs Blender's BVH: disagreement 2.7 % -> 0.47 % of pixel-light pairs);
  * the per-image box offset is stored (boxoff.npy, NaN when the box is absent) so that training can evaluate point /
    spot light geometry for the box domain in world coordinates (protocol fnto_k44, correction part b).
Writes results/fnto_lego_r1/stagea/prep2_<split>/ (r1's prep_<split>/ is untouched).

    python k44_prep.py <split> [...]        (PC or LabServer63 with Mitsuba cuda; mi_env on the server)
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

import k44_common as K  # noqa: F401  (path setup)
import r1_common as C
import fnto_relations as R
import r1_map as M
import r1_relations as RR
import r1_prep as RP

T_MIN = 3e-4
D = C.OUT1 / "stagea"


def visible_t(geo, o, n, L, chunk=2_000_000, tmin=T_MIN):
    """r1_prep.visible with hits closer than tmin to the origin ignored (origin advanced by tmin along the ray)."""
    out = np.zeros(len(o), bool)
    if L[0] == 0:
        w = L[1:4].astype(np.float64); face = (n @ w) > 0; dvec = np.broadcast_to(w, o.shape); tmax = np.full(len(o), 1e4)
    else:
        v = L[1:4].astype(np.float64)[None] - o; dist = np.linalg.norm(v, axis=1); dvec = v / dist[:, None]
        face = (n * dvec).sum(1) > 0
        if L[0] == 2:
            face &= ((-dvec) @ L[4:7].astype(np.float64)) >= L[7]
        tmax = dist - 1e-4
    idx = np.flatnonzero(face & (tmax > tmin + 1e-6))
    mi = geo.mi; f = lambda a: mi.Float(np.ascontiguousarray(a.astype(np.float32)))  # noqa: E731
    for s in range(0, len(idx), chunk):
        ii = idx[s:s + chunk]; dd = np.ascontiguousarray(dvec[ii]); oo = o[ii] + tmin * dd
        ray = mi.Ray3f(mi.Point3f(f(oo[:, 0]), f(oo[:, 1]), f(oo[:, 2])), mi.Vector3f(f(dd[:, 0]), f(dd[:, 1]), f(dd[:, 2])))
        ray.maxt = f(tmax[ii] - tmin)
        out[ii] = ~np.array(geo.scene.ray_test(ray))
    return out


def main(split):
    A = RR.load(); Np = len(A["obj"])
    cfgs = sorted(json.loads((D / split / "configs.json").read_text()), key=lambda c: c["i"])
    nmax = int(os.environ.get("K44_MAX_IMAGES", "0"))
    if nmax:
        cfgs = cfgs[:nmax]
    od = D / (f"prep2_{split}" if not nmax else f"prep2smoke_{split}"); od.mkdir(parents=True, exist_ok=True)
    bV = np.load(C.BOX_T0)["V"]; blo, bhi = bV.min(0), bV.max(0)
    sp_all = A["spos"].astype(np.float64); sn_all = A["snrm"].astype(np.float64); svalid = np.isfinite(sp_all[..., 0])
    nsamp = svalid.sum(1).astype(np.uint8)
    chunks = {k: [] for k in ("pos", "nrm", "cam", "col", "dom", "port", "vbits")}
    start = [0]; LT = np.zeros((len(cfgs), RP.JMAX, 12), np.float32); nl = np.zeros(len(cfgs), np.int32); boff = np.full((len(cfgs), 3), np.nan, np.float32)
    pvis = np.lib.format.open_memmap(od / "port_vis.npy", "w+", np.uint8, (len(cfgs), RP.JMAX, Np))
    rec = {"images": len(cfgs), "dropped_untraced_px": 0, "pixels_in": 0, "t_min_m": T_MIN, "bits_changed_vs_r1": 0, "bits_0to1": 0, "bits_1to0": 0,
           "pixel_light_pairs": 0, "port_counts_changed_vs_r1": 0}
    old = D / f"prep_{split}"; old_vb = np.load(old / "px_vbits.npy", mmap_mode="r"); old_start = np.load(old / "start.npy"); old_pv = np.load(old / "port_vis.npy", mmap_mode="r")
    t0 = time.time()
    for j, c in enumerate(cfgs):
        z = np.load(D / split / f"img_{c['i']:05d}.npz")
        p = z["pos"].astype(np.float64); nn = z["nrm"].astype(np.float64); cam = z["cam"].astype(np.float64)
        nn = np.where(((nn * cam).sum(1) < 0)[:, None], -nn, nn)
        bo = None if c["box_offset"] is None else np.asarray(c["box_offset"], np.float64)
        if bo is not None:
            boff[j] = bo
        dom = np.zeros(len(p), np.int64)
        if bo is not None:
            dom[np.all((p >= blo + bo - 1e-3) & (p <= bhi + bo + 1e-3), 1)] = 1
        geo_cam = R.Geo(np.zeros(3), bo)
        port = np.full(len(p), -1, np.int64); ma = dom == 0
        pa, ok = M.pixels_to_ports(A, geo_cam, p[ma], cam[ma]); port[np.flatnonzero(ma)[ok]] = pa[ok]
        if (dom == 1).any():
            mb = dom == 1
            port[mb] = RR.map_hits(A, geo_cam, np.ones(int(mb.sum()), bool), p[mb], np.zeros(int(mb.sum()), np.int64),
                                   np.full(int(mb.sum()), geo_cam.names.index("B")), -cam[mb])
        keep = port >= 0
        rec["pixels_in"] += len(p); rec["dropped_untraced_px"] += int((~keep).sum())
        geo = R.Geo(np.zeros(3), bo, masks=c["masks"])
        T = RP.light_table(c["lights"]); J = len(T); LT[j, :J] = T; nl[j] = J
        vb = np.zeros(len(p), np.uint32)
        for k in range(J):
            vb |= (visible_t(geo, p + C.EPS1 * nn, nn, T[k]).astype(np.uint32) << np.uint32(k))
        off = np.zeros((Np, 3)); off[A["obj"] == 1] = (bo if bo is not None else 0)
        present = (A["obj"] == 0) | (bo is not None)
        o_s = (sp_all + off[:, None] + C.EPS1 * sn_all)[svalid & present[:, None]]; n_s = sn_all[svalid & present[:, None]]
        owner = np.nonzero(svalid & present[:, None])[0]
        for k in range(J):
            v = visible_t(geo, o_s, n_s, T[k])
            pvis[j, k] = np.bincount(owner, v, Np).astype(np.uint8)
            rec["port_counts_changed_vs_r1"] += int((pvis[j, k] != old_pv[j, k]).sum())
        ov = np.asarray(old_vb[old_start[j]:old_start[j + 1]])
        if len(ov) == keep.sum():
            nv_ = vb[keep]; rec["bits_changed_vs_r1"] += int((nv_ != ov).sum()); rec["pixel_light_pairs"] += int(len(nv_)) * J
            for k in range(J):
                a_ = (ov >> np.uint32(k)) & 1; b_ = (nv_ >> np.uint32(k)) & 1
                rec["bits_0to1"] += int(((a_ == 0) & (b_ == 1)).sum()); rec["bits_1to0"] += int(((a_ == 1) & (b_ == 0)).sum())
        else:
            rec["bits_changed_vs_r1"] = -10 ** 9
        canon = p - np.where(dom[:, None] == 1, bo if bo is not None else 0, 0)
        for key, val in (("pos", canon[keep].astype(np.float32)), ("nrm", nn[keep].astype(np.float16)), ("cam", cam[keep].astype(np.float16)),
                         ("col", z["col"][keep].astype(np.float32)), ("dom", dom[keep].astype(np.uint8)), ("port", port[keep].astype(np.int32)),
                         ("vbits", vb[keep])):
            chunks[key].append(val)
        start.append(start[-1] + int(keep.sum()))
        if j % 50 == 0:
            print(f"{split} {j}/{len(cfgs)} {time.time() - t0:.0f}s lights={J} bits_changed={rec['bits_changed_vs_r1']}", flush=True)
    pvis.flush()
    for k, v in chunks.items():
        np.save(od / f"px_{k}.npy", np.concatenate(v))
    np.save(od / "start.npy", np.asarray(start, np.int64)); np.save(od / "lights.npy", LT); np.save(od / "nlights.npy", nl)
    np.save(od / "nsamp.npy", nsamp); np.save(od / "family.npy", np.array([c["family"] for c in cfgs])); np.save(od / "boxoff.npy", boff)
    rec.update({"pixels_kept": int(start[-1]), "seconds": time.time() - t0, "anchors_sha16": C.sha16(C.OUT1 / "anchors.npz")})
    (od / "meta.json").write_text(json.dumps(rec, indent=1)); print(rec)


if __name__ == "__main__":
    for sp in sys.argv[1:]:
        main(sp)
