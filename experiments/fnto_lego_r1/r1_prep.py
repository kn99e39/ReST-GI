"""WL42 Stage A r1 preprocessing (PC, Mitsuba). Geometry only; the target radiance `col` is copied unchanged.

    python r1_prep.py <split>            # reads results/fnto_lego_r1/stagea/<split>/, writes .../prep_<split>/

Per image (its own geometry: Lego, box at its offset or absent, shadow-only masks):
  pixels : domain (box AABB), canonical position, camera-facing normal, camera direction, target; port by SURFACE
           IDENTITY (the triangle the camera ray hits, r1_map; box: analytic grid) -- pixels whose identity cannot be
           traced are dropped and counted; per-light visibility bits from the pixel position (EPS1 along its normal).
  ports  : per light, the number (0..S) of the port's surface samples that see the light (sun: direction; point/spot:
           segment to the light position; spot: inside its cone).
  lights : [J, 12] = type (0 sun, 1 point, 2 spot), position or direction (3), spot axis (3), cos half-angle,
           intensity (sun: irradiance S; point/spot: P / (4 pi) W/sr), 3 spare.
Incident fields are reconstructed exactly from these (r1_train): E_j(s) = I cos / r^2 x visible fraction (point/spot),
S cos x visible fraction (sun).
"""

from __future__ import annotations

import json
import math
import sys
import time

import numpy as np

import r1_common as C
import fnto_relations as R
import r1_map as M
import r1_relations as RR

D = C.OUT1 / "stagea"
JMAX = 24


def light_table(lights):
    T = np.zeros((len(lights), 12), np.float32)
    for j, L in enumerate(lights):
        if L["type"] == "sun":
            T[j, 0] = 0; T[j, 1:4] = L["dir"]; T[j, 8] = L["strength"]
        elif L["type"] == "point":
            T[j, 0] = 1; T[j, 1:4] = L["pos"]; T[j, 8] = L["power"] / (4 * math.pi)
        else:
            T[j, 0] = 2; T[j, 1:4] = L["pos"]; T[j, 4:7] = L["axis"]; T[j, 7] = L["cos_half"]; T[j, 8] = L["power"] / (4 * math.pi)
    return T


def visible(geo, o, n, L, chunk=2_000_000):
    """Visibility of points o (normals n) to light row L (incl. the facing test and the spot cone)."""
    out = np.zeros(len(o), bool)
    if L[0] == 0:
        w = L[1:4].astype(np.float64); face = (n @ w) > 0; dvec = np.broadcast_to(w, o.shape); tmax = None
    else:
        v = L[1:4].astype(np.float64)[None] - o; dist = np.linalg.norm(v, axis=1); dvec = v / dist[:, None]
        face = (n * dvec).sum(1) > 0
        if L[0] == 2:
            face &= ((-dvec) @ L[4:7].astype(np.float64)) >= L[7]
        tmax = dist - 1e-4
    idx = np.flatnonzero(face)
    for s in range(0, len(idx), chunk):
        ii = idx[s:s + chunk]
        dd = np.ascontiguousarray(dvec[ii])
        if tmax is None:
            out[ii] = ~geo.occluded(o[ii], dd)
        else:
            mi = geo.mi; f = lambda a: mi.Float(np.ascontiguousarray(a.astype(np.float32)))  # noqa: E731
            ray = mi.Ray3f(mi.Point3f(f(o[ii, 0]), f(o[ii, 1]), f(o[ii, 2])), mi.Vector3f(f(dd[:, 0]), f(dd[:, 1]), f(dd[:, 2])))
            ray.maxt = f(tmax[ii])
            out[ii] = ~np.array(geo.scene.ray_test(ray))
    return out


def main(split):
    A = RR.load(); Np = len(A["obj"]); nA = int(A["n_A"])
    cfgs = sorted(json.loads((D / split / "configs.json").read_text()), key=lambda c: c["i"])
    od = D / f"prep_{split}"; od.mkdir(parents=True, exist_ok=True)
    bV = np.load(C.BOX_T0)["V"]; blo, bhi = bV.min(0), bV.max(0)
    sp_all = A["spos"].astype(np.float64); sn_all = A["snrm"].astype(np.float64); svalid = np.isfinite(sp_all[..., 0])
    nsamp = svalid.sum(1).astype(np.uint8)
    chunks = {k: [] for k in ("pos", "nrm", "cam", "col", "dom", "port", "vbits")}
    start = [0]; LT = np.zeros((len(cfgs), JMAX, 12), np.float32); nl = np.zeros(len(cfgs), np.int32)
    pvis = np.lib.format.open_memmap(od / "port_vis.npy", "w+", np.uint8, (len(cfgs), JMAX, Np))
    rec = {"images": len(cfgs), "dropped_untraced_px": 0, "pixels_in": 0}
    t0 = time.time()
    for j, c in enumerate(cfgs):
        z = np.load(D / split / f"img_{c['i']:05d}.npz")
        p = z["pos"].astype(np.float64); nn = z["nrm"].astype(np.float64); cam = z["cam"].astype(np.float64)
        nn = np.where(((nn * cam).sum(1) < 0)[:, None], -nn, nn)
        bo = None if c["box_offset"] is None else np.asarray(c["box_offset"], np.float64)
        dom = np.zeros(len(p), np.int64)
        if bo is not None:
            dom[np.all((p >= blo + bo - 1e-3) & (p <= bhi + bo + 1e-3), 1)] = 1
        geo_cam = R.Geo(np.zeros(3), bo)                                      # identity tracing: no masks
        port = np.full(len(p), -1, np.int64)
        ma = dom == 0
        pa, ok = M.pixels_to_ports(A, geo_cam, p[ma], cam[ma]); port[np.flatnonzero(ma)[ok]] = pa[ok]
        if (dom == 1).any():
            mb = dom == 1; q = p[mb] - bo
            port[mb] = RR.map_hits(A, geo_cam, np.ones(int(mb.sum()), bool), p[mb], np.zeros(int(mb.sum()), np.int64),
                                   np.full(int(mb.sum()), geo_cam.names.index("B")), -cam[mb])
            del q
        keep = port >= 0
        rec["pixels_in"] += len(p); rec["dropped_untraced_px"] += int((~keep).sum())
        geo = R.Geo(np.zeros(3), bo, masks=c["masks"])
        T = light_table(c["lights"]); J = len(T); LT[j, :J] = T; nl[j] = J
        vb = np.zeros(len(p), np.uint32)
        for k in range(J):
            vb |= (visible(geo, p + C.EPS1 * nn, nn, T[k]).astype(np.uint32) << np.uint32(k))
        off = np.zeros((Np, 3)); off[A["obj"] == 1] = (bo if bo is not None else 0)
        present = (A["obj"] == 0) | (bo is not None)
        o_s = (sp_all + off[:, None] + C.EPS1 * sn_all)[svalid & present[:, None]]; n_s = sn_all[svalid & present[:, None]]
        owner = np.nonzero(svalid & present[:, None])[0]
        for k in range(J):
            v = visible(geo, o_s, n_s, T[k])
            pvis[j, k] = np.bincount(owner, v, Np).astype(np.uint8)
        canon = p - np.where(dom[:, None] == 1, bo if bo is not None else 0, 0)
        for key, val in (("pos", canon[keep].astype(np.float32)), ("nrm", nn[keep].astype(np.float16)), ("cam", cam[keep].astype(np.float16)),
                         ("col", z["col"][keep].astype(np.float32)), ("dom", dom[keep].astype(np.uint8)), ("port", port[keep].astype(np.int32)),
                         ("vbits", vb[keep])):
            chunks[key].append(val)
        start.append(start[-1] + int(keep.sum()))
        if j % 50 == 0:
            print(f"{split} {j}/{len(cfgs)} {time.time() - t0:.0f}s lights={J}", flush=True)
    pvis.flush()
    for k, v in chunks.items():
        np.save(od / f"px_{k}.npy", np.concatenate(v))
    np.save(od / "start.npy", np.asarray(start, np.int64)); np.save(od / "lights.npy", LT); np.save(od / "nlights.npy", nl)
    np.save(od / "nsamp.npy", nsamp); np.save(od / "family.npy", np.array([c["family"] for c in cfgs]))
    rec.update({"pixels_kept": int(start[-1]), "seconds": time.time() - t0, "anchors_sha16": C.sha16(C.OUT1 / "anchors.npz")})
    (od / "meta.json").write_text(json.dumps(rec, indent=1)); print(rec)


if __name__ == "__main__":
    for sp in sys.argv[1:]:
        main(sp)
