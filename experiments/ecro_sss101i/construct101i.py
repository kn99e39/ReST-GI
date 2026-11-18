"""WL101I construction (contract v2 instruments). Same proposal stream and semantics as WL101H.

Changes from construct101h (instrumentation only; estimator columns n c v g L LL vL gL are computed
exactly as before):
  * junction class by the REQUESTED POINT's metric distance (<= b_j) to an adjacent junction plane or to
    Box B's surface, not by the origin side;
  * labels assigned after the exclusion class (excluded samples get label "excluded");
  * near-border shell sums (any of a, b, c, d within 0.02 of 0 or 1) for tie/boundary-band radiance;
  * silhouette sensitivity at offsets m/10 and m/100 with flip-level sums (counts, signed weight and
    weighted radiance and their squares) for exact paired differences and their sampling variance;
  * class radiance sums with squares (luminance) for the junction class.
"""
from types import SimpleNamespace

import numpy as np

from projections import Reduction, stack
import gpu101g as GG

LUMA = (0.2126, 0.7152, 0.0722)
CLASSES = ("singular", "ring", "cap", "junction", "inside_box")
LABELS = ("front", "back_panel", "back_sphere", "back_other", "miss_opening", "miss_other", "excluded")
SHELL = 0.02


def construct(reader, oracle, solids, contact, semantics, seed, spp, batch, tol, b_m, launches=None, unregistered=None):
    dr, R, mi = reader.dr, reader.R, reader.mi
    rt = reader.rt
    nested = Reduction(reader, reader.slots * 16)
    perslot = Reduction(reader, reader.slots)
    light = rt.lt.pos[0]
    max_coord, n_launch = 0., (spp // batch if launches is None else launches)
    fsel = lambda m: dr.select(m, R.F(1.), R.F(0.))
    for i in range(n_launch):
        slots, ray, hits, rng, sd, info = oracle.probes(batch, seed, i)
        out, count = oracle.trace(ray, hits, rng, sd, info["valid"], "probe")
        assert not bool(dr.any(count != info["valid"])[0])
        membership = {k: fsel(info[k]) for k in ("valid", "buried", "back", "tie", "invalid")}
        membership["boundary_miss"] = fsel(info["boundary_miss"]) if "boundary_miss" in info else dr.zeros(R.F, dr.width(slots))
        perslot.add(membership, slots)
        ids = dr.compress(info["valid"])
        g = lambda x: dr.gather(type(x), x, ids)
        hit, r, req = g(hits), g(ray), dr.gather(R.U, slots, ids)
        dr.eval(hit, r, req)
        q = SimpleNamespace(point=hit.p, ray=r, pid=req // 64, bn=req % 64)
        slot, s, t, err = reader.addresses(q, tol)
        max_coord = max(max_coord, err)
        a, b, c, d = reader.coordinates(q.point, -q.ray.d, q.pid, q.bn)
        border = dr.minimum(dr.minimum(dr.minimum(a, 1 - a), dr.minimum(b, 1 - b)),
                            dr.minimum(dr.minimum(c, 1 - c), dr.minimum(d, 1 - d)))
        w = -r.d
        cosx = hit.to_local(w).z
        sid = dr.gather(R.U, oracle.ids, q.pid)
        p = hit.p
        conflict = (solids.inside_sphere(p) & (sid != 6) & (dr.squared_norm(p - solids.c) < (solids.r - b_m) ** 2))
        conflict |= solids.inside_box(p) & (sid < 7) & (solids.box_surface_distance(p) > b_m)
        v_nat = hit.spawn_ray(w)
        n_nat = hit.spawn_ray_to(light)
        v_can, v_sing, _ = contact.spawn(hit, w)
        n_can, n_sing, _ = contact.spawn_to(hit, light)
        wl = dr.normalize(light - p)
        nee_on = (hit.to_local(wl).z > 1e-6) & (dr.dot(-wl, rt.lt.axis[0]) > rt.lt.cc[0]) & (sid != 6)
        singular = v_sing | (n_sing & nee_on)
        vray, nray = (v_can, n_can) if semantics == "candidate" else (v_nat, n_nat)
        in_sph = solids.inside_sphere(vray.o) | (nee_on & solids.inside_sphere(nray.o))
        in_box = solids.inside_box(vray.o) | (nee_on & solids.inside_box(nray.o))
        behind = solids.behind_plane(vray.o) | (nee_on & solids.behind_plane(nray.o))
        # Junction band by the requested point (register entries 2 and 3).
        # distance to the nearest OTHER room plane (for room rectangles their own plane is 0)
        others = R.F(1e30)
        for plane_dist, plane_sid in ((dr.abs(p.z), 0), (dr.abs(3 - p.z), 1), (dr.abs(2 - p.y), 2), (dr.abs(2 + p.x), 3), (dr.abs(2 - p.x), 4)):
            others = dr.minimum(others, dr.select(sid == plane_sid, R.F(1e30), plane_dist))
        near_room = (sid < 5) & (others <= b_m)
        near_boxb = ((sid == 0) & (solids.box_surface_distance(p) <= b_m)) | ((sid == 8) & (p.z <= b_m))
        cls = {}
        cls["singular"] = singular if semantics == "candidate" else (singular & mi.Bool(False))
        rest = ~cls["singular"]
        cls["ring"] = rest & in_sph & (sid != 6)
        cls["cap"] = rest & ~cls["ring"] & behind & (sid == 6)
        cls["junction"] = rest & ~cls["ring"] & ~cls["cap"] & (near_room | near_boxb)
        cls["inside_box"] = rest & ~cls["ring"] & ~cls["cap"] & ~cls["junction"] & (in_box | (in_sph & (sid == 6)) | behind)
        include = ~cls["singular"]
        y = rt.scene.ray_intersect(vray)
        dr.eval(y)
        hit_y = y.is_valid()
        front = hit_y & (y.to_local(-w).z > 1e-6)
        ysid = rt.shape_id(y)
        opening, seam = GG.room_exit(vray.o, w, dr, mi, R)
        miss = ~hit_y
        lab = {"front": include & front, "back_panel": include & hit_y & ~front & (ysid == 5),
               "back_sphere": include & hit_y & ~front & (ysid == 6),
               "back_other": include & hit_y & ~front & (ysid != 5) & (ysid != 6),
               "miss_opening": include & miss & opening, "miss_other": include & miss & ~opening,
               "excluded": ~include}
        Lc = [dr.gather(R.F, out[f"f2{k}"], ids) + dr.gather(R.F, out[f"fh{k}"], ids) for k in range(3)]
        Ly = LUMA[0] * Lc[0] + LUMA[1] * Lc[1] + LUMA[2] * Lc[2]
        inc = fsel(include)
        vis = fsel(front & include)
        gw = vis * cosx
        cols = {"n": inc, "c": cosx * inc, "v": vis, "g": gw}
        for k in range(3):
            cols.update({f"L{k}": Lc[k] * inc, f"LL{k}": Lc[k] * Lc[k] * inc, f"vL{k}": vis * Lc[k], f"gL{k}": gw * Lc[k]})
        # Silhouette sensitivity: the same side, offsets m/10 and m/100.
        for tag, scale in (("10", 0.1), ("100", 0.01)):
            o_s = p + (vray.o - p) * scale
            ys = rt.scene.ray_intersect(mi.Ray3f(o_s, w))
            dr.eval(ys)
            fs = fsel(ys.is_valid() & (ys.to_local(-w).z > 1e-6) & include)
            dv = fs - vis
            dg = dv * cosx
            cols.update({f"nf{tag}": dr.abs(dv), f"dg{tag}": dg, f"dgL{tag}": dg * Ly, f"dg2_{tag}": dg * dg,
                         f"dgL2_{tag}": dg * Ly * dg * Ly, f"dgdgL{tag}": dg * dg * Ly,
                         f"dv{tag}": dv, f"dvL{tag}": dv * Ly, f"dvL2_{tag}": dv * dv * Ly * Ly, f"nfL{tag}": dr.abs(dv) * Ly})
        shell = fsel(include & (border < SHELL))
        cols.update({"sh_n": shell, "sh_L": shell * Ly, "sh_LL": shell * Ly * Ly, "sh_v": shell * vis, "sh_vL": shell * vis * Ly,
                     "sh_g": shell * gw, "sh_gL": shell * gw * Ly, "sh_gLL": shell * gw * Ly * Ly})
        for k in CLASSES:
            cols[f"k_{k}"] = fsel(cls[k])
        jm = fsel(cls["junction"])
        cols.update({"j_L": jm * Ly, "j_LL": jm * Ly * Ly, "j_v": jm * vis, "j_g": jm * gw})
        for k in LABELS:
            cols[f"vl_{k}"] = fsel(lab[k])
        nested.add(cols, slot * 16 + s * 4 + t)
        perslot.add({"support_conflict": fsel(conflict)}, req)
        if unregistered is not None:
            bad = cls["inside_box"] | lab["miss_other"] | conflict
            bidx = dr.compress(bad)
            if dr.width(bidx) > 0 and len(unregistered) < 64:
                gg = lambda x: np.asarray(dr.gather(type(x), x, bidx))
                unregistered.append({"seed": seed, "launch": i, "slot": gg(req).tolist(),
                                     "p": [gg(p[k]).tolist() for k in range(3)], "w": [gg(w[k]).tolist() for k in range(3)],
                                     "o": [gg(vray.o[k]).tolist() for k in range(3)],
                                     "inside_box": gg(cls["inside_box"]).tolist(), "miss_other": gg(lab["miss_other"]).tolist(),
                                     "junction": gg(cls["junction"]).tolist(),
                                     "conflict": gg(conflict).tolist()})
    sums = nested.arrays()
    shape = (reader.slots, 4, 4)
    packet = {}
    for k, v in sums.items():
        if k[-1:].isdigit() and k[:-1] in ("L", "LL", "vL", "gL"):
            continue
        packet[k] = v.reshape(shape)
    for k in ("L", "LL", "vL", "gL"):
        packet[k] = stack(sums, k, shape)
    per = perslot.arrays()
    for k, v in per.items():
        packet[f"slot_{k}"] = v
    for k, v in packet.items():
        assert np.isfinite(v).all(), k
    meta = {"seed": seed, "semantics": semantics, "launches": n_launch, "proposals_per_slot": batch * n_launch,
            "coordinate_excursion": max_coord, "membership": {k: int(round(per[k].sum())) for k in per},
            "classes": {k: int(round(packet[f"k_{k}"].sum())) for k in CLASSES},
            "labels": {k: int(round(packet[f"vl_{k}"].sum())) for k in LABELS}}
    return packet, meta
