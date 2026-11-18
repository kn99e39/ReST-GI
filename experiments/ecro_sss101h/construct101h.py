"""WL101H construction with mechanism classes, origin validity and class-split sums (native or candidate).

One proposal stream (WL101E ExposedOracle + a membership rule) is traced under the semantics installed
on the runtime instance. Accumulated per (slot, s, t):

  base (included samples)  n c v g L LL vL gL                       (WL101G columns)
  mechanism counts         k_singular k_ring k_cap k_junction k_inside_box   (all valid samples)
  class radiance           L_ring L_cap L_junction                  (RGB sums over that class)
  visibility labels        vl_front vl_back_panel vl_back_sphere vl_back_other
                           vl_miss_opening vl_miss_junction vl_miss_other
  offset sensitivity       v_small g_small gL_small                 (visibility with 1/10 of the offset)

Per slot (all proposals): membership counts valid buried back tie boundary_miss invalid support_conflict.
Candidate semantics exclude the registered singular class from every included column.
"""
import math
from types import SimpleNamespace

import numpy as np

from projections import Reduction, stack
import gpu101g as GG

CLASSES = ("singular", "ring", "cap", "junction", "inside_box")
LABELS = ("front", "back_panel", "back_sphere", "back_other", "miss_opening", "miss_junction", "miss_other")


def metric_oracle(base_cls, b_m, edge_lengths):
    """New-contract membership: an attachment miss is a boundary miss iff its metric distance to the patch
    boundary is <= b_m (float32 error model, Q1b-validated). Point-in-other-solid beyond b_m is a support
    conflict (semantic, counted)."""
    class MetricOracle(base_cls):
        def probes(self, spp, seed, launch, diagnostic=None):
            dr, mi, R = self.dr, self.mi, self.R
            slots, ray, hits, rng, sd, info = super().probes(spp, seed, launch, diagnostic)
            replay = mi.PCG32(size=self.ns * spp, initstate=sd, initseq=mi.UInt64(0))
            u0, u1 = replay.next_float32(), replay.next_float32()
            pid = slots // 64
            sid = dr.gather(R.U, self.ids, pid)
            l1 = dr.gather(R.F, edge_lengths[0], pid)
            l2 = dr.gather(R.F, edge_lengths[1], pid)
            dist = dr.minimum(dr.minimum(u0, 1 - u0) * l1, dr.minimum(u1, 1 - u1) * l2)
            boundary = info["invalid"] & ~hits.is_valid() & (sid != 6) & (dist <= b_m)
            info["boundary_miss"] = boundary
            info["tie"] = info["tie"] | boundary
            info["invalid"] = info["invalid"] & ~boundary
            return slots, ray, hits, rng, sd, info
    return MetricOracle


class Solids:
    def __init__(self, R, rt, geometry):
        self.R, self.mi, self.dr = R, R.mi, R.dr
        self.c = R.mi.Point3f(*[float(v) for v in rt.SC])
        self.r = float(rt.SR)
        self.boxes = []
        for name in ("A", "B"):
            b = geometry[name]
            cs, sn = math.cos(b["yaw"]), math.sin(b["yaw"])
            self.boxes.append((R.mi.Point3f(*b["c"]), (cs, sn), b["h"]))

    def box_local(self, o, box):
        dr = self.dr
        c, (cs, sn), h = box
        q = o - c
        return q.x * cs + q.y * sn, -q.x * sn + q.y * cs, q.z, h

    def inside_sphere(self, o):
        return self.dr.squared_norm(o - self.c) < self.r * self.r

    def inside_box(self, o):
        dr = self.dr
        res, near = self.R.mi.Bool(False), self.R.mi.Bool(False)
        for box in self.boxes:
            x, y, z, h = self.box_local(o, box)
            res |= (dr.abs(x) < h[0]) & (dr.abs(y) < h[1]) & (dr.abs(z) < h[2])
        return res

    def box_surface_distance(self, p):
        """Smallest |signed distance| of p to a box surface (axis-wise max form)."""
        dr = self.dr
        best = self.R.F(1e30)
        for box in self.boxes:
            x, y, z, h = self.box_local(p, box)
            d = dr.maximum(dr.maximum(dr.abs(x) - h[0], dr.abs(y) - h[1]), dr.abs(z) - h[2])
            best = dr.minimum(best, dr.abs(d))
        return best

    def behind_plane(self, o):
        return (o.z <= 0) | (o.z >= 3) | (o.x <= -2) | (o.x >= 2) | (o.y >= 2)


def construct(reader, oracle, solids, contact, semantics, seed, spp, batch, tol, b_m, launches=None,
              monitor=None, unregistered=None):
    """contact: semantics101h.Contact (classification in native mode; spawning in candidate mode)."""
    dr, R, mi = reader.dr, reader.R, reader.mi
    rt = reader.rt
    nested = Reduction(reader, reader.slots * 16)
    perslot = Reduction(reader, reader.slots)
    light = rt.lt.pos[0]
    max_coord, n_launch = 0., (spp // batch if launches is None else launches)
    for i in range(n_launch):
        slots, ray, hits, rng, sd, info = oracle.probes(batch, seed, i)
        out, count = oracle.trace(ray, hits, rng, sd, info["valid"], "probe")
        assert not bool(dr.any(count != info["valid"])[0])
        membership = {k: dr.select(info[k], R.F(1.), R.F(0.)) for k in ("valid", "buried", "back", "tie", "invalid")}
        membership["boundary_miss"] = dr.select(info["boundary_miss"], R.F(1.), R.F(0.)) if "boundary_miss" in info \
            else dr.zeros(R.F, dr.width(slots))
        ids = dr.compress(info["valid"])
        g = lambda x: dr.gather(type(x), x, ids)
        hit, r, req = g(hits), g(ray), dr.gather(R.U, slots, ids)
        dr.eval(hit, r, req)
        q = SimpleNamespace(point=hit.p, ray=r, pid=req // 64, bn=req % 64)
        slot, s, t, err = reader.addresses(q, tol)
        max_coord = max(max_coord, err)
        w = -r.d
        cosx = hit.to_local(w).z
        sid = dr.gather(R.U, oracle.ids, q.pid)
        # S1: requested point must not lie inside another solid beyond the metric band.
        conflict = (solids.inside_sphere(hit.p) & (sid != 6) &
                    (dr.squared_norm(hit.p - solids.c) < (solids.r - b_m) ** 2))
        conflict |= solids.inside_box(hit.p) & (sid < 7) & (solids.box_surface_distance(hit.p) > b_m)
        # Native and candidate origins (the candidate is computed in both modes for classification).
        v_nat = hit.spawn_ray(w)
        n_nat = hit.spawn_ray_to(light)
        v_can, v_sing, _ = contact.spawn(hit, w)
        n_can, n_sing, _ = contact.spawn_to(hit, light)
        wl0 = dr.normalize(light - hit.p)
        nee_used = (hit.to_local(wl0).z > 1e-6) & (dr.dot(-wl0, rt.lt.axis[0]) > rt.lt.cc[0]) & (sid != 6)
        singular = v_sing | (n_sing & nee_used)
        if semantics == "candidate":
            vray, nray = v_can, n_can
        else:
            vray, nray = v_nat, n_nat
        # The NEE origin matters only where pt100.direct actually uses it (light on the front side, inside
        # the cone, non-SSS vertex); elsewhere spawn_ray_to's offset to the far side is never traced.
        wl = dr.normalize(light - hit.p)
        nee_on = (hit.to_local(wl).z > 1e-6) & (dr.dot(-wl, rt.lt.axis[0]) > rt.lt.cc[0]) & (sid != 6)
        in_sph = solids.inside_sphere(vray.o) | (nee_on & solids.inside_sphere(nray.o))
        in_box = solids.inside_box(vray.o) | (nee_on & solids.inside_box(nray.o))
        behind = solids.behind_plane(vray.o) | (nee_on & solids.behind_plane(nray.o))
        near_box = solids.box_surface_distance(hit.p) <= b_m
        cls = {}
        cls["singular"] = singular if semantics == "candidate" else (singular & mi.Bool(False))
        rest = ~cls["singular"]
        cls["ring"] = rest & in_sph & (sid != 6)
        cls["cap"] = rest & ~cls["ring"] & behind & (sid == 6)
        cls["junction"] = rest & ~cls["ring"] & ~cls["cap"] & (behind | (in_box & near_box))
        cls["inside_box"] = rest & ~cls["ring"] & ~cls["cap"] & ~cls["junction"] & (in_box | (in_sph & (sid == 6)))
        include = ~cls["singular"]
        # Visibility and mechanism labels.
        y = rt.scene.ray_intersect(vray)
        dr.eval(y)
        hit_y = y.is_valid()
        front = hit_y & (y.to_local(-w).z > 1e-6)
        ysid = rt.shape_id(y)
        opening, seam = GG.room_exit(vray.o, w, dr, mi, R)
        miss = ~hit_y
        lab = {"front": front, "back_panel": hit_y & ~front & (ysid == 5), "back_sphere": hit_y & ~front & (ysid == 6),
               "back_other": hit_y & ~front & (ysid != 5) & (ysid != 6), "miss_opening": miss & opening,
               "miss_junction": miss & ~opening & (behind | seam), "miss_other": miss & ~opening & ~(behind | seam)}
        # Offset sensitivity: same side, one tenth of the offset actually used.
        o_small = hit.p + (vray.o - hit.p) * 0.1
        ys = rt.scene.ray_intersect(mi.Ray3f(o_small, w))
        dr.eval(ys)
        front_small = ys.is_valid() & (ys.to_local(-w).z > 1e-6)
        fsel = lambda m: dr.select(m, R.F(1.), R.F(0.))
        inc = fsel(include)
        vis = fsel(front & include)
        gw = vis * cosx
        vis_s = fsel(front_small & include)
        gs = vis_s * cosx
        cols = {"n": inc, "c": cosx * inc, "v": vis, "g": gw, "v_small": vis_s, "g_small": gs}
        for c in range(3):
            L = dr.gather(R.F, out[f"f2{c}"], ids) + dr.gather(R.F, out[f"fh{c}"], ids)
            cols.update({f"L{c}": L * inc, f"LL{c}": L * L * inc, f"vL{c}": vis * L, f"gL{c}": gw * L,
                         f"gL_small{c}": gs * L})
            for k in ("ring", "cap", "junction"):
                cols[f"L_{k}{c}"] = fsel(cls[k]) * L
        for k in CLASSES:
            cols[f"k_{k}"] = fsel(cls[k])
        for k in LABELS:
            cols[f"vl_{k}"] = fsel(lab[k])
        nested.add(cols, slot * 16 + s * 4 + t)
        perslot.add(membership, slots)
        perslot.add({"support_conflict": fsel(conflict)}, req)
        if unregistered is not None:
            bad = cls["inside_box"] | lab["miss_other"] | conflict
            bidx = dr.compress(bad)
            if dr.width(bidx) > 0 and len(unregistered) < 64:
                gg = lambda x: np.asarray(dr.gather(type(x), x, bidx))
                unregistered.append({"seed": seed, "launch": i, "slot": gg(req).tolist(),
                                     "p": [gg(hit.p[k]).tolist() for k in range(3)],
                                     "w": [gg(w[k]).tolist() for k in range(3)],
                                     "inside_box": gg(cls["inside_box"]).tolist(), "miss_other": gg(lab["miss_other"]).tolist(),
                                     "conflict": gg(conflict).tolist()})
        if monitor is not None and (i + 1) % 32 == 0:
            monitor(f"construction {semantics} {seed} {batch * (i + 1)}/{spp}")
    sums = nested.arrays()
    shape = (reader.slots, 4, 4)
    packet = {}
    for k in ("n", "c", "v", "g", "v_small", "g_small", *[f"k_{c}" for c in CLASSES], *[f"vl_{l}" for l in LABELS]):
        packet[k] = sums[k].reshape(shape)
    for k in ("L", "LL", "vL", "gL", "gL_small", "L_ring", "L_cap", "L_junction"):
        packet[k] = stack(sums, k, shape)
    per = perslot.arrays()
    for k, v in per.items():
        packet[f"slot_{k}"] = v
    for k, v in packet.items():
        assert np.isfinite(v).all(), k
    meta = {"seed": seed, "semantics": semantics, "launches": n_launch, "proposals_per_slot": batch * n_launch,
            "coordinate_excursion": max_coord,
            "membership": {k: int(round(per[k].sum())) for k in per},
            "classes": {k: int(round(packet[f"k_{k}"].sum())) for k in CLASSES},
            "labels": {k: int(round(packet[f"vl_{k}"].sum())) for k in LABELS}}
    return packet, meta
