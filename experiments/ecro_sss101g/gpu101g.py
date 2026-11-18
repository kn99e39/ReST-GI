"""WL101G GPU stages: camera-independent construction, per-view controls, held-out readout."""
import time
from types import SimpleNamespace

import numpy as np

from projections import Reduction, stack, weight_y
from query_readout import LUMA, QueryReadout

ROOM = {"x": (-2.0, 2.0), "y": (-2.0, 2.0), "z": (0.0, 3.0)}


def amended_oracle(base_cls, window):
    """WL101G Amendment 1: reclassify planar native misses inside the existing coordinate tie window.

    The WL101E ExposedOracle classification runs unchanged first. A proposal it marks invalid becomes
    a numerical tie only if the native attachment query returned no hit at all (so no wrong-surface
    hit), the requested shape is planar, and min(u0, 1-u0, u1, 1-u1) <= window. u0, u1 are
    regenerated from the identical PCG32 stream.
    """
    class AmendedOracle(base_cls):
        def probes(self, spp, seed, launch, diagnostic=None):
            dr, mi, R = self.dr, self.mi, self.R
            slots, ray, hits, rng, sd, info = super().probes(spp, seed, launch, diagnostic)
            replay = mi.PCG32(size=self.ns * spp, initstate=sd, initseq=mi.UInt64(0))
            u0, u1 = replay.next_float32(), replay.next_float32()
            sid = dr.gather(R.U, self.ids, slots // 64)
            edge = dr.minimum(dr.minimum(u0, 1 - u0), dr.minimum(u1, 1 - u1))
            amended = info["invalid"] & ~hits.is_valid() & (sid != 6) & (edge <= window)
            info["tie"] = info["tie"] | amended
            info["invalid"] = info["invalid"] & ~amended
            info["amended_tie"] = amended
            return slots, ray, hits, rng, sd, info
    return AmendedOracle


def room_exit(o, w, dr, mi, R):
    """Exit face of the room box along o + t w, and whether the exit point is within 1e-3 m of an edge."""
    big = R.F(1e30)
    def plane(lo, hi, oc, wc):
        t_hi = dr.select(wc > 0, (hi - oc) / dr.select(wc > 0, wc, 1.0), big)
        t_lo = dr.select(wc < 0, (lo - oc) / dr.select(wc < 0, wc, -1.0), big)
        return t_lo, t_hi
    tx = plane(*ROOM["x"], o.x, w.x)
    ty = plane(*ROOM["y"], o.y, w.y)
    tz = plane(*ROOM["z"], o.z, w.z)
    t = dr.minimum(dr.minimum(dr.minimum(tx[0], tx[1]), dr.minimum(ty[0], ty[1])), dr.minimum(tz[0], tz[1]))
    p = o + w * t
    opening = (ty[0] <= t) & (p.x >= -2.0 - 1e-3) & (p.x <= 2.0 + 1e-3) & (p.z >= -1e-3) & (p.z <= 3.0 + 1e-3)
    near_edge = R.U(0)
    for axis, (lo, hi) in zip((p.x, p.y, p.z), ROOM.values()):
        near_edge += R.U(dr.minimum(dr.abs(axis - lo), dr.abs(axis - hi)) <= 1e-3)
    return opening, near_edge >= 2


def construct(reader, oracle, cfg, seed, monitor, launches=None):
    """Nested camera-independent sums for Un, V and G at (slot, s, t)."""
    dr, R, mi = reader.dr, reader.R, reader.mi
    spp, batch = cfg["proposals_per_slot"], cfg["launch_proposals"]
    tol = cfg["gates"]["address_coordinate_tolerance"]
    nested = Reduction(reader, reader.slots * 16)
    totals = {k: 0 for k in ("valid", "buried", "back", "tie", "invalid")}
    segments = {k: 0 for k in ("front", "back_receiver", "opening", "seam", "tangency", "unexplained")}
    max_coord, n_launch = 0., (spp // batch if launches is None else launches)
    amended_ties = 0
    scene = oracle.scene
    for i in range(n_launch):
        slots, ray, hits, rng, sd, info = oracle.probes(batch, seed, i)
        out, count = oracle.trace(ray, hits, rng, sd, info["valid"], "probe")
        assert not bool(dr.any(count != info["valid"])[0])
        for key in totals:
            totals[key] += int(dr.count(info[key])[0])
        amended_ties += int(dr.count(info["amended_tie"])[0]) if "amended_tie" in info else 0
        ids = dr.compress(info["valid"])
        grab = lambda x: dr.gather(type(x), x, ids)
        hit, r, requested = grab(hits), grab(ray), dr.gather(R.U, slots, ids)
        q = SimpleNamespace(point=hit.p, ray=r, pid=requested // 64, bn=requested % 64)
        slot, s, t, err = reader.addresses(q, tol)
        max_coord = max(max_coord, err)
        assert bool(dr.all(slot == requested)[0]), "Re-addressed proposal left its requested slot"
        w = -r.d
        cosx = hit.to_local(w).z
        y = scene.ray_intersect(hit.spawn_ray(w))
        hit_y = y.is_valid()
        front = hit_y & (y.to_local(-w).z > 1e-6)
        opening, seam = room_exit(hit.p, w, dr, mi, R)
        miss = ~hit_y
        # Sphere samples within 1 mm of the floor tangency whose spawned ray starts below the floor plane.
        tangent = (dr.gather(R.U, oracle.ids, q.pid) == 6) & (hit.p.z < 1e-3) & (w.z < 0)
        other = miss & ~opening & ~seam
        categories = {"front": front, "back_receiver": hit_y & ~front, "opening": miss & opening,
                      "seam": miss & ~opening & seam, "tangency": other & tangent, "unexplained": other & ~tangent}
        for key, mask in categories.items():
            segments[key] += int(dr.count(mask)[0])
        vis = dr.select(front, R.F(1.), R.F(0.))
        g = vis * cosx
        columns = {"n": dr.full(R.F, 1., dr.width(ids)), "c": cosx, "v": vis, "g": g}
        for c in range(3):
            L = dr.gather(R.F, out[f"f2{c}"], ids) + dr.gather(R.F, out[f"fh{c}"], ids)
            columns.update({f"L{c}": L, f"LL{c}": L * L, f"vL{c}": vis * L, f"gL{c}": g * L})
        nested.add(columns, slot * 16 + s * 4 + t)
        if monitor is not None and (i + 1) % 16 == 0:
            monitor(f"construction {seed} {batch * (i + 1)}/{spp}")
    sums = nested.arrays()
    shape = (reader.slots, 4, 4)
    packet = {k: sums[k].reshape(shape) for k in ("n", "c", "v", "g")}
    for k in ("L", "LL", "vL", "gL"):
        packet[k] = stack(sums, k, shape)
    for k, v in packet.items():
        assert np.isfinite(v).all(), k
    assert int(round(packet["n"].sum())) == totals["valid"]
    assert sum(segments.values()) == totals["valid"]
    meta = {"seed": seed, "launches": n_launch, "proposals_per_slot": batch * n_launch,
            "membership": totals, "segments": segments, "coordinate_excursion": max_coord,
            "amendment1_ties": amended_ties}
    return packet, meta


def train_view(reader, proxy, seed, suffix_seed, spp, batch, tol, monitor):
    """WL101F projection sums with full-frame equal pixel weight on one view (original pixel generator)."""
    dr, R = reader.dr, reader.R
    nested = Reduction(reader, reader.slots * 16)
    queries, max_coord = 0, 0.
    for i in range(spp // batch):
        ray, pix, rng, sd = reader.rays(batch, seed, i)
        _, _, q = reader.prefix(ray, pix, rng, sd)
        slot, s, t, err = reader.addresses(q, tol)
        max_coord = max(max_coord, err)
        queries += dr.width(q.pix)
        suffix = reader.suffix(q, sd, proxy, suffix_seed * 1000003 + i)
        columns = {"count": dr.full(R.F, 1., dr.width(q.pix)), "mass": dr.full(R.F, 1., dr.width(q.pix))}
        for c in range(3):
            light = suffix[f"f2{c}"] + suffix[f"fh{c}"]
            columns.update({f"L{c}": light, f"W{c}": q.weight[c], f"WL{c}": q.weight[c] * light})
        nested.add(columns, slot * 16 + s * 4 + t)
        if monitor is not None:
            monitor()
    result = nested.arrays()
    shape = (reader.slots, 4, 4)
    for k in ("count", "mass"):
        result[k] = result[k].reshape(shape)
    result["count"] = result["count"].astype(np.uint32)
    for k in ("L", "W", "WL"):
        result[k] = stack(result, k, shape)
        for c in range(3):
            del result[f"{k}{c}"]
        assert np.isfinite(result[k]).all() and result[k].min() >= 0
    assert int(result["count"].sum()) == queries
    return result, {"prefix_seed": seed, "suffix_seed": suffix_seed, "spp_per_pixel": spp,
                    "positive_queries": queries, "coordinate_excursion": max_coord}


class DeviceArms:
    """Upload tables once. kind: slot | S | A | SA | AD."""
    def __init__(self, reader, layout=None):
        self.reader, self.R, self.mi, self.dr = reader, reader.R, reader.mi, reader.dr
        self.arms = {}
        self.layout = None
        if layout is not None:
            offsets, refined = layout
            self.layout = (self.R.U(offsets.astype(np.uint32)), self.mi.Bool(refined.astype(bool)))

    def add(self, name, table, native, kind):
        R, mi = self.R, self.mi
        table = np.ascontiguousarray(table, dtype=np.float32)
        native = np.ascontiguousarray(np.broadcast_to(np.asarray(native, dtype=bool), table.shape)).reshape(-1, 3)
        table = table.reshape(-1, 3)
        self.arms[name] = (R.PT.fl(table), tuple(mi.Bool(np.ascontiguousarray(native[:, c])) for c in range(3)), kind)

    def indices(self, slot, s, t):
        dr, R, mi = self.dr, self.R, self.mi
        idx = {"slot": slot, "S": slot * 4 + s, "A": slot * 4 + t, "SA": slot * 16 + s * 4 + t}
        if self.layout is not None:
            offsets, refined = self.layout
            idx["AD"] = dr.gather(R.U, offsets, slot) + dr.select(dr.gather(mi.Bool, refined, slot), s, R.U(0))
        return idx


def camera_pass(reader, proxy, seed, spp, batch, learned, arms, psid, tol, monitor, group=6):
    """One evaluation stream: LOW, learned HIGH, exact physical HIGH and every table arm on shared paths."""
    R, dr, mi = reader.R, reader.dr, reader.mi
    learned_dev = R.PT.fl(learned)
    psid_u = R.U(psid.astype(np.uint32))
    names = list(arms.arms)
    collect = Reduction(reader, reader.n)
    total_queries, coord = 0, 0.
    for i in range(spp // batch):
        ray, pix, rng, sd = reader.rays(batch, seed, i)
        prefix, _, q = reader.prefix(ray, pix, rng, sd, learned_dev)
        collect.add({**{k: v for k, v in prefix.items() if k[:2] in ("e1", "e2")},
                     **{f"learned{c}": prefix[f"hl{c}"] for c in range(3)}}, pix)
        slot, s, t, err = reader.addresses(q, tol)
        coord = max(coord, err)
        total_queries += dr.width(q.pix)
        field = reader.suffix(q, sd, proxy)
        wy = weight_y(q)
        sid = dr.gather(R.U, psid_u, q.pid)
        cols = {"total_weight": wy}
        for c in range(3):
            cols[f"query{c}"] = q.weight[c] * (field[f"f2{c}"] + field[f"fh{c}"])
        for k in range(9):
            cols[f"shape{k}"] = dr.select(sid == k, wy, 0.)
        collect.add(cols, q.pix)
        index = arms.indices(slot, s, t)
        for start in range(0, len(names), group):
            cols = {}
            for name in names[start:start + group]:
                table, native, kind = arms.arms[name]
                idx = index[kind]
                for c, v in enumerate(reader.gather(table, idx, q)):
                    cols[f"{name}:{c}"] = v
                cols[f"{name}:weight"] = sum(LUMA[c] * dr.select(dr.gather(mi.Bool, native[c], idx), q.weight[c], 0.)
                                             for c in range(3))
            collect.add(cols, q.pix)
        if monitor is not None and (i + 1) % 16 == 0:
            monitor(f"camera {seed} {batch * (i + 1)}/{spp}")
    arrays = collect.arrays()
    res = reader.res
    shape = (res, res)
    out = {"total_weight": (arrays["total_weight"] / spp).reshape(shape).astype(np.float32),
           "shape_weight": np.stack([arrays[f"shape{k}"] / spp for k in range(9)]).reshape(9, res, res).astype(np.float32)}
    for key in ("e1", "e2", "learned", "query"):
        out[key] = (stack(arrays, key, shape) / spp).astype(np.float32)
    for name in names:
        out[name] = (stack(arrays, name + ":", shape) / spp).astype(np.float32)
        out[name + ":weight"] = (arrays[name + ":weight"] / spp).reshape(shape).astype(np.float32)
    return out, {"seed": seed, "spp": spp, "positive_queries": total_queries, "coordinate_excursion": coord}


def numpy_subcells(geom, point, w, pid, bn):
    """Independent float64 recomputation of the WL101F s/t subcell coordinates."""
    point, w = np.asarray(point, np.float64), np.asarray(w, np.float64)
    pid, bn = np.asarray(pid, np.int64), np.asarray(bn, np.int64)
    po, e1, e2 = geom.PO[pid], geom.PE1[pid], geom.PE2[pid]
    d = point - po
    with np.errstate(invalid="ignore", divide="ignore"):  # sphere rows are replaced below
        a = np.einsum("ij,ij->i", d, e1) / np.einsum("ij,ij->i", e1, e1)
        b = np.einsum("ij,ij->i", d, e2) / np.einsum("ij,ij->i", e2, e2)
    sph = pid >= geom.SBASE
    qs = (point - np.asarray(geom.SC)) / geom.SR
    ip, iz = geom.PSPH[pid, 0], geom.PSPH[pid, 1]
    a = np.where(sph, (np.arctan2(qs[:, 1], qs[:, 0]) + np.pi) * (8 / (2 * np.pi)) - ip, a)
    b = np.where(sph, (qs[:, 2] + 1) * (4 / 2) - iz, b)
    return a, b


def lookup_checks(reader, learned, seed, tol, refined, offsets):
    """Exact parent-broadcast equivalence, independent subcell coordinates and adaptive index at 64^2 x 1 spp."""
    R, dr, mi = reader.R, reader.dr, reader.mi
    ray, pix, rng, sd = reader.rays(1, seed, 0, res=64)
    _, _, q = reader.prefix(ray, pix, rng, sd)
    slot, s, t, _ = reader.addresses(q, tol)
    rs = np.random.default_rng(seed)
    base = rs.random((reader.slots, 3), dtype=np.float32) + 0.1
    coarse = reader.gather(R.PT.fl(base), slot, q)
    checks = {}
    for kind, factor, idx in (("S", 4, slot * 4 + s), ("A", 4, slot * 4 + t), ("SA", 16, slot * 16 + s * 4 + t)):
        rep = np.repeat(base, factor, axis=0)
        fine = reader.gather(R.PT.fl(rep), idx, q)
        checks[f"{kind}_parent_broadcast_max_abs"] = max(float(dr.max(dr.abs(a - b))[0]) for a, b in zip(coarse, fine))
    compact = np.zeros((int(np.where(refined, 4, 1).sum()), 3), np.float32)
    for k in range(4):
        compact[offsets[refined] + k] = base[refined]
    compact[offsets[~refined]] = base[~refined]
    arms = DeviceArms(reader, (offsets, refined))
    ad_idx = arms.indices(slot, s, t)["AD"]
    adaptive = reader.gather(R.PT.fl(compact), ad_idx, q)
    checks["AD_parent_broadcast_max_abs"] = max(float(dr.max(dr.abs(a - b))[0]) for a, b in zip(coarse, adaptive))
    a, b, c, d = reader.coordinates(q.point, -q.ray.d, q.pid, q.bn)
    pid, bn = np.asarray(q.pid), np.asarray(q.bn)
    point = np.stack([np.asarray(q.point[k]) for k in range(3)], 1)
    na, nb = numpy_subcells(reader.geom, point, -np.stack([np.asarray(q.ray.d[k]) for k in range(3)], 1), pid, bn)
    s_np = (na >= .5).astype(np.int64) + 2 * (nb >= .5)
    s_gpu = np.asarray(s).astype(np.int64)
    tie = (np.abs(na - .5) < 1e-5) | (np.abs(nb - .5) < 1e-5)
    mismatch = s_np != s_gpu
    checks["subcell_queries"] = int(len(s_np))
    checks["subcell_mismatch_non_tie"] = int(np.count_nonzero(mismatch & ~tie))
    checks["subcell_mismatch_tie"] = int(np.count_nonzero(mismatch & tie))
    from tables101g import adaptive_index
    checks["adaptive_index_mismatch"] = int(np.count_nonzero(
        adaptive_index(offsets, refined, pid * 64 + bn, s_gpu) != np.asarray(ad_idx)))
    for key in ("S", "A", "SA", "AD"):
        assert checks[f"{key}_parent_broadcast_max_abs"] <= 1e-6, checks  # float32 kernel-fusion rounding only
    assert checks["subcell_mismatch_non_tie"] == 0 and checks["adaptive_index_mismatch"] == 0, checks
    return checks


def low_invariance(reader, learned, seed):
    R, dr = reader.R, reader.dr
    images = []
    for table in (reader.zero, R.PT.fl(learned)):
        ray, pix, rng, sd = reader.rays(1, seed, 0, res=64)
        out, _, _ = reader.prefix(ray, pix, rng, sd, table)
        images.append({k: np.asarray(out[f"{k}{c}"]) for k in ("e1", "e2") for c in range(3)})
    diff = max(float(np.max(np.abs(images[0][k] - images[1][k]))) for k in images[0])
    assert diff == 0.0, diff
    return {"low_max_abs_zero_vs_learned_table": diff}


def benchmark(reader, arms_dev, seed, tol, reps=30, warm=3):
    """Isolated synchronized lookup timing on one recorded launch."""
    R, dr = reader.R, reader.dr
    ray, pix, rng, sd = reader.rays(32, seed, 0)
    stamp = time.perf_counter()
    _, _, q = reader.prefix(ray, pix, rng, sd)
    R.sync()
    prefix_seconds = time.perf_counter() - stamp
    n = dr.width(q.pix)
    slot, s, t, _ = reader.addresses(q, tol)
    g_table, _, _ = arms_dev.arms["G_0"]
    s_table, _, _ = arms_dev.arms["S_G_0"]
    ad_table, _, _ = arms_dev.arms["S_Gad_0"]
    def coarse():
        return reader.gather(g_table, q.pid * 64 + q.bn, q)
    def spatial():
        a, b, _, _ = reader.coordinates(q.point, -q.ray.d, q.pid, q.bn)
        sub = R.U(a >= .5) + 2 * R.U(b >= .5)
        return reader.gather(s_table, (q.pid * 64 + q.bn) * 4 + sub, q)
    def adaptive():
        a, b, _, _ = reader.coordinates(q.point, -q.ray.d, q.pid, q.bn)
        sub = R.U(a >= .5) + 2 * R.U(b >= .5)
        offsets, refined = arms_dev.layout
        k = q.pid * 64 + q.bn
        idx = dr.gather(R.U, offsets, k) + dr.select(dr.gather(reader.mi.Bool, refined, k), sub, R.U(0))
        return reader.gather(ad_table, idx, q)
    result = {"queries": n, "prefix_launch_seconds_incl_compile": prefix_seconds}
    for label, fn in (("coarse", coarse), ("spatial_2x2", spatial), ("adaptive", adaptive)):
        times = []
        for k in range(reps + warm):
            R.sync()
            stamp = time.perf_counter()
            out = fn()
            dr.eval(out)
            R.sync()
            if k >= warm:
                times.append(time.perf_counter() - stamp)
        result[label] = {"median_ms_per_1e6": 1e3 * float(np.median(times)) * 1e6 / n,
                         "min_ms_per_1e6": 1e3 * float(np.min(times)) * 1e6 / n, "reps": reps}
    return result
