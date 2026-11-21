"""Native curve interactions and ECRO patch fields for the public Fabric scene."""

import json
import math
import sys
from pathlib import Path

import drjit as dr
import mitsuba as mi
import numpy as np

mi.set_variant("cuda_ad_rgb")
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
UPSTREAM = ROOT / "external/8dna26"
sys.path[:0] = [str(HERE.parent / "ecro86"), str(UPSTREAM)]
import pt86 as PT
import scene86 as S
from scenes.fabric import get_scene

P = json.loads((HERE / "protocol.json").read_text(encoding="utf-8"))
F, U = mi.Float, mi.UInt32
NB, K, KAPPA = S.NB, S.K, S.KAPPA


def load_scene(resolution, **camera):
    cfg = get_scene(resolution, **camera)
    cfg["integrator"] = {"type": "path", "max_depth": P["trace_bounces"] + 1, "rr_depth": 100}
    return mi.load_dict(cfg)


def vec(a, cls=mi.Vector3f):
    return cls(*(F(np.ascontiguousarray(x, dtype=np.float32)) for x in a.T))


def array(v):
    return np.stack([np.asarray(v[k]) for k in range(3)], axis=-1)


def spline_probes(scene):
    """Sample tube locations; use only verified native intersections for the pool."""
    par = mi.traverse(scene)
    controls = np.asarray(par["group0.fabric.control_points"]).reshape(-1, 4)
    indices = np.asarray(par["group0.fabric.segment_indices"], dtype=np.int64)
    rng = np.random.default_rng(P["geometry_seed"])
    pick = rng.choice(len(indices), size=min(len(indices), P["segment_probes"]), replace=False)
    cp = controls[indices[pick, None] + np.arange(4)]
    u = rng.random(len(pick))
    u2, u3 = u**2, u**3
    basis = np.stack([(1-u)**3, 4-6*u2+3*u3, 1+3*u+3*u2-3*u3, u3], axis=1) / 6
    deriv = np.stack([-3*(1-u)**2, -12*u+9*u2, 3+6*u-9*u2, 3*u2], axis=1) / 6
    pr = (cp*basis[..., None]).sum(axis=1)
    tangent = (cp[:, :, :3]*deriv[..., None]).sum(axis=1)
    jac = np.linalg.norm(tangent, axis=1)
    tangent /= np.maximum(jac[:, None], 1e-30)
    axis = np.eye(3)[np.argmin(np.abs(tangent), axis=1)]
    transverse = np.cross(tangent, axis)
    transverse /= np.maximum(np.linalg.norm(transverse, axis=1)[:, None], 1e-30)
    phi = rng.uniform(0, 2*np.pi, len(pr))
    radial = np.cos(phi[:, None])*transverse + np.sin(phi[:, None])*np.cross(tangent, transverse)
    offset = P["native_probe_offset_m"]
    origin = (pr[:, :3] + (pr[:, 3:]+offset)*radial).astype(np.float32)
    direction = (-radial).astype(np.float32)
    si = scene.ray_intersect(mi.Ray3f(vec(origin, mi.Point3f), vec(direction)))
    native_points = array(si.p)
    good = (np.asarray(si.is_valid()) & (np.asarray(si.prim_index) == pick)
            & (abs(np.asarray(si.t)-offset) <= P["native_probe_tolerance_m"])
            & (jac > 0))
    area = 2*np.pi * pr[:, 3] * jac * (len(indices) / len(pick))
    return origin[good], direction[good], native_points[good], area[good], {
        "native_segments": len(indices), "native_control_points": len(controls), "probes": len(pick),
        "accepted": int(good.sum()), "acceptance": float(good.mean()),
        "accepted_area_fraction": float(area[good].sum()/area.sum())}


def make_geometry(scene, *, cell=None):
    origin, direction, pts, area, audit = spline_probes(scene)
    if min(audit["acceptance"], audit["accepted_area_fraction"]) < P["min_native_surface_acceptance"]:
        raise ValueError(f"Insufficient native surface probe acceptance: {audit}")
    size = P["patch_cell_m"] if cell is None else cell
    if not np.isfinite(size) or size <= 0:
        raise ValueError("Grid cell size must be finite and positive")
    lo = np.floor(array(scene.bbox().min).reshape(-1, 3)[0] / size).astype(np.int32)
    hi = np.floor(array(scene.bbox().max).reshape(-1, 3)[0] / size).astype(np.int32)
    dims = hi - lo + 1
    xyz = np.floor(pts/size).astype(np.int32) - lo
    code = np.ravel_multi_index(xyz.T, dims)
    keys, inv = np.unique(code, return_inverse=True)
    inverse = np.full(int(np.prod(dims)), -1, dtype=np.int32)
    inverse[keys] = np.arange(len(keys))
    lattice = np.stack(np.unravel_index(np.arange(len(inverse)), dims), axis=1)
    distance = ((lattice[:, None] - lattice[keys])**2).sum(axis=2)
    nearest = np.argmin(distance, axis=1).astype(np.int32)
    rng = np.random.default_rng(P["geometry_seed"]+1)
    chosen, centers, areas = [], [], []
    order = np.argsort(inv, kind="stable")
    bounds = np.searchsorted(inv[order], np.arange(len(keys)+1))
    for l, r in zip(bounds[:-1], bounds[1:]):
        idx = order[l:r]
        w = area[idx]
        mass = w.sum()
        chosen.append(rng.choice(idx, size=P["point_pool_per_cell"], replace=True, p=w/mass))
        centers.append(np.average(pts[idx], axis=0, weights=w))
        areas.append(mass)
    pool = np.stack(chosen)
    n = len(keys)
    side = np.tile([1., -1.], n).astype(np.float32)
    pn = np.column_stack((np.zeros(2*n), np.zeros(2*n), side))
    pt = np.tile([1., 0., 0.], (2*n, 1))
    # A material descriptor for the geometry encoder; native BSDF parameters remain in Mitsuba.
    params = mi.traverse(scene)
    prefix = "group0.fabric.bsdf."
    knob = lambda key: float(np.asarray(params[prefix + key]).reshape(-1)[0])
    absorption = np.asarray(params[prefix+"sigma_a.value"]).reshape(3) * knob("scale")
    row = np.zeros((1, 69), dtype=np.float32)
    row[0, [0, 8, 16]] = absorption
    row[0, 24:31] = knob("longitudinal_roughness")
    row[0, 31:38] = knob("azimuthal_roughness")
    row[0, 53] = knob("scale_tilt")
    row[0, 68] = knob("eta")
    geometry = {
        "pc": np.repeat(np.asarray(centers), 2, axis=0), "pn": pn, "pt": pt,
        "parea": np.repeat(np.asarray(areas)/2, 2),
        "pmat": np.zeros(2*n, dtype=np.int64), "rows": row,
        "grid_min": lo, "grid_dim": dims, "cell_lookup": inverse, "nearest_cell": nearest, "key": keys,
        "origin": origin[pool], "direction": direction[pool],
    }
    if cell is not None:
        geometry["grid_cell_m"] = np.asarray(size, dtype=np.float64)
    return geometry, {**audit, "cells": n, "patches": 2*n, "area_m2_quadrature": float(sum(areas)),
        "min_candidates_per_cell": int(np.min(np.diff(bounds))), "encoder_absorption_rgb": absorption.tolist()}


def mis(pdf_a, pdf_b):
    a, b = dr.sqr(pdf_a), dr.sqr(pdf_b)
    return dr.select(a > 0., a/dr.maximum(a+b, 1e-38), 0.)


class Fabric:
    def __init__(self, scene, geometry=None):
        self.scene = scene
        self.env = scene.environment()
        self.bctx = mi.BSDFContext()
        self.par = mi.traverse(scene)
        self.env_base = mi.Transform4f(self.par["background.to_world"])
        if geometry is not None:
            g = geometry
            self.n = len(g["pc"])
            self.m = g["origin"].shape[1]
            self.cell = float(g.get("grid_cell_m",P["patch_cell_m"]))
            self.lo = g["grid_min"].tolist()
            self.dims = g["grid_dim"].tolist()
            self.lookup = mi.Int32(g["cell_lookup"])
            self.nearest = U(g["nearest_cell"])
            ray = mi.Ray3f(vec(g["origin"].reshape(-1, 3), mi.Point3f), vec(g["direction"].reshape(-1, 3)))
            self.pool = scene.ray_intersect(ray)
            dr.eval(self.pool)
            if not dr.all(self.pool.is_valid()):
                raise ValueError("Native surface pool does not replay")

    def light(self, yaw, tilt):
        rt = mi.Transform4f.rotate([0, 0, 1], yaw) @ mi.Transform4f.rotate([0, 1, 0], tilt) @ self.env_base
        dr.make_opaque(rt)
        self.par["background.to_world"] = rt
        self.par.update()

    def pool_at(self, pid, u):
        k = U(dr.minimum(dr.floor(u*self.m), self.m-1))
        return dr.gather(mi.SurfaceInteraction3f, self.pool, (pid//2)*self.m+k)

    @staticmethod
    def direction(pid, bin_id, u, v):
        z = (F(bin_id//S.NPHI)+u)/S.NZ
        phi = (F(bin_id % S.NPHI)+v)*(2*math.pi/S.NPHI)-math.pi
        s, c = dr.sincos(phi)
        r = dr.sqrt(dr.maximum(0., 1-z*z))
        sign = dr.select(dr.eq(pid % 2, 0), 1., -1.)
        return mi.Vector3f(r*c, r*s*sign, z*sign)

    def address(self, p, w):
        coord = [mi.Int32(dr.floor(p[k]/self.cell))-self.lo[k] for k in range(3)]
        inside = mi.Bool(True)
        for a, d in zip(coord, self.dims):
            inside &= (a >= 0) & (a < d)
        clamped = [U(dr.clamp(a, 0, d-1)) for a, d in zip(coord, self.dims)]
        key = (clamped[0]*self.dims[1]+clamped[1])*self.dims[2]+clamped[2]
        cell = dr.gather(mi.Int32, self.lookup, key)
        exact = inside & (cell >= 0)
        side_index = dr.select(w.z < 0, U(1), U(0))
        pid = dr.select(exact, U(dr.maximum(cell, 0)), dr.gather(U, self.nearest, key))*2 + side_index
        side = dr.select(w.z < 0, -1., 1.)
        bid = PT.Ctx.bin_of(w.x, side*w.y, side*w.z)
        return pid, bid, exact

    def state(self):
        rs = np.random.default_rng(S.P["representation"].get("table_seed", 86001))
        u = rs.random((4, 1, K), dtype=np.float32)
        samples = [F(np.broadcast_to(x, (self.n, K)).copy().ravel()) for x in u]
        lane = dr.arange(U, self.n*K)
        pid, bid = lane//K, (lane % K)//KAPPA
        si = self.pool_at(pid, samples[0])
        w = self.direction(pid, bid, samples[2], samples[3])
        dest = self.scene.ray_intersect(si.spawn_ray(w))
        p, b, exact = self.address(dest.p, -w)
        hit = dest.is_valid()
        valid = np.asarray(hit, dtype=bool)
        exact_fraction = float(np.asarray(exact)[valid].mean())
        return np.asarray(dr.select(hit, mi.Int32(p), -1)).reshape(self.n,K), np.asarray(b).reshape(self.n,K), {
            "state_hit_fraction": float(valid.mean()), "state_exact_hit_fraction": exact_fraction}

    def source(self, seed):
        m = P["source_samples_per_patch"]
        lane = dr.arange(U, self.n*m)
        patch = lane//m
        rng = mi.PCG32(size=self.n*m, initstate=dr.opaque(mi.UInt64, seed))
        si = self.pool_at(patch, rng.next_float32())
        ds, val = self.scene.sample_emitter_direction(si, mi.Point2f(rng.next_float32(), rng.next_float32()), True)
        side = dr.select(dr.eq(patch % 2, 0), 1., -1.)
        z = ds.d.z*side
        enabled = (ds.pdf > 0.) & (z > 0.)
        b = PT.Ctx.bin_of(ds.d.x, ds.d.y*side, z)
        val = dr.select(enabled, val*z/m, 0.)
        ans = PT.Ctx.accum({f"src{k}": val[k] for k in range(3)}, patch*NB+b, self.n*NB)
        return np.stack([ans[f"src{k}"] for k in range(3)], axis=-1).astype(np.float32).reshape(self.n,NB,3)

    def trace(self, si, rng, depth, camera=False, table=None):
        active = si.is_valid()
        thr = mi.Color3f(1.)
        out = {name: mi.Color3f(0.) for name in ("e1", "e2", "h")}
        if camera:
            out["e0"] = dr.select(~active, self.env.eval(si, ~active), 0.)
        for k in range(1, depth+1):
            if table is not None and k == 2:
                p, b, found = self.address(si.p, si.to_world(si.wi))
                i = (p*NB+b)*3
                B = mi.Color3f(*(dr.gather(F, table, i+j, active) for j in range(3)))
                out["hl"] = dr.select(active, thr*B, 0.)
                out["fallback"] = dr.select(active & ~found, F(1.), F(0.))
                out["lookup"] = dr.select(active, F(1.), F(0.))
            bsdf = si.bsdf()
            ds, ew = self.scene.sample_emitter_direction(si, mi.Point2f(rng.next_float32(), rng.next_float32()), True, active)
            lit = active & (ds.pdf > 0.)
            f, p_bs = bsdf.eval_pdf(self.bctx, si, si.to_local(ds.d), lit)
            direct = dr.select(lit, f*ew*mis(ds.pdf, p_bs), 0.)
            bs, w = bsdf.sample(self.bctx, si, rng.next_float32(), mi.Point2f(rng.next_float32(), rng.next_float32()), active)
            scatter = active & (bs.pdf > 0.) & (dr.max(w) > 0.)
            next_si = self.scene.ray_intersect(si.spawn_ray(si.to_world(bs.wo)), scatter)
            escape = scatter & ~next_si.is_valid()
            ds_next = mi.DirectionSample3f(self.scene, next_si, si)
            p_env = self.scene.pdf_emitter_direction(si, ds_next, escape)
            delta = mi.has_flag(bs.sampled_type, mi.BSDFFlags.Delta)
            escaped = dr.select(escape, w*self.env.eval(next_si, escape)*dr.select(delta, 1., mis(bs.pdf,p_env)), 0.)
            slot = "e1" if k == 1 else "e2" if k == 2 else "h"
            out[slot] = out[slot] + thr*(direct+escaped)
            thr *= dr.select(scatter, w, 0.)
            active = scatter & next_si.is_valid()
            si = next_si
            if k < depth:
                dr.eval(thr, active, si, rng.state, *out.values())
        return out

    def field(self, spp, seed):
        n = self.n*NB
        count = n*spp
        lane = dr.arange(U, count)
        # Strided per-bin addresses keep every cell/bin represented in each sample pass.
        pid, bid = (lane % n)//NB, lane % NB
        acc = {a: np.zeros((n,3), np.float64) for a in ("e2","h")}
        batch_spp = max(1, (1 << 17)//n)
        for i, start in enumerate(range(0, spp, batch_spp)):
            k = min(batch_spp, spp-start)*n
            idx = dr.arange(U,k)
            p, b = dr.gather(U,pid,idx), dr.gather(U,bid,idx)
            rng = mi.PCG32(size=k, initstate=dr.opaque(mi.UInt64, seed*1000003+i))
            si = self.pool_at(p, rng.next_float32())
            w = self.direction(p,b,rng.next_float32(),rng.next_float32())
            si.wi = si.to_local(w)
            fields = self.trace(si,rng,P["field_bounces"])
            values = PT.Ctx.accum({a+str(j): fields[a][j] for a in acc for j in range(3)}, idx % n, n)
            for a in acc:
                acc[a] += np.stack([values[a+str(j)] for j in range(3)],-1)
        return tuple((acc[x]/spp).astype(np.float32).reshape(self.n,NB,3) for x in ("e2","h"))

    def pixels(self, spp, seed, table=None):
        sensor = self.scene.sensors()[0]
        w, h = (int(x) for x in sensor.film().crop_size())
        size = w*h
        acc = {}
        launch = max(1, (1<<19)//size)
        for i, start in enumerate(range(0,spp,launch)):
            lanes = size*min(launch,spp-start)
            rng = mi.PCG32(size=lanes, initstate=dr.opaque(mi.UInt64,seed*1000003+i))
            pix = dr.arange(U,lanes) % size
            u = mi.Point2f((F(pix % w)+rng.next_float32())/w, (F(pix//w)+rng.next_float32())/h)
            ray, _ = sensor.sample_ray(0., 0.5, u, mi.Point2f(0.5,0.5))
            si = self.scene.ray_intersect(mi.Ray3f(ray))
            out = self.trace(si,rng,2 if table is not None else P["trace_bounces"], True,table)
            fields = {a+str(j): v[j] for a,v in out.items() if a not in ("lookup","fallback") for j in range(3)}
            fields.update({a:out[a] for a in ("lookup","fallback") if a in out})
            fields["fabric_mask"] = dr.select(si.is_valid(), F(1.), F(0.))
            values = PT.Ctx.accum(fields,pix,size)
            for a,v in values.items():
                acc[a] = acc.get(a,0.)+v
        names = ("e0","e1","e2","hl") if table is not None else ("e0","e1","e2","h")
        ans = {a: np.stack([acc[a+str(j)] for j in range(3)],-1).reshape(h,w,3)/spp for a in names}
        for a in ("lookup","fallback","fabric_mask"):
            if a in acc:
                ans[a] = acc[a].reshape(h,w)/spp
        return ans
