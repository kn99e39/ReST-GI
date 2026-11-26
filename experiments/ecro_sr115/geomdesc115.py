"""WL115 geometry-only descriptors at any sub-cell resolution (no camera, light, radiance or reference).

vhs_k(ctx, Kk, pids, ell): per (patch in pids, Kk x Kk cell): cos-weighted over the 64 bin-centre directions of the patch frame,
  [0] near fraction (hit closer than the patch size L_p = sqrt(area), tracer113.vhs definition),
  [1] mean log10 hit distance over hitting rays,
  [2] escape fraction,
  [3] proximity m = sum cos exp(-d / ell) / sum cos (continuous in the geometry; escape -> 0), ell in metres.
The 4 x 4 version of [0..2] equals tracer113.Ctx113.vhs for the same patches (focused test).
"""
import math
import numpy as np
import tracer115 as T

mi, dr, F, U, S = T.mi, T.dr, T.F, T.U, T.S
NB = T.NB


def vhs_k(ctx, Kk, pids=None, ell=0.0625, chunk=2_000_000):
    pids = np.arange(ctx.N) if pids is None else np.asarray(pids, np.int64)
    P_ = len(pids); NSK = Kk * Kk
    n = P_ * NSK * NB
    PIDS = U(np.ascontiguousarray(pids.astype(np.uint32)))
    PA = F(np.ascontiguousarray(ctx.G.PAREA.astype(np.float32)))
    acc = {k: np.zeros(P_ * NSK) for k in ('w', 'near', 'logd', 'esc', 'prox')}
    for s0 in range(0, n, chunk):
        m = min(chunk, n - s0)
        lane = dr.arange(U, m) + s0
        bn = lane % NB; ps = lane // NB; sub = ps % NSK; pl = ps // NSK
        pid = dr.gather(U, PIDS, pl)
        a = (F(sub % Kk) + 0.5) * (1.0 / Kk); b = (F(sub // Kk) + 0.5) * (1.0 / Kk)
        t, bv, nf = ctx.patch_frame(pid)
        loc = ctx.dir_in_bin(bn, F(0.5), F(0.5))
        w = t * loc.x + bv * loc.y + nf * loc.z
        si, sid, nn = ctx.first_si(pid, a, b, w)
        cosl = dr.dot(w, nn)
        ok = cosl > 1e-3
        ray, sing = ctx.spawn(si, w)
        hit = ctx.scene.ray_intersect(ray, ok & ~sing)
        L = dr.sqrt(dr.gather(F, PA, pid))
        dist = dr.select(hit.is_valid(), hit.t, F(1e3))
        wt = dr.select(ok, cosl, 0.)
        vals = {'w': wt, 'near': dr.select(ok & hit.is_valid() & (hit.t < L), cosl, 0.),
                'logd': dr.select(ok & hit.is_valid(), cosl * dr.log(dr.maximum(dist, 1e-6)) * (1 / math.log(10)), 0.),
                'esc': dr.select(ok & ~hit.is_valid(), cosl, 0.),
                'prox': dr.select(ok & hit.is_valid(), cosl * dr.exp(-dist * (1.0 / ell)), 0.)}
        a_ = ctx.accum(vals, ps, P_ * NSK)
        for k in acc:
            acc[k] += a_[k]
    w_ = np.maximum(acc['w'], 1e-12)
    desc = np.stack([acc['near'] / w_, acc['logd'] / np.maximum(acc['w'] - acc['esc'], 1e-12), acc['esc'] / w_, acc['prox'] / w_], -1)
    return desc.reshape(P_, NSK, 4).astype(np.float32)


def object_of_sid(sid):
    """Rigid object group of a shape id: 0 room (floor, ceiling, walls, panel: shape ids 0-5), 1 sphere (6), 2 box A (7), 3 box B (8)."""
    return dr.select(sid < 6, U(0), sid - 5)


def prox_other(ctx, Kk, pids=None, ells=(0.0625, 0.25), chunk=2_000_000):
    """Per (patch, Kk x Kk cell): cos-weighted proximity to a DIFFERENT rigid object, m_ell = sum cos exp(-d / ell) [hit on another
    object] / sum cos over the 64 bin-centre directions (geometry only; continuous in the geometry). Returns [P, Kk^2, len(ells)]."""
    pids = np.arange(ctx.N) if pids is None else np.asarray(pids, np.int64)
    P_ = len(pids); NSK = Kk * Kk
    n = P_ * NSK * NB
    PIDS = U(np.ascontiguousarray(pids.astype(np.uint32)))
    acc = {k: np.zeros(P_ * NSK) for k in ['w'] + [f'm{i}' for i in range(len(ells))]}
    for s0 in range(0, n, chunk):
        m = min(chunk, n - s0)
        lane = dr.arange(U, m) + s0
        bn = lane % NB; ps = lane // NB; sub = ps % NSK; pl = ps // NSK
        pid = dr.gather(U, PIDS, pl)
        a = (F(sub % Kk) + 0.5) * (1.0 / Kk); b = (F(sub // Kk) + 0.5) * (1.0 / Kk)
        t, bv, nf = ctx.patch_frame(pid)
        loc = ctx.dir_in_bin(bn, F(0.5), F(0.5))
        w = t * loc.x + bv * loc.y + nf * loc.z
        si, sid, nn = ctx.first_si(pid, a, b, w)
        cosl = dr.dot(w, nn)
        ok = cosl > 1e-3
        ray, sing = ctx.spawn(si, w)
        hit = ctx.scene.ray_intersect(ray, ok & ~sing)
        other = hit.is_valid() & dr.neq(object_of_sid(ctx.shape_id(hit)), object_of_sid(sid))
        vals = {'w': dr.select(ok, cosl, 0.)}
        for i, ell in enumerate(ells):
            vals[f'm{i}'] = dr.select(ok & other, cosl * dr.exp(-hit.t * (1.0 / ell)), 0.)
        a_ = ctx.accum(vals, ps, P_ * NSK)
        for k in acc:
            acc[k] += a_[k]
    w_ = np.maximum(acc['w'], 1e-12)
    return np.stack([acc[f'm{i}'] / w_ for i in range(len(ells))], -1).reshape(P_, NSK, len(ells)).astype(np.float32)
