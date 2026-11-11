"""WL87 physical comparators built on the frozen WL86 transport module (pt86 is imported unchanged).

  render_rr      frozen-ER camera path tracer, <= D = 6 scattering events, next-event estimation at every vertex (as pt86), with
                 Russian roulette before tracing vertices >= RR_START (survival q = min(0.95, max_rgb throughput), throughput / q):
                 the expected value is the same truncated path family; only the variance and cost change
  field_orders   path-traced outgoing radiance per (patch, outgoing bin) split by scattering-event count 2..5 (probe of pt86.Ctx.field)
  table fill     pt86.Ctx.field at a given samples-per-slot budget (2..5 events): the deployable physical table (no learning)
"""
import math, time
import numpy as np

import fx87
PT = fx87.PT
mi, dr, S, CD = PT.mi, PT.dr, PT.S, PT.CD
F, U = PT.F, PT.U
NB = S.NB
D = PT.D


def trace_rr(ctx, ray, N, rng, lights, rr_start, kmax=D):
    scene, mats = ctx.scene, ctx.mats
    thr = mi.Color3f(1.0)
    active = mi.Bool(True)
    tot = mi.Color3f(0.0)
    for k in range(1, kmax + 1):
        si = scene.ray_intersect(ray, active)
        active &= si.is_valid()
        mid = dr.gather(U, ctx.SMAT, ctx.shape_id(si))
        wi = si.to_local(-ray.d)
        active &= wi.z > 1e-6
        wi = dr.select(active, wi, mi.Vector3f(0, 0, 1))
        cl = CD.DrClosure83('ER', mats.states, mid)
        c = mi.Color3f(0.0)
        for l in range(lights.n):
            dl = lights.pos[l] - si.p
            d2 = dr.squared_norm(dl)
            wl = dl * dr.rsqrt(d2)
            wo_l = si.to_local(wl)
            on = active & (wo_l.z > 1e-6) & (dr.dot(-wl, lights.axis[l]) > lights.cc[l])
            vis = on & ~scene.ray_test(si.spawn_ray_to(lights.pos[l]), on)
            f = cl.eval(wi, dr.select(vis, wo_l, mi.Vector3f(0, 0, 1)), vis)
            c += dr.select(vis, thr * f * lights.rgb[l] * dr.rcp(d2), 0.0)
        tot += c
        if k == kmax:
            break
        u1 = rng.next_float32(); u2 = mi.Point2f(rng.next_float32(), rng.next_float32())
        wo, pdf, w, valid = cl.sample(wi, u1, u2, active)
        thr = thr * dr.select(valid, w, 0.0)
        active &= valid
        if k + 1 >= rr_start:
            qv = dr.minimum(dr.max(thr), 0.95)
            surv = rng.next_float32() < qv
            active &= surv
            thr = dr.select(active, thr * dr.rcp(dr.maximum(qv, 1e-12)), 0.0)
        ray = si.spawn_ray(si.to_world(wo))
    return tot


def render_rr(ctx, lights, cam, res, spp, seed, rr_start, b=32):
    n = res * res
    acc = None
    T_ = {'trace_s': 0.0, 'eval_s': 0.0, 'gpu_ms': 0.0, 'launches': 0}
    t_all = time.perf_counter()
    done, launch = 0, 0
    while done < spp:
        bb = min(b, spp - done)
        t0 = time.perf_counter()
        dr.kernel_history()
        N = n * bb
        sd = dr.opaque(mi.UInt64, seed * 1000003 + launch)
        rng = mi.PCG32(size=N, initstate=sd, initseq=mi.UInt64(0))
        ray, pix = PT.gen_rays(cam, res, bb, rng)
        tot = trace_rr(ctx, ray, N, rng, lights, rr_start)
        t1 = time.perf_counter()
        a = ctx.accum({'tot0': tot.x, 'tot1': tot.y, 'tot2': tot.z}, pix, n)
        T_['gpu_ms'] += sum(float(h.get('execution_time', 0.0)) for h in dr.kernel_history())
        acc = a if acc is None else {k: acc[k] + v for k, v in a.items()}
        T_['trace_s'] += t1 - t0; T_['eval_s'] += time.perf_counter() - t1; T_['launches'] += 1
        done += bb; launch += 1
    T_['wall_s'] = time.perf_counter() - t_all; T_['spp'] = spp
    return acc, T_


def field_orders(ctx, lights, spp, seed, b=128):
    """[4, N, NB, 3] float64: outgoing radiance of exactly 2, 3, 4, 5 scattering events per (patch, bin) (probe estimator of pt86)."""
    scene, mats = ctx.scene, ctx.mats
    ns = ctx.N * NB
    acc = None
    done, launch = 0, 0
    while done < spp:
        bb = min(b, spp - done)
        N = ns * bb
        sd = dr.opaque(mi.UInt64, seed * 1000003 + launch)
        rng = mi.PCG32(size=N, initstate=sd, initseq=mi.UInt64(0))
        slot = dr.arange(U, N) % ns
        pid = slot // NB; bn = slot % NB
        p, nn, (t, bv, n) = ctx.patch_point(pid, rng.next_float32(), rng.next_float32())
        wl0 = ctx.dir_in_bin(bn, rng.next_float32(), rng.next_float32())
        w = t * wl0.x + bv * wl0.y + n * wl0.z
        active = dr.dot(w, nn) > 1e-3
        ray = mi.Ray3f(p + w * 2e-3, -w)
        thr = mi.Color3f(1.0)
        out = {}
        okp = None
        for k in range(1, D):
            si = scene.ray_intersect(ray, active)
            active &= si.is_valid()
            mid = dr.gather(U, ctx.SMAT, ctx.shape_id(si))
            wi = si.to_local(-ray.d)
            if k == 1:
                active &= dr.norm(si.p - p) < 1e-3
                okp = mi.Bool(active)
            active &= wi.z > 1e-6
            wi = dr.select(active, wi, mi.Vector3f(0, 0, 1))
            cl = CD.DrClosure83('ER', mats.states, mid)
            if k >= 2:
                c = mi.Color3f(0.0)
                for l in range(lights.n):
                    dl = lights.pos[l] - si.p
                    d2 = dr.squared_norm(dl)
                    wl = dl * dr.rsqrt(d2)
                    wo_l = si.to_local(wl)
                    on = active & (wo_l.z > 1e-6) & (dr.dot(-wl, lights.axis[l]) > lights.cc[l])
                    vis = on & ~scene.ray_test(si.spawn_ray_to(lights.pos[l]), on)
                    f = cl.eval(wi, dr.select(vis, wo_l, mi.Vector3f(0, 0, 1)), vis)
                    c += dr.select(vis, thr * f * lights.rgb[l] * dr.rcp(d2), 0.0)
                for ch in range(3):
                    out[f'o{k}_{ch}'] = c[ch]
            if k == D - 1:
                break
            u1 = rng.next_float32(); u2 = mi.Point2f(rng.next_float32(), rng.next_float32())
            wo, pdf, wgt, valid = cl.sample(wi, u1, u2, active)
            thr = thr * dr.select(valid, wgt, 0.0)
            active &= valid
            ray = si.spawn_ray(si.to_world(wo))
        out['cnt'] = dr.select(okp, F(1.0), F(0.0))
        a = ctx.accum(out, slot, ns)
        acc = a if acc is None else {k_: acc[k_] + v for k_, v in a.items()}
        done += bb; launch += 1
    cnt = np.maximum(acc['cnt'].reshape(ctx.N, NB), 1)[..., None]
    return np.stack([np.stack([acc[f'o{k}_{c}'] for c in range(3)], -1).reshape(ctx.N, NB, 3) / cnt for k in range(2, D)])


def table_fill(ctx, lights, spp, seed):
    """Deployable physical table: path-traced 2..5-event field at `spp` samples per (patch, bin) slot (pt86.Ctx.field, one pass)."""
    f, cnt = ctx.field(lights, spp, seed)
    return f.astype(np.float64)
