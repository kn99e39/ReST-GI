"""WL88 physical references on the frozen WL86 transport module (pt86 imported unchanged).

render_fam   the pt86 'cam' ER estimator (identical random-number use; e1 / e2 / h reproduce pt86 bit-for-bit for the same seed) plus
             path-family channels of h (camera vertices x1..xk, next-event contribution at xk = k events, k >= 3), classified by the
             frozen-ER material of x2..xk (mutually exclusive, sum = h):
               Hb  no S1 and no S2 vertex          Hm  S2 vertex, no S1
               H1P exactly one S1 vertex, the panel H1G exactly one S1 vertex, on the glossy mover G
               H2  >= 2 S1 vertices (sharp-to-sharp)
             plus H2G (subset of H2 with a G vertex; not exclusive). WL86 family table: G does not exist (H1G = H2G = 0).
field_orders the probe estimator of pt86.Ctx.field split by event count 1..5 per (patch, outgoing bin) (order 1 = the once-scattered
             table that q estimates; orders 2..5 = the per-order truth of the table recurrence)
"""
import time
import numpy as np

import fx88
PT = fx88.PT
mi, dr, S, CD = PT.mi, PT.dr, PT.S, PT.CD
F, U = PT.F, PT.U
NB = S.NB
D = PT.D
FAM = ['Hb', 'Hm', 'H1P', 'H1G', 'H2', 'H2G']


def trace_fam(ctx, ray, N, rng, lights):
    scene, mats = ctx.scene, ctx.mats
    thr = mi.Color3f(1.0)
    active = mi.Bool(True)
    out = {}
    ns1 = U(0); ns2 = mi.Bool(False); ng = mi.Bool(False)

    def add(name, c, sel=None):
        for i, ch in enumerate((c.x, c.y, c.z)):
            v = ch if sel is None else dr.select(sel, ch, 0.0)
            key = f'{name}{i}'
            out[key] = out[key] + v if key in out else v
    for k in range(1, D + 1):
        si = scene.ray_intersect(ray, active)
        active &= si.is_valid()
        sid = ctx.shape_id(si)
        mid = dr.gather(U, ctx.SMAT, sid)
        wi = si.to_local(-ray.d)
        active &= wi.z > 1e-6
        wi = dr.select(active, wi, mi.Vector3f(0, 0, 1))
        if k >= 2:
            isg = dr.eq(mid, 4) if fx88.FAMILY['name'] == 'wl88' else mi.Bool(False)
            ns1 = dr.select(active & (dr.eq(mid, 1) | isg), ns1 + 1, ns1)
            ns2 |= active & dr.eq(mid, 2)
            ng |= active & isg
        cl = CD.DrClosure83('ER', mats.states, mid)
        c = mi.Color3f(0.0)
        for l in range(lights.n):
            dl = lights.pos[l] - si.p
            d2 = dr.squared_norm(dl)
            wl = dl * dr.rsqrt(d2)
            wo_l = si.to_local(wl)
            on = active & (wo_l.z > 1e-6) & (dr.dot(-wl, lights.axis[l]) > lights.cc[l])
            occ = scene.ray_test(si.spawn_ray_to(lights.pos[l]), on)
            vis = on & ~occ
            wo_s = dr.select(vis, wo_l, mi.Vector3f(0, 0, 1))
            f = cl.eval(wi, wo_s, vis)
            c += dr.select(vis, thr * f * lights.rgb[l] * dr.rcp(d2), 0.0)
        if k == 1:
            add('e1', c)
        elif k == 2:
            add('e2', c)
        else:
            add('h', c)
            z1 = dr.eq(ns1, 0)
            add('Hb', c, z1 & ~ns2); add('Hm', c, z1 & ns2)
            add('H1P', c, dr.eq(ns1, 1) & ~ng); add('H1G', c, dr.eq(ns1, 1) & ng)
            add('H2', c, ns1 >= 2); add('H2G', c, (ns1 >= 2) & ng)
        if k == D:
            break
        u1 = rng.next_float32(); u2 = mi.Point2f(rng.next_float32(), rng.next_float32())
        wo, pdf, w, valid = cl.sample(wi, u1, u2, active)
        thr = thr * dr.select(valid, w, 0.0)
        active &= valid
        ray = si.spawn_ray(si.to_world(wo))
    return out


def render_fam(ctx, lights, cam, res, spp, seed, b=32):
    n = res * res
    acc = {}
    T_ = {'gpu_ms': 0.0, 'launches': 0}
    t_all = time.perf_counter()
    done, launch = 0, 0
    while done < spp:
        bb = min(b, spp - done)
        dr.kernel_history()
        N = n * bb
        sd = dr.opaque(mi.UInt64, seed * 1000003 + launch)
        rng = mi.PCG32(size=N, initstate=sd, initseq=mi.UInt64(0))
        ray, pix = PT.gen_rays(cam, res, bb, rng)
        out = trace_fam(ctx, ray, N, rng, lights)
        a = ctx.accum(out, pix, n)
        T_['gpu_ms'] += sum(float(h.get('execution_time', 0.0)) for h in dr.kernel_history())
        for k_, v in a.items():
            acc[k_] = acc[k_] + v if k_ in acc else v
        done += bb; launch += 1; T_['launches'] += 1
    T_['wall_s'] = time.perf_counter() - t_all; T_['spp'] = spp
    g = lambda k: (acc[k] / spp).reshape(res, res).astype(np.float32)
    o = {name: np.stack([g(f'{name}{c}') for c in range(3)], -1) for name in ['e1', 'e2', 'h'] + FAM}
    o['tot'] = o['e1'] + o['e2'] + o['h']
    return o, T_


def field_orders(ctx, lights, spp, seed, b=128):
    """[5, N, NB, 3] float64: outgoing radiance of exactly 1..5 scattering events per (patch, bin) (pt86 probe estimator)."""
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
    return np.stack([np.stack([acc[f'o{k}_{c}'] for c in range(3)], -1).reshape(ctx.N, NB, 3) / cnt for k in range(1, D)])
