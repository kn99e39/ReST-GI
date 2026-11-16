"""WL97 audit, track E4: independent TAGGED field tracer for CK2 "none" (audit code; nothing here is production).

Definition reproduced (independently coded): table[patch, bin] = mean over valid probe samples of the radiance leaving a uniformly drawn
point of the patch into a uniformly drawn direction of the bin, made of exactly k scattering events (k = 2 .. KMAX), next-event estimation
at every vertex, BSDF sampling of the frozen ER closures, no Russian roulette.

Vertices are counted from the receiver: vertex 1 = receiver r, vertex 2 = s1 (adjacent to r), vertex j = s_(j-1).
A k-event contribution is the NEE made at vertex k.  Tags per contribution:
  cls1  material class of vertex 2 (s1): 0 floor(B, ER), 1 panel(S1, ER sharp), 2 sphere(S2, ER), 3 Lambert (walls / ceiling / boxes)
  ep    1 if ANY vertex j >= 3 (beyond the last sender, including the light-side NEE vertex) lies on the panel
Material lookup is by SHAPE id (original pt86 rule), NOT the mat94 per-patch patch; for the "none" variant both are the same map.

Usage: python e4_field.py <CODE/experiments> <out.npz> <seed> <spp> [kmax=6] [b=32]
Only scene / closure / ray primitives of pt86 are used (Ctx construction, patch_point, dir_in_bin, shape_id, DrClosure83).
"""
import json, os, sys, time
import numpy as np

CODE, OUT, SEED, SPP = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
KMAX = int(sys.argv[5]) if len(sys.argv) > 5 else 6
BL = int(sys.argv[6]) if len(sys.argv) > 6 else 32
sys.path.insert(0, os.path.join(CODE, 'ecro86'))
import pt86 as PT  # noqa: E402
mi, dr, S, CD = PT.mi, PT.dr, PT.S, PT.CD
F, U = PT.F, PT.U
NB = S.NB
NT = 8                                                    # tags per order: cls1 * 2 + ep

SC = json.load(open(os.path.join(CODE, 'ecro_local93d', 'addendum_checker', 'scene_ck2.json')))


def tagged_field(ctx, lt, spp, seed, kmax, b):
    scene, mats = ctx.scene, ctx.mats
    ns = ctx.N * NB
    acc = {k: np.zeros((ns * NT, 3)) for k in range(2, kmax + 1)}
    cnt = np.zeros(ns)
    done, launch = 0, 0
    while done < spp:
        bb = min(b, spp - done)
        N = ns * bb
        sd = dr.opaque(mi.UInt64, seed * 7919 + 13 * launch + 5)          # audit seeding rule (differs from the production rule)
        rng = mi.PCG32(size=N, initstate=sd, initseq=mi.UInt64(97))
        slot = dr.arange(U, N) % ns
        pid0 = slot // NB; bn0 = slot % NB
        p, nn, (t, bv, n) = ctx.patch_point(pid0, rng.next_float32(), rng.next_float32())
        wl0 = ctx.dir_in_bin(bn0, rng.next_float32(), rng.next_float32())
        w = t * wl0.x + bv * wl0.y + n * wl0.z
        active = dr.dot(w, nn) > 1e-3
        ray = mi.Ray3f(p + w * 2e-3, -w)
        thr = mi.Color3f(1.0)
        cls1 = U(0); ep = mi.Bool(False)
        vals, idxs = [], []
        okp = None
        for k in range(1, kmax + 1):
            si = scene.ray_intersect(ray, active)
            active &= si.is_valid()
            sid = ctx.shape_id(si)
            mid = dr.gather(U, ctx.SMAT, sid)
            wi = si.to_local(-ray.d)
            if k == 1:
                active &= dr.norm(si.p - p) < 1e-3
                okp = mi.Bool(active)
            active &= wi.z > 1e-6
            wi = dr.select(active, wi, mi.Vector3f(0, 0, 1))
            cl = CD.DrClosure83('ER', mats.states, mid)
            if k == 2:
                cls1 = dr.minimum(mid, U(3))
            if k >= 3:
                ep = ep | (active & dr.eq(mid, U(1)))
            if k >= 2:
                c = mi.Color3f(0.0)
                for l in range(lt.n):
                    dl = lt.pos[l] - si.p
                    d2 = dr.squared_norm(dl)
                    wl = dl * dr.rsqrt(d2)
                    wo_l = si.to_local(wl)
                    on = active & (wo_l.z > 1e-6) & (dr.dot(-wl, lt.axis[l]) > lt.cc[l])
                    vis = on & ~scene.ray_test(si.spawn_ray_to(lt.pos[l]), on)
                    wo_s = dr.select(vis, wo_l, mi.Vector3f(0, 0, 1))
                    f = cl.eval(wi, wo_s, vis)
                    c += dr.select(vis, thr * f * lt.rgb[l] * dr.rcp(d2), 0.0)
                tag = cls1 * 2 + dr.select(ep, U(1), U(0))
                vals.append(c); idxs.append(slot * NT + tag)
            if k == kmax:
                break
            u1 = rng.next_float32(); u2 = mi.Point2f(rng.next_float32(), rng.next_float32())
            wo, pdf, wgt, valid = cl.sample(wi, u1, u2, active)
            thr = thr * dr.select(valid, wgt, 0.0)
            active &= valid
            ray = si.spawn_ray(si.to_world(wo))
        okf = dr.select(okp, F(1.0), F(0.0))
        flat = [ch for c in vals for ch in (c.x, c.y, c.z)]
        dr.eval(flat, idxs, okf, slot); dr.sync_thread()                  # evaluate BEFORE any scatter (Dr.Jit 0.4.6 lazy-scatter defect)
        bufs = []
        for c, ix in zip(vals, idxs):
            for ch in (c.x, c.y, c.z):
                buf = dr.zeros(F, ns * NT)
                dr.scatter_reduce(dr.ReduceOp.Add, buf, ch, ix)
                bufs.append(buf)
        bc = dr.zeros(F, ns)
        dr.scatter_reduce(dr.ReduceOp.Add, bc, okf, slot)
        dr.eval(bufs, bc); dr.sync_thread()
        for j, k in enumerate(range(2, kmax + 1)):
            for ch in range(3):
                acc[k][:, ch] += np.asarray(bufs[3 * j + ch], np.float64)
        cnt += np.asarray(bc, np.float64)
        done += bb; launch += 1
    cn = np.maximum(cnt.reshape(ctx.N, NB), 1)
    out = {f'k{k}': (acc[k].reshape(ctx.N, NB, NT, 3) / cn[..., None, None]).astype(np.float32) for k in acc}
    out['cnt'] = cnt.reshape(ctx.N, NB).astype(np.float32)
    return out


if __name__ == '__main__':
    t0 = time.time()
    ctx = PT.Ctx(SC['geom'])
    lt = PT.Lights(SC['lights'])
    o = tagged_field(ctx, lt, SPP, SEED, KMAX, BL)
    np.savez_compressed(OUT, seed=SEED, spp=SPP, kmax=KMAX, PMAT=ctx.G.PMAT, PSID=ctx.G.PSID, PAREA=ctx.G.PAREA, **o)
    print('E4FIELD', OUT, 'seed', SEED, 'spp', SPP, 'kmax', KMAX, round(time.time() - t0, 1), 's', flush=True)
    os._exit(0)
