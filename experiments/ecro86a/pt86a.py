"""WL86A diagnostic sources for the frozen WL86 transport module (pt86 is imported unchanged; nothing here is a runtime input of ECRO).

Field probes (radiance leaving a surface point made of 2..5 scattering events, exactly pt86.Ctx.field) under controlled incident fields:
  'true'            the lights of the workload (next-event estimation, as pt86)
  'ang' on a set    same irradiance, visibility and position as the true light, but the incident DIRECTION is redrawn uniformly inside the
                    incident-direction bin it falls in (the angular information the 64-bin encoding keeps) - on the patches of the set only
  'enc' on a set    the ENCODED source re-emitted: every point of a patch receives the patch-average irradiance of each bin from a direction
                    uniform in that bin (what the encoding s keeps, spatially and angularly). This incident field has the same encoding s.
  beams             a collimated beam restricted to one patch (optionally one half of it): direction w, perpendicular irradiance E.
                    Two beams whose directions lie in the same bin and whose E cos(theta) are equal have IDENTICAL encodings by construction.
"""
import math, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'ecro86'))
import pt86 as PT  # noqa: E402
mi, dr, S, CD = PT.mi, PT.dr, PT.S, PT.CD
F, U = PT.F, PT.U
NB = PT.NB
TOPK = 16
BEAM_REACH = 0.25          # m: a beam is emitted by a virtual projector this far from its patch (occlusion is tested inside that distance)


def source_hq(ctx, lights, pts=1024, seed=861):
    """Patch-average direct irradiance by incident bin with `pts` random points per patch (reference for the 16-point estimator)."""
    Np = ctx.N * pts
    lane = dr.arange(U, Np)
    pid = lane // pts
    rng = mi.PCG32(size=Np, initstate=mi.UInt64(seed), initseq=mi.UInt64(0))
    p, nn, (t, bv, n) = ctx.patch_point(pid, rng.next_float32(), rng.next_float32())
    s = np.zeros((ctx.N * NB, 3))
    for l in range(lights.n):
        dl = lights.pos[l] - p
        d2 = dr.squared_norm(dl); dist = dr.sqrt(d2)
        wl = dl * dr.rcp(dist)
        cs = dr.dot(wl, nn)
        on = (cs > 1e-6) & (dr.dot(-wl, lights.axis[l]) > lights.cc[l])
        r = mi.Ray3f(p + nn * 1e-4, wl); r.maxt = dist * (1.0 - 1e-4)
        e = dr.select(on & ~ctx.scene.ray_test(r, on), cs * dr.rcp(d2), 0.0)
        slot = pid * NB + ctx.bin_of(dr.dot(wl, t), dr.dot(wl, bv), dr.dot(wl, n))
        ch = [e * lights.rgb[l].x, e * lights.rgb[l].y, e * lights.rgb[l].z]
        dr.eval(ch, slot); dr.sync_thread()
        sl = np.asarray(slot)
        for c in range(3):
            s[:, c] += np.bincount(sl, weights=np.asarray(ch[c], np.float64), minlength=ctx.N * NB)
    return (s / pts).reshape(ctx.N, NB, 3).astype(np.float32)


def topk_tables(s):
    """Per patch the TOPK incident bins by irradiance: bin ids [N, TOPK], rgb irradiance [N, TOPK, 3], cumulative selection probabilities
    [N, TOPK] (proportional to the bin irradiance), covered fraction of the encoding."""
    lum = s.sum(-1).astype(np.float64)
    idx = np.argsort(-lum, 1)[:, :TOPK]
    val = np.take_along_axis(s, idx[..., None], 1)
    l = np.take_along_axis(lum, idx, 1)
    tot = np.maximum(l.sum(1, keepdims=True), 1e-30)
    cum = np.cumsum(l / tot, 1); cum[:, -1] = 1.0
    prob = l / tot
    return idx.astype(np.uint32), val.astype(np.float32), cum.astype(np.float32), prob.astype(np.float32), float(val.sum() / max(s.sum(), 1e-30))


class Src:
    """Incident-field specification for field_src."""

    def __init__(self, lights=None, mode='true', mats=(), enc=None, beams=()):
        self.lights, self.mode, self.mats, self.beams = lights, mode, tuple(mats), list(beams)
        if enc is not None:
            idx, val, cum, prob, self.enc_cover = topk_tables(enc)
            self.tb = U(np.ascontiguousarray(idx.ravel())); self.te = PT.fl(val); self.tc = PT.fl(cum); self.tp = PT.fl(prob)


def field_src(ctx, src, spp, seed, kmax=PT.D - 1, b=128, split=False):
    """As pt86.Ctx.field, with the incident field of `src`. Returns [N, NB, 3]; split=True returns a dict of four tables that sum to it:
    e2_p / e2_o (exactly 2 scattering events, last sender on the panel / elsewhere) and e3_p / e3_o (3..kmax events)."""
    scene, mats = ctx.scene, ctx.mats
    ns = ctx.N * NB
    acc = None
    pmat = U(np.ascontiguousarray(ctx.G.PMAT.astype(np.uint32)))
    setm = np.zeros(8, np.float32)
    for m in src.mats:
        setm[m] = 1.0
    setm = PT.fl(setm)
    done, launch = 0, 0
    while done < spp:
        bb = min(b, spp - done)
        N = ns * bb
        sd = dr.opaque(mi.UInt64, seed * 1000003 + launch)
        rng = mi.PCG32(size=N, initstate=sd, initseq=mi.UInt64(0))
        slot = dr.arange(U, N) % ns
        pid0 = slot // NB; bn0 = slot % NB
        p, nn, (t, bv, n) = ctx.patch_point(pid0, rng.next_float32(), rng.next_float32())
        wl0 = ctx.dir_in_bin(bn0, rng.next_float32(), rng.next_float32())
        w = t * wl0.x + bv * wl0.y + n * wl0.z
        active = dr.dot(w, nn) > 1e-3
        ray = mi.Ray3f(p + w * 2e-3, -w)
        thr = mi.Color3f(1.0)
        tot = mi.Color3f(0.0)
        parts = [mi.Color3f(0.0) for _ in range(4)]
        v2p = mi.Bool(False)
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
            if k >= 2:
                pid, _ = ctx.patch_bin(si, sid, -ray.d)
                pt_, pb_, pn_ = ctx.patch_frame(pid)
                inset = dr.gather(F, setm, dr.gather(U, pmat, pid)) > 0.5
                c = mi.Color3f(0.0)
                if src.lights is not None:
                    lt = src.lights
                    for l in range(lt.n):
                        dl = lt.pos[l] - si.p
                        d2 = dr.squared_norm(dl)
                        wl = dl * dr.rsqrt(d2)
                        wo_l = si.to_local(wl)
                        on = active & (wo_l.z > 1e-6) & (dr.dot(-wl, lt.axis[l]) > lt.cc[l])
                        if src.mode == 'enc':
                            on &= ~inset                                   # the set's patches are lit by the encoded field below
                        vis = on & ~scene.ray_test(si.spawn_ray_to(lt.pos[l]), on)
                        wo_s = dr.select(vis, wo_l, mi.Vector3f(0, 0, 1))
                        f = cl.eval(wi, wo_s, vis)
                        if src.mode == 'ang':
                            # same irradiance at this point, direction redrawn uniformly in the incident bin of the true direction (patch frame)
                            x, y, z = dr.dot(wl, pt_), dr.dot(wl, pb_), dr.dot(wl, pn_)
                            bn = ctx.bin_of(x, y, z)
                            wr = ctx.dir_in_bin(bn, rng.next_float32(), rng.next_float32())
                            wr_l = si.to_local(pt_ * wr.x + pb_ * wr.y + pn_ * wr.z)
                            use = vis & inset
                            fr = cl.eval(wi, dr.select(use, wr_l, mi.Vector3f(0, 0, 1)), use & (wr_l.z > 1e-6))
                            zb = (F(bn // S.NPHI) + 0.5) * (1.0 / S.NZ)
                            f = dr.select(use, fr * (dr.maximum(z, 0.0) * dr.rcp(zb)), f)
                        c += dr.select(vis, thr * f * lt.rgb[l] * dr.rcp(d2), 0.0)
                if src.mode == 'enc':
                    # one bin drawn in proportion to its irradiance (unbiased), direction uniform in the bin
                    uq = rng.next_float32()
                    q = U(0)
                    for j in range(TOPK - 1):
                        q += dr.select(uq >= dr.gather(F, src.tc, pid * TOPK + j), U(1), U(0))
                    sl = pid * TOPK + q
                    bn = dr.gather(U, src.tb, sl)
                    E = mi.Color3f(dr.gather(F, src.te, sl * 3), dr.gather(F, src.te, sl * 3 + 1), dr.gather(F, src.te, sl * 3 + 2))
                    pq = dr.gather(F, src.tp, sl)
                    wr = ctx.dir_in_bin(bn, rng.next_float32(), rng.next_float32())
                    wr_l = si.to_local(pt_ * wr.x + pb_ * wr.y + pn_ * wr.z)
                    use = active & inset & (wr_l.z > 1e-6) & (pq > 0)
                    fr = cl.eval(wi, dr.select(use, wr_l, mi.Vector3f(0, 0, 1)), use)
                    zb = (F(bn // S.NPHI) + 0.5) * (1.0 / S.NZ)
                    c += dr.select(use, thr * fr * E * dr.rcp(zb * dr.maximum(pq, 1e-30)), 0.0)
                for bm in src.beams:                                        # collimated beam on one patch
                    wb = mi.Vector3f(*[float(v) for v in bm['w']])
                    use = active & dr.eq(pid, U(int(bm['pid'])))
                    if bm.get('half') is not None:
                        g = lambda kk: dr.gather(F, ctx.PTB, pid * 21 + kk)
                        rel = si.p - mi.Point3f(g(0), g(1), g(2))
                        e1 = mi.Vector3f(g(3), g(4), g(5))
                        u = dr.dot(rel, e1) * dr.rcp(dr.squared_norm(e1))
                        use &= (u < 0.5) if bm['half'] == 0 else (u >= 0.5)
                    wo_l = si.to_local(wb)
                    use &= wo_l.z > 1e-6
                    rb = si.spawn_ray(wb); rb.maxt = BEAM_REACH
                    use &= ~scene.ray_test(rb, use)
                    f = cl.eval(wi, dr.select(use, wo_l, mi.Vector3f(0, 0, 1)), use)
                    c += dr.select(use, thr * f * float(bm['E']), 0.0)
                tot += c
                if split:
                    if k == 2:
                        v2p = dr.eq(mid, U(1))
                    for q in range(4):
                        sel = (v2p if q % 2 == 0 else ~v2p) if ((q < 2) == (k == 2)) else mi.Bool(False)
                        parts[q] += dr.select(sel, c, 0.0)
            if k == kmax:
                break
            u1 = rng.next_float32(); u2 = mi.Point2f(rng.next_float32(), rng.next_float32())
            wo, pdf, wgt, valid = cl.sample(wi, u1, u2, active)
            thr = thr * dr.select(valid, wgt, 0.0)
            active &= valid
            ray = si.spawn_ray(si.to_world(wo))
        out = {'f0': tot.x, 'f1': tot.y, 'f2': tot.z, 'cnt': dr.select(okp, F(1.0), F(0.0))}
        if split:
            for q, nm in enumerate(('e2_p', 'e2_o', 'e3_p', 'e3_o')):
                out[f'{nm}0'] = parts[q].x; out[f'{nm}1'] = parts[q].y; out[f'{nm}2'] = parts[q].z
        a = ctx.accum(out, slot, ns)
        acc = a if acc is None else {k_: acc[k_] + v for k_, v in a.items()}
        done += bb; launch += 1
    cnt = acc['cnt'].reshape(ctx.N, NB)
    f = np.stack([acc[f'f{c}'] for c in range(3)], -1).reshape(ctx.N, NB, 3) / np.maximum(cnt, 1)[..., None]
    if split:
        return {nm: (np.stack([acc[f'{nm}{c}'] for c in range(3)], -1).reshape(ctx.N, NB, 3) / np.maximum(cnt, 1)[..., None]).astype(np.float32)
                for nm in ('e2_p', 'e2_o', 'e3_p', 'e3_o')}
    return f.astype(np.float32)


def beam_encoding(ctx, bm):
    """Exact encoding of a beam (patch-average irradiance in the bin of its direction); `visible` = fraction of 256 patch points it reaches."""
    G = ctx.G
    j = int(bm['pid'])
    w = np.array(bm['w'], np.float64)
    loc = np.array([w @ G.PT[j], w @ G.PB[j], w @ G.PN[j]])
    iz = int(np.clip(math.floor(loc[2] * S.NZ), 0, S.NZ - 1)); ip = int(np.clip(math.floor((math.atan2(loc[1], loc[0]) + math.pi) * S.NPHI / (2 * math.pi)), 0, S.NPHI - 1))
    frac = 1.0 if bm.get('half') is None else 0.5
    s = np.zeros((ctx.N, NB, 3), np.float32)
    s[j, iz * S.NPHI + ip] = bm['E'] * loc[2] * frac
    return s, iz * S.NPHI + ip, float(loc[2])
