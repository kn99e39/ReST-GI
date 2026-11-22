"""WL109 C1: prefix-conditioned inner Monte Carlo readout of the frozen ECRO HIGH table.

Frozen estimator (pt100.Ctx.trace, mode 'ecro', kmax 2, read from the code):
    J  pixel jitter (2 uniforms, PT.gen_rays)
    P  first-event continuation: k = 1 SSS entry->first-exit walk (variable count of uniforms, sphere only) and the k = 1
       closure sample (3 uniforms, always drawn); fixes the second hit, its table address a = (patch, bin) and throughput T
    S  k = 2 SSS walk (sphere second hits only); feeds E2 only
    LOW  = E1 + E2,  HIGH = T * B[a] read at the second hit before the k = 2 walk,  TOTAL = LOW + HIGH

C1 keeps every LOW statement and the main PCG32 stream (J, P, S) unchanged and replaces HIGH by
    H_C1(U) = (1/M) sum_m T(U, P'_m) B[a(U, P'_m)]
where U is the first camera interaction (camera ray, first hit, closure, SSS flag; a function of J) and P'_m are M fresh
first-event continuations drawn from an independent TEA-seeded PCG32 stream with exactly the original statements and
distributions (the SSS walk without its NEE, which consumes no random numbers, followed by the closure sample).
E[H_C1 | U] = E[H | U], hence E[TOTAL_C1] = E[TOTAL_R0]. H_C1 is a finite unbiased inner Monte Carlo estimate of the
conditional expectation, not an exact integral.

Test-only switches (never an evaluation arm):
    replay       the inner stream is a copy of the main stream taken before the k = 1 walk: with M = 1 the inner path must
                 equal the original HIGH path (address, throughput, ownership)
    fixed_jitter pixel-centre camera rays (J removed): inner-stream HIGH must then be uncorrelated with every main-stream output
    orig_hl      also emit the original shared-sample HIGH as channel 'ho' (instrumentation; not part of C1's estimate)
"""
import math

KEYS_R0 = ['e10', 'e11', 'e12', 'e20', 'e21', 'e22', 'hl0', 'hl1', 'hl2']
SALT_STATE = 0x6A09E667
SALT_SEQ = 0xBB67AE85
INNER_SEED_MUL = 0x9E3779B97F4A7C15


def inner_seed_value(seed):
    """64-bit inner-stream seed of evidence seed `seed` (distinct value; the stream construction differs from pcg_046)."""
    return (int(seed) * INNER_SEED_MUL + 0x109) & ((1 << 64) - 1)


class C1:
    def __init__(self, R, rt):
        self.R, self.rt = R, rt
        self.mi, self.dr, self.PT = R.mi, R.dr, R.PT
        self.NB = R.PT.NB

    # ------------------------------------------------------------------------------------------- independent inner stream
    def inner_rng(self, N, seed64):
        mi, dr = self.mi, self.dr
        U, U64 = mi.UInt32, mi.UInt64
        lane = dr.arange(U, N)
        lo = U(seed64 & U64(0xFFFFFFFF))
        hi = U(seed64 >> U64(32))
        initstate = mi.sample_tea_64(lo ^ U(SALT_STATE), lane, 8)
        initseq = mi.sample_tea_64(lane, hi ^ U(SALT_SEQ), 8)
        r = mi.PCG32(size=N)
        inc = (initseq << U64(1)) | U64(1)
        r.state = (inc + initstate) * U64(0x5851F42D4C957F2D) + inc     # PCG reference seeding (same form as rt101.pcg_046)
        r.inc = inc
        return r

    def copy_rng(self, rng, N):
        mi = self.mi
        r = mi.PCG32(size=N)
        r.state = mi.UInt64(rng.state)
        r.inc = mi.UInt64(rng.inc)
        return r

    # ------------------------------------------------------------------------------------------- NEE-free SSS walk
    def walk_exit(self, si, ray, rng, active):
        """rt101.RT.sss_walk with the NEE block removed. NEE draws no random number, so the same stream yields the same
        collision sequence, exit point, exit direction and exit throughput."""
        mi, dr, rt = self.mi, self.dr, self.rt
        U = mi.UInt32
        p = mi.Point3f(si.p + ray.d * 1e-6); w = mi.Vector3f(ray.d)
        beta = mi.Color3f(1.)
        ep = mi.Point3f(si.p); ew = mi.Vector3f(ray.d); eb = mi.Color3f(0.)
        act = mi.Bool(active); depth = U(0)
        sigma, albedo = rt.sigma, rt.albedo

        def cond(rng, p, w, beta, ep, ew, eb, act, depth):
            return act & (depth < 128)

        def body(rng, p, w, beta, ep, ew, eb, act, depth):
            end = rt.sphere_exit(p, w)
            distance = -dr.log(dr.maximum(1. - rng.next_float32(), 1e-12)) / sigma
            coll = act & (distance < end)
            leave = act & ~coll
            ep = mi.Point3f(dr.select(leave, p + w * end, ep)); ew = mi.Vector3f(dr.select(leave, w, ew)); eb = mi.Color3f(dr.select(leave, beta, eb))
            p = mi.Point3f(p + w * dr.minimum(distance, end))
            beta = mi.Color3f(beta * albedo)
            u = rng.next_float32(); v = rng.next_float32()
            z = 1 - 2 * u; r = dr.sqrt(dr.maximum(1 - z * z, 0.)); a = 2 * math.pi * v
            w = mi.Vector3f(r * dr.cos(a), r * dr.sin(a), z)
            act = coll; depth = depth + 1
            return rng, p, w, beta, ep, ew, eb, act, depth

        rng2, p, w, beta, ep, ew, eb, act, depth = dr.while_loop((rng, p, w, beta, ep, ew, eb, act, depth), cond, body,
                                                                 labels=('rng', 'p', 'w', 'beta', 'ep', 'ew', 'eb', 'act', 'depth'),
                                                                 label='WL109_inner_first_exit')
        if rng2 is not rng:
            rng.state = rng2.state; rng.inc = rng2.inc
        return ep, ew, eb

    # ------------------------------------------------------------------------------------------- one inner continuation
    def inner_high(self, si, ray, wi, cl, is_sss, active, rng_in, B):
        """T(U, P') * B[a(U, P')] with the statements of pt100.Ctx.trace k = 1 -> k = 2 (HIGH part only)."""
        mi, dr, rt = self.mi, self.dr, self.rt
        F = mi.Float
        thr = mi.Color3f(1.)
        if rt.case == 'sss':
            ep, ew, eb = self.walk_exit(si, ray, rng_in, is_sss)
        wo, pdf, weight, valid = cl.sample(wi, rng_in.next_float32(), mi.Point2f(rng_in.next_float32(), rng_in.next_float32()), active & ~is_sss)
        nr = si.spawn_ray(si.to_world(wo))
        if rt.case == 'sss':
            nr.o = dr.select(is_sss, ep + ew * 1e-4, nr.o)
            nr.d = dr.select(is_sss, ew, nr.d)
            weight = dr.select(is_sss, eb, weight)
            valid = dr.select(is_sss, dr.max(eb) > 0., valid)
        thr *= dr.select(valid, weight, 0.)
        act = active & valid
        s2 = rt.scene.ray_intersect(nr, act)
        act &= s2.is_valid()
        sid2 = rt.shape_id(s2)
        p2, b2 = rt.patch_bin(s2, sid2, -nr.d)
        wi2 = s2.to_local(-nr.d)
        act &= wi2.z > 1e-6
        i = (p2 * self.NB + b2) * 3
        hl = mi.Color3f(*[dr.gather(F, B, i + c, act) for c in range(3)])
        return dr.select(act, thr * hl, 0.)

    # ------------------------------------------------------------------------------------------- trace
    def trace(self, ray, N, rng, rng_in, lights, B, M, orig_hl=False, replay=False):
        """pt100.Ctx.trace (ER, kmax 2, mode 'ecro') with HIGH replaced by the M-sample inner estimate (M = 0: no C1 HIGH)."""
        mi, dr, rt, PT = self.mi, self.dr, self.rt, self.PT
        F, U = mi.Float, mi.UInt32
        thr = mi.Color3f(1.); active = mi.Bool(True)
        out = {}

        def add(name, color):
            for c, ch in enumerate((color.x, color.y, color.z)):
                key = f'{name}{c}'; out[key] = out.get(key, F(0.)) + ch

        # ---- k = 1 (identical statements)
        si = rt.scene.ray_intersect(ray, active); active &= si.is_valid()
        sid = rt.shape_id(si); pid, bn = rt.patch_bin(si, sid, -ray.d)
        mid = dr.gather(U, rt.PMATU, pid, active)
        wi = si.to_local(-ray.d)
        active &= wi.z > 1e-6; wi = dr.select(active, wi, mi.Vector3f(0, 0, 1))
        cl = PT.CD.DrClosure83('ER', rt.mats.states, mid)
        is_sss = active & dr.eq(mid, 2) if rt.case == 'sss' else mi.Bool(False)
        # ---- C1 inner continuations: conditioned on U = (ray, si, wi, cl, is_sss, active); independent of the main P, S
        if M:
            if replay:
                rng_in = self.copy_rng(rng, N)
            hc = mi.Color3f(0.)
            for _ in range(M):
                hc = hc + self.inner_high(si, ray, wi, cl, is_sss, active, rng_in, B)
            add('hl', hc * (1.0 / M))
        # ---- k = 1 LOW and main continuation (identical statements, main stream)
        if rt.case == 'sss':
            qs, ep, ew, eb = rt.sss_walk(si, ray, rng, lights, is_sss)
        else:
            qs = mi.Color3f(0.)
        d, ballistic = rt.direct(si, wi, cl, lights, active, ~is_sss)
        add('e1', thr * (d + qs))
        add('e2', thr * ballistic)
        wo, pdf, weight, valid = cl.sample(wi, rng.next_float32(), mi.Point2f(rng.next_float32(), rng.next_float32()), active & ~is_sss)
        newray = si.spawn_ray(si.to_world(wo))
        if rt.case == 'sss':
            newray.o = dr.select(is_sss, ep + ew * 1e-4, newray.o)
            newray.d = dr.select(is_sss, ew, newray.d)
            weight = dr.select(is_sss, eb, weight)
            valid = dr.select(is_sss, dr.max(eb) > 0., valid)
        thr *= dr.select(valid, weight, 0.); active &= valid; ray = newray
        # ---- k = 2
        si = rt.scene.ray_intersect(ray, active); active &= si.is_valid()
        sid = rt.shape_id(si); pid, bn = rt.patch_bin(si, sid, -ray.d)
        mid = dr.gather(U, rt.PMATU, pid, active)
        wi = si.to_local(-ray.d)
        active &= wi.z > 1e-6; wi = dr.select(active, wi, mi.Vector3f(0, 0, 1))
        if orig_hl:
            i = (pid * self.NB + bn) * 3
            hl = mi.Color3f(*[dr.gather(F, B, i + c, active) for c in range(3)])
            add('ho', dr.select(active, thr * hl, 0.))
        cl = PT.CD.DrClosure83('ER', rt.mats.states, mid)
        is_sss = active & dr.eq(mid, 2) if rt.case == 'sss' else mi.Bool(False)
        if rt.case == 'sss':
            qs, ep, ew, eb = rt.sss_walk(si, ray, rng, lights, is_sss)
        else:
            qs = mi.Color3f(0.)
        d, ballistic = rt.direct(si, wi, cl, lights, active, ~is_sss)
        add('e2', thr * (d + qs))
        return out

    # ------------------------------------------------------------------------------------------- frame program
    def frame_prog(self, scene, tab, lp, Bd, seed, seed_in, cam_res_spp, M, orig_hl=False, replay=False, fixed_jitter=False):
        """Same frame organisation as rt101.RT._frame_prog (seeding, rays, per-pixel sums); keys sorted."""
        mi, dr, rt, R, PT = self.mi, self.dr, self.rt, self.R, self.PT
        rt._bind(tab)
        cam, res, spp = cam_res_spp
        n = res * res
        N = n * spp
        lt = rt.lt
        lt.pos, lt.rgb, lt.axis, lt.cc = list(lp['pos']), list(lp['rgb']), list(lp['axis']), list(lp['cc'])
        rng = R.pcg_046(N, seed, mi.UInt64(0))
        ray, pix = PT.gen_rays(cam, res, spp, None if fixed_jitter else rng)
        rng_in = None if (replay or not M) else self.inner_rng(N, seed_in)
        out = self.trace(ray, N, rng, rng_in, lt, Bd, M, orig_hl=orig_hl, replay=replay)
        keys = sorted(out)
        if spp == 1:
            return [out[k] for k in keys]
        bufs = []
        for k in keys:
            buf = dr.zeros(mi.Float, n)
            dr.scatter_reduce(dr.ReduceOp.Add, buf, out[k], pix)
            bufs.append(buf)
        return bufs

    @staticmethod
    def keys(M, orig_hl):
        k = ['e10', 'e11', 'e12', 'e20', 'e21', 'e22']
        if M:
            k += ['hl0', 'hl1', 'hl2']
        if orig_hl:
            k += ['ho0', 'ho1', 'ho2']
        return sorted(k)
