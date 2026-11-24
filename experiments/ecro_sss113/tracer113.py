"""WL113 tracer: WL112 tracer112 (contract v3, C/N semantics, analytic pinned first surface) extended with

  (i)   4 x 4 sub-patch addresses (16 equal-area cells of the patch parameters; 2 x 2 and coarse are exact poolings),
  (ii)  one deep physical camera pass to 24 events with per-event evaluation (measured 4x faster than the fused depth-6
        pass): events 1-6 are statement- and RNG-identical to the depth-6 estimator (D6), events 7-24 go to 'tail';
        events beyond 24 are < 1e-11 of the total (development check against 40); no roulette,
  (iii) the geometry-only visibility-heterogeneity support (VHS) and per-sub-cell geometric descriptors,
  (iv)  a decorrelated lane seeding option (RNG control; Amendment-1 finding of WL112).
tracer112 is imported unchanged; nothing in a shared production file is edited.
"""
import math, os, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
sys.path.insert(0, str(EXP / 'ecro_sss112'))
import tracer112 as T112  # noqa: E402

T112.NSUB = 16                     # module-level sub-cell count used by tracer112.cam_pass for the event-2 sub index
PT, S, C100 = T112.PT, T112.S, T112.C100
mi, dr, F, U = T112.mi, T112.dr, T112.F, T112.U
NB = S.NB
K = 4
NS = K * K
H_MIN = T112.H_MIN
LUM = T112.LUM
KMAX_FULL = 24
EVAL_EVERY = 1


def decorrelated_pcg(N, sd, salt):
    lane = dr.arange(mi.UInt64, N)
    h = (lane + mi.UInt64(salt) * mi.UInt64(0x9E3779B97F4A7C15)) * mi.UInt64(0xBF58476D1CE4E5B9)
    return mi.PCG32(size=N, initstate=sd, initseq=h ^ (h >> 31))


class Ctx113(T112.Ctx112):
    # ------------------------------------------------------------------ 4 x 4 sub-patch address of a hit
    def subcell(self, si, pid):
        g = lambda k: dr.gather(F, self.PTB, pid * 21 + k)
        o = mi.Point3f(g(0), g(1), g(2)); e1 = mi.Vector3f(g(3), g(4), g(5)); e2 = mi.Vector3f(g(6), g(7), g(8))
        rel = si.p - o
        a = dr.dot(rel, e1) / dr.maximum(dr.squared_norm(e1), 1e-30); b = dr.dot(rel, e2) / dr.maximum(dr.squared_norm(e2), 1e-30)
        q = (si.p - self.cvec) * (1.0 / self.SR)
        zf = (q.z + 1.0) * (0.5 * S.SPH_NZ) - g(20); pf = (dr.atan2(q.y, q.x) + math.pi) * (S.SPH_NPHI / (2 * math.pi)) - g(19)
        sph = g(18) > 0.5
        a = dr.select(sph, pf, a); b = dr.select(sph, zf, b)
        su = U(dr.clamp(dr.floor(a * K), 0, K - 1)); sv = U(dr.clamp(dr.floor(b * K), 0, K - 1))
        return su + K * sv

    # ------------------------------------------------------------------ full-depth trace (copy of trace112, events >= 7 -> tail, per-event evaluation)
    def trace_full(self, ray, N, rng, sd, lights, kmax=KMAX_FULL):
        thr = mi.Color3f(1.)
        active = mi.Bool(True)
        out = {}; bad = mi.Float(0.); badf = mi.Float(0.)

        def add(name, color):
            for c, ch in enumerate((color.x, color.y, color.z)):
                key = f'{name}{c}'
                out[key] = out.get(key, F(0.)) + ch

        def contribution(k, color):
            add('e1' if k == 1 else ('e2' if k == 2 else ('h' if k <= 6 else 'tail')), color)
        for k in range(1, kmax + 1):
            si = self.scene.ray_intersect(ray, active)
            active &= si.is_valid()
            sid = self.shape_id(si); pid, bn = self.patch_bin(si, sid, -ray.d)
            mid = dr.gather(U, self.PMATU, pid, active)
            wi = si.to_local(-ray.d)
            active &= wi.z > 1e-6; wi = dr.select(active, wi, mi.Vector3f(0, 0, 1))
            cl = PT.CD.DrClosure83('ER', self.mats.states, mid)
            is_sss = active & dr.eq(mid, 2)
            qs, ep, ew, eb = self.sss_walk(si, ray, rng, lights, is_sss)
            d, ballistic = self.direct(si, wi, cl, lights, active, ~is_sss)
            bad += dr.select(self._bad_direct[0], 1., 0.); badf += dr.select(self._bad_direct[1], 1., 0.)
            contribution(k, thr * (d + qs))
            if k < kmax:
                contribution(k + 1, thr * ballistic)
            if k == kmax:
                break
            wo, pdf, weight, valid = cl.sample(wi, rng.next_float32(), mi.Point2f(rng.next_float32(), rng.next_float32()), active & ~is_sss)
            newray, hs = self.spawn(si, si.to_world(wo))
            eo, es = self.exit_origin(ep, ew)
            newray.o = dr.select(is_sss, eo, newray.o)
            newray.d = dr.select(is_sss, ew, newray.d)
            weight = dr.select(is_sss, eb, weight)
            valid = dr.select(is_sss, dr.max(eb) > 0., valid)
            valid = valid & ~(hs & ~is_sss) & ~(es & is_sss)
            ins, blw = self.origin_inside(newray.o)
            bad += dr.select(valid & ins, 1., 0.); badf += dr.select(valid & blw, 1., 0.)
            thr *= dr.select(valid, weight, 0.); active &= valid; ray = newray
            if k % EVAL_EVERY == 0:          # EVAL_EVERY = 1 (measured fastest)
                dr.eval(thr, active, ray.o, ray.d, rng.state, rng.inc, list(out.values()), bad, badf)   # bound the kernel size
        out['bad_origin'] = bad; out['bad_floor'] = badf
        return out

    def full_pass(self, lights, cam, res, spp, seed, b=32):
        """Physical deep camera pass to KMAX_FULL events: e1, e2, h (events 3-6) and tail (events 7..KMAX_FULL)."""
        n = res * res
        img = {}; bad = 0.; badf = 0.
        for launch, done in enumerate(range(0, spp, b)):
            bb = min(b, spp - done); Nl = n * bb
            sd = dr.opaque(mi.UInt64, seed * 1000003 + launch)
            rng = mi.PCG32(size=Nl, initstate=sd, initseq=mi.UInt64(0))
            ray, pix = PT.gen_rays(cam, res, bb, rng)
            out = self.trace_full(ray, Nl, rng, sd, lights)
            bad += float(dr.sum(out.pop('bad_origin'))[0]); badf += float(dr.sum(out.pop('bad_floor'))[0])
            a = self.accum(out, pix, n)
            for k_, v in a.items():
                img[k_] = img.get(k_, 0.) + v
        res_img = {nm: np.stack([img.get(f'{nm}{c}', np.zeros(n)) for c in range(3)], -1).reshape(res, res, 3) / spp
                   for nm in ('e1', 'e2', 'h', 'tail')}
        self.last_bad_floor = badf
        dr.flush_malloc_cache()
        return res_img, bad

    # ------------------------------------------------------------------ 4 x 4 camera-independent construction (U measure)
    def construct4(self, lights, n_sub, seed, b=8, decorrelate=False, synthetic=None):
        ncs = self.N * NS * NB
        acc = {}
        for launch, done in enumerate(range(0, n_sub, b)):
            bb = min(b, n_sub - done); Nl = ncs * bb
            sd = dr.opaque(mi.UInt64, seed * 1000003 + launch)
            rng = decorrelated_pcg(Nl, sd, seed * 1000 + launch) if decorrelate else mi.PCG32(size=Nl, initstate=sd, initseq=mi.UInt64(0))
            slot = dr.arange(U, Nl) % ncs
            bn = slot % NB; ps = slot // NB; sub = ps % NS; pid = ps // NS
            u0, u1, u2, u3 = (rng.next_float32() for _ in range(4))
            a = (F(sub % K) + u0) * (1.0 / K); bpar = (F(sub // K) + u1) * (1.0 / K)
            sid = dr.gather(U, self.PSIDU, pid)
            t, bv, n = self.patch_frame(pid)
            local = self.dir_in_bin(bn, u2, u3)
            z = 1.25 * u2 - 1.
            ph = (F(bn % S.NPHI) + u3) * (2 * math.pi / S.NPHI) - math.pi
            rr = dr.sqrt(dr.maximum(1 - z * z, 0.))
            local = dr.select(dr.eq(sid, S.SID_SPHERE) & (bn < S.NPHI), mi.Vector3f(rr * dr.cos(ph), rr * dr.sin(ph), z), local)
            w = t * local.x + bv * local.y + n * local.z
            si, ssid, nn = self.first_si(pid, a, bpar, w)
            bur = self.buried(si.p, sid, nn); sing = ~bur & self.singular_point(si.p, sid, nn)
            ray = mi.Ray3f(si.p + w * 2e-3, -w)
            pre = ~bur & ~sing
            if synthetic is None:
                out, cap = self.trace112(ray, Nl, rng, sd, 5, lights, 'probe', first=si, first_ok=pre)
                cnt = cap['count']
            else:
                cnt = pre & (si.to_local(w).z > 1e-6)
                col = synthetic(si.p, w, nn)
                out = {f'f2{c}': dr.select(cnt, cc, 0.) for c, cc in enumerate((col.x, col.y, col.z))}
                out.update({f'fh{c}': F(0.) for c in range(3)}); out['bad_origin'] = F(0.); out['bad_floor'] = F(0.)
            apid, abn = self.patch_bin(si, sid, w)
            asub = self.subcell(si, apid)
            mism = cnt & (dr.neq(apid, pid) | dr.neq(abn, bn) | dr.neq(asub, sub))
            wt = dr.select(cnt, F(1.), F(0.))
            vals = {'U_w': wt, 'buried': dr.select(bur, F(1.), F(0.)), 'singular': dr.select(sing, F(1.), F(0.)),
                    'back': dr.select(pre & ~cnt, F(1.), F(0.)), 'mismatch': dr.select(mism, F(1.), F(0.)),
                    'bad_origin': out['bad_origin'], 'bad_floor': out['bad_floor']}
            for c in range(3):
                f2c, fhc = out[f'f2{c}'], out[f'fh{c}']
                vals[f'U_f2{c}'] = wt * f2c; vals[f'U_fh{c}'] = wt * fhc; vals[f'U_sq{c}'] = wt * (f2c + fhc) * (f2c + fhc)
                vals[f'misL{c}'] = dr.select(mism, f2c + fhc, 0.)
            keys = sorted(vals); vv = [vals[k] for k in keys]
            dr.eval(vv, slot); dr.sync_thread()
            if not acc:
                acc = {k: dr.zeros(mi.Float64, ncs) for k in keys}
            for k, v in zip(keys, vv):
                dr.scatter_reduce(dr.ReduceOp.Add, acc[k], mi.Float64(v), slot)
            dr.eval([acc[k] for k in keys]); dr.sync_thread()
        acc = {k: np.asarray(v, np.float64) for k, v in acc.items()}
        acc['proposals'] = np.full(ncs, float(n_sub))
        dr.flush_malloc_cache()
        return acc

    # ------------------------------------------------------------------ geometry-only VHS support and sub-cell descriptors
    def vhs(self):
        """Per patch and 4 x 4 sub-cell: cos-weighted fraction of hemisphere rays (64 bin-centre directions of the patch frame)
        whose first hit lies closer than the patch's own size L_p = sqrt(area) [near], cos-weighted mean log10 hit distance,
        cos-weighted escape fraction. Geometry only; no camera, light, radiance or reference."""
        N = self.N
        n = N * NS * NB
        lane = dr.arange(U, n)
        bn = lane % NB; ps = lane // NB; sub = ps % NS; pid = ps // NS
        a = (F(sub % K) + 0.5) * (1.0 / K); b = (F(sub // K) + 0.5) * (1.0 / K)
        t, bv, nf = self.patch_frame(pid)
        loc = self.dir_in_bin(bn, F(0.5), F(0.5))
        w = t * loc.x + bv * loc.y + nf * loc.z
        si, sid, nn = self.first_si(pid, a, b, w)
        cosl = dr.dot(w, nn)
        ok = cosl > 1e-3
        ray, sing = self.spawn(si, w)
        hit = self.scene.ray_intersect(ray, ok & ~sing)
        L = dr.sqrt(dr.gather(F, F(self.G.PAREA.astype(np.float32)), pid))
        dist = dr.select(hit.is_valid(), hit.t, F(1e3))
        wt = dr.select(ok, cosl, 0.)
        vals = {'w': wt, 'near': dr.select(ok & hit.is_valid() & (hit.t < L), cosl, 0.),
                'logd': dr.select(ok & hit.is_valid(), cosl * dr.log(dr.maximum(dist, 1e-6)) * (1 / math.log(10)), 0.),
                'esc': dr.select(ok & ~hit.is_valid(), cosl, 0.)}
        a_ = self.accum(vals, ps, N * NS)
        w_ = np.maximum(a_['w'], 1e-12)
        desc = np.stack([a_['near'] / w_, a_['logd'] / np.maximum(a_['w'] - a_['esc'], 1e-12), a_['esc'] / w_], -1).reshape(N, NS, 3)
        return desc.astype(np.float32)


DELTA = 0.15
SIGMA_T = 12.0


def support_from_desc(desc, G, delta=DELTA):
    """Frozen compact contract VHS u MEDIUM. desc [N,16,3] from Ctx113.vhs(); G the patch tables (PSID, PAREA)."""
    near = desc[..., 0]
    vhs = (near.max(1) - near.min(1)) >= delta
    medium = (G.PSID == S.SID_SPHERE) & (1.0 / SIGMA_T < np.sqrt(G.PAREA))
    return vhs | medium


def support_ring0(G):
    """Diagnostic support control: the 8 lowest-ring sphere patches only."""
    return (G.PSID == S.SID_SPHERE) & (G.PSPH[:, 1] == 0)


def silhouette_coverage(ctx, cam, res=512, n=8):
    """Per-pixel sphere sub-pixel coverage on an n x n grid (geometry only)."""
    cov = np.zeros(res * res)
    o, fwd, right, up, s = PT.cam_basis(cam)
    for i in range(n):
        for j in range(n):
            pix = dr.arange(U, res * res)
            ii, jj = pix // res, pix % res
            x = (F(jj) + (j + 0.5) / n) * (2.0 / res) - 1.0; y = 1.0 - (F(ii) + (i + 0.5) / n) * (2.0 / res)
            d = dr.normalize(mi.Vector3f(*[float(fwd[k]) + s * x * float(right[k]) + s * y * float(up[k]) for k in range(3)]))
            si = ctx.scene.ray_intersect(mi.Ray3f(mi.Point3f(*[F(float(o[k])) for k in range(3)]), d))
            hit = si.is_valid() & dr.eq(ctx.shape_id(si), S.SID_SPHERE)
            cov += np.asarray(dr.select(hit, F(1.), F(0.)))
    return (cov / (n * n)).reshape(res, res)


# ------------------------------------------------------------------------------------ tables (4 x 4 nested)
def tables4(acc, N=1270):
    """Nested U tables from one 4 x 4 construction: coarse [N,64,3], 2x2 [N,4,64,3], 4x4 [N,16,64,3] (empty -> parent),
    plus native-support masks."""
    w = acc['U_w'].reshape(N, K, K, NB)                              # [N, sv, su, bin]
    L = np.stack([acc[f'U_f2{c}'] + acc[f'U_fh{c}'] for c in range(3)], -1).reshape(N, K, K, NB, 3)
    wc = w.sum((1, 2)); Lc = L.sum((1, 2))
    Tc = Lc / np.maximum(wc, 1e-30)[..., None]
    w2 = w.reshape(N, 2, 2, 2, 2, NB).sum((2, 4)); L2 = L.reshape(N, 2, 2, 2, 2, NB, 3).sum((2, 4))   # [N, sv2, su2, bin]
    T2 = np.where((w2 > 0)[..., None], L2 / np.maximum(w2, 1e-30)[..., None], Tc[:, None, None])
    T4 = np.where((w > 0)[..., None], L / np.maximum(w, 1e-30)[..., None],
                  np.repeat(np.repeat(T2, 2, axis=1), 2, axis=2))
    return Tc.astype(np.float32), T2.reshape(N, 4, NB, 3).astype(np.float32), T4.reshape(N, NS, NB, 3).astype(np.float32), \
        {'c': wc > 0, '2': (w2 > 0).reshape(N, 4, NB), '4': (w > 0).reshape(N, NS, NB)}


def to_dense16(T, kind, support=None, coarse=None):
    """Express any table as a dense 4 x 4 readout array [N,16,64,3] (exact broadcast; values unchanged)."""
    N = T.shape[0]
    if kind == 'coarse':
        return np.repeat(T[:, None], NS, axis=1)
    if kind == 'sub2':
        t = T.reshape(N, 2, 2, NB, 3)
        return np.repeat(np.repeat(t, 2, axis=1), 2, axis=2).reshape(N, NS, NB, 3)
    if kind == 'sub4':
        return T
    if kind == 'compact':                                            # refined patches 4x4, others coarse
        out = np.repeat(coarse[:, None], NS, axis=1)
        out[support] = T[support]
        return out
    raise KeyError(kind)


def lum(x):
    return x[..., 0] * LUM[0] + x[..., 1] * LUM[1] + x[..., 2] * LUM[2]
