"""WL115 Stage A2 (post hoc mechanism diagnostic, SELECTION set only; designed after the first Stage A readouts showed that a
spatially refined U table keeps a broad near over-brightness that the camera-conditioned table at the same spatial resolution
removes). Angular sub-bin refinement of the U measure on the diagnostic contact neighbourhood:

  construct_ka: tracer115.construct_k with each of the 64 direction bins split into Ka x Ka sub-bins (stratified in the bin's own
                z / phi parameters: proposals u2' = (az + u2) / Ka, u3' = (ap + u3) / Ka in exactly the tracer114 direction formulas,
                incl. the sphere's extended first bin ring); pooled over sub-bins it is an ordinary K x K construction (nested control)
  'pca' lookup: piecewise constant on per-patch K_p x K_p cells and Ka_p x Ka_p angular sub-bins (Ka_p = 1: the 'pc' lookup).
tracer115 is imported unchanged.
"""
import math
import numpy as np
import tracer115 as T

PT, S, mi, dr, F, U = T.PT, T.S, T.mi, T.dr, T.F, T.U
NB = T.NB


def sub_bin(ctx, pid, w, bn, Ka):
    """Angular sub-bin (az * Ka + ap) of world direction w inside bin bn of patch pid (Ka python int or UInt32 per lane)."""
    t, bv, nn = ctx.patch_frame(pid)
    x, y, z = dr.dot(w, t), dr.dot(w, bv), dr.dot(w, nn)
    iz = bn // S.NPHI; ip = bn % S.NPHI
    sph = dr.eq(dr.gather(U, ctx.PSIDU, pid), S.SID_SPHERE)
    fz = dr.select(sph & dr.eq(iz, 0), (z + 1.0) * (1.0 / 1.25), z * S.NZ - F(iz))
    fp = (dr.atan2(y, x) + math.pi) * (S.NPHI / (2 * math.pi)) - F(ip)
    Kf = F(Ka) if not isinstance(Ka, int) else float(Ka)
    az = U(dr.clamp(dr.floor(fz * Kf), 0, Kf - 1)); ap = U(dr.clamp(dr.floor(fp * Kf), 0, Kf - 1))
    return az * (U(Ka) if not isinstance(Ka, int) else Ka) + ap


class CtxA2(T.Ctx115):
    def construct_ka(self, lights, n_sub, seed, kmax_probe, Kk, Ka, pids, b=1, verbose=False):
        import time
        pids = np.asarray(pids, np.int64)
        P_ = len(pids); NSK = Kk * Kk; NA = Ka * Ka
        ncs = P_ * NSK * NB * NA
        PIDS = U(np.ascontiguousarray(pids.astype(np.uint32)))
        acc = {}
        diag = {'mismatch': 0., 'counted': 0., 'launches': 0}
        deep = T.T114.deep_through(24)
        t0 = time.perf_counter()
        for launch, done in enumerate(range(0, n_sub, b)):
            bb = min(b, n_sub - done); Nl = ncs * bb
            sd = dr.opaque(mi.UInt64, seed * 1000003 + launch)
            rng = mi.PCG32(size=Nl, initstate=sd, initseq=mi.UInt64(0))
            slot = dr.arange(U, Nl) % ncs
            asb = slot % NA; q = slot // NA
            bn = q % NB; ps = q // NB; sub = ps % NSK; pl = ps // NSK
            pid = dr.gather(U, PIDS, pl)
            u0, u1, u2, u3 = (rng.next_float32() for _ in range(4))
            u2 = (F(asb // Ka) + u2) * (1.0 / Ka); u3 = (F(asb % Ka) + u3) * (1.0 / Ka)
            a = (F(sub % Kk) + u0) * (1.0 / Kk); bpar = (F(sub // Kk) + u1) * (1.0 / Kk)
            sid = dr.gather(U, self.PSIDU, pid)
            t, bv, n = self.patch_frame(pid)
            local = self.dir_in_bin(bn, u2, u3)
            z = 1.25 * u2 - 1.
            ph = (F(bn % S.NPHI) + u3) * (2 * math.pi / S.NPHI) - math.pi
            rr_ = dr.sqrt(dr.maximum(1 - z * z, 0.))
            local = dr.select(dr.eq(sid, S.SID_SPHERE) & (bn < S.NPHI), mi.Vector3f(rr_ * dr.cos(ph), rr_ * dr.sin(ph), z), local)
            w = t * local.x + bv * local.y + n * local.z
            si, ssid, nn = self.first_si(pid, a, bpar, w)
            bur = self.buried(si.p, sid, nn); sing = ~bur & self.singular_point(si.p, sid, nn)
            ray = mi.Ray3f(si.p + w * 2e-3, -w)
            pre = ~bur & ~sing
            apid, abn = self.patch_bin(si, sid, w)
            asub = self.subcell_k(si, apid, Kk)
            aab = sub_bin(self, apid, w, abn, Ka)
            wi0 = si.to_local(w)
            cnt = pre & si.is_valid() & (wi0.z > 1e-6)
            wt = dr.select(cnt, F(1.), F(0.))
            mism = cnt & (dr.neq(apid, pid) | dr.neq(abn, bn) | dr.neq(asub, sub) | dr.neq(aab, asb))
            dr.eval(wt, mism, slot, si, ray)
            out, cnt2, d = self.trace_deep(ray, Nl, rng, sd, lights, 'probe', kmax_probe, slot, ncs, wt=wt, first=si, first_ok=pre)
            par = launch % 2
            bufv = dr.zeros(mi.Float64, ncs)
            dr.scatter_reduce(dr.ReduceOp.Add, bufv, mi.Float64(wt), slot)
            acc[f'w{par}'] = acc.get(f'w{par}', 0.) + np.asarray(bufv, np.float64)
            diag['mismatch'] += float(dr.sum(mi.Float64(dr.select(mism, F(1.), F(0.))))[0]); diag['counted'] += float(dr.sum(mi.Float64(wt))[0])
            for c in range(3):
                hf = out.get(f'f2{c}', 0.) + out.get(f'fh{c}', 0.) + sum((out[f'{nm}{c}'] for nm in deep if f'{nm}{c}' in out), 0.)
                acc[f'HF{par}{c}'] = acc.get(f'HF{par}{c}', 0.) + hf
            diag['launches'] += 1
            if verbose and launch % 16 == 0:
                print('  launch', launch, round(time.perf_counter() - t0, 1), flush=True)
            del out, rng, si, ray
        diag['seconds'] = time.perf_counter() - t0
        dr.flush_malloc_cache()
        return {k_: np.asarray(v, np.float64).reshape(P_, NSK, NB, NA) for k_, v in acc.items()}, diag

    def lookup(self, rep, cap):
        if rep['kind'] != 'pca':
            return super().lookup(rep, cap)
        pid, bn, a, b = cap['pid'], cap['bin'], cap['a'], cap['b']
        Kp = dr.gather(U, rep['Kp'], pid); Ka = dr.gather(U, rep['Ka'], pid); off = dr.gather(U, rep['off'], pid)
        asb = sub_bin(self, pid, cap['w'], bn, Ka)
        idx = off + (self.sub_k(a, b, Kp) * NB + bn) * (Ka * Ka) + asb
        return [dr.gather(F, rep['T'], idx * 3 + c) for c in range(3)]


def pca_hybrid(T_base, Kb, fine, Kf, Ka, pids):
    """Dense Kb x Kb (no angular split) everywhere; fine [P, Kf^2, NB, Ka^2, 3] on `pids`."""
    N = T_base.shape[0]
    Kp = np.full(N, Kb, np.uint32); Kap = np.ones(N, np.uint32)
    blocks = [np.asarray(T_base[p], np.float32).reshape(-1) for p in range(N)]
    for i, p in enumerate(pids):
        Kp[p] = Kf; Kap[p] = Ka; blocks[p] = np.asarray(fine[i], np.float32).reshape(-1)
    sizes = np.array([x.size // 3 for x in blocks], np.int64)
    off = np.concatenate([[0], np.cumsum(sizes)[:-1]]).astype(np.uint32)
    return {'kind': 'pca', 'Kp': U(Kp), 'Ka': U(Kap), 'off': U(off), 'T': F(np.ascontiguousarray(np.concatenate(blocks)))}


def pool_ang(x, Ka_from, Ka_to):
    """Sum-pool the angular sub-bin axis (index 3, az * Ka + ap) of [P, K2, NB, Ka^2, ...]."""
    r = Ka_from // Ka_to
    s = x.shape
    t = x.reshape(s[:3] + (Ka_to, r, Ka_to, r) + s[4:])
    return t.sum((4, 6)).reshape(s[:3] + (Ka_to * Ka_to,) + s[4:])


def table_ka(w, L, Kk):
    """U-mean per (cell, bin, sub-bin); an empty sub-bin takes its (cell, bin) U mean, which itself follows the tracer115 nested
    fallback (empty cell -> parent cell), so the Ka-pooled table is exactly the tracer115.table_from_sums contract."""
    Tb = T.table_from_sums(w.sum(3), L.sum(3), Kk)
    Tt = L / np.maximum(w, 1e-30)[..., None]
    return np.where((w > 0)[..., None], Tt, Tb[:, :, :, None]).astype(np.float32)
