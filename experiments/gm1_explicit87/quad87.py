"""WL87 discrete ER scattering on the WL86 patch x bin state (learning-free; frozen ER closures, Dr.Jit).

Notation (patch-local frame (t, b, n) of scene86; 64 equal-solid-angle bins, DOMEGA = 2 pi / 64; eval(wi, wo) = f(wi, wo) cos(wo)).
ER f is reciprocal (f(a, b) = f(b, a): symmetric GGX lobes, symmetric s(mu_i) s(mu_o), Lambert) - test T_SYM.

  fbar(m, w)[bo]   = (1 / DOMEGA) int_{v in bo} f_m(v, w) dv          bin-averaged outgoing response to light arriving from w
                     estimator: one-sample MIS mixture, v ~ 0.5 ER-sampling(wi = w) + 0.5 uniform hemisphere, weight f / p_mix,
                     histogrammed into the outgoing bin of v (all 64 bins from one sample set)
  W_ray[m, r, bo]  = fbar(m, w_r)[bo] cos(w_r) DOMEGA / kappa          receiver-gather weight of state ray r (its exact local direction)
                     (Lambert m: rho / pi cos(w_r) DOMEGA / kappa, exact)
  q[i, bo]         = mean over P stratified points x of patch i of  sum_l E_l(x) fbar(m_i, w_l(x))[bo]      once-scattered radiance
                     (E_l = I_l cos / d^2 with shadow ray and cone; Lambert: rho / pi E exactly)
  sphere           curved patches: W rows per sphere band at each ray's origin point (local frame, cosine and horizon), q bins
                   normalised by the valid solid angle (directions above the local horizon), matching pt86.Ctx.field
  dense table      fbar on 4,096 incident cells (8 x 8 strata per incident bin, cell centres, flat patch frame): used by the encoding
                   control (C2-ENC) and the sender-side control (C3) only
"""
import math
import numpy as np

import fx87
PT = fx87.PT
mi, dr, S, CD = PT.mi, PT.dr, PT.S, PT.CD
F, U = PT.F, PT.U
NB, K, KAPPA, NZ, NPHI = S.NB, S.K, S.KAPPA, S.NZ, S.NPHI
DOMEGA = S.DOMEGA
TWO_PI = 2 * math.pi
ER_MATS = (0, 1, 2)
SUB = 8                                   # dense incident cells: SUB x SUB strata per bin
NCELL = NB * SUB * SUB


def mats():
    z = np.load(fx87.os.path.join(fx87.E86, 'data', 'mats86.npz'))
    return z['states'], z['lambert'].astype(bool), z['albedo'].astype(np.float64)


def ray_dirs(us):
    """Local directions of the K state rays of a realization (us = [4, 1, K] offsets; exactly Ctx.dir_in_bin(k // KAPPA, us[2], us[3]))."""
    k = np.arange(K)
    bn = k // KAPPA
    z = (bn // NPHI + us[2, 0].astype(np.float64)) / NZ
    phi = (bn % NPHI + us[3, 0].astype(np.float64)) * (TWO_PI / NPHI) - math.pi
    r = np.sqrt(np.maximum(1 - z * z, 0))
    return np.stack([r * np.cos(phi), r * np.sin(phi), z], 1)


def cell_dirs():
    """Centres of the dense incident cells: cell = bin * SUB^2 + sz * SUB + sp -> [NCELL, 3]."""
    c = np.arange(NCELL)
    bn, rem = c // (SUB * SUB), c % (SUB * SUB)
    sz, sp = rem // SUB, rem % SUB
    z = (bn // NPHI + (sz + 0.5) / SUB) / NZ
    phi = (bn % NPHI + (sp + 0.5) / SUB) * (TWO_PI / NPHI) - math.pi
    r = np.sqrt(np.maximum(1 - z * z, 0))
    return np.stack([r * np.cos(phi), r * np.sin(phi), z], 1)


def cell_of(x, y, z):
    """Dense cell index of a local direction (Dr.Jit), consistent with cell_dirs and Ctx.bin_of."""
    zz = dr.clamp(z, 0.0, 0.999999) * NZ
    pp = dr.clamp((dr.atan2(y, x) + math.pi) * (NPHI / TWO_PI), 0.0, NPHI - 1e-6)
    iz = U(dr.floor(zz)); ip = U(dr.floor(pp))
    sz = U(dr.clamp(dr.floor((zz - F(iz)) * SUB), 0, SUB - 1)); sp = U(dr.clamp(dr.floor((pp - F(ip)) * SUB), 0, SUB - 1))
    return (iz * NPHI + ip) * (SUB * SUB) + sz * SUB + sp


def _mis_samples(cl, w, rng):
    """One MIS-mixture sample v about the light direction w: returns v, f(w, v) / p_mix (Color3f), valid."""
    pick = rng.next_float32() < 0.5
    u1, u2 = rng.next_float32(), rng.next_float32()
    zu = u1; phu = u2 * TWO_PI - math.pi
    ru = dr.sqrt(dr.maximum(1.0 - zu * zu, 0.0))
    vu = mi.Vector3f(ru * dr.cos(phu), ru * dr.sin(phu), zu)
    vs, _, _, oks = cl.sample(w, rng.next_float32(), mi.Point2f(rng.next_float32(), rng.next_float32()), w.z > 1e-6)
    v = dr.select(pick, vu, vs)
    ok = (w.z > 1e-6) & (v.z > 1e-6) & dr.select(pick, mi.Bool(True), oks)
    v = dr.select(ok, v, mi.Vector3f(0, 0, 1))
    f, p = cl.eval_pdf(w, v, ok)
    pmix = 0.5 * p + 0.5 / TWO_PI
    val = f * dr.rcp(v.z * pmix)
    return v, dr.select(ok, val, 0.0), ok


def fbar(state, dirs, M, seed, chunk=8192):
    """[n, NB, 3] float64: bin-averaged outgoing response (no cosine) of one ER material to unit-irradiance... per unit f: see header."""
    n = len(dirs)
    cl = CD.DrClosure83('ER', state)
    acc = np.zeros((n * NB, 3))
    dx, dy, dz = (PT.fl(dirs[:, k]) for k in range(3))
    done, it = 0, 0
    while done < M:
        m = min(chunk, M - done)
        lanes = n * m
        di = dr.arange(U, lanes) // m
        w = mi.Vector3f(dr.gather(F, dx, di), dr.gather(F, dy, di), dr.gather(F, dz, di))
        rng = mi.PCG32(size=lanes, initstate=dr.opaque(mi.UInt64, seed * 1000003 + it), initseq=mi.UInt64(0))
        v, val, ok = _mis_samples(cl, w, rng)
        slot = di * NB + PT.Ctx.bin_of(v.x, v.y, v.z)
        bufs = []
        dr.eval(val, slot); dr.sync_thread()
        for c, ch in enumerate((val.x, val.y, val.z)):
            b = dr.zeros(F, n * NB)
            dr.scatter_reduce(dr.ReduceOp.Add, b, ch, slot)
            bufs.append(b)
        dr.eval(bufs); dr.sync_thread()
        for c in range(3):
            acc[:, c] += np.asarray(bufs[c], np.float64)
        done += m; it += 1
    return (acc / (M * DOMEGA)).reshape(n, NB, 3)


def integ_frames(state, wp, R, M, seed, chunk=8192):
    """[n, NB, 3]: int over {v in patch-frame bin bo, v above the local horizon (v_p.z > 1e-3)} of f(v, w) dv, for light directions w
    given in the POINT shading frame (wp [n, 3]); R [n, 3, 3] maps point-frame vectors to patch-frame coordinates (rows: patch axes)."""
    n = len(wp)
    cl = CD.DrClosure83('ER', state)
    acc = np.zeros((n * NB, 3))
    wx, wy, wz = (PT.fl(wp[:, k]) for k in range(3))
    Rf = [PT.fl(R[:, a, b]) for a in range(3) for b in range(3)]
    done, it = 0, 0
    while done < M:
        m = min(chunk, M - done)
        lanes = n * m
        di = dr.arange(U, lanes) // m
        g = lambda x: dr.gather(F, x, di)
        w = mi.Vector3f(g(wx), g(wy), g(wz))
        rng = mi.PCG32(size=lanes, initstate=dr.opaque(mi.UInt64, seed * 1000003 + it), initseq=mi.UInt64(0))
        v, val, ok = _mis_samples(cl, w, rng)
        Rg = [g(x) for x in Rf]
        vc = [Rg[3 * a] * v.x + Rg[3 * a + 1] * v.y + Rg[3 * a + 2] * v.z for a in range(3)]
        keep = ok & (v.z > 1e-3) & (vc[2] > 0.0)
        val = dr.select(keep, val, 0.0)
        slot = di * NB + PT.Ctx.bin_of(vc[0], vc[1], vc[2])
        dr.eval(val, slot); dr.sync_thread()
        bufs = []
        for ch in (val.x, val.y, val.z):
            b = dr.zeros(F, n * NB)
            dr.scatter_reduce(dr.ReduceOp.Add, b, ch, slot)
            bufs.append(b)
        dr.eval(bufs); dr.sync_thread()
        for c in range(3):
            acc[:, c] += np.asarray(bufs[c], np.float64)
        done += m; it += 1
    return (acc / M).reshape(n, NB, 3)


def sphere_band_frames(iz, us):
    """For sphere band iz (patch iphi = 0; the other 7 patches of the band are rotations about z, so their local quantities are equal):
    per state ray r: origin point normal, light direction in the point frame, R (point -> patch frame), cos at the point, ok."""
    zc = (iz + 0.5) / S.SPH_NZ * 2 - 1; pc = 0.5 / S.SPH_NPHI * TWO_PI - math.pi
    n = np.array([math.sqrt(1 - zc * zc) * math.cos(pc), math.sqrt(1 - zc * zc) * math.sin(pc), zc])
    t = np.cross([0, 0, 1.0], n); t /= np.linalg.norm(t); b = np.cross(n, t)
    a_, b_ = us[0, 0].astype(np.float64), us[1, 0].astype(np.float64)
    zp = (iz + b_) * (2.0 / S.SPH_NZ) - 1.0; php = a_ * (TWO_PI / S.SPH_NPHI) - math.pi
    rr = np.sqrt(np.maximum(1 - zp * zp, 0))
    ns = np.stack([rr * np.cos(php), rr * np.sin(php), zp], 1)
    tp = np.stack([-ns[:, 1], ns[:, 0], np.zeros(K)], 1); tp /= np.linalg.norm(tp, axis=1, keepdims=True)
    bp = np.cross(ns, tp)
    wl = ray_dirs(us)
    w = wl[:, :1] * t + wl[:, 1:2] * b + wl[:, 2:3] * n
    cosp = (w * ns).sum(1)
    wp = np.stack([(w * tp).sum(1), (w * bp).sum(1), cosp], 1)
    E = np.stack([tp, bp, ns], 2)                                     # columns: point axes in world  [K, 3, 3]
    C = np.stack([t, b, n])                                           # rows: patch axes in world      [3, 3]
    R = np.einsum('aj,kjb->kab', C, E)                                # patch-frame coordinates of the point axes
    return ns, (t, b, n), wp, R, cosp, cosp > 1e-3


def valid_omega(ns, frame, Mv, seed):
    """[n, NB]: solid angle of each patch-frame bin that lies above the local horizon of points with normals ns (uniform sampling).
    frame: patch axes (t, b, n), each [3] (shared) or [n, 3] (per point)."""
    npt = len(ns)
    t, b, n = (np.broadcast_to(np.asarray(x, np.float64), (npt, 3)) for x in frame)
    lanes = npt * NB * Mv
    li = dr.arange(U, lanes)
    pi = li // (NB * Mv); bn = (li // Mv) % NB
    rng = mi.PCG32(size=lanes, initstate=dr.opaque(mi.UInt64, seed), initseq=mi.UInt64(0))
    v = PT.Ctx.dir_in_bin(bn, rng.next_float32(), rng.next_float32())
    gk = lambda a, k: dr.gather(F, PT.fl(a[:, k]), pi)
    vw = [v.x * gk(t, k) + v.y * gk(b, k) + v.z * gk(n, k) for k in range(3)]
    nsf = [gk(ns, k) for k in range(3)]
    ok = dr.select(vw[0] * nsf[0] + vw[1] * nsf[1] + vw[2] * nsf[2] > 1e-3, F(1.0), F(0.0))
    dr.eval(ok); dr.sync_thread()
    return np.asarray(ok, np.float64).reshape(npt, NB, Mv).mean(2) * DOMEGA


class Quad:
    """Quadrature of one state realization: W [6 + SPH_NZ, K, NB, 3] - rows 0..5 by material (flat patches: patch frame = shading
    frame), rows 6 + iz for the sphere bands (each ray at its own origin point: local shading frame, local cosine, local horizon;
    normalised by the bin's mean valid solid angle, as the oracle field pt86.Ctx.field averages over valid directions) - plus dense
    tables (built lazily; flat patch-frame approximation, used only by the ENC / C3 controls)."""

    def __init__(self, us, M_ray, seed_ray, M_dense=None, seed_dense=None):
        self.states, self.lam, self.alb = mats()
        self.dirs = ray_dirs(us)
        cosr = self.dirs[:, 2]
        self.W = np.zeros((6 + S.SPH_NZ, K, NB, 3), np.float64)
        for m in range(6):
            if self.lam[m]:
                self.W[m] = (self.alb[m] / math.pi * cosr * DOMEGA / KAPPA)[:, None, None]
            else:
                fb = fbar(self.states[m], self.dirs, M_ray, seed_ray + 10 * m)
                self.W[m] = fb * (cosr * DOMEGA / KAPPA)[:, None, None]
        self.omega_sph = []
        for iz in range(S.SPH_NZ):
            ns, frame, wp, R, cosp, ok = sphere_band_frames(iz, us)
            I = integ_frames(self.states[2], np.where(ok[:, None], wp, [0, 0, 1.0]), R, M_ray, seed_ray + 100 + 10 * iz)
            om = valid_omega(ns, frame, 256, seed_ray + 200 + iz).mean(0)                           # mean valid solid angle per bin
            self.omega_sph.append(om)
            self.W[6 + iz] = I * (np.where(ok, cosp, 0.0) * DOMEGA / KAPPA)[:, None, None] / np.maximum(om, 1e-12)[None, :, None]
        self.M_dense, self.seed_dense = M_dense, seed_dense
        self._dense = None

    @staticmethod
    def wid(G):
        """W row of every patch: material id, or 6 + iz for sphere patches."""
        w = np.asarray(G.PMAT).copy()
        sph = G.PSID == S.SID_SPHERE
        w[sph] = 6 + G.PSPH[sph, 1].astype(np.int64)
        return w

    def dense(self):
        """fbar on the dense incident cells [6, NCELL, NB, 3] (Lambert: rho / pi)."""
        if self._dense is None:
            cd = cell_dirs()
            D = np.zeros((6, NCELL, NB, 3))
            for m in range(6):
                D[m] = self.alb[m] / math.pi if self.lam[m] else fbar(self.states[m], cd, self.M_dense, self.seed_dense + 10 * m)
            self._dense = D
        return self._dense

    def binq_W(self):
        """C2-BINQ: each ray's weights replaced by the mean over the kappa rays of its incident bin (the scattering response then sees
        only the bin-mean incident radiance: incident direction information inside the bin is discarded; same quadrature samples)."""
        n = self.W.shape[0]
        Mb = self.W.reshape(n, NB, KAPPA, NB, 3).mean(2, keepdims=True)
        return np.broadcast_to(Mb, (n, NB, KAPPA, NB, 3)).reshape(n, K, NB, 3).copy()

    def enc_matrix(self):
        """C2-ENC: Mq[m, bin_in, bin_out] = mean over the incident bin of fbar (the source encoding s carries the cosine)."""
        return self.dense().reshape(6, NB, SUB * SUB, NB, 3).mean(2)


def point_frames(ctx, pid, a, b):
    """Stratified patch points with their shading frame (sphere: frame rotated to the point normal, scene86 construction)."""
    p, nn, (t, bv, n) = ctx.patch_point(pid, a, b)
    sph = dr.gather(F, ctx.PTB, pid * 21 + 18) > 0.5
    ts = dr.normalize(mi.Vector3f(-nn.y, nn.x, 0.0))
    tp = dr.select(sph, ts, t)
    bp = dr.select(sph, dr.cross(nn, ts), bv)
    np_ = dr.select(sph, nn, n)
    return p, nn, (tp, bp, np_), (t, bv, n)


def _q_static(ctx, P):
    """Geometry-independent index tables (the patch layout and materials are fixed for every geometry of the family)."""
    if getattr(ctx, '_q87', None) is None or ctx._q87['P'] != P:
        G = ctx.G
        st, lam, alb = mats()
        erp = np.nonzero(~lam[G.PMAT])[0]
        pts = (erp[:, None] * P + np.arange(P)[None]).ravel().astype(np.uint32)
        ctx._q87 = {'P': P, 'er_pts': U(pts), 'n_er': len(pts), 'pmat': U(G.PMAT.astype(np.uint32)), 'states': st,
                    'lamk': PT.fl(np.where(lam[G.PMAT], alb[G.PMAT] / math.pi, 0.0))}
    return ctx._q87


def qtable(ctx, lt, P, Mq, seed, occlusion=True):
    """Once-scattered outgoing radiance per (patch, outgoing bin) [N, NB, 3] float64 (see header). occlusion=False: test mode.
    GPU-resident: point records and the MIS lanes stay in Dr.Jit; one scatter per channel."""
    qs = _q_static(ctx, P)
    N = ctx.N
    sq = int(round(math.sqrt(P)))
    assert sq * sq == P
    npnt = N * P
    lane = dr.arange(U, npnt)
    pid = lane // P; j = lane % P
    rng = mi.PCG32(size=npnt, initstate=dr.opaque(mi.UInt64, seed), initseq=mi.UInt64(0))
    a = (F(j % sq) + rng.next_float32()) * (1.0 / sq); b = (F(j // sq) + rng.next_float32()) * (1.0 / sq)
    p, nn, (tp, bp, np_), (t, bv, n) = point_frames(ctx, pid, a, b)
    Es, Ws = [], []
    Etot = mi.Color3f(0.0)
    for l in range(lt.n):
        dl = lt.pos[l] - p
        d2 = dr.squared_norm(dl); dist = dr.sqrt(d2)
        wl = dl * dr.rcp(dist)
        cs = dr.dot(wl, nn)
        on = (cs > 1e-6) & (dr.dot(-wl, lt.axis[l]) > lt.cc[l])
        if occlusion:
            r = mi.Ray3f(p + nn * 1e-4, wl); r.maxt = dist * (1.0 - 1e-4)
            on &= ~ctx.scene.ray_test(r, on)
        E = lt.rgb[l] * dr.select(on, cs * dr.rcp(d2), 0.0)
        Es.append(E); Ws.append(mi.Vector3f(dr.dot(wl, tp), dr.dot(wl, bp), dr.dot(wl, np_)))
        Etot += E
    lw = dr.gather(F, qs['lamk'], pid) * (1.0 / P)
    el = Etot * lw
    dr.eval(Es, Ws, t, bv, n, tp, bp, np_, el, pid, nn); dr.sync_thread()   # evaluate BEFORE any scatter (Dr.Jit 0.4.6 lazy-scatter defect, WL84)
    if qs.get('om_seed') != seed:
        # sphere bins: mean valid solid angle (above the local horizon) over the P points of the patch; flat patches: DOMEGA
        G = ctx.G
        sph = np.nonzero(G.PSID == S.SID_SPHERE)[0]
        spts = (sph[:, None] * P + np.arange(P)[None]).ravel()
        nsn = np.stack([np.asarray(nn[k], np.float64)[spts] for k in range(3)], 1)
        pp_ = spts // P
        om = valid_omega(nsn, (G.PT[pp_], G.PB[pp_], G.PN[pp_]), 256, 87990).reshape(len(sph), P, NB).mean(1)
        omega = np.full((N, NB), DOMEGA); omega[sph] = om
        qs['invom'] = PT.fl(1.0 / np.maximum(omega, 1e-12)); qs['om_seed'] = seed; qs['omega'] = omega
    bl = [dr.zeros(F, N) for _ in range(3)]
    for c in range(3):
        dr.scatter_reduce(dr.ReduceOp.Add, bl[c], el[c], pid)
    dr.eval(bl)
    # ER patches: MIS samples about each point's light direction, binned in the PATCH frame
    lanes = qs['n_er'] * Mq
    li = dr.arange(U, lanes)
    pi = dr.gather(U, qs['er_pts'], li // Mq)
    pp = pi // P
    cl = CD.DrClosure83('ER', qs['states'], dr.gather(U, qs['pmat'], pp))
    gv = lambda v: type(v)(dr.gather(F, v.x, pi), dr.gather(F, v.y, pi), dr.gather(F, v.z, pi))
    ft, fb, fn, fp, fq, fr = gv(t), gv(bv), gv(n), gv(tp), gv(bp), gv(np_)
    buf = [dr.zeros(F, N * NB) for _ in range(3)]
    for l in range(lt.n):
        w = gv(Ws[l])
        E = mi.Color3f(dr.gather(F, Es[l].x, pi), dr.gather(F, Es[l].y, pi), dr.gather(F, Es[l].z, pi))
        rr = mi.PCG32(size=lanes, initstate=dr.opaque(mi.UInt64, seed * 7919 + 101 * l), initseq=mi.UInt64(1))
        v, val, ok = _mis_samples(cl, w, rr)
        vworld = fp * v.x + fq * v.y + fr * v.z
        vz = dr.dot(vworld, fn)
        slot = pp * NB + PT.Ctx.bin_of(dr.dot(vworld, ft), dr.dot(vworld, fb), vz)
        keep = ok & (v.z > 1e-3) & (vz > 0.0)                                 # inside a patch-frame bin and above the local horizon
        cv = dr.select(keep, val * E * (dr.gather(F, qs['invom'], slot) * (1.0 / (P * Mq))), 0.0)
        dr.eval(cv, slot); dr.sync_thread()                                    # evaluate BEFORE the scatter (lazy-scatter defect)
        for c in range(3):
            dr.scatter_reduce(dr.ReduceOp.Add, buf[c], cv[c], slot)
        dr.eval(buf)
    dr.sync_thread()
    q = np.stack([np.asarray(x, np.float64) for x in buf], 1).reshape(N, NB, 3)
    ql = np.stack([np.asarray(x, np.float64) for x in bl], 1)
    return q + ql[:, None, :]
