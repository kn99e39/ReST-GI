"""WL90 diagnostics of the common fidelity gap between converged explicit transport and the oracle table (frozen ER; WL86-WL89 code imported
unchanged; nothing here is a deployable representation).

Table recurrence of every WL87-WL89 relation (R0 / R1 / C3UR):   L_1 = q,  L_{k+1} = G(P L_k),  B = L_2 + .. + L_5
  P  projection of the sender radiance onto the patch x 64-bin table (patch-constant in space, bin-constant in direction), applied before
     EVERY gather; G = receiver gather (relation + frozen-ER scattering into the receiver's own patch x bin table).
Oracle table:  B*[i, bo] = receiver patch x bin average of the path-traced 2..5-event radiance = no P at any sender.

Exact-read operators on the pooled R0 relation (receiver gather, WL87 per-ray weights W, NR realizations). For every state ray (x -> hit y on
patch p, direction v0 = y -> x) the gather reads, instead of the table entry L_k[p, bin_p(v0)], a path-traced k-event radiance:
  X(a, s)[k] = G( P_{a,s} Lx_k ),   Lx_k = exact k-event radiance field (frozen-ER path tracing, pt86 estimator: next-event at every vertex)
  P_{a,s}: read at a point y' and direction v':
     s = INF  y' = y (exact point)            s finite: y' uniform in the sub-patch of p containing y (s = g x g sub-patches per patch)
     a = INF  v' = v0 (exact direction)       a finite: v' uniform in the sub-bin containing v0 (a = g x g sub-bins per bin, patch frame)
  (a, s) = (1, 1) is exactly the table's projection applied ONCE to the exact field; (INF, INF) is no projection.
  S samples per ray, averaged over the valid ones (v' above the local horizon at y', probe ray returns to y'), as the table average.
Hybrids (how many gathers must be exact):  M_m = sum_{k<=m} X(INF,INF)[k] + sum_{j=1}^{4-m} T^j X(INF,INF)[m]   (M_0 = table recurrence,
  M_4 = X(INF, INF) = no intermediate projection), T = pooled table operator of the same realizations.
Feasibility probes (deployed single relation; source-independent relation unchanged, only the FIRST gather reads the exact once-scattered
radiance per relation ray, later orders by the table operator):  R0X1 (receiver gather),  C3URX1 (sender-uniform relation: every ray
carries the exact radiance of its own origin point and direction).
"""
import math, time
import numpy as np
import torch

import fx90
PT, Q = fx90.PT, fx90.Q
mi, dr, S, CD = PT.mi, PT.dr, PT.S, PT.CD
F, U = PT.F, PT.U
import rel88 as RL  # noqa: E402
import rel89 as R9  # noqa: E402
import xport87 as X  # noqa: E402
NB, K, NZ, NPHI = S.NB, S.K, S.NZ, S.NPHI
TWO_PI = 2 * math.pi
dev = torch.device('cuda')
INF = 0                                         # code for 'exact' in (a, s)
KMAX = 4


def trace_orders(ctx, ray, lights, rng, kmax, probe_y=None, active0=None):
    """pt86 'cam' estimator split by event count: list of kmax Color3f (contribution of the next-event estimate at vertex k = k events
    arriving along the ray), and the first-hit mask (probe_y: first hit must be within 1 mm of probe_y)."""
    scene, mats = ctx.scene, ctx.mats
    thr = mi.Color3f(1.0)
    active = mi.Bool(True) if active0 is None else mi.Bool(active0)
    out = []
    okp = None
    for k in range(1, kmax + 1):
        si = scene.ray_intersect(ray, active)
        active &= si.is_valid()
        mid = dr.gather(U, ctx.SMAT, ctx.shape_id(si))
        wi = si.to_local(-ray.d)
        if k == 1:
            if probe_y is not None:
                active &= dr.norm(si.p - probe_y) < 1e-3
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
        out.append(c)
        if k == kmax:
            break
        u1 = rng.next_float32(); u2 = mi.Point2f(rng.next_float32(), rng.next_float32())
        wo, pdf, wgt, valid = cl.sample(wi, u1, u2, active)
        thr = thr * dr.select(valid, wgt, 0.0)
        active &= valid
        ray = si.spawn_ray(si.to_world(wo))
    return out, okp


def sub_patch_point(ctx, pid, y, s, u1, u2):
    """Point y' uniform in the sub-patch (s = g x g per patch) of patch pid that contains y; its local normal. s = INF: y itself."""
    g = lambda k: dr.gather(F, ctx.PTB, pid * 21 + k)
    sph = g(18) > 0.5
    if s == INF:
        q = (y - mi.Point3f(*ctx.SC)) * (1.0 / ctx.SR)
        t, bv, n = ctx.patch_frame(pid)
        return y, dr.select(sph, mi.Vector3f(q.x, q.y, q.z), n)
    gs = int(round(math.sqrt(s)))
    assert gs * gs == s
    e1 = mi.Vector3f(g(3), g(4), g(5)); e2 = mi.Vector3f(g(6), g(7), g(8))
    rel = y - mi.Point3f(g(0), g(1), g(2))
    ua_r = dr.dot(rel, e1) * dr.rcp(dr.squared_norm(e1)); ub_r = dr.dot(rel, e2) * dr.rcp(dr.squared_norm(e2))
    q = (y - mi.Point3f(*ctx.SC)) * (1.0 / ctx.SR)
    ua_s = (dr.atan2(q.y, q.x) + math.pi) * (S.SPH_NPHI / TWO_PI) - g(19)
    ub_s = (q.z + 1.0) * (0.5 * S.SPH_NZ) - g(20)
    ua = dr.clamp(dr.select(sph, ua_s, ua_r), 0.0, 1.0 - 1e-6); ub = dr.clamp(dr.select(sph, ub_s, ub_r), 0.0, 1.0 - 1e-6)
    a = (dr.floor(ua * gs) + u1) * (1.0 / gs); b = (dr.floor(ub * gs) + u2) * (1.0 / gs)
    p2, n2, _ = ctx.patch_point(pid, a, b)
    return p2, n2


def sub_bin_dir(ctx, pid, v0, a, u1, u2):
    """World direction v' uniform in the sub-bin (a = g x g per bin, patch frame of pid) that contains v0. a = INF: v0 itself."""
    if a == INF:
        return v0
    ga = int(round(math.sqrt(a)))
    assert ga * ga == a
    t, bv, n = ctx.patch_frame(pid)
    x, y, z = dr.dot(v0, t), dr.dot(v0, bv), dr.dot(v0, n)
    zz = dr.clamp(z, 0.0, 0.999999) * NZ
    pp = dr.clamp((dr.atan2(y, x) + math.pi) * (NPHI / TWO_PI), 0.0, NPHI - 1e-6)
    z2 = (dr.floor(zz * ga) + u1) * (1.0 / (NZ * ga))
    ph = (dr.floor(pp * ga) + u2) * (TWO_PI / (NPHI * ga)) - math.pi
    r = dr.sqrt(dr.maximum(1.0 - z2 * z2, 0.0))
    return t * (r * dr.cos(ph)) + bv * (r * dr.sin(ph)) + n * z2


def xread(ctx, lt, pid_np, y_np, v_np, a, s, Sn, seed, kmax=KMAX, chunk=3_000_000):
    """Projected exact radiance per read: [n, kmax, 3] float32 (mean over the valid ones of Sn samples) and the valid-sample count [n]."""
    n = len(pid_np)
    out = np.zeros((n, kmax, 3), np.float32); cnt = np.zeros(n, np.float32)
    rays_per = max(1, chunk // Sn)
    for s0 in range(0, n, rays_per):
        sl = slice(s0, min(n, s0 + rays_per))
        nn = sl.stop - sl.start
        lanes = nn * Sn
        li = dr.arange(U, lanes)
        ri = li // Sn
        gf = lambda arr: dr.gather(F, PT.fl(arr), ri)
        pid = dr.gather(U, U(np.ascontiguousarray(pid_np[sl].astype(np.uint32))), ri)
        y = mi.Point3f(gf(y_np[sl, 0]), gf(y_np[sl, 1]), gf(y_np[sl, 2]))
        v0 = mi.Vector3f(gf(v_np[sl, 0]), gf(v_np[sl, 1]), gf(v_np[sl, 2]))
        rng = mi.PCG32(size=lanes, initstate=dr.opaque(mi.UInt64, seed * 1000003 + s0), initseq=mi.UInt64(0))
        u = [rng.next_float32() for _ in range(4)]
        y2, n2 = sub_patch_point(ctx, pid, y, s, u[0], u[1])
        v2 = sub_bin_dir(ctx, pid, v0, a, u[2], u[3])
        ok = dr.dot(v2, n2) > 1e-3
        ray = mi.Ray3f(y2 + v2 * 2e-3, -v2)
        cs, okp = trace_orders(ctx, ray, lt, rng, kmax, probe_y=y2, active0=ok)
        w = dr.select(okp, F(1.0), F(0.0))
        vals = [w] + [c[ch] for c in cs for ch in range(3)]
        dr.eval(vals, ri); dr.sync_thread()                                        # evaluate BEFORE any scatter (Dr.Jit 0.4.6 lazy-scatter defect)
        bufs = []
        for v in vals:
            b = dr.zeros(F, nn)
            dr.scatter_reduce(dr.ReduceOp.Add, b, v, ri)
            bufs.append(b)
        dr.eval(bufs); dr.sync_thread()
        c_ = np.asarray(bufs[0], np.float32)
        cnt[sl] = c_
        acc = np.stack([np.asarray(b, np.float32) for b in bufs[1:]], 1).reshape(nn, kmax, 3)
        out[sl] = acc / np.maximum(c_, 1.0)[:, None, None]
    return out, cnt


def state_reads(st):
    """Hit rays of an R0 state: indices, hit patch, hit point y (segment end), direction v0 = y -> origin (world)."""
    idx = np.nonzero(st.hit >= 0)[0]
    o = st.seg[idx, :3].astype(np.float64); e = st.seg[idx, 3:].astype(np.float64)
    v = o - e; v /= np.linalg.norm(v, axis=1, keepdims=True)
    return idx, st.hit[idx].astype(np.int64), e.astype(np.float32), v.astype(np.float32)


def gather_orders(op, N, idx, rad):
    """G of per-ray per-order radiance: rad [n, kmax, 3] at hit rays idx -> [kmax, N, NB, 3] torch (WL87 per-ray weights of op)."""
    kmax = rad.shape[1]
    Lr = torch.zeros(N * K, kmax, 3, device=dev)
    Lr[torch.as_tensor(idx, device=dev)] = torch.as_tensor(rad, device=dev)
    Lr = Lr.view(N, K, kmax, 3)
    return torch.stack([op.scatter(Lr[:, :, k] * op.ok) for k in range(kmax)])


def hybrids(GEE, ops, q):
    """M_0..M_4 tables (torch [N, NB, 3]) from the pooled exact gathers GEE [4, N, NB, 3] and the pooled table operator; also their orders."""
    T = lambda L: sum(op.apply(L) for op in ops) / len(ops)
    out = {}
    for m in range(0, KMAX + 1):
        orders = [GEE[k] for k in range(m)]
        L = GEE[m - 1] if m > 0 else torch.as_tensor(np.asarray(q, np.float32), device=dev)
        for j in range(KMAX - m):
            L = T(L); orders.append(L)
        out[f'M{m}'] = orders
    return out


# ------------------------------------------------------------------------------------------------ feasibility probes (deployed relations)
class RayBuf:
    """Resident Dr.Jit probe rays of a relation (built once per geometry, on the GPU): the exact once-scattered radiance leaving y towards v
    is the next-event estimate at the first hit of the ray (y + v 2e-3, -v). Per source: one launch (point lights: deterministic)."""

    def __init__(self, y_t, v_t):
        """y_t, v_t: torch float32 [n, 3] on the GPU (points and unit directions)."""
        self.n = int(y_t.shape[0])
        p = y_t + v_t * 2e-3
        d = -v_t
        col = lambda t, k: F(t[:, k].contiguous())
        self.o = mi.Point3f(col(p, 0), col(p, 1), col(p, 2)); self.d = mi.Vector3f(col(d, 0), col(d, 1), col(d, 2))
        dr.eval(self.o, self.d)

    def radiance(self, ctx, lt):
        if self.n == 0:
            return torch.zeros(0, 3, device=dev)
        rng = mi.PCG32(size=self.n, initstate=dr.opaque(mi.UInt64, 1), initseq=mi.UInt64(0))
        cs, _ = trace_orders(ctx, mi.Ray3f(self.o, self.d), lt, rng, 1)
        c = cs[0]
        dr.eval(c); dr.sync_thread()
        return torch.stack([c[k].torch() for k in range(3)], 1)


def r0_raybuf(st):
    """Probe rays of the hit rays of an R0 state: hit point y (segment end), direction v = y -> origin."""
    idx = np.nonzero(st.hit >= 0)[0]
    sg = torch.as_tensor(st.seg, device=dev)[torch.as_tensor(idx, device=dev)]
    v = sg[:, :3] - sg[:, 3:]
    v = v / v.norm(dim=1, keepdim=True)
    return idx, RayBuf(sg[:, 3:], v)


def r0_first_exact(op, N, idx, rad):
    """L_2 of R0 from exact per-ray once-scattered radiance rad [n, 3] (torch)."""
    Lr = torch.zeros(N * K, 3, device=dev)
    Lr[torch.as_tensor(idx, device=dev)] = rad
    return op.scatter(Lr.view(N, K, 3) * op.ok)


def c3ur_raybuf(rc):
    """Probe rays of the landed C3UR rays: origin point y (stored origin is y + n 1e-4) and direction v = origin -> hit."""
    keep = torch.nonzero(rc.hit_t >= 0)[:, 0]
    sg = rc.seg_t[keep]
    v = sg[:, 3:] - sg[:, :3]
    v = v / v.norm(dim=1, keepdim=True)
    return RayBuf(sg[:, :3], v)


def c3ur_first(rc, rad):
    """L_2 of C3UR from per-landed-ray sender radiance rad [n_landed, 3] (torch): the unchanged deposition with L[src] replaced by rad."""
    G = rc.G
    e = rad * rc.w_t[:, None]
    E = torch.zeros(G.N * R9.NC, 3, device=dev)
    E.index_add_(0, rc.slot_t, e)
    E = E.view(G.N, R9.NC, 3)
    out = torch.empty(G.N, NB, 3, device=dev)
    for m, ix in rc.groups:
        if rc.lam[m]:
            out[ix] = (E[ix].sum(1) * rc.rho[ix][:, None])[:, None, :].expand(-1, NB, -1)
        else:
            out[ix] = torch.einsum('ncC,cbC->nbC', E[ix], rc.D[m])
    return out


def rec_from(op, L2, J_=KMAX):
    """B = L2 + T L2 + .. (J_ orders in total), given the first gather."""
    L = L2; acc = L2.clone()
    for _ in range(J_ - 1):
        L = op.apply(L); acc = acc + L
    return acc
