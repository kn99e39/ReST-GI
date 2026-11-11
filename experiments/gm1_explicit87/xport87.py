"""WL87 explicit (learning-free) patch x bin transport on the WL86 geometry state.

Primary comparator EXP (receiver gather on the deployed state):
  L_1 = q                                           once-scattered table (quad87.qtable)
  L_{k+1}[i, bo] = sum_r W_ray[m_i, r, bo] L_k[hit_r, dbin_r]      over the K = 512 state rays r of patch i (miss / back face: 0)
  B_exp = L_2 + L_3 + L_4 + L_5 = sum_{j=1}^{4} T_d^j q_d          outgoing radiance of 2..5 scattering events (the WL86 table target)
The camera reads B_exp exactly as ECRO's table (pt86 'ecro' mode): hl = throughput(x1) B[patch(x2), bin(x2 -> x1)], so camera paths
receive 3..6 scattering events from it; LOW (e1 = q, e2 = T q at the camera vertices) is unchanged and B excludes q (no double count).

Controls (same output table, same readout):
  C2-BINQ     W_ray replaced by the bin-averaged incident quadrature (scattering quadrature)
  C2-ENC      L_1 from ECRO's own source encoding s (incident-bin projection of the light at the first event)
  C2-RAYSEND  first gather reads the exact once-scattered radiance at each state ray's hit point (sender projection of the 1-event
              family, incl. light -> panel -> receiver); later orders as EXP
  C3          sender-side deposition: per order, R = N x K rays drawn from the sender table in proportion to emitted power (systematic
              resampling), traced in the current geometry, deposited at the receiver's dense incident cell (separate relation per order)
  POOL        convergence diagnostic (not matched budget): T averaged over several independent state realizations
"""
import math, time
import numpy as np
import torch

import fx87
import quad87 as Q
PT = fx87.PT
mi, dr, S, CD = PT.mi, PT.dr, PT.S, PT.CD
F, U = PT.F, PT.U
NB, K, KAPPA, NZ, NPHI = S.NB, S.K, S.KAPPA, S.NZ, S.NPHI
DOMEGA = S.DOMEGA
dev = torch.device('cuda')
J = 4


def sync():
    dr.sync_thread(); torch.cuda.synchronize()


def realization_us(seed):
    """State-ray offsets of a realization: the deployed rule (pt86.Ctx.__init__: default_rng(seed).random((4, 1, K))). 86001 = deployed."""
    rs = np.random.default_rng(seed)
    return rs.random((4, 1, K), np.float32)


def set_realization(ctx, us):
    ctx.u_state_np = us
    ctx.set_tables(ctx.G)


class XOp:
    """Torch form of T_d for one geometry state: hit / bin per ray and per-material W [6, K, NB, 3]."""

    def __init__(self, G, hit, dbin, W):
        self.N = G.N
        h = torch.as_tensor(hit.astype(np.int64), device=dev).view(G.N, K)
        self.ok = (h >= 0).view(G.N, K, 1).float()
        self.idx = (h.clamp(min=0) * NB + torch.as_tensor(dbin.astype(np.int64), device=dev).view(G.N, K)).view(-1)
        self.W = W if isinstance(W, torch.Tensor) else torch.as_tensor(np.asarray(W, np.float32), device=dev)      # resident (geometry-independent)
        wid = Q.Quad.wid(G)                                                 # material row, or 6 + iz for sphere patches
        self.groups = [(m, torch.as_tensor(np.nonzero(wid == m)[0], device=dev)) for m in range(self.W.shape[0]) if (wid == m).any()]

    def gather(self, L):
        """L [N, NB, 3] -> radiance arriving along every state ray [N, K, 3]."""
        return L.reshape(-1, 3)[self.idx].view(self.N, K, 3) * self.ok

    def scatter(self, Lr):
        out = torch.zeros(self.N, NB, 3, device=dev)
        for m, ix in self.groups:
            out[ix] = torch.einsum('nkc,kbc->nbc', Lr[ix], self.W[m])
        return out

    def apply(self, L):
        return self.scatter(self.gather(L))


def recurrence(ops, q, first=None):
    """B = sum_{k=2}^{5} L_k with L_{k+1} = mean_ops T L_k. first: optional precomputed L_2. Returns B and the orders [L2..L5]."""
    L = torch.as_tensor(np.asarray(q, np.float32), device=dev)
    orders = []
    for j in range(J):
        if j == 0 and first is not None:
            L = torch.as_tensor(np.asarray(first, np.float32), device=dev)
        else:
            L = sum(op.apply(L) for op in ops) / len(ops)
        orders.append(L)
    B = sum(orders)
    return B, orders


def raysend_first(ctx, st, lt, W):
    """C2-RAYSEND: L_2 from the exact once-scattered radiance at each state ray's hit point (towards the ray origin)."""
    idx = np.nonzero(st.hit >= 0)[0]
    o = st.seg[idx, :3].astype(np.float64); e = st.seg[idx, 3:].astype(np.float64)
    d = e - o; d /= np.linalg.norm(d, axis=1, keepdims=True)
    n = len(idx)
    ray = mi.Ray3f(mi.Point3f(*[PT.fl(o[:, k]) for k in range(3)]), mi.Vector3f(*[PT.fl(d[:, k]) for k in range(3)]))
    sd = dr.opaque(mi.UInt64, 1)
    rng = mi.PCG32(size=n, initstate=sd, initseq=mi.UInt64(0))
    out, _ = ctx.trace(ray, n, rng, sd, 'ER', 1, lt, 'cam')
    vals = [out[f'e1{c}'] for c in range(3)]
    dr.eval(vals); dr.sync_thread()
    Lr = np.zeros((st.hit.size, 3), np.float32)
    Lr[idx] = np.stack([np.asarray(v, np.float32) for v in vals], 1)
    op = XOp(ctx.G, st.hit, st.dbin, W)
    return op.scatter(torch.as_tensor(Lr, device=dev).view(ctx.N, K, 3)).cpu().numpy()


def enc_q(G, s, Mq):
    """C2-ENC: once-scattered table from the deployed source encoding s [N, NB(in), 3]: q[i, bo] = sum_bi s[i, bi] Mq[m_i, bi, bo]."""
    q = np.zeros((G.N, NB, 3))
    for m in range(6):
        sel = G.PMAT == m
        if sel.any():
            q[sel] = np.einsum('nic,ioc->noc', s[sel].astype(np.float64), Mq[m])
    return q


# ------------------------------------------------------------------------------------------------ C3: sender-side deposition
def _cosdir_in_bin(bn, u1, u2):
    """Direction in bin bn with density proportional to cos (z^2 uniform within the ring), azimuth uniform."""
    iz = F(bn // NPHI)
    z0 = iz * (1.0 / NZ); z1 = (iz + 1.0) * (1.0 / NZ)
    z = dr.sqrt(z0 * z0 + u1 * (z1 * z1 - z0 * z0))
    phi = (F(bn % NPHI) + u2) * (2 * math.pi / NPHI) - math.pi
    r = dr.sqrt(dr.maximum(1.0 - z * z, 0.0))
    return mi.Vector3f(r * dr.cos(phi), r * dr.sin(phi), z)


def deposit_order(ctx, L, dense, R, seed, chunk=400_000):
    """One order of sender-side transport: table L [N, NB, 3] -> next table. R rays, systematic resampling on luminance power."""
    G = ctx.G
    zc = (np.arange(NB) // NPHI + 0.5) / NZ
    Pw = L.astype(np.float64) * (G.PAREA[:, None] * zc[None] * DOMEGA)[..., None]     # emitted power per (patch, bin, rgb)
    lum = Pw.mean(-1).ravel()
    tot = lum.sum()
    out = np.zeros((G.N, NB, 3))
    if tot <= 0:
        return out
    cdf = np.cumsum(lum) / tot
    u0 = np.random.default_rng(seed).random()
    us = (np.arange(R) + u0) / R
    cell = np.minimum(np.searchsorted(cdf, us, side='right'), len(cdf) - 1)
    pm = lum[cell] / tot
    flux = Pw.reshape(-1, 3)[cell] / (R * pm)[:, None]                               # rgb flux carried by each ray
    pid = cell // NB; bn = cell % NB
    dens = torch.as_tensor(np.asarray(dense, np.float32), device=dev)                 # [6, NCELL, NB, 3]
    outt = torch.zeros(G.N * NB, 3, device=dev)
    area = torch.as_tensor(G.PAREA, device=dev, dtype=torch.float32)
    pmat = torch.as_tensor(G.PMAT, device=dev)
    for s0 in range(0, R, chunk):
        sl = slice(s0, min(R, s0 + chunk))
        n = sl.stop - sl.start
        rng = mi.PCG32(size=n, initstate=dr.opaque(mi.UInt64, seed * 1000003 + s0), initseq=mi.UInt64(0))
        pd = U(np.ascontiguousarray(pid[sl].astype(np.uint32)))
        p, nn, (t, bv, nz) = ctx.patch_point(pd, rng.next_float32(), rng.next_float32())
        wl = _cosdir_in_bin(U(np.ascontiguousarray(bn[sl].astype(np.uint32))), rng.next_float32(), rng.next_float32())
        w = t * wl.x + bv * wl.y + nz * wl.z
        ok = dr.dot(w, nn) > 1e-3
        si = ctx.scene.ray_intersect(mi.Ray3f(p + nn * 1e-4, w), ok)
        valid = ok & si.is_valid()
        sid = ctx.shape_id(si)
        front = si.to_local(-w).z > 1e-6
        ph, _ = ctx.patch_bin(si, sid, -w)
        tr, br, nr = ctx.patch_frame(ph)
        cl = Q.cell_of(dr.dot(-w, tr), dr.dot(-w, br), dr.dot(-w, nr))
        hit = dr.select(valid & front, mi.Int32(ph), mi.Int32(-1))
        dr.eval(hit, cl); dr.sync_thread()
        h = torch.as_tensor(np.asarray(hit).astype(np.int64), device=dev)
        c = torch.as_tensor(np.asarray(cl).astype(np.int64), device=dev)
        fx = torch.as_tensor(flux[sl].astype(np.float32), device=dev)
        keep = h >= 0
        h, c, fx = h[keep], c[keep], fx[keep]
        e = fx / area[h][:, None]                                                     # irradiance contribution per ray
        for m in range(6):
            mm = pmat[h] == m
            if mm.any():
                contrib = dens[m][c[mm]] * e[mm][:, None, :]                          # [n, NB, 3]
                outt.index_add_(0, (h[mm][:, None] * NB + torch.arange(NB, device=dev)[None]).reshape(-1), contrib.reshape(-1, 3))
    return outt.view(G.N, NB, 3).cpu().numpy().astype(np.float64)


def sender_side(ctx, q, dense, R, seed):
    L = np.asarray(q, np.float64)
    orders = []
    for j in range(J):
        L = deposit_order(ctx, L, dense, R, seed + 17 * j)
        orders.append(L)
    return sum(orders), orders


# ------------------------------------------------------------------------------------------------ convenience
def build(ctx, lt, quad, P, Mq, q_seed, st=None):
    """Explicit table on the current geometry: returns B [N, NB, 3] numpy, orders, timing dict (state build only if st is None)."""
    import ecro86 as EC
    T_ = {}
    sync(); t0 = time.perf_counter()
    if st is None:
        st = EC.build_state(ctx)
    sync(); T_['state_s'] = time.perf_counter() - t0
    t0 = time.perf_counter()
    q = Q.qtable(ctx, lt, P, Mq, q_seed)
    sync(); T_['q_s'] = time.perf_counter() - t0
    t0 = time.perf_counter()
    op = XOp(ctx.G, st.hit, st.dbin, quad.W)
    B, orders = recurrence([op], q)
    Bn = B.cpu().numpy()
    sync(); T_['apply_s'] = time.perf_counter() - t0
    return Bn, [o.cpu().numpy() for o in orders], q, st, T_
