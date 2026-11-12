"""WL89 relations. R0 (WL87 EXP) and R1 (WL88 SAM) are imported unchanged from xport87 / rel88. New here:

C3UR  source-independent sender-uniform relation reused for all transport orders (the WL88 Task A control, now a primary candidate):
      kappa_s rays per (sender patch p, outgoing bin b) cell; origin uniform on p, direction cos-weighted inside the bin (xport87),
      front-face first hit h, incident cell c in h's patch frame (WL87 dense table, 8 x 8 cells per bin; flat patch-centre frame for sphere
      receivers, as WL87 C3 - declared). One relation for every order; geometry-only.
        L_{k+1}[h, bo] = sum_{rays r landing on h} fD_{m_h}[c_r, bo] * w_r * L_k[p_r, b_r],   w_r = A_p zc_b DOMEGA / (kappa_s A_h)
      Same discrete operator as R0 / R1 (WL88 target identity: receiver gather and sender deposition of the same piecewise-constant table).
      Per-ray random stream keyed by the global ray id (p * NB + b) * kappa_s + j (Dr.Jit PCG32 initseq offset removed), so a selective
      update (re-trace rays starting on a moved patch or whose segment meets the old / new moved volume) is bit-identical to a full rebuild.
      Application: incident radiance accumulated per (receiver, cell) and scattered per material by one contraction (same operator as
      the WL88 per-ray form, no [n_rays, NB, 3] temporary).
RAYSEND (repaired): exact once-scattered radiance at every u-ray / area-sample hit, traced from an OFFSET origin (x + n_x 1e-4); WL88 RS1
      traced area samples from the unoffset receiver point (self-intersection) and is invalid.
"""
import math, time
import numpy as np
import torch

import fx89
fx88 = fx89.fx88
PT, Q = fx89.PT, fx89.Q
mi, dr, S = PT.mi, PT.dr, PT.S
F, U = PT.F, PT.U
import rel88 as RL  # noqa: E402
import xport87 as X  # noqa: E402
NB, K, KAPPA, NZ, NPHI = S.NB, S.K, S.KAPPA, S.NZ, S.NPHI
DOMEGA = S.DOMEGA
dev = torch.device('cuda')
NC = NB * Q.SUB * Q.SUB


class SendRelR:
    """C3UR relation of one geometry (see module docstring)."""

    def __init__(self, ctx, kappa_s, seed, dense8, timing=None, chunk=2_000_000):
        T_ = timing if timing is not None else {}
        self.kappa_s, self.seed, self.chunk = kappa_s, seed, chunk
        G = ctx.G
        self.G = G
        RL.sync(); t0 = time.perf_counter()
        n = G.N * NB * kappa_s
        self.n_rays = n
        h, c, sg = self._trace(ctx, np.arange(n, dtype=np.int64))
        self.hit_t = torch.as_tensor(h.astype(np.int64), device=dev); self.cell_t = torch.as_tensor(c.astype(np.int64), device=dev)
        self.seg_t = torch.as_tensor(sg, device=dev)
        RL.sync(); T_['trace_s'] = time.perf_counter() - t0; t0 = time.perf_counter()
        self.dens = dense8
        self._finish()
        RL.sync(); T_['coeff_s'] = time.perf_counter() - t0
        self.timing = T_

    def _trace(self, ctx, rid):
        """Trace rays rid (global ids) -> hit [n] int32 (-1 none / back face), cell [n] int32, seg [n, 6] float32 (origin, end)."""
        n = len(rid)
        hit = np.empty(n, np.int32); cell = np.empty(n, np.int32); seg = np.empty((n, 6), np.float32)
        ks = self.kappa_s
        for s0 in range(0, n, self.chunk):
            sl = slice(s0, min(n, s0 + self.chunk))
            nn = sl.stop - sl.start
            ri = mi.UInt64(np.ascontiguousarray(rid[sl].astype(np.uint64)))
            rng = mi.PCG32(size=nn, initstate=dr.opaque(mi.UInt64, self.seed), initseq=ri - dr.arange(mi.UInt64, nn))
            c = U(ri // ks)
            pid = c // NB; bn = c % NB
            p, nrm, (t, bv, nz) = ctx.patch_point(pid, rng.next_float32(), rng.next_float32())
            wl = X._cosdir_in_bin(bn, rng.next_float32(), rng.next_float32())
            w = t * wl.x + bv * wl.y + nz * wl.z
            ok = dr.dot(w, nrm) > 1e-3
            o = p + nrm * 1e-4
            si = ctx.scene.ray_intersect(mi.Ray3f(o, w), ok)
            valid = ok & si.is_valid()
            sid = ctx.shape_id(si)
            front = si.to_local(-w).z > 1e-6
            ph, _ = ctx.patch_bin(si, sid, -w)
            tr, br, nr = ctx.patch_frame(ph)
            cl = Q.cell_of(dr.dot(-w, tr), dr.dot(-w, br), dr.dot(-w, nr))
            h = dr.select(valid & front, mi.Int32(ph), mi.Int32(-1))
            tt = dr.select(valid, si.t, 30.0)
            e = o + w * tt
            vals = [h, cl, o.x, o.y, o.z, e.x, e.y, e.z]
            dr.eval(vals); dr.sync_thread()
            hit[sl] = np.asarray(h); cell[sl] = np.asarray(cl).astype(np.int32)
            seg[sl] = np.stack([np.asarray(v, np.float32) for v in vals[2:]], 1)
        return hit, cell, seg

    # host views (tests, coverage, digest)
    hit = property(lambda self: self.hit_t.cpu().numpy().astype(np.int32))
    cell = property(lambda self: self.cell_t.cpu().numpy().astype(np.int32))
    seg = property(lambda self: self.seg_t.cpu().numpy())

    def _finish(self):
        """Coefficients on the GPU: landed rays, their sender cell, receiver slot and weight (same operator as the host form)."""
        G = self.G
        ks = self.kappa_s
        keep = torch.nonzero(self.hit_t >= 0)[:, 0]
        h = self.hit_t[keep]
        src = keep // ks
        p = src // NB; b = src % NB
        if not hasattr(self, 'D'):
            st, lam, alb = Q.mats()
            self.lam = lam; self.alb = alb
            self.D = {m: torch.as_tensor(np.asarray(self.dens[m], np.float32), device=dev) for m in range(6) if not lam[m]}
            self.zc_t = torch.as_tensor((np.arange(NB) // NPHI + 0.5) / NZ, dtype=torch.float64, device=dev)
        PA = torch.as_tensor(G.PAREA, dtype=torch.float64, device=dev)
        self.n_landed = int(keep.numel())
        self.src_t = src
        self.slot_t = h * NC + self.cell_t[keep]
        self.w_t = (PA[p] * self.zc_t[b] * (DOMEGA / ks) / PA[h]).float()
        self.groups = [(m, torch.as_tensor(np.nonzero(G.PMAT == m)[0], device=dev)) for m in range(6) if (G.PMAT == m).any()]
        self.rho = torch.as_tensor((self.alb[G.PMAT] / math.pi).astype(np.float32), device=dev)

    def apply(self, L):
        G = self.G
        e = L.reshape(-1, 3)[self.src_t] * self.w_t[:, None]
        E = torch.zeros(G.N * NC, 3, device=dev)
        E.index_add_(0, self.slot_t, e)
        E = E.view(G.N, NC, 3)
        out = torch.empty(G.N, NB, 3, device=dev)
        for m, ix in self.groups:
            if self.lam[m]:
                out[ix] = (E[ix].sum(1) * self.rho[ix][:, None])[:, None, :].expand(-1, NB, -1)
            else:
                out[ix] = torch.einsum('ncC,cbC->nbC', E[ix], self.D[m])
        return out

    def update(self, ctx, moved, timing=None):
        """Selective update after ctx.set_geom(new): re-trace rays starting on a moved patch or whose segment meets the old / new moved
        volume (1 mm pad, WL86 rule)."""
        import ecro86 as EC
        T_ = timing if timing is not None else {}
        Gold, Gnew = self.G, ctx.G
        RL.sync(); t0 = time.perf_counter()
        seg = self.seg_t
        d = torch.zeros(seg.shape[0], dtype=torch.bool, device=dev)
        for b in moved:
            d |= EC._seg_obb_t(seg[:, :3], seg[:, 3:], Gold.obb(b)) | EC._seg_obb_t(seg[:, :3], seg[:, 3:], Gnew.obb(b))
            sid = S.SID_BOXA if b == 'A' else S.SID_BOXB
            d |= torch.as_tensor(Gnew.PSID == sid, device=dev).repeat_interleave(NB * self.kappa_s)
        it = torch.nonzero(d)[:, 0]
        idx = it.cpu().numpy()
        RL.sync(); T_['dirty_test_s'] = time.perf_counter() - t0; t0 = time.perf_counter()
        if len(idx):
            h, c, sg = self._trace(ctx, idx)
            self.hit_t[it] = torch.as_tensor(h.astype(np.int64), device=dev); self.cell_t[it] = torch.as_tensor(c.astype(np.int64), device=dev)
            self.seg_t[it] = torch.as_tensor(sg, device=dev)
        RL.sync(); T_['retrace_s'] = time.perf_counter() - t0; t0 = time.perf_counter()
        self.G = Gnew
        self._finish()
        RL.sync(); T_['coeff_s'] = time.perf_counter() - t0; T_['dirty_rays'] = int(len(idx))
        return T_

    def nbytes(self):
        return int(self.hit_t.numel() * 4 + self.cell_t.numel() * 4 + self.seg_t.numel() * 4)       # stored state (int32 hit / cell, float32 segment)

    def digest(self):
        import hashlib
        return hashlib.sha256(self.hit.tobytes() + self.cell.tobytes()).hexdigest()[:16]


def raysend_fixed(ctx, lt, rel, st, table=None):
    """Repaired RAYSEND first gather for a SAM relation rel (m >= 0) on its u state st. Exact once-scattered radiance at every hit, traced
    from offset origins (u rays: stored origin is already p + n 1e-4; area samples: x + n_x 1e-4). table: if given, radiance is read from
    the table at the stored (hit, dbin) instead (plumbing validation: must reproduce rel.apply(table)). Returns L2 torch and the fraction
    of area-sample exact traces whose first hit is the recorded sender patch."""
    G = ctx.G

    def exact(o, d):
        n = len(o)
        if n == 0:
            return np.zeros((0, 3), np.float32), np.zeros(0, np.int64)
        ray = mi.Ray3f(mi.Point3f(*[PT.fl(o[:, k]) for k in range(3)]), mi.Vector3f(*[PT.fl(d[:, k]) for k in range(3)]))
        sd = dr.opaque(mi.UInt64, 1)
        rng = mi.PCG32(size=n, initstate=sd, initseq=mi.UInt64(0))
        out, _ = ctx.trace(ray, n, rng, sd, 'ER', 1, lt, 'cam')
        si = ctx.scene.ray_intersect(ray)
        ph, _ = ctx.patch_bin(si, ctx.shape_id(si), -ray.d)
        vals = [out[f'e1{c}'] for c in range(3)] + [ph]
        dr.eval(vals); dr.sync_thread()
        return np.stack([np.asarray(v, np.float32) for v in vals[:3]], 1), np.asarray(ph).astype(np.int64)
    idx = np.nonzero(st.hit >= 0)[0]
    Lr = np.zeros((st.hit.size, 3), np.float32)
    if table is None:
        o = st.seg[idx, :3].astype(np.float64); e = st.seg[idx, 3:].astype(np.float64)
        d = e - o; d /= np.linalg.norm(d, axis=1, keepdims=True)
        Lr[idx] = exact(o, d)[0]
    else:
        Lr[idx] = table[st.hit[idx].astype(np.int64), st.dbin[idx].astype(np.int64)]
    Lu = torch.as_tensor(Lr, device=RL.dev).view(ctx.N, K, 3)
    frac_ok = 1.0
    if rel.m:
        v = rel.A['valid'].ravel()
        sg = rel.seg_t.cpu().numpy().reshape(-1, 6)[v].astype(np.float64)
        pp = np.repeat(rel.pp, rel.m)[v]
        if table is None:
            ii = np.repeat(rel.pi, rel.m)[v]
            x, y = sg[:, :3], sg[:, 3:]
            d = y - x; d /= np.linalg.norm(d, axis=1, keepdims=True)
            La, ph = exact(x + G.PN[ii] * 1e-4, d)
            frac_ok = float((ph == pp).mean()) if len(ph) else 1.0
        else:
            La = table[pp, rel.A['dbin'].ravel()[v].astype(np.int64)].astype(np.float32)
        La = torch.as_tensor(La, device=RL.dev)
    else:
        La = None
    return rel.first_from_rays(Lu, La), frac_ok
