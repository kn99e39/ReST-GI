"""WL91 isolated C3URX1 candidate path (learning-free; frozen ER; WL86-WL90 code imported unchanged).

C3URX1 = the WL89 source-independent sender-uniform relation C3UR (rel89.SendRelR, unchanged: kappa_s rays per (sender patch, outgoing
bin) cell, reused for all orders, selective update bit-identical to a rebuild) with the WL90 X1 first gather:
   L_2[h, bo] = sum over landed relation rays r of  fD_{m_h}[cell_r, bo] * w_r * Lx_1(y_r, v_r)
   Lx_1(y, v) = exact once-scattered radiance leaving the ray's own origin point y in its own direction v under the CURRENT lights
                (one deterministic next-event evaluation per landed ray and light: 1 probe intersection + 1 shadow ray per light)
   L_3..L_5   = the unchanged C3UR table recurrence applied to L_2;   B = L_2 + L_3 + L_4 + L_5 (read by the unchanged LOW + lookup renderer)
No q table is used. Nothing source-dependent is stored in the relation; the probe-ray buffer is geometry-dependent only and is rebuilt
(GPU-resident) whenever the relation is built or updated. Every table() call evaluates the lights it is given: there is no cached
source state (a geometry stamp guards against using a probe buffer of another geometry).

The class is a consolidation of the WL90 diagnostic functions gap90.c3ur_raybuf / RayBuf.radiance / c3ur_first / rec_from; tests91 checks
it against those frozen reference functions and against the stored WL90 tables.
"""
import math, time
import numpy as np
import torch

import fx91
PT, Q = fx91.PT, fx91.Q
mi, dr, S = PT.mi, PT.dr, PT.S
F, U = PT.F, PT.U
import rel88 as RL  # noqa: E402
import rel89 as R9  # noqa: E402
import gap90 as GP  # noqa: E402  (trace_orders: the pt86 estimator split by event count; frozen WL90 code)
NB, K = S.NB, S.K
dev = torch.device('cuda')
J = 4


def sync():
    dr.sync_thread(); torch.cuda.synchronize()


class C3URX1:
    """Candidate of one geometry. build: C3URX1(ctx, kappa_s, seed, dense8). Source update: table(ctx, lights). Edit: update(ctx, moved)."""

    def __init__(self, ctx, kappa_s, seed, dense8, timing=None):
        T_ = timing if timing is not None else {}
        sync(); t0 = time.perf_counter()
        self.rel = R9.SendRelR(ctx, kappa_s, seed, dense8)
        sync(); T_['relation_build_s'] = time.perf_counter() - t0; t0 = time.perf_counter()
        self._raybuf(ctx)
        sync(); T_['probe_buffer_s'] = time.perf_counter() - t0
        self.counts = {'unique_relation_rays': int(self.rel.n_rays), 'relation_rays_retraced_last_edit': 0}

    def _raybuf(self, ctx):
        rel = self.rel
        keep = torch.nonzero(rel.hit_t >= 0)[:, 0]
        sg = rel.seg_t[keep]
        v = sg[:, 3:] - sg[:, :3]
        v = v / v.norm(dim=1, keepdim=True)
        p = sg[:, :3] + v * 2e-3
        d = -v
        col = lambda t, k: F(t[:, k].contiguous())
        self.o = mi.Point3f(col(p, 0), col(p, 1), col(p, 2)); self.d = mi.Vector3f(col(d, 0), col(d, 1), col(d, 2))
        dr.eval(self.o, self.d)
        self.n = int(keep.numel())
        self.geom_stamp = (rel.n_landed, id(rel.slot_t))          # rel._finish() creates new coefficient tensors on every build / update

    def update(self, ctx, moved, timing=None):
        T_ = timing if timing is not None else {}
        sync(); t0 = time.perf_counter()
        Tr = self.rel.update(ctx, moved)
        sync(); T_['relation_update_s'] = time.perf_counter() - t0; t0 = time.perf_counter()
        self._raybuf(ctx)
        sync(); T_['probe_buffer_s'] = time.perf_counter() - t0
        self.counts['relation_rays_retraced_last_edit'] = int(Tr['dirty_rays'])
        T_['dirty_rays'] = int(Tr['dirty_rays'])
        return T_

    def radiance(self, ctx, lt):
        """Exact once-scattered radiance per landed relation ray under lights lt -> torch [n, 3] (no cache)."""
        assert self.geom_stamp == (self.rel.n_landed, id(self.rel.slot_t)), 'probe buffer does not belong to the current relation'
        rng = mi.PCG32(size=self.n, initstate=dr.opaque(mi.UInt64, 1), initseq=mi.UInt64(0))
        cs, _ = GP.trace_orders(ctx, mi.Ray3f(self.o, self.d), lt, rng, 1)
        c = cs[0]
        dr.eval(c); dr.sync_thread()
        return torch.stack([c[k].torch() for k in range(3)], 1)

    def first_gather(self, rad):
        rc = self.rel
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

    def table(self, ctx, lt, orders=False, J_=J, timing=None):
        """B (torch [N, NB, 3]) for the current geometry and the given lights; orders=True also returns [L2..L5]."""
        T_ = timing if timing is not None else {}
        sync(); t0 = time.perf_counter()
        rad = self.radiance(ctx, lt)
        sync(); T_['source_eval_s'] = time.perf_counter() - t0; t0 = time.perf_counter()
        L = self.first_gather(rad)
        ords = [L]
        for _ in range(J_ - 1):
            L = self.rel.apply(L); ords.append(L)
        B = sum(ords)
        sync(); T_['transport_s'] = time.perf_counter() - t0
        T_['source_eval_rays'] = {'probe_intersections': self.n, 'shadow_rays': self.n * lt.n}
        return (B, ords) if orders else B

    def nbytes(self):
        """Persistent state: relation (hit, cell, segment) + probe buffer (origin, direction); scratch: per-ray radiance + incident-cell buffer."""
        return {'relation': self.rel.nbytes(), 'probe_buffer': self.n * 6 * 4, 'scratch_radiance': self.n * 3 * 4,
                'scratch_incident_cells': self.rel.G.N * R9.NC * 3 * 4}


def c3ur_table(rc, q, orders=False, J_=J):
    """The preserved C3UR table recurrence (rel88.recurrence semantics) on torch."""
    L = torch.as_tensor(np.asarray(q, np.float32), device=dev)
    ords = []
    for _ in range(J_):
        L = rc.apply(L); ords.append(L)
    B = sum(ords)
    return (B, ords) if orders else B
