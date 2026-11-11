"""WL88 transport relations on the patch x bin table (learning-free; frozen ER; WL87 code imported unchanged).

R0  WL87 EXP: receiver gather on the deployed 512-ray state (8 rays per incident bin), per-ray frozen-ER weights W_ray (quad87).
R1  SAM (sender-area multiple-importance relation, the one WL88 candidate). Source-independent, built from current geometry and the
    frozen material rule only. Two proposals per flat receiver patch i, combined by the balance heuristic:
      u  receiver-uniform: the R0 state rays (kappa_u per incident bin; density n_u p_u = kappa_u / DOMEGA per steradian)
      a  sender-area: for every front-facing (flat receiver i, peaked sender patch p) pair, m samples x ~ uniform on i, y ~ stratified
         uniform on p; direction w = (y - x) / |y - x|; density n_a p_a(w | x) = m d^2 / (A_p cos_y)
    The integrand is split by first hit: the term "first hit is p" (p peaked) is estimated by both proposals, the rest by u alone:
      L_{k+1}[i, bo] = sum_{u rays r} fbar_i(w_r)[bo] cos_r L_k[hit_r, dbin_r] / D_r  +  sum_{a samples j, first hit p} fbar_i(w_j)[bo] cos_j L_k[p, dbin_j] / D_j
      D = kappa_u / DOMEGA + m d^2 / (A_p cos_y)  if the first hit is a peaked patch p (and i is flat), else kappa_u / DOMEGA
    u rays keep the WL87 per-ray weights (exact direction); a samples use a dense fbar table (16 x 16 cells per incident bin, cell centre,
    patch frame) for ER receivers and rho / pi for Lambert receivers. Curved (sphere) receivers keep the uniform proposal only
    (no area samples; declared interface limitation). With no peaked sender R1 reduces exactly to R0.
    Peaked materials: protocol relation.peaked_rule (material response only).
Diagnostic controls for the WL87 C3 / EXP difference (Task A):
  C3     WL87 sender-side deposition (power-proportional, fresh relation per order, source-dependent) - xport87.sender_side
  C3U    sender-side, kappa_s rays per (patch, outgoing bin) cell (uniform allocation), fresh relation per order (source-independent)
  C3UR   as C3U but ONE relation reused for all orders (geometry-only sender relation; same lifecycle as R0)
  R0D    R0 rays, scattering weights from the WL87 dense table (8 x 8 cells per bin) at the ray's cell (flat receivers)
  R0F    R0 with an independent state realization per order (fresh relation per order)
  POOLk  R0 averaged over k independent realizations (k x rays)
"""
import math, time
import numpy as np
import torch

import fx88
PT, Q = fx88.PT, fx88.Q
mi, dr, S, CD = PT.mi, PT.dr, PT.S, PT.CD
F, U = PT.F, PT.U
import xport87 as X  # noqa: E402
NB, K, KAPPA, NZ, NPHI = S.NB, S.K, S.KAPPA, S.NZ, S.NPHI
DOMEGA = S.DOMEGA
TWO_PI = 2 * math.pi
dev = torch.device('cuda')
J = 4


def sync():
    dr.sync_thread(); torch.cuda.synchronize()


_PEAK = {}


def peaked():
    """bool[6] per material (cached per family): median_w max_bo fbar(w)[bo] / (a(w) / pi) > threshold (protocol)."""
    fam = fx88.FAMILY['name']
    if fam not in _PEAK:
        st, lam, alb = Q.mats()
        R = fx88.P88['relation']['peaked_rule']
        dirs = S.bin_dirs()
        zc = (np.arange(NB) // NPHI + 0.5) / NZ
        pis = []
        for m in range(6):
            if lam[m]:
                pis.append(1.0)
                continue
            fb = Q.fbar(st[m], dirs, R['M'], R['seed'] + m).mean(-1)
            a = (fb * DOMEGA * zc[None]).sum(1)
            pis.append(float(np.median(fb.max(1) / np.maximum(a / math.pi, 1e-12))))
        _PEAK[fam] = (np.array([p > R['threshold'] for p in pis]), pis)
    return _PEAK[fam]


# ------------------------------------------------------------------------------------------------ dense fbar (SUB x SUB cells per bin)
def cell_dirs(sub):
    c = np.arange(NB * sub * sub)
    bn, rem = c // (sub * sub), c % (sub * sub)
    sz, sp = rem // sub, rem % sub
    z = (bn // NPHI + (sz + 0.5) / sub) / NZ
    phi = (bn % NPHI + (sp + 0.5) / sub) * (TWO_PI / NPHI) - math.pi
    r = np.sqrt(np.maximum(1 - z * z, 0))
    return np.stack([r * np.cos(phi), r * np.sin(phi), z], 1)


def cell_of(x, y, z, sub):
    zz = dr.clamp(z, 0.0, 0.999999) * NZ
    pp = dr.clamp((dr.atan2(y, x) + math.pi) * (NPHI / TWO_PI), 0.0, NPHI - 1e-6)
    iz = U(dr.floor(zz)); ip = U(dr.floor(pp))
    sz = U(dr.clamp(dr.floor((zz - F(iz)) * sub), 0, sub - 1)); sp = U(dr.clamp(dr.floor((pp - F(ip)) * sub), 0, sub - 1))
    return (iz * NPHI + ip) * (sub * sub) + sz * sub + sp


class Dense:
    """fbar on dense incident cells per material, torch [ncell, NB, 3] on the GPU (Lambert: rho / pi). Material-only (cached)."""

    def __init__(self, sub, M, seed):
        st, lam, alb = Q.mats()
        self.sub = sub
        cd = cell_dirs(sub)
        self.T = []
        cache = {}
        for m in range(6):
            if lam[m]:
                self.T.append(torch.full((len(cd), NB, 3), float(alb[m] / math.pi), device=dev))
            else:
                key = st[m].tobytes()
                if key not in cache:
                    cache[key] = torch.as_tensor(Q.fbar(st[m], cd, M, seed + 10 * m).astype(np.float32), device=dev)
                self.T.append(cache[key])
        self.lam = lam; self.alb = alb


# ------------------------------------------------------------------------------------------------ uniform part with per-ray weights
class XOpW(X.XOp):
    """WL87 XOp with a per-ray multiplier mr [N, K] (MIS factor, kappa subset)."""

    def __init__(self, G, hit, dbin, W, mr=None):
        super().__init__(G, hit, dbin, W)
        if mr is not None:
            mr = mr if isinstance(mr, torch.Tensor) else torch.as_tensor(np.asarray(mr, np.float32), device=dev)
            self.ok = self.ok * mr.view(G.N, K, 1)


def kappa_mask(kappa_u):
    """[K] multiplier for a kappa_u-per-bin subset of the deployed state (first kappa_u rays of each bin), reweighted by KAPPA / kappa_u."""
    k = np.arange(K) % KAPPA
    return np.where(k < kappa_u, KAPPA / kappa_u, 0.0)


# ------------------------------------------------------------------------------------------------ R1: pairs and area samples
def patch_corners(G, sph_grid=3):
    """[N, 9, 3] points per patch: rectangle 4 corners (+ repeated), sphere patch a 3 x 3 grid; normals [N, 9, 3]."""
    N = G.N
    P = np.zeros((N, 9, 3)); Nn = np.zeros((N, 9, 3))
    rect = G.PSID != S.SID_SPHERE
    ab = np.array([[a, b] for a in (0.0, 0.5, 1.0) for b in (0.0, 0.5, 1.0)])
    P[rect] = G.PO[rect][:, None] + ab[None, :, :1] * G.PE1[rect][:, None] + ab[None, :, 1:] * G.PE2[rect][:, None]
    Nn[rect] = G.PN[rect][:, None]
    sph = np.nonzero(~rect)[0]
    for t, (a, b) in enumerate(ab):
        z = (G.PSPH[sph, 1] + b) * (2.0 / S.SPH_NZ) - 1.0
        ph = (G.PSPH[sph, 0] + a) * (TWO_PI / S.SPH_NPHI) - math.pi
        r = np.sqrt(np.maximum(1 - z * z, 0))
        ns = np.stack([r * np.cos(ph), r * np.sin(ph), z], 1)
        P[sph, t] = G.SC + G.SR * ns; Nn[sph, t] = ns
    return P, Nn


def _facing(G, recv, send, P):
    """Front-facing test for every (recv, send) combination. Conservative (exact for flat-flat): keep if max over sender points y of
    n_i . (y - c_i) > 1e-6 and (flat sender) max over receiver points x of n_p . (x - c_p) > 1e-6; i != p. Order: (recv, send) ascending."""
    if len(recv) == 0 or len(send) == 0:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    ci = G.PC[recv]; ni = G.PN[recv]
    s1 = (np.einsum('ik,spk->isp', ni, P[send]) - (ni * ci).sum(1)[:, None, None]).max(2) > 1e-6
    flat_s = G.PSID[send] != S.SID_SPHERE
    np_ = G.PN[send]; cp = G.PC[send]
    s2 = (np.einsum('sk,ipk->isp', np_, P[recv]) - (np_ * cp).sum(1)[None, :, None]).max(2) > 1e-6
    keep = s1 & (s2 | ~flat_s[None, :]) & (recv[:, None] != send[None, :])
    ii, ss = np.nonzero(keep)
    return recv[ii].astype(np.int64), send[ss].astype(np.int64)


def pair_list(G, pk_mat):
    """Front-facing (flat receiver, peaked sender) pairs, canonical order (i, p)."""
    P, _ = patch_corners(G)
    return _facing(G, np.nonzero(G.PSID != S.SID_SPHERE)[0], np.nonzero(pk_mat[G.PMAT])[0], P)


def pair_list_update(G, pk_mat, pi_old, pp_old, mov):
    """Same list as pair_list(G) after a rigid edit of the patches `mov`: pairs between unmoved patches keep their facing status."""
    P, _ = patch_corners(G)
    recv = np.nonzero(G.PSID != S.SID_SPHERE)[0]; send = np.nonzero(pk_mat[G.PMAT])[0]
    keep = ~mov[pi_old] & ~mov[pp_old]
    a = _facing(G, recv[mov[recv]], send, P)
    b = _facing(G, recv[~mov[recv]], send[mov[send]], P)
    pi = np.concatenate([pi_old[keep], a[0], b[0]]); pp = np.concatenate([pp_old[keep], a[1], b[1]])
    o = np.argsort(pi * G.N + pp, kind='stable')
    return pi[o], pp[o]


def _pcg(seed, key):
    """One PCG32 stream per sample key, independent of lane order (Dr.Jit adds arange(size) to initseq; subtract it)."""
    n = dr.width(key)
    return mi.PCG32(size=n, initstate=dr.opaque(mi.UInt64, seed), initseq=key - dr.arange(mi.UInt64, n))


def trace_area(ctx, pi, pp, m, kappa_u, seed):
    """Area samples for pairs (pi, pp) [n] numpy: per pair m samples; deterministic per (i, p, j) (stream key = (i * N + p) * 64 + j).
    Returns dict of numpy arrays [n, m]: valid, dbin, cell16, w (= cos_x / D), seg (x, y)."""
    G = ctx.G
    n = len(pi)
    out = {'valid': np.zeros((n, m), bool), 'dbin': np.zeros((n, m), np.uint8), 'cell': np.zeros((n, m), np.int32),
           'w': np.zeros((n, m), np.float32), 'seg': np.zeros((n, m, 6), np.float32)}
    if n == 0:
        return out
    sub = fx88.P88['relation']['dense_sub']
    ctx._pa = PT.fl(G.PAREA)
    gs = int(round(math.sqrt(m)))
    assert gs * gs == m
    chunk = max(1, 2_000_000 // m)
    for s0 in range(0, n, chunk):
        sl = slice(s0, min(n, s0 + chunk))
        nn = sl.stop - sl.start
        lanes = nn * m
        li = dr.arange(U, lanes)
        pr = li // m; j = li % m
        ri = U(np.ascontiguousarray(pi[sl].astype(np.uint32))); si_ = U(np.ascontiguousarray(pp[sl].astype(np.uint32)))
        ig = dr.gather(U, ri, pr); pg = dr.gather(U, si_, pr)
        key = (mi.UInt64(ig) * mi.UInt64(G.N) + mi.UInt64(pg)) * mi.UInt64(64) + mi.UInt64(j)
        rng = _pcg(seed, key)
        x, nx, (ti, bi, ni) = ctx.patch_point(ig, rng.next_float32(), rng.next_float32())
        a = (F(j % gs) + rng.next_float32()) * (1.0 / gs); b = (F(j // gs) + rng.next_float32()) * (1.0 / gs)
        y, ny, _ = ctx.patch_point(pg, a, b)
        d = y - x
        dist = dr.norm(d)
        w = d * dr.rcp(dist)
        cx = dr.dot(nx, w); cy = -dr.dot(ny, w)
        ok = (cx > 1e-3) & (cy > 1e-6)
        sh = ctx.scene.ray_intersect(mi.Ray3f(x + nx * 1e-4, w), ok)
        sid = ctx.shape_id(sh)
        front = sh.to_local(-w).z > 1e-6
        ph, bh = ctx.patch_bin(sh, sid, -w)
        vis = ok & sh.is_valid() & front & dr.eq(ph, pg)
        area = dr.gather(F, ctx._pa, pg)
        D = kappa_u / DOMEGA + m * dist * dist * dr.rcp(area * dr.maximum(cy, 1e-12))
        wt = dr.select(vis, cx * dr.rcp(D), 0.0)
        cl = cell_of(dr.dot(w, ti), dr.dot(w, bi), dr.dot(w, ni), sub)
        vals = [vis, bh, cl, wt, x.x, x.y, x.z, y.x, y.y, y.z]
        dr.eval(vals); dr.sync_thread()
        g = lambda v, t: np.asarray(v).astype(t).reshape(nn, m)
        out['valid'][sl] = g(vis, bool); out['dbin'][sl] = g(bh, np.uint8); out['cell'][sl] = g(cl, np.int32); out['w'][sl] = g(wt, np.float32)
        out['seg'][sl] = np.stack([np.asarray(v, np.float32).reshape(nn, m) for v in vals[4:]], -1)
    return out


class GeoT:
    """Per-geometry patch tables on the GPU (normals, sphere flag, areas, peaked flag)."""

    def __init__(self, G, pk_mat):
        t = lambda a, dt=torch.float32: torch.as_tensor(np.asarray(a), dtype=dt, device=dev)
        self.PN = t(G.PN); self.sph = t(G.PSID == S.SID_SPHERE, torch.bool); self.PA = t(G.PAREA); self.pk = t(pk_mat[G.PMAT], torch.bool)
        self.flat_recv = (~self.sph).repeat_interleave(K)
        self.SC = t(G.SC); self.SR = float(G.SR); self.N = G.N


def uniform_mis(gt, st, keys_t, kappa_u, m):
    """Per-ray multiplier [N * K] (torch) for the u proposal: kappa subset x (KAPPA / kappa_u) x MIS factor (kappa_u / DOMEGA) / D_r.
    Returns the multiplier and the number of u hits on a peaked sender from a flat receiver outside the pair list (must be 0)."""
    N = gt.N
    mr = torch.as_tensor(np.tile(kappa_mask(kappa_u), N).astype(np.float32), device=dev)
    if m == 0:
        return mr, 0
    hit = torch.as_tensor(st.hit.astype(np.int64), device=dev)
    hp = (hit >= 0) & gt.flat_recv & (mr > 0)
    hc = hit.clamp(min=0)
    hp &= gt.pk[hc]
    idx = torch.nonzero(hp)[:, 0]
    if idx.numel() == 0:
        return mr, 0
    seg = torch.as_tensor(st.seg, device=dev)[idx].double()
    o, e = seg[:, :3], seg[:, 3:]
    d = e - o; dist = d.norm(dim=1); w = d / dist[:, None]
    p = hc[idx]
    ny = gt.PN[p].double()
    sp = gt.sph[p]
    ny[sp] = (e[sp] - gt.SC.double()) / gt.SR
    cy = (-(w * ny).sum(1)).clamp(min=1e-12)
    D = kappa_u / DOMEGA + m * dist ** 2 / (gt.PA[p].double() * cy)
    mr[idx] = (mr[idx].double() * (kappa_u / DOMEGA) / D).float()
    keys = (idx // K) * N + p
    missing = int((~torch.isin(keys, keys_t)).sum().item())
    return mr, missing


class SAM:
    """R1 relation of one geometry. Build: SAM(ctx, quad, kappa_u, m, seed, st=None). Apply: .apply(L torch [N, NB, 3])."""

    def __init__(self, ctx, quad, dense, kappa_u, m, seed, st=None, timing=None):
        import ecro86 as EC
        T_ = timing if timing is not None else {}
        G = ctx.G
        self.G, self.quad, self.dense, self.kappa_u, self.m, self.seed = G, quad, dense, kappa_u, m, seed
        ctx._pa = PT.fl(G.PAREA)
        pk, _ = peaked()
        self.pk = pk
        sync(); t0 = time.perf_counter()
        self.st = st if st is not None else EC.build_state(ctx)
        sync(); T_['u_state_s'] = time.perf_counter() - t0; t0 = time.perf_counter()
        self.pi, self.pp = pair_list(G, pk) if m > 0 else (np.zeros(0, np.int64), np.zeros(0, np.int64))
        sync(); T_['pairs_s'] = time.perf_counter() - t0; t0 = time.perf_counter()
        self.A = trace_area(ctx, self.pi, self.pp, m, kappa_u, seed) if m > 0 else None
        if m > 0:
            self.seg_t = torch.as_tensor(self.A.pop('seg'), device=dev)
        sync(); T_['area_trace_s'] = time.perf_counter() - t0; t0 = time.perf_counter()
        self._finish()
        sync(); T_['coeff_s'] = time.perf_counter() - t0
        self.timing = T_

    def _finish(self):
        G = self.G
        self.gt = GeoT(G, self.pk)
        self.keys = self.pi * G.N + self.pp
        mr, self.missing = uniform_mis(self.gt, self.st, torch.as_tensor(self.keys, device=dev), self.kappa_u, self.m)
        self.mr = mr
        self.uop = XOpW(G, self.st.hit, self.st.dbin, self.quad.W if isinstance(self.quad, Q.Quad) else self.quad, mr)
        if self.m > 0:
            v = self.A['valid'].ravel()
            ii = np.repeat(self.pi, self.m)[v]; pp = np.repeat(self.pp, self.m)[v]
            self.n_area_valid = int(v.sum())
            self.a_recv = torch.as_tensor(ii, device=dev)
            self.a_idx = torch.as_tensor(pp * NB + self.A['dbin'].ravel()[v].astype(np.int64), device=dev)
            self.a_w = torch.as_tensor(self.A['w'].ravel()[v], device=dev)
            mat = G.PMAT[ii]
            er = ~self.dense.lam[mat]
            self.a_er = [(mm, torch.as_tensor(np.nonzero(er & (mat == mm))[0], device=dev)) for mm in range(6) if (er & (mat == mm)).any()]
            self.a_cell = torch.as_tensor(self.A['cell'].ravel()[v].astype(np.int64), device=dev)
            lam_sel = np.nonzero(~er)[0]
            self.a_lam = torch.as_tensor(lam_sel, device=dev)
            self.lam_rho = torch.as_tensor((self.dense.alb[G.PMAT] / math.pi).astype(np.float32), device=dev)
        else:
            self.n_area_valid = 0

    def rays(self):
        """Traced rays of this relation (u subset + area samples incl. invalid ones; the u state traces all K, counted at K)."""
        return int(self.G.N * K * self.kappa_u // KAPPA + (len(self.pi) * self.m if self.m else 0))

    def nbytes(self):
        b = self.st.hit.nbytes + self.st.dbin.nbytes + self.st.seg.nbytes + self.mr.numel() * 4
        if self.m:
            b += sum(v.nbytes for v in self.A.values()) + self.pi.nbytes + self.pp.nbytes + self.seg_t.numel() * 4
        return int(b)

    def apply(self, L):
        out = self.uop.apply(L)
        if self.m == 0:
            return out
        e = L.reshape(-1, 3)[self.a_idx] * self.a_w[:, None]                       # [n, 3]
        if len(self.a_lam):
            s = torch.zeros(self.G.N, 3, device=dev)
            s.index_add_(0, self.a_recv[self.a_lam], e[self.a_lam])
            out = out + (s * self.lam_rho[:, None])[:, None, :]
        for mm, ix in self.a_er:
            c = self.dense.T[mm][self.a_cell[ix]] * e[ix][:, None, :]
            out.index_add_(0, self.a_recv[ix], c)
        return out

    def first_from_rays(self, Lr_u, Lr_a):
        """First gather from per-sample incident radiance (RAYSEND): Lr_u [N, K, 3] (u rays), Lr_a [n_valid, 3] (area samples)."""
        out = self.uop.scatter(Lr_u * self.uop.ok)
        if self.m == 0:
            return out
        e = Lr_a * self.a_w[:, None]
        if len(self.a_lam):
            s = torch.zeros(self.G.N, 3, device=dev)
            s.index_add_(0, self.a_recv[self.a_lam], e[self.a_lam])
            out = out + (s * self.lam_rho[:, None])[:, None, :]
        for mm, ix in self.a_er:
            out.index_add_(0, self.a_recv[ix], self.dense.T[mm][self.a_cell[ix]] * e[ix][:, None, :])
        return out

    def update(self, ctx, moved, timing=None):
        """Selective update after ctx.set_geom(new): u rays re-traced where dirty (WL86 rule), pair list rebuilt (cheap geometry), area
        samples re-traced for new pairs, pairs touching a moved patch and samples whose segment meets the old / new moved volume."""
        import ecro86 as EC
        T_ = timing if timing is not None else {}
        Gold, Gnew = self.G, ctx.G
        ctx._pa = PT.fl(Gnew.PAREA)
        sync(); t0 = time.perf_counter()
        st2, nd = EC.update_state(ctx, self.st, moved)
        sync(); T_['u_update_s'] = time.perf_counter() - t0; T_['u_dirty'] = int(nd); t0 = time.perf_counter()
        n_dirty = 0
        if self.m > 0:
            mov = np.zeros(Gnew.N, bool)
            for b in moved:
                mov |= Gnew.PSID == (S.SID_BOXA if b == 'A' else S.SID_BOXB)
            pi, pp = pair_list_update(Gnew, self.pk, self.pi, self.pp, mov)
            keys = pi * Gnew.N + pp
            pos = np.searchsorted(self.keys, keys)
            pos = np.clip(pos, 0, max(len(self.keys) - 1, 0))
            old = (len(self.keys) > 0) & (self.keys[pos] == keys) & ~mov[pi] & ~mov[pp]
            oi = np.nonzero(old)[0]
            pos_t = torch.as_tensor(pos[oi], device=dev)
            seg = self.seg_t[pos_t]                                                       # [n_old, m, 6] (GPU-resident)
            sg = seg.view(-1, 6)
            dseg = torch.zeros(sg.shape[0], dtype=torch.bool, device=dev)
            for b in moved:
                dseg |= EC._seg_obb_t(sg[:, :3], sg[:, 3:], Gold.obb(b)) | EC._seg_obb_t(sg[:, :3], sg[:, 3:], Gnew.obb(b))
            dpair = np.zeros(len(pi), bool)
            dpair[oi] = dseg.view(-1, self.m).any(1).cpu().numpy()
            redo = ~old | dpair
            sync(); T_['dirty_test_s'] = time.perf_counter() - t0; t0 = time.perf_counter()
            Rn = trace_area(ctx, pi[redo], pp[redo], self.m, self.kappa_u, self.seed)
            A = {k: np.empty((len(pi),) + v.shape[1:], v.dtype) for k, v in self.A.items()}
            for k in A:
                A[k][oi] = self.A[k][pos[oi]]
                A[k][redo] = Rn[k]
            seg_t = torch.empty((len(pi), self.m, 6), dtype=torch.float32, device=dev)
            seg_t[torch.as_tensor(oi, device=dev)] = seg
            seg_t[torch.as_tensor(np.nonzero(redo)[0], device=dev)] = torch.as_tensor(Rn['seg'], device=dev)
            n_dirty = int(redo.sum()) * self.m
            self.pi, self.pp, self.A, self.seg_t = pi, pp, A, seg_t
            sync(); T_['area_retrace_s'] = time.perf_counter() - t0; t0 = time.perf_counter()
        self.G, self.st = Gnew, st2
        self._finish()
        sync(); T_['coeff_s'] = time.perf_counter() - t0; T_['area_dirty_samples'] = n_dirty
        return T_

    def digest(self):
        import hashlib
        h = hashlib.sha256(self.st.hit.tobytes() + self.st.dbin.tobytes() + self.mr.cpu().numpy().astype(np.float32).tobytes())
        if self.m:
            for k in ('valid', 'dbin', 'cell', 'w'):
                h.update(np.ascontiguousarray(self.A[k]).tobytes())
            h.update(self.pi.tobytes() + self.pp.tobytes())
        return h.hexdigest()[:16]


def recurrence(ops, q, first=None, J_=J):
    """B = sum_{k=2}^{1+J} L_k, L_{k+1} = mean over ops of op.apply(L_k) (ops: SAM or XOp). Returns B, orders (torch)."""
    L = torch.as_tensor(np.asarray(q, np.float32), device=dev)
    orders = []
    for j in range(J_):
        if j == 0 and first is not None:
            L = torch.as_tensor(np.asarray(first, np.float32), device=dev) if not isinstance(first, torch.Tensor) else first
        else:
            L = sum(op.apply(L) for op in ops) / len(ops)
        orders.append(L)
    return sum(orders), orders


def fresh_recurrence(ops, q):
    """R0F: order j uses ops[j] (independent realization per order)."""
    L = torch.as_tensor(np.asarray(q, np.float32), device=dev)
    orders = []
    for j in range(J):
        L = ops[j].apply(L)
        orders.append(L)
    return sum(orders), orders


def dense_W(quad, dense8):
    """R0D: W rows 0..5 from the WL87 dense table (8 x 8 cells per bin, cell centre) at each state ray's cell; sphere rows unchanged."""
    cd = Q.cell_dirs()
    dirs = quad.dirs
    zz = np.clip(dirs[:, 2], 0, 0.999999) * NZ
    pp = np.clip((np.arctan2(dirs[:, 1], dirs[:, 0]) + math.pi) * (NPHI / TWO_PI), 0, NPHI - 1e-6)
    iz = np.floor(zz).astype(int); ip = np.floor(pp).astype(int)
    sz = np.clip(np.floor((zz - iz) * Q.SUB), 0, Q.SUB - 1).astype(int); sp = np.clip(np.floor((pp - ip) * Q.SUB), 0, Q.SUB - 1).astype(int)
    cell = (iz * NPHI + ip) * Q.SUB * Q.SUB + sz * Q.SUB + sp
    W = quad.W.copy()
    w = dirs[:, 2] * DOMEGA / KAPPA
    for m in range(6):
        W[m] = dense8[m][cell] * w[:, None, None]
    return W


# ------------------------------------------------------------------------------------------------ sender-side uniform relation (C3U / C3UR)
class SendRel:
    """kappa_s rays per (sender patch, outgoing bin) cell, cos-weighted in the bin, deposited at the receiver's WL87 dense cell (8 x 8 per
    bin, flat patch frame as WL87 C3). Geometry-only. apply(L) = next table."""

    def __init__(self, ctx, kappa_s, seed, dense8, chunk=2_000_000):
        G = ctx.G
        self.G = G
        n = G.N * NB * kappa_s
        hs, cs, ws, srcs = [], [], [], []
        zc = (np.arange(NB) // NPHI + 0.5) / NZ
        for s0 in range(0, n, chunk):
            nn = min(chunk, n - s0)
            li = dr.arange(U, nn) + s0
            cell = li // kappa_s
            pid = cell // NB; bn = cell % NB
            rng = mi.PCG32(size=nn, initstate=dr.opaque(mi.UInt64, seed), initseq=mi.UInt64(li))
            p, nnrm, (t, bv, nz) = ctx.patch_point(pid, rng.next_float32(), rng.next_float32())
            wl = X._cosdir_in_bin(bn, rng.next_float32(), rng.next_float32())
            w = t * wl.x + bv * wl.y + nz * wl.z
            ok = dr.dot(w, nnrm) > 1e-3
            si = ctx.scene.ray_intersect(mi.Ray3f(p + nnrm * 1e-4, w), ok)
            valid = ok & si.is_valid()
            sid = ctx.shape_id(si)
            front = si.to_local(-w).z > 1e-6
            ph, _ = ctx.patch_bin(si, sid, -w)
            tr, br, nr = ctx.patch_frame(ph)
            cl = Q.cell_of(dr.dot(-w, tr), dr.dot(-w, br), dr.dot(-w, nr))
            hit = dr.select(valid & front, mi.Int32(ph), mi.Int32(-1))
            dr.eval(hit, cl, cell); dr.sync_thread()
            h = np.asarray(hit).astype(np.int64); keep = h >= 0
            c = np.asarray(cell).astype(np.int64)
            hs.append(h[keep]); cs.append(np.asarray(cl).astype(np.int64)[keep]); srcs.append(c[keep])
        h = np.concatenate(hs); cl = np.concatenate(cs); src = np.concatenate(srcs)
        p = src // NB; b = src % NB
        w = G.PAREA[p] * zc[b] * DOMEGA / kappa_s / G.PAREA[h]
        self.n_rays = n
        self.h = torch.as_tensor(h, device=dev); self.src = torch.as_tensor(src, device=dev)
        self.w = torch.as_tensor(w.astype(np.float32), device=dev); self.cl = torch.as_tensor(cl, device=dev)
        self.dens = [torch.as_tensor(np.asarray(dense8[m], np.float32), device=dev) for m in range(6)]
        pm = G.PMAT[h]
        self.groups = [(m, torch.as_tensor(np.nonzero(pm == m)[0], device=dev)) for m in range(6) if (pm == m).any()]

    def apply(self, L):
        e = L.reshape(-1, 3)[self.src] * self.w[:, None]
        out = torch.zeros(self.G.N, NB, 3, device=dev)
        for m, ix in self.groups:
            out.index_add_(0, self.h[ix], self.dens[m][self.cl[ix]] * e[ix][:, None, :])
        return out
