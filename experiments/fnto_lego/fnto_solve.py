"""FNTO composition: current-geometry free-space coupling C x fixed learned kernels K, multi-bounce state solve, render.

    i_a = e_a + sum_b C_ab(R) b_b,     b_a = K_a[i_a]                        (contract §2; a = b allowed)

State (configuration-dependent, never learned):
  * relation table H [Np, K] of the current geometry (fnto_relations), port sun visibility V [Np] for the view's light;
  * per port incident radiance samples L [Np, K, 3] along its K gather directions (free flight from the hit port);
  * per domain source coefficients gamma (= the compressed incoming boundary state of K_glob).
Kernel tables (persistent, computed once from the frozen K parameters): per port BSDF parameters, p, q, projected
features and the band weights W(port, neighbour, w_o = port normal).

Sender radiance toward a receiver (the compressed outgoing boundary state used by C):
  b_s(w) = I_s + G_s(w),
  I_s  = rho_s/pi E_s + sum_l kappa_l(s) E_ind(s)/pi + sum_m p_m(s) gamma_m + sum_{s'} A_s' sum_b W_b(s, s') h_b(s')
         + sum_c A_c Wm(s, c) E_c(c)
         (isotropic part: diffuse response; glossy reflection of INDIRECT light folded in isotropically with lobe
         albedo 1; band term evaluated at w_o = n_s -- declared sender-side approximations)
  G_s(w) = sum_l kappa_l(s) GGX(w_L, w; alpha_l(s)) E_sun cos V      (the delta sun's glossy reflection, exact; cached per
         view because it does not change across bounces)
Camera pixels use the full per-direction lobes of K_loc on the solved incident samples instead.
Jacobi iteration on L until the relative change of the outgoing energy < TOL (declared), at most MAX_IT iterations.
"""

from __future__ import annotations

import math
import time

import numpy as np
import torch

import fnto_common as C
import fnto_model as M

TOL = 1e-3
MAX_IT = 100
SH_C = (0.282095, 0.488603, 1.092548, 0.315392, 0.546274)


def sh9_t(d):
    x, y, z = d[..., 0], d[..., 1], d[..., 2]
    c0, c1, c2, c3, c4 = SH_C
    return torch.stack([c0 * torch.ones_like(x), c1 * y, c1 * z, c1 * x, c2 * x * y, c2 * y * z, c3 * (3 * z * z - 1), c2 * x * z,
                        c4 * (x * x - y * y)], -1)


def frames_t(n, rot):
    a = torch.where(n[:, 2:3].abs() < 0.9, n.new_tensor([0.0, 0, 1]).expand_as(n), n.new_tensor([1.0, 0, 0]).expand_as(n))
    t = torch.nn.functional.normalize(torch.cross(a, n, dim=1), dim=1)
    b = torch.cross(n, t, dim=1)
    c, s = torch.cos(rot)[:, None], torch.sin(rot)[:, None]
    return c * t + s * b, -s * t + c * b


class Tables:
    """Persistent per-port kernel tables (no gradient) + configuration-invariant gather geometry."""

    def __init__(self, KA, KB, dev, chunk=1 << 15, anchors=None):
        A = np.load(C.OUT / "anchors.npz") if anchors is None else anchors
        T = lambda x, dt=torch.float32: torch.as_tensor(x).to(dev, dt)  # noqa: E731
        self.dev = dev
        self.pos, self.nrm, self.area = T(A["pos"]), T(A["nrm"]), T(A["area"])
        self.obj = T(A["obj"].astype(np.int64), torch.int64); self.mat = T(A["mat"].astype(np.int64), torch.int64)
        self.nbr = T(A["nbr"].astype(np.int64), torch.int64); self.nA = int(A["n_A"]); self.N = len(A["obj"])
        self.f2c = T(A["f2c"].astype(np.int64), torch.int64); self.mid = T(A["mid"].astype(np.int64), torch.int64)
        self.cpos, self.cnrm, self.carea = T(A["cpos"]), T(A["cnrm"]), T(A["carea"])
        self.cmat = T(A["cmat"].astype(np.int64), torch.int64); self.cobj = T(A["cobj"].astype(np.int64), torch.int64); self.Nc = len(A["cobj"])
        self.Ks = [KA, KB]
        rot = T(A["rot"])
        t, b = frames_t(self.nrm, rot)
        L = T(C.cosine_dirs_local())
        self.dirs = (L[None, :, 0:1] * t[:, None] + L[None, :, 1:2] * b[:, None] + L[None, :, 2:3] * self.nrm[:, None]).half()  # [Np, K, 3]
        td, bd = M.det_frame(self.nrm)
        loc = torch.stack([(self.dirs.float() * td[:, None]).sum(-1), (self.dirs.float() * bd[:, None]).sum(-1), (self.dirs.float() * self.nrm[:, None]).sum(-1)], -1)
        self.gbin = M.hemi_bin(loc).to(torch.uint8)                                          # [Np, K] hemispherical bin
        self.detf = (td, bd)
        rho = torch.zeros(self.N, 3, device=dev); kap = torch.zeros(self.N, 2, 3, device=dev); alpha = torch.zeros(self.N, 2, device=dev)
        hp = torch.zeros(self.N, 16, device=dev); self.p = [None, None]; self.q = [None, None]
        Wm = torch.zeros(self.N, C.N_MID, 3, device=dev, dtype=torch.float16); self.hc = [None, None]
        with torch.no_grad():
            for o, K in enumerate(self.Ks):
                ii = torch.arange(0, self.nA, device=dev) if o == 0 else torch.arange(self.nA, self.N, device=dev)
                mat = self.mat[ii] if o == 0 else torch.zeros_like(ii)
                h = K.feat(self.pos[ii], mat)
                r_, k_, a_ = K.bsdf_params(h); rho[ii], kap[ii], alpha[ii] = r_, k_, a_
                p, q = K.p_q(h); self.p[o], self.q[o] = p, q
                hp[ii] = K.proj(h)
                self.hc[o] = K.src_desc(torch.where(self.cobj == 0, self.cmat, 0) if o == 0 else torch.zeros_like(self.cmat))
            for o, K in enumerate(self.Ks):
                lo = 0 if o == 0 else self.nA; hi = self.nA if o == 0 else self.N
                for s in range(lo, hi, chunk):
                    ii = torch.arange(s, min(s + chunk, hi), device=dev)
                    mc = self.mid[ii].clamp(min=0)
                    Wmm = K.mid_w(hp[ii], self.hc[o][mc], self.nrm[ii], self.cnrm[mc], (self.cpos[mc] - self.pos[ii][:, None]) / C.R_MID, self.nrm[ii])
                    Wm[ii] = (Wmm * ((self.mid[ii] >= 0).float() * self.carea[mc])[..., None]).half()
        self.rho, self.kap, self.alpha, self.hp, self.Wm = rho, kap, alpha, hp, Wm
        self.hs = [K.src_desc(torch.where(self.obj == 0, self.mat, 0) if o == 0 else torch.zeros_like(self.mat)) for o, K in enumerate(self.Ks)]
        self.dom = [torch.arange(0, self.nA, device=dev), torch.arange(self.nA, self.N, device=dev)]

    @torch.no_grad()
    def band_sender(self, h, chunk=1 << 14):
        """sum_{s'} A_s' sum_b W_b(s, s', w_o = n_s) h_b(s') for every port (sender-side isotropic approximation),
        evaluated on the fly in chunks (memory-light; the weights do not depend on the configuration)."""
        out = torch.zeros(self.N, 3, device=self.dev)
        for o, K in enumerate(self.Ks):
            lo = 0 if o == 0 else self.nA; hi = self.nA if o == 0 else self.N
            for s in range(lo, hi, chunk):
                ii = torch.arange(s, min(s + chunk, hi), device=self.dev)
                nb = self.nbr[ii].clamp(min=0)
                W = K.band_w(self.hp[ii], self.hs[o][nb], self.nrm[ii], self.nrm[nb], (self.pos[nb] - self.pos[ii][:, None]) / C.R_LOCAL, self.nrm[ii])
                out[ii] = torch.einsum("pkbc,pkbc->pc", W * ((self.nbr[ii] >= 0).float() * self.area[nb])[..., None, None], h[nb])
        return out


class State:
    """Solved incident/outgoing boundary state of one geometry state and one sun direction."""

    def __init__(self, T: Tables, H, V, w_sun, present_B=True):
        self.T = T
        dev = T.dev
        self.H = torch.as_tensor(H.astype(np.int64), device=dev)                            # [Np, K] (-1 miss, -2 enclosed)
        self.w = torch.as_tensor(w_sun, dtype=torch.float32, device=dev)
        cosL = (T.nrm @ self.w).clamp(min=0)
        self.Esun = C.E_SUN * cosL * torch.as_tensor(V, dtype=torch.float32, device=dev)   # [Np]
        if not present_B:
            self.Esun[T.dom[1]] = 0
        td, bd = T.detf
        self.sunbin = M.hemi_bin(torch.stack([td @ self.w, bd @ self.w, T.nrm @ self.w], -1))
        self.present_B = present_B
        self.L = torch.zeros(T.N, C.K_GATHER, 3, device=dev, dtype=torch.float16)
        self.Hc = self.H.clamp(min=0); self.valid = self.H >= 0
        self.spec = torch.empty(T.N, C.K_GATHER, 3, device=dev, dtype=torch.float16)          # delta-sun glossy, per entry
        for s in range(0, T.N, 1 << 14):
            sl = slice(s, s + (1 << 14))
            hit = self.Hc[sl]; wq = -T.dirs[sl].float(); n_s = T.nrm[hit]
            sp = torch.zeros(hit.shape + (3,), device=dev)
            for l in range(2):
                g = M.ggx(n_s, self.w.expand_as(wq), wq, T.alpha[hit][..., l:l + 1])
                sp += T.kap[hit][..., l, :] * g * self.Esun[hit][..., None]
            self.spec[sl] = (sp * self.valid[sl][..., None]).half()

    def incident_aggregates(self, chunk=1 << 15):
        T = self.T
        w = math.pi / C.K_GATHER
        Eind = torch.empty(T.N, 3, device=T.dev); h = torch.zeros(T.N, C.N_BINS_H, 3, device=T.dev)
        for s in range(0, T.N, chunk):
            sl = slice(s, s + chunk)
            Lf = self.L[sl].float()
            Eind[sl] = Lf.sum(1) * w
            h[sl].scatter_add_(1, T.gbin[sl].long()[..., None].expand(-1, -1, 3), Lf * w)
        h[torch.arange(T.N, device=T.dev), self.sunbin] += self.Esun[:, None]
        return Eind, h

    def outgoing_iso(self, Eind, h):
        T = self.T
        Etot = Eind + self.Esun[:, None]
        I = T.rho / math.pi * Etot + T.kap.sum(1) / math.pi * Eind
        gam = []
        for o in (0, 1):
            ii = T.dom[o]
            g = (T.area[ii][:, None, None] * T.q[o] * Etot[ii][:, None, :]).sum(0)           # [M, 3]
            gam.append(g)
            I[ii] += (T.p[o] * g[None]).sum(1)
        Ec = torch.zeros(T.Nc, 3, device=T.dev).index_add_(0, T.f2c, Etot * T.area[:, None]) / T.carea[:, None]
        self.Ec = Ec
        for s in range(0, T.N, 1 << 15):
            sl = slice(s, s + (1 << 15))
            I[sl] += (T.Wm[sl].float() * Ec[T.mid[sl].clamp(min=0)]).sum(1)
        I += T.band_sender(h)
        if not self.present_B:
            I[T.dom[1]] = 0
        return I, gam

    def gather_once(self):
        Eind, h = self.incident_aggregates()
        I, gam = self.outgoing_iso(Eind, h)
        Ih = I.half()
        Lnew = torch.empty_like(self.L)
        for s in range(0, self.T.N, 1 << 16):
            sl = slice(s, s + (1 << 16))
            Lnew[sl] = (Ih[self.Hc[sl]] * self.valid[sl][..., None] + self.spec[sl])
        num = (Lnew.float() - self.L.float()).abs().sum(); den = Lnew.float().abs().sum().clamp(min=1e-12)
        self.L = Lnew
        self.gamma = gam
        return float(num / den)

    def solve(self, tol=TOL, max_it=MAX_IT):
        t = time.time(); hist = []; self.diverged = False
        for it in range(max_it):
            d = self.gather_once(); hist.append(d)
            if not np.isfinite(d) or (it > 5 and d > 1.0):
                self.diverged = True                     # non-passive composed operator: reported, never hidden
                break
            if d < tol and it > 0:
                break
        torch.cuda.synchronize()
        Eind, h = self.incident_aggregates()
        self.h = h; self.Eind = Eind
        _, self.gamma = self.outgoing_iso(Eind, h)
        self.iters = len(hist); self.hist = hist; self.solve_s = time.time() - t
        return self


@torch.no_grad()
def render_pixels(T: Tables, S: State, px, chunk=1 << 14):
    """px: dict(pos canonical, nrm, cam, vis, dom, port [N], port4 [N, 4], w4 [N, 4]) -> radiance [N, 3] and parts."""
    out = torch.zeros(px["pos"].shape[0], 3, device=T.dev); parts = torch.zeros(px["pos"].shape[0], 4, 3, device=T.dev)
    for o in (0, 1):
        K = T.Ks[o]; m = torch.nonzero(px["dom"] == o)[:, 0]
        if not len(m):
            continue
        off = 0 if o == 0 else T.nA
        for s in range(0, len(m), chunk):
            ii = m[s:s + chunk]
            canon, n, wo, port = px["pos"][ii], px["nrm"][ii], px["cam"][ii], px["port"][ii]
            mat = T.mat[port] if o == 0 else torch.zeros_like(port)
            hft = K.feat(canon, mat)
            rho, kap, alpha = K.bsdf_params(hft)
            p, _ = K.p_q(hft)
            # K_loc: delta sun + indirect per gather direction of up to 4 interpolation ports
            Ls = (C.E_SUN * (n * S.w).sum(-1).clamp(min=0) * px["vis"][ii])[:, None, None].expand(-1, 1, 3)
            loc = K.loc_response(n, wo, rho, kap, alpha, S.w.expand_as(n)[:, None], Ls)
            ind = torch.zeros_like(loc)
            for q in range(px["port4"].shape[1]):
                pq = px["port4"][ii, q]; wq = px["w4"][ii, q]
                use = pq >= 0
                if not use.any():
                    continue
                pqc = pq.clamp(min=0)
                Li = S.L[pqc].float() * (math.pi / C.K_GATHER)
                ind += K.loc_response(n, wo, rho, kap, alpha, T.dirs[pqc].float(), Li) * (wq * use)[:, None]
            glob = (p * S.gamma[o][None]).sum(1)
            nb = T.nbr[port]; use = nb >= 0; nbc = nb.clamp(min=0)
            W = K.band_w(K.proj(hft), T.hs[o][nbc], n, T.nrm[nbc], (T.pos[nbc] - canon[:, None]) / C.R_LOCAL, wo)
            band = torch.einsum("nkbc,nkbc->nc", W * (T.area[nbc] * use)[..., None, None], S.h[nbc])
            mc = T.mid[port]; usem = mc >= 0; mcg = mc.clamp(min=0)
            Wmm = K.mid_w(K.proj(hft), T.hc[o][mcg], n, T.cnrm[mcg], (T.cpos[mcg] - canon[:, None]) / C.R_MID, wo)
            midt = (Wmm * (T.carea[mcg] * usem)[..., None] * S.Ec[mcg]).sum(1)
            parts[ii, 0], parts[ii, 1], parts[ii, 2], parts[ii, 3] = loc, ind, glob, band + midt
            out[ii] = loc + ind + glob + band + midt
    return out, parts


@torch.no_grad()
def uniform_albedo(T: Tables):
    """Passivity diagnostic: effective hemispherical albedo pi * I_s / E of every port under a uniform, isotropic unit
    incident radiance field on all ports of its domain (E = pi per port, spread uniformly over the 19 bins by solid
    angle of the cosine-weighted bins). Values > 1 mark ports whose learned response amplifies energy."""
    H = np.full((T.N, C.K_GATHER), -1, np.int64)
    S = State(T, H, np.zeros(T.N, bool), np.array([0.0, 0.0, 1.0]))
    S.L = torch.ones(T.N, C.K_GATHER, 3, device=T.dev, dtype=torch.float16)          # unit radiance from every gather direction
    Eind, h = S.incident_aggregates()
    I, _ = S.outgoing_iso(Eind, h)
    return (math.pi * I / Eind.clamp(min=1e-9)).cpu().numpy()

