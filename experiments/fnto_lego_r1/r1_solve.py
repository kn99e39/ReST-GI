"""WL42 r1 composition: dense ray-cast free-space coupling C (diagnostic-grade, NOT compressed / learned) x frozen K,
multi-bounce state solve, render. Same operator semantics as Batch 6 (fnto_solve) with the corrected inputs:
multi-sample gathers, aggregated band / mid sources with represented-area weights, port sun irradiance from the visible
fraction of surface samples, pixels on their surface-identity port. Solver instrumentation (audit #12 / F6):
  * residual history ||L_k - L_{k-1}|| and per-iteration ratio; DIVERGED when the median ratio of the last 5 iterations
    is >= 1 after iteration 8, or values are non-finite (Batch 6's `d > 1` branch could not trigger);
  * converged when ||L_k - L_{k-1}|| / ||L_k|| < TOL; otherwise UNCONVERGED at MAX_IT (reported separately);
  * spectral_radius(): power iteration of the homogeneous map L -> C K L (no sun), the passivity gate quantity.
"""

from __future__ import annotations

import math
import time

import numpy as np
import torch

import r1_common as C
import fnto_model as M

TOL = 1e-3
MAX_IT = 100


def frames_t(n, rot):
    a = torch.where(n[:, 2:3].abs() < 0.9, n.new_tensor([0.0, 0, 1]).expand_as(n), n.new_tensor([1.0, 0, 0]).expand_as(n))
    t = torch.nn.functional.normalize(torch.cross(a, n, dim=1), dim=1); b = torch.cross(n, t, dim=1)
    c, s = torch.cos(rot)[:, None], torch.sin(rot)[:, None]
    return c * t + s * b, -s * t + c * b


class Tables:
    def __init__(self, KA, KB, dev, chunk=1 << 15):
        A = np.load(C.OUT1 / "anchors.npz")
        T = lambda x, dt=torch.float32: torch.as_tensor(np.asarray(x)).to(dev, dt)  # noqa: E731
        L_ = lambda x: T(x, torch.int64)  # noqa: E731
        self.dev = dev; self.Ks = [KA, KB]
        self.pos, self.nrm, self.area = T(A["pos"]), T(A["nrm"]), T(A["area"]); self.mat, self.obj = L_(A["mat"]), L_(A["obj"])
        self.nA = int(A["n_A"]); self.N = len(A["obj"])
        self.dom = [torch.arange(0, self.nA, device=dev), torch.arange(self.nA, self.N, device=dev)]
        self.nbr, self.nbrA, self.mid, self.midA = L_(A["nbr"]), T(A["nbrA"]), L_(A["mid"]), T(A["midA"])
        self.f2b, self.bpos, self.bnrm, self.barea, self.bmat, self.bobj = L_(A["f2b"]), T(A["bpos"]), T(A["bnrm"]), T(A["barea"]), L_(A["bmat"]), L_(A["bobj"])
        self.f2c, self.cpos, self.cnrm, self.carea, self.cmat, self.cobj = L_(A["f2c"]), T(A["cpos"]), T(A["cnrm"]), T(A["carea"]), L_(A["cmat"]), L_(A["cobj"])
        self.Nb, self.Nc = len(A["barea"]), len(A["carea"])
        # gather directions per (port, k): the frame of the port sample that casts gather k (same as r1_relations.ray_setup)
        K = C.K_GATHER; S = C.S_PORT; per = K // S
        sn = A["snrm"]; valid = np.isfinite(sn[..., 0]); nval = valid.sum(1)
        order = np.argsort(~valid, axis=1, kind="stable"); use = (np.arange(K) // per)[None] % np.maximum(nval, 1)[:, None]
        sidx = np.take_along_axis(order, use, 1)
        Lloc = T(C.cosine_dirs_local()); rot = T(A["rot"]); dirs = torch.empty(self.N, K, 3, device=dev, dtype=torch.float16)
        for s in range(0, self.N, chunk):
            sl = slice(s, s + chunk)
            nn = T(np.take_along_axis(sn[sl], sidx[sl][..., None], 1).reshape(-1, 3))
            t, b = frames_t(nn, rot[sl].repeat_interleave(K))
            L = Lloc.repeat(nn.shape[0] // K, 1)
            dirs[sl] = (L[:, 0:1] * t + L[:, 1:2] * b + L[:, 2:3] * nn).view(-1, K, 3).half()
        self.dirs = dirs
        td, bd = M.det_frame(self.nrm); self.detf = (td, bd)
        loc = torch.stack([(dirs.float() * td[:, None]).sum(-1), (dirs.float() * bd[:, None]).sum(-1), (dirs.float() * self.nrm[:, None]).sum(-1)], -1)
        self.gbin = M.hemi_bin(loc).to(torch.uint8); del loc
        rho = torch.zeros(self.N, 3, device=dev); kap = torch.zeros(self.N, 2, 3, device=dev); alpha = torch.zeros(self.N, 2, device=dev)
        hp = torch.zeros(self.N, 16, device=dev); self.p = [None, None]; self.q = [None, None]; self.hs = [None, None]; self.hc = [None, None]
        with torch.no_grad():
            for o, Kk in enumerate(self.Ks):
                ii = self.dom[o]; mat = self.mat[ii] if o == 0 else torch.zeros_like(ii)
                for s in range(0, len(ii), chunk):
                    jj = ii[s:s + chunk]; h = Kk.feat(self.pos[jj], mat[s:s + chunk])
                    rho[jj], kap[jj], alpha[jj] = Kk.bsdf_params(h); hp[jj] = Kk.proj(h)
                h = Kk.feat(self.pos[ii], mat); p, q = Kk.p_q(h); self.p[o], self.q[o] = p, q
                self.hs[o] = Kk.src_desc(torch.where(self.bobj == 0, self.bmat, 0) if o == 0 else torch.zeros_like(self.bmat))
                self.hc[o] = Kk.src_desc(torch.where(self.cobj == 0, self.cmat, 0) if o == 0 else torch.zeros_like(self.cmat))
        self.rho, self.kap, self.alpha, self.hp = rho, kap, alpha, hp
        self.Wm = torch.zeros(self.N, C.K_MID, 3, device=dev, dtype=torch.float16)
        with torch.no_grad():
            for o, Kk in enumerate(self.Ks):
                ii = self.dom[o]
                for s in range(0, len(ii), chunk):
                    jj = ii[s:s + chunk]; mc = self.mid[jj].clamp(min=0)
                    W = Kk.mid_w(hp[jj], self.hc[o][mc], self.nrm[jj], self.cnrm[mc], (self.cpos[mc] - self.pos[jj][:, None]) / C.R_MID, self.nrm[jj])
                    self.Wm[jj] = (W * (self.midA[jj] * (self.mid[jj] >= 0))[..., None]).half()

    @torch.no_grad()
    def band_sender(self, hB, chunk=1 << 14):
        out = torch.zeros(self.N, 3, device=self.dev)
        for o, Kk in enumerate(self.Ks):
            ii = self.dom[o]
            for s in range(0, len(ii), chunk):
                jj = ii[s:s + chunk]; nb = self.nbr[jj]; nbc = nb.clamp(min=0)
                W = Kk.band_w(self.hp[jj], self.hs[o][nbc], self.nrm[jj], self.bnrm[nbc], (self.bpos[nbc] - self.pos[jj][:, None]) / C.R_LOCAL, self.nrm[jj])
                out[jj] = torch.einsum("pkbc,pkb->pc", W * (self.nbrA[jj] * (nb >= 0))[..., None, None], hB[nbc])
        return out


class State:
    def __init__(self, T: Tables, H, Vfrac, w_sun, sun_strength=C.E_SUN):
        dev = T.dev; self.T = T
        self.H = torch.as_tensor(np.asarray(H).astype(np.int64), device=dev); self.Hc = self.H.clamp(min=0); self.valid = self.H >= 0
        self.w = torch.as_tensor(w_sun, dtype=torch.float32, device=dev)
        self.Esun = sun_strength * (T.nrm @ self.w).clamp(min=0) * torch.as_tensor(Vfrac, dtype=torch.float32, device=dev)
        td, bd = T.detf; self.sunbin = M.hemi_bin(torch.stack([td @ self.w, bd @ self.w, T.nrm @ self.w], -1))
        self.L = torch.zeros(T.N, C.K_GATHER, 3, device=dev, dtype=torch.float16)
        self.spec = torch.empty_like(self.L)
        with torch.no_grad():
            for s in range(0, T.N, 1 << 14):
                sl = slice(s, s + (1 << 14)); hit = self.Hc[sl]; wq = -T.dirs[sl].float(); n_s = T.nrm[hit]
                sp = torch.zeros(hit.shape + (3,), device=dev)
                for l in range(2):
                    g = M.ggx(n_s, self.w.expand_as(wq), wq, T.alpha[hit][..., l:l + 1])
                    sp += T.kap[hit][..., l, :] * g * self.Esun[hit][..., None]
                self.spec[sl] = (sp * self.valid[sl][..., None]).half()

    def aggregates(self, chunk=1 << 15):
        T = self.T; w = math.pi / C.K_GATHER
        Eind = torch.empty(T.N, 3, device=T.dev); h = torch.zeros(T.N, C.N_BINS_H, 3, device=T.dev)
        for s in range(0, T.N, chunk):
            sl = slice(s, s + chunk); Lf = self.L[sl].float()
            Eind[sl] = Lf.sum(1) * w; h[sl].scatter_add_(1, T.gbin[sl].long()[..., None].expand(-1, -1, 3), Lf * w)
        h[torch.arange(T.N, device=T.dev), self.sunbin] += self.Esun[:, None]
        return Eind, h

    def outgoing(self, Eind, h, sun_on=True):
        T = self.T; Etot = Eind + (self.Esun[:, None] if sun_on else 0)
        I = T.rho / math.pi * Etot + T.kap.sum(1) / math.pi * Eind
        gam = []
        for o in (0, 1):
            ii = T.dom[o]; g = (T.area[ii][:, None, None] * T.q[o] * Etot[ii][:, None, :]).sum(0); gam.append(g)
            I[ii] += (T.p[o] * g[None]).sum(1)
        Ec = torch.zeros(T.Nc, 3, device=T.dev).index_add_(0, T.f2c, Etot * T.area[:, None]) / T.carea[:, None].clamp(min=1e-12)
        hB = torch.zeros(T.Nb, C.N_BINS_H, 3, device=T.dev).index_add_(0, T.f2b, h * T.area[:, None, None]) / T.barea[:, None, None].clamp(min=1e-12)
        for s in range(0, T.N, 1 << 15):
            sl = slice(s, s + (1 << 15)); I[sl] += (T.Wm[sl].float() * Ec[T.mid[sl].clamp(min=0)]).sum(1)
        I += T.band_sender(hB)
        self.gamma, self.Ec, self.hB = gam, Ec, hB
        return I

    def step(self, sun_on=True):
        Eind, h = self.aggregates()
        if not sun_on:
            h[torch.arange(self.T.N, device=self.T.dev), self.sunbin] -= self.Esun[:, None]
        I = self.outgoing(Eind, h, sun_on).half()
        Ln = torch.empty_like(self.L)
        for s in range(0, self.T.N, 1 << 16):
            sl = slice(s, s + (1 << 16)); Ln[sl] = I[self.Hc[sl]] * self.valid[sl][..., None] + (self.spec[sl] if sun_on else 0)
        dn = float((Ln.float() - self.L.float()).norm()); nn = float(Ln.float().norm())
        self.L = Ln
        return dn, nn

    @torch.no_grad()
    def solve(self, tol=TOL, max_it=MAX_IT):
        t = time.time(); self.hist = []; self.ratio = []; self.status = "unconverged"
        prev = None
        for it in range(max_it):
            dn, nn = self.step()
            self.hist.append(dn / max(nn, 1e-30))
            if prev:
                self.ratio.append(dn / prev)
            prev = dn
            if not (np.isfinite(dn) and np.isfinite(nn)):
                self.status = "diverged"; break
            if it >= 8 and len(self.ratio) >= 5 and float(np.median(self.ratio[-5:])) >= 1.0:
                self.status = "diverged"; break
            if it > 0 and self.hist[-1] < tol:
                self.status = "converged"; break
        torch.cuda.synchronize()
        Eind, h = self.aggregates(); self.Eind, self.h = Eind, h; self.outgoing(Eind, h)
        self.iters = len(self.hist); self.solve_s = time.time() - t
        return self


@torch.no_grad()
def spectral_radius(T, H, iters=40, seed=0):
    """Power iteration of the homogeneous composed map (no sun): rho = lim ||L_{k+1}|| / ||L_k||."""
    S = State(T, H, np.zeros(T.N), np.array([0.0, 0.0, 1.0]))
    g = torch.Generator(device=T.dev); g.manual_seed(seed)
    S.L = (torch.rand(S.L.shape, generator=g, device=T.dev) * S.valid[..., None]).half()
    rates = []
    for _ in range(iters):
        n0 = float(S.L.float().norm()); S.step(sun_on=False); n1 = float(S.L.float().norm())
        if not np.isfinite(n1) or n1 == 0:
            rates.append(float("inf") if not np.isfinite(n1) else 0.0); break
        rates.append(n1 / n0); S.L = (S.L.float() / n1 * 1e3).half()
    return {"rho_estimate": float(np.median(rates[-10:])) if rates else None, "rates": rates}


@torch.no_grad()
def render_pixels(T, S, px, chunk=1 << 13):
    """px: pos (canonical), nrm, cam, vis (pixel sun visibility), dom, port (surface identity) -> radiance and parts."""
    out = torch.zeros(px["pos"].shape[0], 3, device=T.dev); parts = torch.zeros(px["pos"].shape[0], 4, 3, device=T.dev)
    for o in (0, 1):
        Kk = T.Ks[o]; m = torch.nonzero(px["dom"] == o)[:, 0]
        for s in range(0, len(m), chunk):
            ii = m[s:s + chunk]; canon, n, wo, port = px["pos"][ii], px["nrm"][ii], px["cam"][ii], px["port"][ii]
            mat = T.mat[port] if o == 0 else torch.zeros_like(port)
            h = Kk.feat(canon, mat); rho, kap, alpha = Kk.bsdf_params(h); p, _ = Kk.p_q(h); hx = Kk.proj(h)
            Ls = (C.E_SUN * (n * S.w).sum(-1).clamp(min=0) * px["vis"][ii])[:, None, None].expand(-1, 1, 3)
            loc = Kk.loc_response(n, wo, rho, kap, alpha, S.w.expand_as(n)[:, None], Ls)
            ind = Kk.loc_response(n, wo, rho, kap, alpha, T.dirs[port].float(), S.L[port].float() * (math.pi / C.K_GATHER))
            glob = (p * S.gamma[o][None]).sum(1)
            nb = T.nbr[port]; nbc = nb.clamp(min=0)
            W = Kk.band_w(hx, T.hs[o][nbc], n, T.bnrm[nbc], (T.bpos[nbc] - canon[:, None]) / C.R_LOCAL, wo)
            band = torch.einsum("nkbc,nkbc->nc", W * (T.nbrA[port] * (nb >= 0))[..., None, None], S.hB[nbc])
            mc = T.mid[port]; mcc = mc.clamp(min=0)
            Wm = Kk.mid_w(hx, T.hc[o][mcc], n, T.cnrm[mcc], (T.cpos[mcc] - canon[:, None]) / C.R_MID, wo)
            midt = (Wm * (T.midA[port] * (mc >= 0))[..., None] * S.Ec[mcc]).sum(1)
            parts[ii, 0], parts[ii, 1], parts[ii, 2], parts[ii, 3] = loc, ind, glob, band + midt
            out[ii] = loc + ind + glob + band + midt
    return out, parts
