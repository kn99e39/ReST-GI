"""FNTO v0.1 learned intrinsic optical-domain operator K_theta_a (torch).

One `DomainKernel` per optical domain (A = official Lego object, B = red box); a domain owns ALL of its parameters
(no learned component is shared between domains). K_a maps a spatially and directionally varying incident radiance
field i_a(s, w_i) on the domain's boundary to outgoing radiance b_a(x, w_o). It is LINEAR in i_a by construction:
every learned network below produces coefficients from (position, material, geometry, directions) only and is never
fed radiance; radiance enters only through sums of (non-negative coefficient) x (incident radiance).

    b(x, w_o) = K_loc[i](x, w_o) + K_glob[i](x, w_o) + K_band[i](x, w_o) + K_mid[i](x, w_o)

  K_loc  (local, directionally sharp): a learned per-point BSDF at the receiving point itself,
         f(x, w_i, w_o) = rho(x)/pi + sum_{l=1,2} kappa_l(x) GGX(w_i, w_o; n, alpha_l(x)),   applied to the incident
         radiance at x:  int f L_i cos dw_i  (the sun exactly as a delta direction; indirect radiance per gather direction).
         GGX lobes have learned roughness down to alpha = 0.01, so no low-order angular bottleneck limits sharpness.
  K_glob (nonlocal material-internal, low rank): sum_m p_m(x) q_m(s) per colour channel, entry cosine-weighted:
         K_glob[i](x) = sum_m p_mc(x) gamma_mc,  gamma_mc = sum_s A_s q_mc(s) E_c(s),  E_c(s) = int L_i cos dw.
         p, q >= 0 (softplus): a non-negative kernel. Exit is isotropic (the reference's Burley SSS exit is diffuse).
  K_band (nonlocal material-internal, short range, directional): sum over the N_LOCAL ports s within R_LOCAL of x of
         A_s sum_b W_bc(x, s, w_o) h_bc(s), h_b(s) = int_{bin b} L_i cos dw over 19 hemispherical bins of s's
         deterministic frame. W >= 0; W sees the receiver's features, the source's material and the relative geometry
         (offset, source frame, w_o) in x's frame. Covers near-field SSS, thin-wall and glass entrance -> exit transport.

  K_mid  (nonlocal material-internal, mid range): sum over the N_MID coarse source ports c (H_MID clusters) within R_MID of
         x of A_c Wm_c(x, c, w_o) E_c(c), E_c = area-mean cosine-weighted irradiance of the cluster, Wm >= 0. The
         multiresolution residual between K_band (<= R_LOCAL) and the low-rank K_glob (contract §3.2).

Linearity: K[a i1 + b i2] = a K[i1] + b K[i2] for any i1, i2 (not only non-negative a, b): there is no nonlinearity after
the contraction with radiance. Zero input -> zero output. Non-negativity: all coefficients >= 0, so K maps non-negative
radiance to non-negative radiance (passivity is tested, not assumed).
"""

from __future__ import annotations

import hashlib
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import fnto_common as C


class TriField(nn.Module):
    def __init__(self, res, res_c, ch):
        super().__init__()
        self.fine = nn.Parameter(torch.randn(3, ch, res, res) * 0.01)
        self.coarse = nn.Parameter(torch.randn(3, ch, res_c, res_c) * 0.1)
        self.out_dim = 6 * ch

    def forward(self, x):                                   # x in [0, 1]^3 [N, 3]
        if x.shape[0] == 0:
            return x.new_zeros(0, self.out_dim)
        g = (x * 2 - 1).clamp(-1, 1)
        uv = torch.stack([g[:, [0, 1]], g[:, [1, 2]], g[:, [0, 2]]], 0)[:, None]
        f = F.grid_sample(self.fine, uv, align_corners=True, mode="bilinear")[:, :, 0]
        c = F.grid_sample(self.coarse, uv, align_corners=True, mode="bilinear")[:, :, 0]
        return torch.cat([f, c], 1).permute(2, 0, 1).reshape(x.shape[0], -1)


def mlp(i, w, o, n):
    layers, d = [], i
    for _ in range(n):
        layers += [nn.Linear(d, w), nn.ReLU()]; d = w
    layers.append(nn.Linear(d, o))
    return nn.Sequential(*layers)


def det_frame(n):
    """Deterministic tangent frame (t, b) of normals n [N, 3] (torch); identical to fnto_common.frames(n, 0)."""
    a = torch.where(n[:, 2:3].abs() < 0.9, n.new_tensor([0.0, 0, 1]).expand_as(n), n.new_tensor([1.0, 0, 0]).expand_as(n))
    t = F.normalize(torch.cross(a, n, dim=1), dim=1)
    b = torch.cross(n, t, dim=1)
    return t, b


def hemi_bin(d_local):
    """Local directions [..., 3] -> bin index (identical to fnto_common.hemi_bin_local), torch."""
    z = d_local[..., 2]
    az = torch.remainder(torch.atan2(d_local[..., 1], d_local[..., 0]), 2 * math.pi)
    s6 = 1 + (az / (2 * math.pi) * 6).long().clamp(max=5)
    s12 = 7 + (az / (2 * math.pi) * 12).long().clamp(max=11)
    return torch.where(z > math.cos(math.radians(20)), torch.zeros_like(s6), torch.where(z > math.cos(math.radians(50)), s6, s12))


def ggx(n, wi, wo, alpha):
    """GGX microfacet reflection lobe value (no Fresnel; Cycles' Glossy BSDF), [N, L] for alpha [N, L]."""
    ci = (n * wi).sum(-1, keepdim=True); co = (n * wo).sum(-1, keepdim=True)
    h = F.normalize(wi + wo, dim=-1)
    ch = (n * h).sum(-1, keepdim=True).clamp(min=0)
    a2 = alpha * alpha
    D = a2 / (math.pi * ((ch * ch) * (a2 - 1) + 1) ** 2).clamp(min=1e-12)
    ci_, co_ = ci.clamp(min=1e-4), co.clamp(min=1e-4)
    G1 = lambda c: 2 * c / (c + torch.sqrt(a2 + (1 - a2) * c * c))  # noqa: E731
    val = D * G1(ci_) * G1(co_) / (4 * ci_ * co_)
    return torch.where((ci > 0) & (co > 0), val, torch.zeros_like(val))


class DomainKernel(nn.Module):
    def __init__(self, aabb, n_mat, res=512, res_c=64, ch=8, m_rank=48, width=128):
        super().__init__()
        self.register_buffer("aabb", torch.tensor(aabb, dtype=torch.float32))      # [2, 3] canonical bounds
        self.field = TriField(res, res_c, ch)
        self.mat_emb = nn.Embedding(n_mat, 8)
        hd = self.field.out_dim + 8
        self.bsdf = mlp(hd, width, 3 + 2 * 4, 2)                                     # rho(3); per lobe kappa(3) + alpha(1)
        self.P = mlp(hd, width, 3 * m_rank, 2)
        self.Q = mlp(hd, width, 3 * m_rank, 2)
        self.proj = nn.Linear(hd, 16)
        self.src = nn.Linear(8, 16)                    # band-kernel source descriptor: material embedding (+ geometry below)
        # band-kernel MLP; its first layer is split into receiver / source / geometry parts (same function, cheaper)
        self.W1x = nn.Linear(16, 64); self.W1s = nn.Linear(16, 64, bias=False); self.W1g = nn.Linear(12, 64, bias=False)
        self.W = nn.Sequential(nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, C.N_BINS_H * 3))
        # mid-range kernel MLP (coarse source ports, cosine-weighted irradiance entry), first layer split likewise
        self.M1x = nn.Linear(16, 64); self.M1s = nn.Linear(16, 64, bias=False); self.M1g = nn.Linear(9, 64, bias=False)
        self.Wm = nn.Sequential(nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 3))
        self.m_rank = m_rank
        with torch.no_grad():
            self.bsdf[-1].bias.zero_(); self.bsdf[-1].bias[3:6] = -3.0; self.bsdf[-1].bias[7:10] = -3.0
            self.P[-1].bias.fill_(-4.0); self.Q[-1].bias.fill_(-4.0); self.W[-1].bias.fill_(-6.0); self.Wm[-1].bias.fill_(-8.0)

    # ---- per-point coefficient networks (no radiance input) ------------------------------------------------------------
    def feat(self, canon, mat):
        lo, hi = self.aabb[0], self.aabb[1]
        return torch.cat([self.field(((canon - lo) / (hi - lo)).clamp(0, 1)), self.mat_emb(mat)], -1)

    def bsdf_params(self, h):
        o = self.bsdf(h)
        rho = torch.sigmoid(o[:, 0:3])
        kap = torch.stack([F.softplus(o[:, 3:6]), F.softplus(o[:, 7:10])], 1)          # [N, 2, 3]
        alpha = 0.01 + 0.99 * torch.sigmoid(torch.stack([o[:, 6], o[:, 10]], 1))       # [N, 2]
        return rho, kap, alpha

    def p_q(self, h):
        return F.softplus(self.P(h)).view(-1, self.m_rank, 3), F.softplus(self.Q(h)).view(-1, self.m_rank, 3)

    def src_desc(self, mat):
        return self.src(self.mat_emb(mat))

    def band_w(self, hx, hs, nx, ns, dx, wo):
        """W [N, k, 7, 3] for receivers x (nx [N, 3], hx [N, 16]) and their k ports s (ns/dx [N, k, 3], hs [N, k, 16]).
        All geometric inputs are expressed in x's deterministic frame; dx = (s - x) / R_LOCAL."""
        t, b = det_frame(nx)
        R = torch.stack([t, b, nx], 1)                                                    # [N, 3, 3] rows = x's frame
        ts, _ = det_frame(ns.reshape(-1, 3))
        k = hs.shape[1]
        geo = torch.cat([dx, ts.view_as(ns), ns, wo[:, None].expand(-1, k, -1)], -1)       # [N, k, 12] world
        geo = torch.einsum("nij,nkgj->nkgi", R, geo.view(-1, k, 4, 3)).reshape(-1, k, 12)  # -> x's frame
        with torch.autocast("cuda", dtype=torch.bfloat16):
            z = self.W1x(hx)[:, None] + self.W1s(hs) + self.W1g(geo)
            w = self.W(z)
        return F.softplus(w.float()).view(hx.shape[0], k, C.N_BINS_H, 3)

    def mid_w(self, hx, hs, nx, ns, dx, wo):
        """Mid-range weights [N, k, 3] for receivers x and their k coarse source ports (dx = (c - x) / R_MID)."""
        t, b = det_frame(nx)
        R = torch.stack([t, b, nx], 1)
        k = hs.shape[1]
        geo = torch.cat([dx, ns, wo[:, None].expand(-1, k, -1)], -1)
        geo = torch.einsum("nij,nkgj->nkgi", R, geo.view(-1, k, 3, 3)).reshape(-1, k, 9)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            w = self.Wm(self.M1x(hx)[:, None] + self.M1s(hs) + self.M1g(geo))
        return F.softplus(w.float())

    # ---- operator application (linear in radiance) -----------------------------------------------------------------------
    @staticmethod
    def loc_response(n, wo, rho, kap, alpha, wi, Li):
        """int f L_i cos for incident samples: wi [N, J, 3], Li [N, J, 3] = radiance x cos x solid-angle weight (i.e. the
        cosine-weighted incident power per sample). Returns [N, 3]."""
        J = wi.shape[1]
        nn_ = n[:, None].expand(-1, J, -1); wo_ = wo[:, None].expand(-1, J, -1)
        g = torch.stack([ggx(nn_, wi, wo_, alpha[:, None, l:l + 1].expand(-1, J, 1))[..., 0] for l in range(2)], -1)   # [N, J, 2]
        up = ((nn_ * wi).sum(-1) > 0) & ((nn_ * wo_).sum(-1) > 0)
        diff = (rho[:, None] / math.pi) * up[..., None]
        spec = (g[..., None] * kap[:, None]).sum(2)
        return ((diff + spec) * Li).sum(1)


def make_kernels(dev):
    A = np.load(C.OUT / "anchors.npz")
    pa = A["pos"][A["obj"] == 0]; pb = A["pos"][A["obj"] == 1]
    pad = lambda p, f: np.stack([p.min(0) - f * (p.max(0) - p.min(0)), p.max(0) + f * (p.max(0) - p.min(0))])  # noqa: E731
    KA = DomainKernel(pad(pa, 0.02), 8, m_rank=64).to(dev)
    KB = DomainKernel(pad(pb, 0.05), 1, res=64, res_c=16, m_rank=8).to(dev)
    return KA, KB


def digest(*mods):
    out = {}
    for i, m in enumerate(mods):
        for k, v in m.state_dict().items():
            out[f"{i}.{k}"] = hashlib.sha256(v.detach().cpu().numpy().tobytes()).hexdigest()[:16]
    return out
