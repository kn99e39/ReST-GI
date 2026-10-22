"""SPT-1 model (torch): persistent canonical surface fields + shared heads; dynamic incident state from relation tables.

Persistent learned state (never changed by an edit):
  * per-object triplane fields phi_o (fine + coarse planes) in the object's canonical (T0) frame;
  * shared heads: albedo a(phi) (Lambertian sender reflectance), direct d(phi, n, w_o, w_L) gated by the dataset's
    direct visibility, transfer t(phi, n, w_o) -> 9x3 weights applied linearly to the incident SH state;
  * log E_sun (global sun-irradiance scale).
Dynamic state (recomputed from a relation table R^t, no optimisation):
  * anchor incident state g_a(w_L) = sum_k sh[a,k] * L_k, L_k = a(s_k)/pi * (E_sun max(0, n_s.w_L) vis(s_k, bin(w_L)) + E_ind(s_k));
  * E_ind: multi-bounce anchor irradiance by Jacobi iteration over the same relations (cache, no gradient).
Pixel: L_o = v * d + relu(sum_{j,c} t[j,c] * g[j,c]), g = inverse-distance mix of <= 4 anchors.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import spt_common as C


class TriField(nn.Module):
    def __init__(self, res, res_c, ch):
        super().__init__()
        self.fine = nn.Parameter(torch.randn(3, ch, res, res) * 0.01)
        self.coarse = nn.Parameter(torch.randn(3, ch, res_c, res_c) * 0.1)
        self.out_dim = 6 * ch

    def forward(self, x):                                  # x in [0, 1]^3, [N, 3]
        g = (x * 2 - 1).clamp(-1, 1)
        uv = torch.stack([g[:, [0, 1]], g[:, [1, 2]], g[:, [0, 2]]], 0)[:, None]          # [3, 1, N, 2]
        f = F.grid_sample(self.fine, uv, align_corners=True, mode="bilinear")[:, :, 0]      # [3, ch, N]
        c = F.grid_sample(self.coarse, uv, align_corners=True, mode="bilinear")[:, :, 0]
        return torch.cat([f, c], 1).permute(2, 0, 1).reshape(x.shape[0], -1)


def mlp(i, w, o, n):
    layers, d = [], i
    for _ in range(n):
        layers += [nn.Linear(d, w), nn.ReLU()]; d = w
    layers.append(nn.Linear(d, o))
    return nn.Sequential(*layers)


class SPT(nn.Module):
    def __init__(self, aabb_A, aabb_B, res=512, res_c=64, ch=8, width=256):
        super().__init__()
        self.register_buffer("aabb", torch.tensor(np.stack([aabb_A, aabb_B]), dtype=torch.float32))   # [2, 2, 3]
        self.fields = nn.ModuleList([TriField(res, res_c, ch), TriField(64, 16, ch)])
        fd = self.fields[0].out_dim
        self.albedo = mlp(fd, 64, 3, 2)
        self.direct = mlp(fd + 11, width, 3, 3)
        self.transfer = mlp(fd + 7, width, 27, 3)
        with torch.no_grad():
            self.transfer[-1].bias.zero_(); self.transfer[-1].bias[0:3] = 0.3      # DC response > 0 at init (rows j=0)
        self.log_esun = nn.Parameter(torch.tensor(math.log(5.0)))

    def feat(self, canon, obj):
        out = torch.empty(canon.shape[0], self.fields[0].out_dim, device=canon.device)
        for o in (0, 1):
            m = obj == o
            if m.any():
                lo, hi = self.aabb[o, 0], self.aabb[o, 1]
                out[m] = self.fields[o](((canon[m] - lo) / (hi - lo)).clamp(0, 1))
        return out

    def anchor_albedo(self, R):
        return torch.sigmoid(self.albedo(self.feat(R.pos, R.obj)))

    def decode(self, ft, n, wo, wl, vis, g):
        """ft [N, fd], n/wo/wl [N, 3], vis [N], g [N, 9, 3] -> radiance [N, 3], parts."""
        ndl = (n * wl).sum(-1, keepdim=True); ndo = (n * wo).sum(-1, keepdim=True)
        d = F.softplus(self.direct(torch.cat([ft, n, wo, wl, ndl, ndo], -1))) * vis[:, None]
        t = self.transfer(torch.cat([ft, n, wo, ndo], -1)).view(-1, 9, 3)
        ind = F.relu((t * g).sum(1))
        return d + ind, {"direct": d, "indirect": ind}


class Relations:
    """Anchor tables on one device. hit/vis come from one geometry state (results/spt_lego/rel_<state>.npz)."""

    def __init__(self, state, dev, sh_dtype=torch.float16):
        A = np.load(C.OUT / "anchors.npz")
        self.pos = torch.tensor(A["pos"], device=dev); self.nrm = torch.tensor(A["nrm"], device=dev)
        self.obj = torch.tensor(A["obj"].astype(np.int64), device=dev)
        self.sh = torch.tensor(np.load(C.OUT / "anchor_sh.npy"), device=dev, dtype=sh_dtype)          # [Na, K, 9]
        self.bins = torch.tensor(C.bin_centres(), dtype=torch.float32, device=dev)
        self.set_state(state)

    def set_state(self, state):
        r = np.load(C.OUT / f"rel_{state}.npz")
        dev = self.pos.device
        self.state = state
        self.hit = torch.tensor(r["hit"].astype(np.int64), device=dev)                                # [Na, K]
        self.vis = torch.tensor(np.unpackbits(r["vis"], axis=1)[:, :C.N_BINS], device=dev)            # [Na, N_BINS] uint8
        self.Na, self.K = self.hit.shape

    def visbit(self, s, b):
        return self.vis[s, b].to(torch.float32)


def light_bin_t(d):
    z = d[..., 2].clamp(0, 1 - 1e-6)
    ring = (z * C.N_RING).long().clamp(max=C.N_RING - 1)
    az = torch.remainder(torch.atan2(d[..., 1], d[..., 0]), 2 * math.pi)
    a = (az / (2 * math.pi) * C.N_AZ).long().clamp(max=C.N_AZ - 1)
    return ring * C.N_AZ + a


@torch.no_grad()
def multibounce_cache(model, R, alb=None, iters=4, chunk=256, bins=None):
    """E_ind[a, b, 3]: multi-bounce Lambertian anchor irradiance per light bin (no gradient). bins: subset or all."""
    alb = model.anchor_albedo(R) if alb is None else alb
    bc = R.bins if bins is None else bins
    nb = bc.shape[0]
    esun = model.log_esun.exp()
    cosb = (R.nrm @ bc.T).clamp(min=0)                                                             # [Na, nb]
    if bins is not None:
        raise NotImplementedError
    visb = R.vis.half()
    edir = (esun * cosb * visb).half()                                                              # [Na, nb]
    E = torch.zeros(R.Na, nb, 3, device=R.pos.device, dtype=torch.float16)
    valid = R.hit >= 0; S = R.hit.clamp(min=0)
    for _ in range(iters):
        rad = (alb[:, None, :] / math.pi).half() * (edir[..., None] + E)                            # [Na, nb, 3] sender radiance
        En = torch.empty_like(E)
        for s in range(0, R.Na, chunk):
            Sc, vc = S[s:s + chunk], valid[s:s + chunk]
            En[s:s + chunk] = ((rad[Sc] * vc[..., None, None]).float().sum(1) * (math.pi / R.K)).half()
        E = En
    return E


def anchor_state(model, R, alb, aidx, wl, ecache, chunk=32768):
    """g [M, 9, 3] for anchors aidx [M] under per-row lights wl [M, 3]; ecache [Na, N_BINS, 3] (or [Na, 3] for one light)."""
    out = []
    esun = model.log_esun.exp()
    b_all = light_bin_t(wl)
    for s in range(0, aidx.shape[0], chunk):
        a = aidx[s:s + chunk]; l = wl[s:s + chunk]; b = b_all[s:s + chunk]
        S = R.hit[a]; valid = (S >= 0).float(); Sc = S.clamp(min=0)
        cos = (R.nrm[Sc] * l[:, None, :]).sum(-1).clamp(min=0)
        vis = R.visbit(Sc, b[:, None].expand_as(Sc))
        eind = ecache[Sc] if ecache.dim() == 2 else ecache[Sc, b[:, None].expand_as(Sc)]
        A_s = alb.index_select(0, Sc.reshape(-1)).view(*Sc.shape, 3)        # index_select: backward = index_add (fast)
        L = A_s * (((esun * cos * vis)[..., None] + eind.float()) * (valid / math.pi)[..., None])
        out.append(torch.einsum("mkj,mkc->mjc", R.sh[a].float(), L))
    return torch.cat(out)


def pixel_state(model, R, alb, idx, w, wl, ecache):
    """idx/w [P, 4] (idx -1 = unused), wl [P, 3] -> g [P, 9, 3]."""
    P, n = idx.shape
    use = idx >= 0
    a = idx.clamp(min=0).reshape(-1)
    g = anchor_state(model, R, alb, a, wl[:, None, :].expand(P, n, 3).reshape(-1, 3), ecache).view(P, n, 9, 3)
    return (g * (w * use)[..., None, None]).sum(1)


def digest(model):
    import hashlib

    return {k: hashlib.sha256(v.detach().cpu().numpy().tobytes()).hexdigest()[:16] for k, v in model.state_dict().items()}
