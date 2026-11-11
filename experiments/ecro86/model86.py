"""WL86 ECRO v0 operator (torch): geometry-conditioned, source-independent, linear and non-negative.

Geometry state (no radiance): patch tables, frozen ER states of the patch materials, stratified hemisphere state rays
(hit patch, outgoing bin at the hit patch) -> generator network -> response factors
  c [N, NB, 3, R]   incident-bin -> channel activations   (>= 0)
  a [N, NB, 3, R]   channel activations -> outgoing-bin radiance (>= 0)
  Q [N, 3, RHO, R], P [N, 3, R, RHO]   global encoder / decoder (>= 0)
Application to a source field s [L, N, NB, 3] (direct irradiance by incident bin; never an input of the generator):
  u   = c . s                                   sender activations
  Bq  = a . u                                   learned once-scattered outgoing radiance by bin
  E1  = sum over the state rays of Bq[hit, bin] cos dOmega      (single learned flight, current visibility)
  u1  = c . E1
  yg  = P . (sum_j area_j Q_j . u_j)            low-rank global multi-flight term
  B   = a . (u1 + yg)                           outgoing radiance of 2..5 scattering events per patch and outgoing bin
Linear in s with non-negative coefficients: zero input -> zero, superposition exact, non-negative output.
"""
import math, os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn

import scene86 as S

HERE = os.path.dirname(os.path.abspath(__file__))
NB, K, KAPPA = S.NB, S.K, S.KAPPA
A = S.P['architecture']


def mlp(i, h, o, layers=2):
    mods = []
    d = i
    for _ in range(layers):
        mods += [nn.Linear(d, h), nn.SiLU()]
        d = h
    mods.append(nn.Linear(d, o))
    return nn.Sequential(*mods)


def ray_weights():
    """cos(theta) dOmega / kappa of the K state rays (shared by all patches) -> [K]."""
    rs = np.random.default_rng(S.P['representation'].get('table_seed', 86001))
    us = rs.random((4, 1, K), np.float32)
    k = np.arange(K)
    z = ((k // KAPPA) // S.NPHI + us[2, 0]) / S.NZ
    return (z * S.DOMEGA / KAPPA).astype(np.float32)


class GeoState:
    """Tensors of one geometry on a device."""

    def __init__(self, geom, hit, dbin, dev):
        g = S.Geom(geom) if isinstance(geom, dict) else geom
        t = lambda a, dt=torch.float32: torch.as_tensor(np.asarray(a), dtype=dt, device=dev)
        self.N = g.N
        self.cen, self.nrm, self.tan, self.area, self.mat = t(g.PC), t(g.PN), t(g.PT), t(g.PAREA), t(g.PMAT, torch.long)
        self.hit = t(hit, torch.long).view(g.N, K)
        self.dbin = t(dbin, torch.long).view(g.N, K)


class ECRO(nn.Module):
    def __init__(self):
        super().__init__()
        d, self.R, self.RHO, self.rounds = A['token_dim'], A['channels_R'], A['global_rank'], A['message_rounds']
        mats = np.load(os.path.join(HERE, 'data', 'mats86.npz'))['states']
        self.register_buffer('mat_states', torch.tensor(mats))
        self.register_buffer('bin_dirs', torch.tensor(S.bin_dirs(), dtype=torch.float32))
        self.register_buffer('rw', torch.tensor(ray_weights()))
        self.mat_mlp = mlp(69, 64, 16, 1)
        self.inp = mlp(3 + 3 + 3 + 1 + 16 + 1, d, d, 1)
        self.msg = nn.ModuleList([nn.Linear(d, d) for _ in range(self.rounds)])
        self.glb = nn.ModuleList([nn.Linear(d, d) for _ in range(self.rounds)])
        self.upd = nn.ModuleList([mlp(3 * d, d, d, 1) for _ in range(self.rounds)])
        self.bin_emb = nn.Parameter(torch.randn(NB, 13) * 0.1)
        self.head_c = mlp(d + 16, d, 3 * self.R, 2)
        self.head_a = mlp(d + 16, d, 3 * self.R, 2)
        self.head_Q = mlp(d, d, 3 * self.RHO * self.R, 1)
        self.head_P = mlp(d, d, 3 * self.R * self.RHO, 1)
        with torch.no_grad():
            self.head_c[-1].bias.fill_(-2.0); self.head_a[-1].bias.fill_(-2.0)
            self.head_Q[-1].bias.fill_(-6.0); self.head_P[-1].bias.fill_(-6.0)

    def form_factors(self, gs):
        """Cosine-weighted visibility form factors from the state rays: Ft[i, j] (rows sum to <= 1)."""
        N = gs.N
        w = (self.rw / math.pi).view(1, K).expand(N, K)
        ok = gs.hit >= 0
        idx = (torch.arange(N, device=w.device).view(N, 1) * N + gs.hit.clamp(min=0))[ok]
        return torch.zeros(N * N, device=w.device).index_put_((idx,), w[ok], accumulate=True).view(N, N)

    def factors(self, gs):
        N = gs.N
        Ft = self.form_factors(gs)
        opn = 1.0 - Ft.sum(1, keepdim=True)
        x = torch.cat([gs.cen * 0.5, gs.nrm, gs.tan, torch.log(gs.area).view(N, 1) + 3.0, self.mat_mlp(self.mat_states[gs.mat]), opn], 1)
        t = self.inp(x)
        for l in range(self.rounds):
            m = Ft @ self.msg[l](t)
            gm = self.glb[l](t).mean(0, keepdim=True).expand(N, -1)
            t = t + self.upd[l](torch.cat([t, m, gm], 1))
        be = torch.cat([self.bin_dirs, self.bin_emb], 1)                              # [NB, 16]
        tb = torch.cat([t.view(N, 1, -1).expand(N, NB, -1), be.view(1, NB, 16).expand(N, NB, 16)], 2)
        c = Fn.softplus(self.head_c(tb)).view(N, NB, 3, self.R)
        a = Fn.softplus(self.head_a(tb)).view(N, NB, 3, self.R)
        Q = Fn.softplus(self.head_Q(t)).view(N, 3, self.RHO, self.R)
        Pm = Fn.softplus(self.head_P(t)).view(N, 3, self.R, self.RHO)
        return {'c': c, 'a': a, 'Q': Q, 'P': Pm}

    def apply(self, f, gs, s, use_single=True, use_global=True):
        """s [L, N, NB, 3] -> B [L, N, NB, 3]."""
        Ls, N = s.shape[0], gs.N
        u = torch.einsum('nmcr,lnmc->lncr', f['c'], s)
        y = torch.zeros_like(u)
        if use_single:
            Bq = torch.einsum('ndcr,lncr->lndc', f['a'], u).reshape(Ls, N * NB, 3)
            ok = (gs.hit >= 0).view(1, N * K, 1)
            idx = (gs.hit.clamp(min=0) * NB + gs.dbin).view(-1)
            Lk = Bq[:, idx] * ok
            E1 = (Lk.view(Ls, N, NB, KAPPA, 3) * self.rw.view(1, 1, NB, KAPPA, 1)).sum(3)
            y = y + torch.einsum('nmcr,lnmc->lncr', f['c'], E1)
        if use_global:
            wq = torch.einsum('ncpr,lncr,n->lcp', f['Q'], u, gs.area)
            y = y + torch.einsum('ncrp,lcp->lncr', f['P'], wq)
        return torch.einsum('ndcr,lncr->lndc', f['a'], y)

    def forward(self, gs, s, **kw):
        return self.apply(self.factors(gs), gs, s, **kw)


def loss_fn(pred, tgt):
    """Relative L2 with a denominator from the target's patch mean over bins (64-bin average: low-noise) + a floor."""
    den = tgt.mean(2, keepdim=True) + 0.05 * tgt.mean(dim=(1, 2, 3), keepdim=True)
    return (((pred - tgt) / den) ** 2).mean()


def rel_l1(pred, tgt):
    return float((pred - tgt).abs().sum() / tgt.sum().clamp(min=1e-30))
