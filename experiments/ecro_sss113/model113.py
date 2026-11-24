"""WL113 candidate ECRO-L4: ECRO v0 (model86, unchanged coarse path) + a 4x4 sub-patch head on supported patches.

Sub-head input per (supported patch, sub-cell, outgoing bin): the patch token, the bin embedding (as head_a), a learned 8-d
sub-cell index embedding and the 3 geometric VHS descriptors of the sub-cell (near fraction, mean log10 hit distance, escape
fraction; geometry only). Output a_sub [R, 16, NB, 3, Rch] >= 0; refined table B_sub = a_sub . (u1 + yg) with the SAME channel
activations as the coarse output (linear, non-negative, source independent). Coarse factors (c, a, Q, P) and the internal single
learned flight are model86's. Shared parameters are constructed first, so a fixed seed initialises them identically to model86.
"""
import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as Fn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'ecro86'))
import model86 as M  # noqa: E402
import scene86 as S  # noqa: E402

NB, K, KAPPA = S.NB, S.K, S.KAPPA
NS = 16
SUB_DIM = 8
NDESC = 3


class ECROL4(M.ECRO):
    def __init__(self):
        super().__init__()
        d = M.A['token_dim']
        self.sub_emb = nn.Parameter(torch.randn(NS, SUB_DIM) * 0.1)
        self.head_s = M.mlp(d + 16 + SUB_DIM + NDESC, d, 3 * self.R, 2)
        with torch.no_grad():
            self.head_s[-1].bias.fill_(-2.0)

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
        be = torch.cat([self.bin_dirs, self.bin_emb], 1)
        tb = torch.cat([t.view(N, 1, -1).expand(N, NB, -1), be.view(1, NB, 16).expand(N, NB, 16)], 2)
        f = {'c': Fn.softplus(self.head_c(tb)).view(N, NB, 3, self.R), 'a': Fn.softplus(self.head_a(tb)).view(N, NB, 3, self.R),
             'Q': Fn.softplus(self.head_Q(t)).view(N, 3, self.RHO, self.R), 'P': Fn.softplus(self.head_P(t)).view(N, 3, self.R, self.RHO)}
        sup = gs.sup
        Rn = sup.numel()
        if Rn:
            tbs = tb[sup]                                                     # [R, NB, d+16]
            desc = gs.desc[sup]                                               # [R, NS, 3]
            z = torch.cat([tbs.view(Rn, 1, NB, -1).expand(Rn, NS, NB, tbs.shape[-1]),
                           self.sub_emb.view(1, NS, 1, SUB_DIM).expand(Rn, NS, NB, SUB_DIM),
                           desc.view(Rn, NS, 1, NDESC).expand(Rn, NS, NB, NDESC)], 3)
            f['a_sub'] = Fn.softplus(self.head_s(z)).view(Rn, NS, NB, 3, self.R)
        return f

    def activations(self, f, gs, s, use_single=True, use_global=True):
        """Channel activations y [L, N, 3, R] = c . E1 (single learned flight) + P . (sum area Q . u) (model86.apply)."""
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
        return y

    def apply(self, f, gs, s, use_single=True, use_global=True):
        y = self.activations(f, gs, s, use_single, use_global)
        return torch.einsum('ndcr,lncr->lndc', f['a'], y)

    def apply_sub(self, f, gs, s, use_single=True, use_global=True):
        """Refined outputs of the supported patches [L, R, 16, NB, 3]."""
        y = self.activations(f, gs, s, use_single, use_global)[:, gs.sup]
        return torch.einsum('rsdcq,lrcq->lrsdc', f['a_sub'], y)

    def outputs(self, f, gs, s, use_single=True, use_global=True):
        """(coarse B [L,N,NB,3], refined B_sub [L,R,16,NB,3]) from one activation pass."""
        y = self.activations(f, gs, s, use_single, use_global)
        B = torch.einsum('ndcr,lncr->lndc', f['a'], y)
        Bs = torch.einsum('rsdcq,lrcq->lrsdc', f['a_sub'], y[:, gs.sup]) if 'a_sub' in f else None
        return B, Bs
