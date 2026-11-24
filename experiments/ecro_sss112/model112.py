"""WL112 candidate ECRO-S2: ECRO v0 (model86) with a 2 x 2 sub-patch outgoing decoder.

Minimum output adaptation: head_a additionally receives a learned 8-d sub-patch embedding and produces a_sub [N, 4, NB, 3, R] (>= 0).
The patch-level outgoing factor used inside the operator (single learned flight Bq) is the mean of a_sub over the four equal-area
sub-patches. The output table is B_sub = a_sub . (u1 + yg) [L, N, 4, NB, 3]. Message passing, c, Q, P, source encoding, geometry
state and the linear / non-negative / source-independent application are unchanged.
"""
import math, os, sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as Fn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'ecro86'))
import model86 as M  # noqa: E402
import scene86 as S  # noqa: E402

NB, K, KAPPA = S.NB, S.K, S.KAPPA
NSUB = 4
SUB_DIM = 8


class ECROS2(M.ECRO):
    def __init__(self):
        super().__init__()
        d = M.A['token_dim']
        self.sub_emb = nn.Parameter(torch.randn(NSUB, SUB_DIM) * 0.1)
        self.head_a = M.mlp(d + 16 + SUB_DIM, d, 3 * self.R, 2)
        with torch.no_grad():
            self.head_a[-1].bias.fill_(-2.0)

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
        c = Fn.softplus(self.head_c(tb)).view(N, NB, 3, self.R)
        tb4 = torch.cat([tb.view(N, 1, NB, -1).expand(N, NSUB, NB, -1),
                         self.sub_emb.view(1, NSUB, 1, SUB_DIM).expand(N, NSUB, NB, SUB_DIM)], 3)
        a_sub = Fn.softplus(self.head_a(tb4)).view(N, NSUB, NB, 3, self.R)
        Q = Fn.softplus(self.head_Q(t)).view(N, 3, self.RHO, self.R)
        Pm = Fn.softplus(self.head_P(t)).view(N, 3, self.R, self.RHO)
        return {'c': c, 'a': a_sub.mean(1), 'a_sub': a_sub, 'Q': Q, 'P': Pm}

    def apply(self, f, gs, s, use_single=True, use_global=True):
        """s [L, N, NB, 3] -> B_sub [L, N, 4, NB, 3]."""
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
        return torch.einsum('nsdcr,lncr->lnsdc', f['a_sub'], y)
