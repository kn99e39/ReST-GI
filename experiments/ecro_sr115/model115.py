"""WL115 learned candidate ECRO-CAR: model113 ECROL4 (dense 4 x 4 head and every shared parameter unchanged and constructed FIRST, so a
fixed seed gives the identical shared initialisation as the WL114 B1 / B2 arms) + one refinement head on the CAR-16 active cells.

Refinement head input per (active cell a = (patch p, 4 x 4 cell c), child k of 16, outgoing bin d):
  patch token + bin embedding (as head_a / head_s), the learned 4 x 4 sub-cell embedding of c (shared with head_s), a learned child
  embedding (16 x 8), the parent cell's 3 VHS descriptors, the child's 5 geometric descriptors (vhs_k(16) near / logd / esc / prox and
  other-object proximity). Geometry only: no radiance, camera or reference input.
Output a_child [A, 16, NB, 3, R] >= 0 (softplus); child table = a_child . y with the SAME channel activations y as the dense head
(linear in the source, source independent). The readout blends (1 - w) T4 + w T16 with the geometric activation w (candidate115).
"""
import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as Fn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'ecro_sss113')); sys.path.insert(0, str(HERE.parent / 'ecro86'))
import model113 as M3  # noqa: E402
import model86 as M  # noqa: E402

NB = M3.NB
NCH = 16
CH_DIM = 8
NPD = 3
NCD = 5


class ECROCAR(M3.ECROL4):
    def __init__(self):
        super().__init__()
        d = M.A['token_dim']
        self.child_emb = nn.Parameter(torch.randn(NCH, CH_DIM) * 0.1)
        self.head_r = M.mlp(d + 16 + M3.SUB_DIM + CH_DIM + NPD + NCD, d, 3 * self.R, 2)
        with torch.no_grad():
            self.head_r[-1].bias.fill_(-2.0)

    def n_shared(self):
        return len(list(M3.ECROL4.parameters(self))) - 2      # parameters() order: ECROL4's, then child_emb, head_r (counted below)

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
            tbs = tb[sup]
            desc = gs.desc[sup]
            z = torch.cat([tbs.view(Rn, 1, NB, -1).expand(Rn, M3.NS, NB, tbs.shape[-1]),
                           self.sub_emb.view(1, M3.NS, 1, M3.SUB_DIM).expand(Rn, M3.NS, NB, M3.SUB_DIM),
                           desc.view(Rn, M3.NS, 1, M3.NDESC).expand(Rn, M3.NS, NB, M3.NDESC)], 3)
            f['a_sub'] = Fn.softplus(self.head_s(z)).view(Rn, M3.NS, NB, 3, self.R)
        A_ = gs.act_pid.numel() if getattr(gs, 'act_pid', None) is not None else 0
        if A_:
            ta = tb[gs.act_pid]                                                        # [A, NB, d+16]
            z = torch.cat([ta.view(A_, 1, NB, -1).expand(A_, NCH, NB, ta.shape[-1]),
                           self.sub_emb[gs.act_sub].view(A_, 1, 1, M3.SUB_DIM).expand(A_, NCH, NB, M3.SUB_DIM),
                           self.child_emb.view(1, NCH, 1, CH_DIM).expand(A_, NCH, NB, CH_DIM),
                           gs.desc[gs.act_pid, gs.act_sub].view(A_, 1, 1, NPD).expand(A_, NCH, NB, NPD),
                           gs.cdesc.view(A_, NCH, 1, NCD).expand(A_, NCH, NB, NCD)], 3)
            f['a_ch'] = Fn.softplus(self.head_r(z)).view(A_, NCH, NB, 3, self.R)
        return f

    def outputs_car(self, f, gs, s, use_single=True, use_global=True):
        """(coarse B [L,N,NB,3], dense B_sub [L,N,16,NB,3], child B_ch [L,A,16,NB,3]) from one activation pass."""
        y = self.activations(f, gs, s, use_single, use_global)
        B = torch.einsum('ndcr,lncr->lndc', f['a'], y)
        Bs = torch.einsum('rsdcq,lrcq->lrsdc', f['a_sub'], y[:, gs.sup]) if 'a_sub' in f else None
        Bc = torch.einsum('akdcq,lacq->lakdc', f['a_ch'], y[:, gs.act_pid]) if 'a_ch' in f else None
        return B, Bs, Bc
