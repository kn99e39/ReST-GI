"""WL93 local-operator arms on the unchanged ECRO generator (model86), with the WL92A OWN global branch frozen.

B = S_theta(s) + G_frozen(s)
  G_frozen : the global term of the WL92A own_s<seed> model, evaluated with its own (frozen) factors - identical in every arm
  S_theta  : single-flight term  a_r . (c_r . flight(a_s . (c_s . s)))  produced by a trainable copy of the OWN generator
modes
  tied    c_s = c_r = c, a_s = a_r = a  (heads head_c, head_a; the WL86 / WL92A local operator)
  untied  source factors from (head_c, head_a), receiver factors from duplicated heads (head_c2, head_a2); shared trunk
  tiedcm  capacity-matched tied control: one shared (c, a) = softplus(head(tb) + head2(tb)) - an additive second head of the same
          architecture (hidden layers freshly initialised from a fixed seed, output layer zero) -> a 2x-wide tied head with the
          untied arm's trainable parameter count. (A copy-initialised averaged pair was rejected before the freeze: it is symmetric,
          receives identical updates under Adam and reproduces the tied arm exactly - measured in the dev chain.)
All three equal the WL92A OWN generator at step 0 (untied: duplicated heads are copies; tiedcm: the added head outputs zero).
Linear in s, non-negative, zero -> zero (softplus factors, linear application).
"""
import copy, math
import torch
import torch.nn as nn
import torch.nn.functional as Fn

import model86 as M
import scene86 as S

NB, K, KAPPA = S.NB, S.K, S.KAPPA


def trunk(m, gs):
    """ECRO.factors up to the head input (verbatim copy of model86.ECRO.factors before the heads)."""
    N = gs.N
    Ft = m.form_factors(gs)
    opn = 1.0 - Ft.sum(1, keepdim=True)
    x = torch.cat([gs.cen * 0.5, gs.nrm, gs.tan, torch.log(gs.area).view(N, 1) + 3.0, m.mat_mlp(m.mat_states[gs.mat]), opn], 1)
    t = m.inp(x)
    for l in range(m.rounds):
        msg = Ft @ m.msg[l](t)
        gm = m.glb[l](t).mean(0, keepdim=True).expand(N, -1)
        t = t + m.upd[l](torch.cat([t, msg, gm], 1))
    be = torch.cat([m.bin_dirs, m.bin_emb], 1)
    tb = torch.cat([t.view(N, 1, -1).expand(N, NB, -1), be.view(1, NB, 16).expand(N, NB, 16)], 2)
    return t, tb


class Local(nn.Module):
    def __init__(self, base, mode, cm_seed=0):
        super().__init__()
        assert mode in ('tied', 'untied', 'tiedcm')
        self.mode = mode
        self.g = copy.deepcopy(base)
        if mode != 'tied':
            self.head_c2 = copy.deepcopy(base.head_c)
            self.head_a2 = copy.deepcopy(base.head_a)
        if mode == 'tiedcm':
            gen = torch.Generator().manual_seed(int(cm_seed))
            for h in (self.head_c2, self.head_a2):
                lin = [l for l in h if isinstance(l, nn.Linear)]
                with torch.no_grad():
                    for l in lin[:-1]:                     # nn.Linear default init bound 1/sqrt(fan_in), from the fixed generator
                        b = 1.0 / math.sqrt(l.in_features)
                        l.weight.copy_((torch.rand(l.weight.shape, generator=gen) * 2 - 1) * b)
                        l.bias.copy_((torch.rand(l.bias.shape, generator=gen) * 2 - 1) * b)
                    lin[-1].weight.zero_(); lin[-1].bias.zero_()
        # the copy is trainable regardless of the base's flags (the frozen OWN base is never touched) ...
        for p in self.parameters():
            p.requires_grad_(True)
        # ... except the generator's global heads, unused here (global term comes from the frozen OWN model)
        for p in list(self.g.head_Q.parameters()) + list(self.g.head_P.parameters()):
            p.requires_grad_(False)

    def factors(self, gs):
        N, R = gs.N, self.g.R
        _, tb = trunk(self.g, gs)
        sp = lambda h: Fn.softplus(h).view(N, NB, 3, R)
        if self.mode == 'tied':
            c = sp(self.g.head_c(tb)); a = sp(self.g.head_a(tb))
            return {'c_s': c, 'a_s': a, 'c_r': c, 'a_r': a}
        if self.mode == 'tiedcm':
            c = sp(self.g.head_c(tb) + self.head_c2(tb)); a = sp(self.g.head_a(tb) + self.head_a2(tb))
            return {'c_s': c, 'a_s': a, 'c_r': c, 'a_r': a}
        return {'c_s': sp(self.g.head_c(tb)), 'a_s': sp(self.g.head_a(tb)), 'c_r': sp(self.head_c2(tb)), 'a_r': sp(self.head_a2(tb))}

    def single(self, f, gs, s):
        """s [L, N, NB, 3] -> (S [L, N, NB, 3], Bq [L, N, NB, 3])."""
        Ls, N = s.shape[0], gs.N
        u = torch.einsum('nmcr,lnmc->lncr', f['c_s'], s)
        Bq = torch.einsum('ndcr,lncr->lndc', f['a_s'], u)
        ok = (gs.hit >= 0).view(1, N * K, 1)
        idx = (gs.hit.clamp(min=0) * NB + gs.dbin).view(-1)
        Lk = Bq.reshape(Ls, N * NB, 3)[:, idx] * ok
        E1 = (Lk.view(Ls, N, NB, KAPPA, 3) * self.g.rw.view(1, 1, NB, KAPPA, 1)).sum(3)
        S_ = torch.einsum('ndcr,lncr->lndc', f['a_r'], torch.einsum('nmcr,lnmc->lncr', f['c_r'], E1))
        return S_, Bq, E1

    def trainable(self):
        return [p for p in self.parameters() if p.requires_grad]

    def n_trainable(self):
        return sum(p.numel() for p in self.trainable())


@torch.no_grad()
def global_term(own, gs, s):
    return own.apply(own.factors(gs), gs, s, use_single=False)


def den_of(t):
    return t.mean(2, keepdim=True) + 0.05 * t.mean(dim=(1, 2, 3), keepdim=True)


def losses(S_, G, Bq, t, t2, q):
    den = den_of(t)
    l_sum = (((S_ + G - t) / den) ** 2).mean()
    l_s = (((S_ - t2) / den) ** 2).mean()
    l_g = (((G - (t - t2)) / den) ** 2).mean()
    l_q = (((Bq - q) / den_of(q)) ** 2).mean() if q is not None else torch.zeros((), device=t.device)
    return l_sum, l_s, l_g, l_q
