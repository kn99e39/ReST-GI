"""WL49 candidate DASQ - receiver-centred distance-adaptive source quadrature (amendment 3).

A Morton-ordered octree over the Lego ports; for a receiver x, a node is ACCEPTED as one element when its cell diagonal
<= alpha * |x - centroid| (and the receiver lies outside its cell); support membership is decided per element by its
centroid distance < R (amendment 4: like K44's mid cells; the element-level boundary is part of the candidate's
approximation error against the exact r < R target); single-port nodes are leaves (included if r < R, not self). Element feature = area-mean field over its contiguous Morton range.
"""
import math

import torch

MAXLEV = 21


def morton(q):
    code = torch.zeros(q.shape[0], dtype=torch.int64, device=q.device)
    for b in range(MAXLEV):
        for a in range(3):
            code |= ((q[:, a] >> b) & 1) << (3 * b + a)
    return code


class Octree:
    def __init__(self, pos, area):
        dev = pos.device; self.dev = dev; N = pos.shape[0]
        lo = pos.min(0).values; span = float((pos.max(0).values - lo).max()) * 1.0001
        q = ((pos - lo) / span * (1 << MAXLEV)).long().clamp(0, (1 << MAXLEV) - 1)
        code = morton(q); self.order = torch.argsort(code); cs = code[self.order]
        self.inv = torch.empty_like(self.order); self.inv[self.order] = torch.arange(N, device=dev)
        a = area[self.order].double(); p = pos[self.order].double()
        self.csA = torch.cat([torch.zeros(1, device=dev, dtype=torch.float64), torch.cumsum(a, 0)])
        self.csAP = torch.cat([torch.zeros(1, 3, device=dev, dtype=torch.float64), torch.cumsum(a[:, None] * p, 0)])
        self.st, self.en, self.cen, self.diag, self.area = [], [], [], [], []
        for l in range(MAXLEV + 1):
            pre = cs >> (3 * (MAXLEV - l)); chg = torch.ones(N, dtype=torch.bool, device=dev); chg[1:] = pre[1:] != pre[:-1]
            st = torch.nonzero(chg)[:, 0]; en = torch.cat([st[1:], torch.tensor([N], device=dev)])
            ar = self.csA[en] - self.csA[st]
            self.st.append(st); self.en.append(en); self.area.append(ar)
            self.cen.append(((self.csAP[en] - self.csAP[st]) / ar.clamp(min=1e-30)[:, None]).float())
            self.diag.append(math.sqrt(3) * span / (1 << l))
        self.clo, self.chi = [], []
        for l in range(MAXLEV):
            self.clo.append(torch.searchsorted(self.st[l + 1], self.st[l])); self.chi.append(torch.searchsorted(self.st[l + 1], self.en[l]))

    @torch.no_grad()
    def elements(self, x, self_port, alpha, R):
        """x [n, 3], self_port [n] (port index or -1) -> list of (st, en) per receiver as padded tensors [n, K] (+ count)."""
        dev = self.dev; n = x.shape[0]
        fr = torch.arange(n, device=dev); fn = torch.zeros(n, dtype=torch.long, device=dev)
        self_pos = torch.where(self_port >= 0, self.inv[self_port.clamp(min=0)], torch.full_like(self_port, -1))
        acc_r, acc_s, acc_e = [], [], []
        for l in range(MAXLEV + 1):
            if fr.numel() == 0:
                break
            st, en = self.st[l][fn], self.en[l][fn]; c = self.cen[l][fn]; dg = self.diag[l]
            d = (x[fr] - c).norm(dim=-1); single = (en - st) == 1
            leaf_ok = single & (d < R) & (st != self_pos[fr])
            outside = (d - dg >= R)                                    # no member can be in support
            ok_size = ~single & (dg <= alpha * d)                      # receiver outside the cell (alpha <= 1 or d > dg)
            acc = ok_size & (d < R) & (d > dg)                         # element-level support membership by centroid (amendment 4)
            drop = ok_size & ~acc & (d > dg)
            desc = ~single & ~outside & ~acc & ~drop
            if l == MAXLEV:
                acc = acc | desc; desc = torch.zeros_like(desc)
            for m in (leaf_ok, acc):
                acc_r.append(fr[m]); acc_s.append(st[m]); acc_e.append(en[m])
            if desc.any():
                dn = fn[desc]; lo, hi = self.clo[l][dn], self.chi[l][dn]; cnt = hi - lo
                fr = torch.repeat_interleave(fr[desc], cnt)
                off = torch.arange(int(cnt.sum()), device=dev) - torch.repeat_interleave(torch.cumsum(cnt, 0) - cnt, cnt)
                fn = torch.repeat_interleave(lo, cnt) + off
            else:
                fr = fr[:0]
        r = torch.cat(acc_r); s = torch.cat(acc_s); e = torch.cat(acc_e)
        o = torch.argsort(r * (1 << 40) + s); r, s, e = r[o], s[o], e[o]
        cnt = torch.bincount(r, minlength=n); K = int(cnt.max())
        idx = torch.arange(len(r), device=dev) - torch.repeat_interleave(torch.cumsum(cnt, 0) - cnt, cnt)
        S = torch.full((n, K), -1, dtype=torch.long, device=dev); E = torch.full((n, K), -1, dtype=torch.long, device=dev)
        S[r, idx] = s; E[r, idx] = e
        return S, E, cnt

    def area_of(self, S, E):
        ok = S >= 0
        return torch.where(ok, self.csA[E.clamp(min=0)] - self.csA[S.clamp(min=0)], torch.zeros_like(self.csA[:1]))

    def means(self, field_sorted_cs, S, E):
        """field_sorted_cs = cumsum(A*E) in Morton order [N+1, F] (float64) -> element area-means [n, K, F]."""
        ok = (S >= 0)
        num = field_sorted_cs[E.clamp(min=0)] - field_sorted_cs[S.clamp(min=0)]
        ar = (self.csA[E.clamp(min=0)] - self.csA[S.clamp(min=0)]).clamp(min=1e-30)
        return num / ar[..., None] * ok[..., None]

    def cs_field(self, area, E):
        AE = (area[:, None].double() * E.double())[self.order]
        return torch.cat([torch.zeros(1, AE.shape[1], device=self.dev, dtype=torch.float64), torch.cumsum(AE, 0)])

    @torch.no_grad()
    def ideal_weights(self, pos, area, x, self_port, S, E, kern, R, chunk=8):
        """Exact element weight w = sum_{s in e} k(x, s) A_s (self excluded, r < R): prediction = sum_e w_e * area-mean field_e
        (exact for fields uniform within each element)."""
        n, K = S.shape; W = torch.zeros(n, K, device=self.dev, dtype=torch.float64)
        posS = pos[self.order]; aS = area[self.order].double()
        for i0 in range(0, n, chunk):
            i1 = min(i0 + chunk, n)
            r = torch.cdist(x[i0:i1], posS).clamp(min=2e-3); k = kern(r) * (r < R)
            sp = self_port[i0:i1]
            if (sp >= 0).any():
                k[torch.arange(i1 - i0, device=self.dev)[sp >= 0], self.inv[sp[sp >= 0]]] = 0
            for j in range(i1 - i0):
                st, en = S[i0 + j], E[i0 + j]; ok = st >= 0
                ka = torch.cat([torch.zeros(1, device=self.dev, dtype=torch.float64), torch.cumsum(k[j].double() * aS, 0)])
                W[i0 + j, ok] = ka[en[ok]] - ka[st[ok]]
        return W
