"""WL97 audit track E3: independent float64 port of the frozen ER closure (written from the formula in the docstring of
experiments/fnto_compiled83/closure83_dr.py; validated against the production DrClosure83 by val97_dr.py / cmp97.py).

eval(wi, wo) = f(wi, wo) * cos(theta_o)  (second argument carries the cosine; f is symmetric).
State layout (69 floats): lam 24 (c * 8 + r) | au 7 | av 7 | f0 7 | sampling probs 8 | s 15 (c * 5 + j) | eta 1.
Works with torch tensors (any device, float64). Directions are given as component tuples (x, y, z) in the local shading frame.
"""
import math
import torch

EPS = 1e-7
KNOTS = [0.0, 0.15, 0.4, 0.7, 1.0]
INNER = (3, 4, 5, 6)


class ER:
    def __init__(self, state, device, dtype=torch.float64):
        st = [float(v) for v in state]
        self.lam = [[st[c * 8 + r] for r in range(8)] for c in range(3)]
        self.au, self.av, self.f0 = st[24:31], st[31:38], st[38:45]
        self.s = [[st[53 + c * 5 + j] for j in range(5)] for c in range(3)]
        self.eta = st[68]
        self.device, self.dtype = device, dtype

    def _s(self, mu, c):
        v = torch.full_like(mu, self.s[c][0])
        for j in range(4):
            d = self.s[c][j + 1] - self.s[c][j]
            if d != 0.0:
                v = v + d * torch.clamp((mu - KNOTS[j]) / (KNOTS[j + 1] - KNOTS[j]), 0.0, 1.0)
        return v

    def _refr(self, w):
        ie = 1.0 / self.eta
        wz = torch.clamp(w[2], min=0.0)
        return (w[0] * ie, w[1] * ie, torch.sqrt(max(self.eta * self.eta - 1.0, 0.0) + wz * wz) * ie)

    def _g(self, k, wi, wo):
        au, av, f0 = self.au[k], self.av[k], self.f0[k]
        hx, hy, hz = wi[0] + wo[0], wi[1] + wo[1], wi[2] + wo[2]
        inv = 1.0 / torch.clamp(torch.sqrt(hx * hx + hy * hy + hz * hz), min=1e-6)
        hx, hy, hz = hx * inv, hy * inv, hz * inv
        ci = torch.clamp(wi[2], min=0.0); co = torch.clamp(wo[2], min=0.0)
        den = hx * hx / (au * au) + hy * hy / (av * av) + hz * hz
        D = 1.0 / (math.pi * au * av * den * den)
        si = torch.sqrt(ci * ci + au * au * wi[0] * wi[0] + av * av * wi[1] * wi[1])
        so = torch.sqrt(co * co + au * au * wo[0] * wo[0] + av * av * wo[1] * wo[1])
        m1 = 1.0 - torch.clamp(wi[0] * hx + wi[1] * hy + wi[2] * hz, 0.0, 1.0)
        m2 = m1 * m1
        F = f0 + (1.0 - f0) * (m2 * m2 * m1)
        return F * D / torch.clamp(2.0 * (co * si + ci * so), min=1e-30)

    def eval(self, wi, wo):
        """-> list of 3 tensors (rgb): f * cos(theta_o). Zero where invalid (cos_i, cos_o <= 1e-7)."""
        ci, co = wi[2], wo[2]
        sx, sy, sz = wi[0] + wo[0], wi[1] + wo[1], wi[2] + wo[2]
        valid = (ci > EPS) & (co > EPS) & (torch.sqrt(sx * sx + sy * sy + sz * sz) > 1e-6)
        co_ = torch.clamp(co, min=0.0)
        lamb = co_ / math.pi
        wir, wor = self._refr(wi), self._refr(wo)
        ie2 = 1.0 / (self.eta * self.eta)
        T = []
        for k in range(7):
            if all(self.lam[c][k + 1] == 0.0 for c in range(3)):
                T.append(None); continue
            if k in INNER:
                T.append(self._g(k, wir, wor) * co_ * ie2)
            else:
                T.append(self._g(k, wi, wo) * co_)
        out = []
        for c in range(3):
            f_out = torch.zeros_like(co_); f_in = self.lam[c][0] * lamb
            for k in range(7):
                if T[k] is None or self.lam[c][k + 1] == 0.0:
                    continue
                if k in INNER:
                    f_in = f_in + self.lam[c][k + 1] * T[k]
                else:
                    f_out = f_out + self.lam[c][k + 1] * T[k]
            f = f_out + self._s(ci, c) * self._s(co, c) * f_in
            out.append(torch.where(valid, f, torch.zeros_like(f)))
        return out
