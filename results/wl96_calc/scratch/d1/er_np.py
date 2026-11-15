"""numpy port of the WL83 ER closure eval (semantics copied from experiments/fnto_compiled83/closure83_dr.py; scratch, WL96 design calc)."""
import math
import numpy as np

INV_PI = 1.0 / math.pi
EPS = 1e-7
KNOTS = [0.0, 0.15, 0.4, 0.7, 1.0]
OUTER = [0, 1, 2]
INNER = [3, 4, 5, 6]


class ER:
    def __init__(self, st):
        st = np.asarray(st, np.float64)
        self.lam = st[:24].reshape(3, 8)
        self.au, self.av, self.f0 = st[24:31], st[31:38], st[38:45]
        self.pr = st[45:53]
        self.s = st[53:68].reshape(3, 5)
        self.eta = st[68]

    def _s(self, mu):
        out = []
        for c in range(3):
            v = np.full_like(mu, self.s[c, 0])
            for j in range(4):
                t = np.clip((mu - KNOTS[j]) / (KNOTS[j + 1] - KNOTS[j]), 0, 1)
                v = v + (self.s[c, j + 1] - self.s[c, j]) * t
            out.append(v)
        return np.stack(out, -1)

    def _refr(self, w):
        ie = 1.0 / self.eta
        wz = np.maximum(w[..., 2], 0)
        return np.stack([w[..., 0] * ie, w[..., 1] * ie, np.sqrt(max(self.eta ** 2 - 1, 0) + wz * wz) * ie], -1)

    def _lobe(self, k, wi, wo):
        hs = wi + wo
        h = hs / np.maximum(np.linalg.norm(hs, axis=-1, keepdims=True), 1e-6)
        ci = np.maximum(wi[..., 2], 0); co = np.maximum(wo[..., 2], 0)
        au, av = self.au[k], self.av[k]
        den = h[..., 0] ** 2 / au ** 2 + h[..., 1] ** 2 / av ** 2 + h[..., 2] ** 2
        D = INV_PI / (au * av) / (den * den)
        si = np.sqrt(ci * ci + au ** 2 * wi[..., 0] ** 2 + av ** 2 * wi[..., 1] ** 2)
        so = np.sqrt(co * co + au ** 2 * wo[..., 0] ** 2 + av ** 2 * wo[..., 1] ** 2)
        m1 = 1.0 - np.clip((wi * h).sum(-1), 0, 1)
        F = self.f0[k] + (1 - self.f0[k]) * m1 ** 5
        return F * D / np.maximum(2 * (co * si + ci * so), 1e-30)

    def eval(self, wi, wo):
        """eval(wi, wo) = f cos(theta_o) [..., 3] (local frames, z = normal)."""
        ci, co = wi[..., 2], wo[..., 2]
        valid = (ci > EPS) & (co > EPS) & (np.linalg.norm(wi + wo, axis=-1) > 1e-6)
        co_ = np.maximum(co, 0)
        wi_r, wo_r = self._refr(wi), self._refr(wo)
        ie2 = 1.0 / self.eta ** 2
        f_out = 0; f_in = 0
        for k in range(7):
            refr = k in INNER
            g = self._lobe(k, wi_r if refr else wi, wo_r if refr else wo)
            term = self.lam[:, k + 1] * (g * co_ * (ie2 if refr else 1.0))[..., None]
            if refr:
                f_in = f_in + term
            else:
                f_out = f_out + term
        f = f_out + self._s(ci) * self._s(co) * (f_in + self.lam[:, 0] * (co_ * INV_PI)[..., None])
        return np.where(valid[..., None], f, 0.0)
