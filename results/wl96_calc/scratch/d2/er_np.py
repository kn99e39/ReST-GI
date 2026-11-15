"""numpy port of closure83_dr ER eval (arm ER) for one material state [69]; eval(wi, wo) = f cos(theta_o), local frames (z = normal)."""
import numpy as np
KN = [0.0, 0.15, 0.4, 0.7, 1.0]; INNER = [3, 4, 5, 6]
class ER:
    def __init__(self, st):
        st = np.asarray(st, np.float64)
        self.lam = st[:24].reshape(3, 8); self.au = st[24:31]; self.av = st[31:38]; self.f0 = st[38:45]
        self.s = st[53:68].reshape(3, 5); self.eta = st[68]
    def _s(self, mu):
        out = []
        for c in range(3):
            v = np.full_like(mu, self.s[c, 0])
            for j in range(4):
                t = np.clip((mu - KN[j]) / (KN[j + 1] - KN[j]), 0, 1)
                v = v + (self.s[c, j + 1] - self.s[c, j]) * t
            out.append(v)
        return np.stack(out, -1)
    def _refr(self, w):
        e = self.eta; wz = np.maximum(w[..., 2], 0)
        return np.stack([w[..., 0] / e, w[..., 1] / e, np.sqrt(max(e * e - 1, 0) + wz * wz) / e], -1)
    def _lobe(self, k, wi, wo):
        hs = wi + wo; h = hs / np.maximum(np.linalg.norm(hs, axis=-1, keepdims=True), 1e-6)
        ci = np.maximum(wi[..., 2], 0); co = np.maximum(wo[..., 2], 0)
        au, av = self.au[k], self.av[k]
        den = h[..., 0] ** 2 / au ** 2 + h[..., 1] ** 2 / av ** 2 + h[..., 2] ** 2
        D = 1 / (np.pi * au * av) / den ** 2
        si = np.sqrt(ci * ci + au ** 2 * wi[..., 0] ** 2 + av ** 2 * wi[..., 1] ** 2)
        so = np.sqrt(co * co + au ** 2 * wo[..., 0] ** 2 + av ** 2 * wo[..., 1] ** 2)
        m1 = 1 - np.clip((wi * h).sum(-1), 0, 1); F = self.f0[k] + (1 - self.f0[k]) * m1 ** 5
        return F * D / np.maximum(2 * (co * si + ci * so), 1e-30)
    def eval(self, wi, wo):
        ci, co = wi[..., 2], wo[..., 2]
        valid = (ci > 1e-7) & (co > 1e-7) & (np.linalg.norm(wi + wo, axis=-1) > 1e-6)
        co_ = np.maximum(co, 0)
        wir, wor = self._refr(wi), self._refr(wo); ie2 = 1 / self.eta ** 2
        fo = np.zeros(wi.shape[:-1] + (3,)); fi = np.zeros_like(fo)
        for k in range(7):
            refr = k in INNER
            g = self._lobe(k, wir if refr else wi, wor if refr else wo)
            term = self.lam[:, k + 1] * (g * co_ * (ie2 if refr else 1.0))[..., None]
            if refr: fi += term
            else: fo += term
        f = fo + self._s(ci) * self._s(co) * (fi + self.lam[:, 0] * (co_ / np.pi)[..., None])
        return np.where(valid[..., None], f, 0.0)
