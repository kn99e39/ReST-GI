"""WL83 Dr.Jit runtime for the WL82 closures C0 / E / ER (semantics = experiments/fnto_compiled82/closure82.py, torch reference).

Compiled state per material (69 floats, identical layout for every arm; unused fields are 1):
  lam 24 (c * 8 + r) | au 7 | av 7 | f0 7 | sampling probs 8 | s 15 (c * 5 + j, knots mu = 0, .15, .4, .7, 1) | eta_e 1
Runtime table = state + derived (1/au^2, 1/av^2, 1/(pi au av), au^2, av^2) = 104 floats.

eval(wi, wo) = f cos(theta_o):
  C0: lam_0 cos/pi + sum_k lam_k g_k(wi, wo) cos
  E : sum_{k in OUTER} lam_k g_k cos + s(mu_i) s(mu_o) [lam_0 cos/pi + sum_{k in INNER} lam_k g_k(wi, wo) cos]
  ER: as E, inner lobes g_k(t(wi), t(wo)) / eta^2, t(w) = (w_x / eta, w_y / eta, sqrt(1 - (1 - w_z^2) / eta^2))
  g_k = horizon-safe anisotropic GGX F D / (2 (cos_o sqrt(cos_i^2 + t_i) + cos_i sqrt(cos_o^2 + t_o)))  (WL81 closed form)
  s(mu) = s_0 + sum_j (s_{j+1} - s_j) clamp((mu - k_j) / (k_{j+1} - k_j), 0, 1)  (= piecewise-linear interpolation)
pdf = p_0 cos/pi + sum_k p_k pdf_k; GGX pdf_k = D / (2 (cos_i + sqrt(cos_i^2 + t_i))) (visible normals); refracted inner lobes (ER):
  pdf_vndf(t(wo) | t(wi)) cos_o / (eta^2 cos_o'). Validity as eval (cos_i, cos_o > 1e-7, |wi + wo| > 1e-6).
sample: lobe by wi-independent probabilities; Lambert cosine; GGX VNDF (spherical caps) about wi, or about t(wi) followed by the exit
  refraction for refracted inner lobes (total internal reflection -> invalid); weight = eval / pdf.
"""
import math
import numpy as np

INV_PI = 1.0 / math.pi
EPS = 1e-7
N_STATE = 69
KNOTS = [0.0, 0.15, 0.4, 0.7, 1.0]
OUTER = [0, 1, 2]
INNER = [3, 4, 5, 6]


def _mi():
    import mitsuba as mi
    return mi


def compiled_state(arm, p):
    """torch decoded params (closure82.decode) -> numpy [M, 69]."""
    import torch
    sys_c = __import__('closure82')
    pr = sys_c.sampling_probs(arm, p)
    M = p['lam'].shape[0]
    s = p['s'].reshape(M, 15) if 's' in p else torch.ones(M, 15, device=p['lam'].device)
    eta = p['eta'].reshape(M, 1) if 'eta' in p else torch.ones(M, 1, device=p['lam'].device)
    return torch.cat([p['lam'].reshape(M, 24), p['au'], p['av'], p['f0'], pr, s, eta], 1).detach().cpu().numpy().astype(np.float32)


def expand_state(st):
    st = np.asarray(st, np.float64)
    au, av = st[:, 24:31], st[:, 31:38]
    der = np.concatenate([1 / au ** 2, 1 / av ** 2, INV_PI / (au * av), au ** 2, av ** 2], 1)
    return np.concatenate([st, der], 1).astype(np.float32)


class DrClosure83:
    """arm in {'C0', 'E', 'ER'} (Python constant -> specialised kernels). Uniform (state [69]) or per-lane table (table [M,69] + idx)."""

    def __init__(self, arm, state, idx=None):
        mi = _mi()
        import drjit as dr
        assert arm in ('C0', 'E', 'ER')
        self.arm = arm
        tab = expand_state(np.atleast_2d(np.asarray(state, np.float32)))
        if idx is None:
            assert tab.shape[0] == 1
            vals = []
            for v in tab[0]:
                x = mi.Float(float(v)); dr.make_opaque(x); vals.append(x)
        else:
            M = tab.shape[0]
            flat = mi.Float(np.ascontiguousarray(tab.T).ravel())
            vals = [dr.gather(mi.Float, flat, idx + k * M) for k in range(tab.shape[1])]
        self.lam = [[vals[c * 8 + r] for r in range(8)] for c in range(3)]
        self.au, self.av, self.f0 = vals[24:31], vals[31:38], vals[38:45]
        self.pr = vals[45:53]
        self.s = [[vals[53 + c * 5 + j] for j in range(5)] for c in range(3)]
        self.eta = vals[68]
        self.iau2, self.iav2, self.dn = vals[69:76], vals[76:83], vals[83:90]
        self.au2, self.av2 = vals[90:97], vals[97:104]

    # ---------------------------------------------------------------- helpers
    def _s(self, mu):
        mi = _mi()
        import drjit as dr
        out = []
        for c in range(3):
            v = mi.Float(self.s[c][0])
            for j in range(4):
                t = dr.clamp((mu - KNOTS[j]) * (1.0 / (KNOTS[j + 1] - KNOTS[j])), 0.0, 1.0)
                v = v + (self.s[c][j + 1] - self.s[c][j]) * t
            out.append(v)
        return mi.Color3f(out[0], out[1], out[2])

    def _refr(self, w):
        mi = _mi()
        import drjit as dr
        ie = dr.rcp(self.eta)
        wz = dr.maximum(w.z, 0.0)
        # cancellation-free t_z = sqrt(max(eta^2 - 1, 0) + w_z^2) / eta (algebraically = sqrt(1 - (w_x^2 + w_y^2) / eta^2); see closure83.py)
        return mi.Vector3f(w.x * ie, w.y * ie, dr.sqrt(dr.maximum(self.eta * self.eta - 1.0, 0.0) + wz * wz) * ie)

    def _lobe(self, k, wi, wo, want_eval, want_pdf):
        """g_k(wi, wo) (BRDF value) and visible-normal pdf_k(wo | wi) for directions in the upper hemisphere."""
        mi = _mi()
        import drjit as dr
        hs = wi + wo
        h = hs / dr.maximum(dr.norm(hs), 1e-6)
        ci = dr.maximum(wi.z, 0.0); co = dr.maximum(wo.z, 0.0)
        den = h.x * h.x * self.iau2[k] + h.y * h.y * self.iav2[k] + h.z * h.z
        D = self.dn[k] * dr.rcp(den * den)
        si = dr.sqrt(ci * ci + self.au2[k] * wi.x * wi.x + self.av2[k] * wi.y * wi.y)
        g = p = None
        if want_eval:
            so = dr.sqrt(co * co + self.au2[k] * wo.x * wo.x + self.av2[k] * wo.y * wo.y)
            m1 = 1.0 - dr.clamp(dr.dot(wi, h), 0.0, 1.0)
            m2 = m1 * m1
            F = self.f0[k] + (1.0 - self.f0[k]) * (m2 * m2 * m1)
            g = F * D * dr.rcp(dr.maximum(2.0 * (co * si + ci * so), 1e-30))
        if want_pdf:
            p = D * dr.rcp(dr.maximum(2.0 * (ci + si), 1e-30))
        return g, p

    # ---------------------------------------------------------------- evaluation
    def eval_pdf(self, wi, wo, active=True, want_pdf=True, want_eval=True):
        mi = _mi()
        import drjit as dr
        ci, co = wi.z, wo.z
        valid = mi.Bool(active) & (ci > EPS) & (co > EPS) & (dr.norm(wi + wo) > 1e-6)
        co_ = dr.maximum(co, 0.0)
        lamb = co_ * INV_PI
        lamc = lambda r: mi.Color3f(self.lam[0][r], self.lam[1][r], self.lam[2][r])
        f_out = mi.Color3f(0.0); f_in = mi.Color3f(0.0)
        pdf = self.pr[0] * lamb if want_pdf else None
        if self.arm == 'ER':
            wi_r, wo_r = self._refr(wi), self._refr(wo)
            ie2 = dr.rcp(self.eta * self.eta)
            jac = co_ * ie2 * dr.rcp(dr.maximum(wo_r.z, 1e-30))
        for k in range(7):
            refr = self.arm == 'ER' and k in INNER
            g, p = self._lobe(k, wi_r if refr else wi, wo_r if refr else wo, want_eval, want_pdf)
            if want_eval:
                term = lamc(k + 1) * (g * co_ * (ie2 if refr else 1.0))
                if self.arm != 'C0' and k in INNER:
                    f_in += term
                else:
                    f_out += term
            if want_pdf:
                pdf += self.pr[k + 1] * (p * jac if refr else p)
        f = None
        if want_eval:
            if self.arm == 'C0':
                f = f_out + lamc(0) * lamb
            else:
                f = f_out + self._s(ci) * self._s(co) * (f_in + lamc(0) * lamb)
            f = dr.select(valid, f, 0.0)
        if want_pdf:
            pdf = dr.select(valid, pdf, 0.0)
        return f, pdf

    def eval(self, wi, wo, active=True):
        return self.eval_pdf(wi, wo, active, want_pdf=False)[0]

    def pdf(self, wi, wo, active=True):
        return self.eval_pdf(wi, wo, active, want_eval=False)[1]

    # ---------------------------------------------------------------- sampling
    def sample(self, wi, u1, u2, active=True):
        """Returns wo, pdf, weight (= eval / pdf), valid."""
        mi = _mi()
        import drjit as dr
        active = mi.Bool(active) & (wi.z > EPS)
        sel = mi.UInt32(0)
        acc = mi.Float(self.pr[0])
        for k in range(1, 8):
            sel = dr.select(u1 >= acc, mi.UInt32(k), sel)
            acc = acc + self.pr[k]
        au = mi.Float(1.0); av = mi.Float(1.0)
        for k in range(7):
            m = dr.eq(sel, k + 1)
            au = dr.select(m, self.au[k], au)
            av = dr.select(m, self.av[k], av)
        refr = mi.Bool(False)
        if self.arm == 'ER':
            refr = sel >= 4                                    # lam index 4..7 = inner GGX lobes 3..6
        wb = dr.select(refr, self._refr(wi), wi) if self.arm == 'ER' else wi
        wis = dr.normalize(mi.Vector3f(au * wb.x, av * wb.y, wb.z))
        phi = 2.0 * math.pi * u2.x
        z = (1.0 - u2.y) * (1.0 + wis.z) - wis.z
        st = dr.sqrt(dr.clamp(1.0 - z * z, 0.0, 1.0))
        sp, cp = dr.sincos(phi)
        hs = mi.Vector3f(st * cp, st * sp, z) + wis
        h = dr.normalize(mi.Vector3f(au * hs.x, av * hs.y, dr.maximum(hs.z, 0.0)))
        wr = 2.0 * dr.dot(wb, h) * h - wb
        ok_exit = mi.Bool(True)
        if self.arm == 'ER':
            tx, ty = wr.x * self.eta, wr.y * self.eta
            r2 = tx * tx + ty * ty
            wex = mi.Vector3f(tx, ty, dr.sqrt(dr.maximum(1.0 - r2, 0.0)))
            ok_exit = ~refr | ((r2 < 1.0) & (wr.z > 0.0))
            wr = dr.select(refr, wex, wr)
        wo = dr.select(dr.eq(sel, 0), mi.warp.square_to_cosine_hemisphere(u2), wr)
        f, pdf = self.eval_pdf(wi, wo, active)
        valid = active & ok_exit & (wo.z > EPS) & (pdf > 0)
        w = dr.select(valid, f / dr.select(valid, pdf, 1.0), 0.0)
        return wo, dr.select(valid, pdf, 0.0), w, valid
