"""WL80C compiled closure, runtime implementation in Dr.Jit (Mitsuba types).

Evaluate / Sample / PDF of the frozen R = 8 closure (see closure80c.py for the
definition). The compiled state per material is 53 floats
(lam 24, au 7, av 7, f0 7, sampling probs 8; probs are derived from lam / f0);
at construction the runtime expands it with derived constants
(1/au^2, 1/av^2, 1/(pi au av), au^2, av^2).

Sampling: lobe r with probability p_r (wi-independent); Lambert -> cosine warp;
GGX -> visible-normal sampling with spherical caps (Dupuy & Benyoub 2023),
wo = reflect(wi, h). Reported pdf = sum_r p_r pdf_r(wo), pdf_r(GGX) = G1(wi) D(h) / (4 cos_i)
(sub-normalised: VNDF samples that land below the horizon are failures with weight 0).

alt_sample() is an INDEPENDENT sampler used only by the contract tests:
classic NDF sampling p(h) = D(h) cos(theta_h) (Walter et al. 2007, anisotropic
inverse CDF), pdf_alt(wo) = sum_r p_r D_r(h) cos(theta_h) / (4 wo.h).
"""
import math
import numpy as np

INV_PI = 1.0 / math.pi
EPS = 1e-7
N_STATE = 53


def _mi():
    import mitsuba as mi
    return mi


def expand_state(st):
    """[M,53] compiled state -> [M,88] runtime table (float64 math, float32 store)."""
    st = np.asarray(st, np.float64)
    au, av = st[:, 24:31], st[:, 31:38]
    der = np.concatenate([1 / au ** 2, 1 / av ** 2, INV_PI / (au * av), au ** 2, av ** 2], 1)
    return np.concatenate([st, der], 1).astype(np.float32)


class DrClosure:
    """Closure bound either to one material (uniform: state [53]) or to per-lane
    materials (table [M,53] + idx UInt32 per lane)."""

    def __init__(self, state, idx=None):
        mi = _mi()
        import drjit as dr
        st = np.atleast_2d(np.asarray(state, np.float32))
        tab = expand_state(st)
        if idx is None:
            assert tab.shape[0] == 1
            vals = []
            for v in tab[0]:
                x = mi.Float(float(v))
                dr.make_opaque(x)
                vals.append(x)
        else:
            M = tab.shape[0]
            flat = mi.Float(np.ascontiguousarray(tab.T).ravel())
            vals = [dr.gather(mi.Float, flat, idx + k * M) for k in range(tab.shape[1])]
        self.lam = [[vals[c * 8 + r] for r in range(8)] for c in range(3)]
        self.au, self.av, self.f0 = vals[24:31], vals[31:38], vals[38:45]
        self.pr = vals[45:53]
        self.iau2, self.iav2, self.dn = vals[53:60], vals[60:67], vals[67:74]
        self.au2, self.av2 = vals[74:81], vals[81:88]

    # ---------------- evaluation ----------------
    def eval_pdf(self, wi, wo, active=True, want_pdf=True, want_eval=True):
        mi = _mi()
        import drjit as dr
        ci, co = wi.z, wo.z
        valid = mi.Bool(active) & (ci > EPS) & (co > EPS)
        hs = wi + wo
        hn = dr.norm(hs)
        valid &= hn > 1e-6
        h = hs / dr.maximum(hn, 1e-6)
        ci_ = dr.maximum(ci, EPS)
        inv4ci = dr.rcp(4.0 * ci_)
        c = dr.clamp(dr.dot(wi, h), 0.0, 1.0)
        m1 = 1.0 - c
        m2 = m1 * m1
        m5 = m2 * m2 * m1
        hx2, hy2, hz2 = h.x * h.x, h.y * h.y, h.z * h.z
        wix2, wiy2 = wi.x * wi.x, wi.y * wi.y
        wox2, woy2 = wo.x * wo.x, wo.y * wo.y
        iwiz2 = dr.rcp(dr.maximum(ci * ci, EPS))
        iwoz2 = dr.rcp(dr.maximum(co * co, EPS))
        lamb = dr.maximum(co, 0.0) * INV_PI
        f = mi.Color3f(self.lam[0][0], self.lam[1][0], self.lam[2][0]) * lamb if want_eval else None
        pdf = self.pr[0] * lamb if want_pdf else None
        for k in range(7):
            den = hx2 * self.iau2[k] + hy2 * self.iav2[k] + hz2
            D = self.dn[k] * dr.rcp(den * den)
            Li = 0.5 * (dr.sqrt(1.0 + (self.au2[k] * wix2 + self.av2[k] * wiy2) * iwiz2) - 1.0)
            DG = D * inv4ci
            if want_eval:
                Lo = 0.5 * (dr.sqrt(1.0 + (self.au2[k] * wox2 + self.av2[k] * woy2) * iwoz2) - 1.0)
                F = self.f0[k] + (1.0 - self.f0[k]) * m5
                s = F * DG * dr.rcp(1.0 + Li + Lo)
                f += mi.Color3f(self.lam[0][k + 1], self.lam[1][k + 1], self.lam[2][k + 1]) * s
            if want_pdf:
                pdf += self.pr[k + 1] * DG * dr.rcp(1.0 + Li)
        if want_eval:
            f = dr.select(valid, f, 0.0)
        if want_pdf:
            pdf = dr.select(valid, pdf, 0.0)
        return f, pdf

    def eval(self, wi, wo, active=True):
        return self.eval_pdf(wi, wo, active, want_pdf=False)[0]

    def pdf(self, wi, wo, active=True):
        return self.eval_pdf(wi, wo, active, want_eval=False)[1]

    # ---------------- sampling ----------------
    def _select_lobe(self, u1):
        mi = _mi()
        import drjit as dr
        sel = mi.UInt32(0)
        acc = mi.Float(self.pr[0])
        for k in range(1, 8):
            sel = dr.select(u1 >= acc, mi.UInt32(k), sel)
            acc = acc + self.pr[k]
        au = mi.Float(1.0); av = mi.Float(1.0)
        for k in range(7):
            m = sel == (k + 1)
            au = dr.select(m, self.au[k], au)
            av = dr.select(m, self.av[k], av)
        return sel, au, av

    def sample(self, wi, u1, u2, active=True):
        """Returns wo, pdf, weight (= eval / pdf), valid."""
        mi = _mi()
        import drjit as dr
        active = mi.Bool(active) & (wi.z > EPS)
        sel, au, av = self._select_lobe(u1)
        wo_l = mi.warp.square_to_cosine_hemisphere(u2)
        wis = dr.normalize(mi.Vector3f(au * wi.x, av * wi.y, wi.z))
        phi = 2.0 * math.pi * u2.x
        z = (1.0 - u2.y) * (1.0 + wis.z) - wis.z
        st = dr.sqrt(dr.clamp(1.0 - z * z, 0.0, 1.0))
        sp, cp = dr.sincos(phi)
        hs = mi.Vector3f(st * cp, st * sp, z) + wis
        h = dr.normalize(mi.Vector3f(au * hs.x, av * hs.y, hs.z))
        wo_g = 2.0 * dr.dot(wi, h) * h - wi
        wo = dr.select(sel == 0, wo_l, wo_g)
        f, pdf = self.eval_pdf(wi, wo, active)
        valid = active & (wo.z > EPS) & (pdf > 0)
        w = dr.select(valid, f / dr.select(valid, pdf, 1.0), 0.0)
        return wo, dr.select(valid, pdf, 0.0), w, valid

    def alt_sample(self, wi, u1, u2, active=True):
        """Independent NDF sampler (contract tests only). Returns wo, pdf_alt, valid."""
        mi = _mi()
        import drjit as dr
        active = mi.Bool(active) & (wi.z > EPS)
        sel, au, av = self._select_lobe(u1)
        wo_l = mi.warp.square_to_cosine_hemisphere(u2)
        sp, cp = dr.sincos(2.0 * math.pi * u2.y)
        phi = dr.atan2(av * sp, au * cp)
        sph, cph = dr.sincos(phi)
        inv_a2 = cph * cph / (au * au) + sph * sph / (av * av)
        u = dr.clamp(u2.x, 0.0, 1.0 - 1e-7)
        tan2 = u / ((1.0 - u) * inv_a2)
        ct = dr.rsqrt(1.0 + tan2)
        stt = dr.sqrt(dr.maximum(0.0, 1.0 - ct * ct))
        h = mi.Vector3f(stt * cph, stt * sph, ct)
        wo_g = 2.0 * dr.dot(wi, h) * h - wi
        wo = dr.select(sel == 0, wo_l, wo_g)
        pdf = self.alt_pdf(wi, wo, active)
        valid = active & (wo.z > EPS) & (pdf > 0)
        return wo, dr.select(valid, pdf, 0.0), valid

    def alt_pdf(self, wi, wo, active=True):
        mi = _mi()
        import drjit as dr
        valid = mi.Bool(active) & (wi.z > EPS) & (wo.z > EPS)
        hs = wi + wo
        hn = dr.norm(hs)
        valid &= hn > 1e-6
        h = hs / dr.maximum(hn, 1e-6)
        woh = dr.maximum(dr.dot(wo, h), 1e-7)
        pdf = self.pr[0] * dr.maximum(wo.z, 0.0) * INV_PI
        for k in range(7):
            den = h.x * h.x * self.iau2[k] + h.y * h.y * self.iav2[k] + h.z * h.z
            D = self.dn[k] * dr.rcp(den * den)
            pdf += self.pr[k + 1] * D * h.z / (4.0 * woh)
        return dr.select(valid, pdf, 0.0)

    # per-lobe pieces for the normalisation test
    def lobe_vndf_fail_and_norm(self, k, wi, u2, uh):
        """For GGX lobe k: (a) VNDF sample below-horizon indicator; (b) one-sample
        estimate of int_{H+} pdf_k(wo) dwo using NDF-proportional h (independent)."""
        mi = _mi()
        import drjit as dr
        au, av = self.au[k], self.av[k]
        wis = dr.normalize(mi.Vector3f(au * wi.x, av * wi.y, wi.z))
        phi = 2.0 * math.pi * u2.x
        z = (1.0 - u2.y) * (1.0 + wis.z) - wis.z
        st = dr.sqrt(dr.clamp(1.0 - z * z, 0.0, 1.0))
        sp, cp = dr.sincos(phi)
        hs = mi.Vector3f(st * cp, st * sp, z) + wis
        h = dr.normalize(mi.Vector3f(au * hs.x, av * hs.y, hs.z))
        wo = 2.0 * dr.dot(wi, h) * h - wi
        fail = wo.z <= EPS
        # independent: h ~ D(h) cos(theta_h); integrand pdf_k(wo) * 4 (wi.h) / (D cos theta_h)
        sp2, cp2 = dr.sincos(2.0 * math.pi * uh.y)
        ph = dr.atan2(av * sp2, au * cp2)
        sph, cph = dr.sincos(ph)
        inv_a2 = cph * cph / (au * au) + sph * sph / (av * av)
        u = dr.clamp(uh.x, 0.0, 1.0 - 1e-7)
        tan2 = u / ((1.0 - u) * inv_a2)
        ct = dr.rsqrt(1.0 + tan2)
        stt = dr.sqrt(dr.maximum(0.0, 1.0 - ct * ct))
        h2 = mi.Vector3f(stt * cph, stt * sph, ct)
        wih = dr.dot(wi, h2)
        wo2 = 2.0 * wih * h2 - wi
        Li = 0.5 * (dr.sqrt(1.0 + (au * au * wi.x * wi.x + av * av * wi.y * wi.y) / dr.maximum(wi.z * wi.z, EPS)) - 1.0)
        g = dr.rcp(1.0 + Li) * wih / (dr.maximum(wi.z, EPS) * ct)  # = pdf_k * 4 wi.h / (D cos)
        g = dr.select((wih > 0) & (wo2.z > EPS), g, 0.0)
        return fail, g
