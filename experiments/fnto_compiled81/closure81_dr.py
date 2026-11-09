"""WL81 Dr.Jit runtime closure = WL80D corrected class (closure80d_dr.DrClosure: element-wise lobe / direction selection) with the
horizon-safe GGX closed forms (software fix 2, see closure81.py). Only eval_pdf and the per-lobe normalisation helper are overridden;
sampling, alt_sample, alt_pdf and the state layout are inherited unchanged. `DrClosure` is exported under the WL80C name so the
WL80C contract / bench modules can use this runtime by module substitution (contract80c.CD = closure81_dr).
"""
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'fnto_closure80d'))
sys.path.insert(0, os.path.join(HERE, '..', 'fnto_compiled80c'))
import closure80d_dr as D80  # noqa: E402

INV_PI = D80.INV_PI
EPS = D80.EPS
N_STATE = D80.N_STATE
expand_state = D80.expand_state


class DrClosure81(D80.DrClosure):
    def eval_pdf(self, wi, wo, active=True, want_pdf=True, want_eval=True):
        mi = D80._mi()
        import drjit as dr
        ci, co = wi.z, wo.z
        valid = mi.Bool(active) & (ci > EPS) & (co > EPS)
        hs = wi + wo
        hn = dr.norm(hs)
        valid &= hn > 1e-6
        h = hs / dr.maximum(hn, 1e-6)
        ci_ = dr.maximum(ci, 0.0)
        co_ = dr.maximum(co, 0.0)
        c = dr.clamp(dr.dot(wi, h), 0.0, 1.0)
        m1 = 1.0 - c
        m2 = m1 * m1
        m5 = m2 * m2 * m1
        hx2, hy2, hz2 = h.x * h.x, h.y * h.y, h.z * h.z
        wix2, wiy2 = wi.x * wi.x, wi.y * wi.y
        wox2, woy2 = wo.x * wo.x, wo.y * wo.y
        ci2, co2 = ci_ * ci_, co_ * co_
        lamb = co_ * INV_PI
        f = mi.Color3f(self.lam[0][0], self.lam[1][0], self.lam[2][0]) * lamb if want_eval else None
        pdf = self.pr[0] * lamb if want_pdf else None
        for k in range(7):
            den = hx2 * self.iau2[k] + hy2 * self.iav2[k] + hz2
            D = self.dn[k] * dr.rcp(den * den)
            si = dr.sqrt(ci2 + self.au2[k] * wix2 + self.av2[k] * wiy2)
            if want_eval:
                so = dr.sqrt(co2 + self.au2[k] * wox2 + self.av2[k] * woy2)
                F = self.f0[k] + (1.0 - self.f0[k]) * m5
                s = F * D * co_ * dr.rcp(2.0 * (co_ * si + ci_ * so))
                f += mi.Color3f(self.lam[0][k + 1], self.lam[1][k + 1], self.lam[2][k + 1]) * s
            if want_pdf:
                pdf += self.pr[k + 1] * D * dr.rcp(2.0 * (ci_ + si))
        if want_eval:
            f = dr.select(valid, f, 0.0)
        if want_pdf:
            pdf = dr.select(valid, pdf, 0.0)
        return f, pdf

    def lobe_vndf_fail_and_norm(self, k, wi, u2, uh):
        """As WL80C (VNDF failure indicator; one-sample NDF estimate of int pdf_k) with the horizon-safe pdf_k."""
        mi = D80._mi()
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
        ci_ = dr.maximum(wi.z, 0.0)
        si = dr.sqrt(ci_ * ci_ + au * au * wi.x * wi.x + av * av * wi.y * wi.y)
        g = 2.0 * wih * dr.rcp((ci_ + si) * ct)  # = pdf_k * 4 wi.h / (D cos theta_h), horizon-safe
        g = dr.select((wih > 0) & (wo2.z > EPS) & (wi.z > EPS), g, 0.0)
        return fail, g


DrClosure = DrClosure81
