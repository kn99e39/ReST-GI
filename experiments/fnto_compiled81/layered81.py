"""WL81 qualified reference: a layered COATED-CONDUCTOR BSDF evaluated by stochastic random walks (Dr.Jit, cuda).

Physics (per material): a rough dielectric coating (GGX alpha_i, IOR eta, air above) over a homogeneous isotropic-scattering slab of optical
thickness tau (sigma_t = 1, single-scattering albedo a_rgb) over a rough conductor (GGX alpha_b, complex IOR from a reflectance r_rgb,
relative to the coating medium). Index-matched nothing: refraction, Fresnel, total internal reflection and multiple scattering are all real.

Interfaces: ports of Mitsuba 3.5.1 `roughdielectric` / `roughconductor` (GGX, visible-normal sampling, separable Smith G, exact dielectric /
conductor Fresnel, radiance-mode 1/eta^2 transmission scaling) with PER-LANE parameters (Mitsuba's plugin IOR is a scalar). Equality with the
Mitsuba plugins is a reference-audit test (refaudit81.py).
Layered estimator: a faithful port of pbrt-v4 `LayeredBxDF::f` and `LayeredBxDF::Sample_f` (Guo, Hasan & Zhao 2018 position-free random
walk with next-event estimation and power-heuristic MIS; Russian roulette after depth 3 when max(beta) < 0.25) for the CoatedConductor
configuration (top = dielectric, bottom = conductor, two-sided, g = 0), with pbrt's conductor-reflectance parameterisation
(eta_c = 1, k_c = 2 sqrt(r) / sqrt(1 - r), both divided by the coating IOR).

Conventions (WL80C): local frame z = normal; `wi` = camera-side direction (Mitsuba si.wi, pbrt wo), `wo` = light-side direction (Mitsuba wo,
pbrt wi); eval(wi, wo) = f(wi, wo) cos(theta_wo) (flux measure, same as Mitsuba BSDF::eval). One random walk gives an unbiased estimate;
targets average N independent walks. The layered BSDF is reciprocal and energy conserving physically; the estimator is unbiased up to the
max-depth truncation (energy loss only).
"""
import math
import mitsuba as mi
import drjit as dr

INV4PI = 1.0 / (4.0 * math.pi)
Float = None  # set by init()


def init(variant='cuda_ad_rgb'):
    global Float
    if mi.variant() != variant:
        mi.set_variant(variant)
    Float = mi.Float
    return mi


class Mat:
    """Per-lane material parameters (Dr.Jit arrays of equal width)."""
    def __init__(self, ai, eta, tau, alb, r, ab):
        self.ai, self.eta, self.tau, self.alb, self.ab = mi.Float(ai), mi.Float(eta), mi.Float(tau), mi.Color3f(alb), mi.Float(ab)
        r = dr.clamp(mi.Color3f(r), 0.0, 0.9999)
        self.ce = mi.Color3f(1.0) / self.eta                                   # pbrt: eta_c = 1, k_c = 2 sqrt(r)/sqrt(1-r), / coating eta
        self.ck = (2.0 * dr.sqrt(r) / dr.sqrt(1.0 - r)) / self.eta


def _ggx(alpha):
    return mi.MicrofacetDistribution(mi.MicrofacetType.GGX, alpha, alpha, True)


def mulsign(v, s):
    return dr.mulsign(v, s)


# ------------------------------------------------------------------ rough dielectric (Mitsuba roughdielectric, per-lane eta / alpha)
def diel_eval(m, wi, wo, radiance, refl_on, trans_on, active=True):
    """Mitsuba roughdielectric::eval -> f |cos wo| (grey)."""
    ci, co = wi.z, wo.z
    active = mi.Bool(active) & dr.neq(ci, 0.0)
    reflect = ci * co > 0.0
    eta = dr.select(ci > 0.0, m.eta, dr.rcp(m.eta))
    inv_eta = dr.select(ci > 0.0, dr.rcp(m.eta), m.eta)
    h = dr.normalize(wi + wo * dr.select(reflect, 1.0, eta))
    h = mulsign(h, h.z)
    active &= (dr.dot(wi, h) * ci > 0.0) & (dr.dot(wo, h) * co > 0.0)      # Mitsuba: consistent microfacet orientation
    d = _ggx(m.ai)
    D = d.eval(h)
    F = mi.fresnel(dr.dot(wi, h), m.eta)[0]
    G = d.G(wi, wo, h)
    res = mi.Float(0.0)
    if refl_on:
        r = F * D * G / (4.0 * dr.abs(ci))
        res = dr.select(reflect & active, r, res)
    if trans_on:
        scale = dr.sqr(inv_eta) if radiance else mi.Float(1.0)
        t = dr.abs((scale * (1.0 - F) * D * G * eta * eta * dr.dot(wi, h) * dr.dot(wo, h)) /
                   (ci * dr.sqr(dr.dot(wi, h) + eta * dr.dot(wo, h))))
        res = dr.select((~reflect) & active, t, res)
    return dr.select(active, res, 0.0)


def diel_pdf(m, wi, wo, refl_on, trans_on, active=True):
    """Mitsuba roughdielectric::pdf (component-restricted pdf is conditional, as in Mitsuba)."""
    ci, co = wi.z, wo.z
    active = mi.Bool(active) & dr.neq(ci, 0.0)
    reflect = ci * co > 0.0
    if refl_on and not trans_on:
        active &= reflect
    elif trans_on and not refl_on:
        active &= ~reflect
    eta = dr.select(ci > 0.0, m.eta, dr.rcp(m.eta))
    h = dr.normalize(wi + wo * dr.select(reflect, 1.0, eta))
    h = mulsign(h, h.z)
    active &= (dr.dot(wi, h) * ci > 0.0) & (dr.dot(wo, h) * co > 0.0)
    dwh = dr.select(reflect, dr.rcp(4.0 * dr.dot(wo, h)), (eta * eta * dr.dot(wo, h)) / dr.sqr(dr.dot(wi, h) + eta * dr.dot(wo, h)))
    prob = _ggx(m.ai).pdf(mulsign(wi, ci), h)
    if refl_on and trans_on:
        F = mi.fresnel(dr.dot(wi, h), m.eta)[0]
        prob *= dr.select(reflect, F, 1.0 - F)
    return dr.select(active, prob * dr.abs(dwh), 0.0)


def diel_sample(m, wi, radiance, refl_on, trans_on, u1, u2, active=True):
    """Mitsuba roughdielectric::sample -> (wo, weight = f|cos|/pdf (grey), pdf, is_reflection, valid)."""
    ci = wi.z
    active = mi.Bool(active) & dr.neq(ci, 0.0)
    d = _ggx(m.ai)
    h, pdf = d.sample(mulsign(wi, ci), u2)
    active &= dr.neq(pdf, 0.0)
    F, cos_t, eta_it, eta_ti = mi.fresnel(dr.dot(wi, h), m.eta)
    if refl_on and trans_on:
        sel_r = (u1 <= F) & active
        weight = mi.Float(1.0)
        pdf *= dr.select(sel_r, F, 1.0 - F)
    elif refl_on:
        sel_r = active
        weight = F
    else:
        sel_r = mi.Bool(False)
        weight = 1.0 - F
    sel_t = (~sel_r) & active
    wo_r = mi.reflect(wi, h)
    wo_t = mi.refract(wi, h, cos_t, eta_ti)
    wo = dr.select(sel_r, wo_r, wo_t)
    dwh = dr.select(sel_r, dr.rcp(4.0 * dr.dot(wo, h)),
                    (dr.sqr(eta_it) * dr.dot(wo, h)) / dr.sqr(dr.dot(wi, h) + eta_it * dr.dot(wo, h)))
    if radiance:
        weight = dr.select(sel_t, weight * dr.sqr(eta_ti), weight)
    weight = weight * d.smith_g1(wo, h)
    pdf = pdf * dr.abs(dwh)
    valid = active & dr.neq(pdf, 0.0) & (sel_r | sel_t)
    return wo, dr.select(valid, weight, 0.0), dr.select(valid, pdf, 0.0), sel_r, valid


# ------------------------------------------------------------------ rough conductor (Mitsuba roughconductor, per-lane complex IOR)
def fresnel_conductor(c, eta, k):
    """Mitsuba fresnel_conductor (unpolarised), per channel (Color3f eta, k)."""
    c2 = c * c
    s2 = 1.0 - c2
    s4 = s2 * s2
    t1 = eta * eta - k * k - s2
    a2pb2 = dr.safe_sqrt(t1 * t1 + 4.0 * k * k * eta * eta)
    a = dr.safe_sqrt(0.5 * (a2pb2 + t1))
    term1 = a2pb2 + c2
    term2 = 2.0 * c * a
    rs = (term1 - term2) / (term1 + term2)
    term3 = a2pb2 * c2 + s4
    term4 = term2 * s2
    rp = rs * (term3 - term4) / (term3 + term4)
    return 0.5 * (rs + rp)


def cond_eval(m, wi, wo, active=True):
    ci, co = wi.z, wo.z
    active = mi.Bool(active) & (ci > 0.0) & (co > 0.0)
    h = dr.normalize(wo + wi)
    d = _ggx(m.ab)
    D = d.eval(h)
    active &= dr.neq(D, 0.0)
    G = d.G(wi, wo, h)
    F = fresnel_conductor(mi.Color3f(dr.dot(wi, h)), m.ce, m.ck)
    res = F * (D * G / (4.0 * ci))
    return dr.select(active, res, 0.0)


def cond_pdf(m, wi, wo, active=True):
    ci, co = wi.z, wo.z
    active = mi.Bool(active) & (ci > 0.0) & (co > 0.0)
    h = dr.normalize(wo + wi)
    p = _ggx(m.ab).pdf(wi, h) / (4.0 * dr.dot(wo, h))
    return dr.select(active, p, 0.0)


def cond_sample(m, wi, u2, active=True):
    ci = wi.z
    active = mi.Bool(active) & (ci > 0.0)
    d = _ggx(m.ab)
    h, pdf = d.sample(wi, u2)
    wo = mi.reflect(wi, h)
    active &= dr.neq(pdf, 0.0) & (wo.z > 0.0)
    weight = d.smith_g1(wo, h)
    pdf = pdf / (4.0 * dr.dot(wo, h))
    F = fresnel_conductor(mi.Color3f(dr.dot(wi, h)), m.ce, m.ck)
    return wo, dr.select(active, F * weight, 0.0), dr.select(active, pdf, 0.0), active


# ------------------------------------------------------------------ pbrt-v4 LayeredBxDF::f (CoatedConductor, g = 0)
def _ph(a, b):
    a2, b2 = a * a, b * b
    return dr.select(a2 + b2 > 0.0, a2 / (a2 + b2), 0.0)


def _tr(dz, w):
    return dr.select(dr.abs(dz) <= 1e-30, 1.0, dr.exp(-dr.abs(dz / w.z)))


def _uniform_sphere(u):
    z = 1.0 - 2.0 * u.y
    r = dr.safe_sqrt(1.0 - z * z)
    s, c = dr.sincos(2.0 * math.pi * u.x)
    return mi.Vector3f(r * c, r * s, z)


def layered_eval(m, wi_cam, wo_light, rng, max_depth=64):
    """One random-walk estimate of eval = f(wi_cam, wo_light) cos(wo_light) per lane (pbrt f(wo = wi_cam, wi = wo_light)).
    Both directions in the upper hemisphere (reflection data); other lanes return 0."""
    wo, wi = wi_cam, wo_light                       # pbrt names from here on
    ok = (wo.z > 0.0) & (wi.z > 0.0)
    tau = m.tau
    exitZ = tau                                     # reflection, entered top -> exit through the top
    f = mi.Color3f(diel_eval(m, wo, wi, True, True, False, ok) / dr.maximum(dr.abs(wi.z), 1e-30))   # entrance reflection (pbrt f)
    wos_w, wos_wt, wos_pdf, _, okA = diel_sample(m, wo, True, False, True, rng.next_float32(), mi.Point2f(rng.next_float32(), rng.next_float32()), ok)
    wis_w, wis_wt, wis_pdf, _, okB = diel_sample(m, wi, False, False, True, rng.next_float32(), mi.Point2f(rng.next_float32(), rng.next_float32()), ok)
    okA &= dr.neq(wos_w.z, 0.0)
    okB &= dr.neq(wis_w.z, 0.0)
    wis_fop = wis_wt / dr.maximum(dr.abs(wis_w.z), 1e-30)          # pbrt wis->f / wis->pdf
    beta = mi.Color3f(wos_wt)                                    # f |cos| / pdf
    z = mi.Float(tau)
    w = mi.Vector3f(wos_w)
    active = ok & okA & okB
    depth = mi.UInt32(0)
    loop = mi.Loop(name='layered_eval', state=lambda: (rng, f, beta, z, w, active, depth))
    while loop(active & (depth < max_depth)):
        # Russian roulette
        mb = dr.max(beta)
        rr = (depth > 3) & (mb < 0.25)
        q = dr.maximum(0.0, 1.0 - mb)
        ur = rng.next_float32()
        kill = rr & (ur < q)
        active &= ~kill
        beta = dr.select(rr & active, beta / (1.0 - q), beta)
        # medium: sample distance (sigma_t = 1)
        dz = -dr.log(1.0 - rng.next_float32()) * dr.abs(w.z)
        zp = dr.select(w.z > 0.0, z + dz, z - dz)
        degenerate = dr.eq(zp, z)
        inside = active & (zp > 0.0) & (zp < tau) & ~degenerate
        # (a) scattering event inside the slab
        wt = _ph(wis_pdf, mi.Float(INV4PI))
        f += dr.select(inside, beta * m.alb * (INV4PI * wt * _tr(zp - exitZ, wis_w)) * wis_fop, 0.0)
        wnew = _uniform_sphere(mi.Point2f(rng.next_float32(), rng.next_float32()))
        bnew = beta * m.alb
        toward_exit = inside & (wnew.z > 0.0)           # z < exitZ always inside
        fexit = mi.Color3f(diel_eval(m, -wnew, wi, True, True, True, toward_exit)) / dr.maximum(dr.abs(wi.z), 1e-30)
        # MIS partner density = the density with which the NEE strategy (wis, sampled from wi) generates this inside direction.
        # pbrt-v4 uses exitInterface.PDF(-w, wi) here (density of the reverse query); the two weights then do not sum to one
        # (WL81 Stage A audit: eval-vs-sample albedo and reciprocity biased by ~5 %).
        epdf = diel_pdf(m, wi, -wnew, False, True, toward_exit)
        f += dr.select(toward_exit, bnew * _tr(zp - exitZ, wnew) * fexit * _ph(mi.Float(INV4PI), epdf), 0.0)
        # (b) interface event
        iface = active & ~inside & ~degenerate
        zc = dr.clamp(zp, 0.0, tau)
        at_exit = iface & (zc >= tau)
        at_bot = iface & (zc <= 0.0)
        # (b1) reflection at the exit (top) interface, from below
        r_w, r_wt, r_pdf, _, r_ok = diel_sample(m, -w, True, True, False, rng.next_float32(), mi.Point2f(rng.next_float32(), rng.next_float32()), at_exit)
        r_ok &= dr.neq(r_w.z, 0.0)
        # (b2) bottom conductor: NEE along wis, then sample reflection, then NEE through the exit
        wtb = _ph(wis_pdf, cond_pdf(m, -w, -wis_w, at_bot))
        f += dr.select(at_bot, beta * cond_eval(m, -w, -wis_w, at_bot) * (wtb * _tr(tau, wis_w)) * wis_fop, 0.0)
        c_w, c_wt, c_pdf, c_ok = cond_sample(m, -w, mi.Point2f(rng.next_float32(), rng.next_float32()), at_bot)
        c_ok &= dr.neq(c_w.z, 0.0)
        bc = beta * c_wt
        fexit2 = mi.Color3f(diel_eval(m, -c_w, wi, True, True, True, at_bot & c_ok)) / dr.maximum(dr.abs(wi.z), 1e-30)
        epdf2 = diel_pdf(m, wi, -c_w, False, True, at_bot & c_ok)   # NEE density (see above)
        f += dr.select(at_bot & c_ok, bc * _tr(tau, c_w) * fexit2 * _ph(c_pdf, epdf2), 0.0)
        # state update
        beta = mi.Color3f(dr.select(inside, bnew, dr.select(at_exit, beta * r_wt, dr.select(at_bot, bc, beta))))
        w = mi.Vector3f(dr.select(inside, wnew, dr.select(at_exit, r_w, dr.select(at_bot, c_w, w))))
        z = dr.select(inside, zp, dr.select(iface, zc, z))
        active &= ~(at_exit & ~r_ok) & ~(at_bot & ~c_ok)
        f = mi.Color3f(f)
        depth += 1
    return f * dr.abs(wi.z)                              # eval = f cos(theta_light)


def layered_sample(m, wi_cam, rng, max_depth=64):
    """pbrt LayeredBxDF::Sample_f (radiance mode): returns (wo_light, weight = f cos / pdf (Color3f), valid). Unbiased albedo = E[weight]."""
    wo = wi_cam
    ok = wo.z > 0.0
    tau = m.tau
    b_w, b_wt, b_pdf, b_refl, b_ok = diel_sample(m, wo, True, True, True, rng.next_float32(), mi.Point2f(rng.next_float32(), rng.next_float32()), ok)
    b_ok &= dr.neq(b_w.z, 0.0)
    done = b_ok & b_refl                                 # entrance reflection returns directly
    out_w = mi.Vector3f(b_w)
    out_wt = mi.Color3f(dr.select(done, mi.Color3f(b_wt), 0.0))
    ratio = mi.Color3f(b_wt)                             # running f / pdf with cosines (pbrt keeps f and pdf separately)
    w = mi.Vector3f(b_w)
    z = mi.Float(tau)
    active = b_ok & ~b_refl
    depth = mi.UInt32(0)
    loop = mi.Loop(name='layered_sample', state=lambda: (rng, out_w, out_wt, ratio, w, z, active, depth))
    while loop(active & (depth < max_depth)):
        rrb = dr.max(ratio)
        rr = (depth > 3) & (rrb < 0.25)
        q = dr.maximum(0.0, 1.0 - rrb)
        kill = rr & (rng.next_float32() < q)
        active &= ~kill
        ratio = dr.select(rr & active, ratio / (1.0 - q), ratio)
        active &= dr.neq(w.z, 0.0)
        dz = -dr.log(1.0 - rng.next_float32()) * dr.abs(w.z)
        zp = dr.select(w.z > 0.0, z + dz, z - dz)
        active &= dr.neq(zp, z)
        inside = active & (zp > 0.0) & (zp < tau)
        wnew = _uniform_sphere(mi.Point2f(rng.next_float32(), rng.next_float32()))
        iface = active & ~inside
        zc = dr.clamp(zp, 0.0, tau)
        top = iface & (zc >= tau)
        bot = iface & (zc <= 0.0)
        t_w, t_wt, t_pdf, t_refl, t_ok = diel_sample(m, -w, True, True, True, rng.next_float32(), mi.Point2f(rng.next_float32(), rng.next_float32()), top)
        c_w, c_wt, c_pdf, c_ok = cond_sample(m, -w, mi.Point2f(rng.next_float32(), rng.next_float32()), bot)
        exit_now = top & t_ok & ~t_refl & dr.neq(t_w.z, 0.0)
        out_w = mi.Vector3f(dr.select(exit_now, t_w, out_w))
        out_wt = mi.Color3f(dr.select(exit_now, ratio * t_wt, out_wt))
        ratio = mi.Color3f(dr.select(inside, ratio * m.alb, dr.select(top, ratio * t_wt, dr.select(bot, ratio * c_wt, ratio))))
        w = mi.Vector3f(dr.select(inside, wnew, dr.select(top, t_w, dr.select(bot, c_w, w))))
        z = dr.select(inside, zp, dr.select(iface, zc, z))
        active &= ~exit_now & ~(top & ~t_ok) & ~(bot & ~c_ok)
        depth += 1
    valid = dr.any(dr.neq(out_wt, 0.0)) & (out_w.z > 0.0)
    return out_w, dr.select(valid, out_wt, 0.0), valid
