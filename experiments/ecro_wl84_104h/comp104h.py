"""WL104H exact component graph of the frozen certified ER closure (read-only instrumentation, no shared code is modified).

Native response (closure82.eval_closure('ER') with the closure83 robust refraction; closure83_dr.DrClosure83('ER')), cosine-weighted:

    F = O0 + O1 + O2 + S * (L + I0 + I1 + I2 + I3),   S = s(mu_i) s(mu_o)

    O_k = lam[c, 1+k] g_k(wi, wo) cos_o                         k = 0..2   (GGX lobes 0..2, outer group)
    L   = lam[c, 0] cos_o / pi                                              (Lambert)
    I_k = lam[c, 4+k] g_{3+k}(t(wi), t(wo)) cos_o / eta^2       k = 0..3   (GGX lobes 3..6, refracted inner group)
    t(w) = (w_x / eta, w_y / eta, sqrt(max(eta^2 - 1, 0) + w_z^2) / eta)
    s_c(mu) = piecewise-linear interpolation of 5 knots at mu = 0, .15, .4, .7, 1

Extra terms used only by explicitly declared counterfactuals:
    U_k = lam[c, 4+k] g_{3+k}(wi, wo) cos_o                     (inner lobe without the refracted-coordinate mapping, i.e. eta = 1)

All terms are zero outside the native validity set (cos_i, cos_o > 1e-7, |wi + wo| > 1e-6).
"""
import math
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "ecro_wl84_104b"))
import h104b as H  # noqa: E402  (imports closure82 + the closure83 robust-refraction patch, closure83_dr, Mitsuba)

CL, CD, mi, dr = H.CL, H.CD, H.mi, H.dr
INV_PI = 1.0 / math.pi
OUTER, INNER = (0, 1, 2), (3, 4, 5, 6)
BASE_TERMS = ("O0", "O1", "O2", "L", "I0", "I1", "I2", "I3")

# ------------------------------------------------------------------ intervention table (frozen in the protocol)
# Every intervention is a function of the exact term dict t (keys O0..O2, L, I0..I3, U0..U3, si, so, S) -> F_X.
# validity: VALID_FAMILY = a member (possibly boundary, lam = 0 or s = 1 or eta = 1) of the certified ER family, reciprocal and
#           passive by the WL82 certificate; DIAGNOSTIC_ONLY = breaks reciprocity (not a BSDF of the family).
# family: interventions of the same family are hierarchically nested / act on the same factor; the confounding check is applied
#         between different families only.


def _osum(t):
    return t["O0"] + t["O1"] + t["O2"]


def _isum(t):
    return t["I0"] + t["I1"] + t["I2"] + t["I3"]


def _usum(t):
    return t["U0"] + t["U1"] + t["U2"] + t["U3"]


def native(t):
    return _osum(t) + t["S"] * (_isum(t) + t["L"])


INTERVENTIONS = {
    "LAMBERT_OFF":    dict(level="group", family="lambert", validity="VALID_FAMILY",
                           f=lambda t: _osum(t) + t["S"] * _isum(t), note="lam_0 = 0"),
    "OUTER_OFF":      dict(level="group", family="outer", validity="VALID_FAMILY",
                           f=lambda t: t["S"] * (_isum(t) + t["L"]), note="lam_1..3 = 0"),
    "INNER_OFF":      dict(level="group", family="inner", validity="VALID_FAMILY",
                           f=lambda t: _osum(t) + t["S"] * t["L"], note="lam_4..7 = 0"),
    "ATTEN_OFF":      dict(level="group", family="attenuation", validity="VALID_FAMILY",
                           f=lambda t: _osum(t) + (_isum(t) + t["L"]), note="all 15 knots = 1 (S = 1)"),
    "ATTEN_IN_OFF":   dict(level="factor", family="attenuation", validity="DIAGNOSTIC_ONLY",
                           f=lambda t: _osum(t) + t["so"] * (_isum(t) + t["L"]), note="s(mu_i) -> 1 only; non-reciprocal"),
    "ATTEN_OUT_OFF":  dict(level="factor", family="attenuation", validity="DIAGNOSTIC_ONLY",
                           f=lambda t: _osum(t) + t["si"] * (_isum(t) + t["L"]), note="s(mu_o) -> 1 only; non-reciprocal"),
    "ETA_OFF":        dict(level="group", family="eta", validity="VALID_FAMILY",
                           f=lambda t: _osum(t) + t["S"] * (_usum(t) + t["L"]), note="eta = 1: no refracted mapping, no 1/eta^2"),
    "ETA_ATTEN_OFF":  dict(level="joint", family="eta", validity="VALID_FAMILY",
                           f=lambda t: _osum(t) + (_usum(t) + t["L"]), note="eta = 1 and S = 1 (interaction diagnostic)"),
}
for _k in range(3):
    INTERVENTIONS[f"O{_k}_OFF"] = dict(level="lobe", family="outer", validity="VALID_FAMILY",
                                      f=(lambda k: lambda t: native(t) - t[f"O{k}"])(_k), note=f"lam_{1 + _k} = 0")
for _k in range(4):
    INTERVENTIONS[f"I{_k}_OFF"] = dict(level="lobe", family="inner", validity="VALID_FAMILY",
                                      f=(lambda k: lambda t: native(t) - t["S"] * t[f"I{k}"])(_k), note=f"lam_{4 + _k} = 0")
NAMES = tuple(INTERVENTIONS)


def state_equivalent(name, state):
    """Native 69-float state that realises a VALID_FAMILY intervention exactly (independent check of the term algebra), else None."""
    s = np.array(state, np.float32, copy=True)
    lam = s[:24].reshape(3, 8)
    if name == "LAMBERT_OFF":
        lam[:, 0] = 0
    elif name == "OUTER_OFF":
        lam[:, 1:4] = 0
    elif name == "INNER_OFF":
        lam[:, 4:8] = 0
    elif name.startswith("O") and name.endswith("_OFF") and name[1].isdigit():
        lam[:, 1 + int(name[1])] = 0
    elif name.startswith("I") and name.endswith("_OFF") and name[1].isdigit():
        lam[:, 4 + int(name[1])] = 0
    elif name == "ATTEN_OFF":
        s[53:68] = 1
    elif name == "ETA_OFF":
        s[68] = 1
    elif name == "ETA_ATTEN_OFF":
        s[53:68] = 1
        s[68] = 1
    else:
        return None
    s[:24] = lam.ravel()
    return s


# ------------------------------------------------------------------ Torch (float64) exact decomposition
def terms_torch(p, wi, wo):
    """p: decoded single-material params (closure82.decode), any float dtype, batch 1; wi, wo torch [N, 3]. Returns dict of [N, 3]."""
    n = wi.shape[0]
    q = {k: v.expand(n, *v.shape[1:]) for k, v in p.items()}
    ci, co = wi[:, 2], wo[:, 2]
    co_ = co.clamp_min(0)[:, None]
    lam = q["lam"]
    fo, valid = CL.lobe_f(q["au"][:, list(OUTER)], q["av"][:, list(OUTER)], q["f0"][:, list(OUTER)], wi, wo)
    eta = q["eta"]
    fi, _ = CL.lobe_f(q["au"][:, list(INNER)], q["av"][:, list(INNER)], q["f0"][:, list(INNER)],
                      CL.refract_up(wi, eta), CL.refract_up(wo, eta))
    fi = fi / (eta * eta)[:, None]
    fu, _ = CL.lobe_f(q["au"][:, list(INNER)], q["av"][:, list(INNER)], q["f0"][:, list(INNER)], wi, wo)
    z = torch.zeros_like(lam[:, :, 0])
    m = valid[:, None]
    t = {}
    for k in range(3):
        t[f"O{k}"] = torch.where(m, lam[:, :, 1 + k] * (fo[:, k:k + 1] * co_), z)
    for k in range(4):
        t[f"I{k}"] = torch.where(m, lam[:, :, 4 + k] * (fi[:, k:k + 1] * co_), z)
        t[f"U{k}"] = torch.where(m, lam[:, :, 4 + k] * (fu[:, k:k + 1] * co_), z)
    t["L"] = torch.where(m, lam[:, :, 0] * (co_ / math.pi), z)
    t["si"] = CL.s_interp(q["s"], ci)
    t["so"] = CL.s_interp(q["s"], co)
    t["S"] = t["si"] * t["so"]
    t["valid"] = valid
    return t


def native_torch(p, wi, wo):
    n = wi.shape[0]
    q = {k: v.expand(n, *v.shape[1:]) for k, v in p.items()}
    return CL.eval_closure("ER", q, wi, wo)


# ------------------------------------------------------------------ Dr.Jit exact decomposition (runtime evaluator semantics)
class DrTerms(CD.DrClosure83):
    """closure83_dr.DrClosure83('ER') with every response term exported separately (same operations as eval_pdf)."""

    def __init__(self, state, idx=None):
        super().__init__("ER", state, idx)

    def terms(self, wi, wo, active=True, unrefracted=True):
        ci, co = wi.z, wo.z
        valid = mi.Bool(active) & (ci > CD.EPS) & (co > CD.EPS) & (dr.norm(wi + wo) > 1e-6)
        co_ = dr.maximum(co, 0.0)
        lamb = co_ * INV_PI
        lamc = lambda r: mi.Color3f(self.lam[0][r], self.lam[1][r], self.lam[2][r])
        wi_r, wo_r = self._refr(wi), self._refr(wo)
        ie2 = dr.rcp(self.eta * self.eta)
        t = {}
        for k in range(7):
            refr = k in INNER
            g, _ = self._lobe(k, wi_r if refr else wi, wo_r if refr else wo, True, False)
            term = lamc(k + 1) * (g * co_ * (ie2 if refr else 1.0))
            t[f"O{k}" if k in OUTER else f"I{k - 3}"] = dr.select(valid, term, 0.0)
            if refr and unrefracted:
                gu, _ = self._lobe(k, wi, wo, True, False)
                t[f"U{k - 3}"] = dr.select(valid, lamc(k + 1) * (gu * co_), 0.0)
        t["L"] = dr.select(valid, lamc(0) * lamb, 0.0)
        t["si"] = self._s(ci)
        t["so"] = self._s(co)
        t["S"] = t["si"] * t["so"]
        t["valid"] = valid
        return t

    @staticmethod
    def native_order(t):
        """Native summation order of closure83_dr.eval_pdf (f_out += O_k in k order; f_in += I_k; f = f_out + s s (f_in + L))."""
        f_out = mi.Color3f(0.0) + t["O0"] + t["O1"] + t["O2"]
        f_in = mi.Color3f(0.0) + t["I0"] + t["I1"] + t["I2"] + t["I3"]
        return dr.select(t["valid"], f_out + t["si"] * t["so"] * (f_in + t["L"]), 0.0)


def vec(a):
    a = np.ascontiguousarray(np.asarray(a, np.float32))
    return mi.Vector3f(*[mi.Float(a[:, k]) for k in range(3)])


def rgb(x):
    return np.stack([np.asarray(x[k], np.float64) for k in range(3)], -1)


def terms_dr_numpy(state, wi, wo, chunk=1 << 22):
    """Dr.Jit terms on numpy directions (float32 like the runtime) -> dict of numpy [N, 3] (and [N] valid), plus native eval."""
    out = {}
    for s in range(0, len(wi), chunk):
        e = slice(s, min(len(wi), s + chunk))
        cl = DrTerms(state)
        a, b = vec(wi[e]), vec(wo[e])
        t = cl.terms(a, b)
        f_native = cl.eval(a, b)
        f_order = DrTerms.native_order(t)
        keys = [k for k in t if k != "valid"]
        dr.eval(*[t[k] for k in keys], f_native, f_order, t["valid"])
        part = {k: rgb(t[k]) for k in keys}
        part["valid"] = np.asarray(t["valid"], bool)
        part["native"] = rgb(f_native)
        part["native_order_sum"] = rgb(f_order)
        for k, v in part.items():
            out.setdefault(k, []).append(v)
    return {k: np.concatenate(v) for k, v in out.items()}


# ------------------------------------------------------------------ raw parameter groups (64 learned parameters)
def raw_groups():
    """Orthonormal basis of the 64 raw parameters split into the brief's parameter groups. Returns dict name -> [64, d] and per-lobe
    sub-blocks. Raw layout: z[3k], z[3k+1] = log-alpha_u / log-alpha_v logits, z[3k+2] = F0 logit of GGX lobe k (0..2 outer, 3..6 inner);
    z[21 + 9c + r] = softmax logit of slot r (0 Lambert, 1..3 outer, 4..7 inner, 8 absorption) for channel c; z[48 + 5c + j] = knot j
    of s_c; z[63] = eta logit."""
    E = np.eye(64)
    rough = np.stack([(E[3 * k] + E[3 * k + 1]) / math.sqrt(2) for k in range(7)], 1)
    aniso = np.stack([(E[3 * k] - E[3 * k + 1]) / math.sqrt(2) for k in range(7)], 1)
    fres = np.stack([E[3 * k + 2] for k in range(7)], 1)
    amp = E[:, 21:48]
    att = E[:, 48:63]
    eta = E[:, 63:64]
    groups = {"amplitude": amp, "roughness": rough, "anisotropy": aniso, "fresnel": fres, "attenuation": att, "eta": eta}
    B = np.concatenate(list(groups.values()), 1)
    assert B.shape == (64, 64) and np.allclose(B.T @ B, np.eye(64))
    lobes = {}
    for k in range(7):
        name = f"O{k}" if k in OUTER else f"I{k - 3}"
        slot = 1 + k
        lobes[name] = {"amplitude": np.stack([E[21 + 9 * c + slot] for c in range(3)], 1), "roughness": rough[:, k:k + 1],
                       "anisotropy": aniso[:, k:k + 1], "fresnel": fres[:, k:k + 1]}
    lobes["L"] = {"amplitude": np.stack([E[21 + 9 * c] for c in range(3)], 1)}
    lobes["absorption_slot"] = {"amplitude": np.stack([E[21 + 9 * c + 8] for c in range(3)], 1)}
    return groups, lobes
