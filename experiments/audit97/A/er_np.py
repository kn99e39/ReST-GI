"""WL97 audit, Track A: independent numpy port of the frozen ER closure (experiments/fnto_compiled83/closure83_dr.py, arm 'ER').
Written from the formulas in that file's eval_pdf / _lobe / _refr / _s; no Dr.Jit / Mitsuba / torch. Used only by the audit checks.

eval_cos(state, wi, wo) = f(wi, wo) * cos(theta_o)  (the cosine of the SECOND argument), exactly what DrClosure83.eval returns.
fr(state, wi, wo)       = f(wi, wo)                 (BRDF value, no cosine)
"""
import math
import numpy as np

EPS = 1e-7
KNOTS = [0.0, 0.15, 0.4, 0.7, 1.0]
INNER = (3, 4, 5, 6)


def _unpack(st):
    st = np.asarray(st, np.float64)
    lam = st[:24].reshape(3, 8)
    au, av, f0 = st[24:31], st[31:38], st[38:45]
    s = st[53:68].reshape(3, 5)
    eta = float(st[68])
    return lam, au, av, f0, s, eta


def _s(s, mu):
    """piecewise-linear s(mu) per channel -> [..., 3]."""
    out = []
    for c in range(3):
        v = np.full(np.shape(mu), s[c, 0], np.float64)
        for j in range(4):
            t = np.clip((mu - KNOTS[j]) / (KNOTS[j + 1] - KNOTS[j]), 0.0, 1.0)
            v = v + (s[c, j + 1] - s[c, j]) * t
        out.append(v)
    return np.stack(out, -1)


def _refr(w, eta):
    wz = np.maximum(w[..., 2], 0.0)
    return np.stack([w[..., 0] / eta, w[..., 1] / eta, np.sqrt(max(eta * eta - 1.0, 0.0) + wz * wz) / eta], -1)


def _lobe(au, av, f0, wi, wo):
    hs = wi + wo
    h = hs / np.maximum(np.linalg.norm(hs, axis=-1, keepdims=True), 1e-6)
    ci = np.maximum(wi[..., 2], 0.0); co = np.maximum(wo[..., 2], 0.0)
    den = h[..., 0] ** 2 / au ** 2 + h[..., 1] ** 2 / av ** 2 + h[..., 2] ** 2
    D = (1.0 / (math.pi * au * av)) / (den * den)
    si = np.sqrt(ci * ci + au * au * wi[..., 0] ** 2 + av * av * wi[..., 1] ** 2)
    so = np.sqrt(co * co + au * au * wo[..., 0] ** 2 + av * av * wo[..., 1] ** 2)
    m1 = 1.0 - np.clip((wi * h).sum(-1), 0.0, 1.0)
    F = f0 + (1.0 - f0) * m1 ** 5
    return F * D / np.maximum(2.0 * (co * si + ci * so), 1e-30)


def eval_cos(state, wi, wo):
    """f(wi, wo) cos(theta_o) -> [..., 3] (wi, wo broadcastable [..., 3], local frame, z = normal)."""
    lam, au, av, f0, s, eta = _unpack(state)
    wi, wo = np.broadcast_arrays(np.asarray(wi, np.float64), np.asarray(wo, np.float64))
    ci, co = wi[..., 2], wo[..., 2]
    valid = (ci > EPS) & (co > EPS) & (np.linalg.norm(wi + wo, axis=-1) > 1e-6)
    co_ = np.maximum(co, 0.0)
    lamb = co_ / math.pi
    wi_r, wo_r = _refr(wi, eta), _refr(wo, eta)
    ie2 = 1.0 / (eta * eta)
    f_out = np.zeros(wi.shape[:-1] + (3,)); f_in = np.zeros_like(f_out)
    for k in range(7):
        refr = k in INNER
        g = _lobe(au[k], av[k], f0[k], wi_r if refr else wi, wo_r if refr else wo)
        term = (g * co_ * (ie2 if refr else 1.0))[..., None] * lam[:, k + 1]
        if refr:
            f_in += term
        else:
            f_out += term
    f = f_out + _s(s, ci) * _s(s, co) * (f_in + lamb[..., None] * lam[:, 0])
    return np.where(valid[..., None], f, 0.0)


def fr(state, wi, wo):
    """BRDF value f(wi, wo) (no cosine) -> [..., 3]."""
    wo = np.asarray(wo, np.float64)
    return eval_cos(state, wi, wo) / np.maximum(wo[..., 2], 1e-30)[..., None]


# ---------------------------------------------------------------- bins (scene86 / pt86 layout: bin = iz * NPHI + iphi)
NZ, NPHI, NB = 4, 16, 64
DOMEGA = 2 * math.pi / NB
ZB = ((np.arange(NB) // NPHI) + 0.5) / NZ
WT = ZB * DOMEGA


def bin_of(v):
    iz = np.clip(np.floor(v[..., 2] * NZ), 0, NZ - 1).astype(np.int64)
    ip = np.clip(np.floor((np.arctan2(v[..., 1], v[..., 0]) + math.pi) * (NPHI / (2 * math.pi))), 0, NPHI - 1).astype(np.int64)
    return iz * NPHI + ip


def bin_grid(n):
    """n x n midpoint strata (uniform in z and phi = uniform in solid angle) of every bin -> dirs [NB, n*n, 3]."""
    a = (np.arange(n) + 0.5) / n
    A, B = np.meshgrid(a, a, indexing='ij')
    bn = np.arange(NB)[:, None]
    z = ((bn // NPHI) + A.ravel()[None]) / NZ
    phi = ((bn % NPHI) + B.ravel()[None]) * (2 * math.pi / NPHI) - math.pi
    r = np.sqrt(np.maximum(1 - z * z, 0))
    return np.stack([r * np.cos(phi), r * np.sin(phi), z], -1)
