"""WL81 closure (torch): the WL80C basis with the horizon-safe closed forms (software fix 2) and the two energy arms.

Everything except the per-lobe GGX evaluation is imported from WL80C (closure80c: decode, init_bias, gather, compiled_state,
sampling_probs, raw layout). `install()` replaces closure80c.eval_closure by eval_closure() below so that every reused WL80C helper
(train80c.fit_free, eval80c.lane_metrics, ...) evaluates with the corrected numerics; nothing is copied.

Horizon-safe GGX lobe (algebraically equal to the WL80C form F D G2 / (4 cos_i), G2 = 1 / (1 + Lambda_i + Lambda_o)):
    t(w)   = alpha_u^2 w_x^2 + alpha_v^2 w_y^2
    eval_r = F D cos_o / (2 (cos_o sqrt(cos_i^2 + t_i) + cos_i sqrt(cos_o^2 + t_o)))
    pdf_r  = D / (2 (cos_i + sqrt(cos_i^2 + t_i)))                    (VNDF pdf G1(wi) D / (4 cos_i))
No clamp, symmetric in (wi, wo) after division by cos_o, finite at the grazing limit.
"""
import math
import os
import sys
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'fnto_compiled80c'))
import closure80c as C  # noqa: E402

EPS = C.EPS
R_LOBES, N_GGX, N_RAW = C.R_LOBES, C.N_GGX, C.N_RAW
_ORIGINAL_EVAL = C.eval_closure


def eval_closure(p, wi, wo):
    """Horizon-safe drop-in replacement of closure80c.eval_closure (same signature / semantics)."""
    ci, co = wi[:, 2], wo[:, 2]
    valid = (ci > EPS) & (co > EPS)
    hs = wi + wo
    hn = torch.linalg.norm(hs, dim=1, keepdim=True)
    valid = valid & (hn[:, 0] > 1e-6)
    h = hs / hn.clamp_min(1e-6)
    au, av, f0 = p['au'], p['av'], p['f0']
    hx, hy, hz = h[:, 0:1], h[:, 1:2], h[:, 2:3]
    den = hx ** 2 / au ** 2 + hy ** 2 / av ** 2 + hz ** 2
    D = 1.0 / (math.pi * au * av * den ** 2)
    ci_, co_ = ci.clamp_min(0)[:, None], co.clamp_min(0)[:, None]
    ti = au ** 2 * wi[:, 0:1] ** 2 + av ** 2 * wi[:, 1:2] ** 2
    to = au ** 2 * wo[:, 0:1] ** 2 + av ** 2 * wo[:, 1:2] ** 2
    den2 = 2 * (co_ * torch.sqrt(ci_ ** 2 + ti) + ci_ * torch.sqrt(co_ ** 2 + to))
    c = (wi * h).sum(1, keepdim=True).clamp(0, 1)
    F = f0 + (1 - f0) * (1 - c) ** 5
    spec = F * D * co_ / den2.clamp_min(1e-30)
    lam = p['lam']
    out = lam[:, :, 0] * (co_ / math.pi) + torch.einsum('ncr,nr->nc', lam[:, :, 1:], spec)
    return torch.where(valid[:, None], out, torch.zeros_like(out))


def install():
    C.eval_closure = eval_closure
    return C


def original_eval_closure(p, wi, wo):
    return _ORIGINAL_EVAL(p, wi, wo)


# ---------------------------------------------------------------- per-lobe directional albedo (horizon-safe VNDF estimator)
TWO_PI = 2 * math.pi


def wi_grid_mu(mu, phi_deg, device):
    mu = torch.as_tensor(mu, dtype=torch.float64)
    ph = torch.as_tensor(phi_deg, dtype=torch.float64) * math.pi / 180
    M, P = torch.meshgrid(mu, ph, indexing='ij')
    st = torch.sqrt((1 - M * M).clamp_min(0))
    return torch.stack([st * torch.cos(P), st * torch.sin(P), M], -1).reshape(-1, 3).float().to(device)


def strat_u(n0, n1, seed, device):
    g = torch.Generator(device='cpu').manual_seed(seed)
    i = torch.arange(n0).repeat_interleave(n1).float()
    j = torch.arange(n1).repeat(n0).float()
    u = torch.stack([(i + torch.rand(n0 * n1, generator=g)) / n0, (j + torch.rand(n0 * n1, generator=g)) / n1], 1)
    return u.clamp(0, 1 - 1e-7).to(device)


def lobe_albedo(au, av, f0, wi, u, with_se=False):
    """A_r(wi) = E_{h ~ VNDF(wi)}[F(wi.h) G2/G1(wi) 1(wo.z > 0)], integrand in [0, 1], horizon-safe (exact at cos_i = 0).
    au, av, f0 [B, L]; wi [W, 3]; u [K, 2] -> [B, L, W]."""
    a_u = au[:, :, None, None]; a_v = av[:, :, None, None]; F0 = f0[:, :, None, None]
    w = wi[None, None, :, None, :]
    x, y, z = w[..., 0], w[..., 1], w[..., 2]
    sx, sy, sz = a_u * x, a_v * y, z.expand_as(a_u * x)
    n = torch.sqrt(sx * sx + sy * sy + sz * sz).clamp_min(1e-30)
    sx, sy, sz = sx / n, sy / n, sz / n
    phi = TWO_PI * u[:, 0]; uz = u[:, 1]
    zc = (1 - uz) * (1 + sz) - sz
    st = torch.sqrt((1 - zc * zc).clamp_min(0))
    hx, hy, hz = st * torch.cos(phi) + sx, st * torch.sin(phi) + sy, zc + sz
    hx, hy = a_u * hx, a_v * hy
    hn = torch.sqrt(hx * hx + hy * hy + hz * hz).clamp_min(1e-30)
    hx, hy, hz = hx / hn, hy / hn, hz / hn
    d = (x * hx + y * hy + z * hz)
    wox, woy, woz = 2 * d * hx - x, 2 * d * hy - y, 2 * d * hz - z
    ci = z.clamp_min(0); co = woz.clamp_min(0)
    si = torch.sqrt(ci * ci + a_u ** 2 * x * x + a_v ** 2 * y * y)
    so = torch.sqrt(co * co + a_u ** 2 * wox * wox + a_v ** 2 * woy * woy)
    # G2 / G1(wi) = (1 + L_i) / (1 + L_i + L_o) = (ci + s_i) co / (co s_i + ci s_o)
    ratio = (ci + si) * co / (co * si + ci * so).clamp_min(1e-30)
    F = F0 + (1 - F0) * (1 - d.clamp(0, 1)) ** 5
    v = torch.where(woz > 1e-7, F * ratio, torch.zeros_like(ratio))
    A = v.mean(-1)
    if with_se:
        return A, v.std(-1) / math.sqrt(v.shape[-1])
    return A


def mixture_albedo(lam, A):
    """lam [B,3,8], A [B,7,W] -> a [B,3,W]."""
    return lam[:, :, :1] + torch.einsum('bcr,brw->bcw', lam[:, :, 1:], A)
