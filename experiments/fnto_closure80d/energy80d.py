"""WL80D candidate energy parameterisation (torch).

CANDIDATE: g_c(wi,wo) = sum_r lambda_rc B_r(wi,wo), lambda >= 0 (softplus, no sum bound);
           M_c = sup_wi int g_c cos dwo;  f_c = g_c / max(1, M_c)
Every lobe B_r is the unchanged WL80C basis (Lambert, Schlick-Fresnel anisotropic GGX with
height-correlated Smith G2). The mixture albedo is linear in lambda:
    a_c(wi) = lambda_0c + sum_{r>=1} lambda_rc A_r(wi),
    A_r(wi) = E_{h ~ D_wi}[ F_r(wi.h) G2(wi,wo) / G1(wi) * 1(wo.z > 0) ]   (VNDF estimator, integrand in [0,1])
D is symmetric under x -> -x and y -> -y, so A_r(theta, phi) needs phi in [0, 90 deg] only.

M is estimated numerically (grid over wi x stratified VNDF samples): the result is EMPIRICALLY
passive (sup over a finite wi grid, + 3 standard errors), not a mathematical certificate for the
continuum. Compile once per material; the runtime closure is the WL80C closure with weights
lambda' = lambda / max(1, M) (sum lambda' may exceed 1), i.e. unchanged runtime code and state size.
"""
import math
import numpy as np
import torch
import closure80c as C

TWO_PI = 2 * math.pi


def wi_grid(theta_deg, phi_deg, device):
    th = torch.tensor(np.radians(theta_deg), dtype=torch.float32, device=device)
    ph = torch.tensor(np.radians(phi_deg), dtype=torch.float32, device=device)
    T, P = torch.meshgrid(th, ph, indexing='ij')
    st = torch.sin(T)
    return torch.stack([st * torch.cos(P), st * torch.sin(P), torch.cos(T)], -1).reshape(-1, 3)


def strat_u(n0, n1, seed, device):
    g = torch.Generator(device='cpu').manual_seed(seed)
    i = torch.arange(n0).repeat_interleave(n1).float()
    j = torch.arange(n1).repeat(n0).float()
    u = torch.stack([(i + torch.rand(n0 * n1, generator=g)) / n0, (j + torch.rand(n0 * n1, generator=g)) / n1], 1)
    return u.clamp(0, 1 - 1e-7).to(device)


def lobe_albedo(au, av, f0, wi, u, with_se=False):
    """au, av, f0 [B, L]; wi [W, 3]; u [K, 2] -> A [B, L, W] (and se)."""
    a_u = au[:, :, None, None]; a_v = av[:, :, None, None]; F0 = f0[:, :, None, None]
    w = wi[None, None, :, None, :]                          # [1,1,W,1,3]
    x, y, z = w[..., 0], w[..., 1], w[..., 2]
    sx, sy, sz = a_u * x, a_v * y, z.expand_as(a_u * x)
    n = torch.sqrt(sx * sx + sy * sy + sz * sz)
    sx, sy, sz = sx / n, sy / n, sz / n
    phi = TWO_PI * u[:, 0]; uz = u[:, 1]
    zc = (1 - uz) * (1 + sz) - sz
    st = torch.sqrt((1 - zc * zc).clamp_min(0))
    hx, hy, hz = st * torch.cos(phi) + sx, st * torch.sin(phi) + sy, zc + sz
    hx, hy = a_u * hx, a_v * hy
    hn = torch.sqrt(hx * hx + hy * hy + hz * hz).clamp_min(1e-12)
    hx, hy, hz = hx / hn, hy / hn, hz / hn
    d = (x * hx + y * hy + z * hz)
    wox, woy, woz = 2 * d * hx - x, 2 * d * hy - y, 2 * d * hz - z
    li = 0.5 * (torch.sqrt(1 + (a_u ** 2 * x * x + a_v ** 2 * y * y) / (z * z).clamp_min(1e-12)) - 1)
    lo = 0.5 * (torch.sqrt(1 + (a_u ** 2 * wox * wox + a_v ** 2 * woy * woy) / (woz * woz).clamp_min(1e-12)) - 1)
    F = F0 + (1 - F0) * (1 - d.clamp(0, 1)) ** 5
    v = torch.where(woz > 1e-7, F * (1 + li) / (1 + li + lo), torch.zeros_like(lo))
    A = v.mean(-1)
    if with_se:
        return A, v.std(-1) / math.sqrt(v.shape[-1])
    return A


def mixture_albedo(lam, A):
    """lam [B,3,8], A [B,7,W] -> a [B,3,W]."""
    return lam[:, :, :1] + torch.einsum('bcr,brw->bcw', lam[:, :, 1:], A)


def decode_candidate(z):
    """Raw [B,48] -> unscaled candidate params (lambda = softplus, lobe 0..7; absorption slot unused)."""
    p = C.decode(z, 'nonneg')
    return p


def scaled(p, M):
    """Apply f = g / max(1, M) per channel: lambda' = lambda / max(1, M_c)."""
    s = 1.0 / torch.clamp(M, min=1.0)                       # [B,3]
    q = dict(p)
    q['lam'] = p['lam'] * s[:, :, None]
    return q


def fit_M(p, wi, u):
    A = lobe_albedo(p['au'], p['av'], p['f0'], wi, u)
    return mixture_albedo(p['lam'], A).max(-1).values      # [B,3]


def compile_M(p, wi, u, wi_chunk=16):
    """High-accuracy M_c = max_grid (a_c + 3 se_c); returns M [B,3], argmax index, a_max, se_at_max."""
    B = p['au'].shape[0]
    a_all, se_all = [], []
    with torch.no_grad():
        for s in range(0, wi.shape[0], wi_chunk):
            A, se = lobe_albedo(p['au'], p['av'], p['f0'], wi[s:s + wi_chunk], u, with_se=True)
            a_all.append(mixture_albedo(p['lam'], A))
            # conservative se of the mixture: sum of lambda-weighted lobe se (correlated samples)
            se_all.append(torch.einsum('bcr,brw->bcw', p['lam'][:, :, 1:], se))
    a = torch.cat(a_all, -1); se = torch.cat(se_all, -1)
    ub = a + 3 * se
    M, idx = ub.max(-1)
    return M, idx, a.max(-1).values, torch.gather(se, -1, idx[..., None])[..., 0]
