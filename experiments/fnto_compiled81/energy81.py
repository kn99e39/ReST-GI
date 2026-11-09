"""WL81 energy arms (torch). See protocol energy_arms and amendment 1.

control   (WL80C): lambda = softmax over 9 slots per channel  -> sum_r lambda_rc <= 1 (certificate: every lobe albedo <= 1).
candidate (C81_limit_complete_sup): lambda = softplus (>= 0, no sum bound), f_c = g_c / max(1, M_c),
           M_c = sup over the CLOSED upper hemisphere (including the grazing limit cos theta_i = 0) of the mixture albedo
           a_c(wi) = lambda_0c + sum_r lambda_rc A_r(wi)   (A_r: horizon-safe VNDF estimator, closure81.lobe_albedo).
The runtime closure of the candidate is the corrected WL80C closure with lambda' = lambda / max(1, M_c).
"""
import math
import numpy as np
import torch
import closure81 as K

C = K.C


def decode(arm, z):
    return C.decode(z, 'passive' if arm == 'control' else 'nonneg')


def init_raw(arm, n, device):
    z = C.init_bias().to(device)[None].repeat(n, 1)
    if arm != 'control':     # candidate and the unbounded diagnostic
        lg = z[:, 3 * C.N_GGX:].view(n, 3, C.R_LOBES + 1)
        lg[...] = math.log(math.expm1(1 / 9))        # lambda = 1/9 per slot, as the control's uniform softmax
    return z


def scaled(p, M):
    s = 1.0 / torch.clamp(M, min=1.0)
    q = dict(p)
    q['lam'] = p['lam'] * s[:, :, None]
    return q


# ---------------------------------------------------------------- grids (frozen by amendment 1)
def fit_grid(device):
    mu = [0.0, 0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.7, 0.85, 1.0]
    return K.wi_grid_mu(mu, [0, 30, 60, 90], device)


def compile_grids(device):
    """Base grid G1 and its refinement G2 (midpoints in mu and phi)."""
    t = np.linspace(0, 1, 64)
    mu1 = np.unique(np.concatenate([[0.0, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3], (t ** 2)]))        # dense towards the horizon, exact limit 0
    ph1 = np.linspace(0, 90, 16)
    mu2 = 0.5 * (mu1[1:] + mu1[:-1]); ph2 = 0.5 * (ph1[1:] + ph1[:-1])
    g1 = K.wi_grid_mu(mu1, ph1, device)
    g2 = torch.cat([K.wi_grid_mu(mu2, ph1, device), K.wi_grid_mu(mu1, ph2, device), K.wi_grid_mu(mu2, ph2, device)])
    return g1, g2, (len(mu1), len(ph1))


def fit_M(p, wi, u):
    """Differentiable M estimate during fitting / training (coarse grid incl. mu = 0, common random numbers)."""
    A = K.lobe_albedo(p['au'], p['av'], p['f0'], wi, u)
    return K.mixture_albedo(p['lam'], A).max(-1).values


def _grid_ub(p, wi, u, chunk=8, mchunk=16):
    """Mixture albedo and a conservative standard error on a wi grid (no grad); chunked over materials and directions."""
    B = p['au'].shape[0]
    a_out = torch.empty(B, 3, wi.shape[0], device=wi.device)
    s_out = torch.empty_like(a_out)
    with torch.no_grad():
        for b0 in range(0, B, mchunk):
            bs = slice(b0, min(B, b0 + mchunk))
            for s in range(0, wi.shape[0], chunk):
                ws = slice(s, min(wi.shape[0], s + chunk))
                A, se = K.lobe_albedo(p['au'][bs], p['av'][bs], p['f0'][bs], wi[ws], u, with_se=True)
                a_out[bs, :, ws] = K.mixture_albedo(p['lam'][bs], A)
                s_out[bs, :, ws] = torch.einsum('bcr,brw->bcw', p['lam'][bs, :, 1:], se)   # conservative (correlated samples)
    return a_out, s_out


def compile_M(p, g1, g2, u, shape1):
    """M_c = max over G1 u G2 of (a + 3 se) + refinement margin max(0, max_G2 a - max_G1 a) + local-variation margin
    0.5 max |a(x) - a(y)| over G1 grid neighbours adjacent to the G1 argmax. Returns M [B,3] and diagnostics."""
    a1, s1 = _grid_ub(p, g1, u)
    a2, s2 = _grid_ub(p, g2, u)
    B = a1.shape[0]
    ub = torch.cat([a1 + 3 * s1, a2 + 3 * s2], -1).max(-1).values
    refine = torch.clamp(a2.max(-1).values - a1.max(-1).values, min=0)
    nm, nph = shape1
    A1 = a1.view(B, 3, nm, nph)
    idx = a1.argmax(-1)
    im, ip = idx // nph, idx % nph
    loc = torch.zeros_like(ub)
    for dm, dp in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
        jm = (im + dm).clamp(0, nm - 1); jp = (ip + dp).clamp(0, nph - 1)
        v = torch.gather(A1.view(B, 3, -1), -1, (jm * nph + jp)[..., None])[..., 0]
        loc = torch.maximum(loc, (a1.max(-1).values - v).abs())
    M = ub + refine + 0.5 * loc
    return M, {'ub': ub, 'refine': refine, 'local': 0.5 * loc, 'argmax_mu_index': im, 'argmax_phi_index': ip,
               'a_max': torch.maximum(a1.max(-1).values, a2.max(-1).values)}


def compiled_params(arm, z, grids=None, u=None):
    """Raw z -> runtime parameters (control: decode; candidate: decode + compiled scaling)."""
    p = decode(arm, z)
    if arm == 'control':
        return p, None
    g1, g2, sh = grids
    M, diag = compile_M(p, g1, g2, u, sh)
    return scaled(p, M), (M, diag)
