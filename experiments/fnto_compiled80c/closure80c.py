"""WL80C compiled closure (torch implementation, used for training / fitting / evaluation).

Closure (frozen, R = 8 lobes, local frame z = normal, x = tangent u):
  eval(wi, wo) = f(wi, wo) cos(theta_o)
  f = lambda_0c / pi                                       (lobe 0: Lambert)
    + sum_{r=1..7} lambda_rc * F_r(wi.h) D_r(h) G2_r(wi, wo) / (4 cos_i cos_o)
  D_r : anisotropic GGX (alpha_u_r, alpha_v_r), axis-aligned with the frame
  G2_r: height-correlated Smith 1 / (1 + Lambda(wi) + Lambda(wo))
  F_r : Schlick F0_r + (1 - F0_r)(1 - wi.h)^5   (wi.h = wo.h -> reciprocal)
  lambda_.c = softmax over 9 slots (8 lobes + absorption) per channel c
  -> lambda >= 0, sum_r lambda_rc <= 1; every lobe has albedo <= 1 (G2 <= G1(wi), F <= 1)
  -> passive for every wi; f symmetric -> reciprocal.

Raw compiler output z [B, 48]:
  z[:, 0:21]  -> 7 x (log-alpha_u, log-alpha_v, F0) pre-activations
  z[:, 21:48] -> 3 x 9 lambda logits
"""
import math
import torch

R_LOBES = 8
N_GGX = 7
N_RAW = 3 * N_GGX + 3 * (R_LOBES + 1)  # 48
ALPHA_MIN, ALPHA_MAX = 0.004, 1.0
LOG_AMIN, LOG_AMAX = math.log(ALPHA_MIN), math.log(ALPHA_MAX)
EPS = 1e-7


def decode(z, lam_mode='passive'):
    """Raw [B,48] -> dict(lam [B,3,8], au [B,7], av [B,7], f0 [B,7])."""
    g = z[:, :3 * N_GGX].reshape(-1, N_GGX, 3)
    au = torch.exp(LOG_AMIN + torch.sigmoid(g[..., 0]) * (LOG_AMAX - LOG_AMIN))
    av = torch.exp(LOG_AMIN + torch.sigmoid(g[..., 1]) * (LOG_AMAX - LOG_AMIN))
    f0 = torch.sigmoid(g[..., 2])
    lg = z[:, 3 * N_GGX:].reshape(-1, 3, R_LOBES + 1)
    if lam_mode == 'passive':
        lam = torch.softmax(lg, -1)[..., :R_LOBES]
    elif lam_mode == 'nonneg':  # diagnostic oracle-U only: lambda >= 0, no sum bound
        lam = torch.nn.functional.softplus(lg[..., :R_LOBES])
    else:
        raise ValueError(lam_mode)
    return {'lam': lam, 'au': au, 'av': av, 'f0': f0}


def init_bias():
    """Deterministic initial raw output: log-spaced isotropic alphas 0.01..0.8,
    F0 = 0.5, uniform lambda logits (lambda = 1/9 each)."""
    b = torch.zeros(N_RAW)
    a = torch.exp(torch.linspace(math.log(0.01), math.log(0.8), N_GGX))
    s = (torch.log(a) - LOG_AMIN) / (LOG_AMAX - LOG_AMIN)
    pre = torch.log(s / (1 - s))
    g = b[:3 * N_GGX].view(N_GGX, 3)
    g[:, 0] = pre; g[:, 1] = pre; g[:, 2] = 0.0
    return b


def _lam_ggx(au, av, w):
    """Smith Lambda for anisotropic GGX; au, av [..., L], w [..., 3] -> [..., L]."""
    wz2 = (w[..., 2:3] ** 2).clamp_min(EPS)
    t = (au ** 2 * w[..., 0:1] ** 2 + av ** 2 * w[..., 1:2] ** 2) / wz2
    return 0.5 * (torch.sqrt(1 + t) - 1)


def eval_closure(p, wi, wo):
    """p: decoded params gathered per lane (lam [N,3,8], au/av/f0 [N,7]);
    wi, wo [N,3] unit, local frame. Returns eval = f cos_o, [N,3]."""
    ci, co = wi[:, 2], wo[:, 2]
    valid = (ci > EPS) & (co > EPS)
    ci_ = ci.clamp_min(EPS); co_ = co.clamp_min(EPS)
    hs = wi + wo
    hn = torch.linalg.norm(hs, dim=1, keepdim=True)
    valid = valid & (hn[:, 0] > 1e-6)
    h = hs / hn.clamp_min(1e-6)
    au, av, f0 = p['au'], p['av'], p['f0']
    hx, hy, hz = h[:, 0:1], h[:, 1:2], h[:, 2:3]
    den = hx ** 2 / au ** 2 + hy ** 2 / av ** 2 + hz ** 2
    D = 1.0 / (math.pi * au * av * den ** 2)
    G2 = 1.0 / (1 + _lam_ggx(au, av, wi) + _lam_ggx(au, av, wo))
    c = (wi * h).sum(1, keepdim=True).clamp(0, 1)
    F = f0 + (1 - f0) * (1 - c) ** 5
    spec = F * D * G2 / (4 * ci_[:, None])          # [N,7] = f_r cos_o
    lam = p['lam']
    out = lam[:, :, 0] * (co_[:, None] / math.pi) + torch.einsum('ncr,nr->nc', lam[:, :, 1:], spec)
    return torch.where(valid[:, None], out, torch.zeros_like(out))


def gather(p, idx):
    return {k: v[idx] for k, v in p.items()}


def sampling_probs(p):
    """Lobe selection probabilities (independent of wi): mean_c lambda_rc * w_r,
    w_0 = 1, w_r = F0_r + (1 - F0_r)/21 (Schlick hemispherical average)."""
    lam = p['lam'].mean(1)  # [B,8]
    w = torch.cat([torch.ones_like(p['f0'][:, :1]), p['f0'] + (1 - p['f0']) / 21.0], 1)
    s = lam * w
    tot = s.sum(1, keepdim=True)
    return torch.where(tot > 0, s / tot.clamp_min(1e-30), torch.full_like(s, 1.0 / R_LOBES))


def compiled_state(p):
    """Flat compiled state per material: [B, 53] = lam(24) au(7) av(7) f0(7) probs(8)."""
    pr = sampling_probs(p)
    return torch.cat([p['lam'].reshape(-1, 3 * R_LOBES), p['au'], p['av'], p['f0'], pr], 1)


class Compiler(torch.nn.Module):
    """Shared material compiler: descriptor (9, normalised to [-1,1]) -> raw closure z."""

    def __init__(self, width=128, depth=3):
        super().__init__()
        layers, d = [], 9
        for _ in range(depth):
            layers += [torch.nn.Linear(d, width), torch.nn.SiLU()]
            d = width
        self.body = torch.nn.Sequential(*layers)
        self.head = torch.nn.Linear(d, N_RAW)
        with torch.no_grad():
            self.head.weight.mul_(0.1)
            self.head.bias.copy_(init_bias())

    def forward(self, x):
        return self.head(self.body(x))
