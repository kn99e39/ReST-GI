"""WL82 closures (torch): the WL81 corrected R = 8 basis plus two controlled mechanisms.

Raw layout z [B, 64]:
  z[:, 0:48]   WL80C / WL81 R = 8 raw parameters (closure80c.decode: 7 x (log-alpha_u, log-alpha_v, F0) + 3 x 9 lambda logits)
  z[:, 48:63]  attenuation knots s_c(mu_j), c = RGB, mu_j in MU_KNOTS (sigmoid -> [0, 1]); used by E / Einc / ER
  z[:, 63]     effective coat index eta_e = 1 + (ETA_MAX - 1) sigmoid(z); used by ER only

Lobe groups (fixed by index, identical for every arm): Lambert lobe 0 and GGX lobes 4..7 = INNER group, GGX lobes 1..3 = OUTER group.

  C0   (WL81 closure):            f = lam_0 / pi + sum_r lam_r g_r(wi, wo)
  E    (energy modulation):       f = sum_outer lam_r g_r + s(mu_i) s(mu_o) [lam_0 / pi + sum_inner lam_r g_r(wi, wo)]
  Einc (DIAGNOSTIC, not reciprocal): f = s(mu_i) [C0 closure]           (incident-only energy scale of the whole closure)
  ER   (E + refraction-aware):    f = sum_outer lam_r g_r + s(mu_i) s(mu_o) [lam_0 / pi + eta^-2 sum_inner lam_r g_r(t(wi), t(wo))]
       t(w) = (w_x / eta, w_y / eta, sqrt(1 - (1 - w_z^2) / eta^2))  (direction refracted into the coat, kept in the upper hemisphere)

g_r = horizon-safe anisotropic GGX lobe F D / (2 (cos_o sqrt(cos_i^2 + t_i) + cos_i sqrt(cos_o^2 + t_o)))  (WL81 closure81 form / cos_o).
s_c(mu): piecewise linear in mu = cos(theta) through the knots, values in [0, 1].

Energy modes: 'cert' = WL80C softmax over 9 slots per channel (sum_r lam_rc <= 1); 'unb' = softplus (no bound, diagnostic).
Certificate (cert): every component has directional albedo <= 1 (Lambert 1; outer GGX <= 1 since F <= 1, G2 <= G1; inner GGX
  <= s(mu_i) A_r(wi) <= 1; refracted inner: albedo = s(mu_i) int_cone s g_r(t(wi), w') cos' dw' <= A_r(t(wi)) <= 1 by the solid-angle
  identity cos_o dw_o = eta^2 cos_o' dw_o'), so a_c(wi) <= sum_r lam_rc <= 1 for every wi. Reciprocity of C0 / E / ER is exact by
  construction (each term symmetric in (wi, wo)); Einc is not reciprocal.
"""
import math
import os
import sys
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'fnto_compiled81'))
import closure81 as K  # noqa: E402
C = K.install()

EPS = C.EPS
N_RAW = 64
MU_KNOTS = torch.tensor([0.0, 0.15, 0.4, 0.7, 1.0], dtype=torch.float64)
NK = 5
ETA_MAX = 2.5
OUTER = [0, 1, 2]          # GGX lobe indices (lam index = 1 + k)
INNER = [3, 4, 5, 6]
ARMS = ('C0', 'E', 'Einc', 'ER')


def decode(arm, mode, z):
    p = C.decode(z[:, :48], 'passive' if mode == 'cert' else 'nonneg')
    if arm in ('E', 'Einc', 'ER'):
        p['s'] = torch.sigmoid(z[:, 48:63]).reshape(-1, 3, NK)
    if arm == 'ER':
        p['eta'] = 1.0 + (ETA_MAX - 1.0) * torch.sigmoid(z[:, 63])
    return p


def init_raw(arm, mode, n, device):
    z = torch.zeros(n, N_RAW, device=device)
    z[:, :48] = C.init_bias().to(device)[None]
    if mode == 'unb':
        lg = z[:, 3 * C.N_GGX:48].view(n, 3, C.R_LOBES + 1)
        lg[...] = math.log(math.expm1(1 / 9))        # lambda = 1/9 per slot, as the softmax init
    z[:, 48:63] = 3.0                                 # s = 0.953 at every knot
    z[:, 63] = math.log((0.5 / (ETA_MAX - 1.0)) / (1 - 0.5 / (ETA_MAX - 1.0)))   # eta_e = 1.5
    return z


def n_params(arm):
    return {'C0': 48, 'E': 63, 'Einc': 63, 'ER': 64}[arm]


def gather(p, idx):
    return {k: v[idx] for k, v in p.items()}


def s_interp(s, mu):
    """s [N,3,NK], mu [N] -> [N,3] piecewise linear in mu (clamped to [0, 1])."""
    k = MU_KNOTS.to(mu.device, mu.dtype)
    m = mu.clamp(0, 1)
    j = torch.bucketize(m, k[1:-1], right=True)          # segment index 0..NK-2
    t = (m - k[j]) / (k[j + 1] - k[j])
    jj = j[:, None, None].expand(-1, 3, 1)
    s0 = torch.gather(s, 2, jj)[..., 0]
    s1 = torch.gather(s, 2, jj + 1)[..., 0]
    return s0 + (s1 - s0) * t[:, None]


def refract_up(w, eta):
    """Direction refracted into a medium of relative index eta (>= 1), reported in the upper hemisphere. w [N,3], eta [N]."""
    tx, ty = w[:, 0] / eta, w[:, 1] / eta
    tz = torch.sqrt((1 - tx * tx - ty * ty).clamp_min(0))
    return torch.stack([tx, ty, tz], 1)


def exit_up(w_in, eta):
    """Inverse of refract_up: inside direction -> outside direction; valid iff eta^2 |w_t|^2 < 1 (no total internal reflection)."""
    tx, ty = w_in[:, 0] * eta, w_in[:, 1] * eta
    r2 = tx * tx + ty * ty
    ok = (r2 < 1) & (w_in[:, 2] > 0)
    tz = torch.sqrt((1 - r2).clamp_min(0))
    return torch.stack([tx, ty, tz], 1), ok


def lobe_f(au, av, f0, wi, wo):
    """Horizon-safe GGX lobes (BRDF value, no cosine). au/av/f0 [N,L]; wi, wo [N,3] -> f [N,L], valid [N]."""
    ci, co = wi[:, 2], wo[:, 2]
    valid = (ci > EPS) & (co > EPS)
    hs = wi + wo
    hn = torch.linalg.norm(hs, dim=1, keepdim=True)
    valid = valid & (hn[:, 0] > 1e-6)
    h = hs / hn.clamp_min(1e-6)
    hx, hy, hz = h[:, 0:1], h[:, 1:2], h[:, 2:3]
    den = hx ** 2 / au ** 2 + hy ** 2 / av ** 2 + hz ** 2
    D = 1.0 / (math.pi * au * av * den ** 2)
    ci_, co_ = ci.clamp_min(0)[:, None], co.clamp_min(0)[:, None]
    ti = au ** 2 * wi[:, 0:1] ** 2 + av ** 2 * wi[:, 1:2] ** 2
    to = au ** 2 * wo[:, 0:1] ** 2 + av ** 2 * wo[:, 1:2] ** 2
    den2 = 2 * (co_ * torch.sqrt(ci_ ** 2 + ti) + ci_ * torch.sqrt(co_ ** 2 + to))
    c = (wi * h).sum(1, keepdim=True).clamp(0, 1)
    F = f0 + (1 - f0) * (1 - c) ** 5
    return F * D / den2.clamp_min(1e-30), valid


def eval_closure(arm, p, wi, wo):
    """eval = f cos(theta_o) [N,3] for lanes with gathered parameters p."""
    ci, co = wi[:, 2], wo[:, 2]
    co_ = co.clamp_min(0)[:, None]
    lam = p['lam']
    if arm in ('C0', 'Einc'):
        f, valid = lobe_f(p['au'], p['av'], p['f0'], wi, wo)
        out = lam[:, :, 0] * (co_ / math.pi) + torch.einsum('ncr,nr->nc', lam[:, :, 1:], f * co_)
        if arm == 'Einc':
            out = s_interp(p['s'], ci) * out
        return torch.where(valid[:, None], out, torch.zeros_like(out))
    fo, valid = lobe_f(p['au'][:, OUTER], p['av'][:, OUTER], p['f0'][:, OUTER], wi, wo)
    outer = torch.einsum('ncr,nr->nc', lam[:, :, 1:4], fo * co_)
    if arm == 'E':
        fi, _ = lobe_f(p['au'][:, INNER], p['av'][:, INNER], p['f0'][:, INNER], wi, wo)
    elif arm == 'ER':
        eta = p['eta']
        fi, _ = lobe_f(p['au'][:, INNER], p['av'][:, INNER], p['f0'][:, INNER], refract_up(wi, eta), refract_up(wo, eta))
        fi = fi / (eta * eta)[:, None]
    else:
        raise ValueError(arm)
    inner = lam[:, :, 0] * (co_ / math.pi) + torch.einsum('ncr,nr->nc', lam[:, :, 4:8], fi * co_)
    out = outer + s_interp(p['s'], ci) * s_interp(p['s'], co) * inner
    return torch.where(valid[:, None], out, torch.zeros_like(out))


# ---------------------------------------------------------------- sampling (torch; used by the focused tests)
def vndf_h(au, av, w, u1, u2):
    """Anisotropic GGX visible-normal sample (spherical caps). au, av [N]; w [N,3] (w_z >= 0) -> h [N,3]."""
    sx, sy, sz = au * w[:, 0], av * w[:, 1], w[:, 2]
    n = torch.sqrt(sx * sx + sy * sy + sz * sz).clamp_min(1e-30)
    sx, sy, sz = sx / n, sy / n, sz / n
    phi = 2 * math.pi * u1
    zc = (1 - u2) * (1 + sz) - sz
    st = torch.sqrt((1 - zc * zc).clamp_min(0))
    hx, hy, hz = au * (st * torch.cos(phi) + sx), av * (st * torch.sin(phi) + sy), (zc + sz).clamp_min(0)
    hn = torch.sqrt(hx * hx + hy * hy + hz * hz).clamp_min(1e-30)
    return torch.stack([hx / hn, hy / hn, hz / hn], 1)


def vndf_pdf_wo(au, av, wi, wo):
    """Density of wo = reflect(wi, h), h ~ VNDF(wi): D / (2 (cos_i + sqrt(cos_i^2 + t_i))). au, av [N]."""
    hs = wi + wo
    h = hs / torch.linalg.norm(hs, dim=1, keepdim=True).clamp_min(1e-30)
    den = h[:, 0] ** 2 / au ** 2 + h[:, 1] ** 2 / av ** 2 + h[:, 2] ** 2
    D = 1.0 / (math.pi * au * av * den ** 2)
    ci = wi[:, 2].clamp_min(0)
    ti = au ** 2 * wi[:, 0] ** 2 + av ** 2 * wi[:, 1] ** 2
    p = D / (2 * (ci + torch.sqrt(ci * ci + ti))).clamp_min(1e-30)
    # validity exactly as eval (WL81 runtime closure81_dr: no wi.h condition; wi.h = (1 + wi.wo) / |wi + wo| > 0 mathematically, but a
    # float32 test of it fails at near-mirror grazing pairs and would zero the pdf where eval is non-zero - dev smoke 2026-11-10)
    hn = torch.linalg.norm(hs, dim=1)
    return torch.where((wo[:, 2] > EPS) & (wi[:, 2] > EPS) & (hn > 1e-6), p, torch.zeros_like(p))


def sampling_probs(arm, p):
    """Lobe selection probabilities (wi-independent, WL80C rule on the lambda / F0 part)."""
    return C.sampling_probs({'lam': p['lam'], 'f0': p['f0']})


def pdf_closure(arm, p, wi, wo):
    pr = sampling_probs(arm, p)
    co = wo[:, 2]
    pdf = pr[:, 0] * co.clamp_min(0) / math.pi
    refr = arm == 'ER'
    if refr:
        eta = p['eta']
        wi_r, wo_r = refract_up(wi, eta), refract_up(wo, eta)
        jac = co.clamp_min(0) / (eta * eta * wo_r[:, 2].clamp_min(1e-30))
    for k in range(7):
        if refr and k in INNER:
            pk = vndf_pdf_wo(p['au'][:, k], p['av'][:, k], wi_r, wo_r) * jac
        else:
            pk = vndf_pdf_wo(p['au'][:, k], p['av'][:, k], wi, wo)
        pdf = pdf + pr[:, k + 1] * pk
    ok = (wi[:, 2] > EPS) & (co > EPS)
    return torch.where(ok, pdf, torch.zeros_like(pdf))


def sample_closure(arm, p, wi, u0, u1, u2):
    """Returns wo [N,3], valid [N] (sample produced a direction above the horizon), selected lobe [N]."""
    pr = sampling_probs(arm, p)
    cdf = torch.cumsum(pr, 1)
    r = (u0[:, None] >= cdf[:, :-1]).sum(1)          # selected lobe 0..7
    N = wi.shape[0]
    wo = torch.zeros_like(wi)
    ok = torch.zeros(N, dtype=torch.bool, device=wi.device)
    s = r == 0
    if s.any():
        rr = torch.sqrt(u1[s]); ph = 2 * math.pi * u2[s]
        wo[s] = torch.stack([rr * torch.cos(ph), rr * torch.sin(ph), torch.sqrt((1 - u1[s]).clamp_min(0))], 1)
        ok[s] = wo[s][:, 2] > EPS
    for k in range(7):
        s = r == k + 1
        if not s.any():
            continue
        au, av = p['au'][s, k], p['av'][s, k]
        if arm == 'ER' and k in INNER:
            eta = p['eta'][s]
            w_in = refract_up(wi[s], eta)
            h = vndf_h(au, av, w_in, u1[s], u2[s])
            wr = 2 * (w_in * h).sum(1, keepdim=True) * h - w_in
            out, ex = exit_up(wr, eta)
            wo[s] = out
            ok[s] = ex & (out[:, 2] > EPS)
        else:
            h = vndf_h(au, av, wi[s], u1[s], u2[s])
            w = 2 * (wi[s] * h).sum(1, keepdim=True) * h - wi[s]
            wo[s] = w
            ok[s] = w[:, 2] > EPS
    ok = ok & (wi[:, 2] > EPS)
    return wo, ok, r


def compiled_state_floats(arm):
    """Runtime state per material: WL81 53 floats (lam 24, au/av/f0 21, sampling probs 8) + 15 knots (E / Einc / ER) + 1 eta (ER)."""
    return {'C0': 53, 'E': 68, 'Einc': 68, 'ER': 69}[arm]
