"""numpy port of closure82.eval_closure('ER') for one 69-float compiled state (CPU, float64). Scratch, WL96 design D4."""
import numpy as np

KN = np.array([0.0, 0.15, 0.4, 0.7, 1.0])


def unpack(st):
    st = np.asarray(st, np.float64)
    return dict(lam=st[0:24].reshape(3, 8), au=st[24:31], av=st[31:38], f0=st[38:45], s=st[53:68].reshape(3, 5), eta=st[68])


def s_interp(s, mu):
    m = np.clip(mu, 0, 1)
    return np.stack([np.interp(m, KN, s[c]) for c in range(3)], -1)


def refr(w, eta):
    tx, ty = w[..., 0] / eta, w[..., 1] / eta
    tz = np.sqrt(np.clip(1 - tx * tx - ty * ty, 0, None))
    return np.stack([tx, ty, tz], -1)


def lobe_f(au, av, f0, wi, wo):
    ci, co = wi[..., 2:3], wo[..., 2:3]
    hs = wi + wo
    hn = np.linalg.norm(hs, axis=-1, keepdims=True)
    h = hs / np.maximum(hn, 1e-6)
    den = h[..., 0:1] ** 2 / au ** 2 + h[..., 1:2] ** 2 / av ** 2 + h[..., 2:3] ** 2
    D = 1.0 / (np.pi * au * av * den ** 2)
    ci_, co_ = np.clip(ci, 0, None), np.clip(co, 0, None)
    ti = au ** 2 * wi[..., 0:1] ** 2 + av ** 2 * wi[..., 1:2] ** 2
    to = au ** 2 * wo[..., 0:1] ** 2 + av ** 2 * wo[..., 1:2] ** 2
    den2 = 2 * (co_ * np.sqrt(ci_ ** 2 + ti) + ci_ * np.sqrt(co_ ** 2 + to))
    c = np.clip((wi * h).sum(-1, keepdims=True), 0, 1)
    F = f0 + (1 - f0) * (1 - c) ** 5
    return F * D / np.maximum(den2, 1e-30)


def er_eval(p, wi, wo):
    """eval(wi, wo) = f cos(theta_o) -> [..., 3]; wi, wo local unit vectors [..., 3]."""
    ci, co = wi[..., 2], wo[..., 2]
    valid = (ci > 1e-7) & (co > 1e-7) & (np.linalg.norm(wi + wo, axis=-1) > 1e-6)
    co_ = np.clip(co, 0, None)[..., None]
    O, I = [0, 1, 2], [3, 4, 5, 6]
    fo = lobe_f(p['au'][O], p['av'][O], p['f0'][O], wi[..., None, :].squeeze(-2) if False else wi, wo) if False else None
    # per-lobe evaluation (broadcast lobe axis last)
    def lobes(idx, a, b):
        return lobe_f(p['au'][idx], p['av'][idx], p['f0'][idx], a, b)       # [..., L]
    fo = lobes(O, wi, wo)
    outer = np.einsum('...l,cl->...c', fo * co_, p['lam'][:, 1:4])
    fi = lobes(I, refr(wi, p['eta']), refr(wo, p['eta'])) / p['eta'] ** 2
    inner = p['lam'][:, 0][None] * (co_ / np.pi) + np.einsum('...l,cl->...c', fi * co_, p['lam'][:, 4:8])
    out = outer + s_interp(p['s'], ci) * s_interp(p['s'], co) * inner
    return np.where(valid[..., None], out, 0.0)
