"""Portal estimator v2: stationary-point (image-source) linearised slope proposal in panel coordinates (G1 / LCFG family)."""
import json, math, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import toy96 as T
import toy96b as TB
G, BA = T.G, T.BA


def slope(y, x, pos):
    wl = pos - y; wl /= np.linalg.norm(wl, axis=-1, keepdims=True)
    wx = x - y; wx /= np.linalg.norm(wx, axis=-1, keepdims=True)
    h = wl + wx; h /= np.linalg.norm(h, axis=-1, keepdims=True)
    hz = h @ T.PN_
    return np.stack([(h @ T.PT_) / hz, (h @ T.PB_) / hz], -1)


def portal2(light, Kf, qmc=True, seed=0, n_side=4, pi_u=0.05):
    pos, I = light
    xl = pos - 2 * ((pos - T.ORG) @ T.PN_) * T.PN_
    alphas = np.concatenate([T.AU[T.OUTER], T.AU[T.INNER] * T.ETA])
    wts = np.concatenate([T.LAM[:, 1:4].mean(0), T.LAM[:, 4:8].mean(0)]); wts = wts / wts.sum()
    P, Nr = T.rcv_points(n_side)
    if qmc:
        i = np.arange(Kf)
        U = np.stack([(i + 0.5) / Kf, (i * 0.6180339887498949) % 1.0, (i * 0.7548776662466927) % 1.0], -1)
        U = (U + np.random.default_rng(seed).random(3)) % 1.0
    else:
        U = np.random.default_rng(seed).random((Kf, 3))
    cum = np.cumsum(np.concatenate([[pi_u], (1 - pi_u) * wts]))
    comp = np.searchsorted(cum, U[:, 0], side='right')
    lo = np.concatenate([[0], cum])[comp]; hi = cum[np.clip(comp, 0, len(cum) - 1)]
    ur = np.clip((U[:, 0] - lo) / np.maximum(hi - lo, 1e-12), 1e-9, 1 - 1e-9)
    al = alphas[np.clip(comp - 1, 0, len(alphas) - 1)]
    rr = al * np.sqrt(ur / (1 - ur)); ph = 2 * math.pi * U[:, 1]
    Z = np.stack([rr * np.cos(ph), rr * np.sin(ph)], -1)                       # slope samples [Kf,2]
    Apan = T.W_P * T.H_P
    E = np.zeros((54, 3))
    eps = 1e-3
    for n in range(54):
        x = P[n]
        dxl = xl - x; tt = ((T.ORG - x) @ T.PN_) / (dxl @ T.PN_)
        ys = x + tt[:, None] * dxl                                               # specular (image) point, may lie off the panel
        q = ys - T.ORG; us, vs = q @ T.PT_, q @ T.PB_
        s0 = slope(ys, x, pos)
        su = (slope(ys + eps * T.PT_, x, pos) - slope(ys - eps * T.PT_, x, pos)) / (2 * eps)
        sv = (slope(ys + eps * T.PB_, x, pos) - slope(ys - eps * T.PB_, x, pos)) / (2 * eps)
        J = np.stack([su, sv], -1)                                               # [npts, 2(slope), 2(uv)]
        Ji = np.linalg.inv(J); dJ = np.abs(np.linalg.det(J))
        duv = np.einsum('pij,kj->pki', Ji, Z)              # slope offset relative to the specular slope (0)
        uu = us[:, None] + duv[..., 0]; vv = vs[:, None] + duv[..., 1]
        uu = np.where((comp == 0)[None], U[None, :, 1] * T.W_P, uu); vv = np.where((comp == 0)[None], U[None, :, 2] * T.H_P, vv)
        y = T.ORG + uu[..., None] * T.PT_ + vv[..., None] * T.PB_
        # pdf under the (linear) proposal
        dl = np.stack([uu - us[:, None], vv - vs[:, None]], -1)
        sl = np.einsum('pij,pkj->pki', J, dl)
        r2s = (sl ** 2).sum(-1)
        ps = np.zeros_like(r2s)
        for a, w_ in zip(alphas, wts):
            ps += w_ / (math.pi * a * a * (1 + r2s / (a * a)) ** 2)
        pA = pi_u / Apan + (1 - pi_u) * ps * dJ[:, None]
        dv = x[:, None, :] - y; r2 = (dv * dv).sum(-1); v = dv / np.sqrt(r2)[..., None]
        cy = v @ T.PN_; cx = -(v @ Nr[n])
        inside = (uu >= 0) & (uu <= T.W_P) & (vv >= 0) & (vv <= T.H_P) & (cy > 1e-7) & (cx > 1e-7)
        Lr = TB.L1s(y, v, light)
        f = np.where(inside[..., None], Lr * (cx * cy / r2)[..., None] / pA[..., None], 0.0)
        E[n] = f.mean(1).mean(0)
    return E


if __name__ == '__main__':
    li = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    LIGHTS = [T.LIGHT0, (np.array([-0.9, -0.5, 2.4]), np.array([30.0, 30.0, 30.0]))]
    light = LIGHTS[li]
    Ex = np.load(os.path.join(T.OUT, f'E_exact_L{li}.npy'))
    res = {}
    for Kf in (4, 16, 64, 256):
        t0 = time.time()
        Eq = portal2(light, Kf, True, 1)
        tq = time.time() - t0
        mc = [T.metrics(portal2(light, Kf, False, s), Ex) for s in range(16)]
        res[f'Kf{Kf}'] = {'qmc': T.metrics(Eq, Ex), 'mc_rel_l1_median': float(np.median([m['rel_l1'] for m in mc])),
                          'mc_rel_l1_p90': float(np.quantile([m['rel_l1'] for m in mc], .9)),
                          'mc_rel_l1_worst': float(max(m['rel_l1'] for m in mc)), 'mc_ratio_range': [float(min(m['ratio'] for m in mc)), float(max(m['ratio'] for m in mc))],
                          'pooled16_mc': T.metrics(np.mean([portal2(light, Kf, False, s) for s in range(16)], 0), Ex), 't_qmc_s': tq}
        print(Kf, json.dumps(res[f'Kf{Kf}']), flush=True)
    json.dump(res, open(os.path.join(T.OUT, f'res_portal2_L{li}.json'), 'w'), indent=1)
