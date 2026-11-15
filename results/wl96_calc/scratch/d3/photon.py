"""Light-side lobe-sampled photon estimator (sender-anchored; G1 / ELCF family) for CK2 S1 -> box A exactly-2-event delivery."""
import json, math, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import toy96 as T
import toy96b as TB
G, BA = T.G, T.BA
cA, axA, hA = TB.cA, TB.axA, TB.hA
# box-A patch lookup: faces in Geom order (rect 6..11), 3x3 each
FACES = []
for f in range(6):
    p0 = BA[f * 9]
    FACES.append((G.PO[p0], G.PE1[p0] * 3, G.PE2[p0] * 3, G.PN[p0]))
AREA = G.PAREA[BA]


def hit_box(o, d):
    """first entry into box A: (t, patch index 0..53) or (-1, -1)."""
    ol = (o - cA) @ axA.T; dl = d @ axA.T
    with np.errstate(divide='ignore', invalid='ignore'):
        t1 = (-hA - ol) / dl; t2 = (hA - ol) / dl
    tn = np.where(np.abs(dl) < 1e-12, np.where(np.abs(ol) <= hA, -np.inf, np.inf), np.minimum(t1, t2))
    tf = np.where(np.abs(dl) < 1e-12, np.where(np.abs(ol) <= hA, np.inf, -np.inf), np.maximum(t1, t2))
    tmin = tn.max(-1); tmax = tf.min(-1)
    ok = (tmax >= tmin) & (tmin > 1e-6)
    x = o + tmin[:, None] * d
    pid = np.full(len(o), -1)
    for f, (po, e1, e2, nn) in enumerate(FACES):
        on = ok & (np.abs((x - po) @ nn) < 1e-6) & ((d @ nn) < 0)
        a = ((x - po) @ e1) / (e1 @ e1); b = ((x - po) @ e2) / (e2 @ e2)
        iu = np.clip(np.floor(a * 3), 0, 2).astype(int); iv = np.clip(np.floor(b * 3), 0, 2).astype(int)
        pid = np.where(on & (pid < 0), f * 9 + iu + 3 * iv, pid)
    return pid


def photons(light, M, seed=0, qmc=False, pi_c=0.05):
    pos, I = light
    rng = np.random.default_rng(seed)
    # panel-point pdf proportional to cos_y / d^2 on a 1 cm grid (piecewise constant) -> exact pdf
    Y, dA, Ug, Vg = T.panel_grid(0.01)
    wl = pos - Y; d2 = (wl * wl).sum(1); cz = (wl / np.sqrt(d2)[:, None]) @ T.PN_
    sh = TB.seg_hits_obb(Y, np.broadcast_to(pos, Y.shape))
    w = np.where((cz > 0) & ~sh, cz / d2, 0.0); W = w.sum(); cdf = np.cumsum(w) / W
    if qmc:
        i = np.arange(M)
        U = np.stack([(i + 0.5) / M] + [(i * g) % 1.0 for g in (0.7548776662466927, 0.5698402909980532, 0.6180339887498949, 0.4142135623730951)], -1)
        U = (U + rng.random(5)) % 1.0
    else:
        U = rng.random((M, 5))
    c = np.searchsorted(cdf, U[:, 0]); c = np.clip(c, 0, len(Y) - 1)
    nu = int(round(T.W_P / 0.01)); nv = int(round(T.H_P / 0.01))
    iu, iv = c // nv, c % nv
    y = T.ORG + ((iu + U[:, 1]) * T.W_P / nu)[:, None] * T.PT_ + ((iv + U[:, 2]) * T.H_P / nv)[:, None] * T.PB_
    pdfA = w[c] / W / dA                                                              # pdf over panel area (cell-constant)
    wly = pos - y; d2y = (wly * wly).sum(1); wly /= np.sqrt(d2y)[:, None]
    ok = ~TB.seg_hits_obb(y, np.broadcast_to(pos, y.shape)) & ((wly @ T.PN_) > 1e-7)
    wloc = T.to_local(wly)
    alphas = np.concatenate([T.AU[T.OUTER], T.AU[T.INNER] * T.ETA])
    wts = np.concatenate([T.LAM[:, 1:4].mean(0), T.LAM[:, 4:8].mean(0)]); wts = wts / wts.sum()
    cum = np.cumsum(np.concatenate([[pi_c], (1 - pi_c) * wts]))
    comp = np.searchsorted(cum, U[:, 3], side='right')
    lo = np.concatenate([[0], cum])[comp]; hi = cum[np.clip(comp, 0, len(cum) - 1)]
    ur = np.clip((U[:, 3] - lo) / np.maximum(hi - lo, 1e-12), 1e-9, 1 - 1e-9)
    al = alphas[np.clip(comp - 1, 0, len(alphas) - 1)]
    rr = al * np.sqrt(ur / (1 - ur)); ph = 2 * math.pi * U[:, 4]
    h = np.stack([-rr * np.cos(ph), -rr * np.sin(ph), np.ones_like(rr)], -1); h /= np.linalg.norm(h, axis=1, keepdims=True)
    vref = 2 * (wloc * h).sum(1, keepdims=True) * h - wloc
    # cosine component
    r_ = np.sqrt(ur); vcos = np.stack([r_ * np.cos(ph), r_ * np.sin(ph), np.sqrt(1 - ur)], -1)
    v = np.where((comp == 0)[:, None], vcos, vref)
    # mixture pdf of v
    hh = wloc + v; hh /= np.linalg.norm(hh, axis=1, keepdims=True)
    tan2 = (hh[:, 0] ** 2 + hh[:, 1] ** 2) / np.maximum(hh[:, 2] ** 2, 1e-30)
    ph_ = np.zeros(M)
    for a, w_ in zip(alphas, wts):
        ph_ += w_ / (math.pi * a * a * hh[:, 2] ** 4 * (1 + tan2 / (a * a)) ** 2) * hh[:, 2]
    pv = pi_c * np.clip(v[:, 2], 0, None) / math.pi + (1 - pi_c) * ph_ / (4 * np.maximum((wloc * hh).sum(1), 1e-12))
    ok &= v[:, 2] > 1e-7
    f = T.er_eval(v, wloc)                                       # f cos(theta_light)
    # flux = L1 cos_v dw dA = f cos_l * I/d2 * cos_v ; estimator weight
    wt = f * I[None, :] * (1.0 / d2y)[:, None] * v[:, 2:3] / (pv * pdfA)[:, None] / M
    vw = v[:, 0:1] * T.PT_ + v[:, 1:2] * T.PB_ + v[:, 2:3] * T.PN_
    pid = hit_box(y + vw * 1e-6, vw)
    good = ok & (pid >= 0)
    E = np.zeros((54, 3))
    for ch in range(3):
        E[:, ch] = np.bincount(pid[good], weights=wt[good, ch], minlength=54) / AREA
    return E


if __name__ == '__main__':
    li = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    LIGHTS = [T.LIGHT0, (np.array([-0.9, -0.5, 2.4]), np.array([30.0, 30.0, 30.0]))]
    light = LIGHTS[li]
    Ex = np.load(os.path.join(T.OUT, f'E_exact_L{li}.npy'))
    res = {}
    for M in (10000, 40000, 160000, 640000):
        t0 = time.time()
        Eq = photons(light, M, 1, True); tq = time.time() - t0
        mc = [T.metrics(photons(light, M, 100 + s), Ex) for s in range(16)]
        rl = [m['rel_l1'] for m in mc]
        res[f'M{M}'] = {'qmc': T.metrics(Eq, Ex), 'mc_rel_l1_median': float(np.median(rl)), 'mc_rel_l1_p90': float(np.quantile(rl, .9)),
                        'mc_rel_l1_worst': float(max(rl)), 'mc_ratio_range': [float(min(m['ratio'] for m in mc)), float(max(m['ratio'] for m in mc))], 't_qmc_s': tq}
        print(M, json.dumps(res[f'M{M}']), flush=True)
    json.dump(res, open(os.path.join(T.OUT, f'res_photon_L{li}.json'), 'w'), indent=1)
