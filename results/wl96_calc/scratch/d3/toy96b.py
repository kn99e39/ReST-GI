"""WL96 D3 toy, part 2: carrier ladder, X1 draw variance, portal (image-source lobe-proposal) estimator, second light."""
import json, math, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import toy96 as T
G, S, BA = T.G, T.S, T.BA
OUT = T.OUT
NBASE = 64

# box A OBB for light -> panel shadowing
cA, axA, hA = G.obb('A', pad=0.0)


def seg_hits_obb(p0, p1):
    """p0 [...,3] -> p1 [...,3]: True if the open segment intersects box A."""
    d = p1 - p0
    o = (p0 - cA) @ axA.T; dd = d @ axA.T
    with np.errstate(divide='ignore', invalid='ignore'):
        t1 = (-hA - o) / dd; t2 = (hA - o) / dd
    tmin = np.where(np.abs(dd) < 1e-12, np.where(np.abs(o) <= hA, -np.inf, np.inf), np.minimum(t1, t2)).max(-1)
    tmax = np.where(np.abs(dd) < 1e-12, np.where(np.abs(o) <= hA, np.inf, -np.inf), np.maximum(t1, t2)).min(-1)
    return (tmax >= tmin) & (tmax > 1e-6) & (tmin < 1 - 1e-6)


def L1s(y, v, light):
    L = T.L1(y, v, light)
    pos, _ = light
    sh = seg_hits_obb(y, np.broadcast_to(pos, y.shape))
    return np.where(sh[..., None], 0.0, L)


def rad_exact(Y, v, light, U, V):
    return L1s(Y, v, light)


def rad_centre(Y, v, light, U, V):
    iu = np.clip(np.floor(U / (T.W_P / 6)), 0, 5); iv = np.clip(np.floor(V / (T.H_P / 7)), 0, 6)
    C = T.ORG + ((iu + 0.5) * T.W_P / 6)[:, None] * T.PT_ + ((iv + 0.5) * T.H_P / 7)[:, None] * T.PB_
    return L1s(np.broadcast_to(C, v.shape), v, light)


def deliver_table(light, sub, nz, nphi, n_side=4, h=0.01, ns=None, nd=None):
    """deterministic flight (infinite relation rays) from a physical piecewise-constant table over (sub x sub cells per 20 cm patch) x
    (nz x nphi equal-solid-angle outgoing bins); entries computed lazily for the keys box A reads."""
    nu, nv, NB = 6 * sub, 7 * sub, nz * nphi
    ns = ns or max(2, int(math.ceil(8 / sub))); nd = nd or max(3, int(math.ceil(24 / nz)))
    P, Nr = T.rcv_points(n_side)
    Y, dA, U, V = T.panel_grid(h)
    iu = np.clip(np.floor(U / (T.W_P / nu)), 0, nu - 1).astype(np.int64)
    iv = np.clip(np.floor(V / (T.H_P / nv)), 0, nv - 1).astype(np.int64)
    cid = iu * nv + iv
    per = []
    for n in range(54):
        x = P[n]
        d = x[:, None, :] - Y[None]
        r2 = (d * d).sum(-1); r = np.sqrt(r2); v = d / r[..., None]
        cy = v @ T.PN_; cx = -(v @ Nr[n])
        ok = (cy > 1e-7) & (cx > 1e-7)
        g = (cy * cx / r2 * dA)[ok]
        b = T.bin_of_local(T.to_local(v[ok]), nz, nphi)
        key = np.broadcast_to(cid, ok.shape)[ok] * NB + b
        uk, inv = np.unique(key, return_inverse=True)
        per.append((uk, np.bincount(inv, weights=g, minlength=len(uk)) / x.shape[0]))
    keys = np.unique(np.concatenate([p[0] for p in per]))
    # table entries
    a = (np.arange(ns) + 0.5) / ns; bb = (np.arange(nd) + 0.5) / nd
    A_, B_ = [m.ravel() for m in np.meshgrid(a, a, indexing='ij')]
    Zb, Pb = [m.ravel() for m in np.meshgrid(bb, bb, indexing='ij')]
    Tab = np.zeros((len(keys), 3))
    for k0 in range(0, len(keys), 2000):
        kk = keys[k0:k0 + 2000]
        c, bn = kk // NB, kk % NB
        ci, cj = c // nv, c % nv
        iz, ip = bn // nphi, bn % nphi
        uu = (ci[:, None] + A_[None]) * T.W_P / nu; vv = (cj[:, None] + B_[None]) * T.H_P / nv           # [k, ns2]
        Yk = T.ORG + uu[..., None] * T.PT_ + vv[..., None] * T.PB_                                       # [k, ns2, 3]
        z = (iz[:, None] + Zb[None]) / nz; ph = (ip[:, None] + Pb[None]) / nphi * 2 * math.pi - math.pi  # [k, nd2]
        s_ = np.sqrt(1 - z * z)
        wl = np.stack([s_ * np.cos(ph), s_ * np.sin(ph), z], -1)
        ww = wl[..., 0:1] * T.PT_ + wl[..., 1:2] * T.PB_ + wl[..., 2:3] * T.PN_                          # [k, nd2, 3]
        Lr = L1s(np.broadcast_to(Yk[:, None], (len(kk), ww.shape[1], Yk.shape[1], 3)), np.broadcast_to(ww[:, :, None], (len(kk), ww.shape[1], Yk.shape[1], 3)), light)
        Tab[k0:k0 + 2000] = Lr.mean((1, 2))
    E = np.zeros((54, 3))
    for n, (uk, w) in enumerate(per):
        E[n] = (Tab[np.searchsorted(keys, uk)] * w[:, None]).sum(0)
    return E, {'keys_read': int(len(keys)), 'cells': nu * nv, 'bins': NB, 'samples_per_entry': ns * ns * nd * nd}


def x1_draws(light, D=64, kappa=8, seed=96100, centre=False):
    """receiver-anchored X1 (exact direction; hit point or patch centre), ECRO relation layout: per box-A patch 64 bins x kappa rays."""
    K = 64 * kappa
    out = np.zeros((D, 54, 3)); hits = np.zeros((D, 54))
    PO, E1, E2 = G.PO[BA], G.PE1[BA], G.PE2[BA]
    Tt, Bt, Nt = G.PT[BA], G.PB[BA], G.PN[BA]
    k = np.arange(K); bn = k // kappa; iz = bn // 16; ip = bn % 16
    for d in range(D):
        u = np.random.default_rng(seed + d).random((4, K))                 # shared by all patches, as pt86
        z = (iz + u[2]) / 4; ph = (ip + u[3]) / 16 * 2 * math.pi - math.pi
        s_ = np.sqrt(1 - z * z)
        wl = np.stack([s_ * np.cos(ph), s_ * np.sin(ph), z], -1)           # [K,3]
        o = PO[:, None] + u[0][None, :, None] * E1[:, None] + u[1][None, :, None] * E2[:, None]   # [54,K,3]
        w = wl[None, :, 0:1] * Tt[:, None] + wl[None, :, 1:2] * Bt[:, None] + wl[None, :, 2:3] * Nt[:, None]
        den = w @ T.PN_
        t = ((T.ORG - o) @ T.PN_) / np.where(np.abs(den) > 1e-12, den, 1e-12)
        y = o + t[..., None] * w
        q = y - T.ORG; uu = q @ T.PT_; vv = q @ T.PB_
        ok = (den < -1e-9) & (t > 0) & (uu >= 0) & (uu <= T.W_P) & (vv >= 0) & (vv <= T.H_P)
        if centre:
            iu_ = np.clip(np.floor(uu / (T.W_P / 6)), 0, 5); iv_ = np.clip(np.floor(vv / (T.H_P / 7)), 0, 6)
            y = T.ORG + ((iu_ + 0.5) * T.W_P / 6)[..., None] * T.PT_ + ((iv_ + 0.5) * T.H_P / 7)[..., None] * T.PB_
        L = np.where(ok[..., None], L1s(y, -w, light), 0.0)
        rw = z * (2 * math.pi / 64) / kappa
        out[d] = (L * rw[None, :, None]).sum(1)
        hits[d] = ok.sum(1)
    return out, hits


def portal(light, Kf, qmc=True, seed=0, n_side=4, pi_u=0.1):
    """image-source lobe-proposal estimator (receiver point MIS around the mirror image of the light; G1/LCFG family).
    Mixture over the S1 lobes in slope space (GGX = bivariate t, nu = 2) + defensive uniform panel-area component."""
    pos, I = light
    xl = pos - 2 * ((pos - T.ORG) @ T.PN_) * T.PN_                       # mirror image (diagnostic only)
    alphas = np.concatenate([T.AU[T.OUTER], T.AU[T.INNER] * T.ETA])
    wts = np.concatenate([T.LAM[:, 1:4].mean(0), T.LAM[:, 4:8].mean(0)]); wts = wts / wts.sum()
    P, Nr = T.rcv_points(n_side)
    npts = P.shape[1]
    if qmc:
        # rank-1 lattice / Fibonacci-like 3D points with a fixed shift (deterministic)
        i = np.arange(Kf)
        U = np.stack([(i + 0.5) / Kf, (i * 0.6180339887498949) % 1.0, (i * 0.7548776662466927) % 1.0], -1)
        U = (U + np.random.default_rng(seed).random(3)) % 1.0
    else:
        U = np.random.default_rng(seed).random((Kf, 3))
    E = np.zeros((54, 3))
    Apan = T.W_P * T.H_P
    cum = np.cumsum(np.concatenate([[pi_u], (1 - pi_u) * wts]))
    for n in range(54):
        x = P[n]                                                          # [npts,3]
        # direction from the specular point: fixed incident direction omega* = from the specular point y* to the light
        dxl = xl - x; tt = ((T.ORG - x) @ T.PN_) / (dxl @ T.PN_)
        ys = x + tt[:, None] * dxl
        wst = pos - ys; wst /= np.linalg.norm(wst, axis=1, keepdims=True)                 # omega* [npts,3]
        comp = np.searchsorted(cum, U[:, 0], side='right')                                 # 0 = uniform, 1.. lobes
        ur = np.where(comp == 0, U[:, 0] / pi_u, (U[:, 0] - cum[np.clip(comp - 1, 0, None)]) / np.diff(cum)[np.clip(comp - 1, 0, None)])
        ur = np.clip(ur, 1e-9, 1 - 1e-9)
        al = alphas[np.clip(comp - 1, 0, len(alphas) - 1)]
        # GGX slope sample (isotropic): r = al * sqrt(u/(1-u)), phi = 2 pi u2
        rr = al * np.sqrt(ur / (1 - ur)); phi = 2 * math.pi * U[:, 1]
        hloc = np.stack([-rr * np.cos(phi), -rr * np.sin(phi), np.ones_like(rr)], -1); hloc /= np.linalg.norm(hloc, axis=1, keepdims=True)
        hw = hloc[:, 0:1] * T.PT_ + hloc[:, 1:2] * T.PB_ + hloc[:, 2:3] * T.PN_           # [Kf,3]
        # outgoing v = reflect(omega*, h) per point; panel point y = x - s v on the plane
        vd = 2 * (wst[:, None, :] * hw[None]).sum(-1, keepdims=True) * hw[None] - wst[:, None, :]   # [npts,Kf,3]
        # uniform component: panel point directly
        uu = U[:, 1] * T.W_P; vv = U[:, 2] * T.H_P
        yu = T.ORG + uu[:, None] * T.PT_ + vv[:, None] * T.PB_
        den = vd @ T.PN_
        s = ((x[:, None, :] - T.ORG) @ T.PN_) / np.where(den > 1e-12, den, 1e-12)
        yl = x[:, None, :] - s[..., None] * vd
        y = np.where((comp == 0)[None, :, None], yu[None], yl)
        dv = x[:, None, :] - y; r2 = (dv * dv).sum(-1); r = np.sqrt(r2); v = dv / r[..., None]
        cy = v @ T.PN_; cx = -(v @ Nr[n])
        q = y - T.ORG; qu = q @ T.PT_; qv = q @ T.PB_
        inside = (qu >= 0) & (qu <= T.W_P) & (qv >= 0) & (qv <= T.H_P) & (cy > 1e-7) & (cx > 1e-7)
        inside &= np.where((comp == 0)[None], True, den > 1e-12)
        # mixture pdf over panel area at y
        hh = wst[:, None, :] + v; hh /= np.linalg.norm(hh, axis=-1, keepdims=True)
        hz = hh @ T.PN_; ht = hh @ T.PT_; hb = hh @ T.PB_
        tan2 = (ht ** 2 + hb ** 2) / np.maximum(hz ** 2, 1e-30)
        pdf_h = np.zeros_like(hz)
        for a, w_ in zip(alphas, wts):
            D = 1.0 / (math.pi * a * a * hz ** 4 * (1 + tan2 / (a * a)) ** 2)
            pdf_h += w_ * D * hz
        vh = np.abs((wst[:, None, :] * hh).sum(-1))
        pA = pi_u / Apan + (1 - pi_u) * pdf_h / (4 * np.maximum(vh, 1e-12)) * np.maximum(cy, 0) / r2
        Lr = L1s(y, v, light)
        f = np.where(inside[..., None], Lr * (cx * cy / r2)[..., None] / np.maximum(pA, 1e-30)[..., None], 0.0)
        E[n] = f.mean(1).mean(0)
    return E


if __name__ == '__main__':
    mode = sys.argv[1]
    li = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    LIGHTS = [T.LIGHT0, (np.array([-0.9, -0.5, 2.4]), np.array([30.0, 30.0, 30.0]))]
    light = LIGHTS[li]
    res = {}
    t0 = time.time()
    fx = os.path.join(OUT, f'E_exact_L{li}.npy')
    if mode == 'exact':
        Ex = T.deliver(rad_exact, light, 4, 0.005)
        np.save(fx, Ex)
        Ex2 = T.deliver(rad_exact, light, 4, 0.0025) if li == 0 else None
        if Ex2 is not None:
            res['h5_vs_h2.5'] = T.metrics(Ex, Ex2)
        Ec = T.deliver(rad_centre, light, 4, 0.005)
        res['X1c_like_det'] = T.metrics(Ec, Ex)
        # incidence ring shares at the panel (16 points per patch, unshadowed by panel)
        Y, dA, U, V = T.panel_grid(0.02)
        wl = light[0] - Y; d2 = (wl * wl).sum(1); wl /= np.sqrt(d2)[:, None]
        z = wl @ T.PN_; sh = seg_hits_obb(Y, np.broadcast_to(light[0], Y.shape))
        e = np.where((z > 0) & ~sh, z / d2, 0)
        ring = np.clip(np.floor(z * 4), 0, 3).astype(int)
        res['incident_ring_share'] = [float(e[ring == i].sum() / e.sum()) for i in range(4)]
        res['flux_exitance'] = float(0.5 * Ex.sum())
    elif mode == 'ladder':
        Ex = np.load(fx)
        res['configs'] = []
        cfgs = [(1, 4, 16), (2, 4, 16), (4, 4, 16), (8, 4, 16), (1, 16, 64), (1, 32, 128), (2, 16, 64), (4, 16, 64), (4, 32, 128), (8, 32, 128)]
        if len(sys.argv) > 3:
            cfgs = [cfgs[int(c)] for c in sys.argv[3].split(',')]
        for sub, nz, nphi in cfgs:
            t1 = time.time()
            E, info = deliver_table(light, sub, nz, nphi)
            m = T.metrics(E, Ex); m.update(info); m.update({'sub': sub, 'nz': nz, 'nphi': nphi, 't_s': time.time() - t1})
            res['configs'].append(m); print(json.dumps(m), flush=True)
    elif mode == 'x1':
        Ex = np.load(fx)
        for kappa in (8, 64):
            for cen in (False, True):
                Ed, hits = x1_draws(light, 64, kappa, centre=cen)
                per = [T.metrics(Ed[d], Ex) for d in range(64)]
                rl = np.array([p['rel_l1'] for p in per]); ra = np.array([p['ratio'] for p in per])
                res[f'kappa{kappa}_{"centre" if cen else "point"}'] = {
                    'pooled': T.metrics(Ed.mean(0), Ex), 'per_draw_rel_l1_median': float(np.median(rl)), 'p90': float(np.quantile(rl, .9)),
                    'worst': float(rl.max()), 'per_draw_ratio_p10_p90': [float(np.quantile(ra, .1)), float(np.quantile(ra, .9))],
                    'mean_panel_hits_per_patch': float(hits.mean())}
    elif mode == 'portal':
        Ex = np.load(fx)
        for Kf in (4, 16, 64, 256):
            Eq = portal(light, Kf, qmc=True, seed=1)
            mc = [T.metrics(portal(light, Kf, qmc=False, seed=s), Ex) for s in range(16)]
            res[f'Kf{Kf}'] = {'qmc': T.metrics(Eq, Ex), 'mc_rel_l1_median': float(np.median([m['rel_l1'] for m in mc])),
                              'mc_rel_l1_worst': float(max(m['rel_l1'] for m in mc)), 'mc_ratio_range': [float(min(m['ratio'] for m in mc)), float(max(m['ratio'] for m in mc))]}
            print(Kf, json.dumps(res[f'Kf{Kf}']), flush=True)
    res['t'] = time.time() - t0
    print(json.dumps(res, indent=1))
    json.dump(res, open(os.path.join(OUT, f'res_{mode}_L{li}' + (('_' + sys.argv[3].replace(',', '-')) if len(sys.argv) > 3 else '') + '.json'), 'w'), indent=1)
