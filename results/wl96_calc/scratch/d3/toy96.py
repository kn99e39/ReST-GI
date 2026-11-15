"""WL96 D3 CPU toy (scratch, not evidence of the pipeline): CK2 S1 panel -> box A exactly-2-event delivery under one point light.
Exact deterministic quadrature over the panel vs carriers (binned patch x bin table, refined tables, patch-centre exact direction,
receiver-anchored X1 draws, portal/image-source lobe-proposal estimator). numpy only; ER eval ported from closure82 (ER arm)."""
import json, math, os, sys, time
import numpy as np
REPO = r'c:\Projects\ReST-GI'
sys.path.insert(0, os.path.join(REPO, 'experiments', 'ecro86'))
import scene86 as S
OUT = os.path.dirname(os.path.abspath(__file__))
sc = json.load(open(os.path.join(REPO, 'experiments', 'ecro_local93d', 'addendum_checker', 'scene_ck2.json')))
G = S.Geom(sc['geom'])
S1 = np.arange(1088, 1130); BA = np.arange(1130, 1184)
st = np.load(os.path.join(REPO, 'experiments', 'ecro86', 'data', 'mats86.npz'))['states'][1].astype(np.float64)
LAM = st[:24].reshape(3, 8); AU = st[24:31]; AV = st[31:38]; F0 = st[38:45]; SK = st[53:68].reshape(3, 5); ETA = st[68]
KN = np.array([0.0, .15, .4, .7, 1.0])
OUTER = [0, 1, 2]; INNER = [3, 4, 5, 6]


def s_interp(mu):
    m = np.clip(mu, 0, 1)
    return np.stack([np.interp(m, KN, SK[c]) for c in range(3)], -1)


def refract_up(w):
    tx, ty = w[..., 0] / ETA, w[..., 1] / ETA
    return np.stack([tx, ty, np.sqrt(np.clip(1 - tx * tx - ty * ty, 0, None))], -1)


def lobe_f(ix, wi, wo):
    au, av, f0 = AU[ix], AV[ix], F0[ix]
    ci, co = wi[..., 2:3], wo[..., 2:3]
    hs = wi + wo
    h = hs / np.maximum(np.linalg.norm(hs, axis=-1, keepdims=True), 1e-6)
    den = h[..., 0:1] ** 2 / au ** 2 + h[..., 1:2] ** 2 / av ** 2 + h[..., 2:3] ** 2
    D = 1.0 / (math.pi * au * av * den ** 2)
    ci_, co_ = np.clip(ci, 0, None), np.clip(co, 0, None)
    ti = au ** 2 * wi[..., 0:1] ** 2 + av ** 2 * wi[..., 1:2] ** 2
    to = au ** 2 * wo[..., 0:1] ** 2 + av ** 2 * wo[..., 1:2] ** 2
    den2 = 2 * (co_ * np.sqrt(ci_ ** 2 + ti) + ci_ * np.sqrt(co_ ** 2 + to))
    c = np.clip((wi * h).sum(-1, keepdims=True), 0, 1)
    F = f0 + (1 - f0) * (1 - c) ** 5
    return F * D / np.maximum(den2, 1e-30)


def er_eval(wi, wo):
    """eval = f cos(theta_o), local frames, [...,3] -> [...,3 channels]."""
    ci, co = wi[..., 2], wo[..., 2]
    valid = (ci > 1e-7) & (co > 1e-7) & (np.linalg.norm(wi + wo, axis=-1) > 1e-6)
    co_ = np.clip(co, 0, None)[..., None]
    fo = lobe_f(OUTER, wi, wo)
    outer = np.einsum('cr,...r->...c', LAM[:, 1:4], fo * co_)
    fi = lobe_f(INNER, refract_up(wi), refract_up(wo)) / ETA ** 2
    inner = LAM[:, 0] * (co_ / math.pi) + np.einsum('cr,...r->...c', LAM[:, 4:8], fi * co_)
    out = outer + s_interp(ci) * s_interp(co) * inner
    return np.where(valid[..., None], out, 0.0)


# panel frame
PT_, PB_, PN_ = G.PT[S1[0]], G.PB[S1[0]], G.PN[S1[0]]
PC = np.array(S.FX['panel']['center']); PH = S.FX['panel']['half']
ORG = PC - PT_ * PH[0] - PB_ * PH[1]           # panel corner; y = ORG + u*PT_ + v*PB_, u in [0, 1.2], v in [0, 1.4]
W_P, H_P = 2 * PH[0], 2 * PH[1]


def to_local(w):
    return np.stack([w @ PT_, w @ PB_, w @ PN_], -1)


def L1(y, v, light):
    """once-scattered radiance at panel points y [...,3] towards world directions v [...,3] (unit) -> [...,3]."""
    pos, I = light
    d = pos - y; d2 = (d * d).sum(-1); wl = d / np.sqrt(d2)[..., None]
    return er_eval(to_local(v), to_local(wl)) * I * (1.0 / d2)[..., None]


def bin_of_local(wl, nz, nphi):
    iz = np.clip(np.floor(wl[..., 2] * nz), 0, nz - 1).astype(np.int64)
    ip = np.clip(np.floor((np.arctan2(wl[..., 1], wl[..., 0]) + math.pi) * nphi / (2 * math.pi)), 0, nphi - 1).astype(np.int64)
    return iz * nphi + ip


def rcv_points(n_side):
    a = (np.arange(n_side) + 0.5) / n_side
    A, B = np.meshgrid(a, a, indexing='ij'); A, B = A.ravel(), B.ravel()
    P = G.PO[BA][:, None, :] + A[None, :, None] * G.PE1[BA][:, None, :] + B[None, :, None] * G.PE2[BA][:, None, :]
    return P, G.PN[BA]                                        # [54, n, 3], [54, 3]


def panel_grid(h):
    nu, nv = int(round(W_P / h)), int(round(H_P / h))
    u = (np.arange(nu) + 0.5) * W_P / nu; v = (np.arange(nv) + 0.5) * H_P / nv
    U, V = np.meshgrid(u, v, indexing='ij')
    Y = ORG + U.ravel()[:, None] * PT_ + V.ravel()[:, None] * PB_
    return Y, (W_P / nu) * (H_P / nv), U.ravel(), V.ravel()


def deliver(radfun, light, n_side=4, h=0.005, chunk=4):
    """patch-mean irradiance [54, 3] at box A from panel radiance radfun(y, v, light, U, V) by deterministic quadrature."""
    P, Nr = rcv_points(n_side)
    Y, dA, U, V = panel_grid(h)
    E = np.zeros((54, 3))
    for n in range(54):
        for j0 in range(0, P.shape[1], chunk):
            x = P[n, j0:j0 + chunk]                                   # [c,3]
            d = x[:, None, :] - Y[None]                               # y -> x
            r2 = (d * d).sum(-1); r = np.sqrt(r2); v = d / r[..., None]
            cy = v @ PN_; cx = -(v @ Nr[n])
            ok = (cy > 1e-7) & (cx > 1e-7)
            if not ok.any():
                continue
            g = np.where(ok, cy * cx / r2, 0.0) * dA
            Lr = radfun(np.broadcast_to(Y, v.shape), v, light, U, V)
            E[n] += (Lr * g[..., None]).sum((0, 1))
    return E / P.shape[1]


def rad_exact(Y, v, light, U, V):
    return L1(Y, v, light)


def rad_centre(Y, v, light, U, V):
    """X1c-like: exact direction, radiance evaluated at the centre of the 20 cm patch containing y."""
    iu = np.clip(np.floor(U / (W_P / 6)), 0, 5); iv = np.clip(np.floor(V / (H_P / 7)), 0, 6)
    C = ORG + ((iu + 0.5) * W_P / 6)[:, None] * PT_ + ((iv + 0.5) * H_P / 7)[:, None] * PB_
    return L1(np.broadcast_to(C, v.shape), v, light)


def make_table(light, sub, nz, nphi, ns=6, nd=6):
    """physical binned sender table: mean of L1 over (sub x sub cells of each 20 cm patch) x (nz x nphi equal-solid-angle bins)."""
    nu, nv = 6 * sub, 7 * sub
    a = (np.arange(ns) + 0.5) / ns
    cu = ((np.arange(nu)[:, None] + a[None]) * W_P / nu)               # [nu, ns]
    cv = ((np.arange(nv)[:, None] + a[None]) * H_P / nv)
    b = (np.arange(nd) + 0.5) / nd
    T = np.zeros((nu, nv, nz * nphi, 3))
    zs = ((np.arange(nz)[:, None] + b[None]) / nz)                     # [nz, nd]
    ps = ((np.arange(nphi)[:, None] + b[None]) / nphi) * 2 * math.pi - math.pi
    Zg = zs[:, None, :, None]; Pg = ps[None, :, None, :]
    st_ = np.sqrt(1 - Zg ** 2)
    wl = np.stack(np.broadcast_arrays(st_ * np.cos(Pg), st_ * np.sin(Pg), Zg), -1)    # [nz, nphi, nd, nd, 3] local
    ww = wl[..., 0:1] * PT_ + wl[..., 1:2] * PB_ + wl[..., 2:3] * PN_
    ww = ww.reshape(nz * nphi, nd * nd, 3)
    for i in range(nu):
        for j in range(nv):
            UU, VV = np.meshgrid(cu[i], cv[j], indexing='ij')
            Y = ORG + UU.ravel()[:, None] * PT_ + VV.ravel()[:, None] * PB_         # [ns^2, 3]
            Lr = L1(Y[None, None], ww[:, :, None, :], light)                    # [NB, nd^2, ns^2, 3]
            T[i, j] = Lr.mean((1, 2))
    return T


def rad_table(T, sub, nz, nphi):
    nu, nv = 6 * sub, 7 * sub

    def f(Y, v, light, U, V):
        iu = np.clip(np.floor(U / (W_P / nu)), 0, nu - 1).astype(np.int64)
        iv = np.clip(np.floor(V / (H_P / nv)), 0, nv - 1).astype(np.int64)
        b = bin_of_local(to_local(v), nz, nphi)
        return T[np.broadcast_to(iu, b.shape), np.broadcast_to(iv, b.shape), b]
    return f


def metrics(E, R):
    """flux ratio and abs rel L1 of per-patch exitance (sum over channels, box A Lambert albedo 0.5) against R."""
    e, r = 0.5 * E.sum(1), 0.5 * R.sum(1)
    return {'ratio': float(e.sum() / r.sum()), 'rel_l1': float(np.abs(e - r).sum() / r.sum()), 'corr': float(np.corrcoef(e, r)[0, 1])}


LIGHT0 = (np.array(sc['lights'][0]['pos'], float), np.array(sc['lights'][0]['rgb'], float))

if __name__ == '__main__':
    mode = sys.argv[1]
    res = {}
    if mode == 'validate':
        t0 = time.time()
        Ex = deliver(rad_exact, LIGHT0, 4, 0.005)
        np.save(os.path.join(OUT, 'E_exact_h5.npy'), Ex)
        z = np.load(os.path.join(REPO, 'results', 'ecro_attr95a', 'deliv.npz'))
        ref = 0.5 * (z['ref_a'] + z['ref_b'])[BA]
        e = 0.5 * Ex.sum(1)
        res['toy_vs_ref'] = {'ratio': float(e.sum() / ref.sum()), 'rel_l1': float(np.abs(e - ref).sum() / ref.sum()),
                             'corr': float(np.corrcoef(e, ref)[0, 1]), 'ref_half_noise': float(np.abs(z['ref_a'][BA] - z['ref_b'][BA]).sum() / 2 / ref.sum())}
        for k in ('X1', 'X1c', 'q256', 'A4'):
            p = np.asarray(z[f'draws|{k}']).mean(0)[BA]
            res[f'pooled_{k}_vs_toy'] = {'ratio': float(p.sum() / e.sum()), 'rel_l1': float(np.abs(p - e).sum() / e.sum())}
        res['t'] = time.time() - t0
    print(json.dumps(res, indent=1))
    json.dump(res, open(os.path.join(OUT, f'res_{mode}.json'), 'w'), indent=1)
