"""WL96 D4 scratch 4: extended (indirect-like) incidence on the sharp panel. Emitter = box A's panel-facing face as a Lambertian emitter
with a bright +y column (radiance 1) and dim rest (0.05) (a proxy for the caustic-lit column of an e3_p path light -> A -> panel -> receiver).
Receivers = points on the same face. Compares incident-state carriers at the panel (per control cell, nearest cell) feeding ER emission:
  BINU : 64-bin solid-angle-integrated incident radiance Phi_m x bin-averaged ER response k_m(w) (exact iff L_in uniform within each bin)
  MATOM: Phi_m at the bin's radiance-weighted mean direction (point atom per bin)
  SUB4 : fixed 2 x 2 sub-directions per bin with non-negative weights (256-direction incident table, point-atom response; linear)
  SUB16: 4 x 4 sub-directions per bin
against the exact emitter integral (same panel / receiver quadrature). Simplified geometry, no occlusion. Scratch only."""
import json, os, sys, time
import numpy as np
H = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, H)
import ck2_model as M
import er_np as ER

he = 0.03
ue = -0.45 + (np.arange(30) + 0.5) * he
U, V = np.meshgrid(ue, ue); U = U.ravel(); V = V.ravel()
EP = M.FC + U[:, None] * M.FT + V[:, None] * M.FB                  # emitter points
LE = np.where(U > 0.15, 1.0, 0.05)                                  # +y column bright (iu = 2 <=> u in (0.15, 0.45])
dAe = he * he
xs, dA = M.panel_grid(0.02)
ys = M.recv_points(2).reshape(-1, 3)


def incident_dirs(c):
    """panel point(s) c [3] -> emitter directions (panel local) [E,3] and solid-angle-weighted radiance L dw [E]."""
    d = EP - c; r2 = (d ** 2).sum(1); w = d / np.sqrt(r2)[:, None]
    ce = -(w @ M.FN); cp = w[:, 0]
    ok = (ce > 0) & (cp > 0)
    return M.to_panel_local(w[ok]), (LE * ce / r2 * dAe)[ok]


def Lout_exact(x, wv):
    """x [3], wv [R,3] world outgoing dirs -> [R,3]."""
    wi, Ldw = incident_dirs(x)
    vo = M.to_panel_local(wv)
    f = ER.er_eval(M.P, np.repeat(vo, len(wi), 0), np.tile(wi, (len(vo), 1))).reshape(len(vo), len(wi), 3)
    return (f * Ldw[None, :, None]).sum(1)


def bin_of_local(wl):
    return M.pbin(wl)


def state(c, mode):
    wi, Ldw = incident_dirs(c)
    bn = bin_of_local(wi)
    if mode in ('BINU', 'MATOM'):
        Phi = np.bincount(bn, Ldw, 64)
        if mode == 'BINU':
            return Phi
        mu = np.stack([np.bincount(bn, Ldw * wi[:, k], 64) for k in range(3)], 1)
        nz = Phi > 0
        mu[nz] = mu[nz] / np.linalg.norm(mu[nz], axis=1, keepdims=True)
        return Phi[nz], mu[nz]
    s = 2 if mode == 'SUB4' else 4
    iz = np.clip(np.floor(wi[:, 2] * 4), 0, 3); ip = np.clip(np.floor((np.arctan2(wi[:, 1], wi[:, 0]) + np.pi) * 16 / (2 * np.pi)), 0, 15)
    fz = wi[:, 2] * 4 - iz; fp = (np.arctan2(wi[:, 1], wi[:, 0]) + np.pi) * 16 / (2 * np.pi) - ip
    sub = (bn * s + np.clip(np.floor(fz * s), 0, s - 1)) * s + np.clip(np.floor(fp * s), 0, s - 1)
    W = np.bincount(sub.astype(int), Ldw, 64 * s * s)
    k = np.arange(64 * s * s); b_ = k // (s * s); jz = (k // s) % s; jp = k % s
    dirs = M.bins_dir(None, b_, (jz + 0.5) / s, (jp + 0.5) / s)
    nz = W > 0
    return W[nz], dirs[nz]


# bin-averaged response kernel samples (4 x 4 per bin)
g4 = (np.arange(4) + 0.5) / 4; G2, G3 = np.meshgrid(g4, g4)
KB = np.array([M.bins_dir(None, np.full(16, m), G2.ravel(), G3.ravel()) for m in range(64)])     # [64,16,3]


def Lout_rep(xcell_state, mode, wv):
    vo = M.to_panel_local(wv)
    if mode == 'BINU':
        Phi = xcell_state
        f = ER.er_eval(M.P, np.repeat(vo, 64 * 16, 0), np.tile(KB.reshape(-1, 3), (len(vo), 1))).reshape(len(vo), 64, 16, 3).mean(2)
        return (f * Phi[None, :, None]).sum(1)
    W, dirs = xcell_state
    f = ER.er_eval(M.P, np.repeat(vo, len(dirs), 0), np.tile(dirs, (len(vo), 1))).reshape(len(vo), len(dirs), 3)
    return (f * W[None, :, None]).sum(1)


def deliver(Lfun):
    E = np.zeros((len(ys), 3))
    for i, x in enumerate(xs):
        d = ys - x; r2 = (d ** 2).sum(1); w = d / np.sqrt(r2)[:, None]
        cx = w[:, 0]; cy = -(w @ M.FN); ok = (cx > 0) & (cy > 0)
        if ok.any():
            E[ok] += Lfun(i, x, w[ok]) * (cx[ok] * cy[ok] / r2[ok] * dA)[:, None]
    return E


def cell_index(x, g):
    h = M.PS / g
    j = np.clip(((x[1] - M.PY[0]) / h).astype(int), 0, 6 * g - 1); k = np.clip(((x[2] - M.PZ[0]) / h).astype(int), 0, 7 * g - 1)
    return j, k, np.array([M.PX, M.PY[0] + (j + 0.5) * h, M.PZ[0] + (k + 0.5) * h])


if __name__ == '__main__':
    res = {}; t0 = time.time()
    Eex = deliver(lambda i, x, w: Lout_exact(x, w)); res['exact_s'] = time.time() - t0
    res['E_exact_lum'] = M.lum(Eex).tolist()
    for g in (1, 4):
        for mode in ('BINU', 'MATOM', 'SUB4', 'SUB16'):
            cache = {}
            def Lf(i, x, w, g=g, mode=mode):
                j, k, c = cell_index(x, g)
                if (j, k) not in cache:
                    cache[(j, k)] = state(c, mode)
                return Lout_rep(cache[(j, k)], mode, w)
            t1 = time.time(); E = deliver(Lf)
            m = M.metr(E, Eex); m['s'] = round(time.time() - t1, 1)
            # per receiver-point (not patch-summed) rel L1 too
            m['pointwise_rel_l1'] = float(np.abs(M.lum(E) - M.lum(Eex)).sum() / M.lum(Eex).sum())
            res[f'g{g}_{mode}'] = m
            print(g, mode, m, flush=True)
    # exact incidence but at cell centres only (spatial resolution of the incident state, no angular approximation)
    for g in (1, 4):
        cache = {}
        def Lf(i, x, w, g=g):
            j, k, c = cell_index(x, g)
            return Lout_exact(c, w)
        E = deliver(Lf); m = M.metr(E, Eex); m['pointwise_rel_l1'] = float(np.abs(M.lum(E) - M.lum(Eex)).sum() / M.lum(Eex).sum())
        res[f'g{g}_EXACT_AT_CELL_CENTRE'] = m; print(g, 'exact@cell', m, flush=True)
    json.dump(res, open(os.path.join(H, 'ext_incidence.json'), 'w'), indent=1)
