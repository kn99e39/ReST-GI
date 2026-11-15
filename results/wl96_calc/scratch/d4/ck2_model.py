"""WL96 D4 scratch: simplified CPU model of CK2 S1 -> box A (-x' face) exactly-2-event delivery, to (i) check that a model reproduces the
WL95A ladder (binned q over-delivery, X1c, X1 variance) and (ii) evaluate the candidate carriers' representability / variance.
Simplifications [INF]: only the 9 patches of box A's panel-facing face; no occlusion between light, panel and that face; omni point light;
panel frame t=+y, b=+z (S1 is near-isotropic: au ~ av). Not a measurement of the CK2 renderer."""
import json, os, sys, time
import numpy as np
H = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, H)
import er_np as ER

REPO = 'c:/Projects/ReST-GI'
st = np.load(os.path.join(REPO, 'experiments/ecro86/data/mats86.npz'))['states'][1]
P = ER.unpack(st)
LIGHT = np.array([-1.45, -0.6, 1.7]); I = 30.0
if len(sys.argv) > 1 and sys.argv[1] == 'alt':          # independent non-grazing validation light (design probe only)
    LIGHT = np.array([-0.7, -0.2, 2.4])
TAG = sys.argv[1] if len(sys.argv) > 1 else 'ck2'
# panel: x = -1.5, y in [-0.3, 0.9], z in [0.3, 1.7], normal +x; 6 x 7 patches of 0.2 m; local frame t=+y, b=+z, n=+x
PX = -1.5; PY = (-0.3, 0.9); PZ = (0.3, 1.7); PS = 0.2
# box A -x' face
yaw = 0.15; cA = np.array([-1.08, 0.6, 1.0]); hA = np.array([0.2, 0.45, 0.45])
ex = np.array([np.cos(yaw), np.sin(yaw), 0.0]); ey = np.array([-np.sin(yaw), np.cos(yaw), 0.0]); ez = np.array([0, 0, 1.0])
FC = cA - hA[0] * ex; FN = -ex; FT = ey; FB = ez                       # face centre, normal, in-plane axes (half 0.45 each)
RS = 0.3                                                             # 3 x 3 receiver patches of 0.3 m


def to_panel_local(w):            # world -> panel local (t=+y, b=+z, n=+x)
    return np.stack([w[..., 1], w[..., 2], w[..., 0]], -1)


def L1(x, wout):
    """exact once-scattered S1 radiance leaving panel point(s) x [...,3] in world direction wout [...,3] (unit)."""
    dl = LIGHT - x; d2 = (dl ** 2).sum(-1); wl = dl / np.sqrt(d2)[..., None]
    f = ER.er_eval(P, to_panel_local(wout), to_panel_local(wl))       # eval(wi=viewer, wo=light) incl. cos(light)
    return f * (I / d2)[..., None]


def lum(c):
    return c.sum(-1)


def recv_points(n):
    """n x n receiver points per face patch -> [9, n*n, 3] (patch index iu + 3 iv)."""
    out = []
    for iv in range(3):
        for iu in range(3):
            u = -0.45 + (iu + (np.arange(n) + 0.5) / n) * RS; v = -0.45 + (iv + (np.arange(n) + 0.5) / n) * RS
            U, V = np.meshgrid(u, v)
            out.append(FC + U.ravel()[:, None] * FT + V.ravel()[:, None] * FB)
    return np.array(out)


def panel_grid(h):
    y = np.arange(PY[0] + h / 2, PY[1], h); z = np.arange(PZ[0] + h / 2, PZ[1], h)
    Y, Z = np.meshgrid(y, z)
    return np.stack([np.full(Y.size, PX), Y.ravel(), Z.ravel()], 1), h * h


def E_dense(ys, h):
    """deterministic area quadrature of S1 -> point irradiance at receiver points ys [M,3] (rgb)."""
    xs, dA = panel_grid(h)
    out = np.zeros((len(ys), 3))
    for i, y in enumerate(ys):
        d = y - xs; r2 = (d ** 2).sum(1); r = np.sqrt(r2); w = d / r[:, None]          # panel -> receiver
        cx = w[:, 0]; cy = -(w @ FN)
        ok = (cx > 0) & (cy > 0)
        Lv = L1(xs[ok], w[ok])
        out[i] = (Lv * (cx[ok] * cy[ok] / r2[ok] * dA)[:, None]).sum(0)
    return out


def bins_dir(n, bn, u2, u3):
    iz, ip = bn // 16, bn % 16
    z = (iz + u2) / 4; phi = (ip + u3) * (2 * np.pi / 16) - np.pi
    rr = np.sqrt(np.clip(1 - z * z, 0, None))
    return np.stack([rr * np.cos(phi), rr * np.sin(phi), z], -1)


def hit_panel(o, w):
    """ray o + t w against the panel rectangle (front side). returns x [.,3], ok."""
    t = (PX - o[:, 0]) / np.where(np.abs(w[:, 0]) > 1e-12, w[:, 0], -1e-12)
    x = o + t[:, None] * w
    ok = (w[:, 0] < -1e-9) & (t > 0) & (x[:, 1] >= PY[0]) & (x[:, 1] <= PY[1]) & (x[:, 2] >= PZ[0]) & (x[:, 2] <= PZ[1])
    return x, ok


def patch_of(x):
    iy = np.clip(((x[:, 1] - PY[0]) / PS).astype(int), 0, 5); iz = np.clip(((x[:, 2] - PZ[0]) / PS).astype(int), 0, 6)
    return iy + 6 * iz


def pbin(wl):     # local dir -> bin (iz*16 + iphi), as pt86.bin_of
    iz = np.clip(np.floor(wl[:, 2] * 4), 0, 3).astype(int)
    ip = np.clip(np.floor((np.arctan2(wl[:, 1], wl[:, 0]) + np.pi) * 16 / (2 * np.pi)), 0, 15).astype(int)
    return iz * 16 + ip


def panel_centre(pid):
    iy, iz = pid % 6, pid // 6
    return np.stack([np.full(len(pid), PX), PY[0] + (iy + 0.5) * PS, PZ[0] + (iz + 0.5) * PS], 1)


def q_table(npt=8, ndir=12):
    """binned once-scattered table q[42, 64, 3]: mean over the patch area (npt x npt) and the bin's solid angle (ndir x ndir, z / phi uniform)."""
    q = np.zeros((42, 64, 3))
    g = (np.arange(npt) + 0.5) / npt; gd = (np.arange(ndir) + 0.5) / ndir
    U2, U3 = np.meshgrid(gd, gd); U2 = U2.ravel(); U3 = U3.ravel()
    for pid in range(42):
        iy, iz = pid % 6, pid // 6
        Y, Z = np.meshgrid(PY[0] + (iy + g) * PS, PZ[0] + (iz + g) * PS)
        xs = np.stack([np.full(Y.size, PX), Y.ravel(), Z.ravel()], 1)
        for bn in range(64):
            wl = bins_dir(None, np.full(U2.size, bn), U2, U3)                  # panel local
            ww = np.stack([wl[:, 2], wl[:, 0], wl[:, 1]], 1)                    # local (t=y, b=z, n=x) -> world
            Lv = L1(np.repeat(xs, len(ww), 0), np.tile(ww, (len(xs), 1)))
            q[pid, bn] = Lv.mean(0)
    return q


def recv_frame():
    return FT, np.cross(FN, FT), FN       # t, b, n with t x b = n


def draw(rng, kappa=8, strat=False):
    """one ECRO-like relation draw for the 9 face patches: 64 bins x kappa rays; origin uniform in the patch; direction uniform in the bin
    (strat: a 2 x 4 sub-stratification of the bin). returns per patch: origins, world dirs, rw, sub-cell bounds."""
    t, b, n = recv_frame()
    K = 64 * kappa
    out = []
    for pp in range(9):
        iu, iv = pp % 3, pp // 3
        a = rng.random(K); c = rng.random(K)
        o = FC + (-0.45 + (iu + a) * RS)[:, None] * FT + (-0.45 + (iv + c) * RS)[:, None] * FB
        bn = np.repeat(np.arange(64), kappa)
        if strat:
            k = np.tile(np.arange(kappa), 64); sz, sp = k // 4, k % 4
            lo2, lo3 = sz / 2, sp / 4; w2, w3 = 0.5, 0.25
            u2 = lo2 + rng.random(K) * w2; u3 = lo3 + rng.random(K) * w3
        else:
            lo2 = np.zeros(K); lo3 = np.zeros(K); w2 = 1.0; w3 = 1.0
            u2 = rng.random(K); u3 = rng.random(K)
        wl = bins_dir(None, bn, u2, u3)
        w = wl[:, 0:1] * t + wl[:, 1:2] * b + wl[:, 2:3] * n
        rw = wl[:, 2] * (2 * np.pi / 64) / kappa
        out.append(dict(o=o + n * 1e-6, w=w, rw=rw, bn=bn, lo2=lo2, lo3=lo3, w2=w2, w3=w3))
    return out


def estimate(dr_, mode, q=None, grid=None, foot=0):
    """patch irradiance [9, 3] from one draw. mode: 'q' binned table, 'X1c', 'X1', 'CP' (control-point incident state, grid g),
    'FOOT' (X1 integrated over the ray's sub-cell with foot x foot sub-directions; plane intersection only, no extra visibility)."""
    t, b, n = recv_frame()
    E = np.zeros((9, 3))
    for pp, d in enumerate(dr_):
        if mode == 'FOOT':
            s = foot; g = (np.arange(s) + 0.5) / s
            G2, G3 = np.meshgrid(g, g); G2 = G2.ravel(); G3 = G3.ravel()
            K = len(d['rw'])
            u2 = (d['lo2'][:, None] + G2[None] * d['w2']).ravel(); u3 = (d['lo3'][:, None] + G3[None] * d['w3']).ravel()
            bn = np.repeat(d['bn'], s * s)
            wl = bins_dir(None, bn, u2, u3)
            w = wl[:, 0:1] * t + wl[:, 1:2] * b + wl[:, 2:3] * n
            o = np.repeat(d['o'], s * s, 0)
            x, ok = hit_panel(o, w)
            Lv = np.zeros((len(w), 3)); Lv[ok] = L1(x[ok], -w[ok])
            rw = wl[:, 2] * (2 * np.pi / 64) / (K / 64) / (s * s)
            E[pp] = (Lv * rw[:, None]).sum(0)
            continue
        x, ok = hit_panel(d['o'], d['w'])
        Lv = np.zeros((len(d['w']), 3))
        if mode == 'q':
            pid = patch_of(x[ok]); wl = to_panel_local(-d['w'][ok]); Lv[ok] = q[pid, pbin(wl)]
        elif mode == 'X1':
            Lv[ok] = L1(x[ok], -d['w'][ok])
        elif mode == 'X1c':
            Lv[ok] = L1(panel_centre(patch_of(x[ok])), -d['w'][ok])
        elif mode == 'CP':
            Lv[ok] = L1_cp(x[ok], -d['w'][ok], grid)
        E[pp] = (Lv * d['rw'][:, None]).sum(0)
    return E


def L1_cp(x, wout, g):
    """carrier with the incident (light) state stored at g x g control points per panel patch (cell centres), bilinear in log-amplitude and
    in direction (renormalised) between control points; ER evaluated exactly at the continuous outgoing direction."""
    h = PS / g
    gy = np.arange(PY[0] + h / 2, PY[1], h); gz = np.arange(PZ[0] + h / 2, PZ[1], h)
    def at(y, z):
        c = np.stack([np.full(len(y), PX), y, z], 1); dl = LIGHT - c; d2 = (dl ** 2).sum(1)
        return np.log(I / d2), dl / np.sqrt(d2)[:, None]
    fy = np.clip((x[:, 1] - gy[0]) / h, 0, len(gy) - 1 - 1e-9); fz = np.clip((x[:, 2] - gz[0]) / h, 0, len(gz) - 1 - 1e-9)
    if len(gy) == 1:
        fy = np.zeros(len(x))
    j0, k0 = np.floor(fy).astype(int), np.floor(fz).astype(int); j1 = np.minimum(j0 + 1, len(gy) - 1); k1 = np.minimum(k0 + 1, len(gz) - 1)
    ay, az = fy - j0, fz - k0
    la = 0; wd = 0
    for jj, kk, wgt in ((j0, k0, (1 - ay) * (1 - az)), (j1, k0, ay * (1 - az)), (j0, k1, (1 - ay) * az), (j1, k1, ay * az)):
        l_, w_ = at(gy[jj], gz[kk]); la = la + wgt * l_; wd = wd + wgt[:, None] * w_
    wd = wd / np.linalg.norm(wd, axis=1, keepdims=True)
    f = ER.er_eval(P, to_panel_local(wout), to_panel_local(wd))
    return f * np.exp(la)[:, None]


def metr(Ehat, Eref):
    a, r = lum(Ehat), lum(Eref)
    return dict(ratio=float(a.sum() / r.sum()), rel_l1=float(np.abs(a - r).sum() / r.sum()))


if __name__ == '__main__':
    T = {}
    t0 = time.time()
    ys = recv_points(8)
    Eref = np.stack([E_dense(ys[p], 0.004).mean(0) for p in range(9)])
    T['dense4mm_s'] = time.time() - t0
    Eref2 = np.stack([E_dense(recv_points(4)[p], 0.002).mean(0) for p in range(9)])
    Eref4b = np.stack([E_dense(recv_points(4)[p], 0.004).mean(0) for p in range(9)])
    res = {'light': LIGHT.tolist(), 'Eref_lum_per_patch': lum(Eref).tolist(),
           'dense_conv_2mm_vs_4mm_rel_l1(4x4 pts)': metr(Eref4b, Eref2), 'recv_pts_8x8_vs_4x4_rel_l1': metr(Eref4b, Eref)}
    t0 = time.time(); q = q_table(); T['q_s'] = time.time() - t0
    D = 64
    arms = {'q': {}, 'X1c': {}, 'X1': {}, 'CP1': {'grid': 1}, 'CP2': {'grid': 2}, 'CP4': {'grid': 4}, 'CP8': {'grid': 8}}
    per = {k: [] for k in list(arms) + ['X1_strat', 'FOOT3', 'FOOT6']}
    t0 = time.time()
    for dd in range(D):
        rng = np.random.default_rng(960000 + dd)
        dr_ = draw(rng)
        for k, kw in arms.items():
            mode = 'CP' if k.startswith('CP') else k
            per[k].append(estimate(dr_, mode, q=q, grid=kw.get('grid')))
        rng = np.random.default_rng(961000 + dd)
        ds = draw(rng, strat=True)
        per['X1_strat'].append(estimate(ds, 'X1'))
        per['FOOT3'].append(estimate(ds, 'FOOT', foot=3))
        per['FOOT6'].append(estimate(ds, 'FOOT', foot=6))
    T['draws_s'] = time.time() - t0
    for k, v in per.items():
        v = np.array(v)
        pm = metr(v.mean(0), Eref)
        pd = [metr(x, Eref)['rel_l1'] for x in v]
        res[k] = dict(pooled=pm, per_draw_rel_l1_median=float(np.median(pd)), p90=float(np.percentile(pd, 90)), worst=float(np.max(pd)),
                      per_draw_ratio_sd=float(np.std([metr(x, Eref)['ratio'] for x in v])))
    res['timing_s'] = T
    json.dump(res, open(os.path.join(H, f'ck2_model_{TAG}.json'), 'w'), indent=1)
    print(json.dumps(res, indent=1))
