"""WL98 D1 scratch (design input only, not evidence): flatland toy of sharp first-flight delivery.

Question: how much can a learned, geometry-adaptive, flux-conserving pooling of receiver-gather samples
(the XQT learned role) gain over (a) its learning-free twin (one fixed pooling kernel) and (b) simply
tracing more rays?  Expected errors are computed analytically from the exact element field and the
per-sample variance of the gather estimator (bias^2 + variance of a linear pooling), not from single draws.

Flatland.  Glossy sender segment y=0, x in [-XS, XS], normal +y.  Lambert receiver line y=H, x in [XR0, XR1],
normal -y, NR equal elements.  Optional opaque occluder segment at y=H/2.  Point light below the receiver line.
Lobe in the cos-weighted angular measure: f(th_l, th) cos th = RHO * N(th; -th_l, alpha) (mass beyond the
horizon is lost, so the lobe is passive).  Receiver gather: cosine-weighted direction, exact sender evaluation
(the X1 analogue).  Pooling: column-stochastic Gaussian kernel over receiver elements (conserves the flux of
every element's estimate exactly).

Arms per test configuration and ray budget K (rays per receiver element):
  raw        explicit, no pooling                      (W = I)
  fixed      one bandwidth chosen on training configs  (learning-free twin)
  cfg_oracle best single bandwidth for this configuration (uses the exact field: optimistic)
  col_oracle best bandwidth per source element, coordinate descent with the exact field
             (optimistic bound for any geometry-adaptive linear flux-conserving pooling of these samples)
eta = (rel error raw / rel error arm)^2: how many times more rays the unpooled explicit gather needs to match.
"""
import json, sys, numpy as np

XS, NR, RHO = 0.6, 128, 0.8
rng = np.random.default_rng(98001)
FAM = {
    # grazing: light far to the left and low; lobe leaves at 70-85 deg; receiver line low so the glint lands in span
    'grazing': dict(H=0.35, XR0=-0.6, XR1=3.4, L=((-2.0, -1.0), (0.08, 0.35))),
    # non-grazing: light above-left; lobe leaves at 15-45 deg
    'nongrazing': dict(H=1.5, XR0=-1.0, XR1=3.0, L=((-1.4, -0.2), (0.7, 1.3))),
}


def light_irr(x, L, occ, g):
    dx, dy = L[0] - x, L[1]
    r = np.hypot(dx, dy)
    E = np.maximum(dy / r, 0) / r                     # flatland point light: I cos / r (I = 1)
    yo = g['H'] / 2
    if occ is not None and L[1] > yo:
        xc = x + dx * (yo / L[1])
        E = np.where((xc >= occ[0]) & (xc <= occ[1]), 0.0, E)
    return E, np.arctan2(dx, dy)


def lobe(dth, a):
    return np.exp(-0.5 * (dth / a) ** 2) / (a * np.sqrt(2 * np.pi))


def exact_elements(L, occ, a, g, nsub=24, ns=16000):
    H, yo = g['H'], g['H'] / 2
    w = (g['XR1'] - g['XR0']) / NR
    yx = g['XR0'] + (np.arange(NR * nsub) + 0.5) * w / nsub
    xs = -XS + (np.arange(ns) + 0.5) * 2 * XS / ns
    Ex, thl = light_irr(xs, L, occ, g)
    out = np.zeros(yx.size)
    for c in range(0, yx.size, 256):
        Y = yx[c:c + 256, None]
        dx = Y - xs[None, :]
        r2 = dx ** 2 + H ** 2
        th = np.arctan2(dx, H)                          # direction x -> y, angle from +y
        v = RHO * Ex[None, :] * lobe(th + thl[None, :], a) * H / r2
        if occ is not None:
            xc = xs[None, :] + dx * (yo / H)
            v = np.where((xc >= occ[0]) & (xc <= occ[1]), 0.0, v)
        out[c:c + 256] = v.sum(1) * (2 * XS / ns)
    return out.reshape(NR, nsub).mean(1)


def gather_stats(L, occ, a, g, S=32768):
    """per-element mean and per-sample variance of the cosine-weighted receiver gather with exact sender eval."""
    H, yo = g['H'], g['H'] / 2
    w = (g['XR1'] - g['XR0']) / NR
    m, v = np.zeros(NR), np.zeros(NR)
    for c in range(0, NR, 16):
        n = min(16, NR - c)
        yx = g['XR0'] + (np.arange(c, c + n)[:, None] + rng.random((n, S))) * w
        thp = np.arcsin(rng.random((n, S)) * 2 - 1)     # cosine-weighted in flatland: sin(th) uniform
        xh = yx + H * np.tan(thp)
        ok = np.abs(xh) <= XS
        if occ is not None:
            xc = yx + (H - yo) * np.tan(thp)
            ok &= ~((xc >= occ[0]) & (xc <= occ[1]))
        Ex, thl = light_irr(np.clip(xh, -XS, XS), L, occ, g)
        val = np.where(ok, 2 * RHO * Ex * lobe(-thp + thl, a) / np.cos(thp), 0.0)
        m[c:c + n], v[c:c + n] = val.mean(1), val.var(1)
    return m, v


I = np.arange(NR)


def col(j, b):
    k = np.exp(-0.5 * ((I - j) / b) ** 2) if b > 0 else (I == j).astype(float)
    return k / k.sum()


def pool_matrix(b):
    return np.stack([col(j, b) for j in range(NR)], 1)


def total_mse(W, E, var, K):
    return float(((W @ E - E) ** 2).sum() + ((W ** 2) @ (var / K)).sum())


BGRID = [0, 0.5, 1, 1.5, 2, 3, 4, 6, 8, 12, 16, 24]


def col_oracle(E, var, K, b0):
    """coordinate descent over per-source-element bandwidths with incremental bias / variance updates."""
    bs = [b0] * NR
    W = pool_matrix(b0)
    bias = W @ E - E
    vv = (W ** 2) @ (var / K)
    cols = {(j, b): col(j, b) for j in range(NR) for b in BGRID}
    for _ in range(3):
        for j in range(NR):
            old = W[:, j]
            best = None
            for b in BGRID:
                new = cols[(j, b)]
                nb = bias + (new - old) * E[j]
                nv = vv + (new ** 2 - old ** 2) * var[j] / K
                e = (nb ** 2).sum() + nv.sum()
                if best is None or e < best[0]:
                    best = (e, b, nb, nv)
            _, b, bias, vv = best
            W[:, j] = cols[(j, b)]
            bs[j] = b
    return float((bias ** 2).sum() + vv.sum()), bs


def config(kind):
    f = FAM[kind]
    L = (rng.uniform(*f['L'][0]), rng.uniform(*f['L'][1]))
    occ = None
    if rng.random() < 0.75:                             # occluder somewhere above the sender / in the lobe path
        o1 = rng.uniform(-0.5, 1.5 if kind == 'grazing' else 1.2)
        occ = (o1, o1 + rng.uniform(0.05, 0.5))
    return dict(f, L=L), occ


def main():
    res = {}
    for kind in ('grazing', 'nongrazing'):
        for a in (0.03, 0.10):
            data = []
            for _ in range(40):
                g, occ = config(kind)
                E = exact_elements(g['L'], occ, a, g)
                if E.sum() <= 0:
                    continue
                m, var = gather_stats(g['L'], occ, a, g)
                data.append((E, var, float(np.abs(m - E).sum() / E.sum())))
            tr, te = data[:20], data[20:]
            for K in (16, 64, 256):
                tot = {b: sum(total_mse(pool_matrix(b), E, v, K) / (E ** 2).sum() for E, v, _ in tr) for b in BGRID}
                bstar = min(tot, key=tot.get)
                Wf = pool_matrix(bstar)
                raw, fixed, cfgor, color = [], [], [], []
                for E, v, _ in te:
                    n2 = (E ** 2).sum()
                    raw.append(np.sqrt(total_mse(np.eye(NR), E, v, K) / n2))
                    fixed.append(np.sqrt(total_mse(Wf, E, v, K) / n2))
                    cfgor.append(min(np.sqrt(total_mse(pool_matrix(b), E, v, K) / n2) for b in BGRID))
                    color.append(np.sqrt(col_oracle(E, v, K, bstar)[0] / n2))
                raw, fixed, cfgor, color = map(np.array, (raw, fixed, cfgor, color))
                eta = lambda p: float(np.median((raw / p) ** 2))
                key = f'{kind}_a{a}_K{K}'
                res[key] = dict(n_test=len(te), bstar_elements=bstar, raw_med=float(np.median(raw)),
                                fixed_med=float(np.median(fixed)), cfg_oracle_med=float(np.median(cfgor)),
                                col_oracle_med=float(np.median(color)), eta_fixed=eta(fixed),
                                eta_cfg_oracle=eta(cfgor), eta_col_oracle=eta(color),
                                oracle_over_twin_err_ratio_med=float(np.median(fixed / color)),
                                gather_mean_check_relL1_med=float(np.median([d[2] for d in data])))
                print(key, json.dumps(res[key]))
                sys.stdout.flush()
    json.dump(res, open(sys.argv[1], 'w'), indent=1)


if __name__ == '__main__':
    main()
