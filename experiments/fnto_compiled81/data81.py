"""WL81 Stage A data: material populations, directional populations and reference targets (layered81 random walks on the GPU).

  python data81.py <out_dir> materials            # materials.npz
  python data81.py <out_dir> pairs                # pairs_train / pairs_val / fit_{gate,test_iid,test_block}  (1024 walks)
  python data81.py <out_dir> eval <set>           # eval_<set>.npz with f, fA, fB + noise-floor rule (adaptive doubling)
  python data81.py <out_dir> highlight <set>      # hl_<set>.npz: highlight-grid reference statistics (full and halves)
  [--dev] shrinks every population (dev root only)

Lane layout of eval sets (as WL80C): (material, theta, phi, sample); wo from the analytic defensive mixture with exact density q.
"""
import json, math, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', 'fnto_compiled80c'))
P = json.load(open(os.path.join(HERE, 'protocol', 'compiled81.json')))
A = P['angular']
M = P['materials']
SD = P['seeds']
WALK = P['reference_targets']['walks']
DEV = '--dev' in sys.argv
SIZES = {'train': 3000, 'val': 200, 'gate': 300, 'test_iid': 300, 'test_block': 300, 'train_eval': 300}
if DEV:
    SIZES = {'train': 64, 'val': 16, 'gate': 16, 'test_iid': 16, 'test_block': 16, 'train_eval': 16}
LANES = 1 << 23

_mi = None


def mi_init():
    global _mi, L
    if _mi is None:
        import layered81 as L_
        L = L_
        _mi = L.init(os.environ.get('C80_MI_VARIANT', 'cuda_ad_rgb'))
    return _mi


# ---------------------------------------------------------------- materials
def _lin(u, a, b):
    return a + u * (b - a)


def _log(u, a, b):
    return np.exp(np.log(a) + u * (np.log(b) - np.log(a)))


def sample_box(rng, n):
    b = M['box']
    u = rng.random((n, 10))
    d = np.empty((n, 10))
    d[:, 0] = _log(u[:, 0], *b['alpha_i'][:2])
    d[:, 1] = _lin(u[:, 1], *b['eta'][:2])
    d[:, 2] = _log(u[:, 2], *b['tau'][:2])
    d[:, 3:6] = _lin(u[:, 3:6], *b['a'][:2])
    d[:, 6:9] = _lin(u[:, 6:9], *b['r'][:2])
    d[:, 9] = _log(u[:, 9], *b['alpha_b'][:2])
    return d


def in_block(d):
    k = M['heldout_block']
    return ((d[:, 0] >= k['alpha_i'][0]) & (d[:, 0] <= k['alpha_i'][1]) & (d[:, 2] >= k['tau'][0]) & (d[:, 2] <= k['tau'][1])
            & (d[:, 9] >= k['alpha_b'][0]) & (d[:, 9] <= k['alpha_b'][1]))


def normalise(d):
    b = M['box']
    d = np.asarray(d, np.float64)
    x = np.empty_like(d)
    lg = lambda v, r: 2 * (np.log(v) - np.log(r[0])) / (np.log(r[1]) - np.log(r[0])) - 1
    ln = lambda v, r: 2 * (v - r[0]) / (r[1] - r[0]) - 1
    x[:, 0] = lg(d[:, 0], b['alpha_i']); x[:, 1] = ln(d[:, 1], b['eta']); x[:, 2] = lg(d[:, 2], b['tau'])
    x[:, 3:6] = ln(d[:, 3:6], b['a']); x[:, 6:9] = ln(d[:, 6:9], b['r']); x[:, 9] = lg(d[:, 9], b['alpha_b'])
    return x.astype(np.float32)


def materials(out):
    rng = np.random.default_rng(SD['materials'])
    need = SIZES['train'] + SIZES['val'] + SIZES['gate'] + SIZES['test_iid']
    nb, bl, drawn = [], [], 0
    while sum(len(x) for x in nb) < need or sum(len(x) for x in bl) < SIZES['test_block']:
        d = sample_box(rng, 20000)
        drawn += len(d)
        b = in_block(d)
        nb.append(d[~b]); bl.append(d[b])
    nb = np.concatenate(nb)[:need]; bl = np.concatenate(bl)[:SIZES['test_block']]
    o = np.cumsum([0, SIZES['train'], SIZES['val'], SIZES['gate'], SIZES['test_iid']])
    sp = {'train': nb[o[0]:o[1]], 'val': nb[o[1]:o[2]], 'gate': nb[o[2]:o[3]], 'test_iid': nb[o[3]:o[4]], 'test_block': bl}
    np.savez(os.path.join(out, 'materials.npz'), **sp)
    json.dump({'drawn': drawn, 'block_fraction': float(np.concatenate([in_block(sample_box(np.random.default_rng(1), 200000))]).mean()),
               'sizes': {k: len(v) for k, v in sp.items()}}, open(os.path.join(out, 'materials_meta.json'), 'w'), indent=1)


def mats_of(out, name):
    m = np.load(os.path.join(out, 'materials.npz'))
    if name == 'train_eval':
        return m['train'][:SIZES['train_eval']]
    return m[name]


# ---------------------------------------------------------------- directions (numpy float64)
def sph(theta, phi):
    st = np.sin(theta)
    return np.stack([st * np.cos(phi), st * np.sin(phi), np.cos(theta)], -1)


def uniform_dirs(u):
    z = u[:, 0]; r = np.sqrt(np.maximum(0, 1 - z * z)); ph = 2 * np.pi * u[:, 1]
    return np.stack([r * np.cos(ph), r * np.sin(ph), z], -1)


def cosine_dirs(u):
    r = np.sqrt(u[:, 0]); ph = 2 * np.pi * u[:, 1]
    return np.stack([r * np.cos(ph), r * np.sin(ph), np.sqrt(np.maximum(0, 1 - u[:, 0]))], -1)


def ggx_D(hz, a):
    a2 = a * a
    return a2 / (np.pi * (hz * hz * (a2 - 1) + 1) ** 2)


def vndf_sample(wi, a, u):
    """Isotropic GGX visible-normal sampling (spherical caps) and mirror reflection."""
    s = np.stack([a * wi[:, 0], a * wi[:, 1], wi[:, 2]], 1)
    s /= np.linalg.norm(s, axis=1, keepdims=True)
    ph = 2 * np.pi * u[:, 0]
    z = (1 - u[:, 1]) * (1 + s[:, 2]) - s[:, 2]
    st = np.sqrt(np.maximum(0, 1 - z * z))
    hs = np.stack([st * np.cos(ph), st * np.sin(ph), z], 1) + s
    h = np.stack([a * hs[:, 0], a * hs[:, 1], np.maximum(hs[:, 2], 0)], 1)
    h /= np.maximum(np.linalg.norm(h, axis=1, keepdims=True), 1e-300)
    return 2 * (wi * h).sum(1, keepdims=True) * h - wi


def vndf_pdf(wi, wo, a):
    """Density of vndf_sample: G1(wi) D(h) / (4 cos_i) = D / (2 (cos_i + sqrt(cos_i^2 + a^2 sin_i^2))), wo above the horizon."""
    h = wi + wo
    h /= np.maximum(np.linalg.norm(h, axis=1, keepdims=True), 1e-300)
    ci = wi[:, 2]
    si2 = np.maximum(0, 1 - ci * ci)
    p = ggx_D(h[:, 2], a) / (2 * (ci + np.sqrt(ci * ci + a * a * si2)))
    return np.where((wo[:, 2] > 0) & ((wi * h).sum(1) > 0), p, 0.0)


W_MIX = (0.3, 0.2, 0.3, 0.2)  # cosine, uniform, GGX alpha_i, GGX min(1, eta alpha_b)


def mixture(desc_lane, wi, rng):
    """Returns wo [n,3], q [n] (exact mixture density), use [n] (bool; False = failed GGX sample replaced by a uniform draw)."""
    n = len(wi)
    a1 = desc_lane[:, 0]; a2 = np.minimum(1.0, desc_lane[:, 1] * desc_lane[:, 9])
    c = rng.random(n); u = rng.random((n, 2))
    cw = np.cumsum(W_MIX)
    k = np.searchsorted(cw, c, side='right').clip(0, 3)
    wo = np.empty((n, 3)); use = np.ones(n, bool)
    s = k == 0; wo[s] = cosine_dirs(u[s])
    s = k == 1; wo[s] = uniform_dirs(u[s])
    for kk, aa in [(2, a1), (3, a2)]:
        s = k == kk
        if s.any():
            w = vndf_sample(wi[s], aa[s], u[s])
            bad = w[:, 2] <= 1e-7
            w[bad] = uniform_dirs(rng.random((bad.sum(), 2)))
            use[np.nonzero(s)[0][bad]] = False
            wo[s] = w
    wo /= np.linalg.norm(wo, axis=1, keepdims=True)
    q = (W_MIX[0] * np.maximum(wo[:, 2], 0) / np.pi + W_MIX[1] / (2 * np.pi) + W_MIX[2] * vndf_pdf(wi, wo, a1) + W_MIX[3] * vndf_pdf(wi, wo, a2))
    return wo, q, use


# ---------------------------------------------------------------- reference walks
def ref_eval(desc, mid, wi, wo, nwalk, seed, round_=0, max_depth=1024):
    """Per pair: mean eval over nwalk walks, returned as two halves (fA, fB) [n, 3] float64.
    desc [nm, 10] material table; mid [n] material index per pair."""
    mi = mi_init()
    import drjit as dr
    n = len(wi)
    assert nwalk % 2 == 0
    fA = np.empty((n, 3)); fB = np.empty((n, 3))
    tab = [mi.Float(np.ascontiguousarray(desc[:, j], np.float32)) for j in range(10)]
    per = max(1, LANES // nwalk)
    off = 0
    for s in range(0, n, per):
        e = min(n, s + per)
        npair = e - s
        Lw = npair * nwalk
        lane = dr.arange(mi.UInt32, Lw)
        pidx = lane // nwalk
        m_l = dr.gather(mi.UInt32, mi.UInt32(np.ascontiguousarray(mid[s:e], np.uint32)), pidx)
        g = lambda j: dr.gather(mi.Float, tab[j], m_l)
        mat = L.Mat(g(0), g(1), g(2), mi.Color3f(g(3), g(4), g(5)), mi.Color3f(g(6), g(7), g(8)), g(9))
        V = lambda a: mi.Vector3f(*[dr.gather(mi.Float, mi.Float(np.ascontiguousarray(a[s:e, k], np.float32)), pidx) for k in range(3)])
        rng = mi.PCG32(size=Lw, initstate=seed + 7919 * round_, initseq=dr.arange(mi.UInt64, Lw) + off)
        off += Lw
        ev = L.layered_eval(mat, V(wi), V(wo), rng, max_depth)
        x = np.stack([np.asarray(ev[k]) for k in range(3)], 1).astype(np.float64).reshape(npair, nwalk, 3)
        h = nwalk // 2
        fA[s:e] = x[:, :h].mean(1); fB[s:e] = x[:, h:].mean(1)
    return fA, fB


# ---------------------------------------------------------------- sets
def train_style(desc, n_wi, n_wo, rng):
    band = A['heldout_incident_band_deg']
    nm = len(desc)
    cmin = np.cos(np.radians(88.0))
    tot = nm * n_wi
    th = np.empty(0)
    while len(th) < tot:
        cc = rng.uniform(cmin, 1.0, 2 * tot)
        t = np.degrees(np.arccos(cc))
        th = np.concatenate([th, np.radians(t[(t < band[0]) | (t > band[1])])])
    wi1 = sph(th[:tot], rng.uniform(0, 2 * np.pi, tot))
    mm = np.repeat(np.arange(nm), n_wi * n_wo)
    wi = np.repeat(wi1, n_wo, 0)
    wo, q, use = mixture(desc[mm], wi, rng)
    return mm, wi, wo, q, use


def eval_style(desc, rng):
    nm = len(desc)
    th = np.radians(np.array(A['eval_theta_deg'], np.float64))
    nph, nwo = A['eval_n_phi'], A['eval_wo']
    phis = rng.uniform(0, 2 * np.pi, (nm, nph))
    T = np.broadcast_to(th[None, :, None], (nm, len(th), nph))
    PH = np.broadcast_to(phis[:, None, :], (nm, len(th), nph))
    wi1 = sph(T.ravel(), PH.ravel())
    mm = np.repeat(np.repeat(np.arange(nm), len(th) * nph), nwo)
    wi = np.repeat(wi1, nwo, 0)
    wo, q, use = mixture(desc[mm], wi, rng)
    return mm, wi, wo, q, use, phis, np.array([nm, len(th), nph, nwo])


def pairs(out):
    rng = np.random.default_rng(SD['directions'])
    meta = {}
    for k_set, (name, src, nwi, nwo, nw) in enumerate([('pairs_train', 'train', A['train_wi'], A['train_wo'], WALK['train']),
                                    ('pairs_val', 'val', A['val_wi'], A['val_wo'], WALK['val']),
                                    ('fit_gate', 'gate', A['train_wi'], A['train_wo'], WALK['fit']),
                                    ('fit_test_iid', 'test_iid', A['train_wi'], A['train_wo'], WALK['fit']),
                                    ('fit_test_block', 'test_block', A['train_wi'], A['train_wo'], WALK['fit'])]):
        t0 = time.time()
        d = mats_of(out, src)
        mm, wi, wo, q, use = train_style(d, nwi, nwo, rng)
        fA, fB = ref_eval(d, mm, wi, wo, nw, SD['walks'] + 100 * (k_set + 1))
        f = 0.5 * (fA + fB)
        np.savez(os.path.join(out, f'{name}.npz'), m=mm.astype(np.int32), wi=wi.astype(np.float32), wo=wo.astype(np.float32),
                 f=f.astype(np.float32), fA=fA.astype(np.float32), fB=fB.astype(np.float32))
        meta[name] = {'pairs': len(wi), 'walks': nw, 'seconds': time.time() - t0, 'walks_per_s': len(wi) * nw / (time.time() - t0)}
        print(name, json.dumps(meta[name]), flush=True)
    json.dump(meta, open(os.path.join(out, 'pairs_meta.json'), 'w'), indent=1)


def noise_floor(ev, fA, fB):
    """E_L1 / E_alb of fA vs fB (paired, same samples) / 2 -> per (material, theta, phi) arrays."""
    nm, nth, nph, nwo = [int(v) for v in ev['shape']]
    w = (ev['use'].astype(np.float64) / ev['q'].astype(np.float64))[:, None]
    G = nm * nth * nph
    d = (np.abs(fA - fB) * w).sum(1).reshape(G, nwo).sum(1)
    r = (0.5 * (fA + fB) * w).sum(1).reshape(G, nwo).sum(1)
    aA = (fA * w).reshape(G, nwo, 3).sum(1); aB = (fB * w).reshape(G, nwo, 3).sum(1)
    el1 = 0.5 * d / np.maximum(r, 1e-12)
    ealb = 0.5 * np.abs(aA - aB).sum(1) / np.maximum(0.5 * (aA + aB).sum(1), 1e-12)
    return el1.reshape(nm, nth, nph), ealb.reshape(nm, nth, nph)


def nf_summary(el1, ealb):
    q = lambda x, v: float(np.quantile(x.ravel(), v))
    s = {'E_L1_median': q(el1, .5), 'E_L1_p90': q(el1, .9), 'E_alb_median': q(ealb, .5), 'E_alb_p95': q(ealb, .95)}
    lim = {'E_L1_median': 0.025, 'E_L1_p90': 0.0625, 'E_alb_median': 0.0075, 'E_alb_p95': 0.025}
    s['PASS'] = all(s[k] <= lim[k] for k in lim)
    return s


def eval_set(out, name):
    rng = np.random.default_rng(SD['directions'] + 17 * (['gate', 'test_iid', 'test_block', 'train_eval'].index(name) + 1))
    d = mats_of(out, name)
    mm, wi, wo, q, use, phis, shape = eval_style(d, rng)
    ev = {'m': mm.astype(np.int32), 'wi': wi.astype(np.float32), 'wo': wo.astype(np.float32), 'q': q.astype(np.float32),
          'use': use, 'phis': phis.astype(np.float32), 'shape': shape}
    nw = WALK['eval_initial']
    seed = SD['walks'] + 1000 * (['gate', 'test_iid', 'test_block', 'train_eval'].index(name) + 1)
    t0 = time.time()
    fA, fB = ref_eval(d, mm, wi, wo, nw, seed, 0)
    hist = []
    rnd = 0
    while True:
        el1, ealb = noise_floor(ev, fA, fB)
        s = nf_summary(el1, ealb)
        s.update({'walks': nw, 'seconds': time.time() - t0})
        hist.append(s)
        print(name, json.dumps(s), flush=True)
        if s['PASS'] or nw >= 65536:
            break
        rnd += 1
        gA, gB = ref_eval(d, mm, wi, wo, nw, seed, rnd)
        fA, fB = 0.5 * (fA + gA), 0.5 * (fB + gB)
        nw *= 2
    ev.update({'f': (0.5 * (fA + fB)).astype(np.float32), 'fA': fA.astype(np.float32), 'fB': fB.astype(np.float32)})
    np.savez(os.path.join(out, f'eval_{name}.npz'), **ev)
    np.savez_compressed(os.path.join(out, f'noise_{name}.npz'), el1=el1, ealb=ealb)
    json.dump({'history': hist, 'PASS': hist[-1]['PASS'], 'walks': nw}, open(os.path.join(out, f'noise_{name}.json'), 'w'), indent=1)


# ---------------------------------------------------------------- highlight grids (reference side)
def highlight_set(out, name):
    import eval80c as EV  # grid_dirs / highlight_stats (WL80C, pure numpy)
    hl = A['highlight']
    n = hl['grid']
    rng = np.random.default_rng(SD['directions'] + 991 * (['gate', 'test_iid'].index(name) + 1))
    d = mats_of(out, name)
    ev = np.load(os.path.join(out, f'eval_{name}.npz'))
    phis = ev['phis'][:, 0].astype(np.float64)
    ths = np.radians(np.array(hl['theta_deg'], np.float64))
    nm = len(d)
    mk = np.repeat(np.arange(nm), len(ths)); tk = np.tile(ths, nm)
    wi = sph(tk, phis[mk])
    smax = np.clip(hl['extent_alpha_mult'] * d[mk, 0], hl['extent_min'], hl['extent_max'])
    G = len(mk)
    keys = ['peak_wo', 'vpeak', 'area', 'vx', 'vy', 'touch']
    st = {v: {k: [] for k in keys} for v in ['full', 'A', 'B']}
    aref = np.zeros(G)
    seed = SD['walks'] + 50000 + 1000 * (['gate', 'test_iid'].index(name) + 1)
    chunk = max(1, (1 << 21) // (n * n))
    t0 = time.time()
    for c0 in range(0, G, chunk):
        cs = np.arange(c0, min(G, c0 + chunk))
        wo, sx, sy = EV.grid_dirs(wi[cs], smax[cs], n)
        ok = wo[..., 2] > 1e-4
        wo_c = wo.copy(); wo_c[..., 2] = np.maximum(wo_c[..., 2], 1e-4)
        wo_c /= np.linalg.norm(wo_c, axis=-1, keepdims=True)
        mm = np.repeat(mk[cs], n * n)
        fA, fB = ref_eval(d, mm, np.repeat(wi[cs], n * n, 0), wo_c.reshape(-1, 3), WALK['highlight_grid'], seed, c0)
        for v, arr in [('full', 0.5 * (fA + fB)), ('A', fA), ('B', fB)]:
            vv = np.where(ok, arr.mean(1).reshape(len(cs), -1), -1.0)
            s = EV.highlight_stats(vv, wo, sx, sy, n)
            for k in keys:
                st[v][k].append(s[k])
        nsa = hl['albedo_samples']
        mA = np.repeat(mk[cs], nsa); wiA = np.repeat(wi[cs], nsa, 0)
        woA, qA, useA = mixture(d[mA], wiA, rng)
        gA, gB = ref_eval(d, mA, wiA, woA, WALK['highlight_albedo'], seed + 500, c0)
        aref[cs] = (np.where(useA, 0.5 * (gA + gB).mean(1) / qA, 0).reshape(len(cs), nsa)).mean(1)
        if c0 == 0:
            print(name, 'highlight chunk s', time.time() - t0, flush=True)
    outd = {'mat': mk, 'theta': tk, 'wi': wi, 'smax': smax, 'aref': aref}
    for v in st:
        for k in keys:
            outd[f'{v}_{k}'] = np.concatenate(st[v][k])
    cos_pk = np.clip(outd['full_peak_wo'][:, 2], 0, 1)
    outd['pron'] = outd['full_vpeak'] >= hl['pronounced_factor'] * aref * cos_pk / math.pi
    np.savez(os.path.join(out, f'hl_{name}.npz'), **outd)
    # highlight noise floor: half A vs half B
    pr = outd['pron']
    dots = np.clip((outd['A_peak_wo'] * outd['B_peak_wo']).sum(1), -1, 1)
    pk = np.degrees(np.arccos(dots))
    res = ~(outd['A_touch'] | outd['B_touch'])
    lw = 0.5 * np.log(outd['A_area'] / np.maximum(outd['B_area'], 1))
    s = {'n_cases': int(G), 'n_pronounced': int(pr.sum()), 'peak_err_deg_median_halves': float(np.median(pk[pr])) if pr.any() else None,
         'abs_logw_median_halves': float(np.median(np.abs(lw[pr & res]))) if (pr & res).any() else None, 'seconds': time.time() - t0}
    s['note'] = 'half-vs-half differences; the full-N reference error is about half of these'
    s['PASS'] = (s['peak_err_deg_median_halves'] is None or s['peak_err_deg_median_halves'] / 2 <= 0.25) and \
                (s['abs_logw_median_halves'] is None or s['abs_logw_median_halves'] / 2 <= math.log(1.15) / 4)
    json.dump(s, open(os.path.join(out, f'hl_noise_{name}.json'), 'w'), indent=1)
    print(name, json.dumps(s), flush=True)


if __name__ == '__main__':
    out, what = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    if what == 'materials':
        materials(out)
    elif what == 'pairs':
        pairs(out)
    elif what == 'eval':
        eval_set(out, sys.argv[3])
    elif what == 'highlight':
        highlight_set(out, sys.argv[3])
    os._exit(0)
