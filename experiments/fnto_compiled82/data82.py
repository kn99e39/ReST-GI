"""WL82 data (reference walks = WL81 layered81 via data81.ref_eval; WL81 directional conventions via data81).

  python data82.py <out> <w81data> repro                      # WL81 reference reproduction (eval_gate / fit_gate prefixes, bit-identity)
  python data82.py <out> <w81data> fitset <set>               # fit82_<set>.npz: 64 wi x 64 wo per material, 1024 walks
  python data82.py <out> <w81data> realeval                   # eval82_real.npz (WL81 eval layout) + hl82_real.npz + sphere82_real.npz
  python data82.py <out> <w81data> albset <set> --shard=i/K   # independent albedo samples at the eval incident directions (one shard)
  python data82.py <out> <w81data> albmerge <set> K
sets: gate (development; WL81-inspected), test_iid, test_block (untouched holdouts), real (predeclared named materials)
"""
import json, math, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', 'fnto_compiled81'))
sys.path.insert(0, os.path.join(HERE, '..', 'fnto_compiled80c'))
import data81 as DA  # noqa: E402
P82 = json.load(open(os.path.join(HERE, 'protocol', 'compiled82.json')))
SD = P82['seeds']
SD81 = DA.SD
SETS = ['gate', 'test_iid', 'test_block', 'real']
DEV = '--dev' in sys.argv
DEVN = 16


def real_desc():
    return np.array([m['descriptor'] for m in P82['real_materials']['list']], np.float64)


def descs(w81, name):
    d = real_desc() if name == 'real' else np.load(os.path.join(w81, 'materials.npz'))[name]
    return d[:DEVN] if (DEV and name != 'real') else d


def eval_path(out, w81, name):
    return os.path.join(out, 'eval82_real.npz') if name == 'real' else os.path.join(w81, f'eval_{name}.npz')


# ---------------------------------------------------------------- reference reproduction (stop condition)
def repro(out, w81):
    res = {}
    d = np.load(os.path.join(w81, 'materials.npz'))['gate']
    n = 32768 if not DEV else 4096
    ev = np.load(os.path.join(w81, 'eval_gate.npz'))
    fA, fB = DA.ref_eval(d, ev['m'][:n], ev['wi'][:n], ev['wo'][:n], 4096, SD81['walks'] + 1000 * 1, 0, pair0=0)
    res['eval_gate_prefix'] = {'pairs': n, 'walks': 4096, 'bit_identical_fA': bool(np.array_equal(fA.astype(np.float32), ev['fA'][:n])),
                               'bit_identical_fB': bool(np.array_equal(fB.astype(np.float32), ev['fB'][:n])),
                               'max_abs_diff': float(max(np.abs(fA.astype(np.float32) - ev['fA'][:n]).max(), np.abs(fB.astype(np.float32) - ev['fB'][:n]).max()))}
    ft = np.load(os.path.join(w81, 'fit_gate.npz'))
    gA, gB = DA.ref_eval(d, ft['m'][:n], ft['wi'][:n], ft['wo'][:n], 1024, SD81['walks'] + 100 * 3, 0, pair0=0)
    res['fit_gate_prefix'] = {'pairs': n, 'walks': 1024, 'bit_identical_fA': bool(np.array_equal(gA.astype(np.float32), ft['fA'][:n])),
                              'bit_identical_fB': bool(np.array_equal(gB.astype(np.float32), ft['fB'][:n])),
                              'max_abs_diff': float(max(np.abs(gA.astype(np.float32) - ft['fA'][:n]).max(), np.abs(gB.astype(np.float32) - ft['fB'][:n]).max()))}
    res['REPRODUCED'] = all(v['bit_identical_fA'] and v['bit_identical_fB'] for v in res.values() if isinstance(v, dict))
    json.dump(res, open(os.path.join(out, 'repro.json'), 'w'), indent=1)
    print(json.dumps(res), flush=True)


# ---------------------------------------------------------------- fit sets
def fitset(out, w81, name):
    F = P82['fitset']
    k = SETS.index(name)
    d = descs(w81, name)
    rng = np.random.default_rng([SD['fit'], k])
    t0 = time.time()
    mm, wi, wo, q, use = DA.train_style(d, F['n_wi'], F['n_wo'], rng)
    fA, fB = DA.ref_eval(d, mm, wi, wo, F['walks'], SD['fit_walks'] + 100 * k)
    np.savez(os.path.join(out, f'fit82_{name}.npz'), m=mm.astype(np.int32), wi=wi.astype(np.float32), wo=wo.astype(np.float32),
             q=q.astype(np.float32), use=use, f=(0.5 * (fA + fB)).astype(np.float32), fA=fA.astype(np.float32), fB=fB.astype(np.float32),
             n_wi=F['n_wi'], n_wo=F['n_wo'], n_mat=len(d))
    print(name, 'fitset', len(wi), 'pairs', time.time() - t0, 's', flush=True)


# ---------------------------------------------------------------- independent albedo sets at the evaluation incident directions
def albset(out, w81, name, shard, K):
    Aa = P82['albset']
    k = SETS.index(name)
    d = descs(w81, name)
    ev = np.load(eval_path(out, w81, name))
    nm, nth, nph, nwo = [int(v) for v in ev['shape']]
    nm = min(nm, len(d))
    G = nm * nth * nph
    wiG = ev['wi'][: G * nwo: nwo].astype(np.float64)
    mG = ev['m'][: G * nwo: nwo].astype(np.int64)
    Na = Aa['n_samples'] if not DEV else 512
    g0, g1 = G * shard // K, G * (shard + 1) // K
    rng = np.random.default_rng([SD['alb'], k, shard, K])
    AA = np.zeros((g1 - g0, 3)); AB = np.zeros((g1 - g0, 3)); SE = np.zeros((g1 - g0, 3))
    WO = np.zeros(((g1 - g0) * Na, 3), np.float32); Q = np.zeros((g1 - g0) * Na, np.float32); U = np.zeros((g1 - g0) * Na, bool)
    chunk = max(1, (1 << 21) // Na)
    t0 = time.time()
    for c0 in range(g0, g1, chunk):
        c1 = min(g1, c0 + chunk)
        WI = np.repeat(wiG[c0:c1], Na, 0); M = np.repeat(mG[c0:c1], Na)
        wo, q, use = DA.mixture(d[M], WI, rng)
        fA, fB = DA.ref_eval(d, M, WI, wo, Aa['walks'], SD['alb_walks'] + 1000 * k, 0, pair0=c0 * Na)
        w = (use / q)[:, None]
        sl = slice(c0 - g0, c1 - g0)
        AA[sl] = (fA * w).reshape(c1 - c0, Na, 3).mean(1)
        AB[sl] = (fB * w).reshape(c1 - c0, Na, 3).mean(1)
        v = (0.5 * (fA + fB) * w).reshape(c1 - c0, Na, 3)
        SE[sl] = v.std(1) / math.sqrt(Na)
        ls = slice((c0 - g0) * Na, (c1 - g0) * Na)
        WO[ls] = wo; Q[ls] = q; U[ls] = use
    np.savez(os.path.join(out, f'albshard_{name}_{shard}of{K}.npz'), g0=g0, g1=g1, AA=AA, AB=AB, SE=SE, wo=WO, q=Q, use=U, Na=Na,
             seconds=time.time() - t0)
    print(name, 'albshard', shard, K, g1 - g0, 'groups', time.time() - t0, 's', flush=True)


def albmerge(out, name, K):
    parts = [np.load(os.path.join(out, f'albshard_{name}_{i}of{K}.npz')) for i in range(K)]
    o = {k: np.concatenate([p[k] for p in parts]) for k in ['AA', 'AB', 'SE', 'wo', 'q', 'use']}
    o['Na'] = int(parts[0]['Na'])
    A = 0.5 * (o['AA'] + o['AB'])
    rel_se = (o['SE'] * A).sum(1) / np.maximum(A.sum(1) ** 2, 1e-30)      # energy-weighted relative standard error of A_ref
    walk_half = 0.5 * np.abs(o['AA'] - o['AB']).sum(1) / np.maximum(A.sum(1), 1e-30)
    o['A'] = A
    np.savez(os.path.join(out, f'alb82_{name}.npz'), **o)
    s = {'groups': len(A), 'Na': o['Na'], 'rel_se_median': float(np.median(rel_se)), 'rel_se_p95': float(np.quantile(rel_se, .95)),
         'walk_half_diff_median': float(np.median(walk_half)), 'walk_half_diff_p95': float(np.quantile(walk_half, .95)),
         'note': 'rel_se = sampling standard error of the independent reference albedo (energy-weighted); walk_half = |A_A - A_B| / 2 / A'}
    json.dump(s, open(os.path.join(out, f'alb82_{name}.json'), 'w'), indent=1)
    print(name, json.dumps(s), flush=True)


# ---------------------------------------------------------------- real-material set
def highlight_ref(d, phis, seed):
    import eval80c as EV
    hl = DA.A['highlight']
    n = hl['grid']
    ths = np.radians(np.array(hl['theta_deg'], np.float64))
    nm = len(d)
    mk = np.repeat(np.arange(nm), len(ths)); tk = np.tile(ths, nm)
    wi = DA.sph(tk, phis[mk])
    smax = np.clip(hl['extent_alpha_mult'] * d[mk, 0], hl['extent_min'], hl['extent_max'])
    G = len(mk)
    keys = ['peak_wo', 'vpeak', 'area', 'vx', 'vy', 'touch']
    st = {v: {k: [] for k in keys} for v in ['full', 'A', 'B']}
    aref = np.zeros(G)
    chunk = max(1, (1 << 21) // (n * n))
    for c0 in range(0, G, chunk):
        cs = np.arange(c0, min(G, c0 + chunk))
        rng = np.random.default_rng([seed, c0])
        wo, sx, sy = EV.grid_dirs(wi[cs], smax[cs], n)
        ok = wo[..., 2] > 1e-4
        wo_c = wo.copy(); wo_c[..., 2] = np.maximum(wo_c[..., 2], 1e-4)
        wo_c /= np.linalg.norm(wo_c, axis=-1, keepdims=True)
        fA, fB = DA.ref_eval(d, np.repeat(mk[cs], n * n), np.repeat(wi[cs], n * n, 0), wo_c.reshape(-1, 3), DA.WALK['highlight_grid'], seed, c0)
        for v, arr in [('full', 0.5 * (fA + fB)), ('A', fA), ('B', fB)]:
            vv = np.where(ok, arr.mean(1).reshape(len(cs), -1), -1.0)
            s = EV.highlight_stats(vv, wo, sx, sy, n)
            for k in keys:
                st[v][k].append(s[k])
        nsa = hl['albedo_samples']
        mA = np.repeat(mk[cs], nsa); wiA = np.repeat(wi[cs], nsa, 0)
        woA, qA, useA = DA.mixture(d[mA], wiA, rng)
        gA, gB = DA.ref_eval(d, mA, wiA, woA, DA.WALK['highlight_albedo'], seed + 500, c0)
        aref[cs] = (np.where(useA, 0.5 * (gA + gB).mean(1) / qA, 0).reshape(len(cs), nsa)).mean(1)
    o = {'mat': mk, 'theta': tk, 'wi': wi, 'smax': smax, 'aref': aref}
    for v in st:
        for k in keys:
            o[f'{v}_{k}'] = np.concatenate(st[v][k])
    o['pron'] = o['full_vpeak'] >= hl['pronounced_factor'] * aref * np.clip(o['full_peak_wo'][:, 2], 0, 1) / math.pi
    return o


def sphere_geom(res):
    xs = (np.arange(res) + 0.5) / res * 2 - 1
    X, Y = np.meshgrid(xs, -xs)
    r2 = X * X + Y * Y
    mask = r2 < 0.995
    n = np.stack([X[mask], Y[mask], np.sqrt(1 - r2[mask])], 1)
    t = np.stack([n[:, 2], np.zeros(len(n)), -n[:, 0]], 1)
    t /= np.linalg.norm(t, axis=1, keepdims=True)
    b = np.cross(n, t)
    frame = np.stack([t, b, n], 1)            # rows: tangent, bitangent, normal
    v = np.array([0.0, 0.0, 1.0])
    wi = frame @ v
    lights = [np.array(L, np.float64) / np.linalg.norm(L) for L in P82['real_materials']['sphere']['lights']]
    wos = [frame @ L for L in lights]
    return mask, wi, wos


def realeval(out):
    R = P82['real_materials']
    d = real_desc()
    rng = np.random.default_rng(SD['real_dirs'])
    t0 = time.time()
    mm, wi, wo, q, use, phis, shape = DA.eval_style(d, rng)
    nw = DA.WALK['eval_initial']
    fA, fB = DA.ref_eval(d, mm, wi, wo, nw, SD['real_walks'], 0)
    ev = {'m': mm.astype(np.int32), 'wi': wi.astype(np.float32), 'wo': wo.astype(np.float32), 'q': q.astype(np.float32), 'use': use,
          'phis': phis.astype(np.float32), 'shape': shape}
    hist, rnd = [], 0
    while True:
        el1, ealb = DA.noise_floor(ev, fA, fB)
        s = DA.nf_summary(el1, ealb); s['walks'] = nw; hist.append(s)
        if s['PASS'] or nw >= 65536:
            break
        rnd += 1
        gA, gB = DA.ref_eval(d, mm, wi, wo, nw, SD['real_walks'], rnd)
        fA, fB = 0.5 * (fA + gA), 0.5 * (fB + gB); nw *= 2
    ev.update({'f': (0.5 * (fA + fB)).astype(np.float32), 'fA': fA.astype(np.float32), 'fB': fB.astype(np.float32)})
    np.savez(os.path.join(out, 'eval82_real.npz'), **ev)
    json.dump({'history': hist, 'PASS': hist[-1]['PASS'], 'walks': nw}, open(os.path.join(out, 'noise82_real.json'), 'w'), indent=1)
    hl = highlight_ref(d, phis[:, 0].astype(np.float64), SD['real_walks'] + 50000)
    np.savez(os.path.join(out, 'hl82_real.npz'), **hl)
    # sphere shading reference (direct lighting at the surface only: radiance = eval(view, light) per pixel; no GI)
    S = R['sphere']
    mask, wiS, wos = sphere_geom(S['res'])
    npx = len(wiS)
    img = np.zeros((len(d), len(wos), 2, npx, 3))
    for j, woS in enumerate(wos):
        ok = (woS[:, 2] > 1e-4) & (wiS[:, 2] > 1e-4)
        for m in range(len(d)):
            gA, gB = DA.ref_eval(d, np.full(ok.sum(), m), wiS[ok], woS[ok], S['walks'], SD['real_walks'] + 70000 + 100 * j + m, 0)
            img[m, j, 0, ok] = gA; img[m, j, 1, ok] = gB
    np.savez(os.path.join(out, 'sphere82_real.npz'), img=img.astype(np.float32), mask=mask, wi=wiS.astype(np.float32),
             wo=np.stack(wos).astype(np.float32))
    print('realeval', json.dumps(hist[-1]), time.time() - t0, 's', flush=True)


SHARD = None
for _a in sys.argv:
    if _a.startswith('--shard='):
        _i, _k = _a.split('=')[1].split('/'); SHARD = (int(_i), int(_k))

if __name__ == '__main__':
    out, w81, what = sys.argv[1], sys.argv[2], sys.argv[3]
    os.makedirs(out, exist_ok=True)
    DA.mi_init()
    if what == 'repro':
        repro(out, w81)
    elif what == 'fitset':
        fitset(out, w81, sys.argv[4])
    elif what == 'realeval':
        realeval(out)
    elif what == 'albset':
        albset(out, w81, sys.argv[4], SHARD[0], SHARD[1])
    elif what == 'albmerge':
        albmerge(out, sys.argv[4], int(sys.argv[5]))
    os._exit(0)
