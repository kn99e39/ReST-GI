"""WL93A Task A: physical once-scattered target q - independent realizations over a budget ladder, reliability analysis.

  python qref93a.py cfgs   <R> <R86> <R92A>                                 configuration list -> results/configs.json
  python qref93a.py refgen <R> <R86> <R92A> <B> <cfg_start> <cfg_count>    evaluation-reference realizations (4 per budget)
  python qref93a.py tgen   <R> <R86> <R92A> <B> <train|val>                 training / training-validation targets at budget B
  python qref93a.py taskA  <R> <R86> <R92A> <R93>                           Task A analysis -> results/taskA.json; prints TASKA ...
"""
import glob, json, math, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for d in ('ecro86', 'ecro_attr92', 'ecro_own92a', 'ecro_cf92b', 'ecro_stage93'):
    sys.path.insert(0, os.path.join(HERE, '..', d))
sys.path.insert(0, HERE)
DEV = '--dev' in sys.argv
P = json.load(open(os.path.join(HERE, 'protocol', 'ecro_stage93a.json')))
BUD = P['target']['budgets']
BNAMES = list(BUD)
NREAL = P['target']['realizations']
SEED_REF, SEED_TR, SEED_TV = 93_100_000, 93_500_000, 93_600_000


def jd(o, fn):
    json.dump(o, open(fn, 'w'), indent=1)


def jl(fn):
    return json.load(open(fn))


def budget(b):
    x = BUD[b]
    return (x['P'], 64 if DEV else x['Mq'])


# ------------------------------------------------------------------------------------------------ configurations
def train_population(R86, R92A, sp):
    ex = {n for n, _, _ in jl(os.path.join(R92A, 'results', 'exclusion.json'))['excluded'].get(sp, [])}
    return [f for f in sorted(glob.glob(os.path.join(R86, 'data', sp, 'g*.npz'))) if os.path.basename(f) not in ex
            and os.path.exists(os.path.join(R92A, 'data', sp + '_f2', os.path.basename(f)))]


def cfgs(R, R86, R92A):
    tr = train_population(R86, R92A, 'train')
    sub = tr[::7][:128]
    rep = sub[::8]
    val = train_population(R86, R92A, 'val')
    if DEV:
        sub, rep, val = sub[:24], sub[:24][::8], val[:6]
    C = [{'cfg': i, 'kind': 'train_rep', 'file': os.path.basename(f)} for i, f in enumerate(rep)]
    C += [{'cfg': len(rep) + i, 'kind': 'val', 'file': os.path.basename(f)} for i, f in enumerate(val)]
    C += [{'cfg': len(C) + i, 'kind': 'sharp', 'tag': t} for i, t in enumerate(('E2_T0', 'E2_T1'))]
    out = {'train_subset': [os.path.basename(f) for f in sub], 'train_representative': [os.path.basename(f) for f in rep],
           'validation': [os.path.basename(f) for f in val], 'configs': C, 'train_population_size': len(tr)}
    jd(out, os.path.join(R, 'results', 'configs.json'))
    print('CFGS', len(sub), len(rep), len(val), len(C), flush=True)


def cfg_scene(R86, c):
    """-> (geom, [light-set ...], file index or None)."""
    import fx92
    if c['kind'] == 'sharp':
        geom, ls, e, cfg = fx92.cfg_ls(c['tag'])
        return geom, [fx92.FX['light_sets'][ls]], None
    sp = 'train' if c['kind'] == 'train_rep' else 'val'
    z = np.load(os.path.join(R86, 'data', sp, c['file']))
    return json.loads(str(z['geom'])), json.loads(str(z['lights'])), int(c['file'][1:6])


def ref_seed(cfg, b, r, li):
    return SEED_REF + 1000 * cfg + 40 * BNAMES.index(b) + 4 * r + li


def refgen(R, R86, R92A, b, start, count):
    import fx92
    import quad87 as Q
    PT = fx92.PT
    Pp, Mq = budget(b)
    C = jl(os.path.join(R, 'results', 'configs.json'))['configs']
    out = os.path.join(R, 'data', 'qref'); os.makedirs(out, exist_ok=True)
    ctx = None; t0 = time.time(); n = 0
    for c in C[int(start):int(start) + int(count)]:
        fn = os.path.join(out, f"c{c['cfg']:03d}_{b}.npz")
        if os.path.exists(fn):
            continue
        geom, lights, _ = cfg_scene(R86, c)
        if ctx is None:
            ctx = PT.Ctx(geom)
        else:
            ctx.set_geom(geom)
        q = np.stack([np.stack([Q.qtable(ctx, PT.Lights(l), Pp, Mq, ref_seed(c['cfg'], b, r, li)) for li, l in enumerate(lights)]) for r in range(NREAL)])
        np.savez(fn + '.tmp.npz', q=q.astype(np.float32)); os.replace(fn + '.tmp.npz', fn); n += 1
    print('REFGEN_DONE', b, start, n, round(time.time() - t0, 1), 's', flush=True)


def tgen(R, R86, R92A, b, split):
    import fx92
    import quad87 as Q
    PT = fx92.PT
    Pp, Mq = budget(b)
    cf = jl(os.path.join(R, 'results', 'configs.json'))
    names = cf['train_subset'] if split == 'train' else cf['validation']
    base = SEED_TR if split == 'train' else SEED_TV
    out = os.path.join(R, 'data', split + '_q'); os.makedirs(out, exist_ok=True)
    ctx = None; t0 = time.time()
    for name in names:
        z = np.load(os.path.join(R86, 'data', split, name))
        geom = json.loads(str(z['geom'])); lights = json.loads(str(z['lights'])); idx = int(name[1:6])
        if ctx is None:
            ctx = PT.Ctx(geom)
        else:
            ctx.set_geom(geom)
        q = np.stack([Q.qtable(ctx, PT.Lights(l), Pp, Mq, base + 4 * idx + li) for li, l in enumerate(lights)])
        np.savez_compressed(os.path.join(out, name + '.tmp.npz'), q=q.astype(np.float32), budget=b)
        os.replace(os.path.join(out, name + '.tmp.npz'), os.path.join(out, name))
    print('TGEN_DONE', split, b, len(names), round(time.time() - t0, 1), 's', flush=True)


# ------------------------------------------------------------------------------------------------ metrics (shared with eval93a)
PAIRINGS = [((0, 1), (2, 3)), ((0, 2), (1, 3)), ((0, 3), (1, 2))]


def stat(X, Q4, mask):
    """Per-configuration sums of X [L, N, NB, 3] against four independent realizations Q4 [4, L, N, NB, 3] on patches `mask`."""
    X = X[:, mask].astype(np.float64); Q4 = Q4[:, :, mask].astype(np.float64)
    qm = Q4.mean(0)
    s = {}
    for p, (a, b) in enumerate(PAIRINGS):
        qa, qb = Q4[list(a)].mean(0), Q4[list(b)].mean(0)
        s[f'num{p}'] = float(((X - qa) * (X - qb)).sum()); s[f'den{p}'] = float((qa * qb).sum())
    s['l1'] = float(np.abs(X - qm).sum()); s['sq'] = float(qm.sum())
    s['floor_l1'] = float(0.5 * np.abs(Q4[[0, 1]].mean(0) - Q4[[2, 3]].mean(0)).sum())
    FX, FQ = X.sum((2, 3)), qm.sum((2, 3))                                    # [L, n] patch flux
    s['lvl'] = float(np.abs(FX - FQ).sum())
    with np.errstate(invalid='ignore', divide='ignore'):
        dx = X / np.where(FX > 0, FX, 1)[..., None, None]; dq = qm / np.where(FQ > 0, FQ, 1)[..., None, None]
    s['dir'] = float((FQ * 0.5 * np.abs(dx - dq).sum((2, 3))).sum())
    k = qm.sum(3).argmax(2)                                                    # [L, n] peak bin of the reference mean
    take = lambda A: np.take_along_axis(A.sum(3), k[..., None], 2)[..., 0]
    s['peak_x'] = float(take(X).sum()); s['peak_q'] = float(take(qm).sum())
    return s


def noise_stat(Q4, mask, Qref4):
    """Across-realization noise sums at a budget (unbiased variance) with the B5 reference norm."""
    Q4 = Q4[:, :, mask].astype(np.float64); R4 = Qref4[:, :, mask].astype(np.float64)
    var = Q4.var(0, ddof=1).sum()
    l1 = np.mean([np.abs(Q4[i] - Q4[j]).sum() for i in range(4) for j in range(i + 1, 4)])
    return {'var': float(var), 'den': float((R4[[0, 1]].mean(0) * R4[[2, 3]].mean(0)).sum()), 'l1pair': float(l1), 'sq': float(R4.mean(0).sum())}


def pooled(S):
    """Pooled metrics from a list of per-configuration stat dicts."""
    t = {k: sum(s[k] for s in S) for k in S[0]}
    Ms = [math.sqrt(max(t[f'num{p}'], 0.0) / max(t[f'den{p}'], 1e-30)) for p in range(3)]
    return {'M': Ms[0], 'M_pairings': Ms, 'U': max(Ms) - min(Ms), 'rel_l1': t['l1'] / t['sq'], 'ref_floor_l1': t['floor_l1'] / t['sq'],
            'flux_err': t['lvl'] / t['sq'], 'dir_err': t['dir'] / t['sq'], 'peak_retention': t['peak_x'] / max(t['peak_q'], 1e-30)}


def M_of(S, idx, p=0):
    num = sum(S[i][f'num{p}'] for i in idx); den = sum(S[i][f'den{p}'] for i in idx)
    return math.sqrt(max(num, 0.0) / max(den, 1e-30))


def boot(S, n=2000, seed=93001, other=None):
    """Geometry bootstrap of M (or of M(S) - M(other), paired)."""
    rng = np.random.default_rng(seed); m = len(S); v = []
    for _ in range(n):
        idx = rng.integers(0, m, m)
        v.append(M_of(S, idx) - (M_of(other, idx) if other is not None else 0.0))
    v = np.array(v)
    return {'se': float(v.std(ddof=1)), 'ci95': [float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))]}


def group(C, kind):
    return [c['cfg'] for c in C if c['kind'] == kind]


def load_ref(R, cfg, b):
    return np.load(os.path.join(R, 'data', 'qref', f'c{cfg:03d}_{b}.npz'))['q']


# ------------------------------------------------------------------------------------------------ Task A analysis
def taskA(R, R86, R92A, R93):
    import eval93a as EV
    cf = jl(os.path.join(R, 'results', 'configs.json')); C = cf['configs']
    pm = {c['cfg']: EV.panel_mask(R86, c) for c in C}
    REF = 'B5'
    res = {'budgets': BUD, 'groups': {}, 'per_cfg': {}}
    models = ['enc', 'own_s8601', 'own_s8602']
    for gname in ('train_rep', 'val', 'sharp'):
        ids = group(C, gname)
        noise = {b: [] for b in BNAMES}; meanst = {b: [] for b in BNAMES}; mst = {m: [] for m in models}; w93 = {'enc': [0.0, 0.0], 'own_s8601': [0.0, 0.0], 'own_s8602': [0.0, 0.0], 'q93_sq': 0.0}
        for i in ids:
            Qr = load_ref(R, i, REF)
            for b in BNAMES:
                Qb = Qr if b == REF else load_ref(R, i, b)
                noise[b].append(noise_stat(Qb, pm[i], Qr))
                meanst[b].append(stat(Qb.mean(0), Qr, pm[i]))
            X = EV.load_stage(R, i)
            for m in models:
                mst[m].append(stat(X[m], Qr, pm[i]))
            if gname == 'val' and R93:                                     # WL93 stored q (B0, WL93 seeds) for the same configuration
                c = C[i]; q93 = np.load(os.path.join(R93, 'data', 'val_q', c['file']))['q'].astype(np.float64)[:, pm[i]]
                for m in models:
                    w93[m][0] += float(np.abs(X[m][:, pm[i]].astype(np.float64) - q93).sum())
                w93['q93_sq'] += float(q93.sum())
            ns = noise_stat(Qr, pm[i], Qr)
            res['per_cfg'][str(i)] = {'kind': gname, 'id': C[i].get('file', C[i].get('tag')), 'N_B5': math.sqrt(ns['var'] / max(ns['den'], 1e-30)),
                                      'N_B0': math.sqrt(noise[BNAMES[0]][-1]['var'] / max(noise[BNAMES[0]][-1]['den'], 1e-30)),
                                      'M': {m: M_of([mst[m][-1]], [0]) for m in models}, 'num0': {m: mst[m][-1]['num0'] for m in models}, 'den0': mst['enc'][-1]['den0']}
        g = {'n_cfg': len(ids), 'noise': {}, 'budget_mean_vs_B5': {}, 'discrepancy': {}}
        for b in BNAMES:
            t = {k: sum(s[k] for s in noise[b]) for k in noise[b][0]}
            N = math.sqrt(t['var'] / t['den'])
            pm_ = pooled(meanst[b])
            g['noise'][b] = {'N_single_rel_l2': N, 'pair_rel_l1': t['l1pair'] / t['sq'], 'N_bootstrap_se': None}
            g['budget_mean_vs_B5'][b] = {'M_of_mean': pm_['M'], 'bias_estimate': math.sqrt(max(pm_['M'] ** 2 - N ** 2 / NREAL, 0.0)) if b != REF else 0.0}
        for m in models:
            g['discrepancy'][m] = {**pooled(mst[m]), 'boot': boot(mst[m])}
        if gname == 'val' and R93:
            g['wl93_q_comparison'] = {m: {'rel_l1_vs_wl93_q': w93[m][0] / w93['q93_sq'], 'rel_l1_vs_B5_mean': g['discrepancy'][m]['rel_l1']} for m in models}
        # tail: top-5 shares of the pooled numerator (enc, own)
        tail = {}
        for m in models:
            v = sorted(((res['per_cfg'][str(i)]['num0'][m], res['per_cfg'][str(i)]['id']) for i in ids), reverse=True)
            tot = sum(x for x, _ in v)
            tail[m] = [{'id': n_, 'share': x / tot if tot else None} for x, n_ in v[:5]]
        g['tail_top5'] = tail
        # predeclared high-confidence subgroup (per-geometry B5 noise <= 0.10)
        hc = [k for k, i in enumerate(ids) if res['per_cfg'][str(i)]['N_B5'] <= 0.10]
        g['high_confidence_subgroup'] = {'n_cfg': len(hc), 'ids': [res['per_cfg'][str(ids[k])]['id'] for k in hc],
                                         'discrepancy': {m: pooled([mst[m][k] for k in hc]) for m in models} if hc else None}
        res['groups'][gname] = g
    V = res['groups']['val']; TRp = res['groups']['train_rep']
    # H_TARGET
    if 'wl93_q_comparison' in V:
        e = V['wl93_q_comparison']['enc']; d = abs(e['rel_l1_vs_wl93_q'] - e['rel_l1_vs_B5_mean']); fl = V['discrepancy']['enc']['ref_floor_l1']
        res['H_TARGET'] = {'enc_vs_wl93_q': e['rel_l1_vs_wl93_q'], 'enc_vs_B5_mean': e['rel_l1_vs_B5_mean'], 'abs_change': d, 'B5_mean_floor': fl,
                           'label': 'SUPPORTED' if (d >= 0.05 and fl <= 0.02) else ('NOT SUPPORTED' if d < 0.02 else 'INCONCLUSIVE')}
    # training budget rule (train-representative panel)
    tb = next((b for b in BNAMES[:-1] if TRp['noise'][b]['N_single_rel_l2'] <= 0.10), None)
    if tb is None and TRp['noise'][REF]['N_single_rel_l2'] <= 0.20:
        tb = REF
    own = [V['discrepancy'][f'own_s{s}'] for s in (8601, 8602)]
    delta = 0.10 * float(np.mean([o['M'] for o in own]))
    rel = tb is not None and all(o['U'] <= delta / 3 for o in own)
    res['TARGET_RELIABLE'] = {'train_budget': tb, 'delta': delta, 'U_own': [o['U'] for o in own], 'U_bound': delta / 3, 'RELIABLE': bool(rel)}
    jd(res, os.path.join(R, 'results', 'taskA.json'))
    print('TASKA', 'RELIABLE' if rel else 'UNRELIABLE', 'train_budget', tb, 'delta', round(delta, 4), 'U_own', [round(o['U'], 4) for o in own],
          'H_TARGET', res.get('H_TARGET', {}).get('label'), flush=True)
    print('NOISE train_rep', json.dumps({b: round(TRp['noise'][b]['N_single_rel_l2'], 4) for b in BNAMES}), 'val', json.dumps({b: round(V['noise'][b]['N_single_rel_l2'], 4) for b in BNAMES}), flush=True)
    print('DISCREPANCY val', json.dumps({m: round(V['discrepancy'][m]['M'], 4) for m in models}), flush=True)


if __name__ == '__main__':
    import fx92
    fx92.dr.set_flag(fx92.dr.JitFlag.KernelHistory, True)
    cmd, R, R86, R92A = sys.argv[1:5]
    a = [x for x in sys.argv[5:] if not x.startswith('--')]
    os.makedirs(os.path.join(R, 'results'), exist_ok=True)
    {'cfgs': cfgs, 'refgen': refgen, 'tgen': tgen, 'taskA': taskA}[cmd](R, R86, R92A, *a)
    os._exit(0)
