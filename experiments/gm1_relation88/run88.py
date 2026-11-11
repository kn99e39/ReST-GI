"""WL88 job CLI (one GPU process per call; run88.sh pools them). Roots: R88 (this batch), R86 (frozen WL86: E1 / E2 references, masks).

  wscreen <R88>                           geometry-only screening of the W1 candidates -> w_screen.json, masks M_<name>
  wchoose <R88> <name>                    record the chosen W1, masks of W1 / W2 / held-out sources
  ref <R88> <cfg> <ER|P> <seed> <spp> <tag> [--ls=]      reference render (ER: pt88.render_fam = pt86 'cam' estimator + family channels)
  field <R88> <cfg> <seed> <spp> <tag> [--ls=]           oracle table (pt86.Ctx.field, two halves; output-representation ceiling)
  orders <R88> <cfg> <seed> <spp> [--ls=]                per-order truth fields 1..5 (two halves)
  tables <R88> <cfg> [--ls=]                             R0 / R1 / controls / budget variants / RAYSEND on one configuration
  real <R88> <cfg>                                       16 realizations of R0 and R1, converged R0 / R1 / sender relations, coverage
  render <R88> <cfg> <list> [--ls=]                      LOW + lookup renders of tables (WL86 estimator, 32-spp launches)
Configuration names: W1_T0 / W1_T1 / W2_T0 / W2_T1 (WL88 glossy-mover family), E2_T0 / E2_T1 (historical WL86 family).
"""
import json, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fx88  # noqa: E402
PT, Q = fx88.PT, fx88.Q
mi, dr, S = PT.mi, PT.dr, PT.S
FX = PT.FX
P88 = fx88.P88
E87 = fx88.fx87.P87['explicit']
RP = P88['relation']
NB, K = S.NB, S.K


def opts():
    o = {a.split('=')[0]: a.split('=')[1] for a in sys.argv[2:] if a.startswith('--') and '=' in a}
    a = [x for x in sys.argv[2:] if not x.startswith('--')]
    return o, a


def save(fn, **kw):
    os.makedirs(os.path.dirname(fn), exist_ok=True)
    np.savez_compressed(fn, **kw)


def jd(o, fn):
    os.makedirs(os.path.dirname(fn), exist_ok=True)
    json.dump(o, open(fn, 'w'), indent=1, default=lambda x: x.item() if hasattr(x, 'item') else str(x))


def inject_choice(R):
    """W1 := the chosen candidate (w_choice.json), W2 := the held-out edit, Wdev := the dev geometry."""
    fn = os.path.join(R, 'results', 'w_choice.json')
    ch = json.load(open(fn))['chosen'] if os.path.exists(fn) else None
    W = P88['workloads']
    if ch is not None:
        FX['W1'] = dict(FX[ch]); FX['W1']['role'] = 'W1 primary (candidate ' + ch + ')'
    FX['W2'] = dict(FX[P88['w2']]); FX['W2']['role'] = 'W2 held-out edit'
    return ch


def setup(R, cfg):
    inject_choice(R)
    fx88.setup(cfg)
    geom, ls0 = PT.config(cfg)
    return geom, ls0


# ------------------------------------------------------------------------------------------------ workload choice (geometry only)
def wscreen(R):
    """Geometry-only screening of every W1 candidate (protocol w_order): admissibility rule on mask pixel counts; masks M_<name>.npz."""
    import masks88
    rule = P88['w_rule']
    rows = {}
    out = os.path.join(R, 'results', 'masks'); os.makedirs(out, exist_ok=True)
    for name in P88['w_order']:
        ex = FX[name]
        lsets = [ex['light_set'], 'LMIX88', 'LSPOT88']
        gok = bool(S.geom_ok(ex['T0']) and S.geom_ok(ex['T1']) and all(S.light_ok(l['pos'], ex[t]) for ls in lsets for l in FX['light_sets'][ls] for t in ('T0', 'T1')))
        m, aux = masks88.compute(name)
        cnt = {k: int(v.sum()) for k, v in m.items()}
        ok = gok and all(cnt[k] >= rule['min_pixels'] for k in rule['regions'])
        rows[name] = {'counts': cnt, 'geometry_ok': gok, 'admissible': bool(ok)}
        np.savez_compressed(os.path.join(out, f'M_{name}.npz'), **m, **aux)
        print('W candidate', name, json.dumps(cnt), ok, flush=True)
    jd({'candidates': rows, 'rule': rule, 'order': P88['w_order'], 'note': 'geometry / visibility only; no radiance of any candidate exists'},
       os.path.join(R, 'results', 'w_screen.json'))
    print('W_GEOMETRY_ADMISSIBLE', ' '.join(n for n in P88['w_order'] if rows[n]['admissible']), flush=True)


def wchoose(R, name):
    """Record the chosen W1 candidate and compute the masks of W1, the held-out edit W2 and the held-out sources."""
    import masks88
    import shutil
    out = os.path.join(R, 'results', 'masks')
    jd({'chosen': name, 'note': 'first geometry-admissible candidate (protocol w_order) admitted by the qualification-partition criteria'},
       os.path.join(R, 'results', 'w_choice.json'))
    shutil.copy(os.path.join(out, f'M_{name}.npz'), os.path.join(out, 'W1.npz'))
    inject_choice(R)
    for nm, ls in [('W2', None), ('W1', 'LMIX88'), ('W1', 'LSPOT88')]:
        m, aux = masks88.compute(nm, ls)
        tag = nm + ('_' + ls if ls else '')
        np.savez_compressed(os.path.join(out, f'{tag}.npz'), **m, **aux); jd({k: int(v.sum()) for k, v in m.items()}, os.path.join(out, f'{tag}.json'))
        print('masks', tag, json.dumps({k: int(v.sum()) for k, v in m.items() if k in ('C_ss', 'R_mirror', 'D', 'A', 'stationary_all')}), flush=True)


# ------------------------------------------------------------------------------------------------ references / truth
def ref(R, cfg, kind, seed, spp, tag, ls=None):
    import pt88
    geom, ls0 = setup(R, cfg)
    ctx = PT.Ctx(geom); lt = PT.Lights(FX['light_sets'][ls or ls0])
    out = os.path.join(R, 'results', 'renders')
    if kind == 'ER':
        o, T_ = pt88.render_fam(ctx, lt, FX['camera'], FX['camera']['res'], spp, seed)
    else:
        acc, T_ = ctx.render(lt, kind, FX['camera'], FX['camera']['res'], spp, seed, 'cam', b=32)
        o = PT.pack(acc, FX['camera']['res'], spp)
    save(os.path.join(out, f'{tag}.npz'), **o)
    jd({'cfg': cfg, 'ls': ls or ls0, 'ref': kind, 'seed': seed, 'spp': spp, 'timing': T_}, os.path.join(out, f'{tag}.json'))
    print(tag, json.dumps(T_), flush=True)


def field(R, cfg, seed, spp, tag, ls=None):
    geom, ls0 = setup(R, cfg)
    ctx = PT.Ctx(geom); lt = PT.Lights(FX['light_sets'][ls or ls0])
    fa, ca = ctx.field(lt, spp // 2, seed)
    fb, cb = ctx.field(lt, spp // 2, seed + 500)
    save(os.path.join(R, 'results', 'fields', f'{tag}.npz'), fa=fa, fb=fb, ca=ca, cb=cb, B=(0.5 * (fa + fb)).astype(np.float32))
    print(tag, 'field done', flush=True)


def orders(R, cfg, seed, spp, ls=None):
    import pt88
    geom, ls0 = setup(R, cfg)
    ctx = PT.Ctx(geom); lt = PT.Lights(FX['light_sets'][ls or ls0])
    a = pt88.field_orders(ctx, lt, spp // 2, seed)
    b = pt88.field_orders(ctx, lt, spp // 2, seed + 500)
    tag = cfg + ('_' + ls if ls else '')
    save(os.path.join(R, 'results', 'fields', f'ORD_{tag}.npz'), a=a.astype(np.float32), b=b.astype(np.float32))
    print('orders', tag, flush=True)


# ------------------------------------------------------------------------------------------------ relations
_CACHE = {}


def materials(ctx):
    """Material-only, geometry-independent pieces (built once per process): WL87 dense 8 x 8 table and the R1 dense 16 x 16 table."""
    import rel88 as RL
    fam = fx88.FAMILY['name']
    if fam not in _CACHE:
        q0 = Q.Quad(ctx.u_state_np, 64, E87['seed_ray'], E87['M_dense'], E87['seed_dense'])
        _CACHE[fam] = {'dense8': q0.dense(), 'dense16': RL.Dense(RP['dense_sub'], RP['M_dense'], RP['seed_dense'])}
    return _CACHE[fam]


def quad_of(us, ri):
    return Q.Quad(us, E87['M_ray'], E87['seed_ray'] + 1000 * ri)


def raysend(ctx, lt, rel, st):
    """Exact once-scattered radiance arriving along every u ray [N, K, 3] and every valid area sample [n, 3] (towards the origin)."""
    import torch
    import rel88 as RL

    def exact(o, e):
        n = len(o)
        if n == 0:
            return np.zeros((0, 3), np.float32)
        d = e - o; d /= np.linalg.norm(d, axis=1, keepdims=True)
        ray = mi.Ray3f(mi.Point3f(*[PT.fl(o[:, k]) for k in range(3)]), mi.Vector3f(*[PT.fl(d[:, k]) for k in range(3)]))
        sd = dr.opaque(mi.UInt64, 1)
        rng = mi.PCG32(size=n, initstate=sd, initseq=mi.UInt64(0))
        out, _ = ctx.trace(ray, n, rng, sd, 'ER', 1, lt, 'cam')
        vals = [out[f'e1{c}'] for c in range(3)]
        dr.eval(vals); dr.sync_thread()
        return np.stack([np.asarray(v, np.float32) for v in vals], 1)
    idx = np.nonzero(st.hit >= 0)[0]
    Lr = np.zeros((st.hit.size, 3), np.float32)
    Lr[idx] = exact(st.seg[idx, :3].astype(np.float64), st.seg[idx, 3:].astype(np.float64))
    Lu = torch.as_tensor(Lr, device=RL.dev).view(ctx.N, K, 3)
    if rel.m:
        v = rel.A['valid'].ravel(); sg = rel.seg_t.cpu().numpy().reshape(-1, 6)[v].astype(np.float64)
        La = torch.as_tensor(exact(sg[:, :3], sg[:, 3:]), device=RL.dev)
    else:
        La = None
    return rel.first_from_rays(Lu, La)


def tables(R, cfg, ls=None):
    import torch
    import ecro86 as EC
    import xport87 as X
    import rel88 as RL
    geom, ls0 = setup(R, cfg)
    ls = ls or ls0
    tag = cfg + ('' if ls == ls0 else '_' + ls)
    ctx = PT.Ctx(geom); lt = PT.Lights(FX['light_sets'][ls])
    mat = materials(ctx)
    T = {}
    t0 = time.perf_counter(); quad = Q.Quad(ctx.u_state_np, E87['M_ray'], E87['seed_ray']); T['quad_s'] = time.perf_counter() - t0
    B0, o0, q, st, T['R0_first'] = X.build(ctx, lt, quad, E87['P'], E87['Mq'], E87['seed_q'])
    out = {'q': q, 'R0': B0}
    for j, o in enumerate(o0):
        out[f'R0_L{j + 2}'] = o
    stats = {}
    rels = {}
    for name, kap, m in [('R1', RP['kappa_u'], RP['m']), ('R1M', RP['matched_variant']['kappa_u'], RP['matched_variant']['m']),
                         ('R1m1', RP['kappa_u'], 1), ('R1m16', RP['kappa_u'], 16)]:
        Tm = {}
        rel = RL.SAM(ctx, quad, mat['dense16'], kap, m, RP['seed_area'], st=st, timing=Tm)
        B, ords = RL.recurrence([rel], q)
        out[name] = B.cpu().numpy()
        if name == 'R1':
            for j, o in enumerate(ords):
                out[f'R1_L{j + 2}'] = o.cpu().numpy()
        rels[name] = rel
        stats[name] = {'kappa_u': kap, 'm': m, 'pairs': int(len(rel.pi)), 'area_valid': rel.n_area_valid, 'missing_pairs': rel.missing,
                       'rays': rel.rays(), 'bytes': rel.nbytes(), 'timing_first': Tm, 'digest': rel.digest()}
    # RAYSEND (exact sender radiance in the first gather)
    out['RS1'] = RL.recurrence([rels['R1']], q, first=raysend(ctx, lt, rels['R1'], st))[0].cpu().numpy()
    r0op = X.XOp(ctx.G, st.hit, st.dbin, quad.W)
    out['RS0'] = RL.recurrence([r0op], q, first=raysend(ctx, lt, RL.SAM(ctx, quad, mat['dense16'], RP['kappa_u'], 0, RP['seed_area'], st=st), st))[0].cpu().numpy()
    # Task A controls
    out['R0D'] = RL.recurrence([X.XOp(ctx.G, st.hit, st.dbin, RL.dense_W(quad, mat['dense8']))], q)[0].cpu().numpy()
    ops = []
    for ri, rs in enumerate(P88['stage2']['seeds'][:4]):
        us = X.realization_us(rs); X.set_realization(ctx, us)
        s2 = EC.build_state(ctx)
        ops.append(X.XOp(ctx.G, s2.hit, s2.dbin, quad_of(us, ri + 1).W))
    X.set_realization(ctx, X.realization_us(86001))
    out['R0F'] = RL.fresh_recurrence(ops, q)[0].cpu().numpy()
    out['POOL2'] = RL.recurrence([r0op, ops[0]], q)[0].cpu().numpy()
    out['POOL4'] = RL.recurrence([r0op] + ops[:3], q)[0].cpu().numpy()
    del rels, ops
    torch.cuda.empty_cache()
    X.sync(); t0 = time.perf_counter()
    B3, _ = X.sender_side(ctx, q, mat['dense8'], ctx.N * K, E87['seed_c3'])
    X.sync(); T['C3_s'] = time.perf_counter() - t0
    out['C3'] = B3
    ks = P88['stage1']['kappa_s']
    X.sync(); t0 = time.perf_counter()
    sr = RL.SendRel(ctx, ks, P88['stage1']['seed_send'], mat['dense8'])
    out['C3UR'] = RL.recurrence([sr], q)[0].cpu().numpy()
    X.sync(); T['C3UR_s'] = time.perf_counter() - t0
    L = torch.as_tensor(np.asarray(q, np.float32), device=RL.dev); acc = 0
    for j in range(RL.J):
        L = RL.SendRel(ctx, ks, P88['stage1']['seed_send'] + 17 * (j + 1), mat['dense8']).apply(L); acc = acc + L
    out['C3U'] = acc.cpu().numpy()
    stats['C3UR'] = {'rays': sr.n_rays}
    save(os.path.join(R, 'results', 'tables', f'X_{tag}.npz'), **{k: np.asarray(v, np.float32) for k, v in out.items()})
    meta = {'cfg': cfg, 'ls': ls, 'timing': T, 'relations': stats, 'peaked': RL.peaked()[1], 'state_digest': EC.State.digest(st),
            'finite': bool(all(np.isfinite(v).all() for v in out.values())), 'min': {k: float(np.min(v)) for k, v in out.items()}}
    jd(meta, os.path.join(R, 'results', 'tables', f'X_{tag}.json'))
    print('tables', tag, json.dumps({k: v for k, v in meta.items() if k in ('finite', 'state_digest')}), flush=True)


def pair_energy(ctx, rel_pairs, q, m, seed):
    """Converged pair-resolved irradiance from peaked senders: pure area estimator (no MIS) with m samples per pair -> [n_pairs, 3]."""
    import rel88 as RL
    pi, pp = rel_pairs
    A = RL.trace_area(ctx, pi, pp, m, 1, seed)
    # w = cos_x / (1 / DOMEGA + m d^2 / (A cos_y)) -> recover the pure area weight cos_x A cos_y / (m d^2) from the stored segment
    sg = A['seg'].astype(np.float64)
    G = ctx.G
    x, y = sg[..., :3], sg[..., 3:]
    d = y - x; dist = np.linalg.norm(d, axis=-1); w = d / np.maximum(dist, 1e-12)[..., None]
    nx = G.PN[pi][:, None, :]
    ny = np.broadcast_to(G.PN[pp][:, None, :], y.shape).copy()
    sph = G.PSID[pp] == S.SID_SPHERE
    ny[sph] = (y[sph] - G.SC) / G.SR
    cx = (nx * w).sum(-1); cy = -(ny * w).sum(-1)
    g = np.where(A['valid'], cx * cy * G.PAREA[pp][:, None] / np.maximum(dist ** 2, 1e-12) / m, 0.0)
    Lp = q[pp[:, None], A['dbin'].astype(np.int64)]
    return (g[..., None] * Lp).sum(1)


def real(R, cfg):
    import torch
    import ecro86 as EC
    import xport87 as X
    import rel88 as RL
    geom, ls = setup(R, cfg)
    ctx = PT.Ctx(geom); lt = PT.Lights(FX['light_sets'][ls])
    mat = materials(ctx)
    G = ctx.G
    q = Q.qtable(ctx, lt, E87['P'], E87['Mq'], E87['seed_q'])
    S2 = P88['stage2']
    # converged pair energies (importance of (flat receiver, peaked sender) relations)
    pk, _ = RL.peaked()
    pi, pp = RL.pair_list(G, pk)
    Ep = pair_energy(ctx, (pi, pp), q, S2['pair_m'], S2['pair_seed'])
    El = Ep.mean(1)
    tot_i = np.bincount(pi, weights=El, minlength=G.N)
    imp = El >= S2['important_share'] * tot_i[pi]
    keys = pi * G.N + pp
    out = {'pair_i': pi, 'pair_p': pp, 'pair_E': Ep.astype(np.float32), 'pair_imp': imp}
    cov = {}
    seeds = [86001] + list(S2['seeds'])
    ops0, ops1 = [], []
    for ri, rs in enumerate(seeds):
        us = X.realization_us(rs); X.set_realization(ctx, us)
        st = EC.build_state(ctx)
        quad = quad_of(us, ri) if rs != 86001 else Q.Quad(us, E87['M_ray'], E87['seed_ray'])
        r0 = X.XOp(G, st.hit, st.dbin, quad.W)
        r1 = RL.SAM(ctx, quad, mat['dense16'], RP['kappa_u'], RP['m'], RP['seed_area'] + 1000 * ri, st=st)
        tg = f's{rs}'
        out[f'R0_{tg}'] = RL.recurrence([r0], q)[0].cpu().numpy()
        out[f'R1_{tg}'] = RL.recurrence([r1], q)[0].cpu().numpy()
        if rs != 86001:
            ops0.append(r0); ops1.append(r1)
        # coverage of (flat receiver, peaked sender) relations
        hit = st.hit.astype(np.int64); rid = np.arange(hit.size) // K
        hk = np.unique(rid[hit >= 0] * G.N + hit[hit >= 0])
        s0 = np.isin(keys, hk)
        sa = s0 | r1.A['valid'].any(1)
        cov[tg] = {'R0_energy_cov': float(El[s0].sum() / El.sum()), 'R1_energy_cov': float(El[sa].sum() / El.sum()),
                   'R0_imp_missed': float((imp & ~s0).sum() / max(imp.sum(), 1)), 'R1_imp_missed': float((imp & ~sa).sum() / max(imp.sum(), 1)),
                   'R0_imp_missed_energy': float(El[imp & ~s0].sum() / max(El[imp].sum(), 1e-30)),
                   'R1_imp_missed_energy': float(El[imp & ~sa].sum() / max(El[imp].sum(), 1e-30))}
        print('real', cfg, rs, json.dumps(cov[tg]), flush=True)
    out['R0_POOL16'] = RL.recurrence(ops0, q)[0].cpu().numpy()
    out['R1_POOL16'] = RL.recurrence(ops1, q)[0].cpu().numpy()
    # converged relations (physical-target identity): R0 x 64 realizations, R1 m = 16 x 16 realizations, C3U kappa_s 64 fresh
    for ri, rs in enumerate(S2['inf_seeds']):
        us = X.realization_us(rs); X.set_realization(ctx, us)
        st = EC.build_state(ctx)
        ops0.append(X.XOp(G, st.hit, st.dbin, quad_of(us, 100 + ri).W))
    out['R0_INF'] = RL.recurrence(ops0, q)[0].cpu().numpy()
    del ops0
    ops1 = ops1[:0]
    for ri, rs in enumerate(seeds[1:17]):
        us = X.realization_us(rs); X.set_realization(ctx, us)
        st = EC.build_state(ctx)
        ops1.append(RL.SAM(ctx, quad_of(us, ri + 1), mat['dense16'], RP['kappa_u'], 16, RP['seed_area'] + 500 + ri, st=st))
    out['R1_INF'] = RL.recurrence(ops1, q)[0].cpu().numpy()
    del ops1
    torch.cuda.empty_cache()
    L = torch.as_tensor(np.asarray(q, np.float32), device=RL.dev); acc = 0
    for j in range(RL.J):
        Lj = 0
        for r in range(S2['send_inf_reps']):
            Lj = Lj + RL.SendRel(ctx, S2['send_inf_kappa'], S2['send_inf_seed'] + 101 * j + r, mat['dense8']).apply(L)
        L = Lj / S2['send_inf_reps']; acc = acc + L
    out['C3U_INF'] = acc.cpu().numpy()
    save(os.path.join(R, 'results', 'tables', f'R_{cfg}.npz'), **{k: (v.astype(np.float32) if v.dtype.kind == 'f' else v) for k, v in out.items()})
    jd({'coverage': cov, 'pairs': int(len(pi)), 'important_pairs': int(imp.sum())}, os.path.join(R, 'results', 'tables', f'R_{cfg}.json'))
    print('real done', cfg, flush=True)


# ------------------------------------------------------------------------------------------------ renders
def render(R, cfg, listfn, ls=None):
    geom, ls0 = setup(R, cfg)
    ctx = PT.Ctx(geom); lt = PT.Lights(FX['light_sets'][ls or ls0])
    cam = FX['camera']
    out = os.path.join(R, 'results', 'renders'); os.makedirs(out, exist_ok=True)
    cache = {}
    for line in open(listfn):
        if not line.strip():
            continue
        src, seed, spp, tag = line.split()
        if os.path.exists(os.path.join(out, f'{tag}.npz')):
            continue
        fn, key = src.split(':')
        if fn not in cache:
            cache = {fn: np.load(os.path.join(R, fn) if not fn.startswith('/') else fn)}
        B = cache[fn][key].astype(np.float32)
        acc, T_ = ctx.render(lt, 'ER', cam, cam['res'], int(spp), int(seed), 'ecro', B=PT.fl(B), b=32)
        o = PT.pack(acc, cam['res'], int(spp))
        np.savez_compressed(os.path.join(out, f'{tag}.npz'), hl=o['hl'])
        jd({'cfg': cfg, 'src': src, 'seed': int(seed), 'spp': int(spp), 'timing': T_}, os.path.join(out, f'{tag}.json'))
        print(tag, round(T_['wall_s'], 2), flush=True)


def main():
    dr.set_flag(dr.JitFlag.KernelHistory, True)
    cmd = sys.argv[1]
    o, a = opts()
    if cmd == 'wscreen':
        wscreen(a[0])
    elif cmd == 'wchoose':
        wchoose(a[0], a[1])
    elif cmd == 'ref':
        ref(a[0], a[1], a[2], int(a[3]), int(a[4]), a[5], o.get('--ls'))
    elif cmd == 'field':
        field(a[0], a[1], int(a[2]), int(a[3]), a[4], o.get('--ls'))
    elif cmd == 'orders':
        orders(a[0], a[1], int(a[2]), int(a[3]), o.get('--ls'))
    elif cmd == 'tables':
        tables(a[0], a[1], o.get('--ls'))
    elif cmd == 'real':
        real(a[0], a[1])
    elif cmd == 'render':
        render(a[0], a[1], a[2], o.get('--ls'))


if __name__ == '__main__':
    main()
    os._exit(0)
