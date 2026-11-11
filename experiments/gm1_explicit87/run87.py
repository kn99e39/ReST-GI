"""WL87 job CLI (one GPU process per call; the chain run87.sh pools them). Roots: R87 (this batch), R86 (frozen WL86), R86A (frozen WL86A).

  e3select <R87>                          geometry-only choice of E3 among the protocol candidates (masks, no radiance) -> e3_choice.json
  ref <R87> <cfg> <ER|P> <seed> <spp> <tag>        reference render (pt86 'cam' estimator, unchanged) for E3
  field <R87> <cfg> <seed> <spp> <tag>             oracle field (pt86.Ctx.field, two halves) for E3 (diagnostic ceiling only)
  orders <R87> <cfg> <seed> <spp>                  per-order path-traced field (two halves)
  tables <R87> <R86> <cfg> [--ls=]                 EXP + controls (BINQ, ENC, RAYSEND, C3) on the deployed state; tables + timing
  real <R87> <R86> <cfg>                           16 state realizations: EXP, ECRO through each, panel-family reads, pooled diagnostic
  render <R87> <R86> <cfg> <list> [--ls=] [--cam=2]   render tables (file:key per line, tag) with the WL86 LOW + lookup estimator
  ecro <R87> <R86> <cfg> <model> <variant> <seed> <spp> <tag> [--ls=]   frozen ECRO render (E3; WL86 code unchanged)
"""
import json, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fx87  # noqa: E402
PT = fx87.PT
mi, dr, S = PT.mi, PT.dr, PT.S
FX = PT.FX
P87 = fx87.P87
NB, K = S.NB, S.K


def opts():
    o = {a.split('=')[0]: a.split('=')[1] for a in sys.argv[2:] if a.startswith('--') and '=' in a}
    a = [x for x in sys.argv[2:] if not x.startswith('--')]
    return o, a


def lights_of(cfg, ls=None):
    e = cfg.split('_')[0]
    return FX['light_sets'][ls or FX[e]['light_set']]


def save(fn, **kw):
    os.makedirs(os.path.dirname(fn), exist_ok=True)
    np.savez_compressed(fn, **kw)


# ------------------------------------------------------------------------------------------------ E3 (untouched condition)
def e3select(R):
    import masks86
    rule = P87['e3_rule']
    rows = {}
    chosen = None
    dev = '--dev' in sys.argv
    for name in (['E3dev'] if dev else P87['e3_order']):
        fx87.inject(name=name)
        c = P87['e3_dev_candidate'][name] if dev else P87['e3_candidates'][name]
        gok = bool(S.geom_ok(c['T0']) and S.geom_ok(c['T1']) and S.light_ok(c['light']['pos'], c['T0']) and S.light_ok(c['light']['pos'], c['T1']))
        m, aux = masks86.compute('E3')
        cnt = {k: int(v.sum()) for k, v in m.items()}
        ok = (gok and all(cnt[k] >= rule['min_pixels'] for k in rule['regions'])) or dev
        rows[name] = {'counts': cnt, 'geometry_ok': gok, 'admissible': bool(ok)}
        print('E3 candidate', name, json.dumps(cnt), ok, flush=True)
        if ok and chosen is None:
            chosen = name
            out = os.path.join(R, 'results', 'masks'); os.makedirs(out, exist_ok=True)
            np.savez_compressed(os.path.join(out, 'E3.npz'), **m, **aux)
            json.dump(cnt, open(os.path.join(out, 'E3.json'), 'w'), indent=1)
            break
    json.dump({'chosen': chosen, 'candidates': rows, 'rule': rule, 'note': 'geometry / visibility only; no radiance of any E3 candidate exists'},
              open(os.path.join(R, 'results', 'e3_choice.json'), 'w'), indent=1)
    print('E3_CHOSEN', chosen, flush=True)


def ref(R, cfg, kind, seed, spp, tag, ls=None):
    fx87.inject(R)
    geom, ls0 = PT.config(cfg)
    ctx = PT.Ctx(geom)
    acc, T_ = ctx.render(PT.Lights(FX['light_sets'][ls or ls0]), kind, FX['camera'], FX['camera']['res'], spp, seed, 'cam', b=32)
    out = os.path.join(R, 'results', 'renders'); os.makedirs(out, exist_ok=True)
    np.savez_compressed(os.path.join(out, f'{tag}.npz'), **PT.pack(acc, FX['camera']['res'], spp))
    json.dump({'cfg': cfg, 'ref': kind, 'seed': seed, 'spp': spp, 'timing': T_}, open(os.path.join(out, f'{tag}.json'), 'w'), indent=1)
    print(tag, json.dumps(T_), flush=True)


def field(R, cfg, seed, spp, tag):
    fx87.inject(R)
    geom, ls = PT.config(cfg)
    ctx = PT.Ctx(geom); lt = PT.Lights(FX['light_sets'][ls])
    fa, ca = ctx.field(lt, spp // 2, seed)
    fb, cb = ctx.field(lt, spp // 2, seed + 500)
    save(os.path.join(R, 'results', 'fields', f'{tag}.npz'), fa=fa, fb=fb, ca=ca, cb=cb, B=(0.5 * (fa + fb)).astype(np.float32))
    print(tag, 'field done', flush=True)


def orders(R, cfg, seed, spp):
    import pt87
    fx87.inject(R)
    geom, ls = PT.config(cfg)
    ctx = PT.Ctx(geom); lt = PT.Lights(FX['light_sets'][ls])
    a = pt87.field_orders(ctx, lt, spp // 2, seed)
    b = pt87.field_orders(ctx, lt, spp // 2, seed + 500)
    save(os.path.join(R, 'results', 'fields', f'ORD_{cfg}.npz'), a=a.astype(np.float32), b=b.astype(np.float32))
    print('orders', cfg, flush=True)


# ------------------------------------------------------------------------------------------------ explicit tables (deployed state)
def tables(R, R86, cfg, ls=None):
    import torch
    import ecro86 as EC
    import quad87 as Q
    import xport87 as X
    fx87.inject(R)
    E = P87['explicit']
    geom, ls0 = PT.config(cfg)
    ls = ls or ls0
    tag = cfg + ('' if ls == ls0 else '_' + ls)
    ctx = PT.Ctx(geom); lt = PT.Lights(FX['light_sets'][ls])
    t0 = time.perf_counter()
    quad = Q.Quad(ctx.u_state_np, E['M_ray'], E['seed_ray'], E['M_dense'], E['seed_dense'])
    t_quad = time.perf_counter() - t0
    B, ords, q, st, T_ = X.build(ctx, lt, quad, E['P'], E['Mq'], E['seed_q'])
    out = {'EXP': B, 'q': q}
    for j, o in enumerate(ords):
        out[f'EXP_L{j + 2}'] = o
    t0 = time.perf_counter(); dense = quad.dense(); t_dense = time.perf_counter() - t0
    # C2-BINQ
    opb = X.XOp(ctx.G, st.hit, st.dbin, quad.binq_W())
    out['BINQ'] = X.recurrence([opb], q)[0].cpu().numpy()
    # C2-ENC (deployed 16-point encoding s, as ECRO's input)
    s = ctx.source(lt)
    qe = X.enc_q(ctx.G, s, quad.enc_matrix())
    op = X.XOp(ctx.G, st.hit, st.dbin, quad.W)
    out['ENC'] = X.recurrence([op], qe)[0].cpu().numpy(); out['q_enc'] = qe
    # C2-RAYSEND
    L2r = X.raysend_first(ctx, st, lt, quad.W)
    Br, ordr = X.recurrence([op], q, first=L2r)
    out['RAYSEND'] = Br.cpu().numpy(); out['RAYSEND_L2'] = L2r
    # C3 sender-side deposition, matched ray budget per order
    X.sync(); t0 = time.perf_counter()
    B3, o3 = X.sender_side(ctx, q, dense, ctx.N * K, E['seed_c3'])
    X.sync(); t_c3 = time.perf_counter() - t0
    out['C3'] = B3
    for j, o in enumerate(o3):
        out[f'C3_L{j + 2}'] = o
    save(os.path.join(R, 'results', 'tables', f'X_{tag}.npz'), **{k: np.asarray(v, np.float32) for k, v in out.items()})
    meta = {'cfg': cfg, 'ls': ls, 'timing_first_call': T_, 'quad_ray_s': t_quad, 'dense_s': t_dense, 'c3_s': t_c3, 'state_digest': EC.State.digest(st),
            'finite': bool(all(np.isfinite(v).all() for v in out.values())), 'min': {k: float(np.min(v)) for k, v in out.items()}}
    json.dump(meta, open(os.path.join(R, 'results', 'tables', f'X_{tag}.json'), 'w'), indent=1)
    print('tables', tag, json.dumps(meta), flush=True)


# ------------------------------------------------------------------------------------------------ state realizations (Stage 2)
def real(R, R86, cfg):
    import torch
    import ecro86 as EC
    import quad87 as Q
    import xport87 as X
    fx87.inject(R)
    E = P87['explicit']; S2 = P87['stage2']
    geom, ls = PT.config(cfg)
    ctx = PT.Ctx(geom); lt = PT.Lights(FX['light_sets'][ls])
    q = Q.qtable(ctx, lt, E['P'], E['Mq'], E['seed_q'])
    s = ctx.source(lt)
    models = {m: EC.load_model(R86, m) for m in S2['ecro_models']}
    G = ctx.G
    panel = np.nonzero(G.PMAT == 1)[0]
    pidx = -np.ones(G.N, np.int64); pidx[panel] = np.arange(len(panel))
    ops = []
    out = {}
    seeds = [S2['deployed_seed']] + list(S2['seeds'])
    for ri, rs in enumerate(seeds):
        us = X.realization_us(rs)
        X.set_realization(ctx, us)
        st = EC.build_state(ctx)
        quad = Q.Quad(us, E['M_ray'], E['seed_ray'] + 1000 * ri)       # independent quadrature noise per realization (deployed: seed_ray)
        op = X.XOp(G, st.hit, st.dbin, quad.W)
        if rs != S2['deployed_seed']:
            ops.append(op)
        B, ords = X.recurrence([op], q)
        tagr = f's{rs}'
        out[f'EXP_{tagr}'] = B.cpu().numpy()
        # panel-family (light -> panel -> receiver) through this realization: BIN (sender table q) and RAY (exact sender radiance)
        hit = st.hit.astype(np.int64)
        hp = (hit >= 0) & (G.PMAT[np.clip(hit, 0, None)] == 1)
        qmask = np.zeros_like(q); qmask[panel] = q[panel]
        out[f'BIN_{tagr}'] = op.apply(torch.as_tensor(qmask.astype(np.float32), device=X.dev)).cpu().numpy()
        Lr = np.zeros((hit.size, 3), np.float32)
        idx = np.nonzero(hp)[0]
        if len(idx):
            o = st.seg[idx, :3].astype(np.float64); e = st.seg[idx, 3:].astype(np.float64)
            d = e - o; d /= np.linalg.norm(d, axis=1, keepdims=True)
            ray = mi.Ray3f(mi.Point3f(*[PT.fl(o[:, k]) for k in range(3)]), mi.Vector3f(*[PT.fl(d[:, k]) for k in range(3)]))
            sd = dr.opaque(mi.UInt64, 1)
            rng = mi.PCG32(size=len(idx), initstate=sd, initseq=mi.UInt64(0))
            ov, _ = ctx.trace(ray, len(idx), rng, sd, 'ER', 1, lt, 'cam')
            vals = [ov[f'e1{c}'] for c in range(3)]
            dr.eval(vals); dr.sync_thread()
            Lr[idx] = np.stack([np.asarray(v, np.float32) for v in vals], 1)
        out[f'RAY_{tagr}'] = op.scatter(torch.as_tensor(Lr, device=X.dev).view(G.N, K, 3)).cpu().numpy()
        # receiver x panel-patch relation: ray counts and RAY irradiance (Lambert-equivalent: cos dOmega / kappa weights)
        rw = (Q.ray_dirs(us)[:, 2] * S.DOMEGA / S.KAPPA)
        rid = np.arange(hit.size) // K; kk = np.arange(hit.size) % K
        rel_cnt = np.zeros((G.N, len(panel)), np.int32); rel_E = np.zeros((G.N, len(panel), 3))
        np.add.at(rel_cnt, (rid[hp], pidx[hit[hp]]), 1)
        np.add.at(rel_E, (rid[hp], pidx[hit[hp]]), Lr[hp].astype(np.float64) * rw[kk[hp]][:, None])
        out[f'relcnt_{tagr}'] = rel_cnt; out[f'relE_{tagr}'] = rel_E
        # frozen learned ECRO read through this realization (generator / ray weights see this state; no parameter change)
        for mname, model in models.items():
            zr = Q.ray_dirs(us)[:, 2]
            model.rw = torch.tensor((zr * S.DOMEGA / S.KAPPA).astype(np.float32), device=X.dev)
            Bm, _ = EC.respond(model, st, s)
            out[f'ECRO_{mname}_{tagr}'] = Bm
        print('real', cfg, rs, 'panel-hit rays', int(hp.sum()), flush=True)
    Bp, _ = X.recurrence(ops, q)
    out['EXP_POOL'] = Bp.cpu().numpy()
    save(os.path.join(R, 'results', 'tables', f'R_{cfg}.npz'), **{k: (v.astype(np.float32) if v.dtype.kind == 'f' else v) for k, v in out.items()})
    print('real done', cfg, flush=True)


# ------------------------------------------------------------------------------------------------ renders
def render(R, R86, cfg, listfn, ls=None, cam2=False):
    """Each line of listfn: <npz path relative to R>:<key> <seed> <spp> <tag>. One process, compiled kernels reused."""
    fx87.inject(R)
    geom, ls0 = PT.config(cfg)
    ctx = PT.Ctx(geom); lt = PT.Lights(FX['light_sets'][ls or ls0])
    cam = FX['camera2'] if cam2 else FX['camera']
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
        acc, T_ = ctx.render(lt, 'ER', cam, cam['res'], int(spp), int(seed), 'ecro', B=PT.fl(B), b=32)      # b = 32 as ecro86.render_cfg: full CRN
        o = PT.pack(acc, cam['res'], int(spp))
        np.savez_compressed(os.path.join(out, f'{tag}.npz'), hl=o['hl'])
        json.dump({'cfg': cfg, 'src': src, 'seed': int(seed), 'spp': int(spp), 'timing': T_}, open(os.path.join(out, f'{tag}.json'), 'w'), indent=1)
        print(tag, round(T_['wall_s'], 2), flush=True)


def ecro(R, R86, cfg, model, variant, seed, spp, tag, ls=None):
    import ecro86 as EC
    fx87.inject(R)
    o, B, T_ = EC.render_cfg(R86, cfg, model, variant, seed, spp, ls=ls)
    out = os.path.join(R, 'results', 'renders'); os.makedirs(out, exist_ok=True)
    np.savez_compressed(os.path.join(out, f'{tag}.npz'), hl=o['hl'], B=B.astype(np.float32))
    json.dump({'cfg': cfg, 'model': model, 'variant': variant, 'seed': seed, 'spp': spp, 'timing': T_}, open(os.path.join(out, f'{tag}.json'), 'w'), indent=1)
    print(tag, flush=True)


def main():
    dr.set_flag(dr.JitFlag.KernelHistory, True)
    cmd = sys.argv[1]
    o, a = opts()
    if cmd == 'e3select':
        e3select(a[0])
    elif cmd == 'ref':
        ref(a[0], a[1], a[2], int(a[3]), int(a[4]), a[5], o.get('--ls'))
    elif cmd == 'field':
        field(a[0], a[1], int(a[2]), int(a[3]), a[4])
    elif cmd == 'orders':
        orders(a[0], a[1], int(a[2]), int(a[3]))
    elif cmd == 'tables':
        tables(a[0], a[1], a[2], o.get('--ls'))
    elif cmd == 'real':
        real(a[0], a[1], a[2])
    elif cmd == 'render':
        render(a[0], a[1], a[2], a[3], o.get('--ls'), o.get('--cam') == '2')
    elif cmd == 'ecro':
        ecro(a[0], a[1], a[2], a[3], a[4], int(a[5]), int(a[6]), a[7], o.get('--ls'))


if __name__ == '__main__':
    main()
    os._exit(0)
