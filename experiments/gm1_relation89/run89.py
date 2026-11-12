"""WL89 job CLI (one GPU process per call; run89.sh pools them). Roots: R89 (this batch). WL88 run88 functions are reused unchanged
(their configuration is redirected to the WL89 family by fx89.inject).

  wscreen <R89>                    geometry-only screening of the W1 candidates (protocol w_order) -> w_screen.json, masks M_<name>
  wchoose <R89> <name>             record W1 = name; masks of W1 / W2 / held-out sources
  ref | field | orders | render    as run88 (references, oracle tables, per-order truth, LOW + lookup renders)
  tables <R89> <cfg> [--ls=]       R0, C3UR, R1, R1M, C3UR16, POOL2, C3, C3U, RS0 / RS1 (repaired) + plumbing validation, per-order tables
  real <R89> <cfg>                 16 realizations of R0 / R1 / C3UR, pooled and converged relations, relation coverage and missing energy
"""
import json, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fx89  # noqa: E402
import run88 as R8  # noqa: E402
PT, Q = fx89.PT, fx89.Q
mi, dr, S = PT.mi, PT.dr, PT.S
FX = PT.FX
P89 = fx89.P89
E87 = fx89.fx88.fx87.P87['explicit']
RP = fx89.fx88.P88['relation']
NB, K = S.NB, S.K
jd, save = R8.jd, R8.save


def wscreen(R):
    import masks88
    rule = P89['w_rule']
    rows = {}
    out = os.path.join(R, 'results', 'masks'); os.makedirs(out, exist_ok=True)
    for name in P89['w_order']:
        ex = FX[name]
        lsets = [ex['light_set'], 'LMIX89', 'LSPOT89']
        gok = bool(S.geom_ok(ex['T0']) and S.geom_ok(ex['T1']) and all(S.light_ok(l['pos'], ex[t]) for ls in lsets for l in FX['light_sets'][ls] for t in ('T0', 'T1')))
        m, aux = masks88.compute(name)
        cnt = {k: int(v.sum()) for k, v in m.items()}
        ok = gok and all(cnt[k] >= rule['min_pixels'] for k in rule['regions'])
        rows[name] = {'counts': cnt, 'geometry_ok': gok, 'admissible': bool(ok)}
        np.savez_compressed(os.path.join(out, f'M_{name}.npz'), **m, **aux)
        print('W candidate', name, json.dumps(cnt), ok, flush=True)
    jd({'candidates': rows, 'rule': rule, 'order': P89['w_order']}, os.path.join(R, 'results', 'w_screen.json'))
    print('W_GEOMETRY_ADMISSIBLE', ' '.join(n for n in P89['w_order'] if rows[n]['admissible']), flush=True)


def wchoose(R, name):
    import masks88
    import shutil
    out = os.path.join(R, 'results', 'masks')
    jd({'chosen': name, 'note': 'first geometry-admissible candidate (protocol w_order) admitted by the qualification-partition criteria'},
       os.path.join(R, 'results', 'w_choice.json'))
    shutil.copy(os.path.join(out, f'M_{name}.npz'), os.path.join(out, 'W1.npz'))
    R8.inject_choice(R)
    for nm, ls in [('W2', None), ('W1', 'LMIX89'), ('W1', 'LSPOT89')]:
        m, aux = masks88.compute(nm, ls)
        tag = nm + ('_' + ls if ls else '')
        np.savez_compressed(os.path.join(out, f'{tag}.npz'), **m, **aux); jd({k: int(v.sum()) for k, v in m.items()}, os.path.join(out, f'{tag}.json'))
        print('masks', tag, json.dumps({k: int(v.sum()) for k, v in m.items() if k in ('C_ss', 'R_mirror', 'D', 'A', 'stationary_all')}), flush=True)


def tables(R, cfg, ls=None):
    import torch
    import ecro86 as EC
    import xport87 as X
    import rel88 as RL
    import rel89 as R9
    geom, ls0 = R8.setup(R, cfg)
    ls = ls or ls0
    tag = cfg + ('' if ls == ls0 else '_' + ls)
    ctx = PT.Ctx(geom); lt = PT.Lights(FX['light_sets'][ls])
    mat = R8.materials(ctx)
    T = {}
    quad = Q.Quad(ctx.u_state_np, E87['M_ray'], E87['seed_ray'])
    B0, o0, q, st, T['R0_first'] = X.build(ctx, lt, quad, E87['P'], E87['Mq'], E87['seed_q'])
    out = {'q': q, 'R0': B0}
    for j, o in enumerate(o0):
        out[f'R0_L{j + 2}'] = o
    stats = {}
    S1 = P89['stage1']
    for name, ks, sd in [('C3UR', S1['kappa_s'], S1['seed_send']), ('C3UR16', 2 * S1['kappa_s'], S1['seed_send'] + 1)]:
        Tm = {}
        rel = R9.SendRelR(ctx, ks, sd, mat['dense8'], timing=Tm)
        B, ords = RL.recurrence([rel], q)
        out[name] = B.cpu().numpy()
        if name == 'C3UR':
            for j, o in enumerate(ords):
                out[f'C3UR_L{j + 2}'] = o.cpu().numpy()
        stats[name] = {'kappa_s': ks, 'rays': rel.n_rays, 'landed': rel.n_landed, 'bytes': rel.nbytes(), 'timing_first': Tm, 'digest': rel.digest()}
        del rel
    for name, kap, m in [('R1', RP['kappa_u'], RP['m']), ('R1M', RP['matched_variant']['kappa_u'], RP['matched_variant']['m'])]:
        Tm = {}
        rel = RL.SAM(ctx, quad, mat['dense16'], kap, m, RP['seed_area'], st=st, timing=Tm)
        B, ords = RL.recurrence([rel], q)
        out[name] = B.cpu().numpy()
        if name == 'R1':
            for j, o in enumerate(ords):
                out[f'R1_L{j + 2}'] = o.cpu().numpy()
            # repaired RAYSEND: plumbing validation (table radiance must reproduce the first gather) and the exact first gather
            L2t, _ = R9.raysend_fixed(ctx, lt, rel, st, table=np.asarray(q, np.float32))
            L2a = rel.apply(torch.as_tensor(np.asarray(q, np.float32), device=RL.dev))
            v1 = float((L2t - L2a).abs().max().item() / max(L2a.abs().max().item(), 1e-30))
            L2e, frac1 = R9.raysend_fixed(ctx, lt, rel, st)
            out['RS1'] = RL.recurrence([rel], q, first=L2e)[0].cpu().numpy()
            r0s = RL.SAM(ctx, quad, mat['dense16'], RP['kappa_u'], 0, RP['seed_area'], st=st)
            L2t0, _ = R9.raysend_fixed(ctx, lt, r0s, st, table=np.asarray(q, np.float32))
            L2a0 = r0s.apply(torch.as_tensor(np.asarray(q, np.float32), device=RL.dev))
            v0 = float((L2t0 - L2a0).abs().max().item() / max(L2a0.abs().max().item(), 1e-30))
            L2e0, _ = R9.raysend_fixed(ctx, lt, r0s, st)
            r0op = X.XOp(ctx.G, st.hit, st.dbin, quad.W)
            out['RS0'] = RL.recurrence([r0op], q, first=L2e0)[0].cpu().numpy()
            T['raysend_validation'] = {'RS1_table_vs_apply_maxrel': v1, 'RS0_table_vs_apply_maxrel': v0, 'RS1_area_exact_hit_fraction': frac1}
        stats[name] = {'kappa_u': kap, 'm': m, 'pairs': int(len(rel.pi)), 'area_valid': rel.n_area_valid, 'missing_pairs': rel.missing,
                       'rays': rel.rays(), 'bytes': rel.nbytes(), 'timing_first': Tm, 'digest': rel.digest()}
        del rel
    torch.cuda.empty_cache()
    us = X.realization_us(P89['stage2']['seeds'][0]); X.set_realization(ctx, us)
    s2 = EC.build_state(ctx)
    op2 = X.XOp(ctx.G, s2.hit, s2.dbin, R8.quad_of(us, 1).W)
    X.set_realization(ctx, X.realization_us(86001))
    out['POOL2'] = RL.recurrence([X.XOp(ctx.G, st.hit, st.dbin, quad.W), op2], q)[0].cpu().numpy()
    del op2
    torch.cuda.empty_cache()
    X.sync(); t0 = time.perf_counter()
    out['C3'] = X.sender_side(ctx, q, mat['dense8'], ctx.N * K, E87['seed_c3'])[0]
    X.sync(); T['C3_s'] = time.perf_counter() - t0
    L = torch.as_tensor(np.asarray(q, np.float32), device=RL.dev); acc = 0
    for j in range(RL.J):
        L = R9.SendRelR(ctx, S1['kappa_s'], S1['seed_send'] + 17 * (j + 1), mat['dense8']).apply(L); acc = acc + L
    out['C3U'] = acc.cpu().numpy()
    save(os.path.join(R, 'results', 'tables', f'X_{tag}.npz'), **{k: np.asarray(v, np.float32) for k, v in out.items()})
    meta = {'cfg': cfg, 'ls': ls, 'timing': T, 'relations': stats, 'state_digest': EC.State.digest(st),
            'finite': bool(all(np.isfinite(v).all() for v in out.values())), 'min': {k: float(np.min(v)) for k, v in out.items()}}
    jd(meta, os.path.join(R, 'results', 'tables', f'X_{tag}.json'))
    print('tables', tag, json.dumps({'finite': meta['finite'], 'raysend_validation': T['raysend_validation']}), flush=True)


def real(R, cfg):
    import torch
    import ecro86 as EC
    import xport87 as X
    import rel88 as RL
    import rel89 as R9
    geom, ls = R8.setup(R, cfg)
    ctx = PT.Ctx(geom); lt = PT.Lights(FX['light_sets'][ls])
    mat = R8.materials(ctx)
    G = ctx.G
    q = Q.qtable(ctx, lt, E87['P'], E87['Mq'], E87['seed_q'])
    S2 = P89['stage2']; S1 = P89['stage1']
    pk, _ = RL.peaked()
    pi, pp = RL.pair_list(G, pk)
    Ep = R8.pair_energy(ctx, (pi, pp), q, S2['pair_m'], S2['pair_seed'])
    El = Ep.mean(1)
    tot_i = np.bincount(pi, weights=El, minlength=G.N)
    imp = El >= S2['important_share'] * tot_i[pi]
    keys = pi * G.N + pp
    out = {'pair_i': pi, 'pair_p': pp, 'pair_E': Ep.astype(np.float32), 'pair_imp': imp}
    cov = {}
    miss = {a: np.zeros(G.N) for a in ('R0', 'R1', 'C3UR')}
    seeds = [86001] + list(S2['seeds'])
    ops = {'R0': [], 'R1': [], 'C3UR': []}
    for ri, rs in enumerate(seeds):
        us = X.realization_us(rs); X.set_realization(ctx, us)
        st = EC.build_state(ctx)
        quad = R8.quad_of(us, ri) if rs != 86001 else Q.Quad(us, E87['M_ray'], E87['seed_ray'])
        r0 = X.XOp(G, st.hit, st.dbin, quad.W)
        r1 = RL.SAM(ctx, quad, mat['dense16'], RP['kappa_u'], RP['m'], RP['seed_area'] + 1000 * ri, st=st)
        rc = R9.SendRelR(ctx, S1['kappa_s'], S1['seed_send'] + (0 if rs == 86001 else 1000 * ri), mat['dense8'])
        tg = f's{rs}'
        for a, op in (('R0', r0), ('R1', r1), ('C3UR', rc)):
            out[f'{a}_{tg}'] = RL.recurrence([op], q)[0].cpu().numpy()
            if rs != 86001:
                ops[a].append(op)
        hit = st.hit.astype(np.int64); rid = np.arange(hit.size) // K
        s0 = np.isin(keys, np.unique(rid[hit >= 0] * G.N + hit[hit >= 0]))
        s1 = s0 | r1.A['valid'].any(1)
        hk = rc.hit >= 0
        srcp = (np.arange(rc.hit.size) // (S1['kappa_s'] * NB))
        sc = np.isin(keys, np.unique(rc.hit[hk].astype(np.int64) * G.N + srcp[hk]))
        cov[tg] = {}
        for a, sm in (('R0', s0), ('R1', s1), ('C3UR', sc)):
            cov[tg].update({f'{a}_energy_cov': float(El[sm].sum() / El.sum()), f'{a}_imp_missed': float((imp & ~sm).sum() / max(imp.sum(), 1)),
                            f'{a}_imp_missed_energy': float(El[imp & ~sm].sum() / max(El[imp].sum(), 1e-30))})
            miss[a] += np.bincount(pi[imp & ~sm], weights=El[imp & ~sm], minlength=G.N) / len(seeds)
        if rs == 86001:
            for a, sm in (('R0', s0), ('R1', s1), ('C3UR', sc)):
                out[f'miss_deployed_{a}'] = np.bincount(pi[imp & ~sm], weights=El[imp & ~sm], minlength=G.N).astype(np.float32)
        print('real', cfg, rs, json.dumps(cov[tg]), flush=True)
    for a in miss:
        out[f'miss_mean_{a}'] = miss[a].astype(np.float32)
    out['imp_energy_per_receiver'] = np.bincount(pi[imp], weights=El[imp], minlength=G.N).astype(np.float32)
    for a in ops:
        out[f'{a}_POOL16'] = RL.recurrence(ops[a], q)[0].cpu().numpy()
    del ops['R1']
    torch.cuda.empty_cache()
    for ri, rs in enumerate(S2['inf_seeds']):
        us = X.realization_us(rs); X.set_realization(ctx, us)
        st = EC.build_state(ctx)
        ops['R0'].append(X.XOp(G, st.hit, st.dbin, R8.quad_of(us, 100 + ri).W))
    out['R0_INF'] = RL.recurrence(ops['R0'], q)[0].cpu().numpy()
    del ops
    torch.cuda.empty_cache()
    o1 = []
    for ri, rs in enumerate(seeds[1:17]):
        us = X.realization_us(rs); X.set_realization(ctx, us)
        st = EC.build_state(ctx)
        o1.append(RL.SAM(ctx, R8.quad_of(us, ri + 1), mat['dense16'], RP['kappa_u'], 16, RP['seed_area'] + 500 + ri, st=st))
    out['R1_INF'] = RL.recurrence(o1, q)[0].cpu().numpy()
    del o1
    torch.cuda.empty_cache()
    oc = [R9.SendRelR(ctx, S2['send_inf_kappa'], S2['send_inf_seed'] + r, mat['dense8']) for r in range(S2['send_inf_reps'])]
    out['C3UR_INF'] = RL.recurrence(oc, q)[0].cpu().numpy()
    del oc
    save(os.path.join(R, 'results', 'tables', f'R_{cfg}.npz'), **{k: (v.astype(np.float32) if v.dtype.kind == 'f' else v) for k, v in out.items()})
    jd({'coverage': cov, 'pairs': int(len(pi)), 'important_pairs': int(imp.sum())}, os.path.join(R, 'results', 'tables', f'R_{cfg}.json'))
    print('real done', cfg, flush=True)


def main():
    dr.set_flag(dr.JitFlag.KernelHistory, True)
    cmd = sys.argv[1]
    o, a = R8.opts()
    if cmd == 'wscreen':
        wscreen(a[0])
    elif cmd == 'wchoose':
        wchoose(a[0], a[1])
    elif cmd == 'ref':
        R8.ref(a[0], a[1], a[2], int(a[3]), int(a[4]), a[5], o.get('--ls'))
    elif cmd == 'field':
        R8.field(a[0], a[1], int(a[2]), int(a[3]), a[4], o.get('--ls'))
    elif cmd == 'orders':
        R8.orders(a[0], a[1], int(a[2]), int(a[3]), o.get('--ls'))
    elif cmd == 'tables':
        tables(a[0], a[1], o.get('--ls'))
    elif cmd == 'real':
        real(a[0], a[1])
    elif cmd == 'render':
        R8.render(a[0], a[1], a[2], o.get('--ls'))


if __name__ == '__main__':
    main()
    os._exit(0)
