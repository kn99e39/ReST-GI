"""WL90 job CLI (one GPU process per call). R90 = this batch (holds a copy of the WL89 w_choice.json); references / masks / fields are read from R89.

  xtab <R90> <cfg>          pooled exact-read operators X(a, s), hybrids M_0..M_4 and per-order composites (gap90 docstring) -> tables/X_<cfg>.npz
  feas <R90> <cfg>          deployed + 16 realizations: R0, R0X1, C3UR, C3URX1 -> tables/F_<cfg>.npz
  orders <R90> <cfg> <seed> <spp>     per-order truth fields 1..5 (run88.orders) for configurations WL89 did not cover
  render <R90> <cfg> <list>           LOW + lookup renders (run88.render, unchanged readout)
"""
import json, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fx90  # noqa: E402
import run88 as R8  # noqa: E402
PT, Q = fx90.PT, fx90.Q
mi, dr, S = PT.mi, PT.dr, PT.S
FX = PT.FX
P90, E87 = fx90.P90, fx90.E87
NB, K = S.NB, S.K
DEV = bool(os.environ.get('WL90_DEV'))


def vname(a, s):
    return f"a{'x' if a == 0 else a}s{'x' if s == 0 else s}"


def xtab(R, cfg):
    import torch
    import ecro86 as EC
    import xport87 as X
    import gap90 as GP
    geom, ls = R8.setup(R, cfg)
    ctx = PT.Ctx(geom); lt = PT.Lights(FX['light_sets'][ls])
    N = ctx.N
    q = Q.qtable(ctx, lt, E87['P'], E87['Mq'], E87['seed_q'])
    D = P90['diagnostic']
    seeds = D['pool_seeds'][:4] if DEV else D['pool_seeds']
    variants = [tuple(v) for v in D['variants']]
    Sn = D['samples_per_read']
    acc = {v: [torch.zeros(GP.KMAX, N, NB, 3, device=GP.dev) for _ in range(2)] for v in variants}
    ops = []
    nvalid = {v: 0.0 for v in variants}; nread = 0
    t0 = time.perf_counter()
    for ri, rs in enumerate(seeds):
        us = X.realization_us(rs); X.set_realization(ctx, us)
        st = EC.build_state(ctx)
        op = X.XOp(ctx.G, st.hit, st.dbin, Q.Quad(us, E87['M_ray'], D['quad_seed'] + 1000 * ri).W)
        ops.append(op)
        idx, pid, y, v = GP.state_reads(st)
        nread += len(idx)
        for vi, (a, s) in enumerate(variants):
            exact = a == GP.INF and s == GP.INF
            rad, cnt = GP.xread(ctx, lt, pid, y, v, a, s, 1 if exact else Sn, D['read_seed'] + 7919 * ri + 101 * vi)
            nvalid[(a, s)] += float((cnt > 0).sum())
            acc[(a, s)][ri % 2] += GP.gather_orders(op, N, idx, rad)
        print('xtab', cfg, ri, rs, round(time.perf_counter() - t0, 1), 's', flush=True)
    X.set_realization(ctx, X.realization_us(86001))
    nr = len(seeds)
    G = {v: (acc[v][0] + acc[v][1]) / nr for v in variants}
    out = {'q': np.asarray(q, np.float32)}
    for v in variants:
        nm = vname(*v)
        out[f'X_{nm}'] = G[v].sum(0).cpu().numpy()
        out[f'Xa_{nm}'] = (acc[v][0] / (nr / 2)).sum(0).cpu().numpy(); out[f'Xb_{nm}'] = (acc[v][1] / (nr / 2)).sum(0).cpu().numpy()
        out[f'O_{nm}'] = G[v].cpu().numpy()                                       # per order (gather of the k-event radiance, k = 1..4)
    EE, PP, EA, SE = G[(0, 0)], G[(1, 1)], G[(1, 0)], G[(0, 1)]
    hy = GP.hybrids(EE, ops, q)
    for k, orders in hy.items():
        out[k] = sum(orders).cpu().numpy()
        out[f'O_{k}'] = torch.stack(orders).cpu().numpy()
    comp = {'P1only': PP[0] + EE[1] + EE[2] + EE[3], 'P2plus': EE[0] + PP[1] + PP[2] + PP[3],
            'EA1': EA[0] + EE[1] + EE[2] + EE[3], 'SE1': SE[0] + EE[1] + EE[2] + EE[3]}
    for k, t in comp.items():
        out[k] = t.cpu().numpy()
    R8.save(os.path.join(R, 'results', 'tables', f'X_{cfg}.npz'), **{k: np.asarray(v, np.float32) for k, v in out.items()})
    meta = {'cfg': cfg, 'realizations': nr, 'reads_per_realization': nread / nr, 'valid_read_fraction': {vname(*v): nvalid[v] / nread for v in variants},
            'wall_s': time.perf_counter() - t0, 'finite': bool(all(np.isfinite(v).all() for v in out.values())), 'min': {k: float(np.min(v)) for k, v in out.items()}}
    R8.jd(meta, os.path.join(R, 'results', 'tables', f'X_{cfg}.json'))
    print('xtab done', cfg, json.dumps({'wall_s': round(meta['wall_s'], 1), 'finite': meta['finite'], 'valid': meta['valid_read_fraction']}), flush=True)


def feas(R, cfg):
    import torch
    import ecro86 as EC
    import xport87 as X
    import rel88 as RL
    import rel89 as R9
    import gap90 as GP
    geom, ls = R8.setup(R, cfg)
    ctx = PT.Ctx(geom); lt = PT.Lights(FX['light_sets'][ls])
    mat = R8.materials(ctx)
    q = Q.qtable(ctx, lt, E87['P'], E87['Mq'], E87['seed_q'])
    qt = torch.as_tensor(np.asarray(q, np.float32), device=GP.dev)
    S1 = fx90.P89['stage1']
    seeds = [86001] + list(fx90.P89['stage2']['seeds'])[: (2 if DEV else 16)]
    out = {}
    val = {}
    for ri, rs in enumerate(seeds):
        us = X.realization_us(rs); X.set_realization(ctx, us)
        st = EC.build_state(ctx)
        quad = R8.quad_of(us, ri) if rs != 86001 else Q.Quad(us, E87['M_ray'], E87['seed_ray'])
        r0 = X.XOp(ctx.G, st.hit, st.dbin, quad.W)
        rc = R9.SendRelR(ctx, S1['kappa_s'], S1['seed_send'] + (0 if rs == 86001 else 1000 * ri), mat['dense8'])
        tg = f's{rs}'
        out[f'R0_{tg}'] = RL.recurrence([r0], q)[0].cpu().numpy()
        out[f'C3UR_{tg}'] = RL.recurrence([rc], q)[0].cpu().numpy()
        idx, rb = GP.r0_raybuf(st)
        out[f'R0X1_{tg}'] = GP.rec_from(r0, GP.r0_first_exact(r0, ctx.N, idx, rb.radiance(ctx, lt))).cpu().numpy()
        rbc = GP.c3ur_raybuf(rc)
        out[f'C3URX1_{tg}'] = GP.rec_from(rc, GP.c3ur_first(rc, rbc.radiance(ctx, lt))).cpu().numpy()
        if ri == 0:                                                                # plumbing: table radiance reproduces the table gather
            a = GP.c3ur_first(rc, qt.reshape(-1, 3)[rc.src_t]); b = rc.apply(qt)
            val['c3ur_first_table_vs_apply_maxrel'] = float((a - b).abs().max() / b.abs().max())
            hit = torch.as_tensor(st.hit[idx].astype(np.int64) * NB + st.dbin[idx].astype(np.int64), device=GP.dev)
            a = GP.r0_first_exact(r0, ctx.N, idx, qt.reshape(-1, 3)[hit]); b = r0.apply(qt)
            val['r0_first_table_vs_apply_maxrel'] = float((a - b).abs().max() / b.abs().max())
        print('feas', cfg, rs, flush=True)
    X.set_realization(ctx, X.realization_us(86001))
    R8.save(os.path.join(R, 'results', 'tables', f'F_{cfg}.npz'), **{k: np.asarray(v, np.float32) for k, v in out.items()})
    R8.jd({'cfg': cfg, 'plumbing': val, 'finite': bool(all(np.isfinite(v).all() for v in out.values())), 'min': float(min(np.min(v) for v in out.values()))},
          os.path.join(R, 'results', 'tables', f'F_{cfg}.json'))
    print('feas done', cfg, json.dumps(val), flush=True)


def main():
    dr.set_flag(dr.JitFlag.KernelHistory, True)
    cmd = sys.argv[1]
    o, a = R8.opts()
    if cmd == 'xtab':
        xtab(a[0], a[1])
    elif cmd == 'feas':
        feas(a[0], a[1])
    elif cmd == 'orders':
        R8.orders(a[0], a[1], int(a[2]), int(a[3]), o.get('--ls'))
    elif cmd == 'field':
        R8.field(a[0], a[1], int(a[2]), int(a[3]), a[4], o.get('--ls'))
    elif cmd == 'render':
        R8.render(a[0], a[1], a[2], o.get('--ls'))


if __name__ == '__main__':
    main()
    os._exit(0)
