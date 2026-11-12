"""WL91 job CLI (one GPU process per call). R91 = this batch (copy of the WL89 w_choice.json); references / masks / fields read from R89 / R90.

  tabs <R91> <cfg> [--ls=]     deployed draw + 16 realizations of R0, C3UR, C3URX1 (+ R0X1 on the deployed draw), pooled C3UR / C3URX1,
                               per-order tables of the deployed C3UR / C3URX1; on W1_T1 also the relation-size points kappa_s 4 / 16
  orders <R91> <cfg> <seed> <spp> [--ls=]    per-order truth fields (run88.orders) where WL89 / WL90 have none
  render <R91> <cfg> <list> [--ls=]          LOW + lookup renders (run88.render, unchanged readout)
"""
import json, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fx91  # noqa: E402
import run88 as R8  # noqa: E402
PT, Q = fx91.PT, fx91.Q
mi, dr, S = PT.mi, PT.dr, PT.S
FX = PT.FX
P91, P89, E87 = fx91.P91, fx91.P89, fx91.E87
NB, K = S.NB, S.K
DEV = bool(os.environ.get('WL91_DEV'))


def tabs(R, cfg, ls=None):
    import torch
    import ecro86 as EC
    import xport87 as X
    import rel88 as RL
    import rel89 as R9
    import gap90 as GP
    import cand91 as C
    geom, ls0 = R8.setup(R, cfg)
    ls = ls or ls0
    tag = cfg + ('' if ls == ls0 else '_' + ls)
    ctx = PT.Ctx(geom); lt = PT.Lights(FX['light_sets'][ls])
    mat = R8.materials(ctx)
    q = Q.qtable(ctx, lt, E87['P'], E87['Mq'], E87['seed_q'])
    S1 = P89['stage1']
    seeds = [86001] + list(P89['stage2']['seeds'])[: (2 if DEV else 16)]
    out = {}
    meta = {'cfg': cfg, 'ls': ls, 'draws': seeds, 'rays': {}}
    pool_c, pool_x = [], []
    L2x = 0
    for ri, rs in enumerate(seeds):
        us = X.realization_us(rs); X.set_realization(ctx, us)
        st = EC.build_state(ctx)
        quad = R8.quad_of(us, ri) if rs != 86001 else Q.Quad(us, E87['M_ray'], E87['seed_ray'])
        r0 = X.XOp(ctx.G, st.hit, st.dbin, quad.W)
        cand = C.C3URX1(ctx, S1['kappa_s'], S1['seed_send'] + (0 if rs == 86001 else 1000 * ri), mat['dense8'])
        tg = f's{rs}'
        out[f'R0_{tg}'] = RL.recurrence([r0], q)[0].cpu().numpy()
        Bc, oc = C.c3ur_table(cand.rel, q, orders=True)
        Bx, ox = cand.table(ctx, lt, orders=True)
        out[f'C3UR_{tg}'] = Bc.cpu().numpy(); out[f'C3URX1_{tg}'] = Bx.cpu().numpy()
        if rs == 86001:
            idx, rb = GP.r0_raybuf(st)
            out['R0X1_s86001'] = GP.rec_from(r0, GP.r0_first_exact(r0, ctx.N, idx, rb.radiance(ctx, lt))).cpu().numpy()
            out['O_C3UR'] = torch.stack(oc).cpu().numpy(); out['O_C3URX1'] = torch.stack(ox).cpu().numpy()
            meta['rays'] = {'unique_relation_rays': cand.rel.n_rays, 'landed_rays': cand.n, 'source_eval_probe_intersections': cand.n,
                            'source_eval_shadow_rays': cand.n * lt.n, 'R0_state_rays': int(ctx.N * K)}
            meta['bytes'] = cand.nbytes()
        else:
            pool_c.append(cand.rel); L2x = L2x + ox[0]
        print('tabs', tag, rs, flush=True)
    # pooled (converged) controls: sampling variance vs first-gather projection
    npool = len(pool_c)
    T = lambda L: sum(rc.apply(L) for rc in pool_c) / npool
    L = torch.as_tensor(np.asarray(q, np.float32), device=C.dev); acc = 0
    for _ in range(C.J):
        L = T(L); acc = acc + L
    out['C3UR_POOL'] = acc.cpu().numpy()
    L = L2x / npool; acc = L.clone()
    for _ in range(C.J - 1):
        L = T(L); acc = acc + L
    out['C3URX1_POOL'] = acc.cpu().numpy()
    del pool_c
    torch.cuda.empty_cache()
    if cfg == 'W1_T1' and ls == ls0:                              # relation-size points (descriptive; single draw; never used for a verdict)
        for ks in P91['scaling']['kappa_s_fidelity']:
            cand = C.C3URX1(ctx, ks, S1['seed_send'] + 7, mat['dense8'])
            out[f'C3URX1_k{ks}'] = cand.table(ctx, lt).cpu().numpy()
            out[f'C3UR_k{ks}'] = C.c3ur_table(cand.rel, q).cpu().numpy()
            del cand
    X.set_realization(ctx, X.realization_us(86001))
    R8.save(os.path.join(R, 'results', 'tables', f'T_{tag}.npz'), **{k: np.asarray(v, np.float32) for k, v in out.items()})
    meta['finite'] = bool(all(np.isfinite(v).all() for v in out.values())); meta['min'] = float(min(np.min(v) for v in out.values()))
    R8.jd(meta, os.path.join(R, 'results', 'tables', f'T_{tag}.json'))
    print('tabs done', tag, json.dumps({k: meta[k] for k in ('finite', 'min', 'rays')}), flush=True)


def main():
    dr.set_flag(dr.JitFlag.KernelHistory, True)
    cmd = sys.argv[1]
    o, a = R8.opts()
    if cmd == 'tabs':
        tabs(a[0], a[1], o.get('--ls'))
    elif cmd == 'orders':
        R8.orders(a[0], a[1], int(a[2]), int(a[3]), o.get('--ls'))
    elif cmd == 'render':
        R8.render(a[0], a[1], a[2], o.get('--ls'))


if __name__ == '__main__':
    main()
    os._exit(0)
