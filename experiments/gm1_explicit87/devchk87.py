"""WL87 pre-freeze development check (non-evidence): plumbing, timing and normalisation of the explicit comparators on a DEV geometry
from the WL86 dev split (seed 86300) with a dev light - never E1 / E2 / E3.

  python devchk87.py <out_json> [spp_truth]
"""
import json, math, sys, time
import numpy as np

import fx87
import quad87 as Q
import xport87 as X
import pt87
PT = fx87.PT
mi, dr, S = PT.mi, PT.dr, PT.S
NB, K = S.NB, S.K


def main():
    out_fn = sys.argv[1]
    spp = int(sys.argv[2]) if len(sys.argv) > 2 else 512
    rng = np.random.default_rng(86300)
    g = S.sample_geom(rng)
    light = S.sample_light(rng, g)
    import ecro86 as EC
    E = fx87.P87['explicit']
    ctx = PT.Ctx(g); lt = PT.Lights(light)
    R = {'geom': g, 'light': light}
    t0 = time.perf_counter(); quad = Q.Quad(ctx.u_state_np, E['M_ray'], E['seed_ray'], E['M_dense'], E['seed_dense']); R['quad_ray_s'] = time.perf_counter() - t0
    for rep in range(3):
        B, ords, q, st, T_ = X.build(ctx, lt, quad, E['P'], E['Mq'], E['seed_q'])
        R[f'build_{rep}'] = T_
    t0 = time.perf_counter(); dense = quad.dense(); R['dense_s'] = time.perf_counter() - t0
    X.sync(); t0 = time.perf_counter()
    B3, o3 = X.sender_side(ctx, q, dense, ctx.N * K, E['seed_c3'])
    X.sync(); R['c3_s'] = time.perf_counter() - t0
    opb = X.XOp(ctx.G, st.hit, st.dbin, quad.binq_W())
    Bb = X.recurrence([opb], q)[0].cpu().numpy()
    s = ctx.source(lt)
    op = X.XOp(ctx.G, st.hit, st.dbin, quad.W)
    Be = X.recurrence([op], X.enc_q(ctx.G, s, quad.enc_matrix()))[0].cpu().numpy()
    L2r = X.raysend_first(ctx, st, lt, quad.W)
    Br = X.recurrence([op], q, first=L2r)[0].cpu().numpy()
    t0 = time.perf_counter()
    fo = pt87.field_orders(ctx, lt, spp, 86399)
    R['truth_s'] = time.perf_counter() - t0
    tr = fo.sum(0)
    G = ctx.G
    w = (G.PAREA[:, None] * ((np.arange(NB) // S.NPHI + 0.5) / S.NZ)[None])[..., None]
    cls = {'floor': G.PMAT == 0, 'panel': G.PMAT == 1, 'sphere': G.PMAT == 2, 'walls': G.PMAT == 3, 'boxes': G.PMAT >= 4}
    def cmp(X_, T_):
        o = {}
        for k, m in cls.items():
            o[k] = {'flux_ratio': float((X_[m] * w[m]).sum() / max((T_[m] * w[m]).sum(), 1e-30)),
                    'rel_l1': float(np.abs(X_[m] - T_[m]).sum() / max(T_[m].sum(), 1e-30))}
        return o
    for name, Bx in (('EXP', B), ('BINQ', Bb), ('ENC', Be), ('RAYSEND', Br), ('C3', B3)):
        R[name] = cmp(Bx, tr)
    R['orders_EXP'] = [cmp(ords[j], fo[j])['walls'] for j in range(4)]
    R['orders_truth_flux_share'] = [float((fo[j] * w).sum() / (tr * w).sum()) for j in range(4)]
    json.dump(R, open(out_fn, 'w'), indent=1)
    for k in ('EXP', 'BINQ', 'ENC', 'RAYSEND', 'C3'):
        print(k, json.dumps({c: [round(v['flux_ratio'], 3), round(v['rel_l1'], 3)] for c, v in R[k].items()}))
    print('orders EXP walls', R['orders_EXP'], R['orders_truth_flux_share'])
    print('timing', {k: R[k] for k in R if k.endswith('_s') or k.startswith('build')})


if __name__ == '__main__':
    dr.set_flag(dr.JitFlag.KernelHistory, True)
    main()
    import os
    os._exit(0)
