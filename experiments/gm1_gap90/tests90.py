"""WL90 focused tests (protocol 'tests'). python tests90.py <R90> <R89> [--dev] -> results/tests90.json, last line TESTS PASS|FAIL"""
import json, math, os, sys
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fx90  # noqa: E402
import run88 as R8  # noqa: E402
PT, Q = fx90.PT, fx90.Q
mi, dr, S = PT.mi, PT.dr, PT.S
F, U = PT.F, PT.U
FX = PT.FX
P90, E87 = fx90.P90, fx90.E87
NB, K = S.NB, S.K
DEV = '--dev' in sys.argv


def rel(a, b):
    return float(np.abs(np.asarray(a, np.float64) - b).max() / max(np.abs(b).max(), 1e-30))


def main():
    import ecro86 as EC
    import xport87 as X
    import rel88 as RL
    import rel89 as R9
    import gap90 as GP
    R, R89 = sys.argv[1], sys.argv[2]
    cfg = 'W1_T1'
    geom, ls = R8.setup(R, cfg)
    ctx = PT.Ctx(geom); lt = PT.Lights(FX['light_sets'][ls])
    G = ctx.G; N = ctx.N
    T = {}
    # ---- per-order tracer = pt86 estimator (same random numbers) on camera rays
    res, spp = 128, 16
    n = res * res
    acc, _ = ctx.render(lt, 'ER', FX['camera'], res, spp, 90990, 'cam', b=spp)
    o86 = PT.pack(acc, res, spp)
    sd = dr.opaque(mi.UInt64, 90990 * 1000003)
    rng = mi.PCG32(size=n * spp, initstate=sd, initseq=mi.UInt64(0))
    ray, pix = PT.gen_rays(FX['camera'], res, spp, rng)
    cs, _ = GP.trace_orders(ctx, ray, lt, rng, 6)
    a = ctx.accum({f'o{k}_{c}': cs[k][c] for k in range(6) for c in range(3)}, pix, n)
    g = lambda k: np.stack([(a[f'o{k}_{c}'] / spp).reshape(res, res) for c in range(3)], -1)
    T['T_ORDERS_PT'] = {'e1': rel(g(0), o86['e1']), 'e2': rel(g(1), o86['e2']), 'h': rel(g(2) + g(3) + g(4) + g(5), o86['h'])}
    T['T_ORDERS_PT']['PASS'] = max(T['T_ORDERS_PT'].values()) <= 1e-5
    # ---- sub-bin / sub-patch sampling
    st = EC.build_state(ctx)
    idx, pid, y, v = GP.state_reads(st)
    sel = np.random.default_rng(1).choice(len(idx), 200_000, replace=False)
    pd = U(np.ascontiguousarray(pid[sel].astype(np.uint32)))
    yy = mi.Point3f(*[PT.fl(y[sel, k]) for k in range(3)]); vv = mi.Vector3f(*[PT.fl(v[sel, k]) for k in range(3)])
    rr = mi.PCG32(size=len(sel), initstate=dr.opaque(mi.UInt64, 5), initseq=mi.UInt64(0))
    u = [rr.next_float32() for _ in range(4)]
    sub = {}
    ok = True
    t, bv, nn = ctx.patch_frame(pd)
    loc = lambda w: (dr.dot(w, t), dr.dot(w, bv), dr.dot(w, nn))
    for gsz in (1, 4, 16):
        ga = int(round(math.sqrt(gsz)))
        v2 = GP.sub_bin_dir(ctx, pd, vv, gsz, u[2], u[3])
        c0 = RL.cell_of(*loc(vv), ga); c1 = RL.cell_of(*loc(v2), ga)
        y2, n2 = GP.sub_patch_point(ctx, pd, yy, gsz, u[0], u[1])
        # sub-patch membership: re-derive the sub-cell of y2 with the same parametrisation (a second call with zero jitter gives the cell origin)
        ya, _ = GP.sub_patch_point(ctx, pd, yy, gsz, F(0.0), F(0.0)); yb, _ = GP.sub_patch_point(ctx, pd, y2, gsz, F(0.0), F(0.0))
        dd = dr.norm(ya - yb)
        dr.eval(c0, c1, dd); dr.sync_thread()
        zloc = np.asarray(loc(vv)[2])
        inb = zloc > 1e-3                                                           # reads below the patch horizon (curved sender) are clamped
        fb = float((np.asarray(c0)[inb] == np.asarray(c1)[inb]).mean()); fp = float((np.asarray(dd) < 1e-4).mean())
        sub[f'g{gsz}'] = {'same_sub_bin_fraction': fb, 'same_sub_patch_fraction': fp}
        ok &= fb >= 0.999 and fp >= 0.999
    ve = GP.sub_bin_dir(ctx, pd, vv, GP.INF, u[2], u[3]); ye, _ = GP.sub_patch_point(ctx, pd, yy, GP.INF, u[0], u[1])
    dv = dr.norm(ve - vv); dy = dr.norm(ye - yy)
    dr.eval(dv, dy); dr.sync_thread()
    sub['exact_identity'] = float(max(np.asarray(dv).max(), np.asarray(dy).max()))
    T['T_SUB'] = {**sub, 'PASS': bool(ok and sub['exact_identity'] == 0.0)}
    # ---- pooled exact gather vs truth order fields; single-read table projection of the 1-event field vs the table gather of q
    q = Q.qtable(ctx, lt, E87['P'], E87['Mq'], E87['seed_q'])
    qt = torch.as_tensor(np.asarray(q, np.float32), device=GP.dev)
    nr = 2 if DEV else 8
    D = P90['diagnostic']
    EE = torch.zeros(4, N, NB, 3, device=GP.dev); PP = torch.zeros(4, N, NB, 3, device=GP.dev); TQ = torch.zeros(N, NB, 3, device=GP.dev)
    EE2 = torch.zeros(4, N, NB, 3, device=GP.dev)
    dig = []
    for ri in range(nr):
        us = X.realization_us(90900 + ri); X.set_realization(ctx, us)
        s2 = EC.build_state(ctx)
        op = X.XOp(G, s2.hit, s2.dbin, Q.Quad(us, 4096 if DEV else E87['M_ray'], 90910 + ri).W)
        d0 = EC.State.digest(s2)
        i2, p2, y2_, v2_ = GP.state_reads(s2)
        r_ee, _ = GP.xread(ctx, lt, p2, y2_, v2_, 0, 0, 1, 90920 + ri)
        EE += GP.gather_orders(op, N, i2, r_ee) / nr
        r_e2, _ = GP.xread(ctx, lt, p2, y2_, v2_, 0, 0, 1, 90920 + ri)
        EE2 += GP.gather_orders(op, N, i2, r_e2) / nr
        r_pp, _ = GP.xread(ctx, lt, p2, y2_, v2_, 1, 1, D['samples_per_read'], 90930 + ri)
        PP += GP.gather_orders(op, N, i2, r_pp) / nr
        TQ += op.apply(qt) / nr
        dig.append(d0 == EC.State.digest(s2))
    X.set_realization(ctx, X.realization_us(86001))
    oo = np.load(os.path.join(R89, 'results', 'fields', f'ORD_{cfg}.npz'))
    Tt = 0.5 * (oo['a'].astype(np.float64) + oo['b'])
    w = (G.PAREA[:, None] * ((np.arange(NB) // S.NPHI + 0.5) / S.NZ)[None])[..., None]
    fr = [float((EE[k].cpu().numpy() * w).sum() / (Tt[k + 1] * w).sum()) for k in range(4)]
    T['T_EE_TRUTH'] = {'flux_ratio_orders_2_5': fr, 'PASS': bool(all(abs(f - 1) <= 0.03 + (0.03 if DEV else 0) for f in fr))}
    fq = float((PP[0].cpu().numpy() * w).sum() / (TQ.cpu().numpy() * w).sum())
    T['T_PP_Q'] = {'flux_ratio': fq, 'rel_l1': float((PP[0] - TQ).abs().sum() / TQ.abs().sum()), 'PASS': abs(fq - 1) <= 0.02 + (0.02 if DEV else 0)}
    T['T_REPLAY'] = {'maxrel': float((EE - EE2).abs().max() / EE.abs().max()), 'PASS': float((EE - EE2).abs().max() / EE.abs().max()) <= 1e-5}
    # ---- X1 plumbing and source independence on the deployed relations
    mat = R8.materials(ctx)
    quad = Q.Quad(ctx.u_state_np, 4096 if DEV else E87['M_ray'], E87['seed_ray'])
    r0 = X.XOp(G, st.hit, st.dbin, quad.W)
    S1 = fx90.P89['stage1']
    rc = R9.SendRelR(ctx, S1['kappa_s'], S1['seed_send'], mat['dense8'])
    dg_c, dg_r = rc.digest(), EC.State.digest(st)
    a = GP.c3ur_first(rc, qt.reshape(-1, 3)[rc.src_t]); b = rc.apply(qt)
    hit = torch.as_tensor(st.hit[idx].astype(np.int64) * NB + st.dbin[idx].astype(np.int64), device=GP.dev)
    a0 = GP.r0_first_exact(r0, N, idx, qt.reshape(-1, 3)[hit]); b0 = r0.apply(qt)
    T['T_X1_PLUMBING'] = {'c3ur': float((a - b).abs().max() / b.abs().max()), 'r0': float((a0 - b0).abs().max() / b0.abs().max())}
    T['T_X1_PLUMBING']['PASS'] = max(T['T_X1_PLUMBING'].values()) <= 1e-5
    rbc = GP.c3ur_raybuf(rc); _ = rbc.radiance(ctx, lt)
    i0, rb = GP.r0_raybuf(st); _ = rb.radiance(ctx, lt)
    T['T_SOURCE_INDEPENDENT'] = {'c3ur_digest_unchanged': rc.digest() == dg_c, 'r0_digest_unchanged': EC.State.digest(st) == dg_r, 'pooled_states_unchanged': all(dig),
                                 'PASS': bool(rc.digest() == dg_c and EC.State.digest(st) == dg_r and all(dig))}
    gated = {k: v['PASS'] for k, v in T.items()}
    T['ALL_PASS'] = bool(all(gated.values()))
    R8.jd(T, os.path.join(R, 'results', 'tests90.json'))
    print(json.dumps(gated), flush=True)
    print(json.dumps({'EE_flux': fr, 'PP_q': fq, 'orders_pt': T['T_ORDERS_PT']}), flush=True)
    print('TESTS', 'PASS' if T['ALL_PASS'] else 'FAIL', flush=True)


if __name__ == '__main__':
    dr.set_flag(dr.JitFlag.KernelHistory, True)
    main()
    os._exit(0)
