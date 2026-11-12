"""WL89 Stage 0 correctness tests. python tests89.py <R89> <R87> [--dev] -> results/tests88.json (WL88 frozen tests, unchanged, run on the
WL89 W1) + results/tests89.json (C3UR and repaired-RAYSEND tests); last line TESTS PASS|FAIL (both files must pass).
"""
import inspect, json, math, os, sys
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fx89  # noqa: E402
import run88 as R8  # noqa: E402
import tests88  # noqa: E402
PT, Q = fx89.PT, fx89.Q
mi, dr, S = PT.mi, PT.dr, PT.S
FX = PT.FX
E87 = fx89.fx88.fx87.P87['explicit']
RP = fx89.fx88.P88['relation']
P89 = fx89.P89
NB, K = S.NB, S.K
DEV = '--dev' in sys.argv


def main():
    import ecro86 as EC
    import xport87 as X
    import rel88 as RL
    import rel89 as R9
    R89 = sys.argv[1]
    tests88.main()                                                        # WL88 frozen tests (prints its own TESTS line)
    t88 = json.load(open(os.path.join(R89, 'results', 'tests88.json')))
    R8.inject_choice(R89)
    cfg = 'W1_T1' if 'W1' in FX else 'Wk_T1'
    fx89.fx88.setup(cfg)
    g, ls = PT.config(cfg); g0 = FX[cfg.split('_')[0]]['T0']
    ex = FX[cfg.split('_')[0]]
    ctx = PT.Ctx(g); lt = PT.Lights(FX['light_sets'][ls])
    G = ctx.G
    d8 = Q.Quad(ctx.u_state_np, 64, E87['seed_ray'], 2048 if DEV else E87['M_dense'], E87['seed_dense']).dense()
    ks = P89['stage1']['kappa_s']; sd = P89['stage1']['seed_send']
    T = {}
    rc = R9.SendRelR(ctx, ks, sd, d8)
    # zero / linearity / non-negativity
    z = rc.apply(torch.zeros(G.N, NB, 3, device=RL.dev)).abs().max().item()
    a1 = torch.rand(G.N, NB, 3, device=RL.dev); a2 = torch.rand(G.N, NB, 3, device=RL.dev)
    lin = (rc.apply(2 * a1 + 3 * a2) - 2 * rc.apply(a1) - 3 * rc.apply(a2)).abs().max().item() / rc.apply(2 * a1 + 3 * a2).abs().max().item()
    neg = min(float(rc.w_t.min().item()), float(rc.apply(a1).min().item()), float(min(v.min().item() for v in rc.D.values())))
    T['T_C3UR_ZERO_LIN_NONNEG'] = {'zero': z, 'linearity': lin, 'min': neg, 'PASS': z == 0 and lin <= 1e-5 and neg >= 0}
    # per-ray form equivalence: the aggregated contraction equals the WL88 per-ray deposition form (rel88.SendRel) for the same rays
    Lq = torch.as_tensor(np.random.default_rng(1).random((G.N, NB, 3)).astype(np.float32), device=RL.dev)
    keep = rc.hit >= 0
    src = np.nonzero(keep)[0] // ks; h = rc.hit[keep].astype(np.int64); cl = rc.cell[keep].astype(np.int64)
    zc = (np.arange(NB) // S.NPHI + 0.5) / S.NZ
    w = G.PAREA[src // NB] * zc[src % NB] * S.DOMEGA / ks / G.PAREA[h]
    ref = np.zeros((G.N, NB, 3))
    e = Lq.cpu().numpy().reshape(-1, 3)[src] * w[:, None]
    for m in range(6):
        sel = G.PMAT[h] == m
        if sel.any():
            np.add.at(ref, h[sel], np.asarray(d8[m], np.float64)[cl[sel]] * e[sel][:, None, :])
    got = rc.apply(Lq).cpu().numpy()
    T['T_C3UR_FORM'] = {'maxrel': float(np.abs(got - ref).max() / np.abs(ref).max()), 'PASS': float(np.abs(got - ref).max() / np.abs(ref).max()) <= 1e-4}
    # normalization / target identity on a known field: L = 1 on every peaked patch -> Lambert receivers rho / pi x projected solid angle
    pk, _ = RL.peaked()
    Lk = np.zeros((G.N, NB, 3), np.float32); Lk[pk[G.PMAT]] = 1.0
    Lt = torch.as_tensor(Lk, device=RL.dev)
    npool = 4 if DEV else 16
    acc = 0
    for r in range(npool):
        acc = acc + R9.SendRelR(ctx, ks, 89300 + r, d8).apply(Lt).cpu().numpy()
    acc /= npool
    pi_, pp_ = RL.pair_list(G, pk)
    Ep = R8.pair_energy(ctx, (pi_, pp_), Lk, 64 if DEV else 256, 88330)
    Ea = np.bincount(pi_, weights=Ep[:, 0], minlength=G.N)
    rho = fx89.fx88._FAM['wl88']['albedo'][G.PMAT] / math.pi
    refE = rho * Ea
    lamr = np.isin(G.PMAT, [3, 5]) & (G.PSID != S.SID_SPHERE) & (refE > 0)
    e3 = acc[..., 0].mean(1)
    T['T_C3UR_UNBIASED'] = {'flux_ratio': float(e3[lamr].sum() / refE[lamr].sum()), 'pool_vs_area_relL1': float(np.abs(e3[lamr] - refE[lamr]).sum() / refE[lamr].sum()),
                            'R0_R1_reference': {k: t88['T_UNBIASED'][k] for k in t88['T_UNBIASED'] if k != 'PASS'}}
    T['T_C3UR_UNBIASED']['PASS'] = abs(T['T_C3UR_UNBIASED']['flux_ratio'] - 1) <= 0.02
    # source independence, determinism
    sig = list(inspect.signature(R9.SendRelR.__init__).parameters)
    rc2 = R9.SendRelR(PT.Ctx(g), ks, sd, d8)
    T['T_C3UR_SOURCE_INDEPENDENT'] = {'signature': sig, 'PASS': not any(k in sig for k in ('lights', 'lt', 'q', 'source')) and rc.digest() == rc2.digest()}
    # edits: null, selective = full, move-and-return
    ctxe = PT.Ctx(g0)
    re_ = R9.SendRelR(ctxe, ks, sd, d8)
    dg0 = re_.digest()
    re_.update(ctxe, ex['moved'])
    null_ok = re_.digest() == dg0
    ctxe.set_geom(g)
    Tu = re_.update(ctxe, ex['moved'])
    full = R9.SendRelR(ctxe, ks, sd, d8)
    sel_ok = re_.digest() == full.digest() and np.array_equal(re_.seg, full.seg)
    ctxe.set_geom(g0)
    re_.update(ctxe, ex['moved'])
    mar_ok = re_.digest() == dg0
    T['T_C3UR_EDIT'] = {'null_edit_identical': null_ok, 'selective_equals_full': sel_ok, 'move_and_return_identical': mar_ok, 'edit_dirty_rays': Tu['dirty_rays'],
                        'PASS': bool(null_ok and sel_ok and mar_ok)}
    # repaired RAYSEND: table-radiance mode reproduces the first gather; exact traces hit the recorded sender
    quad = Q.Quad(ctx.u_state_np, 4096 if DEV else E87['M_ray'], E87['seed_ray'])
    d16 = RL.Dense(RP['dense_sub'], 1024 if DEV else RP['M_dense'], RP['seed_dense'])
    st = EC.build_state(ctx)
    r1 = RL.SAM(ctx, quad, d16, RP['kappa_u'], RP['m'], RP['seed_area'], st=st)
    q = Q.qtable(ctx, lt, E87['P'], E87['Mq'], E87['seed_q'])
    L2t, _ = R9.raysend_fixed(ctx, lt, r1, st, table=np.asarray(q, np.float32))
    L2a = r1.apply(torch.as_tensor(np.asarray(q, np.float32), device=RL.dev))
    _, frac = R9.raysend_fixed(ctx, lt, r1, st)
    v = float((L2t - L2a).abs().max().item() / L2a.abs().max().item())
    T['T_RAYSEND_REPAIRED'] = {'table_mode_vs_apply_maxrel': v, 'area_exact_hit_fraction': frac, 'PASS': v <= 1e-5 and frac >= 0.99}
    gated = {k: v['PASS'] for k, v in T.items()}
    T['ALL_PASS'] = bool(all(gated.values()) and t88['ALL_PASS'])
    T['tests88_ALL_PASS'] = t88['ALL_PASS']
    R8.jd(T, os.path.join(R89, 'results', 'tests89.json'))
    print(json.dumps(gated), flush=True)
    print('TESTS', 'PASS' if T['ALL_PASS'] else 'FAIL', flush=True)


if __name__ == '__main__':
    dr.set_flag(dr.JitFlag.KernelHistory, True)
    main()
    os._exit(0)
