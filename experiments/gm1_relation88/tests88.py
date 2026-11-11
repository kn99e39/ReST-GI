"""WL88 Stage 0 correctness tests (protocol stage0.tests). python tests88.py <R88> <R87> [--dev]  -> results/tests88.json, last line TESTS PASS|FAIL
Gated tests must pass before any evidence run; 'report' entries are reported only.
"""
import inspect, json, math, os, sys
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fx88  # noqa: E402
import run88  # noqa: E402
PT, Q = fx88.PT, fx88.Q
mi, dr, S, CD = PT.mi, PT.dr, PT.S, PT.CD
FX = PT.FX
E87 = fx88.fx87.P87['explicit']
RP = fx88.P88['relation']
NB, K = S.NB, S.K
DEV = '--dev' in sys.argv


def rel(a, b):
    return float(np.abs(np.asarray(a, np.float64) - b).max() / max(np.abs(b).max(), 1e-30))


def main():
    import ecro86 as EC
    import xport87 as X
    import rel88 as RL
    import pt88
    R88, R87 = sys.argv[1], sys.argv[2]
    run88.inject_choice(R88)
    T = {}
    # ---- T_WL87: R0 on the historical E2 T1 equals the frozen WL87 EXP table
    fx88.setup('E2_T1')
    g, ls = PT.config('E2_T1')
    ctx = PT.Ctx(g); lt = PT.Lights(FX['light_sets'][ls])
    quad = Q.Quad(ctx.u_state_np, E87['M_ray'], E87['seed_ray'])
    B0, _, q, st, _ = X.build(ctx, lt, quad, E87['P'], E87['Mq'], E87['seed_q'])
    fn = os.path.join(R87, 'results', 'tables', 'X_E2_T1.npz')
    if os.path.exists(fn):
        z = np.load(fn)
        T['T_WL87'] = {'R0_vs_EXP': rel(B0, z['EXP']), 'q_vs_WL87': rel(q, z['q'])}
        T['T_WL87']['PASS'] = T['T_WL87']['R0_vs_EXP'] <= 1e-4 and T['T_WL87']['q_vs_WL87'] <= 1e-4
    else:
        T['T_WL87'] = {'missing': fn, 'PASS': DEV}
    d16_86 = RL.Dense(RP['dense_sub'], 1024 if DEV else RP['M_dense'], RP['seed_dense'])
    r1z = RL.SAM(ctx, quad, d16_86, 8, 0, RP['seed_area'], st=st)
    T['T_REDUCE'] = {'R1_m0_vs_R0': rel(RL.recurrence([r1z], q)[0].cpu().numpy(), B0)}
    T['T_REDUCE']['PASS'] = T['T_REDUCE']['R1_m0_vs_R0'] <= 1e-6
    # ---- WL88 family on W1 T1 (or the dev geometry in dev)
    cfg = 'W1_T1' if 'W1' in FX else 'Wk_T1'
    fx88.setup(cfg)
    g, ls = PT.config(cfg); g0 = FX[cfg.split('_')[0]]['T0']
    ctx = PT.Ctx(g); lt = PT.Lights(FX['light_sets'][ls])
    quad = Q.Quad(ctx.u_state_np, E87['M_ray'], E87['seed_ray'])
    d16 = RL.Dense(RP['dense_sub'], 1024 if DEV else RP['M_dense'], RP['seed_dense'])
    st = EC.build_state(ctx)
    r1 = RL.SAM(ctx, quad, d16, RP['kappa_u'], RP['m'], RP['seed_area'], st=st)
    G = ctx.G
    pk, pis = RL.peaked()
    T['T_PEAKED'] = {'Pi_median': pis, 'peaked': pk.tolist(), 'PASS': bool(pk.tolist() == [False, True, True, False, True, False])}
    T['T_CULL'] = {'missing_pairs': r1.missing, 'PASS': r1.missing == 0}
    # ER consistency: sample weight vs eval / pdf, MIS-mixture albedo vs uniform albedo
    st_, lam, alb = Q.mats()
    rows = {}
    ok = True
    for m in (0, 1, 2):
        cl = CD.DrClosure83('ER', st_[m])
        for th in (0.3, 0.8, 1.2):
            n = 1 << 20
            wi = mi.Vector3f(math.sin(th), 0.0, math.cos(th))
            rng = mi.PCG32(size=n, initstate=dr.opaque(mi.UInt64, 88900 + m), initseq=mi.UInt64(0))
            wo, pdf, w, valid = cl.sample(wi, rng.next_float32(), mi.Point2f(rng.next_float32(), rng.next_float32()), True)
            f, p2 = cl.eval_pdf(wi, wo, valid)
            wr = dr.select(valid & (pdf > 1e-6), f * dr.rcp(dr.maximum(pdf, 1e-12)), 0.0)
            dr.eval(w, wr, pdf, p2); dr.sync_thread()
            a_s = np.asarray(w.x).mean(); a_r = np.asarray(wr.x).mean()
            pv = np.asarray(valid) & (np.asarray(pdf) > 1e-4)
            pdf_rel = float(np.median(np.abs(np.asarray(p2)[pv] - np.asarray(pdf)[pv]) / np.asarray(pdf)[pv]))
            u = mi.warp.square_to_uniform_hemisphere(mi.Point2f(rng.next_float32(), rng.next_float32()))
            fu = cl.eval(wi, u, True)
            dr.eval(fu); dr.sync_thread()
            a_u = np.asarray(fu.x).mean() * 2 * math.pi
            rows[f'm{m}_th{th}'] = {'albedo_sample_weight': float(a_s), 'albedo_eval_over_pdf': float(a_r), 'albedo_uniform': float(a_u), 'pdf_rel_median': pdf_rel}
            ok &= abs(a_s - a_r) <= 0.01 * max(a_s, 1e-3) + 2e-3 and abs(a_s - a_u) <= 0.03 * max(a_s, 1e-3) + 3e-3 and pdf_rel <= 1e-3
    T['T_ER'] = {'rows': rows, 'PASS': bool(ok)}
    # frames: patch frame == shading frame on flat patches (incl. the rotated mover)
    ns = 4096
    rng = np.random.default_rng(88901)
    flat = np.nonzero(G.PSID != S.SID_SPHERE)[0]
    pid = rng.choice(flat, ns)
    p, nn, (t, b, n) = ctx.patch_point(PT.U(pid.astype(np.uint32)), PT.fl(rng.random(ns)), PT.fl(rng.random(ns)))
    d = -n
    si = ctx.scene.ray_intersect(mi.Ray3f(p + n * 1e-3, d))
    a = si.to_local(t); c = si.to_local(n); hs = ctx.shape_id(si)
    dr.eval(a, c, hs); dr.sync_thread()
    same = np.asarray(hs) == G.PSID[pid]                     # coincident faces (box B bottom on the floor) hit another shape: excluded
    pid = pid[same]
    isbox = G.PSID[pid] >= S.SID_BOXA
    ta = np.abs(np.asarray(a.x)[same] - 1)
    ex_r = float(ta[~isbox].max()); ex_b = float(ta[isbox].max()) if isbox.any() else 0.0
    ez = float(np.abs(np.asarray(c.z)[same] - 1).max())
    # S1 (on the box mover G) azimuthal isotropy: f(R wi, R wo) vs f(wi, wo) for rotations R about the normal
    cl1 = CD.DrClosure83('ER', st_[1])
    nn_ = 1 << 16
    rr = mi.PCG32(size=nn_, initstate=dr.opaque(mi.UInt64, 88902), initseq=mi.UInt64(0))
    wi_ = mi.warp.square_to_uniform_hemisphere(mi.Point2f(rr.next_float32(), rr.next_float32()))
    wo_, _, _, vv = cl1.sample(wi_, rr.next_float32(), mi.Point2f(rr.next_float32(), rr.next_float32()), True)
    ph_ = rr.next_float32() * (2 * math.pi)
    rot = lambda v: mi.Vector3f(dr.cos(ph_) * v.x - dr.sin(ph_) * v.y, dr.sin(ph_) * v.x + dr.cos(ph_) * v.y, v.z)
    f0 = cl1.eval(wi_, wo_, vv); f1 = cl1.eval(rot(wi_), rot(wo_), vv)
    dr.eval(f0, f1); dr.sync_thread()
    f0n, f1n = np.asarray(f0.x), np.asarray(f1.x)
    iso = float(np.abs(f1n - f0n).sum() / max(np.abs(f0n).sum(), 1e-30))
    T['T_FRAMES'] = {'excluded_coincident': int((~same).sum()), 'tangent_err_rect': ex_r, 'tangent_err_box': ex_b, 'normal_err': ez, 'S1_azimuthal_anisotropy_relL1': iso,
                     'PASS': ex_r <= 1e-3 and ez <= 1e-3 and (ex_b <= 1e-3 or iso <= 0.02),
                     'note': 'box-face tangent frames may differ from the patch frame by a rotation about n; admissible if S1 is azimuthally isotropic'}
    # dense16 angular quadrature (declared approximation of area-sample scattering): albedo per cell vs exact fbar at random directions
    cd = RL.cell_dirs(RP['dense_sub'])
    ix = rng.choice(len(cd), 64, replace=False)
    zz = 1e-9
    dd = {}
    okd = True
    for m in (0, 1):
        jit = cd[ix] + rng.normal(0, 0.004, (64, 3)); jit /= np.linalg.norm(jit, axis=1, keepdims=True)
        ex = Q.fbar(st_[m], jit, 16384 if not DEV else 2048, 88950 + m)
        dn = d16.T[m][torch.as_tensor(ix, device=RL.dev)].cpu().numpy()
        zc = (np.arange(NB) // S.NPHI + 0.5) / S.NZ
        al_e = (ex.mean(-1) * zc).sum(1); al_d = (dn.mean(-1) * zc).sum(1)
        e_alb = float(np.median(np.abs(al_d - al_e) / np.maximum(al_e, 1e-9)))
        e_bin = float(np.median(np.abs(dn - ex).sum((1, 2)) / np.maximum(ex.sum((1, 2)), 1e-9)))
        dd[f'm{m}'] = {'albedo_rel_median': e_alb, 'per_bin_relL1_median': e_bin}
        okd &= e_alb <= 0.02 + (0.03 if DEV else 0)
    T['T_DENSE'] = {**dd, 'PASS': bool(okd), 'note': 'per-bin error includes the within-cell direction offset (jitter 0.004) and both estimators noise'}
    # unbiasedness on a known field: L = 1 on every peaked patch (all bins) -> L2 at Lambert receivers = rho / pi * (projected solid angle
    # of visible peaked patches). R0 pooled vs R1 pooled vs pure-area estimator (m = 256)
    Lk = np.zeros((G.N, NB, 3), np.float32); Lk[pk[G.PMAT]] = 1.0
    Lt = torch.as_tensor(Lk, device=RL.dev)
    lamr = np.isin(G.PMAT, [3, 5]) & (G.PSID != S.SID_SPHERE)
    npool = 4 if DEV else 16
    acc0 = 0; acc1 = 0
    for ri in range(npool):
        us = X.realization_us(88300 + ri); X.set_realization(ctx, us)
        s2 = EC.build_state(ctx)
        qd = Q.Quad(us, 4096 if DEV else E87['M_ray'], 88310 + ri)
        acc0 = acc0 + X.XOp(G, s2.hit, s2.dbin, qd.W).apply(Lt).cpu().numpy()
        acc1 = acc1 + RL.SAM(ctx, qd, d16, RP['kappa_u'], RP['m'], 88320 + ri, st=s2).apply(Lt).cpu().numpy()
    X.set_realization(ctx, X.realization_us(86001))
    acc0 /= npool; acc1 /= npool
    pi_, pp_ = RL.pair_list(G, pk)
    Ep = run88.pair_energy(ctx, (pi_, pp_), Lk, 64 if DEV else 256, 88330)
    Ea = np.bincount(pi_, weights=Ep[:, 0], minlength=G.N)
    rho = fx88._FAM['wl88']['albedo'][G.PMAT] / math.pi
    ref = rho * Ea
    e0 = acc0[..., 0].mean(1); e1 = acc1[..., 0].mean(1)
    sel = lamr & (ref > 0)
    T['T_UNBIASED'] = {'R0_pool_vs_area_relL1': float(np.abs(e0[sel] - ref[sel]).sum() / ref[sel].sum()),
                       'R1_pool_vs_area_relL1': float(np.abs(e1[sel] - ref[sel]).sum() / ref[sel].sum()),
                       'R0_flux_ratio': float(e0[sel].sum() / ref[sel].sum()), 'R1_flux_ratio': float(e1[sel].sum() / ref[sel].sum())}
    T['T_UNBIASED']['PASS'] = abs(T['T_UNBIASED']['R1_flux_ratio'] - 1) <= 0.01 and abs(T['T_UNBIASED']['R0_flux_ratio'] - 1) <= 0.02 \
        and T['T_UNBIASED']['R1_pool_vs_area_relL1'] <= T['T_UNBIASED']['R0_pool_vs_area_relL1'] + 0.01
    # zero / linearity / non-negativity
    z = r1.apply(torch.zeros(G.N, NB, 3, device=RL.dev)).abs().max().item()
    a1 = torch.rand(G.N, NB, 3, device=RL.dev); a2 = torch.rand(G.N, NB, 3, device=RL.dev)
    lin = (r1.apply(2 * a1 + 3 * a2) - 2 * r1.apply(a1) - 3 * r1.apply(a2)).abs().max().item() / r1.apply(2 * a1 + 3 * a2).abs().max().item()
    neg = min(float(r1.a_w.min().item()), float(r1.mr.min().item()), float(min(t_.min().item() for t_ in d16.T)), float(r1.apply(a1).min().item()))
    T['T_ZERO_LIN_NONNEG'] = {'zero': z, 'linearity': lin, 'min_weight_or_output': neg, 'PASS': z == 0 and lin <= 1e-5 and neg >= 0}
    # source independence: construction takes no source; identical relation for any light set
    sig = list(inspect.signature(RL.SAM.__init__).parameters)
    ctx2 = PT.Ctx(g)
    st2 = EC.build_state(ctx2)
    r1b = RL.SAM(ctx2, quad, d16, RP['kappa_u'], RP['m'], RP['seed_area'], st=st2)
    T['T_SOURCE_INDEPENDENT'] = {'signature': sig, 'digest_a': r1.digest(), 'digest_b': r1b.digest(),
                                 'PASS': not any(k in sig for k in ('lights', 'lt', 'q', 'source')) and r1.digest() == r1b.digest()}
    T['T_DETERMINISTIC'] = {'PASS': r1.digest() == r1b.digest()}
    # edits: null, selective == full, move-and-return
    ex = FX[cfg.split('_')[0]]
    ctxe = PT.Ctx(g0)
    ste = EC.build_state(ctxe)
    re_ = RL.SAM(ctxe, quad, d16, RP['kappa_u'], RP['m'], RP['seed_area'], st=ste)
    dg0 = re_.digest()
    Tn = re_.update(ctxe, ex['moved'])
    null_ok = re_.digest() == dg0
    ctxe.set_geom(g)
    Tu = re_.update(ctxe, ex['moved'])
    full = RL.SAM(ctxe, quad, d16, RP['kappa_u'], RP['m'], RP['seed_area'])
    sel_ok = re_.digest() == full.digest()
    ctxe.set_geom(g0)
    re_.update(ctxe, ex['moved'])
    mar_ok = re_.digest() == dg0
    T['T_EDIT'] = {'null_edit_identical': null_ok, 'null_dirty_samples': Tn.get('area_dirty_samples'), 'selective_equals_full': sel_ok,
                   'move_and_return_identical': mar_ok, 'edit_dirty_samples': Tu.get('area_dirty_samples'), 'edit_u_dirty': Tu.get('u_dirty'),
                   'PASS': bool(null_ok and sel_ok and mar_ok)}
    # family accounting: render_fam reproduces the pt86 estimator and the exclusive classes sum to h
    o, _ = pt88.render_fam(ctx, lt, FX['camera'], 128, 16, 88990)
    acc, _ = ctx.render(lt, 'ER', FX['camera'], 128, 16, 88990, 'cam')
    o86 = PT.pack(acc, 128, 16)
    fs = o['Hb'] + o['Hm'] + o['H1P'] + o['H1G'] + o['H2']
    T['T_FAMILY'] = {'e1': rel(o['e1'], o86['e1']), 'e2': rel(o['e2'], o86['e2']), 'h': rel(o['h'], o86['h']), 'sum_classes_minus_h': rel(fs, o['h']),
                     'H2G_le_H2': bool((o['H2G'] <= o['H2'] + 1e-6).all())}
    T['T_FAMILY']['PASS'] = max(T['T_FAMILY'][k] for k in ('e1', 'e2', 'h', 'sum_classes_minus_h')) <= 1e-5 and T['T_FAMILY']['H2G_le_H2']
    # report only: R1 relation size / no double counting statement
    T['report'] = {'pairs': int(len(r1.pi)), 'area_valid': r1.n_area_valid, 'rays_R1': r1.rays(), 'rays_R0': int(G.N * K),
                   'no_double_counting': 'B = L2..L5 excludes q (as WL87); MIS weights of the u and a proposals sum to 1 for every first-hit-peaked direction by construction (balance heuristic, same D)'}
    gated = {k: v['PASS'] for k, v in T.items() if isinstance(v, dict) and 'PASS' in v}
    T['ALL_PASS'] = bool(all(gated.values()))
    run88.jd(T, os.path.join(R88, 'results', 'tests88.json'))
    print(json.dumps(gated), flush=True)
    print('TESTS', 'PASS' if T['ALL_PASS'] else 'FAIL', flush=True)


if __name__ == '__main__':
    dr.set_flag(dr.JitFlag.KernelHistory, True)
    main()
    os._exit(0)
