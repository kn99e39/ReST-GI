"""WL91 focused correctness tests of the isolated C3URX1 candidate. python tests91.py <R91> <R89> <R90> [--dev] -> results/tests91.json
Last line TESTS PASS|FAIL. Any failure stops the batch (STOPPED_INVALID_COMPARISON).
"""
import inspect, json, math, os, sys
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fx91  # noqa: E402
import run88 as R8  # noqa: E402
PT, Q = fx91.PT, fx91.Q
mi, dr, S = PT.mi, PT.dr, PT.S
FX = PT.FX
P91, P89, E87 = fx91.P91, fx91.P89, fx91.E87
NB, K = S.NB, S.K
DEV = '--dev' in sys.argv


def mrel(a, b):
    return float((a - b).abs().max() / b.abs().max())


def main():
    import ecro86 as EC
    import rel88 as RL
    import rel89 as R9
    import gap90 as GP
    import cand91 as C
    R, R89, R90 = sys.argv[1], sys.argv[2], sys.argv[3]
    cfg = 'W1_T1'
    geom, ls = R8.setup(R, cfg)
    ex = FX['W1']; g0 = ex['T0']
    ctx = PT.Ctx(geom); lt = PT.Lights(FX['light_sets'][ls])
    G = ctx.G
    mat = R8.materials(ctx)
    S1 = P89['stage1']
    T = {}
    cand = C.C3URX1(ctx, S1['kappa_s'], S1['seed_send'], mat['dense8'])
    B, ords = cand.table(ctx, lt, orders=True)
    # ---- equals the frozen WL90 reference functions and the stored WL90 table
    rc = R9.SendRelR(ctx, S1['kappa_s'], S1['seed_send'], mat['dense8'])
    rbc = GP.c3ur_raybuf(rc)
    Bref = GP.rec_from(rc, GP.c3ur_first(rc, rbc.radiance(ctx, lt)))
    T['T_WL90_MECHANISM'] = {'vs_gap90_functions_maxrel': mrel(B, Bref), 'relation_digest_equal': cand.rel.digest() == rc.digest()}
    fn = os.path.join(R90, 'results', 'tables', 'F_W1_T1.npz')
    if os.path.exists(fn) and not DEV:
        z = np.load(fn)
        T['T_WL90_MECHANISM']['vs_stored_WL90_table_maxrel'] = float(np.abs(B.cpu().numpy() - z['C3URX1_s86001']).max() / z['C3URX1_s86001'].max())
        q = Q.qtable(ctx, lt, E87['P'], E87['Mq'], E87['seed_q'])
        T['T_WL90_MECHANISM']['C3UR_vs_stored_WL90_table_maxrel'] = float(np.abs(C.c3ur_table(cand.rel, q).cpu().numpy() - z['C3UR_s86001']).max() / z['C3UR_s86001'].max())
    T['T_WL90_MECHANISM']['PASS'] = bool(all(v <= 1e-5 for k, v in T['T_WL90_MECHANISM'].items() if k.endswith('maxrel')) and T['T_WL90_MECHANISM']['relation_digest_equal'])
    # ---- finite / non-negative; zero light and fully occluded light give zero
    off = PT.Lights([{'pos': FX['light_sets'][ls][0]['pos'], 'rgb': [0.0, 0.0, 0.0]}])
    Bz = cand.table(ctx, off)
    spot = PT.Lights([{'pos': [0.0, 0.0, 1.5], 'rgb': [30.0, 30.0, 30.0], 'aim': [0.0, 0.0, 9.0], 'cone_deg': 0.001}])     # cone lights nothing
    Bn = cand.table(ctx, spot)
    T['T_ZERO_FINITE_NONNEG'] = {'zero_light_max': float(Bz.abs().max()), 'no_visibility_max_over_ref': float(Bn.abs().max() / B.abs().max()),
                                 'finite': bool(torch.isfinite(B).all()), 'min': float(B.min()),
                                 'PASS': bool(Bz.abs().max() == 0 and Bn.abs().max() / B.abs().max() <= 1e-6 and torch.isfinite(B).all() and B.min() >= 0)}
    # ---- linearity in the source (two lights = sum of single lights) and current-source dependence without stale state
    la, lb = FX['light_sets']['LMIX89']
    Ba = cand.table(ctx, PT.Lights([la])); Bb = cand.table(ctx, PT.Lights([lb])); Bab = cand.table(ctx, PT.Lights([la, lb])); Ba2 = cand.table(ctx, PT.Lights([la]))
    T['T_SOURCE_CURRENT'] = {'two_lights_vs_sum_maxrel': mrel(Bab, Ba + Bb), 'A_after_B_equals_A_maxrel': mrel(Ba2, Ba), 'A_differs_from_B': mrel(Ba, Bb) > 0.05,
                             'relation_digest_unchanged': cand.rel.digest() == rc.digest()}
    T['T_SOURCE_CURRENT']['PASS'] = bool(T['T_SOURCE_CURRENT']['two_lights_vs_sum_maxrel'] <= 1e-5 and T['T_SOURCE_CURRENT']['A_after_B_equals_A_maxrel'] <= 1e-5
                                         and T['T_SOURCE_CURRENT']['A_differs_from_B'] and T['T_SOURCE_CURRENT']['relation_digest_unchanged'])
    sig = list(inspect.signature(C.C3URX1.__init__).parameters) + list(inspect.signature(C.C3URX1.update).parameters)
    T['T_RELATION_SOURCE_INDEPENDENT'] = {'build_update_signature': sig, 'PASS': not any(k in sig for k in ('lights', 'lt', 'q', 'source'))}
    # ---- replay
    B2 = cand.table(ctx, lt)
    T['T_REPLAY'] = {'maxrel': mrel(B2, B), 'PASS': mrel(B2, B) <= 1e-5}
    # ---- edits: selective = full (relation digest, probe buffer, table), null edit, move-and-return, stale-geometry guard
    c0 = PT.Ctx(g0)
    ce = C.C3URX1(c0, S1['kappa_s'], S1['seed_send'], mat['dense8'])
    lt0 = lt
    B_t0 = ce.table(c0, lt0); d_t0 = ce.rel.digest()
    ce.update(c0, ex['moved'])
    null_ok = ce.rel.digest() == d_t0 and mrel(ce.table(c0, lt0), B_t0) <= 1e-5
    c0.set_geom(geom)
    ce.update(c0, ex['moved'])
    full = C.C3URX1(c0, S1['kappa_s'], S1['seed_send'], mat['dense8'])
    po = lambda c: torch.stack([torch.stack([c.o[k].torch() for k in range(3)], 1), torch.stack([c.d[k].torch() for k in range(3)], 1)])
    sel = {'relation_digest_equal': ce.rel.digest() == full.rel.digest(), 'segments_equal': bool(torch.equal(ce.rel.seg_t, full.rel.seg_t)),
           'probe_buffer_equal': bool(torch.equal(po(ce), po(full))), 'table_maxrel': mrel(ce.table(c0, lt0), full.table(c0, lt0))}
    sel_ok = sel['relation_digest_equal'] and sel['segments_equal'] and sel['probe_buffer_equal'] and sel['table_maxrel'] <= 1e-5
    fresh = mrel(ce.table(c0, lt0), B)                                             # vs the candidate built on a fresh Ctx(T1) (different scene object)
    c0.set_geom(g0)
    ce.update(c0, ex['moved'])
    mar_ok = ce.rel.digest() == d_t0 and mrel(ce.table(c0, lt0), B_t0) <= 1e-5
    c0.set_geom(geom)
    ce.rel.update(c0, ex['moved'])                                                 # relation updated behind the candidate's back: the probe buffer is stale
    try:
        ce.table(c0, lt0); guard = False
    except AssertionError:
        guard = True
    T['T_EDIT'] = {'null_edit_identical': bool(null_ok), 'selective_vs_full': sel, 'selective_equals_full': bool(sel_ok), 'move_and_return_identical': bool(mar_ok),
                   'vs_fresh_scene_T1_maxrel': fresh, 'stale_geometry_guard_raises': guard, 'PASS': bool(null_ok and sel_ok and mar_ok and guard)}
    # ---- transport family / normalisation: orders of the candidate against the path-traced order fields (flux), pooled over draws
    oo = np.load(os.path.join(R89, 'results', 'fields', f'ORD_{cfg}.npz'))
    Tt = 0.5 * (oo['a'].astype(np.float64) + oo['b'])
    w = (G.PAREA[:, None] * ((np.arange(NB) // S.NPHI + 0.5) / S.NZ)[None])[..., None]
    nr = 2 if DEV else 8
    acc = [0, 0, 0, 0]
    for r in range(nr):
        cd = C.C3URX1(ctx, S1['kappa_s'], 91300 + r, mat['dense8'])
        _, od = cd.table(ctx, lt, orders=True)
        acc = [a + o.cpu().numpy().astype(np.float64) / nr for a, o in zip(acc, od)]
    fr = [float((acc[k] * w).sum() / (Tt[k + 1] * w).sum()) for k in range(4)]
    T['T_ORDER_FLUX'] = {'flux_ratio_orders_2_5': fr, 'PASS': bool(all(abs(f - 1) <= 0.03 + (0.03 if DEV else 0) for f in fr))}
    gated = {k: v['PASS'] for k, v in T.items()}
    T['ALL_PASS'] = bool(all(gated.values()))
    R8.jd(T, os.path.join(R, 'results', 'tests91.json'))
    print(json.dumps(gated), flush=True)
    print(json.dumps({'wl90': T['T_WL90_MECHANISM'], 'flux': fr, 'edit': T['T_EDIT']['selective_vs_full']}), flush=True)
    print('TESTS', 'PASS' if T['ALL_PASS'] else 'FAIL', flush=True)


if __name__ == '__main__':
    dr.set_flag(dr.JitFlag.KernelHistory, True)
    main()
    os._exit(0)
