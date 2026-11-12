"""WL92 focused tests (protocol 'tests'). Run after the truth split, the ladder tables and the fit data, before any render or fit.

  python tests92.py <R> <R86> [--dev]   -> results/tests92.json ; prints TESTS PASS | TESTS FAIL
"""
import glob, json, math, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
DEV = '--dev' in sys.argv


def main():
    R, R86 = sys.argv[1], sys.argv[2]
    import torch
    import fx92
    fx92.inject(R)
    import ecro86 as EC
    import model86 as M
    import diag92 as DG
    PT, FX, S, P92 = fx92.PT, fx92.FX, fx92.S, fx92.P92
    NB = S.NB
    dev = EC.dev
    HIST = P92['conditions']['historical']
    out = {}
    # ---- T_REPRO_TABLE: frozen checkpoints reproduce the tables stored with the WL86 renders
    rows = {}
    for tag in HIST:
        T = np.load(os.path.join(R, 'results', 'tables', f'T_{tag}.npz'))
        for sd in fx92.SEEDS:
            st = np.load(os.path.join(R86, 'results', 'renders', f'EC_{sd}_ecro_{tag}_r0.npz'))
            if 'B' in st.files:
                rows[f'{sd}:{tag}'] = float(np.abs(T[f'B_{sd}'] - st['B']).max() / max(st['B'].max(), 1e-30))
    out['T_REPRO_TABLE'] = {'max_rel': rows, 'PASS': bool(rows and max(rows.values()) <= 1e-4)}
    # ---- T_SPLIT_SUM: path-traced parts against the stored WL86 oracle field
    rows = {}
    ok = True
    for tag in HIST:
        D = np.load(os.path.join(R, 'results', 'fields', f'D_{tag}.npz'))
        Wz = np.load(os.path.join(R, 'results', 'fields', f'W_{tag}.npz'))
        F0 = np.load(os.path.join(R86, 'results', 'fields', f'F_{tag}.npz'))
        sa = sum(D[f'{k}_a'].astype(np.float64) for k in DG.PARTS); sb = sum(D[f'{k}_b'].astype(np.float64) for k in DG.PARTS)
        fa, fb = F0['fa'].astype(np.float64), F0['fb'].astype(np.float64)
        flux = float((sa + sb).sum() / (fa + fb).sum())
        mn = min(float(D[f'{k}_{h}'].min()) for k in DG.PARTS for h in 'ab')
        r = {'flux_ratio': flux, 'min_part': mn}
        if 'C' in Wz.files:
            W = Wz['C'].astype(np.float64).reshape(-1, NB, 3)
            diff = float((W * (0.5 * (sa + sb) - 0.5 * (fa + fb))).sum())
            sig = math.sqrt(float(((W * (sa - sb) * 0.5) ** 2).sum()) + float(((W * (fa - fb) * 0.5) ** 2).sum()))      # slots are independent estimates
            base = float((W * 0.5 * (fa + fb)).sum())
            r.update({'C_read_diff_over_h': diff / base, 'C_read_sigma_over_h': sig / base, 'z': diff / max(sig, 1e-30)})
            ok &= abs(diff) <= 4 * sig or DEV
        ok &= (abs(flux - 1) <= 0.01 or DEV) and mn >= 0
        rows[tag] = r
    out['T_SPLIT_SUM'] = {'cases': rows, 'PASS': bool(ok)}
    # ---- T_OWNERSHIP_CODE and T_FORM (from the ladder metadata) + constructed factor sets
    metas = {tag: json.load(open(os.path.join(R, 'results', 'tables', f'T_{tag}.json'))) for tag in HIST}
    own = max(v for m in metas.values() for v in m['own_sum'].values())
    out['T_FORM'] = {'rel_l1': {t: m['form_rel_l1_ENCB_vs_binq'] for t, m in metas.items()}, 'PASS': bool(max(m['form_rel_l1_ENCB_vs_binq'] for m in metas.values()) <= 0.05 or DEV)}
    # ---- T_LINEAR on the frozen, exact and rank-16 factor sets (E2_T1 geometry)
    D = np.load(os.path.join(R, 'results', 'fields', 'D_E2_T1.npz'))
    nm = np.load(os.path.join(R, 'results', 'nmf.npz'))
    G = S.Geom(FX['E2']['T1'])
    gs = M.GeoState(G, D['hit'], D['dbin'], dev)
    mdl = EC.load_model(R86, fx92.SEEDS[0])
    s1 = torch.as_tensor(D['s16'], device=dev).view(1, G.N, NB, 3)
    s2 = torch.roll(s1, 97, 1) * 0.5
    lin = {}
    with torch.no_grad():
        fz = mdl.factors(gs)
        sets = {'frozen': fz, 'R16': dict(DG.f_nmf(G, nm['Cn'], nm['An'], dev), Q=fz['Q'], P=fz['P']), 'M64': dict(DG.f_exact(G, nm['Mq'].astype(np.float64), dev), Q=None, P=None)}
        for k, f in sets.items():
            kw = {'use_global': False} if f.get('Q') is None else {}
            a, b, ab = mdl.apply(f, gs, s1, **kw), mdl.apply(f, gs, s2, **kw), mdl.apply(f, gs, s1 + s2, **kw)
            sc = float(ab.abs().max())
            lin[k] = {'superposition': float((ab - a - b).abs().max() / sc), 'scaling': float((mdl.apply(f, gs, 3.0 * s1, **kw) - 3.0 * a).abs().max() / sc),
                      'zero': float(mdl.apply(f, gs, torch.zeros_like(s1), **kw).abs().max()), 'min': float(a.min())}
            if f.get('Q') is not None:
                lin[k]['single_plus_global_minus_B'] = float((mdl.apply(f, gs, s1, use_global=False) + mdl.apply(f, gs, s1, use_single=False) - a).abs().max() / sc)
                own = max(own, lin[k]['single_plus_global_minus_B'])
    out['T_LINEAR'] = {'sets': lin, 'PASS': bool(all(v['superposition'] <= 1e-4 and v['scaling'] <= 1e-4 and v['zero'] == 0.0 and v['min'] >= 0 for v in lin.values()))}
    out['T_OWNERSHIP_CODE'] = {'max_single_plus_global_minus_B': float(own), 'parts_nonnegative': bool(all(r['min_part'] >= 0 for r in rows.values())),
                               'PASS': bool(own <= 1e-5 and all(r['min_part'] >= 0 for r in rows.values()))}
    # ---- T_STATE: deterministic rebuild, selective == full, null edit
    ex = FX['E2']
    ctx = PT.Ctx(ex['T0'])
    a = EC.build_state(ctx); b = EC.build_state(ctx)
    ctx.set_geom(ex['T1'])
    sel, nd = EC.update_state(ctx, a, ex['moved'])
    full = EC.build_state(ctx)
    fresh = EC.build_state(PT.Ctx(ex['T1']))
    nul, n0 = EC.update_state(ctx, sel, [])
    out['T_STATE'] = {'rebuild_identical': bool(a.digest() == b.digest()), 'selective_equals_full': bool(sel.digest() == full.digest()), 'equals_fresh_scene': bool(full.digest() == fresh.digest()),
                      'null_edit_identical': bool(nul.digest() == sel.digest() and n0 == 0), 'dirty_rays': int(nd), 'matches_stored_T1': bool(np.array_equal(full.hit, D['hit']))}
    out['T_STATE']['PASS'] = bool(all(v for k, v in out['T_STATE'].items() if isinstance(v, bool)))
    # ---- T_DISJOINT
    fj = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(R, 'results', 'fits', 'fitdata_*.json')))]
    e4 = os.path.join(R, 'results', 'e4_choice.json')
    out['T_DISJOINT'] = {'fit_light_min_distance_m': [f['min_distance_to_test_light_m'] for f in fj], 'e4_choice_recorded': os.path.exists(e4),
                         'e4_radiance_absent': not glob.glob(os.path.join(R, 'results', 'renders', 'REF_*E4*')),
                         'PASS': bool(fj and min(f['min_distance_to_test_light_m'] for f in fj) >= 0.3 and os.path.exists(e4) and not glob.glob(os.path.join(R, 'results', 'renders', 'REF_*E4*')))}
    out['ALL_PASS'] = bool(all(v['PASS'] for v in out.values() if isinstance(v, dict)))
    json.dump(out, open(os.path.join(R, 'results', 'tests92.json'), 'w'), indent=1)
    for k, v in out.items():
        if isinstance(v, dict):
            print(k, 'PASS' if v['PASS'] else 'FAIL', json.dumps({a: (round(b, 6) if isinstance(b, float) else b) for a, b in v.items() if not isinstance(b, dict)})[:300], flush=True)
    print('TESTS', 'PASS' if out['ALL_PASS'] else 'FAIL', flush=True)


if __name__ == '__main__':
    main()
    os._exit(0)
