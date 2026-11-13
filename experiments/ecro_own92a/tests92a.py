"""WL92A focused tests (protocol 'tests').

  python tests92a.py pre <R> <R86> <R92>   T_REPRO, T_STATE (E4), T_DISJOINT, T_REPLAY (frozen)  -> results/tests_pre.json
  python tests92a.py f2  <R>               T_F2_SUM over every regenerated lighting                -> results/tests_f2.json
Post-training contracts (T_INIT, T_CTL, T_LINEAR, T_OWNERSHIP_CODE, T_REPLAY of every model) are in own92a.py fieldeval.
"""
import glob, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'ecro_attr92'))
DEV = '--dev' in sys.argv


def pre(R, R86, R92):
    import torch
    import fx92
    fx92.inject(R)
    import ecro86 as EC
    import model86 as M
    PT, FX, S = fx92.PT, fx92.FX, fx92.S
    out = {}
    rows = {}
    with torch.no_grad():
        for tag in ('E2_T1', 'E2_T0', 'E1_T1'):
            geom, ls, e, cfg = fx92.cfg_ls(tag)
            ctx = PT.Ctx(geom); st = EC.build_state(ctx)
            s = torch.as_tensor(ctx.source(PT.Lights(FX['light_sets'][ls])), device=EC.dev).view(1, ctx.N, S.NB, 3)
            gs = M.GeoState(ctx.G, st.hit, st.dbin, EC.dev)
            for sd in fx92.SEEDS:
                mdl = EC.load_model(R86, sd)
                B = mdl(gs, s)[0].cpu().numpy()
                ref = np.load(os.path.join(R86, 'results', 'renders', f'EC_{sd}_ecro_{tag}_r0.npz'))['B']
                rows[f'{sd}:{tag}'] = float(np.abs(B - ref).max() / max(ref.max(), 1e-30))
                if tag == 'E2_T1':
                    out['T_REPLAY'] = {'max': float((mdl(gs, s) - mdl(gs, s)).abs().max())}
    out['T_REPRO'] = {'max_rel': rows, 'PASS': bool(max(rows.values()) <= 1e-4)}
    out['T_REPLAY']['PASS'] = out['T_REPLAY']['max'] == 0.0
    ex = FX['E4']
    ctx = PT.Ctx(ex['T0'])
    a = EC.build_state(ctx); b = EC.build_state(ctx)
    ctx.set_geom(ex['T1'])
    sel, nd = EC.update_state(ctx, a, ex['moved'])
    full = EC.build_state(ctx); fresh = EC.build_state(PT.Ctx(ex['T1']))
    nul, n0 = EC.update_state(ctx, sel, [])
    out['T_STATE'] = {'rebuild_identical': a.digest() == b.digest(), 'selective_equals_full': sel.digest() == full.digest(), 'equals_fresh_scene': full.digest() == fresh.digest(),
                      'null_edit_identical': bool(nul.digest() == sel.digest() and n0 == 0), 'dirty_rays': int(nd)}
    out['T_STATE']['PASS'] = bool(all(v for k, v in out['T_STATE'].items() if isinstance(v, bool)))
    exj = json.load(open(os.path.join(R, 'results', 'exclusion.json')))
    md = exj['min_distance_kept']
    e4r = glob.glob(os.path.join(R92, 'results', 'renders', '*E4*')) + glob.glob(os.path.join(R, 'results', 'renders', '*E4*'))
    out['T_DISJOINT'] = {'min_distance_kept': md, 'excluded': {k: len(v) for k, v in exj['excluded'].items()}, 'e4_radiance_before_refs': len(e4r),
                         'PASS': bool(all(v['box_m'] >= 0.15 and v['light_m'] >= 0.3 for v in md.values()) and not e4r)}
    out['ALL_PASS'] = bool(all(v['PASS'] for v in out.values() if isinstance(v, dict)))
    json.dump(out, open(os.path.join(R, 'results', 'tests_pre.json'), 'w'), indent=1)
    for k, v in out.items():
        if isinstance(v, dict):
            print(k, 'PASS' if v['PASS'] else 'FAIL', json.dumps(v)[:300], flush=True)
    print('TESTS_PRE', 'PASS' if out['ALL_PASS'] else 'FAIL', flush=True)


def f2(R):
    rows = []
    for sp in ('train', 'val', 'trainhq'):
        for fn in sorted(glob.glob(os.path.join(R, 'data', sp + '_f2', 'check_*.json'))):
            for r in json.load(open(fn))['rows']:
                rows.append(dict(r, split=sp))
    kept = json.load(open(os.path.join(R, 'results', 'exclusion.json')))['kept']
    have = {sp: len(glob.glob(os.path.join(R, 'data', sp + '_f2', 'g*.npz'))) for sp in ('train', 'val', 'trainhq')}
    mx = max((r['max_rel_l1'] for r in rows), default=None); mf = max((r['max_flux_dev'] for r in rows), default=None)
    ok = bool(rows) and mx <= 1e-3 and mf <= 1e-3 and (DEV or all(have[k] == kept[k] for k in have))
    out = {'files': have, 'kept': kept, 'max_rel_l1': mx, 'max_flux_dev': mf, 'worst': sorted(rows, key=lambda r: -r['max_rel_l1'])[:5], 'PASS': ok}
    json.dump(out, open(os.path.join(R, 'results', 'tests_f2.json'), 'w'), indent=1)
    print('T_F2_SUM', 'PASS' if ok else 'FAIL', json.dumps({k: v for k, v in out.items() if k != 'worst'}), flush=True)


if __name__ == '__main__':
    if sys.argv[1] == 'pre':
        pre(sys.argv[2], sys.argv[3], sys.argv[4])
    else:
        f2(sys.argv[2])
    os._exit(0)
