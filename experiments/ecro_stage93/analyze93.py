"""WL93 analysis by the prospective rules (protocol ecro_stage93). numpy only.
  python analyze93.py <R> <R86> <R92A> <R92B>   -> results/final93.json ; prints LABEL <label>
"""
import glob, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for d in ('ecro86', 'ecro_attr92'):
    sys.path.insert(0, os.path.join(HERE, '..', d))
import analyze92 as AN  # noqa: E402
import analyze86 as A  # noqa: E402

P = json.load(open(os.path.join(HERE, 'protocol', 'ecro_stage93.json')))
TH = P['thresholds']
ARMS = list(P['arms'])
SEEDS = P['training']['seeds']
REGS = ['stationary_all', 'A', 'C', 'D', 'obj_panel', 'obj_sphere']


def jl(fn):
    return json.load(open(fn)) if os.path.exists(fn) else None


def main():
    R, R86, R92A, R92B = sys.argv[1:5]
    roots = (R, R92B, R92A, R86)

    def ref(tag):
        for root in roots:
            if glob.glob(os.path.join(root, 'results', 'renders', f'REF_ER_{tag}_r*.npz')):
                return A.load(root, f'REF_ER_{tag}')
        return None
    out = {'recon92b': jl(os.path.join(R, 'results', 'recon92b.json')), 'ident': jl(os.path.join(R, 'results', 'ident.json')),
           'tests_pre': jl(os.path.join(R, 'results', 'tests93_pre.json')), 'tests_post': jl(os.path.join(R, 'results', 'tests93_post.json')),
           'local': jl(os.path.join(R, 'results', 'local93.json')), 'timing': jl(os.path.join(R, 'results', 'timing93.json'))}
    cands = jl(os.path.join(R, 'results', 'e6_candidates.json'))
    adm = [jl(f) for f in sorted(glob.glob(os.path.join(R, 'results', 'e6_admission_k*.json')))]
    ch = jl(os.path.join(R, 'results', 'e6_choice.json'))
    chosen = ch['chosen'] if ch else None
    out['e6'] = {'candidates': None if cands is None else {'disjoint': len(cands['disjoint']), 'geometry_admissible': [c['k'] for c in cands['geometry_admissible']], 'generated_up_to_k': cands['generated_up_to_k']},
                 'admissions': adm, 'chosen': chosen, 'dev_forced': bool(ch and ch.get('dev_forced'))}
    out['training'] = {f'{a}_s{s}': jl(os.path.join(R, 'results', 'models', f'{a}_s{s}.json')) for a in ARMS for s in SEEDS}
    for v in out['training'].values():
        if v:
            v.pop('log', None)
    # ---------------------------------------------------------------- per-condition fidelity, gates
    conds = {'E2': {'T': 'E2', 'src': ['E1_T1_LSPOT']}, 'E5': {'T': f"E5k{P['e5']['candidate']}", 'src': [f"E5k{P['e5']['candidate']}_T1_LSPOT", f"E5k{P['e5']['candidate']}_T1_LMIX"]}}
    if chosen is not None:
        conds['E6'] = {'T': f'E6k{chosen}', 'src': [f'E6k{chosen}_T1_LMIX', f'E6k{chosen}_T1_LSPOT']}
    models = [f'{a}_s{s}' for a in ARMS for s in SEEDS] + [f'{b}92a_s{s}' for b in ('own', 'sum') for s in SEEDS]
    F = {}
    for cn, c in conds.items():
        e = c['T']
        Mk = AN.mask_of(R, R86, f'{e}_T1')
        rf = {t: ref(f'{e}_{t}') for t in ('T0', 'T1')}
        diag = {t: jl(os.path.join(R, 'results', 'tables', f'I_{e}_{t}.json')) or {} for t in ('T0', 'T1')}
        o = {'models': {}, 'controls': {}}
        for n in models:
            ec = {t: AN.hl(R, f'I93_{n}_{e}_{t}') for t in ('T0', 'T1')}
            if any(v is None for v in ec.values()) or any(v is None for v in rf.values()):
                continue
            m = {'fidelity': {t: AN.eh(rf[t], ec[t], Mk, REGS) for t in ('T0', 'T1')}, 'dH': AN.dh(rf['T0'], rf['T1'], ec['T0'], ec['T1'], Mk), 'sources': {},
                 'diag': {t: diag[t].get(n) for t in ('T0', 'T1')}}
            for st in c['src']:
                x = AN.hl(R, f'I93_{n}_{st}'); rs = ref(st)
                if x is not None and rs is not None:
                    m['sources'][st] = AN.eh(rs, x, AN.mask_of(R, R86, st), ['stationary_all', 'C', 'D'])
            fi = m['fidelity']
            m['gates'] = {'F1': {t: bool(fi[t]['stationary_all']['E_H'] <= TH['EH_stationary_max'] and fi[t]['stationary_all']['capture'] <= TH['capture_max']) for t in ('T0', 'T1')},
                          'F2': {k: bool(fi['T1'][k]['E_H'] <= TH['EH_region_max']) for k in ('A', 'C', 'D') if k in fi['T1'] and fi['T1'][k]['pixels'] >= TH['min_pixels']},
                          'F3': m['dH'].get('D', {}).get('F3', 'NA'),
                          'F4': {s: bool(v['stationary_all']['E_H'] <= TH['EH_unseen_source_max']) for s, v in m['sources'].items()}}
            o['models'][n] = m
        for key in ('ORACLE', 'X_EXP'):
            for t in ('T0', 'T1'):
                x = AN.hl(R, f'I93_{key}_{e}_{t}')
                if x is not None and rf[t] is not None:
                    o['controls'][f'{key}_{t}'] = AN.eh(rf[t], x, Mk, REGS)
        F[cn] = o
    out['conditions'] = F
    # ---------------------------------------------------------------- T_REPRO (WL92A own re-rendered = WL92A's own renders, E2 region C)
    rep = {}
    for s in SEEDS:
        for t in ('T0', 'T1'):
            mine = AN.hl(R, f'I93_own92a_s{s}_E2_{t}'); theirs = AN.hl(R92A, f'I92A_own_s{s}_E2_{t}')
            rf = ref(f'E2_{t}')
            if mine is None or theirs is None or rf is None:
                rep[f'own_s{s}:E2_{t}'] = None
                continue
            Mk = AN.mask_of(R, R86, f'E2_{t}')
            a, b = AN.eh(rf, mine, Mk, ['C'])['C']['E_H'], AN.eh(rf, theirs, Mk, ['C'])['C']['E_H']
            rep[f'own_s{s}:E2_{t}'] = {'wl93': a, 'wl92a': b, 'abs_diff': abs(a - b), 'max_abs_hl_diff': float(np.abs(mine['hl'] - theirs['hl']).max())}
    out['repro'] = {'cases': rep, 'PASS': bool(rep and all(v is not None and v['abs_diff'] <= 1e-6 for v in rep.values()))}
    # ---------------------------------------------------------------- criteria
    def mo(cn, a, s):
        return F.get(cn, {}).get('models', {}).get(f'{a}_s{s}')

    def ADV(X, Y):
        det = {}
        ok = 'E6' in F
        for s in SEEDS:
            for t in ('T1', 'T0'):
                x, y = mo('E6', X, s), mo('E6', Y, s)
                if x is None or y is None or 'C' not in x['fidelity'][t]:
                    ok = False; det[f's{s}:{t}'] = None
                    continue
                xc, yc = x['fidelity'][t]['C'], y['fidelity'][t]['C']
                mg = max(xc['noise_bound'], yc['noise_bound'], 0.005)
                p = bool(xc['E_H'] <= yc['E_H'] - mg)
                det[f's{s}:{t}'] = {'X': xc['E_H'], 'Y': yc['E_H'], 'margin': mg, 'pass': p}
                ok &= p
        return bool(ok), det

    def STAGE(X, Y):
        det = {}; ok = 'E6' in F
        for s in SEEDS:
            for cn in ('E6', 'E2'):
                x, y = mo(cn, X, s), mo(cn, Y, s)
                v = lambda m: (((m or {}).get('diag') or {}).get('T1') or {}).get('Bq_vs_q_panel_rel_l1')   # Amendment 1: None when absent (stop path)
                xv, yv = v(x), v(y)
                if xv is None or yv is None:
                    ok = False; det[f's{s}:{cn}'] = None
                    continue
                p = bool(xv <= 0.9 * yv); det[f's{s}:{cn}'] = {'X': xv, 'Y': yv, 'pass': p}; ok &= p
        return bool(ok), det

    def NOREG(X):
        lost, worse, missing = [], [], []
        for s in SEEDS:
            for cn in ('E2', 'E5', 'E6'):
                x, b = mo(cn, X, s), mo(cn, 'T0', s)
                if x is None or b is None:
                    missing.append([cn, s]); continue
                for gk in ('F1', 'F4'):
                    for k, v in b['gates'][gk].items():
                        if v and not x['gates'][gk].get(k, False):
                            lost.append([s, cn, gk, k])
                for k in ('A', 'D'):
                    if b['gates']['F2'].get(k) and not x['gates']['F2'].get(k, False):
                        lost.append([s, cn, 'F2', k])
                for t in ('T0', 'T1'):
                    for k in ('stationary_all', 'D'):
                        xa, xb = b['fidelity'][t].get(k), x['fidelity'][t].get(k)
                        if xa and xb and xb['E_H'] > 1.25 * xa['E_H'] + 0.002:
                            worse.append([s, cn, t, k, xb['E_H'], xa['E_H']])
                for st, v in b['sources'].items():
                    xb = x['sources'].get(st, {}).get('stationary_all'); xa = v.get('stationary_all')
                    if xa and xb and xb['E_H'] > 1.25 * xa['E_H'] + 0.002:
                        worse.append([s, st, 'stationary_all', xb['E_H'], xa['E_H']])
                    if xa and not xb:
                        missing.append([st, s])
        return bool(not lost and not worse and not missing), {'lost': lost, 'worse': worse, 'missing': missing}

    def DYN(X):
        lost = []
        for s in SEEDS:
            for cn in ('E2', 'E5', 'E6'):
                x, b = mo(cn, X, s), mo(cn, 'T0', s)
                if x is None or b is None:
                    lost.append([s, cn, 'missing']); continue
                if b['gates']['F3'] == 'PASS' and x['gates']['F3'] != 'PASS':
                    lost.append([s, cn, x['gates']['F3']])
        return bool(not lost), {'lost': lost}

    def PASS(X, Y, stage):
        a, ad = ADV(X, Y); n, nd = NOREG(X); d, dd = DYN(X)
        r = {'ADV': {'pass': a, 'detail': ad}, 'NOREG': {'pass': n, 'detail': nd}, 'DYN': {'pass': d, 'detail': dd}}
        ok = a and n and d
        if stage:
            g, gd = STAGE(X, Y); r['STAGE'] = {'pass': g, 'detail': gd}; ok = ok and g
        r['PASS'] = bool(ok)
        return r
    C = {'PASS_sup(TQ,T0)': PASS('TQ', 'T0', True), 'PASS_sup(UQ,U0)': PASS('UQ', 'U0', True), 'PASS_sup(UQ,T0)': PASS('UQ', 'T0', True),
         'PASS_share(U0,C0)': PASS('U0', 'C0', False), 'PASS_share(UQ,CQ)': PASS('UQ', 'CQ', False), 'PASS_share(UQ,TQ)': PASS('UQ', 'TQ', False),
         'PASS_sup(CQ,C0)': PASS('CQ', 'C0', True)}
    out['criteria'] = C
    p = lambda k: C[k]['PASS']
    # ---------------------------------------------------------------- stop conditions and label
    stops = []
    if not (out['ident'] and out['ident'].get('PASS')):
        stops.append('STOP_TARGET_NOT_IDENTIFIABLE')
    if chosen is None or out['e6']['dev_forced']:
        stops.append('STOPPED_WORKLOAD_UNQUALIFIED')
    if not (out['tests_pre'] and out['tests_pre'].get('PASS')) or not (out['tests_post'] and out['tests_post'].get('PASS')):
        stops.append('STOPPED_TEST_FAILURE')
    if not out['repro']['PASS']:
        stops.append('STOPPED_BASELINE_NOT_REPRODUCED')
    miss = [k for k, v in out['training'].items() if v is None or v.get('nonfinite_steps')]
    if miss:
        stops.append('STOPPED_TRAINING_INCOMPLETE')
    if stops:
        lab = 'STOPPED-INCONCLUSIVE'
    elif p('PASS_sup(UQ,T0)') and p('PASS_share(UQ,CQ)') and not p('PASS_sup(TQ,T0)') and not p('PASS_share(U0,C0)'):
        lab = 'INTERACTION_SUPPORTED'
    elif p('PASS_share(UQ,CQ)') and p('PASS_share(UQ,TQ)'):
        lab = 'SHARING_SUPPORTED'
    elif p('PASS_sup(TQ,T0)') and not p('PASS_share(UQ,CQ)'):
        lab = 'IDENTIFIABILITY_SUPPORTED'
    elif not any(p(k) for k in ('PASS_sup(TQ,T0)', 'PASS_sup(UQ,U0)', 'PASS_share(U0,C0)', 'PASS_share(UQ,CQ)', 'PASS_sup(UQ,T0)')):
        lab = 'NOT_SUPPORTED'
    else:
        lab = 'INCONCLUSIVE'
    out['label'] = {'label': lab, 'stops': stops, 'note': 'PASS_sup(CQ,C0) is descriptive (not in the label mapping); labels do not authorise adoption'}
    json.dump(out, open(os.path.join(R, 'results', 'final93.json'), 'w'), indent=1)
    print('LABEL', lab, 'stops', stops, flush=True)
    for k, v in C.items():
        print(k, v['PASS'], json.dumps({kk: vv['pass'] for kk, vv in v.items() if kk != 'PASS'}), flush=True)
    for cn, o in F.items():
        for t in ('T1', 'T0'):
            print(cn, t, 'C', json.dumps({n: round(m['fidelity'][t].get('C', {}).get('E_H', -1), 4) for n, m in o['models'].items()}), flush=True)


if __name__ == '__main__':
    main()
