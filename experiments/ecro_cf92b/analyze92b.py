"""WL92B analysis by the prospective rules (protocol ecro_cf92b). numpy only.
  python analyze92b.py <R> <R86> <R92A>   -> results/final92b.json
"""
import glob, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'ecro_attr92'))
import analyze92 as AN  # noqa: E402
import analyze86 as A  # noqa: E402

P = json.load(open(os.path.join(HERE, 'protocol', 'ecro_cf92b.json')))
TH = P['thresholds']
MODELS = ['ecro_s8601', 'ecro_s8602', 'sum_s8601', 'sum_s8602', 'own_s8601', 'own_s8602']
CFS = ['ORIG', 'X_L2', 'X_L3', 'X_M1', 'X_M2', 'X_M12', 'X_Q', 'X_SHQ']
CASES = {'E2_T1': ['C', 'obj_panel', 'stationary_all'], 'E2_T0': ['C', 'obj_panel', 'stationary_all'], 'E1_T1_LSPOT': ['stationary_all', 'C'], 'E4_T1': ['C', 'stationary_all'], 'E4_T0': ['C', 'stationary_all']}
OWN = ['own_s8601', 'own_s8602']


def jl(fn):
    return json.load(open(fn)) if os.path.exists(fn) else None


def ref(R, R86, R92A, tag):
    for root in (R, R92A, R86):
        if glob.glob(os.path.join(root, 'results', 'renders', f'REF_ER_{tag}_r*.npz')):
            return A.load(root, f'REF_ER_{tag}')
    return None


def main():
    R, R86, R92A = sys.argv[1], sys.argv[2], sys.argv[3]
    out = {'repro': jl(os.path.join(R, 'results', 'repro.json')), 'tests': jl(os.path.join(R, 'results', 'tests92b.json')), 'cases': {}, 'local': jl(os.path.join(R, 'results', 'local92b.json'))}
    # ---------------------------------------------------------------- counterfactual attribution
    for tag, regs in CASES.items():
        meta = jl(os.path.join(R, 'results', 'tables', f'CF_{tag}.json'))
        rf = ref(R, R86, R92A, tag)
        if meta is None or rf is None:
            continue
        M = AN.mask_of(R, R86, tag)
        fl = AN.hl(R, f'X92B_ORA_{tag}')
        floor = AN.eh(rf, fl, M, regs) if fl is not None else {}
        o = {'floor': floor, 'models': {}, 'field': meta.get('field'), 'cf_identity': meta['cf_identity'], 'linear': meta['linear']}
        for n in MODELS:
            mo = {'E_H': {}, 'accounted': {}, 'reading': meta['reading'][n]}
            for cf in CFS:
                x = AN.hl(R, f'X92B_{cf}_{n}_{tag}')
                if x is not None:
                    mo['E_H'][cf] = AN.eh(rf, x, M, regs)
            for rk in regs:
                if 'ORIG' not in mo['E_H'] or rk not in mo['E_H']['ORIG'] or rk not in floor:
                    continue
                e0 = mo['E_H']['ORIG'][rk]['E_H']; ef = floor[rk]['E_H']
                den = max(e0 - ef, 1e-4)
                mo['accounted'][rk] = {cf: (e0 - mo['E_H'][cf][rk]['E_H']) / den for cf in CFS if cf in mo['E_H'] and rk in mo['E_H'][cf]}
                if 'X_M1' in mo['E_H'] and 'X_Q' in mo['E_H']:
                    mo['accounted'][rk]['encoding_effect'] = (mo['E_H']['X_M1'][rk]['E_H'] - mo['E_H']['X_Q'][rk]['E_H']) / den
            o['models'][n] = mo
        # seed gap closure for OWN
        gaps = {}
        for rk in regs:
            try:
                e = {cf: [o['models'][n]['E_H'][cf][rk]['E_H'] for n in OWN] for cf in CFS}
            except KeyError:
                continue
            g0 = abs(e['ORIG'][0] - e['ORIG'][1])
            gaps[rk] = {'orig_gap': g0, 'closed': {cf: (1 - abs(v[0] - v[1]) / max(g0, 1e-6)) for cf, v in e.items()}}
        o['own_seed_gap'] = gaps
        out['cases'][tag] = o
    # ---------------------------------------------------------------- hypotheses
    H = {}
    fv = lambda tag, rk, n, cf: out['cases'].get(tag, {}).get('models', {}).get(n, {}).get('accounted', {}).get(rk, {}).get(cf)

    def rule(cf, cases, rk):
        vals = [fv(t, rk, n, cf) for t in cases for n in OWN]
        if any(v is None for v in vals):
            return 'NOT TESTED', vals
        dom = all(min(fv(t, rk, n, cf) for n in OWN) >= 0.5 for t in cases)
        if dom:
            return 'SUPPORTED', vals
        if all(v < 0.25 for v in vals):
            return 'NOT SUPPORTED', vals
        return 'INCONCLUSIVE', vals
    e2 = ['E2_T1', 'E2_T0']
    H['H_local'] = dict(zip(('label', 'values'), rule('X_M12', e2, 'C')))
    H['H_global'] = dict(zip(('label', 'values'), rule('X_L3', e2, 'C')))
    H['single_term'] = dict(zip(('label', 'values'), rule('X_L2', e2, 'C')))
    H['local_source_side'] = dict(zip(('label', 'values'), rule('X_M1', e2, 'C')))
    H['local_receiver_side'] = dict(zip(('label', 'values'), rule('X_M2', e2, 'C')))
    # H_source on E1_T1_LSPOT (stationary)
    shq = [fv('E1_T1_LSPOT', 'stationary_all', n, 'X_SHQ') for n in OWN]
    enc = [fv('E1_T1_LSPOT', 'stationary_all', n, 'encoding_effect') for n in OWN]
    if any(v is None for v in shq + enc):
        H['H_source'] = {'label': 'NOT TESTED'}
    else:
        sup = min(shq) >= 0.5 or min(enc) >= 0.5
        H['H_source'] = {'label': 'SUPPORTED' if sup else ('NOT SUPPORTED' if max(shq) < 0.25 and max(enc) < 0.25 else 'INCONCLUSIVE'), 'X_SHQ': shq, 'encoding_effect': enc,
                         'other_LSPOT_stationary': {cf: [fv('E1_T1_LSPOT', 'stationary_all', n, cf) for n in OWN] for cf in ('X_L2', 'X_L3', 'X_M1', 'X_M2', 'X_M12', 'X_Q')}}
    L = out['local']
    if L:
        gv = [L['geometry_variation'][n]['E2_T0->E2_T1']['panel'] for n in OWN]
        H['H_geometry'] = {'label': 'SUPPORTED' if min(gv) >= 0.10 else ('NOT SUPPORTED' if max(gv) < 0.03 else 'INCONCLUSIVE'), 'G_panel_E2': gv,
                           'seed_variation_panel_E2_T1': L['seed_variation']['own']['E2_T1']['panel']}
    else:
        H['H_geometry'] = {'label': 'NOT TESTED'}
    inter = False
    for t in e2:
        if all(len([cf for cf in ('X_L2', 'X_L3', 'X_M1', 'X_M2', 'X_M12', 'X_Q') if (fv(t, 'C', n, cf) or 0) >= 0.25]) >= 2 for n in OWN):
            inter = True
    no_dom = not any(H[k]['label'] == 'SUPPORTED' for k in ('H_local', 'H_global'))
    H['H_interact'] = {'label': 'SUPPORTED' if (no_dom and inter) else ('NOT SUPPORTED' if not inter else 'INCONCLUSIVE')}
    out['hypotheses'] = H
    # ---------------------------------------------------------------- E5
    adm = [jl(f) for f in sorted(glob.glob(os.path.join(R, 'results', 'e5_admission_k*.json')))]
    cands = jl(os.path.join(R, 'results', 'e5_candidates.json'))
    out['e5'] = {'candidates': None if cands is None else {'disjoint': len(cands['disjoint']), 'geometry_admissible': [c['k'] for c in cands['geometry_admissible']], 'generated_up_to_k': cands['generated_up_to_k']},
                 'admissions': adm}
    chosen = next((a['candidate'] for a in adm if a and a['ADMITTED']), None)
    if chosen is None and os.environ.get('WL92B_DEV_FORCE_E5'):                 # dev plumbing only (never set in the evidence chain)
        chosen = int(os.environ['WL92B_DEV_FORCE_E5'])
    out['e5']['chosen'] = chosen
    C5 = {}
    if chosen is not None:
        e = f'E5k{chosen}'
        Mk = AN.mask_of(R, R86, f'{e}_T1')
        mods = {}
        for n in MODELS:
            m = {'fidelity': {}, 'sources': {}, 'stale': {}}
            ec = {t: AN.hl(R, f'I92B_{n}_{e}_{t}') for t in ('T0', 'T1')}
            rf = {t: ref(R, R86, R92A, f'{e}_{t}') for t in ('T0', 'T1')}
            m['fidelity'] = {t: AN.eh(rf[t], ec[t], Mk) for t in ('T0', 'T1')}
            m['dH'] = AN.dh(rf['T0'], rf['T1'], ec['T0'], ec['T1'], Mk)
            st = AN.hl(R, f'I92B_STALE_{n}_{e}_T1')
            if st is not None:
                m['stale'] = {k: v['E_H'] for k, v in AN.eh(rf['T1'], st, Mk).items()}
            for src in ('LMIX', 'LSPOT'):
                x = AN.hl(R, f'I92B_{n}_{e}_T1_{src}')
                rs = ref(R, R86, R92A, f'{e}_T1_{src}')
                if x is not None and rs is not None:
                    m['sources'][src] = AN.eh(rs, x, AN.mask_of(R, R86, f'{e}_T1_{src}'))
            g = {'F1': {t: bool(m['fidelity'][t]['stationary_all']['E_H'] <= TH['EH_stationary_max'] and m['fidelity'][t]['stationary_all']['capture'] <= TH['capture_max']) for t in ('T0', 'T1')},
                 'F2': {k: bool(m['fidelity']['T1'][k]['E_H'] <= TH['EH_region_max']) for k in ('A', 'C', 'D') if k in m['fidelity']['T1'] and m['fidelity']['T1'][k]['pixels'] >= TH['min_pixels']},
                 'F3': m['dH'].get('D', {}).get('F3', 'NA'), 'F4': {s: bool(v['stationary_all']['E_H'] <= TH['EH_unseen_source_max']) for s, v in m['sources'].items()}}
            m['gates'] = g
            # attribution (region C reading)
            fI, fD, fW = (os.path.join(R, 'results', d, f'{p}_{e}_T1.npz') for d, p in (('tables', 'I'), ('fields', 'D'), ('fields', 'W')))
            if all(os.path.exists(f) for f in (fI, fD, fW)):
                I, D, Wz = np.load(fI), np.load(fD), np.load(fW)
                hmf = lambda ks: sum(0.5 * (D[f'{k}_a'].astype(np.float64) + D[f'{k}_b']) for k in ks)
                L2T, L3T, E2P = hmf(('e2_p', 'e2_o')), hmf(('e3_p', 'e3_o')), hmf(('e2_p',))
                h = AN.reads(Wz, L2T + L3T, 'C')
                m['attribution_C'] = {'panel_family_share': AN.reads(Wz, E2P, 'C') / h, 'panel_family_delivered': AN.reads(Wz, I[f'SP_{n}'], 'C') / max(AN.reads(Wz, E2P, 'C'), 1e-30),
                                      'single_minus_L2_over_h': (AN.reads(Wz, I[f'S_{n}'], 'C') - AN.reads(Wz, L2T, 'C')) / h,
                                      'global_minus_L3_over_h': (AN.reads(Wz, I[f'G_{n}'], 'C') - AN.reads(Wz, L3T, 'C')) / h,
                                      'field_single_vs_L2': float(np.abs(I[f'S_{n}'] - L2T).sum() / L2T.sum()), 'field_global_vs_L3': float(np.abs(I[f'G_{n}'] - L3T).sum() / L3T.sum())}
            mods[n] = m
        ctl = {}
        for key in ('ORACLE', 'X_EXP'):
            for t in ('T0', 'T1'):
                x = AN.hl(R, f'I92B_{key}_{e}_{t}')
                if x is not None:
                    ctl[f'{key}_{t}'] = AN.eh(ref(R, R86, R92A, f'{e}_{t}'), x, Mk)
        out['e5']['models'] = mods; out['e5']['controls'] = ctl
        pair = [('sum_s8601', 'own_s8601'), ('sum_s8602', 'own_s8602')]
        adv = {}
        for a, b in pair:
            adv[b] = {}
            for t in ('T0', 'T1'):
                xa, xb = mods[a]['fidelity'][t]['C'], mods[b]['fidelity'][t]['C']
                mg = max(xa['noise_bound'], xb['noise_bound'], 0.005)
                adv[b][t] = {'own': xb['E_H'], 'sum': xa['E_H'], 'margin': mg, 'pass': bool(xb['E_H'] <= xa['E_H'] - mg)}
        C5['OWN_CONSISTENT_ADVANTAGE'] = {'cases': adv, 'PASS': bool(all(v['pass'] for x in adv.values() for v in x.values()))}
        lost, worse = [], []
        for a, b in pair:
            ga, gb = mods[a]['gates'], mods[b]['gates']
            for gk in ('F1', 'F2', 'F4'):
                for k, v in ga[gk].items():
                    if v and not gb[gk].get(k, False):
                        lost.append([b, gk, k])
            for t in ('T0', 'T1'):
                for k in ('stationary_all', 'D'):
                    xa, xb = mods[a]['fidelity'][t].get(k), mods[b]['fidelity'][t].get(k)
                    if xa and xb and xb['E_H'] > 1.25 * xa['E_H'] + 0.002:
                        worse.append([b, t, k, xb['E_H'], xa['E_H']])
            for src in ('LMIX', 'LSPOT'):
                xa = mods[a]['sources'].get(src, {}).get('stationary_all'); xb = mods[b]['sources'].get(src, {}).get('stationary_all')
                if xa and xb and xb['E_H'] > 1.25 * xa['E_H'] + 0.002:
                    worse.append([b, src, 'stationary_all', xb['E_H'], xa['E_H']])
        C5['NO_REGRESSION'] = {'lost': lost, 'worse': worse, 'PASS': bool(not lost and not worse)}
        f3l = [b for a, b in pair if mods[a]['gates']['F3'] == 'PASS' and mods[b]['gates']['F3'] == 'FAIL']
        C5['DYNAMIC'] = {'F3_lost': f3l, 'C_edit': {n: (mods[n]['dH'].get('C', {}).get('slope'), mods[n]['dH'].get('C', {}).get('corr'), mods[n]['dH'].get('C', {}).get('F3')) for n in MODELS}, 'PASS': not f3l}
    out['e5']['criteria'] = C5
    # ---------------------------------------------------------------- outcome
    sup = [k for k in ('H_local', 'H_global', 'H_source', 'H_geometry') if H.get(k, {}).get('label') == 'SUPPORTED']
    dom = [k for k in ('H_local', 'H_global', 'H_source') if H.get(k, {}).get('label') == 'SUPPORTED']
    rep_ok = bool(out['repro'] and out['repro']['PASS'])
    if not rep_ok or chosen is None:
        oc = 'D'
    elif H['H_interact']['label'] == 'SUPPORTED' or len(sup) > 1:
        oc = 'C'
    elif len(dom) == 1 and C5.get('OWN_CONSISTENT_ADVANTAGE', {}).get('PASS') and C5.get('NO_REGRESSION', {}).get('PASS'):
        oc = 'A'
    elif len(dom) == 1:
        oc = 'B'
    else:
        oc = 'D'
    out['outcome'] = {'label': oc, 'supported': sup, 'baseline_reproduced': rep_ok, 'e5_admitted': chosen is not None}
    json.dump(out, open(os.path.join(R, 'results', 'final92b.json'), 'w'), indent=1)
    print('OUTCOME', oc, 'supported', sup, 'e5', chosen, flush=True)
    for k, v in H.items():
        print(k, json.dumps(v)[:400], flush=True)
    for tag, o in out['cases'].items():
        for rk in CASES[tag][:2]:
            print(tag, rk, 'floor', round(o['floor'].get(rk, {}).get('E_H', -1), 4),
                  json.dumps({n: {cf: round(o['models'][n]['E_H'][cf][rk]['E_H'], 4) for cf in CFS if cf in o['models'][n]['E_H']} for n in MODELS}), flush=True)
    if C5:
        for k, v in C5.items():
            print('E5', k, json.dumps(v)[:500], flush=True)


if __name__ == '__main__':
    main()
