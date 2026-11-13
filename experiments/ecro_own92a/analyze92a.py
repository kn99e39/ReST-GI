"""WL92A analysis by the prospective rules (protocol ecro_own92a: criteria, outcome_mapping). numpy only.

  python analyze92a.py <R> <R86>   -> results/final92a.json
"""
import glob, json, math, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'ecro_attr92'))
import analyze92 as AN  # noqa: E402

P = json.load(open(os.path.join(HERE, 'protocol', 'ecro_own92a.json')))
TH = P['thresholds']
SEEDS = P['training']['seeds']
NB = AN.NB
EDITS = ('E1', 'E2', 'E4')
SRC = {'E1': ['E1_T1_LMIX', 'E1_T1_LSPOT'], 'E4': ['E4_T1_LMIX4', 'E4_T1_LSPOT4']}


def jl(fn):
    return json.load(open(fn)) if os.path.exists(fn) else None


def names(R):
    return ['ecro_s8601', 'ecro_s8602'] + [f'{a}_s{s}' for a in ('sum', 'own') for s in SEEDS if os.path.exists(os.path.join(R, 'results', 'models', f'{a}_s{s}.pt'))]


def main():
    R, R86 = sys.argv[1], sys.argv[2]
    out = {'tests': {'pre': jl(os.path.join(R, 'results', 'tests_pre.json')), 'f2': jl(os.path.join(R, 'results', 'tests_f2.json'))}}
    fe = jl(os.path.join(R, 'results', 'fieldeval.json'))
    out['fieldeval'] = fe
    adm = jl(os.path.join(R, 'results', 'e4_admission.json')) or {'ADMITTED': False}
    out['E4_admitted'] = bool(adm.get('ADMITTED'))
    out['training'] = {n: {k: v for k, v in (jl(os.path.join(R, 'results', 'models', f'{n}.json')) or {}).items() if k != 'log'} for n in names(R) if not n.startswith('ecro')}
    out['training_curves'] = {n: (jl(os.path.join(R, 'results', 'models', f'{n}.json')) or {}).get('log') for n in names(R) if not n.startswith('ecro')}
    NM = names(R)
    M = {}
    out['models'] = {}
    for n in NM:
        m = {'fidelity': {}, 'dH': {}, 'sources': {}, 'stale': {}, 'attribution': {}}
        gates = {'F1': {}, 'F2': {}, 'F4': {}, 'F3': {}}
        for e in EDITS:
            ec = {t: AN.hl(R, f'I92A_{n}_{e}_{t}') for t in ('T0', 'T1')}
            if ec['T0'] is None or ec['T1'] is None:
                continue
            Mk = AN.mask_of(R, R86, f'{e}_T1')
            rf = {t: AN.ref_of(R, R86, f'{e}_{t}') for t in ('T0', 'T1')}
            m['fidelity'][e] = {t: AN.eh(rf[t], ec[t], Mk) for t in ('T0', 'T1')}
            for t in ('T0', 'T1'):
                T = m['fidelity'][e][t]['stationary_all']
                gates['F1'][f'{e}_{t}'] = bool(T['E_H'] <= TH['EH_stationary_max'] and T['capture'] <= TH['capture_max'])
            for k in ('A', 'C', 'D'):
                T = m['fidelity'][e]['T1'].get(k)
                if T and T['pixels'] >= TH['min_pixels']:
                    gates['F2'][f'{e}:{k}'] = bool(T['E_H'] <= TH['EH_region_max'])
            m['dH'][e] = AN.dh(rf['T0'], rf['T1'], ec['T0'], ec['T1'], Mk)
            gates['F3'][e] = m['dH'][e].get('D', {}).get('F3', 'NA')
            st = AN.hl(R, f'I92A_STALE_{n}_{e}_T1')
            if st is not None:
                m['stale'][e] = {k: v['E_H'] for k, v in AN.eh(rf['T1'], st, Mk).items()}
        for e, tags in SRC.items():
            for tag in tags:
                x = AN.hl(R, f'I92A_{n}_{tag}')
                if x is None:
                    continue
                T = AN.eh(AN.ref_of(R, R86, tag), x, AN.mask_of(R, R86, tag))
                m['sources'][tag] = T
                gates['F4'][tag] = bool(T['stationary_all']['E_H'] <= TH['EH_unseen_source_max'])
        m['gates'] = gates
        # reader-weighted attribution (region C) against the path-traced family split
        for tag in ('E2_T1', 'E2_T0', 'E1_T1', 'E4_T1', 'E4_T0'):
            fI, fD, fW = (os.path.join(R, 'results', d, f'{p}_{tag}.npz') for d, p in (('tables', 'I'), ('fields', 'D'), ('fields', 'W')))
            if not all(os.path.exists(f) for f in (fI, fD, fW)):
                continue
            I, D, Wz = np.load(fI), np.load(fD), np.load(fW)
            hm = lambda ks: sum(0.5 * (D[f'{k}_a'].astype(np.float64) + D[f'{k}_b']) for k in ks)
            L2T, L3T, E2P = hm(('e2_p', 'e2_o')), hm(('e3_p', 'e3_o')), hm(('e2_p',))
            row = {'field_single_vs_L2_rel_l1': float(np.abs(I[f'S_{n}'] - L2T).sum() / L2T.sum()), 'field_global_vs_L3_rel_l1': float(np.abs(I[f'G_{n}'] - L3T).sum() / L3T.sum()),
                   'field_whole_rel_l1': float(np.abs(I[f'B_{n}'] - L2T - L3T).sum() / (L2T + L3T).sum())}
            for rk in [k for k in ('C', 'obj_panel', 'D') if k in Wz.files]:
                h = AN.reads(Wz, L2T + L3T, rk)
                row[rk] = {'truth_panel_family_share': AN.reads(Wz, E2P, rk) / h, 'panel_family_delivered': AN.reads(Wz, I[f'SP_{n}'], rk) / max(AN.reads(Wz, E2P, rk), 1e-30),
                           'single_minus_L2_over_h': (AN.reads(Wz, I[f'S_{n}'], rk) - AN.reads(Wz, L2T, rk)) / h,
                           'global_minus_L3_over_h': (AN.reads(Wz, I[f'G_{n}'], rk) - AN.reads(Wz, L3T, rk)) / h,
                           'total_over_h': (AN.reads(Wz, I[f'B_{n}'], rk) - h) / h}
            m['attribution'][tag] = row
        out['models'][n] = m
    # controls on E4
    out['controls'] = {}
    for key in ('ORACLE', 'X_EXP'):
        for t in ('T0', 'T1'):
            x = AN.hl(R, f'I92A_{key}_E4_{t}')
            if x is not None:
                out['controls'][f'{key}_E4_{t}'] = AN.eh(AN.ref_of(R, R86, f'E4_{t}'), x, AN.mask_of(R, R86, 'E4_T1'))
    out['local'] = jl(os.path.join(R, 'results', 'local92a.json'))
    out['timing'] = jl(os.path.join(R, 'results', 'timing92a.json'))
    # ---------------------------------------------------------------- criteria
    V = {}
    tp, f2t = out['tests']['pre'] or {}, out['tests']['f2'] or {}
    stop = []
    if not tp.get('ALL_PASS'):
        stop.append('tests_pre')
    if not f2t.get('PASS'):
        stop.append('T_F2_SUM')
    if fe:
        for k in ('T_INIT', 'T_CTL', 'T_LINEAR', 'T_OWNERSHIP_CODE', 'T_REPLAY'):
            if not fe[k]['PASS']:
                stop.append(k)
    else:
        stop.append('fieldeval_missing')
    if any((out['training'].get(n) or {}).get('nonfinite_steps', 1) > 0 for n in NM if not n.startswith('ecro')):
        stop.append('nonfinite')
    pair = [(f'sum_s{s}', f'own_s{s}') for s in SEEDS]
    have = all(a in out['models'] and b in out['models'] for a, b in pair)
    C = {}
    if fe and have:
        fv = lambda n, k: fe['models'][n]['val'][k]
        C['OWNERSHIP_IMPROVED'] = {'ratios': {b: {'single': fv(b, 'single_vs_L2_rel_l1') / fv(a, 'single_vs_L2_rel_l1'), 'global': fv(b, 'global_vs_L3_rel_l1') / fv(a, 'global_vs_L3_rel_l1')} for a, b in pair}}
        C['OWNERSHIP_IMPROVED']['PASS'] = bool(all(r['single'] <= 0.75 and r['global'] <= 0.75 for r in C['OWNERSHIP_IMPROVED']['ratios'].values()))
        if out['local'] and 'E4_T1' in out['local']:
            pk = lambda n: out['local']['E4_T1'][n]['1']['peak']
            C['PANEL_LOCAL_IMPROVED'] = {'ratios': {b: pk(b) / max(pk(a), 1e-30) for a, b in pair}, 'retention': {n: pk(n) for n in NM}}
            C['PANEL_LOCAL_IMPROVED']['PASS'] = bool(all(v >= 1.5 for v in C['PANEL_LOCAL_IMPROVED']['ratios'].values()))
        ce = lambda n, e, t, k='C': out['models'][n]['fidelity'].get(e, {}).get(t, {}).get(k)
        hs = {}
        for a, b in pair:
            xa, xb = ce(a, 'E4', 'T1'), ce(b, 'E4', 'T1')
            if xa and xb:
                nbd = max(xb['noise_bound'], xa['noise_bound'], 0.005)
                hs[b] = {'own': xb['E_H'], 'sum': xa['E_H'], 'margin': nbd, 'pass': bool(xb['E_H'] <= TH['EH_region_max'] and xb['E_H'] <= xa['E_H'] - nbd)}
        C['HELDOUT_SHARP'] = {'cases': hs, 'E4_admitted': out['E4_admitted'], 'PASS': bool(out['E4_admitted'] and hs and all(v['pass'] for v in hs.values()))}
        lost = []; worse = []
        for a, b in pair:
            ga, gb = out['models'][a]['gates'], out['models'][b]['gates']
            for g in ('F1', 'F2', 'F4'):
                for k, v in ga[g].items():
                    if v and not gb[g].get(k, False):
                        lost.append([b, g, k])
            for e in EDITS:
                for k in ('stationary_all', 'D'):
                    xa, xb = ce(a, e, 'T1', k), ce(b, e, 'T1', k)
                    if xa and xb and xb['E_H'] > 1.25 * xa['E_H'] + 0.002:
                        worse.append([b, f'{e}_T1', k, xb['E_H'], xa['E_H']])
            for tags in SRC.values():
                for tag in tags:
                    xa = out['models'][a]['sources'].get(tag, {}).get('stationary_all'); xb = out['models'][b]['sources'].get(tag, {}).get('stationary_all')
                    if xa and xb and xb['E_H'] > 1.25 * xa['E_H'] + 0.002:
                        worse.append([b, tag, 'stationary_all', xb['E_H'], xa['E_H']])
        C['NO_REGRESSION'] = {'lost_gates': lost, 'worse_broad': worse, 'PASS': bool(not lost and not worse)}
        f3l = [[b, e] for a, b in pair for e in EDITS if out['models'][a]['gates']['F3'].get(e) == 'PASS' and out['models'][b]['gates']['F3'].get(e) == 'FAIL']
        e4c = {}
        for a, b in pair:
            da, db = out['models'][a]['dH'].get('E4', {}).get('C'), out['models'][b]['dH'].get('E4', {}).get('C')
            if da and db:
                imp = bool(da['resolvable'] and da['slope'] is not None and db['slope'] is not None and abs(db['slope'] - 1) < abs(da['slope'] - 1) and (1 - db['corr']) < (1 - da['corr']))
                e4c[b] = {'own': [db['slope'], db['corr']], 'sum': [da['slope'], da['corr']], 'resolvable': da['resolvable'], 'improved': imp}
        C['DYNAMIC'] = {'F3_lost': f3l, 'E4_C_edit_response': e4c, 'PASS': bool(not f3l)}
        C['SOURCE'] = {b: {tag: out['models'][b]['gates']['F4'].get(tag) for tag in SRC['E4']} for _, b in pair}
        C['SOURCE_sum'] = {a: {tag: out['models'][a]['gates']['F4'].get(tag) for tag in SRC['E4']} for a, _ in pair}
    V['criteria'] = C
    V['stop'] = stop
    if stop or not have:
        V['outcome'] = 'D_STOPPED'
    elif not C['OWNERSHIP_IMPROVED']['PASS']:
        V['outcome'] = 'C_NOT_SUPPORTED'
    elif not out['E4_admitted']:
        V['outcome'] = 'D_STOPPED'; V['note'] = 'E4 not admitted: no held-out verdict; ownership improved'
    elif C['HELDOUT_SHARP']['PASS'] and C['NO_REGRESSION']['PASS'] and C['DYNAMIC']['PASS']:
        V['outcome'] = 'A_SUPPORTED'
    else:
        V['outcome'] = 'B_PARTIALLY_SUPPORTED'
    hyp = []
    if C:
        own_imp = C['OWNERSHIP_IMPROVED']['PASS']
        e2 = [out['models'][b]['fidelity'].get('E2', {}).get('T1', {}).get('C', {}).get('E_H', 9) < out['models'][a]['fidelity'].get('E2', {}).get('T1', {}).get('C', {}).get('E_H', 9) for a, b in pair]
        if V['outcome'] == 'A_SUPPORTED':
            hyp.append('H1')
        if own_imp and not C['HELDOUT_SHARP']['PASS']:
            hyp.append('H2')
            if all(e2):
                hyp.append('H3')
        if not own_imp:
            hyp.append('H4')
            if 'PANEL_LOCAL_IMPROVED' in C and not C['PANEL_LOCAL_IMPROVED']['PASS']:
                hyp.append('H5 (consistent with)')
    V['hypotheses'] = hyp
    out['verdict'] = V
    json.dump(out, open(os.path.join(R, 'results', 'final92a.json'), 'w'), indent=1)
    print('VERDICT', V['outcome'], 'stop', stop, 'hyp', hyp, flush=True)
    for k, v in C.items():
        print(k, json.dumps(v)[:600], flush=True)
    for n in NM:
        f = out['models'][n]['fidelity']
        print(n, json.dumps({f'{e}_{t}': {k: round(f[e][t][k]['E_H'], 4) for k in ('stationary_all', 'C', 'D') if k in f[e][t]} for e in f for t in f[e]}), flush=True)


if __name__ == '__main__':
    main()
