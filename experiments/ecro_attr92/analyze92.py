"""WL92 analysis by the prospective rules (protocol ecro_attr92: ladder, fits, identifiability, gate, verdict mapping). numpy only.

  python analyze92.py gate  <R> <R86>     stages S0-S5 -> results/core92.json, results/gate.json ; prints GATE ADMITTED | GATE STOP_INTERVENTION_<gate>
  python analyze92.py e4    <R> <R86>     E4 admission (WL86 stage-0 criteria) -> results/e4_admission.json
  python analyze92.py final <R> <R86>     everything (+ intervention evaluation if it ran) -> results/final92.json
"""
import glob, json, math, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'ecro86'))
import analyze86 as A  # noqa: E402
import scene86 as S  # noqa: E402

P92 = json.load(open(os.path.join(HERE, 'protocol', 'ecro_attr92.json')))
TH = P92['thresholds']
HIST = P92['conditions']['historical']
FAIL = [tuple(x) for x in P92['conditions']['failing_cases']]
SEEDS = ['ecro_s8601', 'ecro_s8602']
REG = ['stationary_all', 'A', 'C', 'D', 'obj_panel', 'obj_sphere']
LADDER = ['ORA2', 'L_RS', 'L_EXP', 'L_ENCQ', 'L_ENCB', 'L_R16', 'P_EXP', 'P_ENCB', 'P_R16', 'X_EXP']
NB = S.NB


def jd(o, fn):
    json.dump(o, open(fn, 'w'), indent=1)


def jl(fn):
    return json.load(open(fn))


def mask_of(R, R86, tag):
    p = tag.split('_')
    name = p[0] + ('_' + p[2] if len(p) > 2 else '')
    for root in (R, R86):
        fn = os.path.join(root, 'results', 'masks', name + '.npz')
        if os.path.exists(fn):
            z = np.load(fn)
            return {k: z[k] for k in z.files if z[k].dtype == bool}
    raise FileNotFoundError(name)


def ref_of(R, R86, tag):
    root = R if glob.glob(os.path.join(R, 'results', 'renders', f'REF_ER_{tag}_r*.npz')) else R86
    return A.load(root, f'REF_ER_{tag}')


def hl(R, pat):
    fns = sorted(glob.glob(os.path.join(R, 'results', 'renders', pat + '_r*.npz')))
    return {'hl': np.stack([np.load(f)['hl'].astype(np.float64) for f in fns])} if fns else None


def eh(ref, ec, M, regs=REG):
    h, ha, hb = A.halves(ref['h']); tot = ref['tot'].mean(0)
    x, xa, xb = A.halves(ec['hl'])
    T = {}
    for k in regs:
        if k not in M or M[k].sum() < 30:
            continue
        m = M[k]
        den = max(tot[m].sum(), 1e-30)
        T[k] = {'E_H': float(np.abs(x[m] - h[m]).sum() / den), 'signed': float((x[m] - h[m]).sum() / den), 'H_share': float(h[m].sum() / den),
                'noise_bound': float((0.5 * np.abs(xa[m] - xb[m]).sum() + 0.5 * np.abs(ha[m] - hb[m]).sum()) / den), 'pixels': int(m.sum())}
        T[k]['capture'] = T[k]['E_H'] / max(T[k]['H_share'], 1e-30)
    return T


def dh(ref0, ref1, ec0, ec1, M, regs=('A', 'C', 'D')):
    Y = A.Y
    x, xa, xb = A.halves(Y(ref1['h']) - Y(ref0['h']))
    y, ya, yb = A.halves(Y(ec1['hl']) - Y(ec0['hl']))
    t0 = Y(ref0['tot']).mean(0)
    T = {}
    for k in regs:
        if k not in M or M[k].sum() < 30:
            continue
        m = M[k]
        sl, co = A.slope_corr(x, xa, xb, y, ya, yb, m)
        b = max(float(t0[m].mean()), 1e-30)
        T[k] = {'slope': sl, 'corr': co, 'true_absdH_rel': float(np.abs(x[m]).mean() / b), 'true_absdH_floor_rel': float(0.5 * np.abs(xa[m] - xb[m]).mean() / b)}
        T[k]['resolvable'] = bool(T[k]['true_absdH_rel'] >= TH['snr'] * T[k]['true_absdH_floor_rel'])
        T[k]['F3'] = 'INDETERMINATE' if (not T[k]['resolvable'] or sl is None) else ('PASS' if TH['dH_slope'][0] <= sl <= TH['dH_slope'][1] and co >= TH['dH_corr_min'] else 'FAIL')
    return T


def reads(Wz, X, rk):
    return float((Wz[rk].astype(np.float64).reshape(-1, NB, 3) * X.astype(np.float64)).sum())


# ------------------------------------------------------------------------------------------------ S0-S5
def core(R, R86):
    out = {'repro': {}, 'conditions': {}, 'fits': {}, 'identifiability': {}}
    f86 = jl(os.path.join(R86, 'results', 'final.json'))
    tests = jl(os.path.join(R, 'results', 'tests92.json')) if os.path.exists(os.path.join(R, 'results', 'tests92.json')) else {}
    out['local'] = jl(os.path.join(R, 'results', 'local.json')) if os.path.exists(os.path.join(R, 'results', 'local.json')) else None
    for tag in HIST:
        if not os.path.exists(os.path.join(R, 'results', 'tables', f'T_{tag}.npz')):
            continue
        e, t = tag.split('_')
        M = mask_of(R, R86, tag); ref = ref_of(R, R86, tag)
        T = np.load(os.path.join(R, 'results', 'tables', f'T_{tag}.npz'))
        Wz = np.load(os.path.join(R, 'results', 'fields', f'W_{tag}.npz'))
        meta = jl(os.path.join(R, 'results', 'tables', f'T_{tag}.json'))
        G = S.Geom(S.FX[e][t])
        o = {'pixels': {k: int(M[k].sum()) for k in REG if k in M}, 'rendered': {}, 'frozen': {}, 'meta': {k: meta[k] for k in ('form_rel_l1_ENCB_vs_binq', 'own_sum', 'state_digest')}}
        # reproduction of WL86 from the stored renders and from the re-rendered tables
        for sd in SEEDS:
            st = eh(ref, A.load(R86, f'EC_{sd}_ecro_{tag}'), M)
            w86 = f86['seeds'][sd]['fidelity'][e][t]
            mine = hl(R, f'X92_B_{sd}_{tag}')
            out['repro'][f'{sd}:{tag}'] = {'stored_vs_final_json_max_abs': float(max(abs(st[k]['E_H'] - w86[k]['E_H']) for k in ('stationary_all', 'C', 'D') if k in st and k in w86)),
                                           'WL86_C': w86.get('C', {}).get('E_H'), 'rerendered_C': eh(ref, mine, M)['C']['E_H'] if mine and 'C' in M else None}
        for nm in LADDER + [f'{p}_{sd}' for sd in SEEDS for p in ('B', 'SL', 'LG')]:
            x = hl(R, f'X92_{nm}_{tag}')
            if x is not None:
                o['rendered'][nm] = eh(ref, x, M)
        # reader-weighted family decomposition (fractions of the higher-order reading of the region)
        L2T, L3T, E2P = T['L2T'], T['L3T'], T['E2P']
        for rk in [k for k in ('C', 'obj_panel', 'D') if k in Wz.files]:
            h = reads(Wz, L2T + L3T, rk)
            fam = {'truth_share': {'L2_via_panel': reads(Wz, E2P, rk) / h, 'L2_other': reads(Wz, L2T - E2P, rk) / h, 'L3plus': reads(Wz, L3T, rk) / h},
                   'two_event_variants_signed_over_h': {v: (reads(Wz, T[f'L2_{v}'], rk) - reads(Wz, L2T, rk)) / h for v in ('RS', 'EXP', 'ENCQ', 'ENCB', 'R16')},
                   'pooled_signed_over_h': {v: (reads(Wz, T[f'P2_{v}'], rk) - reads(Wz, L2T, rk)) / h for v in ('EXP', 'ENCB', 'R16')}, 'models': {}}
            for sd in SEEDS:
                fam['models'][sd] = {'single_minus_L2_over_h': (reads(Wz, T[f'S_{sd}'], rk) - reads(Wz, L2T, rk)) / h,
                                     'global_minus_L3_over_h': (reads(Wz, T[f'G_{sd}'], rk) - reads(Wz, L3T, rk)) / h,
                                     'total_over_h': (reads(Wz, T[f'B_{sd}'], rk) - h) / h,
                                     'panel_family_delivered': reads(Wz, T[f'SP_{sd}'], rk) / max(reads(Wz, E2P, rk), 1e-30),
                                     'single_share_of_L2': reads(Wz, T[f'S_{sd}'], rk) / max(reads(Wz, L2T, rk), 1e-30)}
            o['frozen'][rk] = fam
        # once-scattered table of the frozen generators against the exact q table, by material (field level)
        D = np.load(os.path.join(R, 'results', 'fields', f'D_{tag}.npz'))
        q = D['q'].astype(np.float64)
        o['Bq_vs_q'] = {}
        for sd in SEEDS:
            Bq = T[f'Bq_{sd}'].astype(np.float64)
            o['Bq_vs_q'][sd] = {str(m): {'rel_l1': float(np.abs(Bq[G.PMAT == m] - q[G.PMAT == m]).sum() / max(q[G.PMAT == m].sum(), 1e-30)),
                                         'flux_ratio': float(Bq[G.PMAT == m].sum() / max(q[G.PMAT == m].sum(), 1e-30))} for m in range(6) if (G.PMAT == m).any() and q[G.PMAT == m].sum() > 0}
        o['field'] = {'split_sum_flux': float((L2T + L3T).sum()), 'L2_share_of_field': float(L2T.sum() / (L2T + L3T).sum()),
                      'single_vs_L2_rel_l1': {sd: float(np.abs(T[f'S_{sd}'] - L2T).sum() / L2T.sum()) for sd in SEEDS},
                      'global_vs_L3_rel_l1': {sd: float(np.abs(T[f'G_{sd}'] - L3T).sum() / L3T.sum()) for sd in SEEDS},
                      'variants_vs_L2_rel_l1': {v: float(np.abs(T[f'L2_{v}'] - L2T).sum() / L2T.sum()) for v in ('RS', 'EXP', 'ENCQ', 'ENCB', 'R16')}}
        out['conditions'][tag] = o
        # fits
        fo = {}
        for fn in sorted(glob.glob(os.path.join(R, 'results', 'fits', f'fit_{tag}_*.json'))):
            r = jl(fn)
            v = r['variant']
            row = {'objective': r['objective'], 'init': r['init'], 'converged': r['convergence']['converged'], 'relative_change': r['convergence']['relative_change'], 'seconds': r['seconds'], 'contracts': r['contracts']}
            for ph in ('init_eval', 'final_eval'):
                ev = r[ph]
                rd = ev['workload']['read'].get('C')
                row[ph] = {'val': ev['val'], 'fit32': ev['fit32'], 'workload_field': ev['workload']['field'],
                           'C_read': None if rd is None else {'panel_family_delivered': rd['SP'] / max(rd['L2_panel'], 1e-30), 'single_share_of_L2': rd['S'] / max(rd['L2'], 1e-30),
                                                              'single_minus_L2_over_h': (rd['S'] - rd['L2']) / rd['h'], 'global_minus_L3_over_h': (rd['G'] - rd['L3']) / rd['h'], 'total_over_h': (rd['B'] - rd['h']) / rd['h']}}
            row['rendered'] = {}
            for key in ('B', 'SL', 'SL0', 'B0'):
                x = hl(R, f'X92F_{v}_{key}_{tag}')
                if x is not None:
                    row['rendered'][key] = eh(ref, x, M, ('stationary_all', 'C', 'D', 'obj_panel'))
            fo[v] = row
        out['fits'][tag] = fo
        # identifiability of the SUM contract at this geometry
        sols = {}
        for v, row in fo.items():
            if row['objective'] == 'SUM':
                sols[v] = row['final_eval']
                if row['init'].startswith('f'):
                    sols['frozen_' + row['init'][1:]] = row['init_eval']
        if sols:
            ls = {k: e_['val']['L_sum'] for k, e_ in sols.items()}
            dl = {k: e_['C_read']['panel_family_delivered'] for k, e_ in sols.items() if e_['C_read']}
            fitted = [k for k in sols if not k.startswith('frozen')]
            lsf = [ls[k] for k in fitted]; dlf = [max(dl[k], 1e-6) for k in fitted if k in dl]
            own = {v: {'L_sum_val': row['final_eval']['val']['L_sum'], 'L_comp_val': row['final_eval']['val']['L_comp'],
                       'panel_family_delivered': row['final_eval']['C_read']['panel_family_delivered'] if row['final_eval']['C_read'] else None} for v, row in fo.items() if row['objective'] == 'OWN'}
            k16 = fo.get('OWN-k16', {}).get('init_eval')
            out['identifiability'][tag] = {'L_sum_val': ls, 'panel_family_delivered': dl,
                                           'sum_objective_spread_fitted': (max(lsf) / max(min(lsf), 1e-30)) if lsf else None,
                                           'delivered_ratio_fitted': (max(dlf) / min(dlf)) if dlf else None,
                                           'NON_IDENTIFIABLE': bool(lsf and dlf and max(lsf) / max(min(lsf), 1e-30) <= 1.10 and max(dlf) / min(dlf) >= 2.0),
                                           'OWN_points': own, 'R16_construct_with_frozen_global': None if k16 is None else {'L_sum_val': k16['val']['L_sum'], 'L_comp_val': k16['val']['L_comp']}}
    # ---- gate
    lvl = TH['EH_region_max']
    g = {'level': lvl, 'cases': {}}
    info_ok = rep_ok = True
    for tag, rk in FAIL:
        c = out['conditions'].get(tag)
        if c is None:
            info_ok = rep_ok = False
            continue
        r = c['rendered']
        encb = r.get('L_ENCB', {}).get(rk, {}).get('E_H'); r16 = r.get('L_R16', {}).get(rk, {}).get('E_H')
        fk = out['fits'].get(tag, {}).get('OWN-k16')
        fk_e = fk['rendered'].get('SL', {}).get(rk, {}).get('E_H') if fk else None
        fk_c = bool(fk and fk['converged'])
        best = min([v for v in (r16, fk_e if fk_c else None) if v is not None], default=None)
        g['cases'][f'{tag}:{rk}'] = {'ENCB': encb, 'R16': r16, 'OWN_k16_fit': fk_e, 'OWN_k16_converged': fk_c, 'best_in_family_owned': best,
                                     'frozen_B': {sd: r.get(f'B_{sd}', {}).get(rk, {}).get('E_H') for sd in SEEDS}, 'oracle': r.get('ORA2', {}).get(rk, {}).get('E_H'),
                                     'EXP': r.get('L_EXP', {}).get(rk, {}).get('E_H'), 'RS': r.get('L_RS', {}).get(rk, {}).get('E_H')}
        info_ok &= encb is not None and encb <= lvl
        rep_ok &= best is not None and best <= lvl
    gid = all(tests.get(k, {}).get('PASS', False) for k in ('T_SPLIT_SUM', 'T_OWNERSHIP_CODE', 'T_FORM'))
    g.update({'G_id': bool(gid), 'G_info': bool(info_ok), 'G_rep': bool(rep_ok), 'ADMITTED': bool(gid and info_ok and rep_ok)})
    g['STOP'] = None if g['ADMITTED'] else ('STOP_INTERVENTION_' + ('G_id' if not gid else 'G_info' if not info_ok else 'G_rep'))
    out['gate'] = g
    return out


def gate(R, R86):
    out = core(R, R86)
    jd(out, os.path.join(R, 'results', 'core92.json'))
    jd(out['gate'], os.path.join(R, 'results', 'gate.json'))
    for k, v in out['gate']['cases'].items():
        print(k, json.dumps({a: (round(b, 4) if isinstance(b, float) else b) for a, b in v.items() if not isinstance(b, dict)}), flush=True)
    print('GATE', 'ADMITTED' if out['gate']['ADMITTED'] else out['gate']['STOP'], flush=True)


# ------------------------------------------------------------------------------------------------ E4 admission (WL86 stage-0 criteria)
def e4(R, R86):
    G0 = A.G0
    e = 'E4'
    M = mask_of(R, R86, 'E4_T1')
    p0, p1 = A.load(R, f'REF_P_{e}_T0'), A.load(R, f'REF_P_{e}_T1')
    r1 = A.load(R, f'REF_ER_{e}_T1'); r0 = A.load(R, f'REF_ER_{e}_T0')
    Y = A.Y
    d, da, db = A.halves(Y(p1['tot']) - Y(p0['tot']))
    dh_, dha, dhb = A.halves(Y(p1['h']) - Y(p0['h']))
    t0 = Y(p0['tot']).mean(0); t1, t1a, t1b = A.halves(p1['tot']); e1r, e1a, e1b = A.halves(r1['tot'])
    Rg = {}
    for k in A.REG:
        if k not in M:
            continue
        m = M[k]
        if m.sum() < 30:
            Rg[k] = {'pixels': int(m.sum())}
            continue
        b = max(float(t0[m].mean()), 1e-30)
        Rg[k] = {'pixels': int(m.sum()), 'abs_change_rel': float(np.abs(d[m]).mean() / b), 'abs_change_floor_rel': float(0.5 * np.abs(da[m] - db[m]).mean() / b),
                 'absH_change_rel': float(np.abs(dh_[m]).mean() / b), 'absH_change_floor_rel': float(0.5 * np.abs(dha[m] - dhb[m]).mean() / b),
                 'H_share_T1': float(p1['h'].mean(0)[m].sum() / t1[m].sum()), 'uncertainty_P_T1': 0.5 * A.rl(t1a, t1b, m),
                 'uncertainty_ER_T1': 0.5 * A.rl(e1a, e1b, m), 'g_ER_vs_P_T1': A.rl(e1r, t1, m)}
    mC = M['C']; sm_recv = M['obj_panel'] | M['obj_sphere']
    dc = np.abs((r1['c2'] - r0['c2']).mean(0)) + np.abs((r1['ch'] - r0['ch']).mean(0))
    allc = dc.sum(-1)
    sm = float((allc[mC & sm_recv].sum() + dc[..., :2].sum(-1)[mC & ~sm_recv].sum()) / max(allc[mC].sum(), 1e-30)) if mC.sum() else 0.0
    g = lambda k, f: Rg.get(k, {}).get(f, 0.0)
    sig = lambda k, f='abs_change_rel', fl_='abs_change_floor_rel': g(k, f) >= G0['snr'] * g(k, fl_)
    q = {}
    q['Q1_A_direct_change'] = {'value': g('A', 'abs_change_rel'), 'PASS': bool(g('A', 'abs_change_rel') >= G0['A_min'] and sig('A'))}
    q['Q2_B_indirect_change'] = {'value': g('Bset', 'abs_change_rel'), 'PASS': bool(g('Bset', 'abs_change_rel') >= G0['B_min'] and sig('Bset'))}
    q['Q3_C_sharp_moderate'] = {'value': g('C', 'abs_change_rel'), 'sm_share': sm, 'pixels': Rg['C']['pixels'],
                                'PASS': bool(Rg['C']['pixels'] >= G0['min_pixels'] and g('C', 'abs_change_rel') >= G0['C_min'] and sig('C') and sm >= G0['C_sm_share_min'])}
    ratio = g('D', 'absH_change_rel') / max(g('D', 'abs_change_rel'), 1e-30)
    q['Q4_D_broad_multibounce'] = {'value': g('D', 'abs_change_rel'), 'absH_over_abs': ratio,
                                   'PASS': bool(g('D', 'abs_change_rel') >= G0['D_min'] and sig('D') and ratio >= G0['D_H_ratio_min'] and sig('D', 'absH_change_rel', 'absH_change_floor_rel'))}
    q['Q6_H_share'] = {'value': g('stationary_all', 'H_share_T1'), 'PASS': bool(g('stationary_all', 'H_share_T1') >= G0['H_share_min'])}
    un = max(g('stationary_all', 'uncertainty_P_T1'), g('stationary_all', 'uncertainty_ER_T1'))
    q['Q7_uncertainty'] = {'value': un, 'PASS': bool(un <= G0['uncertainty_max'])}
    adm = bool(all(v['PASS'] for v in q.values()))
    jd({'regions': Rg, 'qualification': q, 'ADMITTED': adm, 'g_stationary': g('stationary_all', 'g_ER_vs_P_T1')}, os.path.join(R, 'results', 'e4_admission.json'))
    print('E4', 'ADMITTED' if adm else 'NOT_ADMITTED', json.dumps({k: v['PASS'] for k, v in q.items()}), flush=True)


# ------------------------------------------------------------------------------------------------ intervention evaluation
def intervention(R, R86):
    mdir = os.path.join(R, 'results', 'models')
    names = SEEDS + sorted(os.path.basename(f)[:-3] for f in glob.glob(os.path.join(mdir, '*.pt')))
    if len(names) == len(SEEDS):
        return None
    adm = jl(os.path.join(R, 'results', 'e4_admission.json')) if os.path.exists(os.path.join(R, 'results', 'e4_admission.json')) else {'ADMITTED': False}
    o = {'models': {}, 'E4_admitted': adm['ADMITTED'], 'training': {n: {k: v for k, v in jl(os.path.join(mdir, n + '.json')).items() if k != 'log'} for n in names if os.path.exists(os.path.join(mdir, n + '.json'))}}
    edits = ['E1', 'E2'] + (['E4'] if glob.glob(os.path.join(R, 'results', 'renders', 'REF_ER_E4_T1_r*.npz')) else [])
    srcs = [('E1_T1_LMIX', 'E1'), ('E1_T1_LSPOT', 'E1')] + ([('E4_T1_LMIX4', 'E4'), ('E4_T1_LSPOT4', 'E4')] if 'E4' in edits else [])
    for n in names:
        m = {'fidelity': {}, 'dH': {}, 'sources': {}, 'attribution': {}}
        f1 = f2 = f4 = True; f3 = {}
        for e in edits:
            M = mask_of(R, R86, f'{e}_T1')
            ec = {t: hl(R, f'I92_{n}_{e}_{t}') for t in ('T0', 'T1')}
            if ec['T0'] is None or ec['T1'] is None:
                continue
            rf = {t: ref_of(R, R86, f'{e}_{t}') for t in ('T0', 'T1')}
            m['fidelity'][e] = {t: eh(rf[t], ec[t], M) for t in ('T0', 'T1')}
            for t in ('T0', 'T1'):
                T = m['fidelity'][e][t]['stationary_all']
                f1 &= T['E_H'] <= TH['EH_stationary_max'] and T['capture'] <= TH['capture_max']
            for k in ('A', 'C', 'D'):
                if k in m['fidelity'][e]['T1'] and m['fidelity'][e]['T1'][k]['pixels'] >= TH['min_pixels']:
                    f2 &= m['fidelity'][e]['T1'][k]['E_H'] <= TH['EH_region_max']
            m['dH'][e] = dh(rf['T0'], rf['T1'], ec['T0'], ec['T1'], M)
            f3[e] = m['dH'][e].get('D', {}).get('F3', 'NA')
        for tag, e in srcs:
            x = hl(R, f'I92_{n}_{tag}')
            if x is None:
                continue
            T = eh(ref_of(R, R86, tag), x, mask_of(R, R86, tag))
            m['sources'][tag] = T
            f4 &= T['stationary_all']['E_H'] <= TH['EH_unseen_source_max']
        for tag in ('E2_T1', 'E2_T0', 'E4_T1', 'E4_T0', 'E1_T1'):
            fnI, fnD, fnW = (os.path.join(R, 'results', d, f'{p}_{tag}.npz') for d, p in (('tables', 'I'), ('fields', 'D'), ('fields', 'W')))
            if not all(os.path.exists(f) for f in (fnI, fnD, fnW)):
                continue
            I, D, Wz = np.load(fnI), np.load(fnD), np.load(fnW)
            hm = lambda ks: sum(0.5 * (D[f'{k}_a'].astype(np.float64) + D[f'{k}_b']) for k in ks)
            L2T, L3T, E2P = hm(('e2_p', 'e2_o')), hm(('e3_p', 'e3_o')), hm(('e2_p',))
            if 'C' not in Wz.files:
                continue
            h = reads(Wz, L2T + L3T, 'C')
            m['attribution'][tag] = {'truth_panel_family_share': reads(Wz, E2P, 'C') / h, 'panel_family_delivered': reads(Wz, I[f'SP_{n}'], 'C') / max(reads(Wz, E2P, 'C'), 1e-30),
                                     'single_minus_L2_over_h': (reads(Wz, I[f'S_{n}'], 'C') - reads(Wz, L2T, 'C')) / h, 'global_minus_L3_over_h': (reads(Wz, I[f'G_{n}'], 'C') - reads(Wz, L3T, 'C')) / h,
                                     'field_single_vs_L2_rel_l1': float(np.abs(I[f'S_{n}'] - L2T).sum() / L2T.sum()), 'field_global_vs_L3_rel_l1': float(np.abs(I[f'G_{n}'] - L3T).sum() / L3T.sum())}
        m['gates'] = {'F1': bool(f1), 'F2': bool(f2), 'F3_D': f3, 'F4': bool(f4)}
        o['models'][n] = m
    own = [n for n in names if n.startswith('own_')]; ctl = [n for n in names if n.startswith('ctl_')] or SEEDS
    cases = [('E2', 'T1'), ('E2', 'T0')] + ([('E4', 'T1')] if 'E4' in edits and adm['ADMITTED'] else [])
    cval = lambda n, e, t: o['models'][n]['fidelity'].get(e, {}).get(t, {}).get('C', {}).get('E_H')
    f2c = all(cval(n, e, t) is not None and cval(n, e, t) <= TH['EH_region_max'] for n in own for e, t in cases) if own else False
    lost = [(n, k) for n, c in zip(own, ctl) for k in ('F1', 'F4') if o['models'][c]['gates'][k] and not o['models'][n]['gates'][k]]
    dl = lambda n, tag: o['models'][n]['attribution'].get(tag, {}).get('panel_family_delivered')
    ratio = [dl(n, tag) / max(dl(c, tag), 1e-6) for n, c in zip(own, ctl) for tag in ('E2_T1', 'E2_T0') if dl(n, tag) is not None and dl(c, tag) is not None]
    o['effect'] = {'own': own, 'control': ctl, 'F2_C_all_cases': bool(f2c), 'lost_gates': lost, 'delivered_ratio_own_over_control': ratio,
                   'EFFECTIVE': bool(own and f2c and not lost and ratio and min(ratio) >= 2.0),
                   'C_E_H': {n: {f'{e}_{t}': cval(n, e, t) for e in edits for t in ('T0', 'T1')} for n in names}}
    return o


def final(R, R86):
    out = core(R, R86)
    iv = intervention(R, R86)
    out['intervention'] = iv
    g = out['gate']
    C = out['conditions']
    lvl = TH['EH_region_max']
    V = {}
    fits_conv = all(out['fits'].get(tag, {}).get('OWN-k16', {}).get('converged', False) for tag, _ in FAIL)
    if g['G_rep']:
        V['A_representation_capacity'] = 'SUPPORTED'
    elif g['G_info'] and fits_conv:
        V['A_representation_capacity'] = 'NOT SUPPORTED'
    else:
        V['A_representation_capacity'] = 'INCONCLUSIVE'
    V['B_input_information'] = 'SUPPORTED' if g['G_info'] else 'NOT SUPPORTED'
    # C: a constructed / OWN-fitted in-family single-flight term against the frozen generators' (rendered, + path-traced L3+)
    better = []
    for tag, rk in FAIL:
        r = C.get(tag, {}).get('rendered', {})
        best = g['cases'].get(f'{tag}:{rk}', {}).get('best_in_family_owned')
        for sd in SEEDS:
            fz = r.get(f'SL_{sd}', {}).get(rk, {}).get('E_H')
            if best is not None and fz is not None:
                better.append(fz / max(best, 1e-30))
    V['C_learning_optimisation'] = ('INCONCLUSIVE' if not better else 'NOT SUPPORTED' if min(better) >= 1.25 else 'SUPPORTED' if max(better) < 1.25 else 'INCONCLUSIVE')
    idn = [v['NON_IDENTIFIABLE'] for v in out['identifiability'].values()]
    V['D_component_identifiability'] = {'sum_contract': ('NON-IDENTIFIABLE' if idn and all(idn) else 'IDENTIFIABLE BY THE RULE' if idn and not any(idn) else 'MIXED' if idn else 'NOT TESTED'),
                                        'component_targets_valid': bool(g['G_id'])}
    if iv is None:
        V['E_heldout_sharp_fidelity'] = 'NOT TESTED'; V['F_economics'] = 'NOT TESTED'
        V['completion'] = 1 if not (g['G_info'] and g['G_rep']) else 5
    else:
        V['E_heldout_sharp_fidelity'] = ('SUPPORTED' if iv['effect']['EFFECTIVE'] and iv['E4_admitted'] else 'NOT SUPPORTED' if not iv['effect']['F2_C_all_cases'] else 'INCONCLUSIVE')
        V['F_economics'] = 'runtime path unchanged (WL86 economics apply); offline costs in intervention.training'
        V['completion'] = 3 if iv['effect']['EFFECTIVE'] else 2
    material = sum([V['A_representation_capacity'] == 'NOT SUPPORTED', V['B_input_information'] == 'NOT SUPPORTED', V['C_learning_optimisation'] == 'NOT SUPPORTED',
                    V['D_component_identifiability']['sum_contract'] == 'NON-IDENTIFIABLE'])
    V['material_causes'] = int(material)
    if material >= 2:
        V['completion_note'] = 'more than one of A-D carries a material share (completion 4 applies alongside)'
    out['verdicts'] = V
    jd(out, os.path.join(R, 'results', 'final92.json'))
    print('VERDICT', json.dumps(V), flush=True)
    print('GATE', json.dumps({k: v for k, v in g.items() if k != 'cases'}), flush=True)
    for tag in C:
        r = C[tag]['rendered']
        print(tag, 'C E_H', json.dumps({k: round(v['C']['E_H'], 4) for k, v in r.items() if 'C' in v}), flush=True)


if __name__ == '__main__':
    {'gate': gate, 'e4': e4, 'final': final}[sys.argv[1]](sys.argv[2], sys.argv[3])
