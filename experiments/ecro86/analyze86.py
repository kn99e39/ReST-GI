"""WL86 analysis and prospective gates (protocol sections stage0 / stage1 / stage3).

  python analyze86.py qualify <root>     Stage 0: workload admission on REF-P (class shares on REF-ER), E1 and E2
  python analyze86.py stage1 <root>      Stage 1: higher-order identifiability and oracle-projection ceiling
  python analyze86.py final <root>       Stage 3: learned operator fidelity, generalisation, edit response, controls
Error measures (region r, channel-wise sums):
  rel_l1(X, R)  = sum |X - R| / sum R
  E_H(X)        = sum |hl_X - h_ref| / sum tot_ref     additional higher-order error relative to total radiance (LOW estimator shared)
  H share       = sum h_ref / sum tot_ref               (= the error of LOW)
"""
import glob, json, math, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
P = json.load(open(os.path.join(HERE, 'protocol', 'ecro86.json')))
G0, G1, G3 = P['stage0'], P['stage1'], P['stage3']
CFGS = ['E1_T0', 'E1_T1', 'E2_T0', 'E2_T1']
REG = ['stationary_all', 'A', 'Bset', 'C', 'D', 'E', 'B_other', 'obj_floor', 'obj_ceiling', 'obj_walls', 'obj_panel', 'obj_sphere', 'obj_boxes_static']
Y = lambda x: x.mean(-1)


def load(root, pat):
    fns = sorted(glob.glob(os.path.join(root, 'results', 'renders', pat + '_r*.npz')))
    assert fns, pat
    arrs = {}
    for fn in fns:
        z = np.load(fn)
        for k in z.files:
            if k != 'B':
                arrs.setdefault(k, []).append(z[k].astype(np.float64))
    d = {k: np.stack(v) for k, v in arrs.items()}
    if 'tot' not in d:
        d['tot'] = d['e1'] + d['e2'] + (d['h'] if 'h' in d else d['hl'])
    return d


def halves(x):
    n = x.shape[0]
    return x.mean(0), x[:n // 2].mean(0), x[n // 2:].mean(0)


def masks(root, e):
    z = np.load(os.path.join(root, 'results', 'masks', f'{e}.npz'))
    return {k: z[k] for k in z.files if z[k].dtype == bool}, z


def rl(a, b, m):
    return float(np.abs(a[m] - b[m]).sum() / max(b[m].sum(), 1e-30))


def slope_corr(x, xa, xb, y, ya, yb, m):
    sxx = float((xa[m] * xb[m]).sum()); syy = float((ya[m] * yb[m]).sum()); sxy = float((x[m] * y[m]).sum())
    if sxx <= 0 or syy <= 0:
        return None, None
    return sxy / sxx, sxy / math.sqrt(sxx * syy)


def qualify(root):
    out = {}
    ok_all = True
    for e in ('E1', 'E2'):
        M, _ = masks(root, e)
        p0, p1 = load(root, f'REF_P_{e}_T0'), load(root, f'REF_P_{e}_T1')
        r1 = load(root, f'REF_ER_{e}_T1'); r0 = load(root, f'REF_ER_{e}_T0')
        d, da, db = halves(Y(p1['tot']) - Y(p0['tot']))
        dh, dha, dhb = halves(Y(p1['h']) - Y(p0['h']))
        t0 = Y(p0['tot']).mean(0); t1, t1a, t1b = halves(p1['tot']); e1r, e1a, e1b = halves(r1['tot'])
        R = {}
        for k in REG:
            m = M[k]
            if m.sum() < 30:
                R[k] = {'pixels': int(m.sum())}
                continue
            b = max(float(t0[m].mean()), 1e-30)
            R[k] = {'pixels': int(m.sum()), 'T0': b, 'signed_change_rel': float(d[m].mean() / b), 'abs_change_rel': float(np.abs(d[m]).mean() / b),
                    'abs_change_floor_rel': float(0.5 * np.abs(da[m] - db[m]).mean() / b),
                    'absH_change_rel': float(np.abs(dh[m]).mean() / b), 'absH_change_floor_rel': float(0.5 * np.abs(dha[m] - dhb[m]).mean() / b),
                    'H_share_T1': float(p1['h'].mean(0)[m].sum() / t1[m].sum()),
                    'uncertainty_P_T1': 0.5 * rl(t1a, t1b, m), 'uncertainty_ER_T1': 0.5 * rl(e1a, e1b, m),
                    'g_ER_vs_P_T1': rl(e1r, t1, m), 'g_signed_T1': float((e1r[m].sum() - t1[m].sum()) / t1[m].sum()),
                    'g_ER_vs_P_T0': rl(r0['tot'].mean(0), p0['tot'].mean(0), m)}
        # sharp / moderate involvement in the CHANGE of the >= 2-event radiance in C (REF-ER, common random numbers): panel / sphere
        # receivers count fully, otherwise the sender-class channels S, M
        mC = M['C']
        sm_recv = M['obj_panel'] | M['obj_sphere']
        dc = np.abs((r1['c2'] - r0['c2']).mean(0)) + np.abs((r1['ch'] - r0['ch']).mean(0))          # [H, W, 3 sender classes]
        allc = dc.sum(-1)
        sm = float((allc[mC & sm_recv].sum() + dc[..., :2].sum(-1)[mC & ~sm_recv].sum()) / max(allc[mC].sum(), 1e-30)) if mC.sum() else 0.0
        q = {}
        g = lambda k, f: R[k].get(f, 0.0)
        sig = lambda k, f='abs_change_rel', fl_='abs_change_floor_rel': g(k, f) >= G0['snr'] * g(k, fl_)
        q['Q1_A_direct_change'] = {'value': g('A', 'abs_change_rel'), 'PASS': bool(g('A', 'abs_change_rel') >= G0['A_min'] and sig('A'))}
        q['Q2_B_indirect_change'] = {'value': g('Bset', 'abs_change_rel'), 'floor': g('Bset', 'abs_change_floor_rel'),
                                     'PASS': bool(g('Bset', 'abs_change_rel') >= G0['B_min'] and sig('Bset'))}
        q['Q3_C_sharp_moderate'] = {'value': g('C', 'abs_change_rel'), 'sm_share': sm, 'pixels': R['C']['pixels'],
                                    'PASS': bool(R['C']['pixels'] >= G0['min_pixels'] and g('C', 'abs_change_rel') >= G0['C_min'] and sig('C') and sm >= G0['C_sm_share_min'])}
        ratio = g('D', 'absH_change_rel') / max(g('D', 'abs_change_rel'), 1e-30)
        q['Q4_D_broad_multibounce'] = {'value': g('D', 'abs_change_rel'), 'absH_over_abs': ratio, 'absH': g('D', 'absH_change_rel'), 'absH_floor': g('D', 'absH_change_floor_rel'),
                                       'PASS': bool(g('D', 'abs_change_rel') >= G0['D_min'] and sig('D') and ratio >= G0['D_H_ratio_min']
                                                    and sig('D', 'absH_change_rel', 'absH_change_floor_rel'))}
        q['Q5_E_unaffected'] = {'pixels': R['E']['pixels'], 'value': g('E', 'abs_change_rel'), 'ratio_to_D': g('E', 'abs_change_rel') / max(g('D', 'abs_change_rel'), 1e-30),
                                'PASS': bool(R['E']['pixels'] >= G0['min_pixels'] and g('E', 'abs_change_rel') <= G0['E_max_ratio_to_D'] * g('D', 'abs_change_rel'))}
        q['Q6_H_share'] = {'value': g('stationary_all', 'H_share_T1'), 'PASS': bool(g('stationary_all', 'H_share_T1') >= G0['H_share_min'])}
        un = max(g('stationary_all', 'uncertainty_P_T1'), g('stationary_all', 'uncertainty_ER_T1'))
        q['Q7_uncertainty'] = {'value': un, 'PASS': bool(un <= G0['uncertainty_max'])}
        q['QUALIFIED'] = bool(all(v['PASS'] for k, v in q.items() if not k.startswith('Q5')))      # Q5 is a reported control class
        ok_all &= q['QUALIFIED']
        out[e] = {'regions': R, 'qualification': q}
    tj = os.path.join(root, 'results', 'tests', 'tests86.json')
    out['tests_pass'] = bool(json.load(open(tj))['PASS']) if os.path.exists(tj) else False
    ta = os.path.join(root, 'results', 'tests', 'tests86_amend1.json')          # amendment 1 (post hoc): V3 in-place clause re-evaluated
    if os.path.exists(ta):
        out['tests_pass_frozen'] = out['tests_pass']; out['amendment'] = 1
        out['tests_pass'] = bool(json.load(open(ta))['PASS'])
    out['ADMITTED'] = bool(ok_all and out['tests_pass'])
    json.dump(out, open(os.path.join(root, 'results', 'qualify.json'), 'w'), indent=1)
    print(json.dumps({e: {k: (v['PASS'] if isinstance(v, dict) else v) for k, v in out[e]['qualification'].items()} for e in ('E1', 'E2')}))
    print('ADMITTED' if out['ADMITTED'] else 'STOPPED_UNQUALIFIED_WORKLOAD', flush=True)


def eh_table(root, ref, ec, M):
    """E_H of an ECRO-type render set against the pooled REF-ER channels, per region, with noise bounds."""
    h, ha, hb = halves(ref['h']); tot = ref['tot'].mean(0)
    hl, hla, hlb = halves(ec['hl'])
    T = {}
    for k in REG:
        m = M[k]
        if m.sum() < 30:
            continue
        den = max(tot[m].sum(), 1e-30)
        T[k] = {'E_H': float(np.abs(hl[m] - h[m]).sum() / den), 'H_share': float(h[m].sum() / den),
                'noise_bound': float((0.5 * np.abs(hla[m] - hlb[m]).sum() + 0.5 * np.abs(ha[m] - hb[m]).sum()) / den),
                'hl_over_h': float(hl[m].sum() / max(h[m].sum(), 1e-30))}
        T[k]['capture'] = T[k]['E_H'] / max(T[k]['H_share'], 1e-30)
    return T


def dh_table(ref0, ref1, ec0, ec1, M):
    x, xa, xb = halves(Y(ref1['h']) - Y(ref0['h']))
    y, ya, yb = halves(Y(ec1['hl']) - Y(ec0['hl']))
    t0 = Y(ref0['tot']).mean(0)
    T = {}
    for k in REG:
        m = M[k]
        if m.sum() < 30:
            continue
        sl, co = slope_corr(x, xa, xb, y, ya, yb, m)
        b = max(float(t0[m].mean()), 1e-30)
        T[k] = {'slope': sl, 'corr': co, 'true_absdH_rel': float(np.abs(x[m]).mean() / b), 'true_absdH_floor_rel': float(0.5 * np.abs(xa[m] - xb[m]).mean() / b),
                'err_absdH_rel': float(np.abs(y[m] - x[m]).mean() / b)}
        T[k]['resolvable'] = bool(T[k]['true_absdH_rel'] >= G1['snr'] * T[k]['true_absdH_floor_rel'])
    return T


def stage1(root):
    out = {}
    ident = True; rep = True
    for e in ('E1', 'E2'):
        M, _ = masks(root, e)
        refs = {t: load(root, f'REF_ER_{e}_{t}') for t in ('T0', 'T1')}
        ora = {t: load(root, f'ORA_{e}_{t}') for t in ('T0', 'T1')}
        o = {'oracle': {t: eh_table(root, refs[t], ora[t], M) for t in ('T0', 'T1')}, 'oracle_dH': dh_table(refs['T0'], refs['T1'], ora['T0'], ora['T1'], M)}
        tot, ta, tb = halves(refs['T1']['tot'])
        o['ref_uncertainty_T1'] = 0.5 * rl(ta, tb, M['stationary_all'])
        hs = o['oracle']['T1']['stationary_all']['H_share']; hd = o['oracle']['T1']['D']['H_share']
        o['I1_H_identifiable'] = {'H_share_stationary': hs, 'H_share_D': hd,
                                  'PASS': bool(hs >= max(G1['H_share_min'], G1['snr'] * o['ref_uncertainty_T1']) and hd >= G1['H_share_min'])}
        o['I2_dH_resolvable_D'] = o['oracle_dH']['D']['resolvable']
        r1 = True
        for t in ('T0', 'T1'):
            T = o['oracle'][t]
            r1 &= T['stationary_all']['E_H'] <= G1['oracle_EH_stationary_max'] and T['stationary_all']['capture'] <= G1['oracle_capture_max']
            for k in ('A', 'C', 'D'):
                if k in T:
                    r1 &= T[k]['E_H'] <= G1['oracle_EH_region_max']
        o['R1_representation'] = {'PASS': bool(r1)}
        ident &= o['I1_H_identifiable']['PASS']; rep &= bool(r1)
        out[e] = o
    out['STATUS'] = 'PASS' if ident and rep else ('STOPPED_HIGH_ORDER_BENEFIT_NOT_IDENTIFIABLE' if not ident else 'STOPPED_REPRESENTATION_INADEQUATE')
    json.dump(out, open(os.path.join(root, 'results', 'stage1.json'), 'w'), indent=1)
    for e in ('E1', 'E2'):
        print(e, json.dumps({t: {k: [round(v['E_H'], 4), round(v['H_share'], 3), round(v['noise_bound'], 4)] for k, v in out[e]['oracle'][t].items() if k in ('stationary_all', 'A', 'C', 'D', 'E', 'obj_panel', 'obj_sphere')} for t in ('T0', 'T1')}))
    print(out['STATUS'], flush=True)


def final(root):
    out = {'seeds': {}}
    seeds = [f'ecro_s{s}' for s in P['training']['seeds']]
    refs = {c: load(root, f'REF_ER_{c}') for c in CFGS}
    Ms = {e: masks(root, e)[0] for e in ('E1', 'E2')}
    s1 = json.load(open(os.path.join(root, 'results', 'stage1.json')))
    src = {}
    for ls in ('LMIX', 'LSPOT'):
        mm, _ = masks(root, f'E1_{ls}') if os.path.exists(os.path.join(root, 'results', 'masks', f'E1_{ls}.npz')) else (Ms['E1'], None)
        src[ls] = (load(root, f'REF_ER_E1_T1_{ls}'), mm)
    for sd in seeds:
        o = {'fidelity': {}, 'dH': {}, 'stale': {}, 'ablation': {}, 'sources': {}}
        mj = json.load(open(os.path.join(root, 'results', 'models', f'{sd}.json')))
        ex = mj
        o['field'] = {'val': ex['val'], 'train_hq': ex['trainhq'], 'val_no_global': ex['val_no_global'], 'val_no_single': ex['val_no_single'],
                      'params': mj['params'], 'train_s': mj['train_s']}
        gen_ok = ex['val']['rel_l1'] <= G3['val_over_train_max'] * ex['trainhq']['rel_l1']
        f1 = f2 = True
        for e in ('E1', 'E2'):
            M = Ms[e]
            ec = {t: load(root, f'EC_{sd}_ecro_{e}_{t}') for t in ('T0', 'T1')}
            o['fidelity'][e] = {t: eh_table(root, refs[f'{e}_{t}'], ec[t], M) for t in ('T0', 'T1')}
            for t in ('T0', 'T1'):
                T = o['fidelity'][e][t]
                f1 &= T['stationary_all']['E_H'] <= G3['EH_stationary_max'] and T['stationary_all']['capture'] <= G3['capture_max']
            for k in ('A', 'C', 'D'):
                if k in o['fidelity'][e]['T1']:
                    f2 &= o['fidelity'][e]['T1'][k]['E_H'] <= G3['EH_region_max']
            o['dH'][e] = dh_table(refs[f'{e}_T0'], refs[f'{e}_T1'], ec['T0'], ec['T1'], M)
            st = load(root, f'EC_{sd}_stale_{e}_T1')
            o['stale'][e] = {'E_H': eh_table(root, refs[f'{e}_T1'], st, M), 'dH': dh_table(refs[f'{e}_T0'], refs[f'{e}_T1'], ec['T0'], st, M)}
            o['ablation'][e] = {v: eh_table(root, refs[f'{e}_T1'], load(root, f'EC_{sd}_{v}_{e}_T1'), M) for v in ('nog', 'nos')}
        f3 = {}
        for e in ('E1', 'E2'):
            d = o['dH'][e]['D']
            f3[e] = 'INDETERMINATE' if not d['resolvable'] else ('PASS' if (G3['dH_slope'][0] <= d['slope'] <= G3['dH_slope'][1] and d['corr'] >= G3['dH_corr_min']) else 'FAIL')
        f4 = True
        for ls, (rf, mm) in src.items():
            T = eh_table(root, rf, load(root, f'EC_{sd}_ecro_E1_T1_{ls}'), mm)
            o['sources'][ls] = T
            f4 &= T['stationary_all']['E_H'] <= G3['EH_unseen_source_max']
        need = {}
        for e in ('E1', 'E2'):
            need[e] = {k: (o['stale'][e]['E_H'][k]['E_H'] / max(o['fidelity'][e]['T1'][k]['E_H'], 1e-30)) for k in ('A', 'C', 'D', 'obj_panel', 'obj_sphere', 'obj_boxes_static')
                       if k in o['stale'][e]['E_H'] and k in o['fidelity'][e]['T1']}
        f5 = any(v >= G3['stale_ratio_min'] for e in need for v in need[e].values())
        o['gates'] = {'G1_field_generalisation': bool(gen_ok), 'F1_stationary': bool(f1), 'F2_regions': bool(f2), 'F3_dH_D': f3, 'F4_unseen_sources': bool(f4),
                      'F5_stale_control_update_needed': bool(f5), 'stale_over_ecro_EH': need}
        o['FIDELITY_PASS'] = bool(f1 and f2 and f4 and all(v != 'FAIL' for v in f3.values()))
        o['GENERALISATION_PASS'] = bool(gen_ok and f1 and f2)
        out['seeds'][sd] = o
    n_ok = sum(v['FIDELITY_PASS'] and v['GENERALISATION_PASS'] for v in out['seeds'].values())
    train_ok = any(v['field']['train_hq']['rel_l1'] <= G3['train_field_fit_max'] for v in out['seeds'].values())
    out['passing_seeds'] = n_ok
    out['STATUS'] = 'PASS' if n_ok == len(seeds) else ('PASS_ONE_SEED' if n_ok else ('STOPPED_GENERALIZATION_NOT_SUPPORTED' if train_ok else 'FIDELITY_NOT_SUPPORTED'))
    out['economics_seed'] = next((k for k, v in out['seeds'].items() if v['FIDELITY_PASS'] and v['GENERALISATION_PASS']), seeds[0])
    json.dump(out, open(os.path.join(root, 'results', 'final.json'), 'w'), indent=1)
    for sd, o in out['seeds'].items():
        print(sd, json.dumps(o['gates']), json.dumps({e: {t: {k: round(v['E_H'], 4) for k, v in o['fidelity'][e][t].items() if k in ('stationary_all', 'A', 'C', 'D', 'E')} for t in ('T0', 'T1')} for e in ('E1', 'E2')}))
    print(out['STATUS'], flush=True)


if __name__ == '__main__':
    {'qualify': qualify, 'stage1': stage1, 'final': final}[sys.argv[1]](sys.argv[2])
