"""WL89 analysis and prospective rules (protocol stage0 / stage1 / stage2 / stage3 / verdict_mapping). numpy only.

  python analyze89.py wqualify <R89> <name>   qualification on the independent qualification partition (REF_ERQ, 8 x 8192 spp per T)
  python analyze89.py stage1 <R89>            fidelity, target identity, decomposition, coverage, matched-budget comparisons
  python analyze89.py stage2 <R89>            realizations, held-out edit / sources, source independence, edit validity
  python analyze89.py stage3 <R89>            lifecycle economics, quality matching, bottleneck, scaling
  python analyze89.py verdict <R89>           H0 / H1 / H2 and verdict items -> final89.json
E_H(X) = sum |hl_X - h_REF-ER| / sum tot_REF-ER (WL86); signed S(X) = sum (hl_X - h_REF-ER) / sum tot_REF-ER.
"""
import glob, json, math, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'ecro86'))
sys.path.insert(0, os.path.join(HERE, '..', 'gm1_relation88'))
P89 = json.load(open(os.path.join(HERE, 'protocol', 'gm1_relation89.json')))
if os.environ.get('WL89_DEV'):
    P89['w_order'] = [P89['dev_workload']]; P89['w2'] = P89['dev_workload']
import analyze86 as A  # noqa: E402
import analyze88 as A8  # noqa: E402  (numpy only; WL88 helpers eh / dh / field_cmp / interp / ci95, unchanged)
import scene86 as S  # noqa: E402
NB = S.NB
TH = P89['stage1']['thresholds']
Y = A.Y
REG = ['stationary_all', 'A', 'C_ss', 'R_mirror', 'D', 'E', 'obj_floor', 'obj_walls', 'obj_panel', 'obj_ceiling']
SHARP = ['C_ss', 'R_mirror']
ARMS = P89['stage1']['arms']
RARMS = P89['stage1']['real_arms']
CAND = ['R0', 'C3UR', 'R1']
jd, jl, load_hl, eh, dh, ci95, interp = A8.jd, A8.jl, A8.load_hl, A8.eh, A8.dh, A8.ci95, A8.interp


def f3(d, k):
    if k not in d:
        return 'NA'
    v = d[k]
    if not v['resolvable'] or v['slope'] is None:
        return 'INDETERMINATE'
    return 'PASS' if TH['dH_slope'][0] <= v['slope'] <= TH['dH_slope'][1] and v['corr'] >= TH['dH_corr_min'] else 'FAIL'


def masks(R, name):
    z = np.load(os.path.join(R, 'results', 'masks', f'{name}.npz'))
    return {k: z[k] for k in z.files if z[k].dtype == bool}


def w1_admitted(R):
    return os.path.exists(os.path.join(R, 'results', 'masks', 'W1.npz'))


def geom_of(R, cfg):
    e, t = cfg.split('_')[:2]
    name = jl(os.path.join(R, 'results', 'w_choice.json'))['chosen'] if e == 'W1' else P89['w2']
    return S.Geom(P89['workloads'][name][t]) if name in P89['workloads'] else S.Geom(A8.P88['workloads'][name][t])


# ------------------------------------------------------------------------------------------------ Stage 0: qualification
def wqualify(R, name):
    """Floors are computed on the qualification partition itself, which has the final evaluation reference budget (8 x 8192 spp per T):
    a change counts as resolvable iff it is >= snr x its floor at the budget the evaluation will have (WL88 Q5 inconsistency removed)."""
    Q0 = P89['stage0']['qualification']
    M = masks(R, f'M_{name}')
    e0, e1 = A.load(R, f'REF_ERQ_{name}_T0'), A.load(R, f'REF_ERQ_{name}_T1')
    t0 = Y(e0['tot']).mean(0)
    Rg = {}
    for k in ['stationary_all', 'A', 'Bset', 'C_ss', 'R_mirror', 'D', 'E']:
        m = M[k]
        if m.sum() < 30:
            Rg[k] = {'pixels': int(m.sum())}
            continue
        b = max(float(t0[m].mean()), 1e-30)
        r = {'pixels': int(m.sum())}
        for key in ('tot', 'h', 'H2', 'H1G', 'H2G'):
            x, xa, xb = A.halves(Y(e1[key]) - Y(e0[key]))
            r[f'abs_{key}'] = float(np.abs(x[m]).mean() / b); r[f'floor_{key}'] = float(0.5 * np.abs(xa[m] - xb[m]).mean() / b)
        dh_ = np.abs((Y(e1['h']) - Y(e0['h'])).mean(0))[m].sum()
        r['H2_share'] = float(np.abs((Y(e1['H2']) - Y(e0['H2'])).mean(0))[m].sum() / max(dh_, 1e-30))
        r['G_share'] = float((np.abs((Y(e1['H1G']) - Y(e0['H1G'])).mean(0))[m].sum() + np.abs((Y(e1['H2G']) - Y(e0['H2G'])).mean(0))[m].sum()) / max(dh_, 1e-30))
        tt, ta, tb = A.halves(e1['tot'])
        r['H_share_T1'] = float(e1['h'].mean(0)[m].sum() / tt[m].sum())
        r['uncertainty_T1'] = 0.5 * A.rl(ta, tb, m)
        hh, ha, hb = A.halves(e1['h'])
        r['h_noise_rel'] = float(0.5 * np.abs(ha[m] - hb[m]).sum() / tt[m].sum())
        Rg[k] = r
    g = lambda k, f: Rg[k].get(f, 0.0)
    sig = lambda k, key='tot': g(k, f'abs_{key}') >= Q0['snr'] * g(k, f'floor_{key}')
    q = {}
    q['Q1_A_direct_change'] = {'value': g('A', 'abs_tot'), 'PASS': bool(g('A', 'abs_tot') >= Q0['A_min'] and sig('A'))}
    q['Q2_B_indirect_change'] = {'value': g('Bset', 'abs_tot'), 'PASS': bool(g('Bset', 'abs_tot') >= Q0['B_min'] and sig('Bset'))}
    q['Q3_C_ss_sharp_to_sharp'] = {'pixels': Rg['C_ss']['pixels'], 'value': g('C_ss', 'abs_tot'), 'H2_share': g('C_ss', 'H2_share'),
                                   'PASS': bool(Rg['C_ss']['pixels'] >= Q0['min_pixels'] and g('C_ss', 'abs_tot') >= Q0['C_min'] and sig('C_ss')
                                                and g('C_ss', 'H2_share') >= Q0['family_share_min'] and sig('C_ss', 'H2'))}
    q['Q4_R_mirror_moving_sharp_sender'] = {'pixels': Rg['R_mirror']['pixels'], 'value': g('R_mirror', 'abs_tot'), 'G_share': g('R_mirror', 'G_share'),
                                            'PASS': bool(Rg['R_mirror']['pixels'] >= Q0['min_pixels'] and g('R_mirror', 'abs_tot') >= Q0['C_min']
                                                         and sig('R_mirror') and g('R_mirror', 'G_share') >= Q0['family_share_min'])}
    ratio = g('D', 'abs_h') / max(g('D', 'abs_tot'), 1e-30)
    q['Q5_D_broad_multibounce'] = {'value': g('D', 'abs_tot'), 'absH_over_abs': ratio, 'snr_h': g('D', 'abs_h') / max(g('D', 'floor_h'), 1e-30),
                                   'PASS': bool(g('D', 'abs_tot') >= Q0['D_min'] and sig('D') and ratio >= Q0['D_H_ratio_min'] and sig('D', 'h'))}
    q['Q6_H_share'] = {'value': g('stationary_all', 'H_share_T1'), 'PASS': bool(g('stationary_all', 'H_share_T1') >= Q0['H_share_min'])}
    q['Q7_uncertainty'] = {'value': g('stationary_all', 'uncertainty_T1'), 'PASS': bool(g('stationary_all', 'uncertainty_T1') <= Q0['uncertainty_max'])}
    rn = max(g('C_ss', 'h_noise_rel'), g('R_mirror', 'h_noise_rel'))
    q['Q8_reference_resolution_sharp'] = {'value': rn, 'PASS': bool(rn <= Q0['sharp_ref_noise_max'])}
    adm = bool(all(v['PASS'] for v in q.values()))
    jd({'candidate': name, 'regions': Rg, 'qualification': q, 'ADMITTED': adm}, os.path.join(R, 'results', f'wq_{name}.json'))
    print('WQUALIFY', name, 'ADMITTED' if adm else 'NOT_ADMITTED', json.dumps({k: v['PASS'] for k, v in q.items()}), flush=True)


# ------------------------------------------------------------------------------------------------ Stage 1
def field_block(R, cfg, xz, rr):
    G = geom_of(R, cfg)
    oo = np.load(os.path.join(R, 'results', 'fields', f'ORD_{cfg}.npz'))
    a, b = oo['a'].astype(np.float64), oo['b'].astype(np.float64)
    T = 0.5 * (a + b); Tn = 0.5 * np.abs(a - b)
    mz = np.load(os.path.join(R, 'results', 'masks', 'W1.npz'))
    pids = np.unique(mz['pid'][mz['C_ss'] | mz['R_mirror']])
    fp = np.zeros(G.N, bool); fp[pids[(pids >= 0) & (pids < G.N)]] = True
    Bt, Btn = T[1:].sum(0), np.sqrt((Tn[1:] ** 2).sum(0))
    o = {'field': {arm: A8.field_cmp(xz[arm].astype(np.float64), Bt, Btn, G, fp) for arm in xz.files if arm in ARMS}}
    o['orders'] = {'q_vs_order1': A8.field_cmp(xz['q'].astype(np.float64), T[0], Tn[0], G, fp)['all']}
    for arm in ('R0', 'R1', 'C3UR'):
        o['orders'][arm] = [A8.field_cmp(xz[f'{arm}_L{j + 2}'].astype(np.float64), T[j + 1], Tn[j + 1], G, fp) for j in range(4)]
    if rr is not None:
        for arm in ('R0_INF', 'R1_INF', 'C3UR_INF', 'R0_POOL16', 'R1_POOL16', 'C3UR_POOL16'):
            o['field'][arm] = A8.field_cmp(rr[arm].astype(np.float64), Bt, Btn, G, fp)
        o['target_identity_field'] = {k: A8.field_cmp(rr[k].astype(np.float64), rr['R0_INF'].astype(np.float64), np.zeros_like(Bt), G, fp)
                                      for k in ('R1_INF', 'C3UR_INF')}
    return o


def stage1(R):
    out = {}
    cfgs = P89['stage1']['configs']
    for cfg in cfgs:
        M = masks(R, 'W1')
        ref = A.load(R, f'REF_ER_{cfg}')
        rends = {}
        for arm in ARMS:
            if glob.glob(os.path.join(R, 'results', 'renders', f'X89_{arm}_{cfg}_r*.npz')):
                rends[arm] = load_hl(R, f'X89_{arm}_{cfg}')
        for arm in RARMS:
            if glob.glob(os.path.join(R, 'results', 'renders', f'X89R_{arm}_{cfg}_r*.npz')):
                rends[arm] = load_hl(R, f'X89R_{arm}_{cfg}')
        o = {'fidelity': {arm: eh(ref, ec, M, REG) for arm, ec in rends.items()}}
        xz = np.load(os.path.join(R, 'results', 'tables', f'X_{cfg}.npz'))
        rz = os.path.join(R, 'results', 'tables', f'R_{cfg}.npz')
        o.update(field_block(R, cfg, xz, np.load(rz) if os.path.exists(rz) else None))
        out[cfg] = o
    M = masks(R, 'W1')
    r0, r1 = A.load(R, 'REF_ER_W1_T0'), A.load(R, 'REF_ER_W1_T1')
    out['dH'] = {arm: dh(r0, r1, load_hl(R, f'X89_{arm}_W1_T0'), load_hl(R, f'X89_{arm}_W1_T1'), M, REG) for arm in ARMS
                 if glob.glob(os.path.join(R, 'results', 'renders', f'X89_{arm}_W1_T0_r*.npz'))}
    # signed change maps reference (for visuals): stored by viz
    # target identity (converged relations agree; physical target unchanged by the construction)
    ident = {}
    tol = P89['stage1']['target_identity_signed_tol']
    for cfg in cfgs:
        F_ = out[cfg]['fidelity']
        if 'R0_INF' not in F_:
            continue
        ident[cfg] = {k: {a: {'d_signed': F_[a][k]['signed'] - F_['R0_INF'][k]['signed'], 'noise': max(F_[a][k]['noise_bound'], F_['R0_INF'][k]['noise_bound'])}
                          for a in ('R1_INF', 'C3UR_INF') if a in F_ and k in F_[a]} for k in SHARP + ['stationary_all', 'D'] if k in F_['R0_INF']}
    ok = all(abs(v['d_signed']) <= tol + v['noise'] for c in ident.values() for r in c.values() for v in r.values())
    fld = all(abs(v['all']['flux_ratio'] - 1) <= P89['stage1']['target_identity_flux_tol'] for cfg in cfgs if 'target_identity_field' in out[cfg]
              for v in out[cfg]['target_identity_field'].values())
    out['target_identity'] = {'rendered': ident, 'PASS': bool(ok and fld and len(ident) > 0), 'rendered_ok': bool(ok), 'field_flux_ok': bool(fld)}
    # RAYSEND repair validation (gated before RS0 / RS1 are used)
    rv = {cfg: jl(os.path.join(R, 'results', 'tables', f'X_{cfg}.json'))['timing']['raysend_validation'] for cfg in cfgs}
    out['raysend_validation'] = {'per_config': rv, 'PASS': all(v['RS1_table_vs_apply_maxrel'] <= 1e-5 and v['RS0_table_vs_apply_maxrel'] <= 1e-5
                                                             and v['RS1_area_exact_hit_fraction'] >= 0.99 for v in rv.values())}
    # gates per arm
    gates = {}
    for arm in ARMS + RARMS:
        res = {}; fails = []
        for cfg in cfgs:
            F_ = out[cfg]['fidelity'].get(arm)
            if F_ is None:
                continue
            v = F_['stationary_all']
            res[f'{cfg}:F1'] = bool(v['E_H'] <= TH['EH_stationary_max'] and v['capture'] <= TH['capture_max'])
            if not res[f'{cfg}:F1']:
                fails.append({'where': f'{cfg}:stationary_all', 'excess': v['E_H'] - TH['EH_stationary_max'], 'noise_bound': v['noise_bound']})
            if cfg.endswith('T1'):
                for k in ('A', 'C_ss', 'R_mirror', 'D'):
                    if k in F_:
                        res[f'{cfg}:F2:{k}'] = bool(F_[k]['E_H'] <= TH['EH_region_max'])
                        if not res[f'{cfg}:F2:{k}']:
                            fails.append({'where': f'{cfg}:{k}', 'excess': F_[k]['E_H'] - TH['EH_region_max'], 'noise_bound': F_[k]['noise_bound']})
        if arm in out['dH']:
            for k in ['D'] + SHARP:
                res[f'W1:F3:{k}'] = f3(out['dH'][arm], k)
        gates[arm] = {'results': res, 'failing': fails, 'F1': all(v for k, v in res.items() if ':F1' in k),
                      'F2': all(v for k, v in res.items() if ':F2:' in k), 'F2_sharp': all(v for k, v in res.items() if ':F2:C_ss' in k or ':F2:R_mirror' in k),
                      'F3': all(v != 'FAIL' for k, v in res.items() if ':F3:' in k), 'F3_values': {k: v for k, v in res.items() if ':F3:' in k}}
    out['gates'] = gates
    # comparisons (deployed draws; both renders' noise)
    cmp_ = {}
    pairs = [('C3UR', 'R0'), ('R1', 'R0'), ('R1M', 'R0'), ('C3UR', 'R1M'), ('C3UR', 'R1'), ('C3UR16', 'POOL2'), ('C3', 'C3UR'), ('C3U', 'C3UR')]
    for cfg in cfgs:
        F_ = out[cfg]['fidelity']
        cmp_[cfg] = {f'{a}_vs_{b}': {k: {'d_E_H': F_[a][k]['E_H'] - F_[b][k]['E_H'], 'd_signed': F_[a][k]['signed'] - F_[b][k]['signed'],
                                         'noise': max(F_[a][k]['noise_bound'], F_[b][k]['noise_bound'])} for k in F_[a] if k in F_[b]}
                     for a, b in pairs if a in F_ and b in F_}
    out['comparisons'] = cmp_
    # error decomposition in the sharp regions (signed, T1): sampling = deployed - converged; first-gather source projection = RS - arm;
    # relation discretisation = converged - oracle table; output representation = oracle - reference
    dec = {}
    for cfg in cfgs:
        F_ = out[cfg]['fidelity']
        dec[cfg] = {}
        for k in SHARP + ['D', 'stationary_all']:
            if k not in F_.get('R0', {}):
                continue
            s = lambda a: F_[a][k]['signed'] if a in F_ and k in F_[a] else None
            d = {'output_representation (ORACLE - REF)': s('ORACLE')}
            for arm, inf, rs in (('R0', 'R0_INF', 'RS0'), ('R1', 'R1_INF', 'RS1'), ('C3UR', 'C3UR_INF', None)):
                if s(arm) is None:
                    continue
                d[f'{arm}: sampling (deployed - converged)'] = (s(arm) - s(inf)) if s(inf) is not None else None
                d[f'{arm}: relation discretisation (converged - ORACLE)'] = (s(inf) - s('ORACLE')) if s(inf) is not None and s('ORACLE') is not None else None
                if rs and s(rs) is not None:
                    d[f'{arm}: first-gather sender projection (arm - RS)'] = s(arm) - s(rs)
            d['noise_bound'] = F_['R0'][k]['noise_bound']
            dec[cfg][k] = d
    out['decomposition'] = dec
    cov = {}
    for cfg in P89['stage2']['real_configs']:
        fn = os.path.join(R, 'results', 'tables', f'R_{cfg}.json')
        if os.path.exists(fn):
            c = jl(fn)['coverage']
            cov[cfg] = {k: {'mean': float(np.mean([v[k] for v in c.values()])), 'min': float(np.min([v[k] for v in c.values()])),
                            'max': float(np.max([v[k] for v in c.values()]))} for k in next(iter(c.values()))}
    out['coverage'] = cov
    lim = {}
    for cfg in cfgs:
        F_ = out[cfg]['fidelity']
        if cfg.endswith('T1'):
            for k in SHARP:
                if k in F_.get('R0', {}):
                    lim[f'{cfg}:{k}'] = {a: bool(F_[a][k]['E_H'] <= TH['EH_region_max']) for a in ('ORACLE', 'R0', 'C3UR', 'R1', 'C3', 'R0_INF', 'R1_INF', 'C3UR_INF') if a in F_}
    out['limit_attribution'] = lim
    meta = {cfg: jl(os.path.join(R, 'results', 'tables', f'X_{cfg}.json')) for cfg in cfgs}
    out['tables_meta'] = meta
    t89 = jl(os.path.join(R, 'results', 'tests89.json'))
    out['validity'] = {'tests_pass': t89['ALL_PASS'], 'finite_nonneg': all(m['finite'] and min(m['min'].values()) >= 0 for m in meta.values()),
                       'target_identity': out['target_identity']['PASS'], 'raysend_repaired': out['raysend_validation']['PASS']}
    jd(out, os.path.join(R, 'results', 'stage1_89.json'))
    for cfg in cfgs:
        F_ = out[cfg]['fidelity']
        print(cfg, json.dumps({arm: {k: round(F_[arm][k]['E_H'], 4) for k in ('stationary_all', 'C_ss', 'R_mirror', 'D') if k in F_[arm]}
                               for arm in ('R0', 'C3UR', 'R1', 'R1M', 'C3', 'ORACLE', 'R0_INF', 'C3UR_INF', 'R1_INF') if arm in F_}), flush=True)
    print('STAGE1 validity', json.dumps(out['validity']), flush=True)


# ------------------------------------------------------------------------------------------------ Stage 2
def stage2(R):
    out = {}
    S2 = P89['stage2']
    real = {}
    seeds = S2['seeds']
    for cfg in S2['real_configs']:
        M = masks(R, 'W1')
        ref = A.load(R, f'REF_ER_{cfg}')
        regs = SHARP + ['stationary_all', 'D']
        per = {a: {sd: eh(ref, load_hl(R, f'X89S_{a}_{cfg}_s{sd}'), M, regs) for sd in seeds} for a in CAND}
        rr = {}
        for a in CAND:
            rr[a] = {}
            for k in regs:
                v = [per[a][sd][k]['E_H'] for sd in seeds if k in per[a][sd]]
                sg = [per[a][sd][k]['signed'] for sd in seeds if k in per[a][sd]]
                if not v:
                    continue
                lim_ = TH['EH_stationary_max'] if k == 'stationary_all' else TH['EH_region_max']
                rr[a][k] = {'E_H_mean': float(np.mean(v)), 'E_H_sd': float(np.std(v, ddof=1)), 'E_H_min': float(np.min(v)), 'E_H_max': float(np.max(v)),
                            'signed_mean': float(np.mean(sg)), 'signed_ci95': ci95(sg), 'pass_frac_region': float(np.mean(np.array(v) <= lim_))}
            if cfg.endswith('T1'):
                rr[a]['pass_frac_all_regions'] = float(np.mean([all(per[a][sd][k]['E_H'] <= (TH['EH_stationary_max'] if k == 'stationary_all' else TH['EH_region_max'])
                                                                    for k in per[a][sd]) for sd in seeds]))
        rr['paired'] = {}
        for a, b in (('C3UR', 'R0'), ('R1', 'R0'), ('C3UR', 'R1')):
            rr['paired'][f'{a}_minus_{b}'] = {}
            for k in regs:
                d = [per[a][sd][k]['E_H'] - per[b][sd][k]['E_H'] for sd in seeds if k in per[a][sd] and k in per[b][sd]]
                if d:
                    rr['paired'][f'{a}_minus_{b}'][k] = {'mean': float(np.mean(d)), 'ci95': ci95(d), 'frac_first_better': float(np.mean(np.array(d) < 0))}
        real[cfg] = rr
    out['realizations'] = real
    ho = {}
    for cfg, ls in [('W2_T0', None), ('W2_T1', None), ('W1_T1', 'LMIX89'), ('W1_T1', 'LSPOT89')]:
        tag = cfg + ('_' + ls if ls else '')
        M = masks(R, ('W2' if cfg.startswith('W2') else 'W1') + ('_' + ls if ls else ''))
        ref = A.load(R, f'REF_ER_{tag}')
        ho[tag] = {arm: eh(ref, load_hl(R, f'X89_{arm}_{tag}'), M, REG) for arm in S2['heldout_arms']
                   if glob.glob(os.path.join(R, 'results', 'renders', f'X89_{arm}_{tag}_r*.npz'))}
    out['heldout'] = ho
    M = masks(R, 'W2')
    r0, r1 = A.load(R, 'REF_ER_W2_T0'), A.load(R, 'REF_ER_W2_T1')
    out['dH_W2'] = {arm: dh(r0, r1, load_hl(R, f'X89_{arm}_W2_T0'), load_hl(R, f'X89_{arm}_W2_T1'), M, REG) for arm in S2['heldout_arms']}
    gates = {}
    for arm in S2['heldout_arms']:
        r = {}
        for tag, T in ho.items():
            if arm not in T:
                continue
            v = T[arm]
            if 'LMIX' in tag or 'LSPOT' in tag:
                r[f'{tag}:F4'] = bool(v['stationary_all']['E_H'] <= TH['EH_unseen_source_max'])
                for k in SHARP:
                    if k in v and v[k]['pixels'] >= P89['stage0']['qualification']['min_pixels']:
                        r[f'{tag}:F2s:{k}'] = bool(v[k]['E_H'] <= TH['EH_region_max'])
            else:
                r[f'{tag}:F1'] = bool(v['stationary_all']['E_H'] <= TH['EH_stationary_max'] and v['stationary_all']['capture'] <= TH['capture_max'])
                if tag.endswith('T1'):
                    for k in ('A', 'C_ss', 'R_mirror', 'D'):
                        if k in v and v[k]['pixels'] >= P89['stage0']['qualification']['min_pixels']:
                            r[f'{tag}:F2:{k}'] = bool(v[k]['E_H'] <= TH['EH_region_max'])
        for k in ['D'] + SHARP:
            r[f'W2:F3:{k}'] = f3(out['dH_W2'][arm], k)
        gates[arm] = r
    out['heldout_gates'] = gates
    t89 = jl(os.path.join(R, 'results', 'tests89.json'))
    t88 = jl(os.path.join(R, 'results', 'tests88.json'))
    ecp = os.path.join(R, 'results', 'econ', 'econ89.json')
    ec = jl(ecp) if os.path.exists(ecp) else {}
    out['source_independence'] = {'C3UR_construction_no_source': t89['T_C3UR_SOURCE_INDEPENDENT']['PASS'], 'R1_construction_no_source': t88['T_SOURCE_INDEPENDENT']['PASS'],
                                  'digest_constant_across_sources': ec.get('source_reuse_digest_constant')}
    out['edit_validity'] = {'C3UR': t89['T_C3UR_EDIT'], 'R1': t88['T_EDIT'], 'econ_selective_equals_full': ec.get('selective_equals_full')}
    jd(out, os.path.join(R, 'results', 'stage2_89.json'))
    for cfg, rr in real.items():
        print('REAL', cfg, json.dumps({a: {k: [round(rr[a][k]['E_H_mean'], 4), round(rr[a][k]['E_H_sd'], 4), round(rr[a][k]['signed_mean'], 4)]
                                           for k in rr[a] if isinstance(rr[a][k], dict)} for a in CAND}), flush=True)
        print('PAIRED', cfg, json.dumps(rr['paired']), flush=True)
    print('STAGE2 heldout', json.dumps(gates), flush=True)


# ------------------------------------------------------------------------------------------------ Stage 3
def stage3(R):
    ec = jl(os.path.join(R, 'results', 'econ', 'econ89.json'))
    sc = jl(os.path.join(R, 'results', 'econ', 'scale89.json'))
    S3 = P89['stage3']
    em = ec['edit_median']
    base = {'A_geometry': em['A_geometry_s'], 'D_source_q': em['D_q_s'], 'F_transfer': em['F_transfer_s'], 'G_first_frame_extra': em['G_first_frame_extra_s']}
    life = {'R0': {**base, 'B_relation_discovery': em['B_R0_state_s'], 'C_coefficients': em['C_R0_coeff_s'], 'E_application': em['E_R0_apply_s']},
            'C3UR': {**base, 'BC_relation_update': em['C3UR_update_s'], 'E_application': em['E_C3UR_apply_s']},
            'R1': {**base, 'BC_relation_update': em['R1_update_s'], 'E_application': em['E_R1_apply_s']},
            'R1M': {**base, 'BC_relation_update': em['R1M_update_s'], 'E_application': em['E_R1M_apply_s']},
            'C3': {**base, 'BCE_sender_side_per_source': em['C3_s']},
            'PT': {'A_geometry': ec['pt_edit_median']['geometry_update_s'], 'G_first_frame_extra': ec['pt_edit_median']['first_frame_extra_s']}}
    for b, v in ec['tf_cost'].items():
        life[f'TF{b}'] = {**base, 'BCDE_fill': v['fill_s']}
        life[f'TF{b}'].pop('D_source_q')
    for a in life:
        life[a]['total_per_edit'] = float(sum(life[a].values()))
    out = {'lifecycle_per_edit': life}
    bott = {}
    for a in ('R0', 'C3UR', 'R1'):
        rel = life[a].get('BC_relation_update', life[a].get('B_relation_discovery', 0) + life[a].get('C_coefficients', 0))
        bott[a] = {'relation_update_s': rel, 'application_s': life[a]['E_application'], 'source_q_s': life[a]['D_source_q'],
                   'relation_update_share': rel / life[a]['total_per_edit'], 'application_share': life[a]['E_application'] / life[a]['total_per_edit']}
    out['bottleneck'] = bott
    out['update_parts'] = {'R1': ec['R1_update_parts_median'], 'C3UR': ec['C3UR_update_parts_median']}
    su = ec['source_update_median']
    out['source_only_update'] = {a: su['q_s'] + su[f'{a}_apply_s'] for a in ('R0', 'R1', 'C3UR')}
    out['source_only_update']['C3'] = su['q_s'] + su['C3_s']
    out['multi_source'] = {f'S{n}': {a: v * n for a, v in out['source_only_update'].items()} for n in S3['source_reuse_counts']}
    lm = ec['ladder_median']
    lv = S3['levels']
    qm = {}
    for key, lvls in [('err_stationary_all', lv['stationary']), ('err_C_ss', lv['sharp']), ('err_R_mirror', lv['sharp']), ('err_D', lv['region'])]:
        qm[key] = {}
        for l_ in lvls:
            row = {}
            for arm in lm:
                p = interp(lm[arm], l_, key)
                fixed = life[arm]['total_per_edit'] if arm in life else 0.0
                row[arm] = None if p is None else {**p, 'edit_to_first_frame_s': fixed + p['wall_s'], 'sustained_frame_s': p['wall_s']}
            qm[key][str(l_)] = row
    out['quality_matched'] = qm
    out['quality_floor'] = {arm: {k: min(r[k] for r in lm[arm]) for k in ('err_stationary_all', 'err_C_ss', 'err_R_mirror', 'err_D')} for arm in lm}
    adv = {}
    for key in qm:
        for l_, row in qm[key].items():
            for a in ('C3UR', 'R1', 'R1M'):
                k = f'{key}@{l_}'
                if row.get(a) and row.get('R0'):
                    adv.setdefault(f'{a}_over_R0', {})[k] = row['R0']['edit_to_first_frame_s'] / row[a]['edit_to_first_frame_s']
                elif row.get(a):
                    adv.setdefault(f'{a}_over_R0', {})[k] = 'R0 does not reach'
                elif row.get('R0'):
                    adv.setdefault(f'{a}_over_R0', {})[k] = f'{a} does not reach'
            for a in ('C3UR', 'R0', 'R1'):
                if row.get(a) and row.get('PT'):
                    adv.setdefault(f'PT_over_{a}', {})[f'{key}@{l_}'] = row['PT']['edit_to_first_frame_s'] / row[a]['edit_to_first_frame_s']
    out['speedups'] = adv
    V = {}
    l_ = str(lv['stationary'][-1])
    for n in S3['view_counts']:
        row = qm['err_stationary_all'][l_]
        V[f'V{n}'] = {a: (life[a]['total_per_edit'] if a in life else 0) + n * row[a]['wall_s'] for a in row if row[a]}
    out['view_amortisation_stationary'] = V
    out['memory'] = ec['memory']
    sc_out = {}
    for gb in (False, True):
        rr = [r for r in sc['rows'] if r['glossy_B'] == gb]
        if len(rr) < 2:
            continue
        N = np.log([r['N'] for r in rr])
        fit = lambda key: float(np.polyfit(N, np.log([max(r[key], 1e-9) for r in rr]), 1)[0])
        sc_out['glossy_B' if gb else 'G_only'] = {'N': [r['N'] for r in rr], 'exp': {k: fit(k) for k in ('R0_state_s', 'C3UR_build_s', 'R1_build_s', 'R1_pairs', 'C3UR_bytes',
                                                                                                           'R1_bytes', 'R0_apply_J4_s', 'C3UR_apply_J4_s', 'R1_apply_J4_s',
                                                                                                           'C3UR_update_s', 'R1_update_s', 'R0_update_s')}, 'rows': rr}
    out['scaling'] = sc_out
    jd(out, os.path.join(R, 'results', 'stage3_89.json'))
    print('STAGE3', json.dumps({a: round(life[a]['total_per_edit'], 4) for a in life}), json.dumps(bott), flush=True)


# ------------------------------------------------------------------------------------------------ verdict
def verdict(R):
    VM = P89['verdict_mapping']
    adm = {os.path.basename(f)[3:-5]: jl(f) for f in sorted(glob.glob(os.path.join(R, 'results', 'wq_*.json')))}
    if not w1_admitted(R):
        V = {'WORKLOAD_QUALIFICATION': 'NOT SUPPORTED (WORKLOAD_UNQUALIFIED: no predeclared candidate admitted)'}
        for k in ('EXPERIMENTAL_VALIDITY', 'C3UR_PHYSICAL_FIDELITY', 'C3UR_SAMPLING_RELATION_QUALITY', 'SOURCE_INDEPENDENT_REUSE', 'DYNAMIC_EDIT_VALIDITY',
                  'RELATION_CONSTRUCTION_ECONOMICS', 'OPERATOR_APPLICATION_ECONOMICS', 'QUALITY_MATCHED_ADVANTAGE', 'SCALING', 'H0', 'H1', 'H2'):
            V[k] = 'NOT TESTED'
        jd({'verdicts': V, 'qualification': {n: {k: v['PASS'] for k, v in a['qualification'].items()} for n, a in adm.items()}},
           os.path.join(R, 'results', 'final89.json'))
        print('VERDICT', json.dumps(V), flush=True)
        return
    s1 = jl(os.path.join(R, 'results', 'stage1_89.json'))
    s2 = jl(os.path.join(R, 'results', 'stage2_89.json'))
    s3 = jl(os.path.join(R, 'results', 'stage3_89.json'))
    V = {'WORKLOAD_QUALIFICATION': 'SUPPORTED (' + jl(os.path.join(R, 'results', 'w_choice.json'))['chosen'] + ' admitted)'}
    val = s1['validity']
    V['EXPERIMENTAL_VALIDITY'] = 'SUPPORTED' if all(val.values()) else 'NOT SUPPORTED (' + ', '.join(k for k, v in val.items() if not v) + ')'
    stop = not all(val.values())
    g = s1['gates']
    rr = s2['realizations']['W1_T1']
    pc = rr['paired']['C3UR_minus_R0']
    imp = [k for k in SHARP if k in pc and pc[k]['ci95'][1] < 0]
    wor = [k for k in pc if pc[k]['ci95'][0] > 0]
    hg = s2['heldout_gates']

    def fid(arm):
        G_ = g[arm]
        ok = G_['F1'] and G_['F2'] and G_['F3']
        h = hg.get(arm, {})
        hok = all(v not in (False, 'FAIL') for v in h.values())
        return ('SUPPORTED' if ok and hok else 'CONDITIONALLY SUPPORTED' if G_['F2_sharp'] and G_['F1'] else 'NOT SUPPORTED'), ok, hok
    V['C3UR_PHYSICAL_FIDELITY'] = fid('C3UR')[0]
    V['R0_PHYSICAL_FIDELITY'] = fid('R0')[0]
    V['R1_PHYSICAL_FIDELITY'] = fid('R1')[0]
    pf = rr['C3UR'].get('pass_frac_all_regions', 0)
    V['C3UR_SAMPLING_RELATION_QUALITY'] = ('SUPPORTED' if imp and not wor and pf >= VM['realization_pass_frac_supported'] else
                                          'CONDITIONALLY SUPPORTED' if imp and not wor else 'NOT SUPPORTED')
    si = s2['source_independence']; ev = s2['edit_validity']
    src_gate = all(v for k, v in hg.get('C3UR', {}).items() if 'LMIX' in k or 'LSPOT' in k)
    V['SOURCE_INDEPENDENT_REUSE'] = ('SUPPORTED' if si['C3UR_construction_no_source'] and si['digest_constant_across_sources'] and src_gate else
                                     'CONDITIONALLY SUPPORTED' if si['C3UR_construction_no_source'] and si['digest_constant_across_sources'] else 'NOT SUPPORTED')
    w2ok = all(v not in (False, 'FAIL') for k, v in hg.get('C3UR', {}).items() if k.startswith('W2'))
    V['DYNAMIC_EDIT_VALIDITY'] = ('SUPPORTED' if ev['C3UR']['PASS'] and (ev['econ_selective_equals_full'] or {}).get('C3UR') and g['C3UR']['F3'] and w2ok else
                                  'CONDITIONALLY SUPPORTED' if ev['C3UR']['PASS'] and (g['C3UR']['F3'] or w2ok) else 'NOT SUPPORTED')
    b = s3['bottleneck']
    r_rel = b['C3UR']['relation_update_s'] / max(b['R0']['relation_update_s'], 1e-12)
    r_app = b['C3UR']['application_s'] / max(b['R0']['application_s'], 1e-12)
    V['RELATION_CONSTRUCTION_ECONOMICS'] = {'C3UR_over_R0': r_rel, 'verdict': 'SUPPORTED' if r_rel <= VM['cost_ratio_supported'] else
                                            'CONDITIONALLY SUPPORTED' if r_rel <= VM['cost_ratio_conditional'] else 'NOT SUPPORTED'}
    V['OPERATOR_APPLICATION_ECONOMICS'] = {'C3UR_over_R0': r_app, 'verdict': 'SUPPORTED' if r_app <= VM['cost_ratio_supported'] else
                                           'CONDITIONALLY SUPPORTED' if r_app <= VM['cost_ratio_conditional'] else 'NOT SUPPORTED'}
    sp = s3['speedups'].get('C3UR_over_R0', {})
    sharp_keys = [k for k in sp if 'C_ss' in k or 'R_mirror' in k]
    win = [k for k in sharp_keys if (isinstance(sp[k], float) and sp[k] >= VM['meaningful_advantage']) or sp[k] == 'R0 does not reach']
    loss = [k for k in sharp_keys if (isinstance(sp[k], float) and sp[k] <= 1 / VM['meaningful_advantage']) or sp[k] == 'C3UR does not reach']
    eligible = V['C3UR_PHYSICAL_FIDELITY'] != 'NOT SUPPORTED' and V['SOURCE_INDEPENDENT_REUSE'] != 'NOT SUPPORTED'
    V['QUALITY_MATCHED_ADVANTAGE'] = {'wins': win, 'losses': loss, 'verdict': ('NOT SUPPORTED (ineligible)' if not eligible else
                                      'SUPPORTED' if win and not loss else 'CONDITIONALLY SUPPORTED' if win else 'NOT SUPPORTED')}
    scg = s3['scaling'].get('G_only', {}).get('exp', {})
    e_ = max(scg.get('C3UR_build_s', 9), scg.get('C3UR_update_s', 9))
    V['SCALING'] = {'C3UR_exponents': {k: v for k, v in scg.items() if k.startswith('C3UR')},
                    'verdict': 'SUPPORTED' if e_ <= VM['scaling_exp_supported'] else 'CONDITIONALLY SUPPORTED' if e_ <= VM['scaling_exp_conditional'] else 'NOT SUPPORTED'}
    h1 = (not stop) and eligible and ((bool(imp) and not wor) or bool(win)) and not loss
    h2 = (not stop) and bool(imp) and not win and bool(loss) and b['C3UR']['relation_update_s'] > b['R0']['relation_update_s']
    h0 = (not stop) and not imp and not win
    V['H1'] = 'SUPPORTED' if h1 else 'NOT SUPPORTED'
    V['H2'] = 'SUPPORTED' if h2 else 'NOT SUPPORTED'
    V['H0'] = 'SUPPORTED' if h0 else 'NOT SUPPORTED'
    cause = []
    if not h1:
        cv = s1['coverage'].get('W1_T1', {})
        if cv and cv.get('C3UR_imp_missed_energy', {}).get('mean', 0) >= cv.get('R0_imp_missed_energy', {}).get('mean', 1):
            cause.append('relation coverage')
        lim = s1['limit_attribution']
        if any(not v.get('C3UR_INF', True) and not v.get('ORACLE', True) for v in lim.values()):
            cause.append('representation (output)')
        elif any(not v.get('C3UR_INF', True) for v in lim.values()):
            cause.append('representation (relation discretisation)')
        if any(v.get('C3UR_INF', False) and not v.get('C3UR', True) for v in lim.values()):
            cause.append('sampling')
        if not ev['C3UR']['PASS'] or not w2ok:
            cause.append('invalidation / edit response')
        if h2 or V['RELATION_CONSTRUCTION_ECONOMICS']['verdict'] == 'NOT SUPPORTED':
            cause.append('economics')
        if not cause:
            cause.append('no measured advantage (fidelity and cost within noise / threshold of R0)')
    V['FAILURE_ATTRIBUTION'] = cause if not h1 else []
    jd({'verdicts': V, 'w_choice': jl(os.path.join(R, 'results', 'w_choice.json'))['chosen']}, os.path.join(R, 'results', 'final89.json'))
    print('VERDICT', json.dumps(V), flush=True)


if __name__ == '__main__':
    cmd = sys.argv[1]
    if cmd == 'wqualify':
        wqualify(sys.argv[2], sys.argv[3])
    else:
        {'stage1': stage1, 'stage2': stage2, 'stage3': stage3, 'verdict': verdict}[cmd](sys.argv[2])
