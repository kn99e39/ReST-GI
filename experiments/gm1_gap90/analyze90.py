"""WL90 analysis by the prospective rules of protocol 'analysis'. numpy only.  python analyze90.py <R90> <R89>  -> results/final90.json

E_H(X) = sum |hl_X - h_REF| / sum tot_REF, signed S(X) = sum (hl_X - h_REF) / sum tot_REF (WL86), against the WL89 REF-ER.
Arm-to-arm differences use the common render seeds: noise of a difference = half |difference(seed 0) - difference(seed 1)|.
"""
import glob, json, math, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for d in ('ecro86', 'gm1_relation88', 'gm1_relation89'):
    sys.path.insert(0, os.path.join(HERE, '..', d))
P90 = json.load(open(os.path.join(HERE, 'protocol', 'gm1_gap90.json')))
import analyze86 as A  # noqa: E402
import analyze88 as A8  # noqa: E402
import analyze89 as A9  # noqa: E402
import scene86 as S  # noqa: E402
NB = S.NB
jd, jl = A8.jd, A8.jl
REG = ['R_mirror', 'C_ss', 'stationary_all', 'D', 'A', 'obj_panel', 'obj_floor', 'obj_walls']
DEVM = bool(os.environ.get('WL90_DEV'))


def hl2(R, tag):
    fns = sorted(glob.glob(os.path.join(R, 'results', 'renders', tag + '_r*.npz')))
    return np.stack([np.load(f)['hl'].astype(np.float64) for f in fns]) if fns else None


def stats(ref, hl, M):
    """Per region: E_H, signed (mean of the two render seeds) and the per-seed values (for common-random-number differences)."""
    h = ref['h'].mean(0); tot = ref['tot'].mean(0)
    o = {}
    for k in REG:
        if k not in M or M[k].sum() < 30:
            continue
        m = M[k]; den = tot[m].sum()
        eh = [float(np.abs(x[m] - h[m]).sum() / den) for x in hl]
        sg = [float((x[m] - h[m]).sum() / den) for x in hl]
        ehm = float(np.abs(hl.mean(0)[m] - h[m]).sum() / den)
        o[k] = {'E_H': ehm, 'signed': float(np.mean(sg)), 'E_H_seeds': eh, 'signed_seeds': sg}
    return o


def diff(Sx, a, b, k, key='E_H'):
    """a - b with CRN noise (per-seed differences)."""
    if a not in Sx or b not in Sx or k not in Sx[a] or k not in Sx[b]:
        return None
    ks = key + '_seeds'
    d = [x - y for x, y in zip(Sx[a][k][ks], Sx[b][k][ks])]
    val = Sx[a][k][key] - Sx[b][k][key]
    return {'value': float(val), 'noise': float(0.5 * abs(d[0] - d[1])) if len(d) == 2 else None}


def field_orders(R, R89, cfg, G):
    """Per-order field-level comparison against the path-traced order fields (orders 2..5 of the table = gathers of events 1..4)."""
    fn = os.path.join(R89, 'results', 'fields', f'ORD_{cfg}.npz')
    if not os.path.exists(fn):
        fn = os.path.join(R, 'results', 'fields', f'ORD_{cfg}.npz')
    oo = np.load(fn)
    a, b = oo['a'].astype(np.float64), oo['b'].astype(np.float64)
    T = 0.5 * (a + b); Tn = 0.5 * np.abs(a - b)
    xz = np.load(os.path.join(R, 'results', 'tables', f'X_{cfg}.npz'))
    out = {}
    for nm in ('O_M0', 'O_axsx', 'O_a1s1', 'O_a1sx', 'O_axs1'):
        out[nm] = [{c: v for c, v in A8.field_cmp(xz[nm][j].astype(np.float64), T[j + 1], Tn[j + 1], G).items() if c in ('all', 'panel', 'G_or_boxA', 'floor', 'walls', 'sphere')}
                   for j in range(4)]
    out['q_vs_order1'] = A8.field_cmp(xz['q'].astype(np.float64), T[0], Tn[0], G)['all']
    return out


def main():
    R, R89 = sys.argv[1], sys.argv[2]
    AN = P90['analysis']
    out = {'configs': {}}
    for cfg in P90['configs']:
        if not os.path.exists(os.path.join(R, 'results', 'tables', f'X_{cfg}.npz')):
            continue
        wname = cfg.split('_')[0]
        M = A9.masks(R89, wname)
        ref = A.load(R89, f'REF_ER_{cfg}')
        Sx = {}
        for f in sorted(glob.glob(os.path.join(R, 'results', 'renders', f'X90_*_{cfg}_r0.npz'))):
            arm = os.path.basename(f)[len('X90_'):-len(f'_{cfg}_r0.npz')]
            Sx[arm] = stats(ref, hl2(R, f'X90_{arm}_{cfg}'), M)
        o = {'stats': Sx}
        G = A9.geom_of(R89, cfg) if not DEVM else A9.geom_of(R89, cfg)
        o['field_orders'] = field_orders(R, R89, cfg, G)
        o['tables_meta'] = jl(os.path.join(R, 'results', 'tables', f'X_{cfg}.json'))
        res = {}
        for k in ('R_mirror', 'C_ss', 'stationary_all', 'D'):
            if 'M0' not in Sx or k not in Sx['M0']:
                continue
            r = {}
            for key in ('E_H', 'signed'):
                d = lambda a, b: diff(Sx, a, b, k, key)
                g0 = d('M0', 'ORACLE')
                r[key] = {'gap_M0_minus_ORACLE': g0, 'EE_minus_ORACLE': d('axsx', 'ORACLE'),
                          'joint_J_PP_minus_EE': d('a1s1', 'axsx'), 'angular_A_EA_minus_EE': d('a1sx', 'axsx'), 'spatial_S_SE_minus_EE': d('axs1', 'axsx'),
                          'compounding_M0_minus_PP': d('M0', 'a1s1'), 'P1only_minus_EE': d('P1only', 'axsx'), 'P2plus_minus_EE': d('P2plus', 'axsx'),
                          'EA1_minus_EE': d('EA1', 'axsx'), 'SE1_minus_EE': d('SE1', 'axsx'),
                          'hybrids_minus_ORACLE': {f'M{m}': d(f'M{m}', 'ORACLE') for m in range(5)},
                          'ladder_minus_EE': {nm: d(nm, 'axsx') for nm in ('a1sx', 'a4sx', 'a16sx', 'axs1', 'axs4', 'axs16', 'a1s1', 'a4s4', 'a16s16')}}
            e = r['E_H']
            g0 = e['gap_M0_minus_ORACLE']
            if g0 is None:
                continue
            G0 = g0['value']
            rules = {'gap': G0, 'gap_noise': g0['noise'], 'gap_exceeds_2x_noise': bool(G0 > 2 * (g0['noise'] or 0))}
            ee = e['EE_minus_ORACLE']['value']
            rules['isolation'] = 'ISOLATED' if ee <= 0.25 * G0 else 'PARTIAL' if ee <= 0.5 * G0 else 'NOT_ISOLATED'
            rules['gap_closed_by_no_projection'] = 1 - ee / G0 if G0 > 0 else None
            J = e['joint_J_PP_minus_EE']['value']; Aa = e['angular_A_EA_minus_EE']['value']; Sp = e['spatial_S_SE_minus_EE']['value']
            rules['J'] = J; rules['A'] = Aa; rules['S'] = Sp; rules['interaction'] = J - Aa - Sp
            rules['factor'] = ('ANGULAR' if J > 0 and Aa >= 2 / 3 * J and Sp <= 1 / 3 * J else 'SPATIAL' if J > 0 and Sp >= 2 / 3 * J and Aa <= 1 / 3 * J else
                               'BOTH_OR_INTERACTING' if J > 0 else 'NO_JOINT_LOSS')
            hy = e['hybrids_minus_ORACLE']
            ms = [m for m in range(5) if hy[f'M{m}'] and hy[f'M{m}']['value'] <= 0.25 * G0]
            rules['m_star'] = ms[0] if ms else None
            rules['first_order_share'] = e['P1only_minus_EE']['value'] / J if J > 0 else None
            rules['compounding_share_of_gap'] = e['compounding_M0_minus_PP']['value'] / G0 if G0 > 0 else None
            lad = e['ladder_minus_EE']
            def first(names):
                for nm in names:
                    if lad[nm] and lad[nm]['value'] <= 0.25 * J:
                        return nm
                return None
            rules['ladder_first_within_25pct_of_J'] = {'angular_only': first(['a1sx', 'a4sx', 'a16sx']), 'spatial_only': first(['axs1', 'axs4', 'axs16']),
                                                       'joint': first(['a1s1', 'a4s4', 'a16s16'])}
            r['rules'] = rules
            res[k] = r
        o['regions'] = res
        fo = o['field_orders']['O_M0']
        o['first_field_order_exceeding_2x_noise'] = {c: next((j + 2 for j in range(4) if c in fo[j] and fo[j][c]['rel_l1'] > 2 * fo[j][c]['truth_noise']), None)
                                                     for c in ('panel', 'G_or_boxA', 'floor', 'walls')}
        out['configs'][cfg] = o
    # feasibility probe (deployed + 16 realizations)
    feas = {}
    for cfg in P90['feasibility']['configs']:
        if not glob.glob(os.path.join(R, 'results', 'renders', f'X90F_R0_{cfg}_s*_r0.npz')):
            continue
        M = A9.masks(R89, cfg.split('_')[0]); ref = A.load(R89, f'REF_ER_{cfg}')
        seeds = sorted({os.path.basename(f).split('_s')[1].split('_r')[0] for f in glob.glob(os.path.join(R, 'results', 'renders', f'X90F_R0_{cfg}_s*_r0.npz'))})
        per = {a: {sd: stats(ref, hl2(R, f'X90F_{a}_{cfg}_s{sd}'), M) for sd in seeds} for a in P90['feasibility']['arms']}
        fz = {'deployed': {a: {k: {'E_H': per[a]['86001'][k]['E_H'], 'signed': per[a]['86001'][k]['signed']} for k in ('R_mirror', 'C_ss', 'stationary_all', 'D') if k in per[a]['86001']}
                           for a in per if '86001' in per[a]}}
        rs = [s for s in seeds if s != '86001']
        summ = {}
        for a in per:
            summ[a] = {}
            for k in ('R_mirror', 'C_ss', 'stationary_all', 'D'):
                v = [per[a][s][k]['E_H'] for s in rs if k in per[a][s]]
                sg = [per[a][s][k]['signed'] for s in rs if k in per[a][s]]
                if v:
                    summ[a][k] = {'E_H_mean': float(np.mean(v)), 'E_H_sd': float(np.std(v, ddof=1)) if len(v) > 1 else None, 'E_H_min': float(np.min(v)),
                                  'E_H_max': float(np.max(v)), 'signed_mean': float(np.mean(sg)), 'pass_frac_0.06': float(np.mean(np.array(v) <= 0.06))}
        paired = {}
        for a, b in (('R0X1', 'R0'), ('C3URX1', 'C3UR'), ('C3URX1', 'R0X1')):
            paired[f'{a}_minus_{b}'] = {}
            for k in ('R_mirror', 'C_ss', 'stationary_all', 'D'):
                d = [per[a][s][k]['E_H'] - per[b][s][k]['E_H'] for s in rs if k in per[a][s]]
                if d:
                    paired[f'{a}_minus_{b}'][k] = {'mean': float(np.mean(d)), 'ci95': A8.ci95(d), 'frac_first_better': float(np.mean(np.array(d) < 0))}
        fz['realizations'] = summ; fz['paired'] = paired
        fz['plumbing'] = jl(os.path.join(R, 'results', 'tables', f'F_{cfg}.json'))['plumbing']
        feas[cfg] = fz
    out['feasibility'] = feas
    ecp = os.path.join(R, 'results', 'econ', 'econ90.json')
    if os.path.exists(ecp):
        ec = jl(ecp); em = ec['edit_median']; sm = ec['source_update_median_single_light']
        s3 = jl(os.path.join(R89, 'results', 'stage3_89.json')) if os.path.exists(os.path.join(R89, 'results', 'stage3_89.json')) else None
        geo = s3['lifecycle_per_edit']['C3UR']['A_geometry'] if s3 else None
        cost = {'per_source': {'C3UR': sm['q_s'] + sm['C3UR_apply4_s'], 'C3URX1': sm['C3UR_exactL1_s'] + sm['C3URX1_first_plus_3_s'],
                               'R0': sm['q_s'] + sm['R0_apply4_s'], 'R0X1': sm['R0_exactL1_s'] + sm['R0X1_first_plus_3_s']},
                'per_edit_excluding_geometry': {'C3UR': em['C3UR_update_s'] + em['q_s'] + em['C3UR_apply4_s'],
                                                'C3URX1': em['C3UR_update_s'] + em['C3UR_raybuf_s'] + em['C3UR_exactL1_s'] + em['C3URX1_first_plus_3_s'],
                                                'R0': em['R0_state_update_s'] + em['R0_coeff_s'] + em['q_s'] + em['R0_apply4_s'],
                                                'R0X1': em['R0_state_update_s'] + em['R0_coeff_s'] + em['R0_raybuf_s'] + em['R0_exactL1_s'] + em['R0X1_first_plus_3_s']},
                'geometry_update_s_WL89': geo, 'parts_median': em, 'memory': ec['memory']}
        if geo is not None:
            cost['per_edit_including_geometry'] = {k: v + geo for k, v in cost['per_edit_excluding_geometry'].items()}
            pe = cost['per_edit_including_geometry']
            cost['ratio_per_edit_C3URX1_over_C3UR'] = pe['C3URX1'] / pe['C3UR']
        cost['ratio_per_source_C3URX1_over_C3UR'] = cost['per_source']['C3URX1'] / cost['per_source']['C3UR']
        out['costs'] = cost
    # projected (not measured) storage / work of the fixed counterfactual grid, relative to the present table and C3UR relation
    out['projection_labelled'] = {'note': 'PROJECTION, not a measurement: a table refined by a sub-bins and s sub-patches has a x s times the entries; a sender relation with the same rays per cell has a x s times the rays, storage and update work; table application scales with the landed rays',
                                  'factors': {nm: {'table_entries_x': a * s, 'relation_rays_x': a * s} for nm, a, s in (('a4', 4, 1), ('a16', 16, 1), ('s4', 1, 4), ('s16', 1, 16), ('a4s4', 4, 4), ('a16s16', 16, 16))}}
    # verdicts by the prospective rules (primary: W1 T1 R_mirror)
    pc = P90['primary']['config']; pk = P90['primary']['region']
    V = {}
    tests = jl(os.path.join(R, 'results', 'tests90.json'))
    V['TESTS'] = 'PASS' if tests['ALL_PASS'] else 'FAIL'
    if pc in out['configs'] and pk in out['configs'][pc]['regions']:
        r = out['configs'][pc]['regions'][pk]['rules']
        V['COMMON_GAP_VERIFIED'] = 'SUPPORTED' if r['gap_exceeds_2x_noise'] else 'NOT SUPPORTED (within noise: STOP)'
        V['INTERMEDIATE_PROJECTION_ISOLATED'] = r['isolation']
        V['DOMINANT_FACTOR'] = r['factor']
        V['M_STAR'] = r['m_star']; V['FIRST_ORDER_SHARE'] = r['first_order_share']; V['COMPOUNDING_SHARE_OF_GAP'] = r['compounding_share_of_gap']
        V['LADDER'] = r['ladder_first_within_25pct_of_J']
        V['PRIMARY_HYPOTHESIS'] = ('SUPPORTED' if r['gap_exceeds_2x_noise'] and r['isolation'] == 'ISOLATED' else
                                   'CONDITIONALLY SUPPORTED' if r['gap_exceeds_2x_noise'] and r['isolation'] == 'PARTIAL' else 'NOT SUPPORTED')
        other = {c: {k: out['configs'][c]['regions'][k]['rules']['isolation'] for k in out['configs'][c]['regions']} for c in out['configs']}
        V['ISOLATION_ALL_CONFIGS_REGIONS'] = other
        st = out['configs'][pc]['stats']
        V['OUTPUT_TABLE_SUFFICIENT_SHARP'] = bool(all(st['ORACLE'][k]['E_H'] <= 0.06 for k in ('R_mirror', 'C_ss') if k in st['ORACLE']))
        interp = r['isolation'] in ('ISOLATED', 'PARTIAL') and (r['m_star'] == 1 or (r['first_order_share'] or 0) >= 0.5)
        V['FEASIBILITY_PROBE_INTERPRETABLE'] = bool(interp)
        if interp and pc in feas:
            p = feas[pc]['paired']['C3URX1_minus_C3UR'].get(pk)
            V['X1_MATERIAL'] = bool(p and p['ci95'][1] < 0)
            if 'costs' in out and 'ratio_per_edit_C3URX1_over_C3UR' in out['costs']:
                c = out['costs']
                V['X1_ECONOMICALLY_PLAUSIBLE'] = bool(c['ratio_per_edit_C3URX1_over_C3UR'] <= 1.25 and c['ratio_per_source_C3URX1_over_C3UR'] <= 2.0)
                V['X1_COST_RATIOS'] = {'per_edit': c['ratio_per_edit_C3URX1_over_C3UR'], 'per_source': c['ratio_per_source_C3URX1_over_C3UR']}
    out['verdicts'] = V
    jd(out, os.path.join(R, 'results', 'final90.json'))
    print('VERDICT', json.dumps(V), flush=True)
    for cfg, o in out['configs'].items():
        for k, r in o['regions'].items():
            e = r['E_H']
            print(cfg, k, json.dumps({'G0': round(r['rules']['gap'], 4), 'EE-ORA': round(e['EE_minus_ORACLE']['value'], 4), 'J': round(r['rules']['J'], 4),
                                      'A': round(r['rules']['A'], 4), 'S': round(r['rules']['S'], 4), 'iso': r['rules']['isolation'], 'factor': r['rules']['factor'],
                                      'm*': r['rules']['m_star']}), flush=True)


if __name__ == '__main__':
    main()
