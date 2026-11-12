"""WL91 analysis by the prospective rules (protocol fidelity_rules / economics / verdict_mapping). numpy only.
  python analyze91.py <R91> <R89> <R90>   -> results/final91.json
"""
import glob, json, math, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for d in ('ecro86', 'gm1_relation88', 'gm1_relation89'):
    sys.path.insert(0, os.path.join(HERE, '..', d))
P91 = json.load(open(os.path.join(HERE, 'protocol', 'gm1_c3urx1_91.json')))
import analyze86 as A  # noqa: E402
import analyze88 as A8  # noqa: E402
import analyze89 as A9  # noqa: E402
import scene86 as S  # noqa: E402
TH = P91['thresholds']
jd, jl, ci95 = A8.jd, A8.jl, A8.ci95
REG = ['stationary_all', 'A', 'C_ss', 'R_mirror', 'D']
ARMS = ['R0', 'C3UR', 'C3URX1']
TAGS = ['W1_T0', 'W1_T1', 'W2_T0', 'W2_T1', 'W1_T1_LMIX89', 'W1_T1_LSPOT89']
HELD = ['W2_T1', 'W1_T1_LMIX89', 'W1_T1_LSPOT89']


def mask_name(tag):
    p = tag.split('_')
    return p[0] + ('_' + p[2] if len(p) > 2 else '')


def hl(R, pat):
    fns = sorted(glob.glob(os.path.join(R, 'results', 'renders', pat + '_r*.npz')))
    return {'hl': np.stack([np.load(f)['hl'].astype(np.float64) for f in fns])} if fns else None


def f3(d, k):
    if k not in d:
        return 'NA'
    v = d[k]
    if not v['resolvable'] or v['slope'] is None:
        return 'INDETERMINATE'
    return 'PASS' if TH['dH_slope'][0] <= v['slope'] <= TH['dH_slope'][1] and v['corr'] >= TH['dH_corr_min'] else 'FAIL'


def gates(T, tag):
    """Applicable gates of one stats table (one draw / arm) of a condition."""
    g = {}
    src = 'LMIX' in tag or 'LSPOT' in tag
    v = T['stationary_all']
    if src:
        g['F4'] = bool(v['E_H'] <= TH['EH_unseen_source_max'])
    else:
        g['F1'] = bool(v['E_H'] <= TH['EH_stationary_max'] and v['capture'] <= TH['capture_max'])
    if tag.endswith('T1') or src:
        for k in (['C_ss', 'R_mirror'] if src else ['A', 'C_ss', 'R_mirror', 'D']):
            if k in T and T[k]['pixels'] >= TH['min_pixels']:
                g[f'F2:{k}'] = bool(T[k]['E_H'] <= TH['EH_region_max'])
    return g


def main():
    R, R89, R90 = sys.argv[1], sys.argv[2], sys.argv[3]
    out = {'conditions': {}}
    refs = {}
    for tag in TAGS:
        if not glob.glob(os.path.join(R, 'results', 'renders', f'X91_C3URX1_{tag}_s86001_r*.npz')):
            continue
        M = A9.masks(R89, mask_name(tag))
        ref = A.load(R89, f'REF_ER_{tag}'); refs[tag] = (ref, M)
        seeds = sorted({os.path.basename(f).split('_s')[-1].split('_r')[0] for f in glob.glob(os.path.join(R, 'results', 'renders', f'X91_C3URX1_{tag}_s*_r0.npz'))})
        rs = [s for s in seeds if s != '86001']
        per = {a: {sd: A8.eh(ref, hl(R, f'X91_{a}_{tag}_s{sd}'), M, REG) for sd in seeds} for a in ARMS}
        o = {'pixels': {k: int(M[k].sum()) for k in REG if k in M}, 'deployed': {a: per[a]['86001'] for a in ARMS}}
        for nm, pat in (('R0X1', f'X91_R0X1_{tag}_s86001'), ('ORACLE', f'X91_ORACLE_{tag}'), ('C3UR_POOL', f'X91P_C3UR_POOL_{tag}'), ('C3URX1_POOL', f'X91P_C3URX1_POOL_{tag}')):
            h = hl(R, pat)
            if h is not None:
                o['deployed'][nm] = A8.eh(ref, h, M, REG)
        summ = {}
        for a in ARMS:
            summ[a] = {}
            for k in REG:
                v = [per[a][s][k]['E_H'] for s in rs if k in per[a][s]]
                sg = [per[a][s][k]['signed'] for s in rs if k in per[a][s]]
                if v:
                    summ[a][k] = {'E_H_mean': float(np.mean(v)), 'E_H_sd': float(np.std(v, ddof=1)) if len(v) > 1 else None, 'E_H_min': float(np.min(v)),
                                  'E_H_max': float(np.max(v)), 'signed_mean': float(np.mean(sg)), 'signed_ci95': ci95(sg)}
            gl = [gates(per[a][s], tag) for s in rs]
            summ[a]['gate_pass_frac'] = {g: float(np.mean([x[g] for x in gl])) for g in gl[0]} if gl else {}
            summ[a]['all_gates_pass_frac'] = float(np.mean([all(x.values()) for x in gl])) if gl else None
            summ[a]['deployed_gates'] = gates(per[a]['86001'], tag)
        o['realizations'] = summ
        paired = {}
        for a, b in (('C3URX1', 'C3UR'), ('C3URX1', 'R0'), ('C3UR', 'R0')):
            paired[f'{a}_minus_{b}'] = {}
            for k in REG:
                d = [per[a][s][k]['E_H'] - per[b][s][k]['E_H'] for s in rs if k in per[a][s] and k in per[b][s]]
                if d:
                    paired[f'{a}_minus_{b}'][k] = {'mean': float(np.mean(d)), 'ci95': ci95(d), 'frac_first_better': float(np.mean(np.array(d) < 0)), 'n': len(d)}
        o['paired'] = paired
        dep = o['deployed']
        dec = {}
        for k in ('R_mirror', 'C_ss'):
            if k in dep.get('C3UR_POOL', {}) and k in summ['C3UR']:
                dec[k] = {'C3UR_sampling (realization mean - pooled)': summ['C3UR'][k]['E_H_mean'] - dep['C3UR_POOL'][k]['E_H'],
                          'C3URX1_sampling (realization mean - pooled)': summ['C3URX1'][k]['E_H_mean'] - dep['C3URX1_POOL'][k]['E_H'],
                          'first_gather_projection (C3UR_POOL - C3URX1_POOL)': dep['C3UR_POOL'][k]['E_H'] - dep['C3URX1_POOL'][k]['E_H'],
                          'remaining (C3URX1_POOL - ORACLE)': dep['C3URX1_POOL'][k]['E_H'] - dep['ORACLE'][k]['E_H'] if 'ORACLE' in dep and k in dep['ORACLE'] else None,
                          'pixels': int(M[k].sum())}
        o['decomposition'] = dec
        # per-order field level (deployed draw)
        ofn = None
        for root, name in ((R89, f'ORD_{tag}.npz'), (R90, f'ORD_{tag}.npz'), (R, f'ORD_{tag}.npz')):
            if os.path.exists(os.path.join(root, 'results', 'fields', name)):
                ofn = os.path.join(root, 'results', 'fields', name); break
        if ofn:
            oo = np.load(ofn); a_, b_ = oo['a'].astype(np.float64), oo['b'].astype(np.float64)
            Tt = 0.5 * (a_ + b_); Tn = 0.5 * np.abs(a_ - b_)
            G = A9.geom_of(R89, '_'.join(tag.split('_')[:2]))
            tz = np.load(os.path.join(R, 'results', 'tables', f'T_{tag}.npz'))
            o['field_orders'] = {nm: [{c: v for c, v in A8.field_cmp(tz[nm][j].astype(np.float64), Tt[j + 1], Tn[j + 1], G).items() if c in ('all', 'panel', 'G_or_boxA', 'floor', 'walls')}
                                      for j in range(4)] for nm in ('O_C3UR', 'O_C3URX1')}
        o['tables_meta'] = jl(os.path.join(R, 'results', 'tables', f'T_{tag}.json'))
        o['_per'] = per
        out['conditions'][tag] = o
    # edit response (F3): deployed draw and per-realization pass fractions
    dH = {}
    for e in ('W1', 'W2'):
        t0, t1 = f'{e}_T0', f'{e}_T1'
        if t0 not in refs or t1 not in refs:
            continue
        r0, M = refs[t0]; r1, _ = refs[t1]
        seeds = sorted(out['conditions'][t1]['_per']['C3UR'])
        dH[e] = {}
        for a in ARMS:
            dep = A8.dh(r0, r1, hl(R, f'X91_{a}_{t0}_s86001'), hl(R, f'X91_{a}_{t1}_s86001'), M, REG)
            res = {k: [] for k in ('D', 'C_ss', 'R_mirror')}; slopes = {k: [] for k in res}
            for sd in seeds:
                if sd == '86001':
                    continue
                d = A8.dh(r0, r1, hl(R, f'X91_{a}_{t0}_s{sd}'), hl(R, f'X91_{a}_{t1}_s{sd}'), M, REG)
                for k in res:
                    res[k].append(f3(d, k))
                    if k in d and d[k]['slope'] is not None:
                        slopes[k].append(d[k]['slope'])
            dH[e][a] = {'deployed': {k: {'slope': dep[k]['slope'], 'corr': dep[k]['corr'], 'resolvable': dep[k]['resolvable'], 'F3': f3(dep, k)} for k in res if k in dep},
                        'realization_F3': {k: {'pass': int(sum(x == 'PASS' for x in v)), 'fail': int(sum(x == 'FAIL' for x in v)), 'indeterminate': int(sum(x == 'INDETERMINATE' for x in v)),
                                               'slope_mean': float(np.mean(slopes[k])) if slopes[k] else None, 'slope_sd': float(np.std(slopes[k], ddof=1)) if len(slopes[k]) > 1 else None}
                                           for k, v in res.items()}}
    out['dH'] = dH
    for o in out['conditions'].values():
        o.pop('_per')
    # relation-size points (descriptive)
    kp = {}
    if 'W1_T1' in refs:
        ref, M = refs['W1_T1']
        for f in sorted(glob.glob(os.path.join(R, 'results', 'renders', 'X91K_*_W1_T1_r0.npz'))):
            nm = os.path.basename(f)[len('X91K_'):-len('_W1_T1_r0.npz')]
            T = A8.eh(ref, hl(R, f'X91K_{nm}_W1_T1'), M, REG)
            kp[nm] = {k: {'E_H': T[k]['E_H'], 'signed': T[k]['signed']} for k in ('R_mirror', 'C_ss', 'stationary_all') if k in T}
    out['relation_size_points_descriptive'] = kp
    # ---------------------------------------------------------------- held-out rules
    C = out['conditions']
    surv = {}; harm = []
    for tag in HELD:
        if tag not in C:
            continue
        p = C[tag]['paired']['C3URX1_minus_C3UR']
        px = C[tag]['pixels']
        k = 'R_mirror' if px.get('R_mirror', 0) >= TH['min_pixels'] else max((r for r in ('C_ss', 'R_mirror') if px.get(r, 0) >= TH['min_pixels']), key=lambda r: px[r], default=None)
        surv[tag] = {'region': k, 'paired': p.get(k) if k else None, 'survives': bool(k and p.get(k) and p[k]['ci95'][1] < 0),
                     'R_mirror_reported': p.get('R_mirror'), 'R_mirror_pixels': px.get('R_mirror', 0)}
    for tag in C:
        for k, v in C[tag]['paired']['C3URX1_minus_C3UR'].items():
            if v['ci95'][0] > 0:
                harm.append({'condition': tag, 'region': k, **v})
        gx, gc = C[tag]['realizations']['C3URX1']['gate_pass_frac'], C[tag]['realizations']['C3UR']['gate_pass_frac']
        for g in gx:
            if gc.get(g, 0) >= 0.9 and gx[g] < 0.9:
                harm.append({'condition': tag, 'gate': g, 'C3UR': gc[g], 'C3URX1': gx[g]})
    hold_gates = {tag: C[tag]['realizations']['C3URX1']['all_gates_pass_frac'] for tag in HELD if tag in C}
    f3x = {e: {k: dH[e]['C3URX1']['deployed'][k]['F3'] for k in dH[e]['C3URX1']['deployed']} for e in dH}
    V = {}
    tests = jl(os.path.join(R, 'results', 'tests91.json'))
    V['CANDIDATE_IMPLEMENTATION'] = 'QUALIFIED' if tests['ALL_PASS'] else 'NOT QUALIFIED'
    w1 = C.get('W1_T1', {}).get('paired', {}).get('C3URX1_minus_C3UR', {}).get('R_mirror')
    V['W1_REPRODUCTION'] = 'REPRODUCED' if w1 and w1['ci95'][1] < 0 and tests['T_WL90_MECHANISM']['PASS'] else 'NOT REPRODUCED'
    nsurv = sum(v['survives'] for v in surv.values())
    g_ok = all((v or 0) >= 0.9 for v in hold_gates.values()) and all(x != 'FAIL' for x in f3x.get('W2', {}).values())
    V['HELDOUT_MECHANISM_VALIDITY'] = ('SUPPORTED' if not harm and g_ok and nsurv >= 2 else 'CONDITIONALLY SUPPORTED' if not harm and g_ok and nsurv == 1 else 'NOT SUPPORTED')
    out['heldout'] = {'advantage_survives': surv, 'n_survive': nsurv, 'harm': harm, 'heldout_gate_pass_frac_C3URX1': hold_gates, 'F3_C3URX1_deployed': f3x}
    # ---------------------------------------------------------------- economics
    ep = os.path.join(R, 'results', 'econ')
    if os.path.exists(os.path.join(ep, 'life91.json')) and os.path.exists(os.path.join(ep, 'ladder91.json')):
        life = jl(os.path.join(ep, 'life91.json')); lad = jl(os.path.join(ep, 'ladder91.json'))
        Lf = life['arms']
        E = {'lifecycle_ms': {a: {'edit_total': Lf[a]['edit']['wall_s'] * 1e3, 'edit_without_render': Lf[a]['edit']['wall_minus_render_s'] * 1e3,
                                  'source_total': Lf[a]['source_single_light']['wall_s'] * 1e3, 'source_without_render': Lf[a]['source_single_light']['wall_minus_render_s'] * 1e3,
                                  'source_two_lights_without_render': Lf[a]['source_two_lights']['wall_minus_render_s'] * 1e3 if Lf[a].get('source_two_lights') else None,
                                  'frame_64spp': Lf[a]['frame']['wall_s'] * 1e3, 'edit_cpu_ms': Lf[a]['edit']['cpu_s'] * 1e3, 'edit_gpu_kernel_ms': Lf[a]['edit']['gpu_kernel_ms'],
                                  'source_cpu_ms': Lf[a]['source_single_light']['cpu_s'] * 1e3, 'source_gpu_kernel_ms': Lf[a]['source_single_light']['gpu_kernel_ms'],
                                  'edit_parts_ms': {k: v * 1e3 for k, v in Lf[a]['edit']['parts'].items()}, 'source_parts_ms': {k: v * 1e3 for k, v in Lf[a]['source_single_light']['parts'].items()},
                                  'parts_sum_over_wall': Lf[a]['edit']['parts_sum_over_wall'], 'amortisation_measured_s': Lf[a]['amortisation_sequence_measured_s'],
                                  'amortisation_predicted_s': Lf[a]['amortisation_sequence_predicted_s'],
                                  'amortisation_raw_s': Lf[a].get('amortisation_sequence_raw_s'), 'amortisation_compile_s': Lf[a].get('amortisation_sequence_compile_s'),
                                  'compilation_excluded': {k: {'events_with_compilation': Lf[a][k]['events_with_compilation'], 'compile_s_total': Lf[a][k]['compile_s_total'],
                                                               'n': Lf[a][k]['n'], 'wall_raw_median_ms': Lf[a][k]['wall_raw_s'] * 1e3}
                                                           for k in ('edit', 'source_single_light', 'frame')},
                                  'torch_peak_mib': Lf[a]['torch_peak_mib'], 'process_gpu_mem': Lf[a]['process_gpu_mem'],
                                  'bytes': Lf[a].get('bytes'), 'rays': Lf[a].get('rays'), 'relation_rays_retraced_per_edit': Lf[a]['relation_rays_retraced_per_edit']} for a in Lf}}
        x, c = E['lifecycle_ms']['C3URX1'], E['lifecycle_ms']['C3UR']
        E['lifecycle_ratio_C3URX1_over_C3UR'] = {k: x[k] / c[k] for k in ('edit_total', 'edit_without_render', 'source_total', 'source_without_render')}
        E['source_edit_validity_e2e'] = {'C3URX1_selective_equals_full': Lf['C3URX1'].get('selective_equals_full'), 'C3URX1_relation_digest_constant': Lf['C3URX1']['relation_digest_constant_across_sources']}
        E['cold'] = {a: jl(os.path.join(ep, f'cold91_{a}.json')) for a in ('C3URX1', 'C3UR', 'R0', 'PT') if os.path.exists(os.path.join(ep, f'cold91_{a}.json'))}
        tiers = P91['economics']['tiers']; mean_ = P91['economics']['meaningful']
        fixed = {a: {'edit': Lf[a]['edit']['wall_minus_render_s'], 'source': Lf[a]['source_single_light']['wall_minus_render_s']} for a in Lf}
        TT = {}
        a_ok = True; b_cfg = {}; c_ok = True; c_detail = {}
        for tag, z in lad['configs'].items():
            lm = z['ladder_median']
            TT[tag] = {}
            draws = sorted({k.split('_s')[1] for k in lm if '_s' in k})
            for key, lvls in tiers.items():
                ek = 'err_' + key
                if ek not in lm['PT'][0]:
                    continue
                for lv in lvls:
                    row = {}
                    for arm in lm:
                        p = A8.interp(lm[arm], lv, ek)
                        base = arm.split('_s')[0]
                        row[arm] = None if p is None else {'spp': p['spp'], 'render_s': p['wall_s'], 'edit_to_frame_s': fixed[base]['edit'] + p['wall_s'], 'source_to_frame_s': fixed[base]['source'] + p['wall_s']}
                    TT[tag][f'{key}@{lv}'] = row
            # rule (a) and (b)
            gain = {}
            for sd in draws:
                g = False
                for tk, row in TT[tag].items():
                    xx, cc = row.get(f'C3URX1_s{sd}'), row.get(f'C3UR_s{sd}')
                    if xx and cc:
                        for path in ('edit_to_frame_s', 'source_to_frame_s'):
                            if xx[path] / cc[path] > mean_:
                                a_ok = False
                    if tk.startswith('R_mirror'):
                        if xx and not cc:
                            g = True
                        elif xx and cc and cc['edit_to_frame_s'] / xx['edit_to_frame_s'] >= mean_:
                            g = True
                gain[sd] = g
            b_cfg[tag] = {'draw_gain': gain, 'config_counts': bool(sum(gain.values()) >= 2)}
            # rule (c): deployed draw vs PT
            cd = {}
            for tk, row in TT[tag].items():
                xx, pt = row.get('C3URX1_s86001'), row.get('PT')
                if xx:
                    cd[tk] = (pt['edit_to_frame_s'] / xx['edit_to_frame_s']) if pt else 'PT does not reach'
                    if pt and pt['edit_to_frame_s'] / xx['edit_to_frame_s'] < mean_:
                        c_ok = False
                else:
                    cd[tk] = 'NOT REACHED'
            c_detail[tag] = cd
        E['time_to_tier'] = TT
        E['rule_a_no_slowdown'] = a_ok; E['rule_b_sharp_gain'] = b_cfg; E['rule_c_vs_PT'] = {'ok': c_ok, 'PT_over_C3URX1': c_detail}
        nb = sum(v['config_counts'] for v in b_cfg.values())
        nb_any = sum(any(v['draw_gain'].values()) for v in b_cfg.values())
        V['QUALITY_MATCHED_ECONOMIC_UTILITY'] = ('SUPPORTED' if a_ok and c_ok and nb >= 2 else 'CONDITIONALLY SUPPORTED' if a_ok and c_ok and nb_any >= 1 else 'NOT SUPPORTED')
        lr = E['lifecycle_ratio_C3URX1_over_C3UR']
        V['LIFECYCLE_COST'] = 'SUPPORTED' if max(lr.values()) <= mean_ else 'NOT SUPPORTED'
        e2e = E['source_edit_validity_e2e']
        V['SOURCE_EDIT_VALIDITY'] = ('SUPPORTED' if tests['T_SOURCE_CURRENT']['PASS'] and tests['T_RELATION_SOURCE_INDEPENDENT']['PASS'] and tests['T_EDIT']['PASS']
                                     and e2e['C3URX1_selective_equals_full'] and e2e['C3URX1_relation_digest_constant'] else 'NOT SUPPORTED')
        sp = os.path.join(ep, 'scale91.json')
        if os.path.exists(sp):
            E['scaling'] = jl(sp)['rows']
        out['economics'] = E
    else:
        for k in ('QUALITY_MATCHED_ECONOMIC_UTILITY', 'LIFECYCLE_COST', 'SOURCE_EDIT_VALIDITY'):
            V[k] = 'NOT TESTED'
    need = {'CANDIDATE_IMPLEMENTATION': V['CANDIDATE_IMPLEMENTATION'] == 'QUALIFIED', 'W1_REPRODUCTION': V['W1_REPRODUCTION'] == 'REPRODUCED',
            'HELDOUT_MECHANISM_VALIDITY': V['HELDOUT_MECHANISM_VALIDITY'] == 'SUPPORTED', 'SOURCE_EDIT_VALIDITY': V['SOURCE_EDIT_VALIDITY'] == 'SUPPORTED',
            'LIFECYCLE_COST': V['LIFECYCLE_COST'] == 'SUPPORTED', 'QUALITY_MATCHED_ECONOMIC_UTILITY': V['QUALITY_MATCHED_ECONOMIC_UTILITY'] in ('SUPPORTED', 'CONDITIONALLY SUPPORTED')}
    V['BASELINE_ADOPTION_EVIDENCE'] = {'verdict': 'SUFFICIENT' if all(need.values()) else 'INSUFFICIENT', 'failing': [k for k, v in need.items() if not v],
                                       'note': 'evidence statement only; adoption is the owner decision'}
    V['FINAL_NOVELTY_EVIDENCE'] = 'INSUFFICIENT by scope (not evaluated: no external learned baseline measured, one scene family, point / cone lights)'
    out['verdicts'] = V
    jd(out, os.path.join(R, 'results', 'final91.json'))
    print('VERDICT', json.dumps(V), flush=True)
    for tag in C:
        s = C[tag]['realizations']
        k = 'R_mirror' if 'R_mirror' in s['C3UR'] else 'C_ss'
        print(tag, k, json.dumps({a: [round(s[a][k]['E_H_mean'], 4), round(s[a][k]['E_H_sd'] or 0, 4), s[a]['all_gates_pass_frac']] for a in ARMS}),
              json.dumps(C[tag]['paired']['C3URX1_minus_C3UR'].get(k)), flush=True)


if __name__ == '__main__':
    main()
