"""J7 decisive check: WL86 F2 founding failure vs its own rule, state-draw resampling (WL87) and held-out E4/E5 (WL92A/WL92B).
Reads stored JSON only (numpy-free). Run from repo root: python experiments/audit97/J7/founding_failure_check.py"""
import json, math
RULE = 0.06
out = {}
f86 = json.load(open('results/ecro86/results/final.json'))
out['WL86_F2_E2_T1_C'] = {s: {'E_H': v['fidelity']['E2']['T1']['C']['E_H'], 'noise_bound': v['fidelity']['E2']['T1']['C']['noise_bound'],
                              'excess_over_rule_in_noise_bounds': (v['fidelity']['E2']['T1']['C']['E_H'] - RULE) / v['fidelity']['E2']['T1']['C']['noise_bound'],
                              'F2_regions': v['gates']['F2_regions']} for s, v in f86['seeds'].items()}
out['WL86_other_F2_cells_max'] = max(v['fidelity'][e]['T1'][r]['E_H'] for v in f86['seeds'].values() for e in ('E1', 'E2') for r in ('A', 'D')) 
s2 = json.load(open('results/gm1_explicit87/results/stage2_87.json'))['per_edit']['E2']
ec, ex = s2['ECRO']['rows'], {r['seed']: r for r in s2['EXP']['rows']}
for t in ('T0', 'T1'):
    a = [r[t]['C']['E_H'] for r in ec]
    d = [r[t]['C']['E_H'] - ex[r['seed']][t]['C']['E_H'] for r in ec]
    m = sum(d) / len(d); sd = math.sqrt(sum((x - m) ** 2 for x in d) / (len(d) - 1))
    out[f'WL87_E2_{t}_C_ecro_s8601_state_draws'] = {'n': len(a), 'draw_seeds': [r['seed'] for r in ec], 'pass_all': sum(x <= RULE for x in a),
        'pass_excl_deployed_86001': sum(x <= RULE for x in a[1:]), 'deployed_86001': a[0], 'mean': sum(a) / len(a), 'min': min(a), 'max': max(a),
        'paired_ecro_minus_exp_mean': m, 'paired_se': sd / math.sqrt(len(d)), 'n_ecro_worse': sum(x > 0 for x in d),
        'exp_pass': sum(ex[k][t]['C']['E_H'] <= RULE for k in ex)}
f92b = json.load(open('results/ecro_cf92b/results/final92b.json'))['e5']
X = f92b['controls']
out['WL92B_E5k995'] = {m: {t: {'C': v['fidelity'][t]['C']['E_H'], 'obj_panel': v['fidelity'][t]['obj_panel']['E_H'],
                               'C_minus_XEXP': v['fidelity'][t]['C']['E_H'] - X[f'X_EXP_{t}']['C']['E_H'], 'nb_C': v['fidelity'][t]['C']['noise_bound']}
                           for t in ('T0', 'T1')} for m, v in f92b['models'].items()}
out['WL92B_E5k995_controls'] = {k: {'C': X[k]['C']['E_H'], 'obj_panel': X[k].get('obj_panel', {}).get('E_H')} for k in X}
f92a = json.load(open('results/ecro_own92a/results/final92a.json'))
out['WL92A_E4_T1_C'] = {m: v['fidelity']['E4']['T1']['C']['E_H'] for m, v in f92a['models'].items()}
out['WL92A_E4_T1_C']['X_EXP'] = f92a['controls']['X_EXP_E4_T1']['C']['E_H']
L = json.load(open('results/gm1_explicit87/results/final87.json'))['learned_vs_explicit_regions']
out['WL87_region_rows'] = {'n': len(L), 'ECRO_better': sum(x['ECRO_better'] for x in L), 'ECRO_worse': sum(x['ECRO_worse'] for x in L)}
json.dump(out, open('results/audit97/J7/founding_failure_check.json', 'w', encoding='utf-8', newline='\n'), indent=1)
for k, v in out.items(): print(k, json.dumps(v)[:260])
