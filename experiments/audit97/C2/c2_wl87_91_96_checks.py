"""WL97 audit C2: quoted-number checks for WL87 (learned-vs-explicit region counts), WL90 (P2plus), WL91 lifecycle anchors used by the WL96 Amdahl
estimate, and the WL96A cost ratios.  json only.  Run from the repository root."""
import json
import numpy as np
o = {}
f = json.load(open('results/gm1_explicit87/results/final87.json'))
rows = f['learned_vs_explicit_regions']
d = np.array([[y - r['EXP'] for y in r['ECRO']] for r in rows])
o['wl87'] = {'rows': len(rows), 'rule_better': len(f['ECRO_better']), 'rule_worse': f['ECRO_worse'], 'rule_equal': len(rows) - len(f['ECRO_better']) - len(f['ECRO_worse']),
             'raw_both_seeds_worse': int((d > 0).all(1).sum()), 'raw_both_seeds_better': int((d < 0).all(1).sum()), 'raw_mixed': int(((d > 0).any(1) & (d < 0).any(1)).sum()),
             'raw_both_better_rows': [r['where'] for r, x in zip(rows, d) if (x < 0).all()],
             'mean_diff_ECRO_minus_EXP_per_seed': d.mean(0).tolist(), 'median_diff_per_seed': np.median(d, 0).tolist(),
             'verdict_LEARNED_OPERATOR_ADDITIONAL_UTILITY': f['verdicts']['LEARNED_OPERATOR_ADDITIONAL_UTILITY']}
g = json.load(open('results/gm1_gap90/results/final90.json'))
o['wl90_P2plus_minus_EE_R_mirror'] = {c: g['configs'][c]['regions']['R_mirror']['E_H']['P2plus_minus_EE'] for c in g['configs']}
o['wl90_W1T1_R_mirror'] = {k: g['configs']['W1_T1']['regions']['R_mirror']['E_H'][k] for k in ('gap_M0_minus_ORACLE', 'angular_A_EA_minus_EE', 'spatial_S_SE_minus_EE', 'P1only_minus_EE')}
c = json.load(open('results/gm1_c3urx1_91/results/econ/life91.json'))
am = {}
for arm, v in c['arms'].items():
    if arm == 'PT':
        continue
    e, s = v['edit'], v['source_single_light']
    sw_e, sw_s = e['parts']['transport_orders'], s['parts']['transport_orders']
    am[arm] = {'edit_wall_ms': 1e3 * e['wall_s'], 'edit_minus_render_ms': 1e3 * e['wall_minus_render_s'], 'source_wall_ms': 1e3 * s['wall_s'],
               'source_minus_render_ms': 1e3 * s['wall_minus_render_s'], 'sweeps_edit_ms': 1e3 * sw_e, 'sweeps_source_ms': 1e3 * sw_s,
               'amdahl_edit': e['wall_s'] / (e['wall_s'] - sw_e), 'amdahl_source_own_sweep': s['wall_s'] / (s['wall_s'] - sw_s),
               'amdahl_source_edit_sweep(WL96 arithmetic)': s['wall_s'] / (s['wall_s'] - sw_e),
               'amdahl_edit_no_render': e['wall_minus_render_s'] / (e['wall_minus_render_s'] - sw_e),
               'amdahl_source_no_render_own_sweep': s['wall_minus_render_s'] / (s['wall_minus_render_s'] - sw_s),
               'n_edit': e['n'], 'n_source': s['n'], 'render_share_of_edit': e['parts']['render'] / e['wall_s'], 'gpu_kernel_share_of_edit_wall': e['gpu_kernel_ms'] / (1e3 * e['wall_s'])}
o['wl91_amdahl'] = am
k = json.load(open('results/ecro_util96a/cost_s1.0.json'))
A = k['arms']
o['wl96a_cost'] = {'U_recomputed': {a: {ev: A['R0'][ev]['wall_s'] / A[a][ev]['wall_s'] for ev in ('edit', 'source')} for a in ('LAM24', 'LAM64', 'ZERO')}, 'U_stored': k['U'],
                   'tail_orders_share_of_R0_edit': A['R0']['edit']['parts']['tail_orders'] / A['R0']['edit']['wall_s'],
                   'tail_orders_share_of_R0_source': A['R0']['source']['parts']['tail_orders'] / A['R0']['source']['wall_s'],
                   'q_table_ms_by_arm_edit': {a: 1e3 * A[a]['edit']['parts']['source_q_table'] for a in A},
                   'render_share_R0_edit': A['R0']['edit']['parts']['render'] / A['R0']['edit']['wall_s']}
print(json.dumps(o, indent=1))
json.dump(o, open('results/audit97/C2/wl87_91_96_checks.json', 'w'), indent=1)
