"""WL97 audit track D (resumed run): independent spot-checks of ceilings.json and a few extra conventions, from the recorded JSON only.
  python addendum.py -> results/audit97/D/addendum.json"""
import json, os
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
J = lambda *p: json.load(open(os.path.join(ROOT, *p)))
c = J('results', 'ecro_util96a', 'cost_s1.0.json'); L = J('results', 'gm1_c3urx1_91', 'results', 'econ', 'life91.json')
sc = J('results', 'gm1_c3urx1_91', 'results', 'econ', 'scale91.json'); e87 = J('results', 'gm1_explicit87', 'results', 'econ', 'econ87.json')
ms = lambda x: 1e3 * x
A = c['arms']; R0e, R0s = A['R0']['edit'], A['R0']['source']
o = {}
# 1. spot-check: recorded U and analytic ceilings
o['spot'] = {'U_ZERO_edit': R0e['wall_s'] / A['ZERO']['edit']['wall_s'], 'U_ZERO_source': R0s['wall_s'] / A['ZERO']['source']['wall_s'],
             'U_LAM24_edit': R0e['wall_s'] / A['LAM24']['edit']['wall_s'], 'U_LAM64_edit': R0e['wall_s'] / A['LAM64']['edit']['wall_s'],
             'analytic_edit': R0e['wall_s'] / (R0e['wall_s'] - R0e['parts']['tail_orders']),
             'analytic_source': R0s['wall_s'] / (R0s['wall_s'] - R0s['parts']['tail_orders'])}
X = L['arms']['C3URX1']
for k, ev in (('edit', 'edit'), ('source', 'source_single_light')):
    w = X[ev]['wall_s']; sw = X[ev]['parts']['transport_orders']; rd = X[ev]['parts']['render']
    o['spot']['WL91_C3URX1_%s_to_frame_own_sweeps' % k] = w / (w - sw)
    o['spot']['WL91_C3URX1_%s_no_render_own_sweeps' % k] = (w - rd) / (w - rd - sw)
we = X['edit']['parts']['transport_orders']; ws = X['source_single_light']['wall_s']; rs = X['source_single_light']['parts']['render']
o['spot']['WL91_critic_source_to_frame_edit_sweeps'] = ws / (ws - we)
o['spot']['WL91_critic_source_no_render_edit_sweeps'] = (ws - rs) / (ws - rs - we)
# 2. LAM edit decomposition
for a in ('LAM24', 'LAM64'):
    e = A[a]['edit']; p = e['parts']
    dq = p['source_q_table'] - R0e['parts']['source_q_table']; gen = p['generator']; iso = c['components']['generator_factors_s']
    o[a] = {'deficit_ms': ms(e['wall_s'] - R0e['wall_s']), 'q_inflation_ms': ms(dq), 'generator_in_event_ms': ms(gen), 'generator_isolated_ms': ms(iso),
            'geostate_and_other_in_generator_part_ms': ms(gen - iso), 'artefact_ms': ms(dq + gen - iso),
            'U_recorded': R0e['wall_s'] / e['wall_s'], 'U_minus_artefacts': R0e['wall_s'] / (e['wall_s'] - dq - (gen - iso)),
            'cpu_ms': ms(e['cpu_s']), 'cpu_R0_ms': ms(R0e['cpu_s']),
            'source_event_q_ms': ms(A[a]['source']['parts']['source_q_table']), 'R0_source_event_q_ms': ms(R0s['parts']['source_q_table'])}
# 3. edit lifecycle under work-only conventions (recorded numbers; torch / numpy parts have no kernel timer and stay at synchronised wall)
E2 = e87['E2']['edit_median']
qk = E2['q_gpu_ms']; stk = E2['state_selective_gpu_ms']
frame_kernel = L['arms']['R0']['frame']['gpu_kernel_ms']
geo = ms(R0e['parts']['geometry_update']); coef = ms(R0e['parts']['relation_coefficients']); pre = ms(R0e['parts']['prefix_order']); tail = ms(R0e['parts']['tail_orders'])


def U(fixed):
    return (fixed + pre + tail) / (fixed + pre)


o['edit_conventions'] = {
    'inputs_ms': {'q_kernel_WL87_E2': qk, 'state_update_kernel_WL87_E2': stk, 'frame_kernel_WL91_R0_64spp': frame_kernel, 'geometry_update_wall': geo,
                  'relation_coefficients_wall': coef, 'prefix_wall': pre, 'tail_wall': tail},
    'edit_to_table_kernel_geo_at_wall': U(geo + stk + coef + qk), 'edit_to_table_kernel_geo_excluded': U(stk + coef + qk),
    'edit_to_frame64_kernel_geo_at_wall': U(geo + stk + coef + qk + frame_kernel), 'edit_to_frame64_kernel_geo_excluded': U(stk + coef + qk + frame_kernel)}
# 4. overhead shares of the R0 lifecycles (WL96A): Dr.Jit kernel ms / wall, CPU / wall
o['overhead_WL96A_R0'] = {}
for k in ('edit', 'source'):
    r = A['R0'][k]
    o['overhead_WL96A_R0'][k] = {'wall_ms': ms(r['wall_s']), 'drjit_kernel_ms': r['gpu_kernel_ms'], 'cpu_ms': ms(r['cpu_s']),
                                 'kernel_over_wall': r['gpu_kernel_ms'] / ms(r['wall_s']),
                                 'torch_parts_wall_ms': ms(sum(r['parts'].get(x, 0) for x in ('prefix_order', 'tail_orders'))),
                                 'non_kernel_non_torch_ms': ms(r['wall_s']) - r['gpu_kernel_ms'] - ms(sum(r['parts'].get(x, 0) for x in ('prefix_order', 'tail_orders')))}
# 5. N scaling (scale91, kappa_s = 8 rows): growth ratios and asymptotic source->table ceiling from per-patch slopes
rows = sorted([r for r in sc['rows'] if r['kappa_s'] == 8], key=lambda r: r['N'])
g = lambda k: rows[-1][k] / rows[0][k]
sl = lambda k: (rows[-1][k] - rows[0][k]) / (rows[-1]['N'] - rows[0]['N'])
o['scale91'] = {'N': [r['N'] for r in rows], 'growth_q_table': g('q_table_s'), 'growth_one_order': g('apply_one_order_s'), 'growth_C3UR_J4': g('C3UR_table_J4_s'),
                'growth_relation_build': g('C3UR_build_s'), 'growth_update': g('C3URX1_update_s'),
                'ceiling_source_to_table_slopes_q_plus_4_orders_remove_3': (sl('q_table_s') + 4 * sl('apply_one_order_s')) / (sl('q_table_s') + sl('apply_one_order_s')),
                'ceiling_rows_source_to_frame_q_plus_4_orders_remove_3_frame54ms': {
                    str(r['N']): (r['q_table_s'] + 4 * r['apply_one_order_s'] + 0.0545) / (r['q_table_s'] + r['apply_one_order_s'] + 0.0545) for r in rows}}
# 6. WL96 critic DQ estimate with the audit's dirty-pair counts (dirty_xor.json) at the critic's own throughput (1.05 M rays per 11.5 ms)
dx = J('results', 'audit97', 'D', 'dirty_xor.json')['shifts']['minus0.30y_critic']
thr = 11.5 / 1.05e6
D = {'critic_union_pairs': dx['union'] + dx['front_onA'], 'xor_pairs_plus_moved': dx['xor'] + dx['front_onA'],
     'rays_strictly_needed_old_only_plus_moved': dx['old'] - dx['both'] + dx['front_onA'], 'ECRO_edit_ms': 77.0}
for k, n in (('critic', D['critic_union_pairs']), ('xor', D['xor_pairs_plus_moved']), ('needed', D['rays_strictly_needed_old_only_plus_moved'])):
    D[k + '_ms'] = n * thr; D[k + '_total_over_ECRO'] = (77.0 + n * thr) / 77.0
o['DQ_edit'] = D
json.dump(o, open(os.path.join(ROOT, 'results', 'audit97', 'D', 'addendum.json'), 'w'), indent=1)


def show(d, ind=0):
    for k, v in d.items():
        if isinstance(v, dict):
            print(' ' * ind + str(k) + ':'); show(v, ind + 2)
        else:
            print(' ' * ind + str(k) + ':', round(v, 4) if isinstance(v, float) else v)


show(o)
