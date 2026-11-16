"""WL97 audit, phase 2, track J5 (adjudication of the P0-4 cost half and the WL96 cost-critic arithmetic).
Recomputes, from stored JSON only (no timing, no GPU), every number the J5 rulings rest on.
Read-only on all inputs; writes results/audit97/J5/j5_check.json.   python j5_check.py
"""
import json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
J = lambda *p: json.load(open(os.path.join(ROOT, *p)))
ms = lambda x: 1e3 * x
out = {}

# ------------------------------------------------------------------ 1. WL96A Stage 3 as recorded
c = J('results', 'ecro_util96a', 'cost_s1.0.json'); A = c['arms']
W = {a: {e: ms(A[a][e]['wall_s']) for e in ('edit', 'source')} for a in A}
P = {a: {e: {k: ms(v) for k, v in A[a][e]['parts'].items()} for e in ('edit', 'source')} for a in A}
o = {'wall_ms': W, 'U_recomputed': {a: {e: W['R0'][e] / W[a][e] for e in ('edit', 'source')} for a in ('ZERO', 'LAM24', 'LAM64')}, 'U_recorded': c['U']}
o['amdahl_from_R0_parts'] = {e: W['R0'][e] / (W['R0'][e] - P['R0'][e]['tail_orders']) for e in ('edit', 'source')}
o['ZERO_edit_gain_ms_vs_tail_removed_ms'] = [W['R0']['edit'] - W['ZERO']['edit'], P['R0']['edit']['tail_orders']]
o['no_frame_wall'] = {e: (W['R0'][e] - P['R0'][e]['render']) / (W['R0'][e] - P['R0'][e]['render'] - P['R0'][e]['tail_orders']) for e in ('edit', 'source')}
o['kernel_over_wall_R0'] = {e: A['R0'][e]['gpu_kernel_ms'] / W['R0'][e] for e in ('edit', 'source')}
# LAM edit deficit decomposition (conflict 2)
dec = {}
for a in ('LAM24', 'LAM64'):
    d = W[a]['edit'] - W['R0']['edit']
    qinf = P[a]['edit']['source_q_table'] - P['R0']['edit']['source_q_table']
    gen = P[a]['edit']['generator']; lam = P[a]['edit']['lambda_apply']; tail = P['R0']['edit']['tail_orders']
    other = d - (qinf + gen + lam - tail)
    dec[a] = {'deficit_ms': d, 'q_table_inflation_ms': qinf, 'generator_in_event_ms': gen, 'lambda_ms': lam, 'tail_saved_ms': -tail, 'other_parts_ms': other,
              'q_inflation_share_of_deficit': qinf / d, 'cpu_over_wall': A[a]['edit']['cpu_s'] / A[a]['edit']['wall_s'],
              'q_table_in_source_event_ms': P[a]['source']['source_q_table'],
              'U_edit_recorded': W['R0']['edit'] / W[a]['edit'],
              'U_edit_minus_q_inflation': W['R0']['edit'] / (W[a]['edit'] - qinf),
              'U_edit_minus_q_inflation_generator_at_isolated_5.3': W['R0']['edit'] / (W[a]['edit'] - qinf - gen + ms(c['components']['generator_factors_s']))}
o['LAM_edit_decomposition'] = dec
o['R0_cpu_over_wall_edit'] = A['R0']['edit']['cpu_s'] / A['R0']['edit']['wall_s']
o['components_ms'] = {k: ms(v) for k, v in c['components'].items()}
out['1_WL96A_recorded'] = o

# ------------------------------------------------------------------ 2. E5 in-event generator controls (raw run files)
g = {}
for run in ('run1_s1.0', 'run3_s1.0', 'run7_passive_s1.0', 'run2_passive_s1.0'):
    r = J('results', 'audit97', 'E5', run + '.json'); ev = r['events']
    def med(block, key):
        s = ev[block]['summary']
        return s[key]['wall_ms']['med'] if key in s else None
    def cpu(block, key):
        s = ev[block]['summary']
        return s[key]['cpu_ms']['med'] if key in s else None
    row = {}
    r0_first = med('A_orig_order', 'R0|edit|TOTAL')
    r0_after = med('D_R0_after_LAM', 'R0|edit|TOTAL') if 'D_R0_after_LAM' in ev else None
    for blk in ev:
        s = ev[blk]['summary']
        if 'LAM24|edit|TOTAL' not in s:
            continue
        ref = r0_first if blk == 'A_orig_order' or r0_after is None else r0_after
        row[blk] = {'LAM24_edit_ms': med(blk, 'LAM24|edit|TOTAL'), 'R0_edit_ms': ref, 'U': ref / med(blk, 'LAM24|edit|TOTAL'),
                    'q_table_ms': med(blk, 'LAM24|edit|source_q_table'), 'generator_ms': med(blk, 'LAM24|edit|generator'),
                    'event_cpu_ms': cpu(blk, 'LAM24|edit|TOTAL'), 'gen_mode': ev[blk].get('gen_mode'), 'threads': ev[blk].get('threads')}
    row['R0_q_table_ms'] = med('A_orig_order', 'R0|edit|source_q_table'); row['R0_tail_ms'] = med('A_orig_order', 'R0|edit|tail_orders')
    row['env'] = {k: r['env'].get(k) for k in ('torch_threads', 'env')}
    g[run] = row
out['2_E5_generator_controls_raw'] = g

# ------------------------------------------------------------------ 3. WL96 critic Amdahl table (life91.json, C3URX1)
L = J('results', 'gm1_c3urx1_91', 'results', 'econ', 'life91.json')['arms']
X = L['C3URX1']; e, s = X['edit'], X['source_single_light']
we, ws = ms(e['wall_s']), ms(s['wall_s']); se, ss = ms(e['parts']['transport_orders']), ms(s['parts']['transport_orders']); re_, rs = ms(e['parts']['render']), ms(s['parts']['render'])
t = {'edit_wall': we, 'source_wall': ws, 'sweeps_edit': se, 'sweeps_source': ss, 'edit_no_render': we - re_, 'source_no_render': ws - rs,
     'edit_to_frame': we / (we - se), 'edit_no_render_ratio': (we - re_) / (we - re_ - se),
     'source_to_frame_critic_(edit_sweeps)': ws / (ws - se), 'source_to_frame_own_sweeps': ws / (ws - ss),
     'source_no_render_critic_(edit_sweeps)': (ws - rs) / (ws - rs - se), 'source_no_render_own_sweeps': (ws - rs) / (ws - rs - ss)}
# 'generator about 11 ms instead of 5.6-11.6 ms of sweeps => ratio <= 1'
gen11 = 11.0
t['per_edit_ratio_generator_11ms'] = {arm: ms(L[arm]['edit']['wall_s']) / (ms(L[arm]['edit']['wall_s']) - ms(L[arm]['edit']['parts']['transport_orders']) + gen11) for arm in ('R0', 'C3UR', 'C3URX1')}
t['per_edit_ratio_generator_5.3ms'] = {arm: ms(L[arm]['edit']['wall_s']) / (ms(L[arm]['edit']['wall_s']) - ms(L[arm]['edit']['parts']['transport_orders']) + 5.3) for arm in ('R0', 'C3UR', 'C3URX1')}
t['sweeps_ms_by_arm_edit'] = {arm: ms(L[arm]['edit']['parts']['transport_orders']) for arm in ('R0', 'C3UR', 'C3URX1')}
t['all_sweeps_ceiling_by_arm'] = {arm: {'edit': ms(L[arm]['edit']['wall_s']) / (ms(L[arm]['edit']['wall_s']) - ms(L[arm]['edit']['parts']['transport_orders'])),
                                         'source': ms(L[arm]['source_single_light']['wall_s']) / (ms(L[arm]['source_single_light']['wall_s']) - ms(L[arm]['source_single_light']['parts']['transport_orders']))} for arm in ('R0', 'C3UR', 'C3URX1')}
out['3_WL96_critic_amdahl'] = t

# ------------------------------------------------------------------ 4. scale91 (WL96A line 194; conflict 1)
rows = sorted([r for r in J('results', 'gm1_c3urx1_91', 'results', 'econ', 'scale91.json')['rows'] if r['kappa_s'] == 8], key=lambda r: r['N'])
gr = lambda k: rows[-1][k] / rows[0][k]
sc = {'N': [r['N'] for r in rows], 'growth_1270_to_5080': {k: gr(k) for k in ('q_table_s', 'apply_one_order_s', 'C3UR_table_J4_s', 'C3UR_build_s', 'C3URX1_update_s', 'X1_source_eval_s')},
      'abs_increase_ms': {k: ms(rows[-1][k] - rows[0][k]) for k in ('q_table_s', 'apply_one_order_s', 'C3UR_table_J4_s', 'C3UR_build_s')}}
F = 54.5   # WL91 frame wall, ms (constant in N)
sc['source_ceiling_remove_3_of_4_orders'] = {str(r['N']): {'with_frame_%.1fms' % F: (F + ms(r['q_table_s']) + 4 * ms(r['apply_one_order_s'])) / (F + ms(r['q_table_s']) + ms(r['apply_one_order_s'])),
                                                             'no_frame': (ms(r['q_table_s']) + 4 * ms(r['apply_one_order_s'])) / (ms(r['q_table_s']) + ms(r['apply_one_order_s'])),
                                                             'tail_share_of_table_work': 3 * ms(r['apply_one_order_s']) / (ms(r['q_table_s']) + 4 * ms(r['apply_one_order_s']))} for r in rows}
allr = J('results', 'gm1_c3urx1_91', 'results', 'econ', 'scale91.json')['rows']
sc['X1_source_eval_ms_vs_landed_rays'] = sorted([[r['landed_rays'], ms(r['X1_source_eval_s'])] for r in allr])
lr = np.array([r['landed_rays'] for r in allr], float); y = np.array([ms(r['X1_source_eval_s']) for r in allr])
sl, ic = np.polyfit(lr, y, 1); sc['X1_fit'] = {'slope_ns_per_ray': sl * 1e6, 'intercept_ms': ic, 'corr': float(np.corrcoef(lr, y)[0, 1]),
                                               'critic_linear_rate_ms_for_5.5M': 5.5e6 * 11.5 / 1.05e6, 'fit_ms_for_5.5M': sl * 5.5e6 + ic}
out['4_scale91'] = sc

# E5 own refinement runs (CK2, R0 XOp): event medians straight from the raw run files
e5 = {}
for run in ('run1_s1.0', 'run4_s1.5', 'run5_s2.0', 'run6_s3.0'):
    r = J('results', 'audit97', 'E5', run + '.json'); ev = r['events']
    row = {'N': r['N']}
    blk = 'B_interleaved_R0_ZERO' if 'B_interleaved_R0_ZERO' in ev else 'A_orig_order'
    row['block'] = blk
    S = ev[blk]['summary']
    for e_ in ('edit', 'source'):
        r0 = S['R0|%s|TOTAL' % e_]['wall_ms']['med']; z = S['ZERO|%s|TOTAL' % e_]['wall_ms']['med']; tl = S['R0|%s|tail_orders' % e_]['wall_ms']['med']
        q = S['R0|%s|source_q_table' % e_]['wall_ms']['med']; rd = S['R0|%s|render' % e_]['wall_ms']['med']
        row[e_] = {'R0_ms': r0, 'ZERO_ms': z, 'U_measured': r0 / z, 'U_amdahl': r0 / (r0 - tl), 'tail_ms': tl, 'q_ms': q, 'render_ms': rd,
                   'U_amdahl_no_frame': (r0 - rd) / (r0 - rd - tl)}
        if e_ == 'edit':
            row[e_]['geometry_plus_relation_ms'] = sum(S['R0|edit|%s' % k]['wall_ms']['med'] for k in ('geometry_update', 'relation_update', 'relation_coefficients'))
    e5[run] = row
out['4b_E5_refinement_raw'] = e5

# ------------------------------------------------------------------ 5. conventions on one consistent basis (conflict 3)
sup = J('results', 'audit97', 'E5', 'e5_summary.json')
r1 = J('results', 'audit97', 'E5', 'run1_s1.0.json')
comp = r1['components']; tp = r1['torch_profiler']; kho = r1['components_kh_off']
cm = lambda k: comp[k]['wall_ms']['med']
kx = lambda k: comp[k]['exec_ms']['med']
conv = {'inputs_ms': {'q_wall': cm('q_table'), 'q_kernel': kx('q_table'), 'one_order_wall': cm('one_order'), 'one_order_gpu': tp['one_order']['gpu_ms_per_call'],
                      'tail_wall': cm('tail_3_orders'), 'tail_gpu': tp['tail_3_orders']['gpu_ms_per_call'], 'handover_wall': cm('table_handover')}}
i = conv['inputs_ms']
frame_wall = P['R0']['source']['render']; frame_kernel = e5['run1_s1.0']['source']['render_ms']  # placeholder, replaced below
S1 = r1['events']['A_orig_order']['summary']
frame_wall = S1['R0|source|render']['wall_ms']['med']; frame_kernel = S1['R0|source|render']['exec_ms']['med']
conv['inputs_ms'].update(frame_wall_512x64=frame_wall, frame_kernel_512x64=frame_kernel)
# in-event medians for the wall basis (the in-event prefix part includes the q upload); kernel basis adds the profiled q upload
ev_ = lambda k: S1['R0|source|%s' % k]['wall_ms']['med']
i.update(q_wall=ev_('source_q_table'), one_order_wall=ev_('prefix_order'), tail_wall=ev_('tail_orders'), handover_wall=ev_('table_handover'),
         q_kernel=S1['R0|source|source_q_table']['exec_ms']['med'], one_order_gpu=tp['one_order']['gpu_ms_per_call'] + tp['q_upload']['gpu_ms_per_call'])
def U(fixed, pre, tail):
    return (fixed + pre + tail) / (fixed + pre)
conv['resolvent_only_source'] = {
    'wall_frame64': U(i['q_wall'] + i['handover_wall'] + frame_wall, i['one_order_wall'], i['tail_wall']),
    'wall_no_frame': U(i['q_wall'] + i['handover_wall'], i['one_order_wall'], i['tail_wall']),
    'kernel_frame64_consistent': U(i['q_kernel'] + frame_kernel, i['one_order_gpu'], i['tail_gpu']),
    'kernel_no_frame_consistent': U(i['q_kernel'], i['one_order_gpu'], i['tail_gpu']),
    'mixed_basis_D03_style_frame64_(tail_at_wall)': U(i['q_kernel'] + frame_kernel, i['one_order_wall'], i['tail_wall']),
    'mixed_basis_D03_style_no_frame_(tail_at_wall)': U(i['q_kernel'], i['one_order_wall'], i['tail_wall']),
    'operator_only_(not_a_lifecycle)': U(0.0, i['one_order_wall'], i['tail_wall'])}
# full learned operator replacing q table + all orders on a source update (D-04 / F0-06; derived, NOT measured, NOT quality matched)
e87 = J('results', 'gm1_explicit87', 'results', 'econ', 'econ87.json')
src_enc = [ms(e87[E]['edit_median']['source_s']) for E in ('E1', 'E2', 'E3')]
app87 = [ms(e87[E]['source_update']['ecro_application_s']) for E in ('E1', 'E2', 'E3')]
R0s = W['R0']['source']; ps = P['R0']['source']
expl = ps['source_q_table'] + ps['prefix_order'] + ps['tail_orders']
conv['full_operator_source_to_frame_wall_derived'] = {
    'explicit_q_plus_4_orders_ms': expl, 'ecro_source_encoding_ms_WL87': src_enc, 'ecro_apply_isolated_ms_WL96A': ms(c['components']['ecro_apply_s']),
    'ecro_application_incl_geostate_ms_WL87': app87,
    'zero_cost_bound': R0s / (R0s - expl),
    'with_encoding_and_isolated_apply': [R0s / (R0s - expl + s_ + ms(c['components']['ecro_apply_s'])) for s_ in (min(src_enc), max(src_enc))],
    'with_encoding_and_WL87_application': [R0s / (R0s - expl + s_ + a_) for s_, a_ in ((min(src_enc), min(app87)), (max(src_enc), max(app87)))],
    'WL87_source_only_no_frame_EXP_over_ECRO': [(ms(e87[E]['source_update']['exp_q_s']) + ms(e87[E]['source_update']['exp_apply_s'])) / (ms(e87[E]['source_update']['ecro_source_s']) + ms(e87[E]['source_update']['ecro_application_s'])) for E in ('E1', 'E2', 'E3')],
    'WL87_q_wall_ms_vs_q_gpu_ms_edit_median': [[ms(e87[E]['edit_median']['q_s']), e87[E]['edit_median']['q_gpu_ms']] for E in ('E1', 'E2', 'E3')]}
out['5_conventions'] = conv

# ------------------------------------------------------------------ 6. orders sensitivity (E5-6)
one = cm('one_order')          # isolated single order (the in-event prefix part also contains the q upload)
def need(L0):
    k = 1
    while (L0 + k * one) / L0 < 1.25:
        k += 1
    return k
Ls, Le = W['R0']['source'], W['R0']['edit']
out['6_tail_orders_needed_for_1.25_zero_cost'] = {'source_frame': need(Ls - P['R0']['source']['tail_orders']), 'edit_frame': need(Le - P['R0']['edit']['tail_orders']),
                                                 'source_no_frame': need(Ls - P['R0']['source']['render'] - P['R0']['source']['tail_orders']),
                                                 'edit_no_frame': need(Le - P['R0']['edit']['render'] - P['R0']['edit']['tail_orders']), 'one_order_ms': one}

os.makedirs(os.path.join(ROOT, 'results', 'audit97', 'J5'), exist_ok=True)
with open(os.path.join(ROOT, 'results', 'audit97', 'J5', 'j5_check.json'), 'w', newline='\n') as f:
    json.dump(out, f, indent=1)

r3 = lambda v: round(v, 4) if isinstance(v, float) else v
def flat(d, pre=''):
    for k, v in d.items():
        if isinstance(v, dict):
            flat(v, pre + str(k) + '.')
        elif isinstance(v, list):
            print(pre + str(k), [([r3(x) for x in u] if isinstance(u, list) else r3(u)) for u in v])
        else:
            print(pre + str(k), r3(v))
for sec in out:
    print('##', sec); flat(out[sec])
