"""WL97 audit track D: recomputation of every cost ratio / Amdahl ceiling from the recorded JSON artifacts (pure Python + numpy, no GPU).
Inputs (read-only): results/ecro_util96a/cost_s1.0.json, results/gm1_c3urx1_91/results/econ/{life91,scale91}.json,
results/gm1_explicit87/results/econ/econ87.json, results/ecro86/results/econ/econ86.json.  Output: results/audit97/D/ceilings.json.
Every number below is arithmetic on recorded values; conventions are stated in the keys.  Nothing is re-timed."""
import hashlib, json, os
import numpy as np
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..')
P = lambda *a: os.path.join(ROOT, *a)
FN = {'c96a': P('results', 'ecro_util96a', 'cost_s1.0.json'), 'life91': P('results', 'gm1_c3urx1_91', 'results', 'econ', 'life91.json'),
      'scale91': P('results', 'gm1_c3urx1_91', 'results', 'econ', 'scale91.json'), 'econ87': P('results', 'gm1_explicit87', 'results', 'econ', 'econ87.json'),
      'econ86': P('results', 'ecro86', 'results', 'econ', 'econ86.json')}
J = {k: json.load(open(v)) for k, v in FN.items()}
out = {'inputs_sha256_16': {k: hashlib.sha256(open(v, 'rb').read()).hexdigest()[:16] for k, v in FN.items()}}
ms = lambda s: s * 1e3

# ---------------------------------------------------------------- A. WL96A Stage 3 recomputation
c = J['c96a']; A = c['arms']; comp = {k: ms(v) for k, v in c['components'].items()}
R0e, R0s = A['R0']['edit'], A['R0']['source']
pe = {k: ms(v) for k, v in R0e['parts'].items()}; ps = {k: ms(v) for k, v in R0s['parts'].items()}
We, Ws = ms(R0e['wall_s']), ms(R0s['wall_s'])
a = {'R0_edit_ms': We, 'R0_source_ms': Ws, 'R0_edit_parts_ms': pe, 'R0_source_parts_ms': ps,
     'U_recorded': c['U'],
     'U_recomputed': {arm: {k: A['R0'][k]['wall_s'] / A[arm][k]['wall_s'] for k in ('edit', 'source')} for arm in ('LAM24', 'LAM64', 'ZERO')},
     'tail_share': {'edit': pe['tail_orders'] / We, 'source': ps['tail_orders'] / Ws}}
# analytic Amdahl ceiling from the R0 arm alone (remove the tail part) vs the measured ZERO arm
a['ceiling_analytic_R0_minus_tail'] = {'edit': We / (We - pe['tail_orders']), 'source': Ws / (Ws - ps['tail_orders'])}
a['ZERO_minus_R0_unexplained_ms'] = {k: (ms(A['R0'][k]['wall_s']) - ms(A['ZERO'][k]['wall_s'])) - ms(A['R0'][k]['parts']['tail_orders']) for k in ('edit', 'source')}
a['arm_to_arm_drift_of_common_parts_ms'] = {arm: {p: ms(A[arm]['edit']['parts'][p]) - pe[p] for p in ('geometry_update', 'relation_update', 'relation_coefficients', 'source_q_table', 'prefix_order', 'render')} for arm in ('LAM24', 'LAM64', 'ZERO')}
# without the frame
nf = lambda arm, k: ms(A[arm][k]['wall_s']) - ms(A[arm][k]['parts']['render'])
a['ceiling_no_frame'] = {'measured_ZERO': {k: nf('R0', k) / nf('ZERO', k) for k in ('edit', 'source')},
                         'analytic': {'edit': (We - pe['render']) / (We - pe['render'] - pe['tail_orders']), 'source': (Ws - ps['render']) / (Ws - ps['render'] - ps['tail_orders'])}}
# LAM edit: decomposition of the deficit and corrected U
lam = {}
for arm in ('LAM24', 'LAM64'):
    pl = {k: ms(v) for k, v in A[arm]['edit']['parts'].items()}
    Wl = ms(A[arm]['edit']['wall_s'])
    q_infl = pl['source_q_table'] - pe['source_q_table']
    lam_ap = comp['lambda24_apply_s'] if arm == 'LAM24' else comp['lambda64_apply_s']
    lam[arm] = {'edit_ms': Wl, 'deficit_vs_R0_ms': Wl - We, 'generator_in_lifecycle_ms': pl['generator'], 'generator_isolated_ms': comp['generator_factors_s'],
                'q_table_inflation_ms': q_infl, 'lambda_apply_in_lifecycle_ms': pl['lambda_apply'], 'tail_saved_ms': pe['tail_orders'],
                'cpu_ms': ms(A[arm]['edit']['cpu_s']), 'cpu_extra_vs_R0_ms': ms(A[arm]['edit']['cpu_s'] - R0e['cpu_s']),
                'U_recorded': We / Wl,
                'U_without_q_inflation': We / (Wl - q_infl),
                'U_isolated_generator_no_q_inflation': We / (We - pe['tail_orders'] + comp['generator_factors_s'] + lam_ap),
                'U_break_even_generator_ms': pe['tail_orders'] - lam_ap}
a['LAM_edit'] = lam
a['cpu_over_wall'] = {arm: {k: A[arm][k]['cpu_s'] / A[arm][k]['wall_s'] for k in ('edit', 'source')} for arm in A}
a['drjit_kernel_ms_over_wall'] = {arm: {k: A[arm][k]['gpu_kernel_ms'] / ms(A[arm][k]['wall_s']) for k in ('edit', 'source')} for arm in A}
out['A_WL96A'] = a

# ---------------------------------------------------------------- B. WL96 critic Amdahl table (from WL91 C3URX1 lifecycle)
L = J['life91']['arms']
x = L['C3URX1']
e, s = x['edit'], x['source_single_light']
sw_e, sw_s = ms(e['parts']['transport_orders']), ms(s['parts']['transport_orders'])
b = {'edit_ms': ms(e['wall_s']), 'source_ms': ms(s['wall_s']), 'sweeps_in_edit_ms': sw_e, 'sweeps_in_source_ms': sw_s,
     'edit_to_frame': ms(e['wall_s']) / (ms(e['wall_s']) - sw_e), 'edit_no_render': ms(e['wall_minus_render_s']) / (ms(e['wall_minus_render_s']) - sw_e),
     'source_to_frame_critic_used_edit_sweeps': ms(s['wall_s']) / (ms(s['wall_s']) - sw_e), 'source_to_frame_own_sweeps': ms(s['wall_s']) / (ms(s['wall_s']) - sw_s),
     'source_no_render_critic': ms(s['wall_minus_render_s']) / (ms(s['wall_minus_render_s']) - sw_e), 'source_no_render_own_sweeps': ms(s['wall_minus_render_s']) / (ms(s['wall_minus_render_s']) - sw_s)}
# same table for the R0 arm (receiver gather; the operator WL96A actually timed) and C3UR
for arm in ('R0', 'C3UR'):
    ee, ss = L[arm]['edit'], L[arm]['source_single_light']
    b[arm] = {'edit_to_frame_all_sweeps': ms(ee['wall_s']) / (ms(ee['wall_s']) - ms(ee['parts']['transport_orders'])),
              'source_to_frame_all_sweeps': ms(ss['wall_s']) / (ms(ss['wall_s']) - ms(ss['parts']['transport_orders'])),
              'sweeps_ms': ms(ss['parts']['transport_orders'])}
# "real generator ~11 ms => guaranteed loss": per-edit ratio with the WL96A isolated generator instead
b['edit_ratio_generator_11ms'] = ms(e['wall_s']) / (ms(e['wall_s']) - sw_e + 11.0)
b['edit_ratio_generator_isolated_5.3ms'] = ms(e['wall_s']) / (ms(e['wall_s']) - sw_e + comp['generator_factors_s'])
b['edit_ratio_generator_isolated_keep_1_of_4_orders'] = ms(e['wall_s']) / (ms(e['wall_s']) - 0.75 * sw_e + comp['generator_factors_s'])
b['lifecycle_ratios_C3URX1_over_C3UR'] = {'edit': e['wall_s'] / L['C3UR']['edit']['wall_s'], 'edit_no_render': e['wall_minus_render_s'] / L['C3UR']['edit']['wall_minus_render_s'],
                                          'source': s['wall_s'] / L['C3UR']['source_single_light']['wall_s'], 'source_no_render': s['wall_minus_render_s'] / L['C3UR']['source_single_light']['wall_minus_render_s']}
out['B_WL96_critic_from_WL91'] = b

# ---------------------------------------------------------------- C. overhead profile (wall vs Dr.Jit kernel ms)
E87 = J['econ87']
o = {}
for ed in E87['edits']:
    em = E87[ed]['edit_median']
    lad = {r['spp']: r for r in E87[ed]['ladder_median']['EXP']}
    o[ed] = {'q_wall_ms': ms(em['q_s']), 'q_gpu_kernel_ms': em['q_gpu_ms'], 'q_kernel_share': em['q_gpu_ms'] / ms(em['q_s']),
             'state_selective_wall_ms': ms(em['state_selective_s']), 'state_selective_gpu_kernel_ms': em['state_selective_gpu_ms'],
             'source_enc_wall_ms': ms(em['source_s']), 'source_enc_gpu_kernel_ms': em['source_gpu_ms'],
             'ecro_inference_wall_ms': ms(em['ecro_inference_s']), 'exp_apply_wall_ms': ms(em['exp_apply_s']),
             'table_frame_by_spp': {spp: {'wall_ms': ms(r['wall_s']), 'python_trace_ms': ms(r['trace_s']), 'gpu_kernel_ms': r['gpu_ms'], 'kernel_share': r['gpu_ms'] / ms(r['wall_s'])} for spp, r in lad.items() if spp <= 256},
             'source_only_wall': {'ecro_ms': ms(E87[ed]['source_update']['ecro_source_s'] + E87[ed]['source_update']['ecro_application_s']),
                                  'exp_ms': ms(E87[ed]['source_update']['exp_q_s'] + E87[ed]['source_update']['exp_apply_s']),
                                  'exp_q_ms_2lights_single_unwarmed_call': ms(E87[ed]['source_update']['exp_q_s'])}}
    so = o[ed]['source_only_wall']; so['ratio_ecro_over_exp'] = so['ecro_ms'] / so['exp_ms']
out['C_overhead_WL87'] = o
# source-only update in a work-based convention (GPU kernel / synchronised torch walls of the pure applications)
q_k = float(np.median([o[ed]['q_gpu_kernel_ms'] for ed in o]))
src_k = float(np.median([o[ed]['source_enc_gpu_kernel_ms'] for ed in o]))
out['C_source_only_work_convention'] = {'explicit_ms': q_k + 4 * comp['one_order_s'], 'learned_ms': src_k + comp['ecro_apply_s'],
                                        'ratio_learned_over_explicit': (src_k + comp['ecro_apply_s']) / (q_k + 4 * comp['one_order_s']),
                                        'note': 'q / source at Dr.Jit kernel ms (WL87, one light); 4 explicit orders and ECRO apply at WL96A isolated synchronised walls (resident tensors)'}
# WL91: per-event kernel share, two-light scaling
k91 = {}
for arm in L:
    k91[arm] = {ev: {'wall_ms': ms(L[arm][ev]['wall_s']), 'drjit_kernel_ms': L[arm][ev]['gpu_kernel_ms'], 'kernel_share': L[arm][ev]['gpu_kernel_ms'] / ms(L[arm][ev]['wall_s']),
                     'parts_ms': {p: ms(v) for p, v in L[arm][ev]['parts'].items()}} for ev in ('edit', 'source_single_light', 'source_two_lights', 'frame') if L[arm][ev]}
    rows = L[arm]['edit_rows']; k91[arm]['edit_wall_sd_ms'] = float(np.std([ms(r['wall_s']) for r in rows], ddof=1)); k91[arm]['edit_wall_range_ms'] = [min(ms(r['wall_s']) for r in rows), max(ms(r['wall_s']) for r in rows)]
    rows = [r for r in L[arm]['source_rows'] if r['lights'] == 1]; k91[arm]['source_wall_sd_ms'] = float(np.std([ms(r['wall_s']) for r in rows], ddof=1))
out['C_overhead_WL91'] = k91

# ---------------------------------------------------------------- D. ceilings under alternative, explicitly stated conventions (WL96A R0 parts)
fr = o['E2']['table_frame_by_spp']                                                       # table frame ladder (WL87 E2, same renderer / resolution)
pre, tl, ho = ps['prefix_order'], ps['tail_orders'], ps['table_handover']
conv = {}


def ceil(fixed, name, note):
    conv[name] = {'fixed_ms': fixed, 'prefix_ms': pre, 'tail_ms': tl, 'U_ZERO': (fixed + pre + tl) / (fixed + pre), 'note': note}


ceil(ps['source_q_table'] + ho + ps['render'], 'source_to_frame_wall_as_recorded', 'WL96A parts: q wall + hand-over + 64-spp frame wall')
ceil(ps['source_q_table'] + ho, 'source_to_table_wall_no_frame', 'no frame')
for spp in (4, 16, 256):
    ceil(ps['source_q_table'] + ho + fr[spp]['wall_ms'], f'source_to_frame_wall_{spp}spp', f'frame wall at {spp} spp from the WL87 E2 EXP ladder (512^2, plain)')
ceil(q_k + ho + fr[64]['gpu_kernel_ms'], 'source_to_frame_gpu_kernel_64spp', 'q and frame at Dr.Jit kernel ms (WL87); torch sweeps kept at their synchronised wall (upper bound of their GPU time)')
ceil(q_k + ho + fr[4]['gpu_kernel_ms'], 'source_to_frame_gpu_kernel_4spp', 'as above, 4-spp frame')
ceil(q_k + ho + fr[64]['gpu_kernel_ms'] / 64.0, 'source_to_frame_gpu_kernel_1spp_scaled', 'as above, 1-spp frame (64-spp kernel ms / 64; scaled, not measured)')
ceil(q_k + ho, 'source_to_table_gpu_kernel_no_frame', 'as above, no frame')
ceil(0.0, 'transport_only_prefix_plus_tail', 'only the operator applications (1 prefix + 3 tail orders)')
edit_fixed = pe['geometry_update'] + pe['relation_update'] + pe['relation_coefficients']
ceil(edit_fixed + pe['source_q_table'] + ho + pe['render'], 'edit_to_frame_wall_as_recorded', 'WL96A R0 edit parts')
ceil(edit_fixed + pe['source_q_table'] + ho, 'edit_to_table_wall_no_frame', 'no frame')
for k in conv:
    for rho, la in ((24, comp['lambda24_apply_s']), (64, comp['lambda64_apply_s'])):
        conv[k][f'U_LAM{rho}_no_generator'] = (conv[k]['fixed_ms'] + pre + tl) / (conv[k]['fixed_ms'] + pre + la)
out['D_conventions'] = conv

# ---------------------------------------------------------------- E. sensitivities from recorded numbers
sens = {}
# lights: WL91 R0 source events with 1 and 2 lights (sweeps act on the summed field)
r1, r2 = L['R0']['source_single_light'], L['R0']['source_two_lights']
sens['lights_WL91_R0'] = {'q_ms_1': ms(r1['parts']['source_q_table']), 'q_ms_2': ms(r2['parts']['source_q_table']), 'render_ms_1': ms(r1['parts']['render']), 'render_ms_2': ms(r2['parts']['render']),
                          'sweeps_ms_1': ms(r1['parts']['transport_orders']), 'sweeps_ms_2': ms(r2['parts']['transport_orders']),
                          'ceiling_remove_3_of_4_orders_1light': ms(r1['wall_s']) / (ms(r1['wall_s']) - 0.75 * ms(r1['parts']['transport_orders'])),
                          'ceiling_remove_3_of_4_orders_2lights': ms(r2['wall_s']) / (ms(r2['wall_s']) - 0.75 * ms(r2['parts']['transport_orders'])), 'n_two_light_events': r2['n']}
# orders J (R0 XOp: one order from the WL96A component median); summed-field recurrence with J orders, learned arm keeps 1 prefix order
one = comp['one_order_s']
sens['orders_R0_wall'] = {J_: {'source_to_frame': (Ws + (J_ - 4) * one) / (Ws + (J_ - 4) * one - (J_ - 1) * one), 'edit_to_frame': (We + (J_ - 4) * one) / (We + (J_ - 4) * one - (J_ - 1) * one)} for J_ in (4, 8, 16, 32)}
# N scaling: WL91 scale91 (C3UR sender relation, the only recorded patch scaling); linear fits through the 3 kappa_s = 8 rows
rows = [r for r in J['scale91']['rows'] if r['kappa_s'] == 8]
Nn = np.array([r['N'] for r in rows], float)
fit = {}
for key in ('q_table_s', 'apply_one_order_s', 'C3UR_table_J4_s', 'C3UR_table_J8_s', 'C3URX1_update_s', 'X1_source_eval_s'):
    y = np.array([ms(r[key]) for r in rows]); sl, ic = np.polyfit(Nn, y, 1)
    fit[key] = {'values_ms': y.tolist(), 'ratio_last_over_first': float(y[-1] / y[0]), 'slope_us_per_patch': float(sl * 1e3), 'intercept_ms': float(ic), 'intercept_share_at_N1270': float(ic / y[0])}
sens['N_scaling_scale91_fits'] = {'N': Nn.tolist(), **fit}
frame = ms(L['C3UR']['frame']['wall_s'])


def u_src(N, with_frame=True, intercepts=True):
    q = fit['q_table_s']['slope_us_per_patch'] * 1e-3 * N + (fit['q_table_s']['intercept_ms'] if intercepts else 0.0)
    sw = fit['C3UR_table_J4_s']['slope_us_per_patch'] * 1e-3 * N + (fit['C3UR_table_J4_s']['intercept_ms'] if intercepts else 0.0)
    f = frame if with_frame else 0.0
    return (f + q + sw) / (f + q + 0.25 * sw)


sens['N_scaling_ceiling_source_C3UR_J4_remove_3_of_4'] = {
    'measured_rows': {int(r['N']): (frame + ms(r['q_table_s']) + ms(r['C3UR_table_J4_s'])) / (frame + ms(r['q_table_s']) + 0.25 * ms(r['C3UR_table_J4_s'])) for r in rows},
    'measured_rows_no_frame': {int(r['N']): (ms(r['q_table_s']) + ms(r['C3UR_table_J4_s'])) / (ms(r['q_table_s']) + 0.25 * ms(r['C3UR_table_J4_s'])) for r in rows},
    'linear_extrapolation_with_frame': {N: u_src(N) for N in (1270, 5080, 20000, 40000, 100000, 1000000)},
    'slopes_only_no_frame_any_N': u_src(1.0, False, False)}
# solve N for 1.25 (extrapolation far outside the measured range N <= 5,080)
qs, qi = fit['q_table_s']['slope_us_per_patch'] * 1e-3, fit['q_table_s']['intercept_ms']
ss_, si = fit['C3UR_table_J4_s']['slope_us_per_patch'] * 1e-3, fit['C3UR_table_J4_s']['intercept_ms']
num = 1.25 * (frame + qi + 0.25 * si) - (frame + qi + si); den = (qs + ss_) - 1.25 * (qs + 0.25 * ss_)
sens['N_scaling_ceiling_source_C3UR_J4_remove_3_of_4']['N_for_1.25_with_frame_extrapolated'] = float(num / den) if den > 0 else None
# source count (separate tables per source), WL91
sens['source_count_scale91'] = [r for r in J['scale91']['rows'] if 'source_count' in r][0]['source_count']
# X1 source evaluation vs ray count (the 'throughput' used by the WL96 critic for the DQ dirty-pair cost)
allr = J['scale91']['rows']
sens['X1_source_eval_vs_landed_rays'] = sorted([(r['landed_rays'], ms(r['X1_source_eval_s'])) for r in allr])
lr = np.array([r['landed_rays'] for r in allr], float); y = np.array([ms(r['X1_source_eval_s']) for r in allr]); sl, ic = np.polyfit(lr, y, 1)
sens['X1_source_eval_fit'] = {'slope_ns_per_landed_ray': float(sl * 1e6), 'intercept_ms': float(ic), 'corr': float(np.corrcoef(lr, y)[0, 1]),
                             'critic_assumed_ms_per_5.5M_retests': 5.5e6 / 1.05e6 * 11.5, 'fit_predicted_ms_for_5.5M_landed_rays': float(ic + sl * 5.5e6),
                             'max_measured_ms': float(y.max()), 'max_measured_landed_rays': float(lr.max())}
rr = np.array([r['unique_relation_rays'] for r in allr], float); y = np.array([ms(r['C3URX1_update_s']) for r in allr]); sl, ic = np.polyfit(rr, y, 1)
sens['relation_update_vs_relation_size_fit'] = {'points': sorted([(r['unique_relation_rays'], r['relation_rays_retraced'], ms(r['C3URX1_update_s'])) for r in allr]),
                                                'slope_ns_per_relation_ray': float(sl * 1e6), 'intercept_ms': float(ic), 'predicted_ms_for_9.09M_pair_cache': float(ic + sl * 9.09e6)}
out['E_sensitivity'] = sens

# ---------------------------------------------------------------- F. work measures
N, K, NB = c['N'], 512, 64
D = N * NB
out['F_work'] = {'relation_rays': N * K, 'dirty_rays_per_edit': c['rays']['dirty_rays_per_edit_median'], 'dirty_fraction': c['rays']['dirty_rays_per_edit_median'] / (N * K),
                 'q_shadow_rays_per_light': c['rays']['q_shadow_rays_per_light'], 'render_rays_nominal_per_frame': c['rays']['render_rays_nominal_per_frame'],
                 'frame_rays_over_all_update_rays': c['rays']['render_rays_nominal_per_frame'] / (c['rays']['dirty_rays_per_edit_median'] + c['rays']['q_shadow_rays_per_light']),
                 'MAC_one_explicit_order': N * K * NB * 3, 'MAC_three_tail_orders': 3 * N * K * NB * 3,
                 'MAC_lambda24': 2 * 3 * D * 24, 'MAC_lambda64': 2 * 3 * D * 64,
                 'MAC_ratio_tail_over_lambda24': 3 * N * K * NB * 3 / (2 * 3 * D * 24), 'time_ratio_tail_over_lambda24': comp['tail_3_orders_s'] / comp['lambda24_apply_s'],
                 'MAC_ratio_tail_over_lambda64': 3 * N * K * NB * 3 / (2 * 3 * D * 64), 'time_ratio_tail_over_lambda64': comp['tail_3_orders_s'] / comp['lambda64_apply_s'],
                 'time_ratio_4_orders_over_ecro_apply': 4 * comp['one_order_s'] / comp['ecro_apply_s'],
                 'LAM64_factor_floats': 2 * 3 * D * 64, 'LAM24_factor_floats': 2 * 3 * D * 24, 'OWN_factor_floats_WL87': E87['E1']['memory']['ecro_factor_bytes'] // 4,
                 'generator_dense_form_factor_floats_N2': N * N}

# ---------------------------------------------------------------- G. WL86 fixed per-edit costs
g = {}
for ed in ('E1', 'E2'):
    em = J['econ86'][ed]['edit_median']
    g[ed] = {'ecro_fixed_wall_ms': ms(J['econ86'][ed]['ecro_fixed_wall_s']), 'pt_fixed_wall_ms': ms(J['econ86'][ed]['pt_fixed_wall_s']),
             'state_selective_ms': ms(em['state_selective_s']), 'state_full_ms': ms(em['state_full_s']), 'state_selective_gpu_kernel_ms': em['state_selective_gpu_ms'], 'state_full_gpu_kernel_ms': em['state_full_gpu_ms'],
             'operator_ms': ms(em['inference_s']), 'selective_equals_full_all': all(x['selective_equals_full'] for x in J['econ86'][ed]['edits']),
             'matched_speedups': {k: {kk: vv for kk, vv in v.items() if 'speedup' in kk} for k, v in J['econ86'][ed]['matched'].items()}}
out['G_WL86'] = g
jf = P('results', 'audit97', 'D', 'ceilings.json')
json.dump(out, open(jf, 'w'), indent=1)
print('written', os.path.normpath(jf))
