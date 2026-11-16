"""WL97 E5 supplement (numpy only): derived numbers quoted in the report that analyze_e5.py does not print.
  python supp_e5.py  ->  results/audit97/E5/e5_supplement.json (+ stdout)
Reads only results/audit97/E5/run*.json, gen_s1.0.json and the original results/ecro_util96a/cost_s1.0.json, life91.json."""
import json, math, os

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
O = os.path.join(ROOT, 'results', 'audit97', 'E5')
L = lambda n: json.load(open(os.path.join(O, n + '.json')))
RUNS = ['run1_s1.0', 'run3_s1.0', 'run7_passive_s1.0', 'run2_passive_s1.0', 'run4_s1.5', 'run5_s2.0', 'run6_s3.0']
R = {k: L(k) for k in RUNS}
m = lambda p, key='wall_ms': p[key]['med']
out = {}


def ev(d, tag):
    g = {}
    for k, v in d['events'][tag]['summary'].items():
        a, kd, p = k.split('|')
        g.setdefault((a, kd), {})[p] = v
    return g


# (2) Amdahl ceiling from the R0 event alone: U = L / (L - t_tail), next to the measured R0 / ZERO ratio
am = {}
for rk in RUNS:
    for tag in R[rk]['events']:
        g = ev(R[rk], tag)
        if ('R0', 'edit') not in g:
            continue
        row = {}
        for kd in ('edit', 'source'):
            Lr = m(g[('R0', kd)]['TOTAL']); t = m(g[('R0', kd)]['tail_orders'])
            row[kd] = {'L_ms': Lr, 'tail_ms': t, 'tail_share': t / Lr, 'U_amdahl': Lr / (Lr - t),
                       'U_measured_R0_over_ZERO': (Lr / m(g[('ZERO', kd)]['TOTAL'])) if ('ZERO', kd) in g else None, 'n': g[('R0', kd)]['TOTAL']['n']}
        am[f'{rk}|{tag}'] = row
out['amdahl'] = am

# (4) generator arms relative to the adjacent R0 block
gen = {}
for rk in ('run1_s1.0', 'run3_s1.0'):
    r0 = m(ev(R[rk], 'D_R0_after_LAM')[('R0', 'edit')]['TOTAL'])
    for tag in ('A_orig_order', 'E_LAM24_np_geostate', 'F_LAM24_threads1', 'G_LAM24_default_again'):
        g = ev(R[rk], tag)
        base = m(g[('R0', 'edit')]['TOTAL']) if ('R0', 'edit') in g else r0
        p = g[('LAM24', 'edit')]
        gen[f'{rk}|{tag}'] = {'LAM24_edit_ms': m(p['TOTAL']), 'R0_edit_ms_used': base, 'U': base / m(p['TOTAL']), 'q_table_ms': m(p['source_q_table']),
                              'q_table_cpu_ms': m(p['source_q_table'], 'cpu_ms'), 'generator_ms': m(p['generator']), 'generator_cpu_ms': m(p['generator'], 'cpu_ms'),
                              'event_cpu_ms': m(p['TOTAL'], 'cpu_ms')}
for rk in ('run7_passive_s1.0', 'run2_passive_s1.0'):
    g = ev(R[rk], 'A_orig_order'); p = g[('LAM24', 'edit')]; base = m(g[('R0', 'edit')]['TOTAL'])
    gen[f'{rk}|A_orig_order'] = {'LAM24_edit_ms': m(p['TOTAL']), 'R0_edit_ms_used': base, 'U': base / m(p['TOTAL']), 'q_table_ms': m(p['source_q_table']),
                                 'q_table_cpu_ms': m(p['source_q_table'], 'cpu_ms'), 'generator_ms': m(p['generator']), 'generator_cpu_ms': m(p['generator'], 'cpu_ms'),
                                 'event_cpu_ms': m(p['TOTAL'], 'cpu_ms'), 'R0_q_table_ms': m(g[('R0', 'edit')]['source_q_table']),
                                 'load_before': R[rk]['env']['uptime_before'].split('load average:')[1].strip()}
out['generator_arms'] = gen
G = L('gen_s1.0')
c1 = R['run1_s1.0']['components']
g1 = ev(R['run1_s1.0'], 'B_interleaved_R0_ZERO')
r0e = m(g1[('R0', 'edit')]['TOTAL']); tail_e = m(g1[('R0', 'edit')]['tail_orders'])
lam = m(c1['lambda24_apply'])
trim = {'tail_3_orders_wall_ms': m(c1['tail_3_orders']), 'tail_3_orders_gpu_ms': R['run1_s1.0']['torch_profiler']['tail_3_orders']['gpu_ms_per_call'],
        'full_factors_wall_ms': G['timings']['factors_full_production']['wall_med_ms'] if 'timings' in G else None}
# gen_s1.0.json layout: flat dict of step -> stats (plus scalars)
steps = {k: v for k, v in (G.get('timings') or G).items() if isinstance(v, dict) and 'wall_med_ms' in v}
trim['steps'] = {k: {'wall_ms': v['wall_med_ms'], 'gpu_ms': v['gpu_ms']} for k, v in steps.items()}
go = [v for k, v in steps.items() if k.startswith('global_only')][0]
full = [v for k, v in steps.items() if k.startswith('factors_full')][0]
gs = m(c1['geostate_np'])
trim['derived_edit_U_not_measured_in_event'] = {
    'global_only_generator_no_geostate': r0e / (r0e - tail_e + lam + go['wall_med_ms']),
    'global_only_generator_plus_geostate': r0e / (r0e - tail_e + lam + go['wall_med_ms'] + gs),
    'full_factors_plus_geostate(isolated walls)': r0e / (r0e - tail_e + lam + full['wall_med_ms'] + gs)}
# ECRO-class operator replacing XOp coefficients + 4 orders (isolated walls; accuracy out of scope)
ecro = gs + m(c1['generator_factors']) + m(c1['ecro_apply'])
expl = m(c1['xop_build_prod']) + m(c1['four_orders'])
trim['ecro_full_operator_vs_explicit_coeff_plus_4_orders_ms'] = {'ecro_geostate+factors+apply': ecro, 'explicit_xop_build+4_orders': expl, 'edit_U_if_swapped': r0e / (r0e - expl + ecro)}
out['generator_trim'] = trim

# (3) overhead shares (run1 / run3 components): Dr.Jit kernel execution or torch GPU time versus wall
ov = {}
for rk in ('run1_s1.0', 'run3_s1.0'):
    c = R[rk]['components']; pf = R[rk]['torch_profiler']
    row = {}
    for k in ('q_table', 'render_512_64', 'render_512_1', 'render_256_1', 'set_geom', 'sg_Geom_construct', 'sg_params_update', 'sg_set_tables', 'update_state', 'state_rays_dirty', 'build_state_full'):
        row[k] = {'wall_ms': m(c[k]), 'drjit_exec_ms': c[k]['exec_ms']['med'], 'codegen_ms': c[k]['codegen_ms']['med'], 'cpu_main_ms': m(c[k], 'cpu_main_ms'), 'gpu_share': c[k]['exec_ms']['med'] / m(c[k])}
    for k in ('one_order', 'tail_3_orders', 'dirty_rays', 'xop_build_residentW', 'generator_factors', 'lambda24_apply', 'q_upload'):
        row[k] = {'wall_ms': m(c[k]), 'torch_gpu_ms': pf[k]['gpu_ms_per_call'], 'gpu_events': pf[k]['gpu_events_per_call'], 'gpu_share': pf[k]['gpu_ms_per_call'] / m(c[k])}
    row['render_fixed_overhead_ms(512x64 wall - exec)'] = m(c['render_512_64']) - c['render_512_64']['exec_ms']['med']
    ov[rk] = row
out['overhead'] = ov

# sensitivity: number of tail orders (at the measured wall per order) a zero-cost replacement would need to remove for U >= 1.25
sens = {}
one = m(c1['one_order'])
for tag, kd in (('B_interleaved_R0_ZERO', 'edit'), ('B_interleaved_R0_ZERO', 'source'), ('C_noframe_R0_ZERO', 'edit'), ('C_noframe_R0_ZERO', 'source')):
    z = m(ev(R['run1_s1.0'], tag)[('ZERO', kd)]['TOTAL'])
    sens[f'{tag}|{kd}'] = {'lifecycle_without_tail_ms': z, 'tail_ms_needed_for_1.25': 0.25 * z, 'tail_orders_needed': math.ceil(0.25 * z / one), 'one_order_wall_ms': one}
out['orders_sensitivity_run1'] = sens

# (5) scaling exponents between N = 1,270 and N = 11,430 (log-log slope of medians)
a, b = R['run1_s1.0'], R['run6_s3.0']
ex = {}
for k in ('one_order', 'tail_3_orders', 'q_table', 'update_state', 'set_geom', 'xop_build_prod', 'render_512_64', 'generator_factors', 'form_factors', 'geostate_np', 'lambda24_apply', 'build_state_full'):
    ex[k] = {'N1270_ms': m(a['components'][k]), 'N11430_ms': m(b['components'][k]), 'exponent': math.log(m(b['components'][k]) / m(a['components'][k])) / math.log(b['N'] / a['N'])}
ex['q_table_drjit_exec'] = {'N1270_ms': a['components']['q_table']['exec_ms']['med'], 'N11430_ms': b['components']['q_table']['exec_ms']['med']}
ex['tail_3_orders_gpu'] = {'N1270_ms': a['torch_profiler']['tail_3_orders']['gpu_ms_per_call'], 'N11430_ms': b['torch_profiler']['tail_3_orders']['gpu_ms_per_call']}
out['scaling_exponents'] = ex

# WL96 cross-workload ceilings re-derived from the WL91 anchors
l91 = json.load(open(os.path.join(ROOT, 'results', 'gm1_c3urx1_91', 'results', 'econ', 'life91.json')))
x = l91['arms']['C3URX1']
w = {}
for kd in ('edit', 'source_single_light'):
    Lr = x[kd]['wall_s'] * 1e3; t = x[kd]['parts']['transport_orders'] * 1e3; r = x[kd]['parts']['render'] * 1e3
    t_edit = x['edit']['parts']['transport_orders'] * 1e3
    w[kd] = {'L_ms': Lr, 'sweep_ms_own_event': t, 'U_own_sweep': Lr / (Lr - t), 'U_with_edit_event_sweep(WL96 arithmetic)': Lr / (Lr - t_edit),
             'U_noframe_own_sweep': (Lr - r) / (Lr - r - t), 'U_noframe_edit_sweep(WL96 arithmetic)': (Lr - r) / (Lr - r - t_edit)}
out['wl96_anchor_check'] = w

json.dump(out, open(os.path.join(O, 'e5_supplement.json'), 'w'), indent=1)
rd = lambda o: {k: rd(v) for k, v in o.items()} if isinstance(o, dict) else (round(o, 4) if isinstance(o, float) else o)
for k in out:
    print('##', k)
    for a_, b_ in out[k].items():
        print('  ', a_, json.dumps(rd(b_)))
