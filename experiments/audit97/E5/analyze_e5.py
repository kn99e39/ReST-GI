"""WL97 E5: consolidate the timing runs (numpy only).  python analyze_e5.py  ->  results/audit97/E5/e5_summary.json + tables on stdout."""
import json, os, sys
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
O = os.path.join(ROOT, 'results', 'audit97', 'E5')
orig = json.load(open(os.path.join(ROOT, 'results', 'ecro_util96a', 'cost_s1.0.json')))
RUNS = {k: os.path.join(O, k + '.json') for k in ('run1_s1.0', 'run2_passive_s1.0', 'run3_s1.0', 'run4_s1.5', 'run5_s2.0', 'run6_s3.0')}
R = {k: json.load(open(v)) for k, v in RUNS.items() if os.path.exists(v)}
out = {}


def ev(d, tag):
    g = {}
    for k, v in d['events'][tag]['summary'].items():
        arm, kind, part = k.split('|')
        g.setdefault((arm, kind), {})[part] = v
    return g


def med(p, key='wall_ms'):
    return p[key]['med']


def parts_of(g):
    return {k: med(v) for k, v in g.items() if k != 'TOTAL'}


print('## 1. Original vs reproduced, R0 (ms; medians)')
names = ['geometry_update', 'relation_update', 'relation_coefficients', 'source_q_table', 'prefix_order', 'tail_orders', 'table_handover', 'render']
rows = {}
for kind in ('edit', 'source'):
    o = orig['arms']['R0'][kind]
    line = {'orig_wall': o['wall_s'] * 1e3, 'orig_parts': {k: v * 1e3 for k, v in o['parts'].items()}, 'orig_cpu': o['cpu_s'] * 1e3, 'orig_gpu_kernel': o['gpu_kernel_ms']}
    for rk in ('run1_s1.0', 'run3_s1.0'):
        if rk in R:
            for tag in ('A_orig_order', 'B_interleaved_R0_ZERO', 'D_R0_after_LAM'):
                g = ev(R[rk], tag)[('R0', kind)]
                line[f'{rk}|{tag}'] = {'wall': med(g['TOTAL']), 'q25': g['TOTAL']['wall_ms']['q25'], 'q75': g['TOTAL']['wall_ms']['q75'], 'n': g['TOTAL']['n'],
                                       'parts': parts_of(g), 'cpu': med(g['TOTAL'], 'cpu_ms'), 'drjit_exec': sum(v['exec_ms']['med'] for k, v in g.items() if k != 'TOTAL')}
    rows[kind] = line
    print(f"R0 {kind}: orig {line['orig_wall']:.1f}", {k: round(line['orig_parts'].get(k, 0), 2) for k in names if k in line['orig_parts']})
    for k, v in line.items():
        if '|' in k:
            print(f"   {k}: {v['wall']:.1f} [{v['q25']:.1f},{v['q75']:.1f}] ratio {v['wall'] / line['orig_wall']:.3f}", {a: round(b, 2) for a, b in v['parts'].items()})
out['repro_R0'] = rows

print('\n## 2. U (R0 / arm), medians')
U = {'orig': orig['U']}
for rk in R:
    if R[rk]['s'] != 1.0:
        continue
    for tag in R[rk]['events']:
        g = ev(R[rk], tag)
        if ('R0', 'edit') not in g:
            continue
        for (arm, kind), p in g.items():
            if arm != 'R0':
                U.setdefault(f'{rk}|{tag}', {}).setdefault(arm, {})[kind] = med(g[('R0', kind)]['TOTAL']) / med(p['TOTAL'])
print(json.dumps(U, indent=0))
out['U'] = U

print('\n## 3. LAM arms: generator artifact (edit event, ms)')
G = {}
for rk in R:
    if R[rk]['s'] != 1.0:
        continue
    for tag in R[rk]['events']:
        g = ev(R[rk], tag)
        for (arm, kind), p in g.items():
            if arm.startswith('LAM') and kind == 'edit':
                G[f'{rk}|{tag}|{arm}'] = {'wall': med(p['TOTAL']), 'cpu': med(p['TOTAL'], 'cpu_ms'), 'q_table': med(p['source_q_table']), 'q_cpu': med(p['source_q_table'], 'cpu_ms'),
                                          'generator': med(p['generator']), 'generator_cpu': med(p['generator'], 'cpu_ms'), 'threads': R[rk]['events'][tag]['threads'],
                                          'gen_mode': R[rk]['events'][tag]['gen_mode'], 'omp_env': R[rk]['env']['env']}
                print(f"{rk}|{tag}|{arm}:", {k: (round(v, 2) if isinstance(v, float) else v) for k, v in G[f'{rk}|{tag}|{arm}'].items()})
out['generator'] = G

print('\n## 4. Components (ms): wall median [q25, q75], Dr.Jit kernel exec, torch-profiler GPU')
CT = {}
for rk in R:
    if 'components' not in R[rk]:
        continue
    C = R[rk]['components']; Pf = R[rk].get('torch_profiler', {})
    t = {}
    for k, v in C.items():
        t[k] = {'wall': med(v), 'q25': v['wall_ms']['q25'], 'q75': v['wall_ms']['q75'], 'min': v['wall_ms']['min'], 'max': v['wall_ms']['max'], 'n': v['n'], 'cpu': med(v, 'cpu_ms'),
                'drjit_exec': v['exec_ms']['med'], 'codegen': v['codegen_ms']['med'], 'nk': v['nk']['med'], 'miss': v['miss']['sum']}
        if k in Pf and 'gpu_ms_per_call' in Pf[k]:
            t[k]['torch_gpu'] = Pf[k]['gpu_ms_per_call']
    CT[rk] = {'N': R[rk]['N'], 's': R[rk]['s'], 'comp': t, 'kh_off': {k: med(v) for k, v in R[rk].get('components_kh_off', {}).items()}}
    print(f"-- {rk} N={R[rk]['N']}")
    for k, v in t.items():
        print(f"  {k:22s} {v['wall']:8.3f} [{v['q25']:.3f},{v['q75']:.3f}] n={v['n']} drjit_exec {v['drjit_exec']:.3f} torch_gpu {v.get('torch_gpu', float('nan')):.3f} miss {v['miss']:.0f}")
out['components'] = CT


def conventions(c):
    """Lifecycle ceilings U = L / (L - t_tail) under frame conventions; wall basis and GPU-work basis."""
    w = lambda k: c[k]['wall']
    gpu = lambda k: c[k].get('torch_gpu', c[k]['drjit_exec'])
    res = {}
    tail_w, tail_g = w('tail_3_orders'), gpu('tail_3_orders')
    src_w = w('q_table') + w('q_upload') + w('one_order') + tail_w + w('table_handover')
    edit_w = src_w + w('set_geom') + w('update_state') + w('xop_build_prod')
    src_g = c['q_table']['drjit_exec'] + gpu('q_upload') + gpu('one_order') + tail_g
    # GPU-work edit: BVH / table refresh bounded above by the params.update wall; dirty test + dirty re-trace + coefficient indices
    edit_g_hi = src_g + c['sg_params_update']['wall'] + gpu('dirty_rays') + c['state_rays_dirty']['drjit_exec'] + gpu('xop_build_residentW')
    edit_g_lo = src_g + c['sg_params_update']['drjit_exec'] + c['state_rays_dirty']['drjit_exec']
    frames = [('512x512x64', 'render_512_64'), ('512x512x16', 'render_512_16'), ('512x512x4', 'render_512_4'), ('512x512x1', 'render_512_1'),
              ('256x256x64', 'render_256_64'), ('256x256x16', 'render_256_16'), ('256x256x4', 'render_256_4'), ('256x256x1', 'render_256_1'), ('no frame', None)]
    for name, key in frames:
        fw = w(key) if key else 0.0
        fg = c[key]['drjit_exec'] if key else 0.0
        res[name] = {'wall': {'source_ms': src_w + fw, 'edit_ms': edit_w + fw, 'U_source': (src_w + fw) / (src_w + fw - tail_w), 'U_edit': (edit_w + fw) / (edit_w + fw - tail_w),
                              'tail_share_source': tail_w / (src_w + fw), 'tail_share_edit': tail_w / (edit_w + fw)},
                     'gpu_work': {'source_ms': src_g + fg, 'edit_ms_hi': edit_g_hi + fg, 'edit_ms_lo': edit_g_lo + fg, 'U_source': (src_g + fg) / (src_g + fg - tail_g),
                                  'U_edit_hi_denominator': (edit_g_hi + fg) / (edit_g_hi + fg - tail_g), 'U_edit_lo_denominator': (edit_g_lo + fg) / (edit_g_lo + fg - tail_g)}}
    res['_inputs'] = {'tail_wall': tail_w, 'tail_gpu': tail_g, 'src_table_wall': src_w, 'edit_table_wall': edit_w, 'src_table_gpu': src_g, 'edit_table_gpu_hi': edit_g_hi, 'edit_table_gpu_lo': edit_g_lo}
    return res


print('\n## 5. Ceilings under conventions (ZERO-cost replacement of the three tail orders)')
CV = {}
for rk in CT:
    CV[rk] = conventions(CT[rk]['comp'])
    print(f"-- {rk} N={CT[rk]['N']}  inputs", {k: round(v, 3) for k, v in CV[rk]['_inputs'].items()})
    for name, v in CV[rk].items():
        if name.startswith('_'):
            continue
        print(f"  {name:11s} wall: src {v['wall']['source_ms']:6.1f} U {v['wall']['U_source']:.3f} | edit {v['wall']['edit_ms']:6.1f} U {v['wall']['U_edit']:.3f} || gpu-work: src {v['gpu_work']['source_ms']:6.2f} U {v['gpu_work']['U_source']:.3f} "
              f"| edit {v['gpu_work']['edit_ms_lo']:.2f}..{v['gpu_work']['edit_ms_hi']:.2f} U {v['gpu_work']['U_edit_hi_denominator']:.3f}..{v['gpu_work']['U_edit_lo_denominator']:.3f}")
out['conventions'] = CV

print('\n## 6. Scaling with patch refinement (ms)')
SC = {}
keys = ['one_order', 'tail_3_orders', 'q_table', 'update_state', 'set_geom', 'xop_build_prod', 'render_512_64', 'generator_factors', 'geostate_np', 'lambda24_apply', 'build_state_full', 'sg_Geom_construct']
for rk in CT:
    c = CT[rk]['comp']
    row = {'N': CT[rk]['N'], 's': CT[rk]['s']}
    for k in keys:
        row[k] = c[k]['wall']
        if 'torch_gpu' in c[k]:
            row[k + '_gpu'] = c[k]['torch_gpu']
    row['q_table_exec'] = c['q_table']['drjit_exec']; row['render_512_64_exec'] = c['render_512_64']['drjit_exec']
    g = ev(R[rk], 'B_interleaved_R0_ZERO') if 'B_interleaved_R0_ZERO' in R[rk]['events'] else None
    if g:
        for kind in ('edit', 'source'):
            row[f'R0_{kind}'] = med(g[('R0', kind)]['TOTAL']); row[f'ZERO_{kind}'] = med(g[('ZERO', kind)]['TOTAL'])
            row[f'U_ZERO_{kind}'] = row[f'R0_{kind}'] / row[f'ZERO_{kind}']
            row[f'tail_in_{kind}'] = med(g[('R0', kind)]['tail_orders'])
        g2 = ev(R[rk], 'C_noframe_R0_ZERO')
        for kind in ('edit', 'source'):
            row[f'U_ZERO_{kind}_noframe'] = med(g2[('R0', kind)]['TOTAL']) / med(g2[('ZERO', kind)]['TOTAL'])
            row[f'R0_{kind}_noframe'] = med(g2[('R0', kind)]['TOTAL'])
    row['dirty_rays'] = R[rk]['probe_check']['dirty_rays']; row['relation_rays'] = R[rk]['probe_check']['relation_rays']
    row['selective_equals_full'] = R[rk]['probe_check']['selective_equals_full']
    SC[rk] = row
    print(rk, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()})
out['scaling'] = SC
out['env'] = {rk: {k: R[rk]['env'].get(k) for k in ('uptime_before', 'uptime_after', 'apps_before', 'apps_after', 'torch', 'drjit', 'mitsuba', 'torch_threads', 'env', 'total_s', 'torch_peak_mib', 'code_commit')} for rk in R}
out['probe_check'] = {rk: R[rk]['probe_check'] for rk in R}
json.dump(out, open(os.path.join(O, 'e5_summary.json'), 'w'), indent=1)
print('\nwritten', os.path.join(O, 'e5_summary.json'))
