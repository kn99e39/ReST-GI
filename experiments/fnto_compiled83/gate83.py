"""WL83 gates and verdict (protocol decision section).

  python gate83.py stageA <root_results>        -> QUALIFIED | NOT_QUALIFIED (+ writes stageA83.json)
  python gate83.py verdict <root_results> <out.json>
"""
import json, math, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
P83 = json.load(open(os.path.join(HERE, 'protocol', 'compiled83.json')))
G = P83['gates']
SEEDS = P83['compiler']['seeds']


def rd(R, p):
    f = os.path.join(R, p)
    return json.load(open(f)) if os.path.exists(f) else None


def stageA(R):
    a = {}
    rep = rd(R, 'A/repro/repro.json')
    a['reference_reproduced'] = bool(rep and rep['REPRODUCED'])
    t82 = rd(R, 'A/tests82/tests.json')
    a['wl82_focused_tests'] = bool(t82 and t82['PASS'])
    c82 = rd(R, 'A/contract82/contract.json')
    a['wl82_cert_contracts'] = bool(c82 and all(c82[t]['PHYSICAL_CONTRACT_PASS'] for t in ['C0_cert_metric', 'E_cert_metric', 'ER_cert_metric']))
    ind = rd(R, 'A/independence/independence.json')
    a['population_independence'] = bool(ind and ind['PASS'])
    # stored WL82 direct fits re-evaluated through the WL83 evaluator == WL82 recorded medians
    w82 = json.load(open(os.path.expanduser('~/rest_compiled82/results/eval/eval82_test_iid.json')))['arms']
    e83 = rd(R, 'A/eval_unpatched/eval83_wl82_test_iid.json')
    dev = {}
    for arm in ['C0', 'E', 'ER']:
        x = w82[f'{arm}_cert_metric_12000_generic']; y = e83['closures'][f'DF_{arm}']
        dev[arm] = max(abs(x['summary']['E_L1']['median'] - y['summary']['E_L1']['median']), abs(x['summary']['E_alb']['median'] - y['summary']['E_alb']['median']),
                       abs(x['shape']['median'] - y['shape']['median']))
    a['wl82_direct_fit_reproduction_max_abs_dev'] = dev
    a['wl82_direct_fit_reproduced'] = all(v <= 1e-9 for v in dev.values())
    a['QUALIFIED'] = all(v for k, v in a.items() if isinstance(v, bool))
    json.dump(a, open(os.path.join(R, 'stageA83.json'), 'w'), indent=1)
    print('QUALIFIED' if a['QUALIFIED'] else 'NOT_QUALIFIED')


def verdict(R, out):
    ev = {s: rd(R, f'eval/eval83_{s}.json') for s in ['test83_iid', 'test83_block', 'test83_hard', 'val83', 'train_eval', 'wl82_test_iid', 'real83']}
    A = rd(R, 'stageA83.json'); rt = rd(R, 'runtime/runtime_tests.json'); au = rd(R, 'runtime/audit.json')
    rv = rd(R, 'render/validation.json'); be = rd(R, 'bench/bench83.json'); rn = rd(R, 'render/render83.json')
    C = lambda s, n: ev[s]['closures'][n]
    med = lambda s, n: C(s, n)['summary']['E_L1']['median']
    validity = {'stageA_qualified': bool(A and A['QUALIFIED']), 'runtime_tests': bool(rt and rt['PASS']), 'runtime_passivity_audit': bool(au and au['PASS']),
                'renderer_validation': bool(rv and rv['PASS'])}
    rep = {arm: C('test83_iid', f'DF_{arm}')['gate'] for arm in ['C0', 'E', 'ER']}
    rep_ok = {arm: bool(rep[arm]['PASS']) for arm in rep}
    LG = G['learned_rule']
    learned = {}
    for arm in ['C0', 'E', 'ER']:
        per = {}
        df = C('test83_iid', f'DF_{arm}')['summary']
        for sd in SEEDS:
            n = f'L_{arm}_s{sd}'
            ti, tb, tr = C('test83_iid', n), C('test83_block', n), C('train_eval', n)
            x = {'L1_wl81_gate_test83_iid': bool(ti['gate']['PASS']),
                 'L2_transfer': ti['summary']['E_L1']['median'] <= max(LG['vs_df_ratio'] * df['E_L1']['median'], df['E_L1']['median'] + LG['vs_df_abs']),
                 'L3_generalisation_gap': ti['summary']['E_L1']['median'] <= LG['gen_gap_ratio'] * tr['summary']['E_L1']['median'],
                 'L4_block': tb['summary']['E_L1']['median'] <= LG['block_E_L1_median'] and tb['summary']['E_L1']['p90'] <= LG['block_E_L1_p90']
                             and tb['summary']['E_alb']['median'] <= LG['block_E_alb_median'],
                 'L5_grazing_band': ti['summary']['grazing_median'] <= LG['grazing_vs_all_ratio'] * ti['summary']['E_L1']['median']
                             and ti['summary']['band_median'] <= LG['band_vs_neighbour_ratio'] * ti['summary']['band_neighbour_median']}
            x['PASS'] = bool(all(x.values()))
            per[n] = x
        per['PASS'] = bool(all(per[f'L_{arm}_s{sd}']['PASS'] for sd in SEEDS))
        learned[arm] = per
    econ = {arm: bool(be and be['economics'][arm]['PASS_speed'] and not be['foreign_gpu_process']) for arm in ['C0', 'E', 'ER']}
    rend = {arm: bool(rn and rn['RENDER_rule_per_arm'].get(f'{arm}_L', False)) for arm in ['E', 'ER']}
    full = {arm: learned[arm]['PASS'] and econ[arm] and rend[arm] for arm in ['E', 'ER']}
    hard = {arm: float(np.mean([med('test83_hard', f'L_{arm}_s{sd}') for sd in SEEDS])) for arm in ['E', 'ER']}
    cost = None
    if be:
        rows = [r for r in be['rows'] if r['mode'] == 'io' and r['op'] == 'eval_pdf' and r['n'] == max(rr['n'] for rr in be['rows'] if rr['mode'] == 'io')]
        er = {r['material']: r['gpu_ms_median'] for r in rows if r['arm'] == 'ER'}; e = {r['material']: r['gpu_ms_median'] for r in rows if r['arm'] == 'E'}
        cost = float(np.median([er[k] / e[k] for k in e]))
    worth = bool(hard['ER'] <= G['er_worth']['hard_ratio'] * hard['E'] and cost is not None and cost <= G['er_worth']['cost_ratio_max'])
    valid = all(validity.values())
    if not valid:
        outcome = 'E'
    elif not (rep_ok['E'] or rep_ok['ER']):
        outcome = 'E (representation not reproduced on the new population)'
    elif full['ER'] and (worth or not full['E']):
        outcome = 'B'
    elif full['E']:
        outcome = 'A'
    elif not learned['E']['PASS'] and not learned['ER']['PASS']:
        outcome = 'C'
    else:
        outcome = 'D'
    table = {}
    for s, e in ev.items():
        if e is None:
            continue
        table[s] = {n: {'E_L1_median': r['summary']['E_L1']['median'], 'E_L1_p90': r['summary']['E_L1']['p90'], 'E_L1_max': r['summary']['E_L1']['max'],
                        'E_alb_median': r['summary']['E_alb']['median'], 'E_alb_p95': r['summary']['E_alb']['p95'], 'E_shape_median': r['shape']['median'],
                        'grazing': r['summary']['grazing_median'], 'transfer_median': r.get('transfer', {}).get('median'),
                        'highlight': {k: r['highlight'][k] for k in ['peak_err_deg_median', 'abs_logw_median']} if 'highlight' in r else None,
                        'gate': r['gate'], 'hard_box_members': r['hard_box_members'], 'cluster': r['failure_cluster_set_terciles']}
                    for n, r in e['closures'].items()}
    res = {'validity': validity, 'VALID': valid, 'representation_DF_test83_iid': rep, 'learned': learned, 'economics': econ, 'rendering': rend,
           'full': full, 'hard_median_learned': hard, 'ER_over_E_eval_pdf_cost': cost, 'ER_worth': worth, 'OUTCOME': outcome, 'table': table,
           'noise': {s: e['noise_floor'] for s, e in ev.items() if e}}
    json.dump(res, open(out, 'w'), indent=1, default=float)
    print(outcome)


if __name__ == '__main__':
    if sys.argv[1] == 'stageA':
        stageA(sys.argv[2])
    else:
        verdict(sys.argv[2], sys.argv[3])
