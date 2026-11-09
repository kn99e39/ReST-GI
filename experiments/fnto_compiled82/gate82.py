"""WL82 gates (protocol decision section).

  python gate82.py repro <data82>                 -> REPRODUCED | NOT_REPRODUCED
  python gate82.py tests <tests_dir>              -> PASS | FAIL
  python gate82.py conv <eval_gate.json> <out_steps.json>   -> list of arms to escalate (space separated) or NONE
  python gate82.py conv2 <eval_gate.json> <steps.json>      -> records the 24000 vs 12000 check for escalated arms
  python gate82.py verdict <root_results> <out.json>
"""
import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
P82 = json.load(open(os.path.join(HERE, 'protocol', 'compiled82.json')))
F = dict(P82['fitting'])
for _k, _e in [('steps_primary', 'W82_STEP1'), ('steps_convergence', 'W82_STEP0'), ('steps_escalated', 'W82_STEP2')]:
    if os.environ.get(_e):
        F[_k] = int(os.environ[_e])          # dev smoke only
TAGS = ['_'.join(a) for a in P82['arms']['fit_arms']]


def obj_median(r, tag):
    return r['shape']['median'] if tag.endswith('_shape') else r['summary']['E_L1']['median']


def conv(ev_path, out):
    ev = json.load(open(ev_path))['arms']
    S = {'rule': P82['fitting']['convergence_diagnostic'], 'arms': {}}
    esc = []
    for t in TAGS:
        a, b, c = (ev.get(f'{t}_{F["steps_primary"]}_generic'), ev.get(f'{t}_{F["steps_convergence"]}_generic'),
                   ev.get(f'{t}_{F["steps_primary"]}_reinit'))
        d = abs(obj_median(a, t) - obj_median(b, t))
        S['arms'][t] = {'median_12000': obj_median(a, t), 'median_6000': obj_median(b, t), 'delta': d,
                        'median_reinit_12000': obj_median(c, t), 'reinit_spread': abs(obj_median(a, t) - obj_median(c, t)),
                        'converged': d <= 0.005, 'steps': F['steps_primary'] if d <= 0.005 else F['steps_escalated']}
        if d > 0.005:
            esc.append(t)
    json.dump(S, open(out, 'w'), indent=1)
    print(' '.join(esc) if esc else 'NONE')


def conv2(ev_path, steps_path):
    ev = json.load(open(ev_path))['arms']
    S = json.load(open(steps_path))
    for t, v in S['arms'].items():
        if v['steps'] == F['steps_escalated']:
            a = ev[f'{t}_{F["steps_escalated"]}_generic']; b = ev[f'{t}_{F["steps_primary"]}_generic']
            v['median_24000'] = obj_median(a, t)
            v['delta_24000_vs_12000'] = abs(obj_median(a, t) - obj_median(b, t))
            v['OPTIMISATION_LIMITED'] = v['delta_24000_vs_12000'] > 0.005
    json.dump(S, open(steps_path, 'w'), indent=1)
    print('ok')


def steps_of(S, t):
    return S['arms'][t]['steps'] if S and t in S['arms'] else F['steps_primary']


def verdict(R, out):
    rd = lambda p: json.load(open(os.path.join(R, p))) if os.path.exists(os.path.join(R, p)) else None
    rep = rd('data/repro.json'); tst = rd('tests/tests.json'); con = rd('contract/contract.json'); S = rd('steps.json')
    ev = {s: rd(f'eval/eval82_{s}.json') for s in ['gate', 'test_iid', 'test_block', 'real']}
    key = lambda t: f'{t}_{steps_of(S, t)}_generic'
    A = ev['test_iid']['arms']
    th = P82['evaluation']['wl81_thresholds']
    val = {'reference_reproduced': bool(rep and rep['REPRODUCED']), 'focused_tests_pass': bool(tst and tst['PASS']),
           'cert_physical_contracts_pass': bool(con and all(con[t]['PHYSICAL_CONTRACT_PASS'] for t in ['C0_cert_metric', 'E_cert_metric', 'ER_cert_metric'])),
           'shape_noise_rule_test_iid': ev['test_iid']['noise_floor']['E_shape_median'] <= 0.025,
           'split_separation': bool(tst and tst['T9_split']['PASS'])}
    valid = all(val.values())
    g = lambda t: A[key(t)]['gate']['PASS']
    med = lambda t, k='el1': A[key(t)]['summary']['E_L1']['median'] if k == 'el1' else A[key(t)]['shape']['median']
    oracle_r1 = A[key('C0_cert_metric')]['gate_oracle']['R1'] or A[key('C0_unb_shape')]['gate_oracle']['R1']
    er_better = med('ER_cert_metric') <= 0.75 * med('E_cert_metric') and med('ER_cert_metric', 'sh') <= 0.75 * med('E_cert_metric', 'sh')
    if not valid:
        outcome = 'E'
    elif g('E_cert_metric'):
        outcome = 'A'
    elif oracle_r1:
        outcome = 'C'
    elif g('ER_cert_metric') and er_better:
        outcome = 'B'
    else:
        outcome = 'D'
    nz = ev['test_iid']['noise_floor']['E_L1_median']
    c0 = A[key('C0_cert_metric')]
    share = (c0['summary']['E_L1']['median'] - c0['oracle']['E_L1']['median']) / max(c0['summary']['E_L1']['median'] - nz, 1e-12)
    table = {}
    for s in ['gate', 'test_iid', 'test_block', 'real']:
        if ev[s] is None:
            continue
        T = {}
        for t in TAGS:
            r = ev[s]['arms'].get(key(t))
            if r is None:
                continue
            T[t] = {'E_L1_median': r['summary']['E_L1']['median'], 'E_L1_p90': r['summary']['E_L1']['p90'], 'E_L1_max': r['summary']['E_L1']['max'],
                    'E_alb_median': r['summary']['E_alb']['median'], 'E_alb_p95': r['summary']['E_alb']['p95'],
                    'E_shape_median': r['shape']['median'], 'E_shape_p90': r['shape']['p90'], 'grazing_E_L1': r['summary']['grazing_median'],
                    'oracle_E_L1_median': r['oracle']['E_L1']['median'], 'oracle_E_L1_p90': r['oracle']['E_L1']['p90'],
                    'gate': r.get('gate', r.get('gate_no_highlight')), 'gate_oracle': r.get('gate_oracle', r.get('gate_oracle_no_highlight')),
                    'highlight': r.get('highlight'), 'failure_cluster': r['failure_cluster']}
        if s == 'gate':
            for a in ['control', 'candidate', 'unbounded']:
                r = ev[s]['arms'].get(f'W81_{a}')
                if r:
                    T[f'W81_{a}'] = {'E_L1_median': r['summary']['E_L1']['median'], 'E_L1_p90': r['summary']['E_L1']['p90'],
                                     'E_alb_median': r['summary']['E_alb']['median'], 'E_shape_median': r['shape']['median'],
                                     'oracle_E_L1_median': r['oracle']['E_L1']['median'], 'gate': r.get('gate'), 'gate_oracle': r.get('gate_oracle')}
        table[s] = T
    res = {'validity': val, 'VALID': valid, 'ORACLE_R1_test_iid': bool(oracle_r1), 'GATE_E_cert_test_iid': bool(g('E_cert_metric')),
           'GATE_ER_cert_test_iid': bool(g('ER_cert_metric')), 'ER_vs_E_substantial': bool(er_better), 'OUTCOME': outcome,
           'energy_share_test_iid': share, 'thresholds': th, 'steps': S, 'table': table,
           'test_block_region_check': {t: table.get('test_block', {}).get(t, {}).get('gate') for t in TAGS}}
    json.dump(res, open(out, 'w'), indent=1, default=float)
    print(outcome)


if __name__ == '__main__':
    w = sys.argv[1]
    if w == 'repro':
        r = json.load(open(os.path.join(sys.argv[2], 'repro.json')))
        print('REPRODUCED' if r['REPRODUCED'] else 'NOT_REPRODUCED')
    elif w == 'tests':
        r = json.load(open(os.path.join(sys.argv[2], 'tests.json')))
        print('PASS' if r['PASS'] else 'FAIL')
    elif w == 'conv':
        conv(sys.argv[2], sys.argv[3])
    elif w == 'conv2':
        conv2(sys.argv[2], sys.argv[3])
    elif w == 'verdict':
        verdict(sys.argv[2], sys.argv[3])
