"""WL93C analysis by the prospective rules (protocol ecro_stage93c). numpy only.
  python analyze93c.py <R> <R86> <R92A> <R92B>   -> results/final93c.json ; prints VERDICT <label>
"""
import glob, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for d in ('ecro86', 'ecro_attr92'):
    sys.path.insert(0, os.path.join(HERE, '..', d))
import analyze92 as AN  # noqa: E402
import analyze86 as A  # noqa: E402

P = json.load(open(os.path.join(HERE, 'protocol', 'ecro_stage93c.json')))
TH = json.load(open(os.path.join(HERE, '..', 'ecro_stage93', 'protocol', 'ecro_stage93.json')))['thresholds']
SEEDS = [8601, 8602]
TAGS = P['renders']['tags']
REGS = ['stationary_all', 'A', 'C', 'D', 'obj_panel']


def jl(fn):
    return json.load(open(fn)) if os.path.exists(fn) else None


def main():
    R, R86, R92A, R92B = sys.argv[1:5]
    out = {k: jl(os.path.join(R, 'results', f)) for k, f in (('reuse', 'reuse.json'), ('tests_pre', 'tests93c_pre.json'), ('tests_post', 'tests93c_post.json'),
                                                               ('oracle_gate', 'oracle_gate.json'), ('contrasts', 'contrasts.json'))}
    ora = jl(os.path.join(R, 'results', 'oracle.json')); acc = jl(os.path.join(R, 'results', 'account.json'))
    out['oracle_groups'] = ora['groups'] if ora else None
    out['account_groups'] = acc['groups'] if acc else None
    stops = []
    if not (out['reuse'] and out['reuse']['PASS']):
        stops.append('STOP_REUSE_NOT_VERIFIED')
    if not (out['tests_pre'] and out['tests_pre']['PASS']):
        stops.append('STOP_TEST_FAILURE_PRE')
    feasible = bool(out['oracle_gate'] and out['oracle_gate']['ORACLE_FEASIBLE'])
    if feasible and not (out['tests_post'] and out['tests_post']['PASS']):
        stops.append('STOP_TEST_FAILURE_POST')

    def ref(tag):
        for root in (R, R92B, R92A, R86):
            if glob.glob(os.path.join(root, 'results', 'renders', f'REF_ER_{tag}_r*.npz')):
                return A.load(root, f'REF_ER_{tag}')
    models = [f'{a}_s{s}' for a in ('ORA', 'T0', 'TQ', 'TR', 'TQR') for s in SEEDS]
    rend, dH = {}, {}
    for tag in TAGS:
        rf = ref(tag); Mk = AN.mask_of(R, R86, tag)
        if rf is None:
            continue
        rend[tag] = {n: AN.eh(rf, AN.hl(R, f'I93C_{n}_{tag}'), Mk, REGS) for n in models if AN.hl(R, f'I93C_{n}_{tag}') is not None}
    for e in ('E2', 'E5k995'):
        if f'{e}_T0' in rend and f'{e}_T1' in rend:
            Mk = AN.mask_of(R, R86, f'{e}_T1'); r0, r1 = ref(f'{e}_T0'), ref(f'{e}_T1')
            dH[e] = {n: AN.dh(r0, r1, AN.hl(R, f'I93C_{n}_{e}_T0'), AN.hl(R, f'I93C_{n}_{e}_T1'), Mk) for n in models if AN.hl(R, f'I93C_{n}_{e}_T0') is not None}
    out['renders'] = rend; out['renders_dH'] = dH

    def gates(n):
        g = {}
        for e in ('E2', 'E5k995'):
            for t in ('T0', 'T1'):
                x = rend.get(f'{e}_{t}', {}).get(n)
                if x:
                    g[f'F1:{e}_{t}'] = bool(x['stationary_all']['E_H'] <= TH['EH_stationary_max'] and x['stationary_all']['capture'] <= TH['capture_max'])
            x = rend.get(f'{e}_T1', {}).get(n)
            if x:
                for k in ('A', 'D'):
                    if k in x and x[k]['pixels'] >= TH['min_pixels']:
                        g[f'F2:{e}:{k}'] = bool(x[k]['E_H'] <= TH['EH_region_max'])
            g[f'F3:{e}'] = dH.get(e, {}).get(n, {}).get('D', {}).get('F3', 'NA')
        x = rend.get('E1_T1_LSPOT', {}).get(n)
        if x:
            g['F4:E1_LSPOT'] = bool(x['stationary_all']['E_H'] <= TH['EH_unseen_source_max'])
        return g
    out['gates'] = {n: gates(n) for n in models if any(n in v for v in rend.values())}
    crit = {}
    lab = None
    if not feasible:
        lab = 'ORACLE_COMPOSITION_LIMITED'
    elif not stops and acc and out['contrasts']:
        D = out['contrasts']; V = acc['groups']['val']
        per = {}
        for s in SEEDS:
            n = lambda a: f'{a}_s{s}'
            m = lambda a, k: V[n(a)][{'SRC': 'SRC', 'RECV': 'RECV_Q', 'SINGLE': 'SINGLE', 'WHOLE': 'WHOLE'}[k]]
            gain = m('T0', 'SRC') - m('TQ', 'SRC'); gap = m('TQ', 'WHOLE') - m('T0', 'WHOLE')
            c = lambda x, y, k: D[f'{n(x)}-{n(y)}:{k}']
            a_ = m('TQR', 'SRC') <= m('T0', 'SRC') - 0.5 * gain and c('TQR', 'TQ', 'SRC')['diff'] <= c('TQR', 'TQ', 'SRC')['margin']
            r_ = c('TQR', 'T0', 'RECV')['diff'] <= -c('TQR', 'T0', 'RECV')['margin'] and c('TQR', 'TQ', 'RECV')['diff'] <= -c('TQR', 'TQ', 'RECV')['margin']
            b_ = c('TQR', 'TQ', 'WHOLE')['diff'] <= -max(c('TQR', 'TQ', 'WHOLE')['margin'], 0.5 * gap) and c('TQR', 'TQ', 'SINGLE')['diff'] <= c('TQR', 'TQ', 'SINGLE')['margin']
            sh = []
            for tag in ('E2_T0', 'E2_T1', 'E5k995_T0', 'E5k995_T1'):
                x, y = rend.get(tag, {}).get(n('TQR'), {}).get('C'), rend.get(tag, {}).get(n('TQ'), {}).get('C')
                if x and y and x['E_H'] > 1.25 * y['E_H'] + 0.002:
                    sh.append([tag, 'C', x['E_H'], y['E_H']])
                for k in ('stationary_all', 'D'):
                    x, y = rend.get(tag, {}).get(n('TQR'), {}).get(k), rend.get(tag, {}).get(n('T0'), {}).get(k)
                    if x and y and x['E_H'] > 1.25 * y['E_H'] + 0.002:
                        sh.append([tag, k, x['E_H'], y['E_H']])
            x, y = rend.get('E1_T1_LSPOT', {}).get(n('TQR'), {}).get('stationary_all'), rend.get('E1_T1_LSPOT', {}).get(n('T0'), {}).get('stationary_all')
            if x and y and x['E_H'] > 1.25 * y['E_H'] + 0.002:
                sh.append(['E1_T1_LSPOT', 'stationary_all', x['E_H'], y['E_H']])
            gT, gX = gates(n('T0')), gates(n('TQR'))
            lost = [k for k, v in gT.items() if v in (True, 'PASS') and gX.get(k) not in (True, 'PASS')]
            s_ = not sh and not lost and bool(rend)
            per[s] = {'a': bool(a_), 'r': bool(r_), 'b': bool(b_), 's': bool(s_), 'sharp_regressions': sh, 'gates_lost': lost, 'gain': gain, 'gap': gap,
                      'values': {x: {k: m(x, k) for k in ('SRC', 'RECV', 'SINGLE', 'WHOLE')} for x in ('T0', 'TQ', 'TR', 'TQR')}}
        crit['per_seed'] = per
        al = lambda k: all(per[s][k] for s in SEEDS)
        if al('a') and al('r') and al('b') and al('s'):
            lab = 'DUAL_SUPERVISION_SIGNAL'
        elif al('a') and al('r'):
            lab = 'STAGE_ONLY_IMPROVEMENT'
        elif not any(per[s]['r'] for s in SEEDS) or (not any(per[s]['a'] for s in SEEDS) and not any(per[s]['b'] for s in SEEDS)):
            lab = 'NO_SIGNAL_AT_SMOKE_SCALE'
        else:
            lab = 'INCONCLUSIVE'
    if stops:
        lab = 'INCONCLUSIVE'
    out['criteria'] = crit
    out['verdict'] = {'label': lab, 'stops': stops, 'oracle_feasible': feasible, 'limiting_factor': (out['oracle_gate'] or {}).get('limiting_factor'),
                      'note': 'smoke-scale feasibility conclusion only; no adoption'}
    json.dump(out, open(os.path.join(R, 'results', 'final93c.json'), 'w'), indent=1, default=str)
    print('VERDICT', lab, 'stops', stops, 'oracle_feasible', feasible, 'limiting', out['verdict']['limiting_factor'], flush=True)
    for s, v in crit.get('per_seed', {}).items():
        print('SEED', s, {k: v[k] for k in ('a', 'r', 'b', 's')}, json.dumps({a: {k: round(x, 4) for k, x in vv.items()} for a, vv in v['values'].items()}), flush=True)
    for tag, v in rend.items():
        print('RENDER', tag, json.dumps({n: round(x.get('C', x['stationary_all'])['E_H'], 4) for n, x in v.items()}), flush=True)


if __name__ == '__main__':
    main()
