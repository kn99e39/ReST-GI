"""WL93A analysis by the prospective rules (protocol ecro_stage93a). numpy only.
  python analyze93a.py <R> <R86> <R92A> <R92B>   -> results/final93a.json ; prints VERDICT <label>
"""
import glob, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for d in ('ecro86', 'ecro_attr92'):
    sys.path.insert(0, os.path.join(HERE, '..', d))
import analyze92 as AN  # noqa: E402
import analyze86 as A  # noqa: E402

P = json.load(open(os.path.join(HERE, 'protocol', 'ecro_stage93a.json')))
SEEDS = P['task_b']['training']['seeds']
ARMS = list(P['task_b']['arms'])
TAGS = ['E2_T0', 'E2_T1', 'E1_T1_LSPOT', 'E5k995_T0', 'E5k995_T1']
REGS = ['stationary_all', 'C', 'D', 'obj_panel']


def jl(fn):
    return json.load(open(fn)) if os.path.exists(fn) else None


def main():
    R, R86, R92A, R92B = sys.argv[1:5]
    out = {'taskA': jl(os.path.join(R, 'results', 'taskA.json')), 'tests_pre': jl(os.path.join(R, 'results', 'tests93a_pre.json')),
           'tests_post': jl(os.path.join(R, 'results', 'tests93a_post.json')), 'evalB': jl(os.path.join(R, 'results', 'evalB.json'))}
    stops = []
    TA = out['taskA']
    rel = bool(TA and TA['TARGET_RELIABLE']['RELIABLE'])
    if not rel:
        stops.append('TARGET_UNRELIABLE')
    if rel and (not (out['tests_pre'] and out['tests_pre']['PASS']) or not (out['tests_post'] and out['tests_post']['PASS'])):
        stops.append('STOPPED_TEST_FAILURE')
    # ---- rendered smoke (diagnostic)
    def ref(tag):
        for root in (R, R92B, R92A, R86):
            if glob.glob(os.path.join(root, 'results', 'renders', f'REF_ER_{tag}_r*.npz')):
                return A.load(root, f'REF_ER_{tag}')
    rend = {}
    models = [f'{a}_s{s}' for a in ARMS + ['own92a'] for s in SEEDS]
    for tag in TAGS:
        rf = ref(tag); x0 = AN.hl(R, f'I93A_{models[0]}_{tag}')
        if rf is None or x0 is None:
            continue
        Mk = AN.mask_of(R, R86, tag)
        rend[tag] = {n: AN.eh(rf, AN.hl(R, f'I93A_{n}_{tag}'), Mk, REGS) for n in models if AN.hl(R, f'I93A_{n}_{tag}') is not None}
    out['renders'] = rend
    for e in ('E2', 'E5k995'):
        if f'{e}_T0' in rend and f'{e}_T1' in rend:
            Mk = AN.mask_of(R, R86, f'{e}_T1'); r0, r1 = ref(f'{e}_T0'), ref(f'{e}_T1')
            out.setdefault('renders_dH', {})[e] = {n: AN.dh(r0, r1, AN.hl(R, f'I93A_{n}_{e}_T0'), AN.hl(R, f'I93A_{n}_{e}_T1'), Mk) for n in models}
    # ---- criteria
    C = {}
    lab = None
    if rel and 'STOPPED_TEST_FAILURE' not in stops and out['evalB']:
        V = out['evalB']['groups']['val']; delta = TA['TARGET_RELIABLE']['delta']
        per = {}
        for s in SEEDS:
            d = V[f'dM_s{s}']; thr = max(delta / 2, 2 * d['boot']['se'], d['spread'])
            tr = out['evalB']['training']
            noreg = tr[f'TQ_s{s}']['val']['whole_rel_l1'] <= 1.05 * tr[f'T0_s{s}']['val']['whole_rel_l1'] and tr[f'TQ_s{s}']['val']['single_vs_L2_rel_l1'] <= 1.05 * tr[f'T0_s{s}']['val']['single_vs_L2_rel_l1']
            stable = tr[f'TQ_s{s}']['nonfinite_steps'] == 0 and tr[f'T0_s{s}']['nonfinite_steps'] == 0
            per[s] = {'dM': d['dM'], 'threshold': thr, 'signal': bool(d['dM'] <= -thr), 'noreg': bool(noreg), 'stable': bool(stable),
                      'se': d['boot']['se'], 'spread': d['spread'], 'ci95': d['boot']['ci95']}
        C['per_seed'] = per; C['delta'] = delta
        if all(p['signal'] and p['noreg'] and p['stable'] for p in per.values()):
            lab = 'LEARNING_SIGNAL'
        elif all(p['stable'] for p in per.values()) and not any(p['signal'] for p in per.values()):
            lab = 'NO_SIGNAL'
        else:
            lab = 'INCONCLUSIVE'
    out['criteria'] = C
    if 'TARGET_UNRELIABLE' in stops:
        verdict = 'TARGET_UNRELIABLE'
    elif stops or lab is None:
        verdict = 'INCONCLUSIVE'
    else:
        verdict = {'LEARNING_SIGNAL': 'FEASIBLE_FOR_FULL_SCALE_EVALUATION', 'NO_SIGNAL': 'NO_SIGNAL_AT_SMOKE_SCALE', 'INCONCLUSIVE': 'INCONCLUSIVE'}[lab]
    out['verdict'] = {'label': verdict, 'classification': lab, 'stops': stops, 'H_TARGET': (TA or {}).get('H_TARGET', {}).get('label'),
                      'note': 'feasibility assessment only; no adoption, no full-scale run, no successor'}
    json.dump(out, open(os.path.join(R, 'results', 'final93a.json'), 'w'), indent=1)
    print('VERDICT', verdict, 'classification', lab, 'stops', stops, 'H_TARGET', out['verdict']['H_TARGET'], flush=True)
    if C:
        print('CRITERIA', json.dumps(C['per_seed']), flush=True)
    for tag, v in rend.items():
        print('RENDER', tag, json.dumps({n: {k: round(x['E_H'], 4) for k, x in m.items()} for n, m in v.items()}), flush=True)


if __name__ == '__main__':
    main()
