"""WL93B analysis by the prospective rules (protocol ecro_stage93b). numpy only.
  python analyze93b.py <R> <R86> <R92A> <R92B> <R93A>   -> results/final93b.json ; prints LABEL <label>
"""
import glob, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for d in ('ecro86', 'ecro_attr92'):
    sys.path.insert(0, os.path.join(HERE, '..', d))
import analyze92 as AN  # noqa: E402
import analyze86 as A  # noqa: E402

P = json.load(open(os.path.join(HERE, 'protocol', 'ecro_stage93b.json')))
TH = json.load(open(os.path.join(HERE, '..', 'ecro_stage93', 'protocol', 'ecro_stage93.json')))['thresholds']
ARMS = list(P['arms']); SEEDS = [8601, 8602]
TAGS = P['renders']['tags']
REGS = ['stationary_all', 'A', 'C', 'D', 'obj_panel']


def jl(fn):
    return json.load(open(fn)) if os.path.exists(fn) else None


def pooled(rows, key):
    if key == 'SRC':
        n = sum(r['src']['l1'] for r in rows); d = sum(r['src']['sq'] for r in rows)
    else:
        n = sum(r[key][0] for r in rows); d = sum(r[key][1] for r in rows)
    return n / d


KEYS = {'SRC': 'SRC', 'WHOLE': 'whole_l1', 'SINGLE': 'single_l1'}


def main():
    R, R86, R92A, R92B, R93A = sys.argv[1:6]
    out = {k: jl(os.path.join(R, 'results', f)) for k, f in (('reuse', 'reuse.json'), ('tests_pre', 'tests93b_pre.json'), ('tests_post', 'tests93b_post.json'),
                                                               ('operators', 'operators.json'), ('pattern', 'lpat.json'))}
    acc = jl(os.path.join(R, 'results', 'account.json'))
    out['account'] = {g: v for g, v in acc['groups'].items()} if acc else None
    stops = []
    if not (out['reuse'] and out['reuse']['PASS']):
        stops.append('STOP_REUSE_NOT_VERIFIED')
    if not (out['tests_pre'] and out['tests_pre']['PASS']) or not (out['tests_post'] and out['tests_post']['PASS']):
        stops.append('STOP_TEST_FAILURE')
    # ---- reproduction of the historical controls (training-time validation metrics)
    rep = {}
    for a in ('T0', 'TQ'):
        for s in SEEDS:
            x, y = jl(os.path.join(R, 'results', 'models', f'{a}_s{s}.json')), jl(os.path.join(R, 'results', 'models', f'R_{a}_s{s}.json'))
            if x is None or y is None:
                rep[f'{a}_s{s}'] = None; continue
            rep[f'{a}_s{s}'] = {k: {'wl93a': x['val'][k], 'retrained': y['val'][k], 'rel_diff': abs(y['val'][k] - x['val'][k]) / x['val'][k]}
                                for k in ('whole_rel_l1', 'single_vs_L2_rel_l1', 'Bq_vs_trainval_q_panel_rel_l1')}
    rep_ok = all(v is not None and all(z['rel_diff'] <= 0.02 for z in v.values()) for v in rep.values())
    out['reproduction'] = {'cases': rep, 'PASS': bool(rep_ok)}
    if not rep_ok:
        stops.append('STOP_HISTORICAL_CONTROLS_NOT_REPRODUCED')
    # ---- run-to-run variation and paired bootstrap on validation
    crit = {}
    if acc:
        C = jl(os.path.join(R, 'results', 'configs.json'))['configs']
        vidx = [i for i, c in enumerate(C) if c['kind'] == 'val']
        rows = {n: [acc['per_cfg'][n][i] for i in vidx] for n in acc['per_cfg']}
        m = lambda n, k: pooled(rows[n], KEYS[k])
        r = {k: max(abs(m(f'R_{a}_s{s}', k) - m(f'{a}_s{s}', k)) for a in ('T0', 'TQ') for s in SEEDS if f'R_{a}_s{s}' in rows) if any(f'R_{a}_s{s}' in rows for a in ('T0', 'TQ') for s in SEEDS) else 0.0
             for k in KEYS}
        rng = np.random.default_rng(93002); B = [rng.integers(0, len(vidx), len(vidx)) for _ in range(2000)]

        def diff(x, y, k):
            d0 = m(x, k) - m(y, k)
            bs = [pooled([rows[x][j] for j in b], KEYS[k]) - pooled([rows[y][j] for j in b], KEYS[k]) for b in B]
            se = float(np.std(bs, ddof=1))
            return {'diff': d0, 'se': se, 'ci95': [float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))], 'margin': max(2 * se, r[k])}
        crit['run_to_run'] = r
        per = {}
        for s in SEEDS:
            n = lambda a: f'{a}_s{s}'
            gap = m(n('TQ'), 'WHOLE') - m(n('T0'), 'WHOLE'); gain = m(n('T0'), 'SRC') - m(n('TQ'), 'SRC')
            D = {f'{x}-{y}:{k}': diff(n(x), n(y), k) for x, y in (('UQ', 'TQ'), ('UQ', 'CQ'), ('U0', 'T0'), ('UQ', 'U0'), ('CQ', 'C0'), ('C0', 'T0'), ('CQ', 'TQ'), ('TQ', 'T0')) for k in KEYS}
            a_ = m(n('UQ'), 'SRC') <= m(n('T0'), 'SRC') - 0.5 * gain and D['UQ-TQ:SRC']['diff'] <= D['UQ-TQ:SRC']['margin']
            mw = lambda key: max(D[key]['margin'], 0.5 * gap)
            b_ = D['UQ-TQ:WHOLE']['diff'] <= -mw('UQ-TQ:WHOLE') and D['UQ-TQ:SINGLE']['diff'] <= D['UQ-TQ:SINGLE']['margin']
            c_ = D['UQ-CQ:WHOLE']['diff'] <= -mw('UQ-CQ:WHOLE')
            bCQ = D['CQ-TQ:WHOLE']['diff'] <= -mw('CQ-TQ:WHOLE') and D['CQ-TQ:SINGLE']['diff'] <= D['CQ-TQ:SINGLE']['margin']
            u0 = D['U0-T0:WHOLE']['diff'] <= -D['U0-T0:WHOLE']['margin']
            per[s] = {'values': {a: {k: m(n(a), k) for k in KEYS} for a in ARMS}, 'gap': gap, 'gain': gain, 'contrasts': D,
                      'a': bool(a_), 'b': bool(b_), 'c': bool(c_), 'b_CQ': bool(bCQ), 'U0_whole_improves': bool(u0)}
        crit['per_seed'] = per
    # ---- renders
    def ref(tag):
        for root in (R, R92B, R92A, R86):
            if glob.glob(os.path.join(root, 'results', 'renders', f'REF_ER_{tag}_r*.npz')):
                return A.load(root, f'REF_ER_{tag}')
    models = [f'{a}_s{s}' for a in ARMS + ['own92a'] for s in SEEDS]
    rend, dH = {}, {}
    for tag in TAGS:
        rf = ref(tag)
        try:
            Mk = AN.mask_of(R, R86, tag)
        except FileNotFoundError:
            continue
        if rf is None:
            continue
        rend[tag] = {n: AN.eh(rf, AN.hl(R, f'I93B_{n}_{tag}'), Mk, REGS) for n in models if AN.hl(R, f'I93B_{n}_{tag}') is not None}
    for e in ('E2', 'E5k995'):
        if f'{e}_T0' in rend and f'{e}_T1' in rend:
            Mk = AN.mask_of(R, R86, f'{e}_T1'); r0, r1 = ref(f'{e}_T0'), ref(f'{e}_T1')
            dH[e] = {n: AN.dh(r0, r1, AN.hl(R, f'I93B_{n}_{e}_T0'), AN.hl(R, f'I93B_{n}_{e}_T1'), Mk) for n in models}
    out['renders'] = rend; out['renders_dH'] = dH
    # T_RENDER_REF: own92a re-render = WL93A render
    rr = {}
    for s in SEEDS:
        for t in ('E2_T0', 'E2_T1'):
            a_, b_ = AN.hl(R, f'I93B_own92a_s{s}_{t}'), AN.hl(R93A, f'I93A_own92a_s{s}_{t}')
            if a_ is not None and b_ is not None and t in rend:
                Mk = AN.mask_of(R, R86, t); rf = ref(t)
                rr[f's{s}:{t}'] = abs(AN.eh(rf, a_, Mk, ['C'])['C']['E_H'] - AN.eh(rf, b_, Mk, ['C'])['C']['E_H'])
    out['render_ref_consistency'] = {'cases': rr, 'PASS': bool(rr and all(v <= 1e-6 for v in rr.values()))}
    if not out['render_ref_consistency']['PASS']:
        stops.append('STOP_RENDER_REFERENCE_INCONSISTENT')

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

    def noreg(X, s):
        x, b = f'{X}_s{s}', f'T0_s{s}'
        gx, gb = gates(x), gates(b)
        lost = [k for k, v in gb.items() if v in (True, 'PASS') and gx.get(k) not in (True, 'PASS')]
        worse = []
        for tag in ('E2_T0', 'E2_T1', 'E5k995_T0', 'E5k995_T1', 'E1_T1_LSPOT'):
            for k in ('stationary_all', 'D'):
                xa, xb = rend.get(tag, {}).get(b, {}).get(k), rend.get(tag, {}).get(x, {}).get(k)
                if xa and xb and xb['E_H'] > 1.25 * xa['E_H'] + 0.002:
                    worse.append([tag, k, xb['E_H'], xa['E_H']])
        return {'lost': lost, 'worse': worse, 'PASS': bool(not lost and not worse and rend)}
    out['gates'] = {n: gates(n) for n in models}
    if crit.get('per_seed'):
        for s in SEEDS:
            crit['per_seed'][s]['d'] = noreg('UQ', s)
    out['criteria'] = crit
    # ---- label
    lab = 'INCONCLUSIVE'
    if not stops and crit.get('per_seed'):
        p = crit['per_seed']
        al = lambda k: all(p[s][k] if k != 'd' else p[s]['d']['PASS'] for s in SEEDS)
        if al('a') and al('b') and al('c') and al('d'):
            lab = 'SHARING_SIGNAL'
        elif al('a') and al('b') and not al('c') and al('b_CQ'):
            lab = 'CAPACITY_EXPLANATION'
        elif al('a') and not any(p[s]['b'] for s in SEEDS):
            lab = 'TRADEOFF_REMAINS'
        elif not any(p[s]['b'] for s in SEEDS) and not al('a') and not any(p[s]['U0_whole_improves'] for s in SEEDS):
            lab = 'NO_ADVANTAGE'
    out['label'] = {'label': lab, 'stops': stops, 'note': 'smoke-scale feasibility evidence only; no architecture validation'}
    json.dump(out, open(os.path.join(R, 'results', 'final93b.json'), 'w'), indent=1, default=str)
    print('LABEL', lab, 'stops', stops, flush=True)
    for s, v in crit.get('per_seed', {}).items():
        print('SEED', s, {k: v[k] for k in ('a', 'b', 'c', 'b_CQ', 'U0_whole_improves')}, 'd', v['d']['PASS'], 'gap', round(v['gap'], 5), 'gain', round(v['gain'], 4),
              json.dumps({a: {k: round(x, 4) for k, x in vv.items()} for a, vv in v['values'].items()}), flush=True)
    for tag, v in rend.items():
        print('RENDER', tag, json.dumps({n: round(x.get('C', x['stationary_all'])['E_H'], 4) for n, x in v.items()}), flush=True)


if __name__ == '__main__':
    main()
