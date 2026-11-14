"""WL93D analysis by the prospective rules (protocol ecro_local93d). numpy only.
  python analyze93d.py <R> <R86> <R92A> <R92B> <R93A>   -> results/final93d.json ; prints LABELS ...
"""
import glob, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for d in ('ecro86', 'ecro_attr92'):
    sys.path.insert(0, os.path.join(HERE, '..', d))
import analyze92 as AN  # noqa: E402
import analyze86 as A  # noqa: E402

P = json.load(open(os.path.join(HERE, 'protocol', 'ecro_local93d.json')))
FAMS = ['own', 'T0']; SEEDS = [8601, 8602]
ARMS = ['C0', 'C1_src', 'C1_rcv', 'C1_both', 'C2_src', 'C2_rcv', 'C2_both']
REGS = ['stationary_all', 'C', 'D', 'obj_panel']


def jl(fn):
    return json.load(open(fn)) if os.path.exists(fn) else None


def main():
    R, R86, R92A, R92B, R93A = sys.argv[1:6]
    out = {k: jl(os.path.join(R, 'results', f)) for k, f in (('ckpt', 'ckpt93d.json'), ('tests', 'tests93d.json'), ('functional', 'functional93d.json'),
                                                               ('contrasts', 'contrasts93d.json'), ('e6_choice', 'e6_choice.json'))}
    acc = jl(os.path.join(R, 'results', 'account93d.json'))
    out['account_summary'] = acc['summary'] if acc else None
    out['e6_admissions'] = [jl(f) for f in sorted(glob.glob(os.path.join(R, 'results', 'e6_admission_k*.json')))]
    stops = []
    for k in ('ckpt', 'tests'):
        if not (out[k] and out[k].get('PASS')):
            stops.append(f'STOP_{k.upper()}_FAILED')
    e6 = (out['e6_choice'] or {}).get('chosen')
    tconds = ['E2', 'E5k995'] + ([f'E6k{e6}'] if e6 is not None else [])

    def ref(tag):
        for root in (R, R92B, R92A, R86):
            if glob.glob(os.path.join(root, 'results', 'renders', f'REF_ER_{tag}_r*.npz')):
                return A.load(root, f'REF_ER_{tag}')
    rend, dH = {}, {}
    for e in tconds:
        for t in ('T0', 'T1'):
            tag = f'{e}_{t}'; rf = ref(tag)
            if rf is None:
                continue
            Mk = AN.mask_of(R, R86, tag)
            for fam in FAMS:
                for sd in SEEDS:
                    for arm in ARMS:
                        x = AN.hl(R, f'I93D_{fam}_s{sd}_{arm}_{tag}')
                        if x is not None:
                            rend.setdefault(tag, {})[f'{fam}_s{sd}:{arm}'] = AN.eh(rf, x, Mk, REGS)
        r0, r1 = ref(f'{e}_T0'), ref(f'{e}_T1')
        if r0 is None or r1 is None:
            continue
        Mk = AN.mask_of(R, R86, f'{e}_T1')
        for fam in FAMS:
            for sd in SEEDS:
                for arm in ARMS:
                    t0arm = arm if arm.startswith('C2') or arm == 'C0' else 'C0'
                    x0, x1 = AN.hl(R, f'I93D_{fam}_s{sd}_{t0arm}_{e}_T0'), AN.hl(R, f'I93D_{fam}_s{sd}_{arm}_{e}_T1')
                    if x0 is not None and x1 is not None:
                        dH.setdefault(e, {})[f'{fam}_s{sd}:{arm}'] = AN.dh(r0, r1, x0, x1, Mk)
    out['renders'] = rend; out['renders_dH'] = dH
    # ---- T_REPRO (render): own deployed (C0) on E2 T1 / T0 vs WL93A I93A_own92a renders
    rr = {}
    for sd in SEEDS:
        for t in ('E2_T0', 'E2_T1'):
            a_, b_ = AN.hl(R, f'I93D_own_s{sd}_C0_{t}'), AN.hl(R93A, f'I93A_own92a_s{sd}_{t}')
            if a_ is not None and b_ is not None:
                Mk = AN.mask_of(R, R86, t); rf = ref(t)
                rr[f's{sd}:{t}'] = abs(AN.eh(rf, a_, Mk, ['C'])['C']['E_H'] - AN.eh(rf, b_, Mk, ['C'])['C']['E_H'])
    out['render_repro'] = {'cases': rr, 'PASS': bool(rr and all(v <= 1e-6 for v in rr.values()))}
    if not out['render_repro']['PASS']:
        stops.append('STOP_RENDER_REPRO_FAILED')
    # ---- criteria
    F = out['functional']; CON = out['contrasts'] or {}
    crit = {}
    for fam in FAMS:
        c = {}
        for sd in SEEDS:
            f = F['families'][fam] if F else None
            if f is None:
                continue
            W = {u: f['S1'][u]['per_seed'][str(sd)] if str(sd) in f['S1'][u]['per_seed'] else f['S1'][u]['per_seed'][sd] for u in ('src', 'rcv')}
            L = {u: f['LAM'][u]['per_seed'][str(sd)] if str(sd) in f['LAM'][u]['per_seed'] else f['LAM'][u]['per_seed'][sd] for u in ('src', 'rcv')}
            g1 = all(W[u]['D_geo'] >= 0.10 and W[u]['D_geo'] >= 2 * L[u]['D_geo'] for u in ('src', 'rcv'))
            g2 = all(W[u]['S_geo'] >= 0.25 for u in ('src', 'rcv'))
            rlab = all(W[u]['E_phys'] >= 0.30 and W[u]['S_geo'] < 0.25 for u in ('src', 'rcv'))
            # G3: rendered region-C |C1_both - C0| on T1 conditions + validation single |diff| >= 2 SE
            cells = []
            for e in tconds:
                x = rend.get(f'{e}_T1', {})
                a_, b_ = x.get(f'{fam}_s{sd}:C1_both', {}).get('C'), x.get(f'{fam}_s{sd}:C0', {}).get('C')
                if a_ and b_:
                    mg = max(a_['noise_bound'], b_['noise_bound'], 0.005)
                    cells.append({'cond': e, 'C1_both': a_['E_H'], 'C0': b_['E_H'], 'margin': mg, 'differs': bool(abs(a_['E_H'] - b_['E_H']) >= mg)})
            kv = CON.get(f'{fam}_s{sd}:C1_both-C0:single_l1')
            g3_val = bool(kv and abs(kv['diff']) >= 2 * kv['se'])
            g3 = sum(x['differs'] for x in cells) >= 2 and g3_val
            # co-adaptation
            ccells = []
            for e in tconds:
                x = rend.get(f'{e}_T1', {})
                a_, b_ = x.get(f'{fam}_s{sd}:C2_both', {}).get('C'), x.get(f'{fam}_s{sd}:C0', {}).get('C')
                if a_ and b_:
                    mg = max(a_['noise_bound'], b_['noise_bound'], 0.005)
                    ccells.append({'cond': e, 'C2_both': a_['E_H'], 'C0': b_['E_H'], 'better': bool(a_['E_H'] <= b_['E_H'] - mg)})
            k2 = CON.get(f'{fam}_s{sd}:C2_both-C0:single_l1')
            v_not_better = bool(k2 and not (k2['diff'] <= -2 * k2['se']))
            ds = {a: CON.get(f'{fam}_s{sd}:{a}-C0:single_l1', {}).get('diff') for a in ('C2_src', 'C2_rcv', 'C2_both')}
            inter = None if None in ds.values() else bool(abs(ds['C2_both'] - (ds['C2_src'] + ds['C2_rcv'])) > max(abs(ds['C2_src']), abs(ds['C2_rcv'])))
            c[sd] = {'workload_S1': W, 'workload_LAM': L, 'G1': bool(g1), 'G2': bool(g2), 'R_cond': bool(rlab), 'G3': bool(g3), 'G3_cells': cells, 'G3_val': kv,
                     'C2_cells': ccells, 'C2_val': k2, 'C2_val_not_better': v_not_better, 'C2_interaction': inter, 'C2_deltas': ds,
                     'C1_deltas': {a: CON.get(f'{fam}_s{sd}:{a}-C0:single_l1', {}).get('diff') for a in ('C1_src', 'C1_rcv', 'C1_both')}}
        if len(c) == 2:
            al = lambda k: all(c[s][k] for s in SEEDS)
            labels = []
            if al('G1') and al('G2') and al('G3'):
                labels.append('GEOMETRY_SPECIALIZATION_SUPPORTED')
            if al('R_cond') and not al('G3'):
                labels.append('REPRESENTATION_OR_OPTIMIZATION_DOMINANT')
            cells_all = [x for s in SEEDS for x in c[s]['C2_cells']]
            co_a = bool(cells_all) and sum(not x['better'] for x in cells_all) >= len(cells_all) / 2
            co_b = all(c[s]['C2_val_not_better'] for s in SEEDS)
            co_c = all(c[s]['C2_interaction'] for s in SEEDS)
            if co_a or co_b or co_c:
                labels.append('COADAPTATION_LIMITED')
            c['labels'] = labels; c['coadapt_parts'] = {'renders_not_better_half': co_a, 'val_not_better_both': co_b, 'interaction_both': bool(co_c)}
        crit[fam] = c
    out['criteria'] = crit
    prim = crit.get('own', {}).get('labels', [])
    sec = crit.get('T0', {}).get('labels', [])
    conflict = ('GEOMETRY_SPECIALIZATION_SUPPORTED' in prim and 'REPRESENTATION_OR_OPTIMIZATION_DOMINANT' in sec) or ('REPRESENTATION_OR_OPTIMIZATION_DOMINANT' in prim and 'GEOMETRY_SPECIALIZATION_SUPPORTED' in sec)
    head = ['INCONCLUSIVE'] if (stops or conflict or not prim) else prim
    out['verdict'] = {'headline': head, 'primary_family_own': prim, 'secondary_family_T0': sec, 'stops': stops, 'family_conflict': conflict,
                      'e6_admitted': e6 is not None, 'note': 'labels evaluated independently; multiple mechanisms allowed; frozen-checkpoint attribution only'}
    json.dump(out, open(os.path.join(R, 'results', 'final93d.json'), 'w'), indent=1, default=str)
    print('LABELS', head, 'own', prim, 'T0', sec, 'stops', stops, 'e6', e6, flush=True)
    for fam in FAMS:
        for s in SEEDS:
            x = crit.get(fam, {}).get(s)
            if x:
                print(fam, s, {k: x[k] for k in ('G1', 'G2', 'R_cond', 'G3', 'C2_val_not_better', 'C2_interaction')}, flush=True)
    for tag, v in rend.items():
        print('RENDER', tag, json.dumps({k: round(m['C']['E_H'], 4) for k, m in v.items() if 'C' in m}), flush=True)


if __name__ == '__main__':
    main()
