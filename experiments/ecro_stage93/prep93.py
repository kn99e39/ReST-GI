"""WL93 preparation: WL92B rule reconciliation, source-stage targets, held-out E6 qualification.

  python prep93.py recon92b <R> <R92B>                 re-evaluate WL92B's H_interact / outcome under its written frozen rule (no change to WL92B)
  python prep93.py qgen     <R> <R86> <R92A> <split> <start> <count>   source-stage targets q (quad87.qtable) for the WL92A training items
  python prep93.py e6gen    <R> <R86>                  ordered geometry-only E6 candidates (disjoint from all training files), masks
  python prep93.py e6admit  <R> <R86> <k>              E6 candidate admission (WL86 stage-0 incl. Q4 SNR 4 + sharp family, WL92B rule)
  python prep93.py call     <R> <R86> <module.func> <args...>   run a WL92 function with E5 / E6 injected
"""
import glob, json, math, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'ecro_attr92'))
sys.path.insert(0, os.path.join(HERE, '..', 'ecro_own92a'))
sys.path.insert(0, os.path.join(HERE, '..', 'ecro_cf92b'))
sys.path.insert(0, HERE)
DEV = '--dev' in sys.argv
P = json.load(open(os.path.join(HERE, 'protocol', 'ecro_stage93.json')))
R92B_DEFAULT = os.path.expanduser('~/rest_x92b')


def jd(o, fn):
    json.dump(o, open(fn, 'w'), indent=1)


def jl(fn):
    return json.load(open(fn))


def inject(R):
    """E4 (WL92), E5 = WL92B candidate 995 as FX['E5k995'], every recorded E6 candidate as FX['E6k<k>']."""
    import fx92
    fx92.inject(R)
    c5 = [c for c in jl(os.path.join(R, 'results', 'e5_candidates.json'))['geometry_admissible'] if c['k'] == P['e5']['candidate']][0]
    fx92.FX[f"E5k{c5['k']}"] = {'T0': c5['T0'], 'T1': c5['T1'], 'moved': [c5['moved']], 'light_set': c5['light_set']}
    fn = os.path.join(R, 'results', 'e6_candidates.json')
    if os.path.exists(fn):
        for c in jl(fn)['geometry_admissible']:
            fx92.FX[f"E6k{c['k']}"] = {'T0': c['T0'], 'T1': c['T1'], 'moved': [c['moved']], 'light_set': c['light_set']}
    return fx92


# ------------------------------------------------------------------------------------------------ WL92B reconciliation
def recon92b(R, R92B):
    F = jl(os.path.join(R92B, 'results', 'final92b.json'))
    PB = jl(os.path.join(HERE, '..', 'ecro_cf92b', 'protocol', 'ecro_cf92b.json'))
    own = ['own_s8601', 'own_s8602']
    cfs = ['X_L2', 'X_L3', 'X_M1', 'X_M2', 'X_M12', 'X_Q', 'X_SHQ']
    f = lambda t, n, cf: F['cases'][t]['models'][n]['accounted']['C'][cf]
    dom = {cf: all(min(f(t, n, cf) for n in own) >= 0.5 for t in ('E2_T1', 'E2_T0')) for cf in cfs}
    inter = any(all(len([cf for cf in ('X_L2', 'X_L3', 'X_M1', 'X_M2', 'X_M12', 'X_Q') if f(t, n, cf) >= 0.25]) >= 2 for n in own) for t in ('E2_T1', 'E2_T0'))
    written = 'SUPPORTED' if (not any(dom.values()) and inter) else ('NOT SUPPORTED' if not inter else 'INCONCLUSIVE')
    H = F['hypotheses']
    sup = [k for k in ('H_local', 'H_global', 'H_source', 'H_geometry') if H[k]['label'] == 'SUPPORTED']
    domh = [k for k in ('H_local', 'H_global', 'H_source') if H[k]['label'] == 'SUPPORTED']
    if written == 'SUPPORTED' or len(sup) > 1:
        oc = 'C'
    elif len(domh) == 1:
        oc = 'A/B'
    else:
        oc = 'D (code fall-through; the written mapping has no clause for exactly one non-dominance hypothesis SUPPORTED)'
    out = {'written_rule': PB['attribution_rules']['H_interact'], 'implemented_rule': 'no_dom = not (H_local or H_global SUPPORTED)  (only X_M12 and X_L3 checked)',
           'dominant_counterfactuals_both_E2_cases': {k: v for k, v in dom.items()}, 'interaction_clause': inter,
           'H_interact_official': H['H_interact']['label'], 'H_interact_written_rule': written,
           'outcome_official': F['outcome']['label'], 'outcome_written_rule': oc,
           'measurements_affected': False, 'note': 'WL92B files are not modified; this record is the WL93 clarification'}
    jd(out, os.path.join(R, 'results', 'recon92b.json'))
    print('RECON92B', json.dumps({k: out[k] for k in ('H_interact_official', 'H_interact_written_rule', 'outcome_official', 'outcome_written_rule')}), flush=True)


# ------------------------------------------------------------------------------------------------ source-stage targets
def qgen(R, R86, R92A, split, start, count):
    import fx92
    import quad87 as Q
    PT = fx92.PT; E87 = fx92.E87
    start, count = int(start), int(count)
    ex = {n for n, _, _ in jl(os.path.join(R92A, 'results', 'exclusion.json'))['excluded'].get(split, [])}
    out = os.path.join(R, 'data', split + '_q'); os.makedirs(out, exist_ok=True)
    ctx = None; t0 = time.time(); n = 0
    for idx in range(start, start + count):
        name = f'g{idx:05d}.npz'
        src = os.path.join(R86, 'data', split, name)
        if not os.path.exists(src) or name in ex or os.path.exists(os.path.join(out, name)):
            continue
        z = np.load(src)
        geom = json.loads(str(z['geom'])); lights = json.loads(str(z['lights']))
        if ctx is None:
            ctx = PT.Ctx(geom)
        else:
            ctx.set_geom(geom)
        q = np.stack([Q.qtable(ctx, PT.Lights(l), E87['P'], 64 if DEV else E87['Mq'], P['source_stage']['q_seed'] + 4 * idx + li) for li, l in enumerate(lights)])
        np.savez_compressed(os.path.join(out, name + '.tmp.npz'), q=q.astype(np.float16))
        os.replace(os.path.join(out, name + '.tmp.npz'), os.path.join(out, name)); n += 1
        if n % 32 == 0:
            print(split, idx, n, round(time.time() - t0, 1), 's', flush=True)
    print('QGEN_DONE', split, start, n, round(time.time() - t0, 1), 's', flush=True)


# ------------------------------------------------------------------------------------------------ E6
def e6gen(R, R86):
    import fx92
    fx92.inject(R)
    import masks86
    S, FX = fx92.S, fx92.FX
    E = P['e6']
    tr = {'A': [], 'B': []}
    for sp in ('train', 'val', 'trainhq'):
        for fn in sorted(glob.glob(os.path.join(R86, 'data', sp, 'g*.npz'))):
            g = json.loads(str(np.load(fn)['geom']))
            for b in 'AB':
                tr[b].append(g[b]['c'])
    tr = {b: np.array(v) for b, v in tr.items()}
    disjoint, admissible = [], []
    nmax = 400 if DEV else E['max_candidates']
    out_m = os.path.join(R, 'results', 'masks'); os.makedirs(out_m, exist_ok=True)
    t0 = time.time(); k = -1
    for k in range(nmax):
        rng = np.random.default_rng([E['rng_seed'], k])
        g0 = S.sample_geom(rng)
        m = 'A' if rng.random() < 0.5 else 'B'
        g1 = json.loads(json.dumps(g0)); found = False
        for _ in range(200):
            h = g0[m]['h']
            c = [rng.uniform(-1.5, 1.5), rng.uniform(-1.5, 1.5), rng.uniform(h[2] + 0.02, 2.9 - h[2])] if m == 'A' else [rng.uniform(-1.5, 1.5), rng.uniform(-1.5, 1.5), h[2]]
            if np.linalg.norm(np.array(c) - np.array(g0[m]['c'])) < 0.6:
                continue
            g1[m] = {'c': [float(x) for x in c], 'h': h, 'yaw': float(rng.uniform(0, math.pi / 2))}
            if S.geom_ok(g1):
                found = True
                break
        if not found:
            continue
        ls = 'L1' if k % 2 == 0 else 'L2'
        lpos = FX['light_sets'][ls][0]['pos']
        dmin = min(float(np.linalg.norm(tr[b] - np.array(g[b]['c']), axis=1).min()) for g in (g0, g1) for b in 'AB')
        if dmin < (0.0 if DEV else 0.15) or not (S.light_ok(lpos, g0) and S.light_ok(lpos, g1)):
            continue
        cand = {'k': k, 'T0': g0, 'T1': g1, 'moved': m, 'light_set': ls, 'min_box_distance_to_training_m': dmin}
        disjoint.append(cand)
        FX['E6c'] = {'T0': g0, 'T1': g1, 'moved': [m], 'light_set': ls}
        mk, aux = masks86.compute('E6c')
        cnt = {kk: int(v.sum()) for kk, v in mk.items()}
        cand['counts'] = cnt
        rule = all(cnt[r] >= 1000 for r in ('A', 'C', 'D', 'E')) and cnt['C_footprint'] >= 1000
        cand['geometry_admissible'] = bool(rule)
        print('E6 candidate', k, m, ls, round(dmin, 3), json.dumps({r: cnt[r] for r in ('A', 'C', 'D', 'E', 'C_footprint', 'C_mirror')}), rule, flush=True)
        if rule:
            admissible.append(cand)
            np.savez_compressed(os.path.join(out_m, f'E6k{k}.npz'), **mk, **aux)
            for src in ('LMIX', 'LSPOT'):
                ms, auxs = masks86.compute('E6c', ls=src)
                np.savez_compressed(os.path.join(out_m, f'E6k{k}_{src}.npz'), **ms, **auxs)
            if len(admissible) >= E['reference_attempts']:
                break
    jd({'rule': E, 'generated_up_to_k': k, 'disjoint': disjoint, 'geometry_admissible': admissible, 'seconds': time.time() - t0,
        'note': 'geometry / visibility only; no radiance of any candidate exists at this point'}, os.path.join(R, 'results', 'e6_candidates.json'))
    print('E6GEN', 'disjoint', len(disjoint), 'admissible', [c['k'] for c in admissible], flush=True)


def e6admit(R, R86, k):
    """WL92B admission rule (cf92b.e5admit) for an E6 candidate: the same code path with the configuration name E6k<k>."""
    import cf92b
    import analyze86 as A
    fx92 = inject(R)
    import analyze92 as AN
    G0 = A.G0
    e = f'E6k{k}'
    M = AN.mask_of(R, R86, f'{e}_T1')
    p0, p1 = A.load(R, f'REF_P_{e}_T0'), A.load(R, f'REF_P_{e}_T1')
    r1 = A.load(R, f'REF_ER_{e}_T1'); r0 = A.load(R, f'REF_ER_{e}_T0')
    Y = A.Y
    d, da, db = A.halves(Y(p1['tot']) - Y(p0['tot']))
    dh_, dha, dhb = A.halves(Y(p1['h']) - Y(p0['h']))
    t0 = Y(p0['tot']).mean(0); t1, t1a, t1b = A.halves(p1['tot']); e1r, e1a, e1b = A.halves(r1['tot'])
    Rg = {}
    for kk in A.REG:
        if kk not in M or M[kk].sum() < 30:
            Rg[kk] = {'pixels': int(M[kk].sum()) if kk in M else 0}
            continue
        m = M[kk]; b = max(float(t0[m].mean()), 1e-30)
        Rg[kk] = {'pixels': int(m.sum()), 'abs_change_rel': float(np.abs(d[m]).mean() / b), 'abs_change_floor_rel': float(0.5 * np.abs(da[m] - db[m]).mean() / b),
                  'absH_change_rel': float(np.abs(dh_[m]).mean() / b), 'absH_change_floor_rel': float(0.5 * np.abs(dha[m] - dhb[m]).mean() / b),
                  'H_share_T1': float(p1['h'].mean(0)[m].sum() / t1[m].sum()), 'uncertainty_P_T1': 0.5 * A.rl(t1a, t1b, m), 'uncertainty_ER_T1': 0.5 * A.rl(e1a, e1b, m)}
    mC = M['C']; sm_recv = M['obj_panel'] | M['obj_sphere']
    dc = np.abs((r1['c2'] - r0['c2']).mean(0)) + np.abs((r1['ch'] - r0['ch']).mean(0))
    allc = dc.sum(-1)
    sm = float((allc[mC & sm_recv].sum() + dc[..., :2].sum(-1)[mC & ~sm_recv].sum()) / max(allc[mC].sum(), 1e-30)) if mC.sum() else 0.0
    g = lambda kk, f: Rg.get(kk, {}).get(f, 0.0)
    sig = lambda kk, f='abs_change_rel', fl_='abs_change_floor_rel': g(kk, f) >= G0['snr'] * g(kk, fl_)
    q = {'Q1_A_direct_change': {'value': g('A', 'abs_change_rel'), 'PASS': bool(g('A', 'abs_change_rel') >= G0['A_min'] and sig('A'))},
         'Q2_B_indirect_change': {'value': g('Bset', 'abs_change_rel'), 'PASS': bool(g('Bset', 'abs_change_rel') >= G0['B_min'] and sig('Bset'))},
         'Q3_C_sharp_moderate': {'value': g('C', 'abs_change_rel'), 'sm_share': sm, 'pixels': Rg['C']['pixels'],
                                 'PASS': bool(Rg['C']['pixels'] >= G0['min_pixels'] and g('C', 'abs_change_rel') >= G0['C_min'] and sig('C') and sm >= G0['C_sm_share_min'])}}
    ratio = g('D', 'absH_change_rel') / max(g('D', 'abs_change_rel'), 1e-30)
    q['Q4_D_broad_multibounce'] = {'value': g('D', 'abs_change_rel'), 'absH_over_abs': ratio, 'absH': g('D', 'absH_change_rel'), 'absH_floor': g('D', 'absH_change_floor_rel'),
                                   'PASS': bool(g('D', 'abs_change_rel') >= G0['D_min'] and sig('D') and ratio >= G0['D_H_ratio_min'] and sig('D', 'absH_change_rel', 'absH_change_floor_rel'))}
    q['Q6_H_share'] = {'value': g('stationary_all', 'H_share_T1'), 'PASS': bool(g('stationary_all', 'H_share_T1') >= G0['H_share_min'])}
    un = max(g('stationary_all', 'uncertainty_P_T1'), g('stationary_all', 'uncertainty_ER_T1'))
    q['Q7_uncertainty'] = {'value': un, 'PASS': bool(un <= G0['uncertainty_max'])}
    D = np.load(os.path.join(R, 'results', 'fields', f'D_{e}_T1.npz')); Wz = np.load(os.path.join(R, 'results', 'fields', f'W_{e}_T1.npz'))
    sh = {}
    if 'C' in Wz.files:
        W = Wz['C'].astype(np.float64).reshape(-1, fx92.S.NB, 3)
        for hh in 'ab':
            tot = sum(D[f'{p}_{hh}'].astype(np.float64) for p in ('e2_p', 'e2_o', 'e3_p', 'e3_o'))
            sh[hh] = float((W * D[f'e2_p_{hh}']).sum() / max((W * tot).sum(), 1e-30))
        share = 0.5 * (sh['a'] + sh['b']); noise = 0.5 * abs(sh['a'] - sh['b'])
    else:
        share, noise = 0.0, 1.0
    q['SHARP_FAMILY'] = {'share': share, 'half_split_noise': noise, 'PASS': bool(share >= P['e6']['sharp_family_min'] and share >= 4.0 * noise)}
    adm = bool(all(v['PASS'] for v in q.values()))
    jd({'candidate': k, 'regions': Rg, 'qualification': q, 'ADMITTED': adm}, os.path.join(R, 'results', f'e6_admission_k{k}.json'))
    print('E6ADMIT', k, 'ADMITTED' if adm else 'NOT_ADMITTED', json.dumps({kk: v['PASS'] for kk, v in q.items()}), round(share, 4), round(noise, 4), flush=True)


def call(R, R86, fn, *args):
    inject(R)
    mod, f = fn.split('.')
    getattr(__import__(mod), f)(R, R86, *args)


if __name__ == '__main__':
    import fx92
    fx92.dr.set_flag(fx92.dr.JitFlag.KernelHistory, True)
    cmd, R = sys.argv[1], sys.argv[2]
    a = [x for x in sys.argv[3:] if not x.startswith('--')]
    for d in ('fields', 'tables', 'renders', 'masks', 'lists'):
        os.makedirs(os.path.join(R, 'results', d), exist_ok=True)
    {'recon92b': recon92b, 'qgen': qgen, 'e6gen': e6gen, 'e6admit': e6admit, 'call': call}[cmd](R, *a)
    os._exit(0)
