"""WL87 analysis and prospective rules (protocol stage1 / stage2 / stage3 / verdict_mapping).

  python analyze87.py e3qualify <R87>              E3 admission (WL86 stage0 criteria on REF-P / REF-ER of E3) -> e3_admission.json
  python analyze87.py stage1 <R87> <R86> <R86A>    explicit operator fidelity (+ controls, field / order level, family reading)
  python analyze87.py stage2 <R87> <R86> <R86A>    state-realization statistics and classification
  python analyze87.py stage3 <R87>                 matched-quality economics from econ87.json
  python analyze87.py verdict <R87>                verdict mapping -> final87.json
E_H(X) = sum |hl_X - h_REF-ER| / sum tot_REF-ER (WL86); signed S(X) = sum (hl_X - h_REF-ER) / sum tot_REF-ER.
"""
import glob, json, math, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'ecro86'))
P87 = json.load(open(os.path.join(HERE, 'protocol', 'gm1_explicit87.json')))
import analyze86 as A  # noqa: E402  (numpy only)
import scene86 as S  # noqa: E402
NB, K = S.NB, S.K
TH = P87['stage1']['thresholds_wl86']
ARMS = ['EXP', 'BINQ', 'ENC', 'RAYSEND', 'C3']
REGS = ['stationary_all', 'A', 'C', 'D', 'E', 'obj_panel', 'obj_sphere', 'obj_walls', 'obj_floor']
FAMILY = [('E2_T1', 'C'), ('E2_T0', 'C'), ('E1_T1', 'obj_panel')]


def jd(o, fn):
    json.dump(o, open(fn, 'w'), indent=1, default=lambda x: x.item() if hasattr(x, 'item') else str(x))


def load_hl(root, pat):
    fns = sorted(glob.glob(os.path.join(root, 'results', 'renders', pat + '_r*.npz')))
    assert fns, pat
    return {'hl': np.stack([np.load(f)['hl'].astype(np.float64) for f in fns])}


def edits(R87):
    adm = os.path.join(R87, 'results', 'e3_admission.json')
    e3 = os.path.exists(adm) and (json.load(open(adm)).get('ADMITTED') or bool(os.environ.get('WL87_DEV_FORCE_E3')))      # override: dev smoke only
    return ['E1', 'E2'] + (['E3'] if e3 else [])


def root_of(e, R87, R86):
    return R87 if e == 'E3' else R86


def signed(ref, ec, m):
    return float((ec['hl'].mean(0)[m] - ref['h'].mean(0)[m]).sum() / max(ref['tot'].mean(0)[m].sum(), 1e-30))


# ------------------------------------------------------------------------------------------------ E3 admission
def e3qualify(R87):
    G0 = A.G0
    e = 'E3'
    M, _ = A.masks(R87, e)
    p0, p1 = A.load(R87, f'REF_P_{e}_T0'), A.load(R87, f'REF_P_{e}_T1')
    r1 = A.load(R87, f'REF_ER_{e}_T1'); r0 = A.load(R87, f'REF_ER_{e}_T0')
    Y = A.Y
    d, da, db = A.halves(Y(p1['tot']) - Y(p0['tot']))
    dh, dha, dhb = A.halves(Y(p1['h']) - Y(p0['h']))
    t0 = Y(p0['tot']).mean(0); t1, t1a, t1b = A.halves(p1['tot']); e1r, e1a, e1b = A.halves(r1['tot'])
    Rg = {}
    for k in A.REG:
        m = M[k]
        if m.sum() < 30:
            Rg[k] = {'pixels': int(m.sum())}
            continue
        b = max(float(t0[m].mean()), 1e-30)
        Rg[k] = {'pixels': int(m.sum()), 'abs_change_rel': float(np.abs(d[m]).mean() / b), 'abs_change_floor_rel': float(0.5 * np.abs(da[m] - db[m]).mean() / b),
                 'absH_change_rel': float(np.abs(dh[m]).mean() / b), 'absH_change_floor_rel': float(0.5 * np.abs(dha[m] - dhb[m]).mean() / b),
                 'H_share_T1': float(p1['h'].mean(0)[m].sum() / t1[m].sum()), 'uncertainty_P_T1': 0.5 * A.rl(t1a, t1b, m),
                 'uncertainty_ER_T1': 0.5 * A.rl(e1a, e1b, m), 'g_ER_vs_P_T1': A.rl(e1r, t1, m)}
    mC = M['C']; sm_recv = M['obj_panel'] | M['obj_sphere']
    dc = np.abs((r1['c2'] - r0['c2']).mean(0)) + np.abs((r1['ch'] - r0['ch']).mean(0))
    allc = dc.sum(-1)
    sm = float((allc[mC & sm_recv].sum() + dc[..., :2].sum(-1)[mC & ~sm_recv].sum()) / max(allc[mC].sum(), 1e-30)) if mC.sum() else 0.0
    g = lambda k, f: Rg[k].get(f, 0.0)
    sig = lambda k, f='abs_change_rel', fl_='abs_change_floor_rel': g(k, f) >= G0['snr'] * g(k, fl_)
    q = {}
    q['Q1_A_direct_change'] = {'value': g('A', 'abs_change_rel'), 'PASS': bool(g('A', 'abs_change_rel') >= G0['A_min'] and sig('A'))}
    q['Q2_B_indirect_change'] = {'value': g('Bset', 'abs_change_rel'), 'PASS': bool(g('Bset', 'abs_change_rel') >= G0['B_min'] and sig('Bset'))}
    q['Q3_C_sharp_moderate'] = {'value': g('C', 'abs_change_rel'), 'sm_share': sm, 'pixels': Rg['C']['pixels'],
                                'PASS': bool(Rg['C']['pixels'] >= G0['min_pixels'] and g('C', 'abs_change_rel') >= G0['C_min'] and sig('C') and sm >= G0['C_sm_share_min'])}
    ratio = g('D', 'absH_change_rel') / max(g('D', 'abs_change_rel'), 1e-30)
    q['Q4_D_broad_multibounce'] = {'value': g('D', 'abs_change_rel'), 'absH_over_abs': ratio,
                                   'PASS': bool(g('D', 'abs_change_rel') >= G0['D_min'] and sig('D') and ratio >= G0['D_H_ratio_min'] and sig('D', 'absH_change_rel', 'absH_change_floor_rel'))}
    q['Q5_E_unaffected'] = {'pixels': Rg['E']['pixels'], 'ratio_to_D': g('E', 'abs_change_rel') / max(g('D', 'abs_change_rel'), 1e-30)}
    q['Q6_H_share'] = {'value': g('stationary_all', 'H_share_T1'), 'PASS': bool(g('stationary_all', 'H_share_T1') >= G0['H_share_min'])}
    un = max(g('stationary_all', 'uncertainty_P_T1'), g('stationary_all', 'uncertainty_ER_T1'))
    q['Q7_uncertainty'] = {'value': un, 'PASS': bool(un <= G0['uncertainty_max'])}
    adm = bool(all(v['PASS'] for k, v in q.items() if 'PASS' in v))
    out = {'regions': Rg, 'qualification': q, 'ADMITTED': adm, 'g_stationary': g('stationary_all', 'g_ER_vs_P_T1')}
    jd(out, os.path.join(R87, 'results', 'e3_admission.json'))
    print('E3', 'ADMITTED' if adm else 'NOT_ADMITTED', json.dumps({k: v.get('PASS') for k, v in q.items()}), flush=True)


# ------------------------------------------------------------------------------------------------ Stage 1
def field_cmp(Xt, fa, fb, G):
    T = 0.5 * (fa + fb)
    w = (G.PAREA[:, None] * ((np.arange(NB) // S.NPHI + 0.5) / S.NZ)[None])[..., None]
    cls = {'all': np.ones(G.N, bool), 'floor': G.PMAT == 0, 'panel': G.PMAT == 1, 'sphere': G.PMAT == 2, 'walls': G.PMAT == 3, 'boxes': G.PMAT >= 4}
    o = {}
    for k, m in cls.items():
        o[k] = {'rel_l1': float(np.abs(Xt[m] - T[m]).sum() / max(T[m].sum(), 1e-30)), 'truth_noise': float(0.5 * np.abs(fa[m] - fb[m]).sum() / max(T[m].sum(), 1e-30)),
                'flux_ratio': float((Xt[m] * w[m]).sum() / max((T[m] * w[m]).sum(), 1e-30))}
    return o


def family(R86A, R86, cfg, rk, tabs, G):
    """WL86A amendment-2 reading of the light -> panel -> Lambert family for tables X [N, NB, 3] (Lambert receivers)."""
    lam = np.isin(G.PMAT, [3, 4, 5])
    W = np.load(os.path.join(R86A, 'stageB', f'importance_{cfg}.npz'))[rk].astype(np.float64).reshape(G.N, NB, 3)
    zo = np.load(os.path.join(R86, 'results', 'fields', f'F_{cfg}.npz')); Bo = 0.5 * (zo['fa'].astype(np.float64) + zo['fb'])
    full = (W * Bo).sum()
    Wp = W.sum(1)
    sp = np.load(os.path.join(R86A, 'results', 'fields', f'D_SPLIT_{cfg}.npz'))
    Ta = sp['e2_p_a'].astype(np.float64).mean(1); Tb = sp['e2_p_b'].astype(np.float64).mean(1); T = 0.5 * (Ta + Tb)
    rd = lambda Xp: float((Wp[lam] * Xp[lam]).sum() / full)
    m = lam & (Wp.sum(1) > 0.002 * Wp.sum())
    tt = T[m].sum(1)
    out = {'truth_read': rd(T), 'truth_halves': [rd(Ta), rd(Tb)], 'important_patches': int(m.sum()),
           'truth_patch_noise_rel_rms': float(math.sqrt(((Ta[m].sum(1) - Tb[m].sum(1)) ** 2).sum() / 4 / max((tt ** 2).sum(), 1e-30)))}
    for name, Xt in tabs.items():
        Xp = Xt.mean(1)
        x = Xp[m].sum(1)
        out[name] = {'read': rd(Xp), 'read_over_truth': rd(Xp) / max(rd(T), 1e-30),
                     'patch_rel_rms': float(math.sqrt(((x - tt) ** 2).sum() / max((tt ** 2).sum(), 1e-30))),
                     'patch_corr': float(np.corrcoef(x, tt)[0, 1]) if m.sum() > 2 else None}
    return out, m, W, Bo


def stage1(R87, R86, R86A):
    out = {}
    E = edits(R87)
    for e in E:
        root = root_of(e, R87, R86)
        M, _ = A.masks(root, e)
        o = {'fidelity': {}, 'signed': {}, 'dH': {}, 'field': {}, 'orders': {}}
        refs = {t: A.load(root, f'REF_ER_{e}_{t}') for t in ('T0', 'T1')}
        rends = {}
        for t in ('T0', 'T1'):
            cfg = f'{e}_{t}'
            rends[t] = {arm: load_hl(R87, f'X87_{arm}_{cfg}') for arm in ARMS}
            for sd in ('ecro_s8601', 'ecro_s8602'):
                rends[t][f'ECRO_{sd}'] = load_hl(R86, f'EC_{sd}_ecro_{cfg}') if e != 'E3' else load_hl(R87, f'EC87_{sd}_ecro_{cfg}')
            rends[t]['ORACLE'] = load_hl(R86, f'ORA_{cfg}') if e != 'E3' else load_hl(R87, f'ORA87_{cfg}')
            if e == 'E3' or t == 'T1':
                for sd in ('ecro_s8601', 'ecro_s8602'):
                    if t == 'T1':
                        rends[t][f'STALE_{sd}'] = load_hl(R86, f'EC_{sd}_stale_{cfg}') if e != 'E3' else load_hl(R87, f'EC87_{sd}_stale_{cfg}')
            o['fidelity'][t] = {arm: A.eh_table(None, refs[t], ec, M) for arm, ec in rends[t].items()}
            o['signed'][t] = {arm: {k: signed(refs[t], ec, M[k]) for k in REGS if M[k].sum() >= 30} for arm, ec in rends[t].items()}
            # field level against the oracle field
            fz = np.load(os.path.join(root, 'results', 'fields', f'F_{cfg}.npz' if e != 'E3' else f'F87_{cfg}.npz'))
            fa, fb = fz['fa'].astype(np.float64), fz['fb'].astype(np.float64)
            Gc = S.Geom(S.FX[e][t]) if e != 'E3' else _geom_e3(R87, t)
            xz = np.load(os.path.join(R87, 'results', 'tables', f'X_{cfg}.npz'))
            o['field'][t] = {arm: field_cmp(xz[arm].astype(np.float64), fa, fb, Gc) for arm in ARMS}
            for sd in ('ecro_s8601', 'ecro_s8602'):
                fn = os.path.join(R86, 'results', 'renders', f'EC_{sd}_ecro_{cfg}_r0.npz') if e != 'E3' else os.path.join(R87, 'results', 'renders', f'EC87_{sd}_ecro_{cfg}_r0.npz')
                o['field'][t][f'ECRO_{sd}'] = field_cmp(np.load(fn)['B'].astype(np.float64), fa, fb, Gc)
            # per order
            oz = os.path.join(R87, 'results', 'fields', f'ORD_{cfg}.npz')
            if os.path.exists(oz):
                oo = np.load(oz)
                o['orders'][t] = {}
                for arm in ('EXP', 'C3'):
                    o['orders'][t][arm] = [field_cmp(xz[f'{arm}_L{j + 2}'].astype(np.float64), oo['a'][j].astype(np.float64), oo['b'][j].astype(np.float64), Gc)['all']
                                           for j in range(4)]
                o['orders'][t]['truth_sum_vs_oracle'] = field_cmp(0.5 * (oo['a'].sum(0) + oo['b'].sum(0)).astype(np.float64), fa, fb, Gc)['all']
        for arm in ARMS + ['ECRO_ecro_s8601', 'ECRO_ecro_s8602', 'ORACLE']:
            o['dH'][arm] = A.dh_table(refs['T0'], refs['T1'], rends['T0'][arm], rends['T1'][arm], M)
        out[e] = o
    # held-out sources on E1 T1
    src = {}
    for ls in ('LMIX', 'LSPOT'):
        Mm, _ = A.masks(R86, f'E1_{ls}')
        ref = A.load(R86, f'REF_ER_E1_T1_{ls}')
        src[ls] = {arm: A.eh_table(None, ref, load_hl(R87, f'X87_{arm}_E1_T1_{ls}'), Mm) for arm in ARMS}
        for sd in ('ecro_s8601', 'ecro_s8602'):
            src[ls][f'ECRO_{sd}'] = A.eh_table(None, ref, load_hl(R86, f'EC_{sd}_ecro_E1_T1_{ls}'), Mm)
    out['sources'] = src
    # family reading (WL86A cases), deployed state
    fam = {}
    for cfg, rk in FAMILY:
        e, t = cfg.split('_')
        G = S.Geom(S.FX[e][t])
        rz = np.load(os.path.join(R87, 'results', 'tables', f'R_{cfg}.npz'))
        xz = np.load(os.path.join(R87, 'results', 'tables', f'X_{cfg}.npz'))
        tabs = {'BIN_deployed': rz['BIN_s86001'], 'RAY_deployed': rz['RAY_s86001'], 'BIN_pool': np.mean([rz[k] for k in rz.files if k.startswith('BIN_s8720') or k.startswith('BIN_s8721')], 0),
                'RAY_pool': np.mean([rz[k] for k in rz.files if k.startswith('RAY_s8720') or k.startswith('RAY_s8721')], 0)}
        f, m, W, Bo = family(R86A, R86, cfg, rk, {k: v.astype(np.float64) for k, v in tabs.items()}, G)
        # importance-weighted signed table error of each arm (share of total radiance of the region)
        root = R86
        ref = A.load(root, f'REF_ER_{cfg}')
        Mm, _ = A.masks(root, e)
        totr = ref['tot'].mean(0)[Mm[rk]].sum()
        f['P_signed'] = {arm: float((W * (xz[arm].astype(np.float64) - Bo)).sum() / totr) for arm in ARMS}
        for sd in ('ecro_s8601', 'ecro_s8602'):
            Bm = np.load(os.path.join(R86, 'results', 'renders', f'EC_{sd}_ecro_{cfg}_r0.npz'))['B'].astype(np.float64)
            f['P_signed'][f'ECRO_{sd}'] = float((W * (Bm - Bo)).sum() / totr)
        fam[f'{cfg}:{rk}'] = f
    out['family'] = fam
    # classification of EXP (protocol stage1.classification) on the admitted edits
    gates = {}
    for arm in ARMS + ['ECRO_ecro_s8601', 'ECRO_ecro_s8602', 'ORACLE']:
        f1 = f2 = True; f3 = {}; fails = []
        for e in E:
            for t in ('T0', 'T1'):
                T = out[e]['fidelity'][t][arm]
                v = T['stationary_all']
                ok = v['E_H'] <= TH['EH_stationary_max'] and v['capture'] <= TH['capture_max']
                f1 &= ok
                if not ok:
                    fails.append({'where': f'{e}_{t}:stationary_all', 'excess': max(v['E_H'] - TH['EH_stationary_max'], (v['capture'] - TH['capture_max']) * v['H_share']),
                                  'noise_bound': v['noise_bound']})
            for k in ('A', 'C', 'D'):
                if k in out[e]['fidelity']['T1'][arm]:
                    v = out[e]['fidelity']['T1'][arm][k]
                    ok = v['E_H'] <= TH['EH_region_max']
                    f2 &= ok
                    if not ok:
                        fails.append({'where': f'{e}_T1:{k}', 'excess': v['E_H'] - TH['EH_region_max'], 'noise_bound': v['noise_bound']})
            d = out[e]['dH'][arm]['D']
            f3[e] = 'INDETERMINATE' if not d['resolvable'] else ('PASS' if (TH['dH_slope'][0] <= d['slope'] <= TH['dH_slope'][1] and d['corr'] >= TH['dH_corr_min']) else 'FAIL')
        f4 = all(src[ls][arm]['stationary_all']['E_H'] <= TH['EH_unseen_source_max'] for ls in src) if arm in src['LMIX'] else None
        gates[arm] = {'F1': bool(f1), 'F2': bool(f2), 'F3': f3, 'F4': f4, 'failing': fails,
                      'all_failures_within_1.5_noise': bool(fails) and all(x['excess'] <= 1.5 * x['noise_bound'] for x in fails)}
    out['gates'] = gates
    meta = {}
    for e in E:
        for t in ('T0', 'T1'):
            meta[f'{e}_{t}'] = json.load(open(os.path.join(R87, 'results', 'tables', f'X_{e}_{t}.json')))
    out['tables_meta'] = meta
    g = gates['EXP']
    finite = all(m['finite'] for m in meta.values()) and all(min(m['min'].values()) >= 0 for m in meta.values())
    fluxok = all(0.8 <= out[e]['field'][t]['EXP']['all']['flux_ratio'] <= 1.2 for e in E for t in ('T0', 'T1'))
    tests = json.load(open(os.path.join(R87, 'results', 'tests', 'tests87.json')))
    if not (tests['PASS'] and finite and fluxok):
        cls = 'NUMERICALLY_INVALID'
    elif g['F1'] and g['F2'] and all(v != 'FAIL' for v in g['F3'].values()) and g['F4']:
        cls = 'ACCURATE_UNDER_FROZEN_REPRESENTATION'
    elif g['all_failures_within_1.5_noise'] and all(v != 'FAIL' for v in g['F3'].values()) and g['F4']:
        cls = 'INSUFFICIENTLY_RESOLVED'
    else:
        # localisation (protocol): every failing quantity passes for the oracle table (the continuous table is adequate there), and the
        # excess is visible as table discretisation: a per-order field error > 2 x the truth noise, or a C2 / C3 control moving the
        # failing E_H by >= 20 %
        loc = []
        for x in g['failing']:
            cfgk, k = x['where'].split(':'); e, t = cfgk.split('_')
            o_ok = out[e]['fidelity'][t]['ORACLE'][k]['E_H'] <= (TH['EH_stationary_max'] if k == 'stationary_all' else TH['EH_region_max'])
            od = out[e]['orders'].get(t, {}).get('EXP', [])
            order_vis = any(r['rel_l1'] > 2 * r['truth_noise'] for r in od)
            base = out[e]['fidelity'][t]['EXP'][k]['E_H']
            ctrl = any(abs(out[e]['fidelity'][t][c][k]['E_H'] - base) >= 0.2 * base for c in ('BINQ', 'ENC', 'RAYSEND', 'C3'))
            loc.append({'where': x['where'], 'oracle_passes': bool(o_ok), 'order_error_visible': bool(order_vis), 'control_moves_it': bool(ctrl)})
        out['EXP_localisation'] = loc
        cls = 'DISCRETISATION_LIMITED' if all(l['oracle_passes'] and (l['order_error_visible'] or l['control_moves_it']) for l in loc) else 'FAILING_UNLOCALISED'
    out['EXP_classification'] = {'class': cls, 'finite_nonneg': finite, 'flux_within_bound': fluxok, 'tests_pass': tests['PASS']}
    jd(out, os.path.join(R87, 'results', 'stage1_87.json'))
    for e in E:
        print(e, json.dumps({arm: {k: round(out[e]['fidelity']['T1'][arm][k]['E_H'], 4) for k in ('stationary_all', 'A', 'C', 'D', 'obj_panel') if k in out[e]['fidelity']['T1'][arm]}
                             for arm in ['EXP', 'ECRO_ecro_s8601', 'ORACLE']}))
    print('STAGE1', cls, json.dumps(gates['EXP']), flush=True)


def _geom_e3(R87, t):
    ch = json.load(open(os.path.join(R87, 'results', 'e3_choice.json')))['chosen']
    c = P87['e3_candidates'][ch] if ch in P87['e3_candidates'] else P87['e3_dev_candidate'][ch]
    return S.Geom(c[t])


# ------------------------------------------------------------------------------------------------ Stage 2
def ci95(x):
    x = np.asarray(x, np.float64)
    n = len(x)
    tq = {16: 2.131, 17: 2.120, 15: 2.145}.get(n, 2.0)
    return float(x.mean()), float(x.std(ddof=1)), [float(x.mean() - tq * x.std(ddof=1) / math.sqrt(n)), float(x.mean() + tq * x.std(ddof=1) / math.sqrt(n))]


def stage2(R87, R86, R86A):
    S2 = P87['stage2']
    seeds = S2['seeds']
    E = edits(R87)
    out = {'per_edit': {}, 'family': {}}
    for e in E:
        root = root_of(e, R87, R86)
        M, _ = A.masks(root, e)
        refs = {t: A.load(root, f'REF_ER_{e}_{t}') for t in ('T0', 'T1')}
        oe = {}
        for arm in ('EXP', 'ECRO'):
            rows = []
            for sd in [S2['deployed_seed']] + seeds:
                r = {'seed': sd}
                ec = {t: load_hl(R87, f'X87R_{arm}_{e}_{t}_s{sd}') for t in ('T0', 'T1')}
                for t in ('T0', 'T1'):
                    T = A.eh_table(None, refs[t], ec[t], M)
                    r[t] = {k: {'E_H': T[k]['E_H'], 'signed': signed(refs[t], ec[t], M[k]), 'noise_bound': T[k]['noise_bound'], 'capture': T[k]['capture']}
                            for k in ('stationary_all', 'C', 'D', 'A', 'obj_panel') if k in T}
                dh = A.dh_table(refs['T0'], refs['T1'], ec['T0'], ec['T1'], M)
                r['dH'] = {k: {'slope': dh[k]['slope'], 'corr': dh[k]['corr']} for k in ('C', 'D') if k in dh}
                rows.append(r)
            summ = {}
            real = [r for r in rows if r['seed'] != S2['deployed_seed']]
            for t in ('T0', 'T1'):
                for k in ('stationary_all', 'C', 'D', 'A', 'obj_panel'):
                    if k in real[0][t]:
                        summ[f'{t}:{k}:E_H'] = ci95([r[t][k]['E_H'] for r in real])
                        summ[f'{t}:{k}:signed'] = ci95([r[t][k]['signed'] for r in real])
                        summ[f'{t}:{k}:frac_pass_0.06'] = float(np.mean([r[t][k]['E_H'] <= TH['EH_region_max'] for r in real]))
            for k in ('C', 'D'):
                if k in real[0]['dH']:
                    summ[f'dH:{k}:slope'] = ci95([r['dH'][k]['slope'] for r in real]); summ[f'dH:{k}:corr'] = ci95([r['dH'][k]['corr'] for r in real])
            summ['frac_F1_F2_pass'] = float(np.mean([all(r['T1'][k]['E_H'] <= TH['EH_region_max'] for k in ('A', 'C', 'D') if k in r['T1'])
                                                    and all(r[t]['stationary_all']['E_H'] <= TH['EH_stationary_max'] and r[t]['stationary_all']['capture'] <= TH['capture_max']
                                                            for t in ('T0', 'T1')) for r in real]))
            oe[arm] = {'rows': rows, 'summary': summ}
        pool = {}
        for t in ('T0', 'T1'):
            ec = load_hl(R87, f'X87R_POOL_{e}_{t}')
            T = A.eh_table(None, refs[t], ec, M)
            pool[t] = {k: {'E_H': T[k]['E_H'], 'signed': signed(refs[t], ec, M[k])} for k in ('stationary_all', 'C', 'D', 'A', 'obj_panel') if k in T}
        oe['POOL'] = pool
        # classification for region C at T0 / T1 (EXP)
        cl = {}
        s1 = json.load(open(os.path.join(R87, 'results', 'stage1_87.json')))
        for t in ('T0', 'T1'):
            mu, sd_, ci = oe['EXP']['summary'][f'{t}:C:signed']
            dep = oe['EXP']['rows'][0][t]['C']['signed']
            pl = pool[t]['C']['signed']
            c3 = s1[e]['signed'][t]['C3']['C']
            exps = s1[e]['signed'][t]['EXP']['C']
            a_ = (pl / mu >= 0.5 if mu != 0 else False) and (ci[0] > 0 or ci[1] < 0)
            b_ = sd_ >= abs(mu) and abs(pl) <= 0.5 * abs(mu)
            c_ = abs(c3) <= 0.5 * abs(exps)
            cl[t] = {'realization_mean_signed': mu, 'sd': sd_, 'ci95': ci, 'deployed_signed': dep, 'pool_signed': pl, 'C3_signed': c3,
                     'A_SYSTEMATIC': bool(a_), 'B_VARIANCE': bool(b_), 'C3_halves_error': bool(c_)}
        oe['classification_C'] = cl
        out['per_edit'][e] = oe
    # family cases: hit counts, missed relations, BIN / RAY per realization
    fam = {}
    c3_halves = []
    for cfg, rk in FAMILY:
        e, t = cfg.split('_')
        G = S.Geom(S.FX[e][t])
        rz = np.load(os.path.join(R87, 'results', 'tables', f'R_{cfg}.npz'))
        lam = np.isin(G.PMAT, [3, 4, 5])
        W = np.load(os.path.join(R86A, 'stageB', f'importance_{cfg}.npz'))[rk].astype(np.float64).reshape(G.N, NB, 3)
        Wp = W.sum(1)
        imp = lam & (Wp.sum(1) > 0.002 * Wp.sum())
        relE = np.mean([rz[f'relE_s{s}'].sum(-1) for s in seeds], 0)                            # [N, panel] pooled RAY panel irradiance
        tot = relE.sum(1, keepdims=True)
        important_rel = (relE >= 0.05 * np.maximum(tot, 1e-30)) & imp[:, None] & (tot > 0)
        rows = []
        for sd in [S2['deployed_seed']] + seeds:
            cnt = rz[f'relcnt_s{sd}']
            hits = cnt.sum(1)[imp]
            miss = important_rel & (cnt == 0)
            f, _, _, _ = family(R86A, R86, cfg, rk, {'BIN': rz[f'BIN_s{sd}'].astype(np.float64), 'RAY': rz[f'RAY_s{sd}'].astype(np.float64)}, G)
            rows.append({'seed': sd, 'panel_hits_important_min_med_max': [int(hits.min()), int(np.median(hits)), int(hits.max())],
                         'missed_relations_frac': float(miss.sum() / max(important_rel.sum(), 1)),
                         'missed_relations_weighted': float((relE * miss).sum() / max((relE * important_rel).sum(), 1e-30)),
                         'BIN_read_over_truth': f['BIN']['read_over_truth'], 'RAY_read_over_truth': f['RAY']['read_over_truth'],
                         'BIN_patch_rel_rms': f['BIN']['patch_rel_rms'], 'RAY_patch_rel_rms': f['RAY']['patch_rel_rms'], 'truth_read': f['truth_read']})
        real = rows[1:]
        summ = {k: ci95([r[k] for r in real]) for k in ('missed_relations_frac', 'missed_relations_weighted', 'BIN_read_over_truth', 'RAY_read_over_truth',
                                                          'BIN_patch_rel_rms', 'RAY_patch_rel_rms')}
        fam[f'{cfg}:{rk}'] = {'rows': rows, 'summary': summ, 'important_relations': int(important_rel.sum()), 'important_patches': int(imp.sum())}
        s1 = json.load(open(os.path.join(R87, 'results', 'stage1_87.json')))
        sg = s1[e]['signed'][t]
        c3_halves.append({'case': f'{cfg}:{rk}', 'EXP_rendered_signed': sg['EXP'][rk], 'C3_rendered_signed': sg['C3'][rk],
                          'halved': bool(abs(sg['C3'][rk]) <= 0.5 * abs(sg['EXP'][rk]))})
    out['family'] = fam
    out['C_RECEIVER_SAMPLING'] = {'per_case': c3_halves, 'holds': bool(sum(c['halved'] for c in c3_halves) >= 2),
                                  'rule': 'rendered region signed error of the deployed state, C3 vs EXP, >= 2 of 3 family cases'}
    for e, oe in out['per_edit'].items():
        for t, c in oe['classification_C'].items():
            flags = [n for n, v in (('A', c['A_SYSTEMATIC']), ('B', c['B_VARIANCE']), ('C', out['C_RECEIVER_SAMPLING']['holds'])) if v]
            names = {'A': 'A_SYSTEMATIC', 'B': 'B_VARIANCE', 'C': 'C_RECEIVER_SAMPLING'}
            c['label'] = names[flags[0]] if len(flags) == 1 else 'D_MIXTURE'
            c['flags'] = flags
    jd(out, os.path.join(R87, 'results', 'stage2_87.json'))
    for e, oe in out['per_edit'].items():
        print(e, json.dumps({t: {k: (round(v, 4) if isinstance(v, float) else v) for k, v in c.items() if k not in ('ci95',)} for t, c in oe['classification_C'].items()}))
    print('STAGE2 done', flush=True)


# ------------------------------------------------------------------------------------------------ Stage 3
def interp(med, q, key, errk='err_stationary_all'):
    ee = [m[errk] for m in med]; cc = [m[key] for m in med]; ss = [m['spp'] for m in med]
    if q >= ee[0]:
        return {'value': cc[0], 'spp': ss[0], 'how': 'first ladder point already at or below q', 'region_err': {k: med[0]['err_' + k] for k in ('A', 'C', 'D')}}
    for i in range(len(ee) - 1):
        if ee[i] >= q >= ee[i + 1] and ee[i + 1] < ee[i]:
            s_ = math.log(q / ee[i]) / math.log(ee[i + 1] / ee[i])
            reg = {k: float(med[i]['err_' + k] * (med[i + 1]['err_' + k] / max(med[i]['err_' + k], 1e-30)) ** s_) for k in ('A', 'C', 'D')}
            return {'value': float(cc[i] * (cc[i + 1] / cc[i]) ** s_), 'spp': float(ss[i] * (ss[i + 1] / ss[i]) ** s_), 'how': 'interpolated', 'region_err': reg}
    return {'value': None, 'how': 'not reached on the ladder', 'best_error': float(min(ee))}


def stage3(R87, R86):
    R = json.load(open(os.path.join(R87, 'results', 'econ', 'econ87.json')))
    q86 = json.load(open(os.path.join(R86, 'results', 'qualify.json')))
    out = {}
    for e in R['edits']:
        Re = R[e]
        if e == 'E3':
            qa = json.load(open(os.path.join(R87, 'results', 'e3_admission.json')))
            g = qa['g_stationary']; unc = qa['regions']['stationary_all']['uncertainty_ER_T1']
        else:
            g = q86[e]['regions']['stationary_all']['g_ER_vs_P_T1']; unc = q86[e]['regions']['stationary_all']['uncertainty_ER_T1']
        g_ok = unc <= g / 2                                        # protocol: g evaluated only if the pooled reference noise <= g / 2
        em = Re['edit_median']; pm = Re['pt_edit_median']
        ffe = max(em['table_first_frame_extra_s'], 0.0)
        fixed = {'PT': pm['geometry_update_s'] + max(pm['first_frame_extra_s'], 0.0),
                 'PT_RR': pm['geometry_update_s'] + max(pm['first_frame_extra_s'], 0.0),
                 'ECRO_s8601': em['geometry_update_s'] + em['state_selective_s'] + em['source_s'] + em['ecro_inference_s'] + em['ecro_transfer_s'] + ffe,
                 'EXP': em['geometry_update_s'] + em['state_selective_s'] + em['q_s'] + em['exp_apply_s'] + em['exp_transfer_s'] + ffe,
                 'C3': em['geometry_update_s'] + em['q_s'] + em['c3_s'] + em['exp_transfer_s'] + ffe}
        fixed_gpu = {'PT': 0.0, 'PT_RR': 0.0, 'ECRO_s8601': em['state_selective_gpu_ms'] + em['source_gpu_ms'] + 1e3 * em['ecro_inference_s'],
                     'EXP': em['state_selective_gpu_ms'] + em['q_gpu_ms'] + 1e3 * em['exp_apply_s'], 'C3': em['q_gpu_ms'] + 1e3 * em['c3_s']}      # torch / numpy parts: wall as GPU proxy
        for b, v in Re['tf_cost'].items():
            fixed[f'TF{b}'] = em['geometry_update_s'] + v['fill_s'] + em['exp_transfer_s'] + ffe
            fixed_gpu[f'TF{b}'] = v['fill_gpu_ms']
        lm = Re['ladder_median']
        levels = [('0.10', 0.10), ('0.05', 0.05), ('0.03', 0.03)] + ([('g', g)] if g_ok else [])
        res = {'g': g, 'g_reference_noise': unc, 'g_evaluated': g_ok, 'fixed_wall_s': fixed, 'fixed_gpu_ms': fixed_gpu, 'levels': {}}
        for name, q in levels:
            row = {}
            for arm in lm:
                w = interp(lm[arm], q, 'wall_s'); gg = interp(lm[arm], q, 'gpu_ms')
                row[arm] = {'spp': w.get('spp'), 'frame_wall_s': w['value'], 'frame_gpu_ms': gg['value'], 'how': w['how'], 'best_error': w.get('best_error'),
                            'region_err': w.get('region_err')}
                if w['value'] is not None:
                    row[arm]['edit_to_first_frame_wall_s'] = fixed[arm] + w['value']
                    row[arm]['edit_to_first_frame_gpu_ms'] = fixed_gpu[arm] + gg['value']
                    for V in (1, 4, 16):
                        row[arm][f'amortised_wall_V{V}'] = fixed[arm] + V * w['value']
            p = row['PT']
            for arm in row:
                if arm == 'PT' or row[arm]['frame_wall_s'] is None or p['frame_wall_s'] is None:
                    continue
                c = row[arm]
                c['vs_PT_edit_to_first_frame_wall'] = p['edit_to_first_frame_wall_s'] / c['edit_to_first_frame_wall_s']
                c['vs_PT_edit_to_first_frame_gpu'] = p['edit_to_first_frame_gpu_ms'] / max(c['edit_to_first_frame_gpu_ms'], 1e-9)
                c['vs_PT_frame_gpu'] = p['frame_gpu_ms'] / c['frame_gpu_ms']
                rp, rc = p.get('region_err'), c.get('region_err')
                c['region_matched'] = bool(rp and rc and all(rc[k] <= 1.1 * rp[k] for k in rp)) if (rp and rc) else None
            if row['EXP']['frame_wall_s'] is not None and row['ECRO_s8601']['frame_wall_s'] is not None:
                row['ECRO_over_EXP_cost'] = {'edit_to_first_frame_wall': row['ECRO_s8601']['edit_to_first_frame_wall_s'] / row['EXP']['edit_to_first_frame_wall_s'],
                                             'edit_to_first_frame_gpu': row['ECRO_s8601']['edit_to_first_frame_gpu_ms'] / row['EXP']['edit_to_first_frame_gpu_ms']}
            res['levels'][name] = row
        # per-region time-to-quality (rule ii)
        rq = {}
        for reg in ('C', 'D'):
            rq[reg] = {}
            for name, q in (('0.10', 0.10), ('0.05', 0.05), ('0.03', 0.03)):
                rq[reg][name] = {arm: (lambda w: {'spp': w.get('spp'), 'frame_wall_s': w['value'], 'best_error': w.get('best_error')})(interp(lm[arm], q, 'wall_s', 'err_' + reg))
                                 for arm in lm}
        res['region_time_to_quality'] = rq
        res['floors_at_max_spp'] = {arm: {k: lm[arm][-1]['err_' + k] for k in ('stationary_all', 'A', 'C', 'D')} for arm in lm}
        res['source_update'] = Re['source_update']
        res['view2_64spp'] = {a: {'wall_s': v['wall_s'], 'gpu_ms': v['gpu_ms']} for a, v in Re['view2_64spp'].items()}
        res['view2_frame_ratio_PT_over_table'] = {'wall': Re['view2_64spp']['PT']['wall_s'] / Re['view2_64spp']['EXP']['wall_s'],
                                                  'gpu': Re['view2_64spp']['PT']['gpu_ms'] / Re['view2_64spp']['EXP']['gpu_ms']}
        res['g_note'] = None if g_ok else 'g not evaluable (reference noise > g / 2): SUPPORTED unreachable on this edit, label capped at CONDITIONALLY'
        # per-arm verdict vs PT (protocol)
        vv = {}
        for arm in lm:
            if arm == 'PT':
                continue
            def adv(lvl):
                c = res['levels'].get(lvl, {}).get(arm)
                return bool(c and c.get('frame_wall_s') is not None and c.get('vs_PT_edit_to_first_frame_wall', 0) >= 1.25
                            and c.get('vs_PT_edit_to_first_frame_gpu', 0) >= 1.25 and c.get('region_matched'))
            if all(adv(l) for l in ('0.05', '0.03', 'g')):
                vv[arm] = 'SUPPORTED'
            elif adv('0.05') and adv('0.03'):
                vv[arm] = 'CONDITIONALLY SUPPORTED'
            else:
                vv[arm] = 'NOT SUPPORTED'
        res['verdict_vs_PT'] = vv
        out[e] = res
    jd(out, os.path.join(R87, 'results', 'stage3_87.json'))
    for e in out:
        print(e, json.dumps(out[e]['verdict_vs_PT']), json.dumps({l: {a: round(v.get('vs_PT_edit_to_first_frame_wall', 0) or 0, 2) for a, v in r.items() if isinstance(v, dict) and 'spp' in v}
                                                                   for l, r in out[e]['levels'].items()}))
    print('STAGE3 done', flush=True)


def verdict(R87):
    """Protocol verdict_mapping, applied mechanically to the stage outputs -> final87.json."""
    rd = lambda n: json.load(open(os.path.join(R87, 'results', n)))
    s1, s2, s3, tests = rd('stage1_87.json'), rd('stage2_87.json'), rd('stage3_87.json'), rd('tests/tests87.json')
    E = [e for e in ('E1', 'E2', 'E3') if e in s1]
    V = {}
    c = s1['EXP_classification']
    V['EXPLICIT_OPERATOR_CORRECTNESS'] = 'SUPPORTED' if (tests['PASS'] and c['finite_nonneg'] and c['flux_within_bound']) else 'NOT SUPPORTED'
    g = s1['gates']['EXP']
    broad_ok = g['F1'] and all(s1[e]['fidelity']['T1']['EXP']['D']['E_H'] <= TH['EH_region_max'] for e in E) and all(v != 'FAIL' for v in g['F3'].values())
    frac = {e: s2['per_edit'][e]['EXP']['summary']['frac_F1_F2_pass'] for e in E}
    only_C = bool(g['failing']) and all(x['where'].endswith(':C') for x in g['failing'])
    if c['class'] == 'ACCURATE_UNDER_FROZEN_REPRESENTATION' and all(v >= 0.9 for v in frac.values()):
        V['FROZEN_STATE_ADEQUACY'] = 'SUPPORTED'
    elif broad_ok and (only_C or c['class'] == 'ACCURATE_UNDER_FROZEN_REPRESENTATION'):
        V['FROZEN_STATE_ADEQUACY'] = 'CONDITIONALLY SUPPORTED'
    elif not broad_ok:
        V['FROZEN_STATE_ADEQUACY'] = 'NOT SUPPORTED'
    else:
        V['FROZEN_STATE_ADEQUACY'] = 'INDETERMINATE'
    sharp = {}
    for e in E:
        pe = s2['per_edit'][e]
        dep = s1[e]['fidelity']['T1']['EXP']['C']['E_H']
        fr = pe['EXP']['summary']['T1:C:frac_pass_0.06']
        mean = pe['EXP']['summary']['T1:C:E_H'][0]
        pool = pe['POOL']['T1']['C']['E_H']
        sharp[e] = {'deployed_E_H': dep, 'realization_frac_pass': fr, 'realization_mean_E_H': mean, 'pool_E_H': pool,
                    'pass': dep <= TH['EH_region_max'] and fr >= 0.9,
                    'systematic_fail': dep > TH['EH_region_max'] and mean > TH['EH_region_max'] and pool > TH['EH_region_max']}
    V['SHARP_TRANSPORT_REPRESENTABILITY'] = ('SUPPORTED' if all(v['pass'] for v in sharp.values()) else
                                             'NOT SUPPORTED' if any(v['systematic_fail'] for v in sharp.values()) else 'INDETERMINATE')
    V['BROAD_TRANSPORT_FIDELITY'] = 'SUPPORTED' if broad_ok else 'NOT SUPPORTED'
    # learned vs explicit, per region class (both ECRO seeds must agree), T0 and T1
    better, worse, cmp_rows = [], [], []
    for e in E:
        for t in ('T0', 'T1'):
            F_ = s1[e]['fidelity'][t]
            for k in ('stationary_all', 'A', 'C', 'D', 'obj_panel', 'obj_sphere'):
                if k not in F_['EXP']:
                    continue
                x = F_['EXP'][k]
                ys = [F_[f'ECRO_ecro_s{sd}'][k] for sd in (8601, 8602)]
                nb = [max(x['noise_bound'], y['noise_bound']) for y in ys]
                b = all(y['E_H'] < x['E_H'] - n for y, n in zip(ys, nb)); w = all(y['E_H'] > x['E_H'] + n for y, n in zip(ys, nb))
                cmp_rows.append({'where': f'{e}_{t}:{k}', 'EXP': x['E_H'], 'ECRO': [y['E_H'] for y in ys], 'ECRO_better': b, 'ECRO_worse': w})
                (better if b else worse if w else []).append(f'{e}_{t}:{k}')
    cost = {}
    cheaper = False
    for e in s3:
        lv = s3[e]['levels']
        cost[e] = {l: lv[l].get('ECRO_over_EXP_cost') for l in lv}
        su = s3[e]['source_update']
        cost[e]['source_update_ECRO_over_EXP'] = (su['ecro_source_s'] + su['ecro_application_s']) / (su['exp_q_s'] + su['exp_apply_s'])
        for l, r in lv.items():
            oc = r.get('ECRO_over_EXP_cost')
            if oc and r['ECRO_s8601'].get('region_matched') is not False and oc['edit_to_first_frame_wall'] <= 1 / 1.25 and oc['edit_to_first_frame_gpu'] <= 1 / 1.25:
                cheaper = True
        if cost[e]['source_update_ECRO_over_EXP'] <= 1 / 1.25:
            cheaper = True
    exp_costlier = cheaper
    if (better and not worse) or cheaper:
        V['LEARNED_OPERATOR_ADDITIONAL_UTILITY'] = 'SUPPORTED'
    elif not better and not exp_costlier:
        V['LEARNED_OPERATOR_ADDITIONAL_UTILITY'] = 'NOT SUPPORTED'
    else:
        V['LEARNED_OPERATOR_ADDITIONAL_UTILITY'] = 'CONDITIONALLY SUPPORTED'
    order = ['NOT SUPPORTED', 'CONDITIONALLY SUPPORTED', 'SUPPORTED']
    econ = {}
    for arm in set(a for e in s3 for a in s3[e]['verdict_vs_PT']):
        labs = [s3[e]['verdict_vs_PT'][arm] for e in s3 if arm in s3[e]['verdict_vs_PT']]
        lab = min(labs, key=order.index)
        if lab == 'SUPPORTED' and not all(s3[e]['g_evaluated'] for e in s3):
            lab = 'CONDITIONALLY SUPPORTED'
        econ[arm] = lab
    V['QUALITY_MATCHED_ECONOMICS'] = econ
    f3 = g['F3']
    dc = {e: s1[e]['dH']['EXP'].get('C', {}) for e in E}
    V['DYNAMIC_EDIT_VALIDITY'] = ('NOT SUPPORTED' if any(v == 'FAIL' for v in f3.values()) else
                                  'CONDITIONALLY SUPPORTED' if any(v == 'INDETERMINATE' for v in f3.values()) else 'SUPPORTED')
    V['GM1_ARCHITECTURE_IMPLICATION'] = 'conditional statement over H0 / H1 / H2 written in the worklog from the labels above (no successor selected)'
    out = {'verdicts': V, 'sharp_by_edit': sharp, 'learned_vs_explicit_regions': cmp_rows, 'ECRO_better': better, 'ECRO_worse': worse,
           'cost_ECRO_over_EXP': cost, 'F3_EXP': f3, 'dH_C_EXP': dc, 'frac_F1_F2_pass_EXP': frac, 'edits': E}
    jd(out, os.path.join(R87, 'results', 'final87.json'))
    print('VERDICTS', json.dumps(V), flush=True)


if __name__ == '__main__':
    cmd = sys.argv[1]
    if cmd == 'verdict':
        verdict(sys.argv[2])
    if cmd == 'e3qualify':
        e3qualify(sys.argv[2])
    elif cmd == 'stage1':
        stage1(sys.argv[2], sys.argv[3], sys.argv[4])
    elif cmd == 'stage2':
        stage2(sys.argv[2], sys.argv[3], sys.argv[4])
    elif cmd == 'stage3':
        stage3(sys.argv[2], sys.argv[3])
