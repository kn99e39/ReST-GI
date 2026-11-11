"""WL88 analysis and prospective rules (protocol stage0 / stage1 / stage2 / stage3 / verdict_mapping). numpy only.

  python analyze88.py wqualify <R88> <name>      workload qualification on the qualification partition (REF_PQ / REF_ERQ) -> wq_<name>.json
  python analyze88.py stage1 <R88> <R86>         information sufficiency, decomposition, Task A factors, physical-target identity
  python analyze88.py stage2 <R88> <R86>         realizations, held-out edit / sources, source independence, edit validity
  python analyze88.py stage3 <R88>               lifecycle economics, quality matching, scaling
  python analyze88.py verdict <R88>              verdict mapping -> final88.json
E_H(X) = sum |hl_X - h_REF-ER| / sum tot_REF-ER (WL86); signed S(X) = sum (hl_X - h_REF-ER) / sum tot_REF-ER.
"""
import glob, json, math, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'ecro86'))
P88 = json.load(open(os.path.join(HERE, 'protocol', 'gm1_relation88.json')))
if os.environ.get('WL88_DEV'):                 # dev smoke only
    P88['w_order'] = [P88['dev_workload']]; P88['w2'] = P88['dev_workload']
import analyze86 as A  # noqa: E402  (numpy only)
import scene86 as S  # noqa: E402
NB, K = S.NB, S.K
TH = P88['stage1']['thresholds']
Y = A.Y
REG_W = ['stationary_all', 'A', 'C_ss', 'R_mirror', 'D', 'E', 'obj_floor', 'obj_walls', 'obj_panel', 'obj_ceiling']
REG_E = ['stationary_all', 'A', 'C', 'D', 'E', 'obj_panel', 'obj_floor', 'obj_walls']
SHARP_W = ['C_ss', 'R_mirror']
ARMS = P88['stage1']['arms']
RARMS = P88['stage1']['real_arms']


def jd(o, fn):
    json.dump(o, open(fn, 'w'), indent=1, default=lambda x: x.item() if hasattr(x, 'item') else str(x))


def jl(fn):
    return json.load(open(fn))


def load_hl(root, pat):
    fns = sorted(glob.glob(os.path.join(root, 'results', 'renders', pat + '_r*.npz')))
    assert fns, pat
    return {'hl': np.stack([np.load(f)['hl'].astype(np.float64) for f in fns])}


def masks(root, name):
    z = np.load(os.path.join(root, 'results', 'masks', f'{name}.npz'))
    return {k: z[k] for k in z.files if z[k].dtype == bool}, z


def eh(ref, ec, M, regs):
    h, ha, hb = A.halves(ref['h']); tot = ref['tot'].mean(0)
    hl, hla, hlb = A.halves(ec['hl'])
    T = {}
    for k in regs:
        if k not in M or M[k].sum() < 30:
            continue
        m = M[k]
        den = max(tot[m].sum(), 1e-30)
        T[k] = {'E_H': float(np.abs(hl[m] - h[m]).sum() / den), 'H_share': float(h[m].sum() / den),
                'signed': float((hl[m] - h[m]).sum() / den),
                'noise_bound': float((0.5 * np.abs(hla[m] - hlb[m]).sum() + 0.5 * np.abs(ha[m] - hb[m]).sum()) / den),
                'ref_noise': float(0.5 * np.abs(ha[m] - hb[m]).sum() / den), 'hl_over_h': float(hl[m].sum() / max(h[m].sum(), 1e-30)), 'pixels': int(m.sum())}
        T[k]['capture'] = T[k]['E_H'] / max(T[k]['H_share'], 1e-30)
    return T


def dh(ref0, ref1, ec0, ec1, M, regs):
    x, xa, xb = A.halves(Y(ref1['h']) - Y(ref0['h']))
    y, ya, yb = A.halves(Y(ec1['hl']) - Y(ec0['hl']))
    t0 = Y(ref0['tot']).mean(0)
    T = {}
    for k in regs:
        if k not in M or M[k].sum() < 30:
            continue
        m = M[k]
        sl, co = A.slope_corr(x, xa, xb, y, ya, yb, m)
        b = max(float(t0[m].mean()), 1e-30)
        T[k] = {'slope': sl, 'corr': co, 'true_absdH_rel': float(np.abs(x[m]).mean() / b), 'true_absdH_floor_rel': float(0.5 * np.abs(xa[m] - xb[m]).mean() / b)}
        T[k]['resolvable'] = bool(T[k]['true_absdH_rel'] >= TH['snr'] * T[k]['true_absdH_floor_rel'])
    return T


def f3(d, k):
    if k not in d:
        return 'NA'
    v = d[k]
    if not v['resolvable'] or v['slope'] is None:
        return 'INDETERMINATE'
    return 'PASS' if TH['dH_slope'][0] <= v['slope'] <= TH['dH_slope'][1] and v['corr'] >= TH['dH_corr_min'] else 'FAIL'


# ------------------------------------------------------------------------------------------------ Stage 0: workload qualification
def wqualify(R, name):
    Q0 = P88['stage0']['qualification']
    M, _ = masks(R, f'M_{name}')
    p0, p1 = A.load(R, f'REF_PQ_{name}_T0'), A.load(R, f'REF_PQ_{name}_T1')
    e0, e1 = A.load(R, f'REF_ERQ_{name}_T0'), A.load(R, f'REF_ERQ_{name}_T1')
    d, da, db = A.halves(Y(p1['tot']) - Y(p0['tot']))
    t0 = Y(p0['tot']).mean(0)
    nrep = p1['tot'].shape[0]
    Rg = {}
    for k in ['stationary_all', 'A', 'Bset', 'C_ss', 'R_mirror', 'D', 'E']:
        m = M[k]
        if m.sum() < 30:
            Rg[k] = {'pixels': int(m.sum())}
            continue
        b = max(float(t0[m].mean()), 1e-30)
        r = {'pixels': int(m.sum()), 'abs_change_rel': float(np.abs(d[m]).mean() / b), 'abs_change_floor_rel': float(0.5 * np.abs(da[m] - db[m]).mean() / b)}
        dh_, dha, dhb = A.halves(Y(p1['h']) - Y(p0['h']))
        r['absH_change_rel'] = float(np.abs(dh_[m]).mean() / b); r['absH_change_floor_rel'] = float(0.5 * np.abs(dha[m] - dhb[m]).mean() / b)
        t1, t1a, t1b = A.halves(p1['tot'])
        r['H_share_T1'] = float(p1['h'].mean(0)[m].sum() / t1[m].sum())
        r['uncertainty_P_T1'] = 0.5 * A.rl(t1a, t1b, m)
        ee, ea, eb = A.halves(e1['tot'])
        r['uncertainty_ER_T1'] = 0.5 * A.rl(ea, eb, m)
        r['g_ER_vs_P_T1'] = A.rl(ee, t1, m)
        # family shares of the change of h (REF-ER, common random numbers between T0 and T1)
        dhE = np.abs(Y(e1['h']).mean(0) - Y(e0['h']).mean(0))[m].sum()
        fam = {}
        for c in ('H2', 'H2G', 'H1G', 'H1P', 'Hm', 'Hb'):
            x, xa, xb = A.halves(Y(e1[c]) - Y(e0[c]))
            fam[c] = {'abs_change_share': float(np.abs(x[m]).sum() / max(dhE, 1e-30)), 'abs_change': float(np.abs(x[m]).mean() / b),
                      'floor': float(0.5 * np.abs(xa[m] - xb[m]).mean() / b),
                      'share_of_h_T1': float(Y(e1[c]).mean(0)[m].sum() / max(Y(e1['h']).mean(0)[m].sum(), 1e-30))}
        r['family'] = fam
        hh, ha, hb = A.halves(e1['h'])
        r['h_noise_rel_final'] = float(0.5 * np.abs(ha[m] - hb[m]).sum() / e1['tot'].mean(0)[m].sum()) * math.sqrt(nrep / P88['stage0']['references']['ER']['replicas'])
        Rg[k] = r
    g = lambda k, f: Rg[k].get(f, 0.0)
    sig = lambda k, f='abs_change_rel', fl='abs_change_floor_rel': g(k, f) >= Q0['snr'] * g(k, fl)
    fam = lambda k, c, f: Rg[k].get('family', {}).get(c, {}).get(f, 0.0)
    q = {}
    q['Q1_A_direct_change'] = {'value': g('A', 'abs_change_rel'), 'PASS': bool(g('A', 'abs_change_rel') >= Q0['A_min'] and sig('A'))}
    q['Q2_B_indirect_change'] = {'value': g('Bset', 'abs_change_rel'), 'PASS': bool(g('Bset', 'abs_change_rel') >= Q0['B_min'] and sig('Bset'))}
    ss = fam('C_ss', 'H2', 'abs_change_share')
    q['Q3_C_ss_sharp_to_sharp'] = {'pixels': Rg['C_ss']['pixels'], 'value': g('C_ss', 'abs_change_rel'), 'H2_change_share': ss,
                                   'PASS': bool(Rg['C_ss']['pixels'] >= Q0['min_pixels'] and g('C_ss', 'abs_change_rel') >= Q0['C_min'] and sig('C_ss')
                                                and ss >= Q0['family_share_min'] and fam('C_ss', 'H2', 'abs_change') >= Q0['snr'] * fam('C_ss', 'H2', 'floor'))}
    gs = fam('R_mirror', 'H1G', 'abs_change_share') + fam('R_mirror', 'H2G', 'abs_change_share')
    q['Q4_R_mirror_moving_sharp_sender'] = {'pixels': Rg['R_mirror']['pixels'], 'value': g('R_mirror', 'abs_change_rel'), 'G_change_share': gs,
                                            'PASS': bool(Rg['R_mirror']['pixels'] >= Q0['min_pixels'] and g('R_mirror', 'abs_change_rel') >= Q0['C_min']
                                                         and sig('R_mirror') and gs >= Q0['family_share_min'])}
    ratio = g('D', 'absH_change_rel') / max(g('D', 'abs_change_rel'), 1e-30)
    q['Q5_D_broad_multibounce'] = {'value': g('D', 'abs_change_rel'), 'absH_over_abs': ratio,
                                   'PASS': bool(g('D', 'abs_change_rel') >= Q0['D_min'] and sig('D') and ratio >= Q0['D_H_ratio_min']
                                                and sig('D', 'absH_change_rel', 'absH_change_floor_rel'))}
    q['Q6_H_share'] = {'value': g('stationary_all', 'H_share_T1'), 'PASS': bool(g('stationary_all', 'H_share_T1') >= Q0['H_share_min'])}
    un = max(g('stationary_all', 'uncertainty_P_T1'), g('stationary_all', 'uncertainty_ER_T1')) * math.sqrt(nrep / P88['stage0']['references']['ER']['replicas'])
    q['Q7_uncertainty_final'] = {'value': un, 'PASS': bool(un <= Q0['uncertainty_max'])}
    rn = max(g('C_ss', 'h_noise_rel_final'), g('R_mirror', 'h_noise_rel_final'))
    q['Q8_reference_resolution_sharp'] = {'value': rn, 'PASS': bool(rn <= Q0['sharp_ref_noise_max'])}
    adm = bool(all(v['PASS'] for v in q.values()))
    jd({'candidate': name, 'regions': Rg, 'qualification': q, 'ADMITTED': adm}, os.path.join(R, 'results', f'wq_{name}.json'))
    print('WQUALIFY', name, 'ADMITTED' if adm else 'NOT_ADMITTED', json.dumps({k: v['PASS'] for k, v in q.items()}), flush=True)


# ------------------------------------------------------------------------------------------------ helpers for stage 1 / 2
def cfg_root(cfg, R, R86):
    return R86 if cfg.startswith('E') else R


def cfg_masks(cfg, R, R86, ls=None):
    e = cfg.split('_')[0]
    if e == 'E2':
        return masks(R86, 'E2')[0], REG_E
    return masks(R, e + ('_' + ls if ls else ''))[0], REG_W


def field_cmp(Xt, T, Tn, G, sel=None):
    """relative L1 / flux ratio of a table against a truth table T (with its half-difference noise Tn) per patch class."""
    w = (G.PAREA[:, None] * ((np.arange(NB) // S.NPHI + 0.5) / S.NZ)[None])[..., None]
    cls = {'all': np.ones(G.N, bool), 'floor': G.PMAT == 0, 'panel': G.PMAT == 1, 'sphere': G.PMAT == 2, 'walls': G.PMAT == 3, 'G_or_boxA': G.PMAT == 4,
           'boxB': G.PMAT == 5}
    if sel is not None:
        cls['footprint'] = sel
    o = {}
    for k, m in cls.items():
        if m.sum() == 0:
            continue
        den = max(T[m].sum(), 1e-30)
        o[k] = {'rel_l1': float(np.abs(Xt[m] - T[m]).sum() / den), 'truth_noise': float(Tn[m].sum() / den),
                'flux_ratio': float((Xt[m] * w[m]).sum() / max((T[m] * w[m]).sum(), 1e-30))}
    return o


def geom_of(cfg, R):
    e, t = cfg.split('_')[:2]
    fx = P88['workloads']
    if e == 'E2':
        return S.Geom(S.FX['E2'][t])
    name = jl(os.path.join(R, 'results', 'w_choice.json'))['chosen'] if e == 'W1' else P88['w2']
    return S.Geom(fx[name][t])


# ------------------------------------------------------------------------------------------------ Stage 1
def stage1(R, R86):
    out = {}
    cfgs = P88['stage1']['configs']
    for cfg in cfgs:
        root = cfg_root(cfg, R, R86)
        M, regs = cfg_masks(cfg, R, R86)
        ref = A.load(root, f'REF_ER_{cfg}')
        o = {'fidelity': {}, 'field': {}, 'orders': {}}
        rends = {}
        for arm in ARMS:
            try:
                rends[arm] = load_hl(R, f'X88_{arm}_{cfg}')
            except AssertionError:
                pass
        if cfg in P88['stage2']['real_configs']:
            for arm in RARMS:
                try:
                    rends[arm] = load_hl(R, f'X88R_{arm}_{cfg}')
                except AssertionError:
                    pass
        o['fidelity'] = {arm: eh(ref, ec, M, regs) for arm, ec in rends.items()}
        G = geom_of(cfg, R)
        oz = os.path.join(R, 'results', 'fields', f'ORD_{cfg}.npz')
        if os.path.exists(oz):
            oo = np.load(oz)
            a, b = oo['a'].astype(np.float64), oo['b'].astype(np.float64)
            T = 0.5 * (a + b); Tn = 0.5 * np.abs(a - b)
            xz = np.load(os.path.join(R, 'results', 'tables', f'X_{cfg}.npz'))
            fp = None
            if 'pid' in np.load(os.path.join(root, 'results', 'masks', ('E2' if cfg.startswith('E2') else cfg.split('_')[0]) + '.npz')).files and not cfg.startswith('E2'):
                mz = np.load(os.path.join(R, 'results', 'masks', cfg.split('_')[0] + '.npz'))
                pids = np.unique(mz['pid'][mz['C_ss'] | mz['R_mirror']])
                fp = np.zeros(G.N, bool); fp[pids[(pids >= 0) & (pids < G.N)]] = True
            Bt, Btn = T[1:].sum(0), np.sqrt((Tn[1:] ** 2).sum(0))
            o['field'] = {arm: field_cmp(xz[arm].astype(np.float64), Bt, Btn, G, fp) for arm in xz.files if arm in ARMS}
            o['orders'] = {'q_vs_order1': field_cmp(xz['q'].astype(np.float64), T[0], Tn[0], G, fp)['all']}
            for arm in ('R0', 'R1'):
                o['orders'][arm] = [field_cmp(xz[f'{arm}_L{j + 2}'].astype(np.float64), T[j + 1], Tn[j + 1], G, fp) for j in range(4)]
            rz = os.path.join(R, 'results', 'tables', f'R_{cfg}.npz')
            if os.path.exists(rz):
                rr = np.load(rz)
                for arm in ('R0_INF', 'R1_INF', 'C3U_INF', 'R0_POOL16', 'R1_POOL16'):
                    o['field'][arm] = field_cmp(rr[arm].astype(np.float64), Bt, Btn, G, fp)
                o['target_identity_field'] = {k: field_cmp(rr[k].astype(np.float64), rr['R0_INF'].astype(np.float64), np.zeros_like(Bt), G, fp)
                                              for k in ('R1_INF', 'C3U_INF')}
        out[cfg] = o
    # dH (T1 - T0) for edits with both configurations
    dH = {}
    for e in P88['stage1']['edits']:
        root = cfg_root(f'{e}_T1', R, R86)
        M, regs = cfg_masks(f'{e}_T1', R, R86)
        r0, r1 = A.load(root, f'REF_ER_{e}_T0'), A.load(root, f'REF_ER_{e}_T1')
        dH[e] = {}
        for arm in ARMS:
            try:
                dH[e][arm] = dh(r0, r1, load_hl(R, f'X88_{arm}_{e}_T0'), load_hl(R, f'X88_{arm}_{e}_T1'), M, regs)
            except AssertionError:
                pass
    out['dH'] = dH
    # Task A factor decomposition (signed and E_H differences in the sharp regions)
    fac = {}
    pairs = [('allocation (C3 - C3U)', 'C3', 'C3U'), ('fresh relation per order (C3U - C3UR)', 'C3U', 'C3UR'),
             ('sender vs receiver placement, equal rays, dense quadrature (C3UR - R0D)', 'C3UR', 'R0D'),
             ('scattering quadrature dense vs per-ray (R0D - R0)', 'R0D', 'R0'), ('4x work (POOL4 - R0)', 'POOL4', 'R0'),
             ('fresh realization per order (R0F - R0)', 'R0F', 'R0'), ('sender projection, first gather (RS0 - R0)', 'RS0', 'R0')]
    for cfg in cfgs:
        F_ = out[cfg]['fidelity']
        regs = ['C'] if cfg.startswith('E2') else SHARP_W + ['D']
        fac[cfg] = {}
        for k in regs:
            if 'R0' not in F_ or k not in F_['R0']:
                continue
            fac[cfg][k] = {lab: {'d_signed': F_[a][k]['signed'] - F_[b][k]['signed'], 'd_E_H': F_[a][k]['E_H'] - F_[b][k]['E_H'],
                                 'noise': max(F_[a][k]['noise_bound'], F_[b][k]['noise_bound'])}
                           for lab, a, b in pairs if a in F_ and b in F_}
    out['taskA_factors'] = fac
    # physical-target identity: converged relations agree (rendered, region signed)
    ident = {}
    for cfg in P88['stage2']['real_configs']:
        F_ = out[cfg]['fidelity']
        regs = ['C', 'stationary_all'] if cfg.startswith('E2') else SHARP_W + ['stationary_all', 'D']
        if 'R0_INF' not in F_:
            continue
        ident[cfg] = {k: {a: {'d_signed': F_[a][k]['signed'] - F_['R0_INF'][k]['signed'], 'noise': max(F_[a][k]['noise_bound'], F_['R0_INF'][k]['noise_bound'])}
                          for a in ('R1_INF', 'C3U_INF') if a in F_ and k in F_[a]} for k in regs if k in F_['R0_INF']}
    tol = P88['stage1']['target_identity_signed_tol']
    ok = all(abs(v['d_signed']) <= tol + v['noise'] for c in ident.values() for r in c.values() for v in r.values())
    fld = all(abs(v['all']['flux_ratio'] - 1) <= P88['stage1']['target_identity_flux_tol'] for cfg in ident if 'target_identity_field' in out[cfg]
              for v in out[cfg]['target_identity_field'].values())
    out['target_identity'] = {'rendered': ident, 'PASS': bool(ok and fld), 'rendered_ok': bool(ok), 'field_flux_ok': bool(fld)}
    # gates per arm (W1 and E2; W2 and sources in stage 2)
    gates = {}
    for arm in ARMS + RARMS:
        fails = []; res = {}
        for cfg in cfgs:
            F_ = out[cfg]['fidelity'].get(arm)
            if F_ is None:
                continue
            t = cfg.split('_')[1]
            v = F_['stationary_all']
            res[f'{cfg}:F1'] = bool(v['E_H'] <= TH['EH_stationary_max'] and v['capture'] <= TH['capture_max'])
            if not res[f'{cfg}:F1']:
                fails.append({'where': f'{cfg}:stationary_all', 'excess': v['E_H'] - TH['EH_stationary_max'], 'noise_bound': v['noise_bound']})
            if t == 'T1':
                for k in (['A', 'C', 'D'] if cfg.startswith('E2') else ['A', 'C_ss', 'R_mirror', 'D']):
                    if k in F_:
                        okk = F_[k]['E_H'] <= TH['EH_region_max']
                        res[f'{cfg}:F2:{k}'] = bool(okk)
                        if not okk:
                            fails.append({'where': f'{cfg}:{k}', 'excess': F_[k]['E_H'] - TH['EH_region_max'], 'noise_bound': F_[k]['noise_bound']})
        for e in dH:
            if arm in dH[e]:
                for k in (['D', 'C'] if e == 'E2' else ['D'] + SHARP_W):
                    res[f'{e}:F3:{k}'] = f3(dH[e][arm], k)
        gates[arm] = {'results': res, 'failing': fails,
                      'F1': all(v for k, v in res.items() if ':F1' in k), 'F2': all(v for k, v in res.items() if ':F2:' in k),
                      'F2_sharp': all(v for k, v in res.items() if ':F2:C' in k or ':F2:R_mirror' in k),
                      'F3': all(v != 'FAIL' for k, v in res.items() if ':F3:' in k), 'F3_values': {k: v for k, v in res.items() if ':F3:' in k}}
    out['gates'] = gates
    # R1 vs R0 per region (both seeds' noise) and the matched-ray pair
    cmp_ = {}
    for cfg in cfgs:
        F_ = out[cfg]['fidelity']
        cmp_[cfg] = {}
        for a, b in (('R1', 'R0'), ('R1M', 'R0'), ('R1', 'POOL2'), ('R1M', 'POOL2')):
            if a in F_ and b in F_:
                cmp_[cfg][f'{a}_vs_{b}'] = {k: {'d_E_H': F_[a][k]['E_H'] - F_[b][k]['E_H'], 'noise': max(F_[a][k]['noise_bound'], F_[b][k]['noise_bound'])}
                                            for k in F_[a] if k in F_[b]}
    out['R1_vs_R0'] = cmp_
    worse = [(c, p, k) for c in cmp_ for p, d in cmp_[c].items() if p == 'R1_vs_R0' for k, v in d.items() if v['d_E_H'] > v['noise']]
    better = [(c, p, k) for c in cmp_ for p, d in cmp_[c].items() if p == 'R1_vs_R0' for k, v in d.items() if v['d_E_H'] < -v['noise']]
    out['R1_vs_R0_summary'] = {'worse_beyond_noise': worse, 'better_beyond_noise': better}
    # coverage (information)
    cov = {}
    for cfg in P88['stage2']['real_configs']:
        fn = os.path.join(R, 'results', 'tables', f'R_{cfg}.json')
        if os.path.exists(fn):
            c = jl(fn)['coverage']
            cov[cfg] = {k: {'mean': float(np.mean([v[k] for v in c.values()])), 'min': float(np.min([v[k] for v in c.values()])),
                            'max': float(np.max([v[k] for v in c.values()]))} for k in next(iter(c.values()))}
    out['coverage'] = cov
    meta = {cfg: jl(os.path.join(R, 'results', 'tables', f'X_{cfg}.json')) for cfg in cfgs}
    out['tables_meta'] = meta
    tests = jl(os.path.join(R, 'results', 'tests88.json'))
    finite = all(m['finite'] and min(m['min'].values()) >= 0 for m in meta.values())
    g1 = gates['R1']
    sharp_ok = g1['F2_sharp'] and all(v != 'FAIL' for k, v in g1['F3_values'].items() if ':F3:C' in k or ':F3:R_mirror' in k)
    if not (tests['ALL_PASS'] and finite):
        cls = 'NUMERICALLY_INVALID'
    elif g1['F1'] and g1['F2'] and g1['F3'] and not worse:
        cls = 'SUFFICIENT'
    elif sharp_ok and not any(k in SHARP_W + ['C'] for _, _, k in worse):
        cls = 'SHARP_SUFFICIENT_OTHER_FAIL'
    elif g1['failing'] and all(x['excess'] <= 1.5 * x['noise_bound'] for x in g1['failing']):
        cls = 'INSUFFICIENTLY_RESOLVED'
    else:
        cls = 'INSUFFICIENT'
    # where the converged relation (any construction) fails while the oracle table passes: relation-discretisation limit; oracle fails:
    # output representation limit (protocol stage1.limit_attribution)
    lim = {}
    for cfg in cfgs:
        F_ = out[cfg]['fidelity']
        for k in (['C'] if cfg.startswith('E2') else SHARP_W):
            if cfg.endswith('T1') and k in F_.get('R1', {}):
                o_ok = 'ORACLE' in F_ and F_['ORACLE'][k]['E_H'] <= TH['EH_region_max']
                inf_ok = any(a in F_ and F_[a][k]['E_H'] <= TH['EH_region_max'] for a in ('R0_INF', 'R1_INF', 'C3U_INF'))
                c3_ok = 'C3' in F_ and F_['C3'][k]['E_H'] <= TH['EH_region_max']
                lim[f'{cfg}:{k}'] = {'oracle_passes': bool(o_ok), 'converged_relation_passes': bool(inf_ok), 'C3_passes': bool(c3_ok),
                                     'R1_passes': bool(F_['R1'][k]['E_H'] <= TH['EH_region_max']), 'R0_passes': bool(F_['R0'][k]['E_H'] <= TH['EH_region_max'])}
    out['limit_attribution'] = lim
    out['R1_classification'] = {'class': cls, 'tests_pass': tests['ALL_PASS'], 'finite_nonneg': finite}
    jd(out, os.path.join(R, 'results', 'stage1_88.json'))
    for cfg in cfgs:
        F_ = out[cfg]['fidelity']
        print(cfg, json.dumps({arm: {k: round(F_[arm][k]['E_H'], 4) for k in F_[arm] if k in ('stationary_all', 'C', 'C_ss', 'R_mirror', 'D')}
                               for arm in ('R0', 'R1', 'R1M', 'C3', 'ORACLE', 'R0_INF', 'R1_INF') if arm in F_}), flush=True)
    print('STAGE1', cls, 'target identity', out['target_identity']['PASS'], flush=True)


# ------------------------------------------------------------------------------------------------ Stage 2
def ci95(x):
    x = np.asarray(x, np.float64)
    n = len(x)
    if n < 2:
        return [float('nan')] * 2
    se = x.std(ddof=1) / math.sqrt(n)
    t = 2.131 if n == 16 else (2.12 if n == 17 else 2.0)
    return [float(x.mean() - t * se), float(x.mean() + t * se)]


def stage2(R, R86):
    out = {}
    S2 = P88['stage2']
    real = {}
    seeds = [86001] + S2['seeds']
    for cfg in S2['real_configs']:
        root = cfg_root(cfg, R, R86)
        M, regs = cfg_masks(cfg, R, R86)
        ref = A.load(root, f'REF_ER_{cfg}')
        regs_s = (['C', 'stationary_all', 'D'] if cfg.startswith('E2') else SHARP_W + ['stationary_all', 'D'])
        rr = {}
        for arm in ('R0', 'R1'):
            vals = {k: [] for k in regs_s}; sig = {k: [] for k in regs_s}; passes = []
            for sd in seeds[1:]:
                T = eh(ref, load_hl(R, f'X88S_{arm}_{cfg}_s{sd}'), M, regs_s)
                for k in regs_s:
                    if k in T:
                        vals[k].append(T[k]['E_H']); sig[k].append(T[k]['signed'])
                passes.append(all(T[k]['E_H'] <= (TH['EH_stationary_max'] if k == 'stationary_all' else TH['EH_region_max']) for k in T))
            rr[arm] = {k: {'E_H_mean': float(np.mean(vals[k])), 'E_H_sd': float(np.std(vals[k], ddof=1)), 'E_H_min': float(np.min(vals[k])),
                           'E_H_max': float(np.max(vals[k])), 'signed_mean': float(np.mean(sig[k])), 'signed_ci95': ci95(sig[k]),
                           'pass_frac_region': float(np.mean(np.array(vals[k]) <= (TH['EH_stationary_max'] if k == 'stationary_all' else TH['EH_region_max'])))}
                       for k in regs_s if vals[k]}
            rr[arm]['pass_frac_all_regions'] = float(np.mean(passes)) if cfg.endswith('T1') else None
        rr['R1_minus_R0_paired'] = {}
        for k in regs_s:
            a = []
            for sd in seeds[1:]:
                T0_ = eh(ref, load_hl(R, f'X88S_R0_{cfg}_s{sd}'), M, [k]); T1_ = eh(ref, load_hl(R, f'X88S_R1_{cfg}_s{sd}'), M, [k])
                if k in T0_:
                    a.append(T1_[k]['E_H'] - T0_[k]['E_H'])
            if a:
                rr['R1_minus_R0_paired'][k] = {'mean': float(np.mean(a)), 'ci95': ci95(a), 'frac_R1_better': float(np.mean(np.array(a) < 0))}
        real[cfg] = rr
    out['realizations'] = real
    # held-out edit W2 and held-out sources on W1 T1
    ho = {}
    for cfg, ls in [('W2_T0', None), ('W2_T1', None), ('W1_T1', 'LMIX88'), ('W1_T1', 'LSPOT88')]:
        tag = cfg + ('_' + ls if ls else '')
        M, regs = cfg_masks(cfg, R, R86, ls)
        try:
            ref = A.load(R, f'REF_ER_{tag}')
        except AssertionError:
            continue
        ho[tag] = {arm: eh(ref, load_hl(R, f'X88_{arm}_{tag}'), M, regs) for arm in S2['heldout_arms'] if glob.glob(os.path.join(R, 'results', 'renders', f'X88_{arm}_{tag}_r*.npz'))}
    out['heldout'] = ho
    try:
        M, regs = cfg_masks('W2_T1', R, R86)
        r0, r1 = A.load(R, 'REF_ER_W2_T0'), A.load(R, 'REF_ER_W2_T1')
        out['dH_W2'] = {arm: dh(r0, r1, load_hl(R, f'X88_{arm}_W2_T0'), load_hl(R, f'X88_{arm}_W2_T1'), M, regs) for arm in S2['heldout_arms']}
    except AssertionError:
        out['dH_W2'] = {}
    gates = {}
    for arm in S2['heldout_arms']:
        r = {}
        for tag, T in ho.items():
            if arm not in T:
                continue
            v = T[arm]
            if 'LMIX' in tag or 'LSPOT' in tag:
                r[f'{tag}:F4'] = bool(v['stationary_all']['E_H'] <= TH['EH_unseen_source_max'])
                for k in SHARP_W:
                    if k in v:
                        r[f'{tag}:F2s:{k}'] = bool(v[k]['E_H'] <= TH['EH_region_max'])
            else:
                r[f'{tag}:F1'] = bool(v['stationary_all']['E_H'] <= TH['EH_stationary_max'] and v['stationary_all']['capture'] <= TH['capture_max'])
                if tag.endswith('T1'):
                    for k in ['A', 'C_ss', 'R_mirror', 'D']:
                        if k in v:
                            r[f'{tag}:F2:{k}'] = bool(v[k]['E_H'] <= TH['EH_region_max'])
        if arm in out['dH_W2']:
            for k in ['D'] + SHARP_W:
                r[f'W2:F3:{k}'] = f3(out['dH_W2'][arm], k)
        gates[arm] = r
    out['heldout_gates'] = gates
    tests = jl(os.path.join(R, 'results', 'tests88.json'))
    econ = os.path.join(R, 'results', 'econ', 'econ88.json')
    ec = jl(econ) if os.path.exists(econ) else {}
    out['source_independence'] = {'construction_has_no_source_input': tests['T_SOURCE_INDEPENDENT']['PASS'],
                                  'digest_constant_across_sources': ec.get('source_reuse_digest_constant'),
                                  'heldout_source_gates_R1': {k: v for k, v in gates.get('R1', {}).items() if 'LMIX' in k or 'LSPOT' in k}}
    out['edit_validity'] = {'tests': tests['T_EDIT'], 'econ_selective_equals_full': ec.get('selective_equals_full_digest')}
    jd(out, os.path.join(R, 'results', 'stage2_88.json'))
    for cfg, rr in real.items():
        print('REAL', cfg, json.dumps({a: {k: [round(rr[a][k]['E_H_mean'], 4), round(rr[a][k]['E_H_sd'], 4), round(rr[a][k]['signed_mean'], 4)]
                                           for k in rr[a] if isinstance(rr[a][k], dict)} for a in ('R0', 'R1')}), flush=True)
    print('STAGE2 heldout', json.dumps(gates), flush=True)


# ------------------------------------------------------------------------------------------------ Stage 3
def interp(med, lvl, key):
    """spp and wall at which err[key] first reaches lvl (log-log interpolation; first point counts; no extrapolation)."""
    pts = [(r['spp'], r['wall_s'], r['gpu_ms'], r[key]) for r in med]
    for i, (s, w, g, e) in enumerate(pts):
        if e <= lvl:
            if i == 0:
                return {'spp': s, 'wall_s': w, 'gpu_ms': g, 'first_point': True}
            s0, w0, g0, e0 = pts[i - 1]
            f = (math.log(e0) - math.log(lvl)) / max(math.log(e0) - math.log(e), 1e-12)
            lg = lambda a, b: math.exp(math.log(a) + f * (math.log(b) - math.log(a)))
            return {'spp': lg(s0, s), 'wall_s': lg(w0, w), 'gpu_ms': lg(max(g0, 1e-9), max(g, 1e-9)), 'first_point': False}
    return None


def stage3(R):
    ec = jl(os.path.join(R, 'results', 'econ', 'econ88.json'))
    sc = jl(os.path.join(R, 'results', 'econ', 'scale88.json'))
    S3 = P88['stage3']
    em = ec['edit_median']
    out = {}
    # per-edit lifecycle (warm medians), stages A..G
    life = {
        'R0': {'A_geometry': em['A_geometry_s'], 'B_relation_discovery': em['B_R0_state_s'], 'C_coefficients': em['C_R0_coeff_s'], 'D_source_q': em['D_q_s'],
               'E_application': em['E_R0_apply_s'], 'F_transfer': em['F_transfer_s'], 'G_first_frame_extra': em['G_first_frame_extra_s']},
        'R1': {'A_geometry': em['A_geometry_s'], 'B_C_relation_update': em['R1_update_s'], 'D_source_q': em['D_q_s'], 'E_application': em['E_R1_apply_s'],
               'F_transfer': em['F_transfer_s'], 'G_first_frame_extra': em['G_first_frame_extra_s']},
        'R1M': {'A_geometry': em['A_geometry_s'], 'B_C_relation_update': em['R1M_update_s'], 'D_source_q': em['D_q_s'], 'E_application': em['E_R1M_apply_s'],
                'F_transfer': em['F_transfer_s'], 'G_first_frame_extra': em['G_first_frame_extra_s']},
        'C3': {'A_geometry': em['A_geometry_s'], 'D_source_q': em['D_q_s'], 'BCE_sender_side_per_source': em['C3_s'], 'F_transfer': em['F_transfer_s'],
               'G_first_frame_extra': em['G_first_frame_extra_s']},
        'PT': {'A_geometry': ec['pt_edit_median']['geometry_update_s'], 'G_first_frame_extra': ec['pt_edit_median']['first_frame_extra_s']}}
    for b, v in ec['tf_cost'].items():
        life[f'TF{b}'] = {'A_geometry': em['A_geometry_s'], 'BCDE_fill': v['fill_s'], 'F_transfer': em['F_transfer_s'], 'G_first_frame_extra': em['G_first_frame_extra_s']}
    for a in life:
        life[a]['total_per_edit'] = float(sum(life[a].values()))
    out['lifecycle_per_edit'] = life
    r1 = life['R1']
    rel_upd = r1['B_C_relation_update']; r0_rel = life['R0']['B_relation_discovery'] + life['R0']['C_coefficients']
    out['bottleneck_R1'] = {'relation_update_share': rel_upd / r1['total_per_edit'], 'application_share': r1['E_application'] / r1['total_per_edit'],
                            'source_q_share': r1['D_source_q'] / r1['total_per_edit'], 'relation_update_over_R0': rel_upd / max(r0_rel, 1e-12),
                            'application_over_R0': r1['E_application'] / max(life['R0']['E_application'], 1e-12),
                            'R1_update_parts_median': {k: float(np.median([e['R1_update_parts'][k] for e in ec['edits']])) for k in ec['edits'][0]['R1_update_parts']}}
    su = ec['source_update_median']
    out['source_only_update'] = {'R0': su['q_s'] + su['R0_apply_s'], 'R1': su['q_s'] + su['R1_apply_s'], 'C3': su['q_s'] + su['C3_s'],
                                 'R1_relation_rebuilt': False, 'digest_constant': ec['source_reuse_digest_constant']}
    out['multi_source'] = {f'S{n}': {a: out['source_only_update'][a] * n for a in ('R0', 'R1', 'C3')} for n in S3['source_reuse_counts']}
    # quality matching at T1 (edit-to-first-frame = per-edit + render to the level)
    lm = ec['ladder_median']
    lvls = S3['levels']
    qm = {}
    for key, lv_list in [('err_stationary_all', lvls['stationary']), ('err_C_ss', lvls['sharp']), ('err_R_mirror', lvls['sharp']), ('err_D', lvls['region'])]:
        qm[key] = {}
        for lv in lv_list:
            row = {}
            for arm in lm:
                p = interp(lm[arm], lv, key)
                if p is None:
                    row[arm] = None
                    continue
                fixed = life['PT']['total_per_edit'] if arm == 'PT' else life[arm]['total_per_edit'] if arm in life else None
                row[arm] = {**p, 'edit_to_first_frame_s': (fixed or 0) + p['wall_s']}
            qm[key][str(lv)] = row
    out['quality_matched'] = qm
    floors = {arm: {k: min(r[k] for r in lm[arm]) for k in ('err_stationary_all', 'err_C_ss', 'err_R_mirror', 'err_D')} for arm in lm}
    out['quality_floor'] = floors
    adv = {}
    for key in qm:
        for lv, row in qm[key].items():
            for a in ('R1', 'R1M', 'R0', 'C3'):
                if row.get(a) and row.get('PT'):
                    adv.setdefault(a, {})[f'{key}@{lv}'] = row['PT']['edit_to_first_frame_s'] / row[a]['edit_to_first_frame_s']
            if row.get('R1') and row.get('R0'):
                adv.setdefault('R1_over_R0', {})[f'{key}@{lv}'] = row['R0']['edit_to_first_frame_s'] / row['R1']['edit_to_first_frame_s']
            elif row.get('R1') and not row.get('R0'):
                adv.setdefault('R1_over_R0', {})[f'{key}@{lv}'] = 'R0 does not reach'
            elif row.get('R0') and not row.get('R1'):
                adv.setdefault('R1_over_R0', {})[f'{key}@{lv}'] = 'R1 does not reach'
    out['speedups'] = adv
    V = {}
    for n in S3['view_counts']:
        lv = str(lvls['stationary'][-1])
        row = qm['err_stationary_all'][lv]
        V[f'V{n}'] = {a: (life[a]['total_per_edit'] if a in life else 0) + n * row[a]['wall_s'] for a in row if row[a]}
    out['view_amortisation_stationary_0.03'] = V
    out['memory'] = ec['memory']
    rows = sc['rows']
    sc_out = {}
    for gb in (False, True):
        rr = [r for r in rows if r['glossy_B'] == gb]
        if len(rr) < 2:
            continue
        N = np.log([r['N'] for r in rr])
        fit = lambda key: float(np.polyfit(N, np.log([max(r[key], 1e-9) for r in rr]), 1)[0])
        sc_out['glossy_B' if gb else 'G_only'] = {'N': [r['N'] for r in rr], 'peaked_patches': [r['peaked_patches'] for r in rr],
                                                  'exp_R0_state': fit('R0_state_s'), 'exp_R1_build': fit('R1_build_s'), 'exp_R1_pairs': fit('R1_pairs'),
                                                  'exp_R1_bytes': fit('R1_bytes'), 'exp_R0_apply_J4': fit('R0_apply_J4_s'), 'exp_R1_apply_J4': fit('R1_apply_J4_s'),
                                                  'exp_R1_update': fit('R1_update_s'), 'rows': rr}
    out['scaling'] = sc_out
    jd(out, os.path.join(R, 'results', 'stage3_88.json'))
    print('STAGE3', json.dumps({'R1_total': r1['total_per_edit'], 'R0_total': life['R0']['total_per_edit'], 'bottleneck': out['bottleneck_R1']['relation_update_share'],
                                'scaling': {k: (v['exp_R1_build'], v['exp_R1_pairs']) for k, v in sc_out.items()}}), flush=True)


# ------------------------------------------------------------------------------------------------ verdict
def verdict(R):
    VM = P88['verdict_mapping']
    s1 = jl(os.path.join(R, 'results', 'stage1_88.json'))
    s2 = jl(os.path.join(R, 'results', 'stage2_88.json'))
    s3p = os.path.join(R, 'results', 'stage3_88.json')
    s3 = jl(s3p) if os.path.exists(s3p) else None
    tests = jl(os.path.join(R, 'results', 'tests88.json'))
    wq = jl(os.path.join(R, 'results', 'w_choice.json'))
    adm = [jl(f) for f in sorted(glob.glob(os.path.join(R, 'results', 'wq_*.json')))]
    V = {}
    V['WORKLOAD_QUALIFICATION'] = 'SUPPORTED' if any(a['ADMITTED'] for a in adm) else 'NOT SUPPORTED'
    V['PHYSICAL_CORRECTNESS'] = 'SUPPORTED' if tests['ALL_PASS'] and s1['target_identity']['PASS'] else (
        'CONDITIONALLY SUPPORTED' if tests['ALL_PASS'] else 'NOT SUPPORTED')
    cls = s1['R1_classification']['class']
    g1 = s1['gates']['R1']
    real = s2['realizations']
    sharp_cfgs = [c for c in real if c.startswith('W1') and c.endswith('T1')]
    frac = [real[c]['R1']['pass_frac_all_regions'] for c in sharp_cfgs if real[c]['R1'].get('pass_frac_all_regions') is not None]
    paired = [v for c in sharp_cfgs for k, v in real[c]['R1_minus_R0_paired'].items() if k in SHARP_W]
    improve = any(v['ci95'][1] < 0 for v in paired)
    if cls == 'SUFFICIENT' and frac and min(frac) >= VM['realization_pass_frac_supported']:
        V['SHARP_RELATION_INFORMATION_SUFFICIENCY'] = 'SUPPORTED'
    elif cls in ('SUFFICIENT', 'SHARP_SUFFICIENT_OTHER_FAIL') or (g1['F2_sharp'] and improve):
        V['SHARP_RELATION_INFORMATION_SUFFICIENCY'] = 'CONDITIONALLY SUPPORTED'
    elif cls == 'INSUFFICIENTLY_RESOLVED':
        V['SHARP_RELATION_INFORMATION_SUFFICIENCY'] = 'INDETERMINATE'
    else:
        V['SHARP_RELATION_INFORMATION_SUFFICIENCY'] = 'NOT SUPPORTED'
    si = s2['source_independence']
    src_ok = si['construction_has_no_source_input'] and si['digest_constant_across_sources']
    src_gate = all(v for v in si['heldout_source_gates_R1'].values()) if si['heldout_source_gates_R1'] else None
    V['SOURCE_INDEPENDENT_REUSE'] = ('SUPPORTED' if src_ok and src_gate else 'CONDITIONALLY SUPPORTED' if src_ok and src_gate is not None else
                                     'NOT SUPPORTED' if not src_ok else 'INDETERMINATE')
    ev = s2['edit_validity']
    hg = s2['heldout_gates'].get('R1', {})
    w2_ok = all(v not in (False, 'FAIL') for k, v in hg.items() if k.startswith('W2'))
    f3_ok = g1['F3']
    V['DYNAMIC_EDIT_VALIDITY'] = ('SUPPORTED' if ev['tests']['PASS'] and ev.get('econ_selective_equals_full') and f3_ok and w2_ok else
                                  'CONDITIONALLY SUPPORTED' if ev['tests']['PASS'] and (f3_ok or w2_ok) else 'NOT SUPPORTED')
    if s3:
        b = s3['bottleneck_R1']
        V['RELATION_CONSTRUCTION_ECONOMICS'] = ('SUPPORTED' if b['relation_update_over_R0'] <= VM['cost_ratio_supported'] else
                                                'CONDITIONALLY SUPPORTED' if b['relation_update_over_R0'] <= VM['cost_ratio_conditional'] else 'NOT SUPPORTED')
        V['OPERATOR_APPLICATION_ECONOMICS'] = ('SUPPORTED' if b['application_over_R0'] <= VM['cost_ratio_supported'] else
                                               'CONDITIONALLY SUPPORTED' if b['application_over_R0'] <= VM['cost_ratio_conditional'] else 'NOT SUPPORTED')
        sp = s3['speedups']
        r1o = sp.get('R1_over_R0', {})
        sharp_keys = [k for k in r1o if 'C_ss' in k or 'R_mirror' in k]
        sharp_win = any((isinstance(r1o[k], float) and r1o[k] >= VM['meaningful_advantage']) or r1o[k] == 'R0 does not reach' for k in sharp_keys)
        sharp_loss = any((isinstance(r1o[k], float) and r1o[k] <= 1 / VM['meaningful_advantage']) or r1o[k] == 'R1 does not reach' for k in sharp_keys)
        vs_pt = [v for k, v in sp.get('R1', {}).items()]
        eligible = V['SHARP_RELATION_INFORMATION_SUFFICIENCY'] in ('SUPPORTED', 'CONDITIONALLY SUPPORTED') and V['SOURCE_INDEPENDENT_REUSE'] != 'NOT SUPPORTED'
        if not eligible:
            V['QUALITY_MATCHED_END_TO_END_ECONOMICS'] = 'NOT SUPPORTED (ineligible: fidelity / source-independence requirement not met)'
        elif sharp_win and not sharp_loss and vs_pt and min(vs_pt) >= VM['meaningful_advantage']:
            V['QUALITY_MATCHED_END_TO_END_ECONOMICS'] = 'SUPPORTED'
        elif (sharp_win or not sharp_loss) and vs_pt and max(vs_pt) >= VM['meaningful_advantage']:
            V['QUALITY_MATCHED_END_TO_END_ECONOMICS'] = 'CONDITIONALLY SUPPORTED'
        else:
            V['QUALITY_MATCHED_END_TO_END_ECONOMICS'] = 'NOT SUPPORTED'
        scg = s3['scaling'].get('G_only', {})
        e_b = scg.get('exp_R1_build'); e_p = scg.get('exp_R1_pairs')
        V['REPRESENTATION_SCALABILITY'] = ('INDETERMINATE' if e_b is None else 'SUPPORTED' if max(e_b, e_p) <= VM['scaling_exp_supported'] else
                                           'CONDITIONALLY SUPPORTED' if max(e_b, e_p) <= VM['scaling_exp_conditional'] else 'NOT SUPPORTED')
    else:
        for k in ('RELATION_CONSTRUCTION_ECONOMICS', 'OPERATOR_APPLICATION_ECONOMICS', 'QUALITY_MATCHED_END_TO_END_ECONOMICS', 'REPRESENTATION_SCALABILITY'):
            V[k] = 'NOT TESTED'
    # outcome A..F (protocol stage4)
    lim = s1['limit_attribution']
    suff = V['SHARP_RELATION_INFORMATION_SUFFICIENCY'] in ('SUPPORTED', 'CONDITIONALLY SUPPORTED')
    c3_only = any(v['C3_passes'] and not v['R1_passes'] for v in lim.values())
    none_ok = any(not v['R1_passes'] and not v['C3_passes'] and not v['converged_relation_passes'] for v in lim.values())
    outc = []
    if suff and V['SOURCE_INDEPENDENT_REUSE'] in ('SUPPORTED', 'CONDITIONALLY SUPPORTED') and s3:
        if V['RELATION_CONSTRUCTION_ECONOMICS'] != 'NOT SUPPORTED' and V['QUALITY_MATCHED_END_TO_END_ECONOMICS'] in ('SUPPORTED', 'CONDITIONALLY SUPPORTED'):
            outc.append('A_RELATION_FEASIBILITY')
        b = s3['bottleneck_R1']
        if b['relation_update_share'] >= VM['dominance_share']:
            outc.append('C_DISCOVERY_DOMINATED')
        elif b['application_share'] + b['source_q_share'] >= VM['dominance_share']:
            outc.append('B_EVALUATION_DOMINATED')
    if not suff and c3_only:
        outc.append('D_SOURCE_INDEPENDENCE_LIMIT')
    if none_ok:
        outc.append('E_REPRESENTATION_LIMIT')
    if not outc:
        outc.append('F_MIXED_INDETERMINATE')
    V['GM1_ARCHITECTURE_IMPLICATION'] = {'outcomes': outc, 'note': 'classification of the measured result, not an architecture selection'}
    jd({'verdicts': V, 'R1_class': cls, 'w_choice': wq.get('chosen'), 'limit_attribution': lim}, os.path.join(R, 'results', 'final88.json'))
    print('VERDICT', json.dumps(V), flush=True)


if __name__ == '__main__':
    cmd = sys.argv[1]
    if cmd == 'wqualify':
        wqualify(sys.argv[2], sys.argv[3])
    elif cmd == 'stage1':
        stage1(sys.argv[2], sys.argv[3])
    elif cmd == 'stage2':
        stage2(sys.argv[2], sys.argv[3])
    elif cmd == 'stage3':
        stage3(sys.argv[2])
    elif cmd == 'verdict':
        verdict(sys.argv[2])
