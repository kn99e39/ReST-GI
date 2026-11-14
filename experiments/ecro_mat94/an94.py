"""WL94 analysis: frozen Stage B metrics and gate, Stage C fit / outcome mapping per seed, final hypothesis labels, figures.

  python an94.py stageB <R>              -> results/an/stageB.json (prints the gate only)
  python an94.py stageC <R> <seed>       -> results/an/stageC_s<seed>.json (prints FIT / run-next only)
  python an94.py final  <R> <out_dir>    -> results/an/final.json + figures + manifest in <out_dir>
Metric definitions: protocol ecro_mat94 stage_B.metrics / stage_C.
"""
import glob, hashlib, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
P = json.load(open(os.path.join(HERE, 'protocol', 'ecro_mat94.json')))
VARIANTS = ['none', 'U07', 'U03', 'bx75', 'bx38', 'bx38i']
TEX = VARIANTS[1:]
CHECK = ['bx75', 'bx38', 'bx38i']
SEEDS = [8601, 8602]
ARMS_B = ['C0', 'C2_src', 'C2_rcv', 'C2_both', 'C2_srcS1']
NB, NPHI, NZ = 64, 8, 8
TAG = 'CK2_T1'


def jd(o, fn):
    os.makedirs(os.path.dirname(fn), exist_ok=True)
    json.dump(o, open(fn, 'w'), indent=1, default=float)


def jl(fn):
    return json.load(open(fn))


W = (((np.arange(NB) // NPHI) + 0.5) / NZ * (2 * np.pi / NB)).astype(np.float64)


def ex(X):
    """per-patch exitance (sum over bins and channels, cos dOmega weights) of [N, NB, 3]."""
    return (X.astype(np.float64) * W[None, :, None]).sum((1, 2))


def slope(m, r):
    return float((m * r).sum() / max((r * r).sum(), 1e-30))


def corr(m, r):
    return float(np.corrcoef(m, r)[0, 1]) if len(m) > 2 and np.std(m) > 0 and np.std(r) > 0 else None


# ------------------------------------------------------------------------------------------------ references
def fref(R):
    out = {}
    for v in VARIANTS:
        z = np.load(os.path.join(R, 'results', 'fields', f'CKF_{v}.npz'))
        d = {}
        for h in 'ab':
            d[f'Sp_{h}'] = z[f'e2_p_{h}'].astype(np.float64); d[f'So_{h}'] = z[f'e2_o_{h}'].astype(np.float64)
            d[f'S_{h}'] = d[f'Sp_{h}'] + d[f'So_{h}']; d[f'G_{h}'] = (z[f'e3_p_{h}'] + z[f'e3_o_{h}']).astype(np.float64)
            d[f'B_{h}'] = d[f'S_{h}'] + d[f'G_{h}']
        for k in ('Sp', 'So', 'S', 'G', 'B'):
            d[k] = 0.5 * (d[f'{k}_a'] + d[f'{k}_b'])
        out[v] = d
    return out


def masks(R):
    return dict(np.load(os.path.join(R, 'results', 'pre', 'masks94.npz')))


# ------------------------------------------------------------------------------------------------ table metrics
def comp_pairs(arm):
    """model key, reference key per component."""
    return {'B': (f'B_{arm}', 'B'), 'S': (f'S_{arm}', 'S'), 'G': ('G', 'G')}


def response(tab, F, v, comp_m, comp_r, P_):
    """log-response amplitude / corr over the patch set P_ (bool [N]) and the half-split noise of the reference response."""
    em_v, em_0 = ex(tab[f'{v}|{comp_m}'])[P_], ex(tab[f'none|{comp_m}'])[P_]
    er_v, er_0 = ex(F[v][comp_r])[P_], ex(F['none'][comp_r])[P_]
    ok = (em_v > 0) & (em_0 > 0) & (er_v > 0) & (er_0 > 0)
    rm, rr = np.log(em_v[ok] / em_0[ok]), np.log(er_v[ok] / er_0[ok])
    ra = np.log(ex(F[v][comp_r + '_a'])[P_][ok] / ex(F['none'][comp_r + '_a'])[P_][ok])
    rb = np.log(ex(F[v][comp_r + '_b'])[P_][ok] / ex(F['none'][comp_r + '_b'])[P_][ok])
    nz = 0.5 * np.abs(ra - rb)
    r_mod, r_ref = em_v[ok] / em_0[ok], er_v[ok] / er_0[ok]
    dv = max(er_v[ok].sum(), 1e-30)
    return {'amplitude': slope(rm, rr), 'corr': corr(rm, rr), 'patches': int(ok.sum()),
            'ref_snr_median': float(np.median(np.abs(rr) / np.maximum(nz, 1e-12))) if ok.any() else None,
            'baseline_err': float(np.abs(em_0[ok] - er_0[ok]).sum() / max(er_0[ok].sum(), 1e-30)),
            'textured_err': float(np.abs(em_v[ok] - er_v[ok]).sum() / dv),
            'err_baseline_carried': float(np.abs(r_mod * (em_0[ok] - er_0[ok])).sum() / dv),
            'err_response': float(np.abs(er_0[ok] * (r_mod - r_ref)).sum() / dv),
            '_rho': (rm.tolist(), rr.tolist())}


def bin_err(X, Fv, P_, binmask):
    """per patch-bin rel L1 (channels summed) of X vs Fv over patches P_ and bins selected by binmask [N, NB]."""
    sel = binmask & P_[:, None]
    x, f = X.astype(np.float64).sum(2)[sel], Fv.astype(np.float64).sum(2)[sel]
    return float(np.abs(x - f).sum() / max(f.sum(), 1e-30)), int(sel.sum())


def bin_group_amp(tab, F, v, arm, P_, binmask):
    """amplitude of the B response pooled over the bins of one group, per patch."""
    def eg(X):
        return (X.astype(np.float64) * W[None, :, None] * binmask[..., None]).sum((1, 2))[P_]
    m_v, m_0 = eg(tab[f'{v}|B_{arm}']), eg(tab[f'none|B_{arm}'])
    r_v, r_0 = eg(F[v]['B']), eg(F['none']['B'])
    ok = (m_v > 0) & (m_0 > 0) & (r_v > 0) & (r_0 > 0)
    rm, rr = np.log(m_v[ok] / m_0[ok]), np.log(r_v[ok] / r_0[ok])
    return {'amplitude': slope(rm, rr), 'corr': corr(rm, rr), 'patches': int(ok.sum())}


def table_metrics(tab, F, M, arms):
    T_, Tp, To = M['T'], M['T_panel_face'], M['T_other_faces']
    pfb = M['panel_facing_bins']; allb = np.ones_like(pfb); oth = M['T'][:, None] & ~pfb
    out = {}
    for arm in arms:
        a = {}
        for v in TEX:
            for cn, (km, kr) in comp_pairs(arm).items():
                for sn, Pset in (('T', T_), ('T_panel_face', Tp), ('T_other_faces', To)):
                    a[f'{v}|{cn}|{sn}'] = response(tab, F, v, km, kr, Pset)
            a[f'{v}|B|panel_facing_bins'] = bin_group_amp(tab, F, v, arm, T_, pfb)
            a[f'{v}|B|other_bins'] = bin_group_amp(tab, F, v, arm, T_, oth)
        for v in VARIANTS:
            e_p, n_p = bin_err(tab[f'{v}|B_{arm}'], F[v]['B'], T_, pfb)
            e_o, n_o = bin_err(tab[f'{v}|B_{arm}'], F[v]['B'], T_, oth)
            a[f'{v}|binerr'] = {'panel_facing': e_p, 'other': e_o, 'all': bin_err(tab[f'{v}|B_{arm}'], F[v]['B'], T_, allb)[0], 'n_pf': n_p, 'n_other': n_o}
        # receivers of the box material (box as a sender): S1 panel patches vs other non-box patches
        nonT = ~T_
        for v in TEX:
            er_v, er_0 = ex(F[v]['B']), ex(F['none']['B'])
            ra = np.log(np.maximum(ex(F[v]['B_a']), 1e-30) / np.maximum(ex(F['none']['B_a']), 1e-30))
            rb = np.log(np.maximum(ex(F[v]['B_b']), 1e-30) / np.maximum(ex(F['none']['B_b']), 1e-30))
            rr = np.log(np.maximum(er_v, 1e-30) / np.maximum(er_0, 1e-30))
            q = nonT & (er_0 > 0) & (np.abs(rr) > 4 * 0.5 * np.abs(ra - rb))
            mv, m0 = ex(tab[f'{v}|B_{arm}']), ex(tab[f'none|B_{arm}'])
            for nm, sel in (('Q_S1', q & M['S1']), ('Q_nonS1', q & ~M['S1'])):
                ok = sel & (mv > 0) & (m0 > 0)
                rm = np.log(mv[ok] / m0[ok])
                a[f'{v}|receivers|{nm}'] = {'amplitude': slope(rm, rr[ok]), 'corr': corr(rm, rr[ok]), 'patches': int(ok.sum())}
        out[arm] = a
    # sender split of the single flight (C0 learned receiver; C2_rcv exact receiver) vs e2_p / e2_o, untextured and per variant
    ss = {}
    for nm, km in (('C0_fromS1', 'S_C0_fromS1'), ('C0_fromOther', 'S_C0_fromOther'), ('C2_rcv_fromS1', 'S_C2_rcv_fromS1'), ('C2_rcv_fromOther', 'S_C2_rcv_fromOther')):
        kr = 'Sp' if nm.endswith('S1') else 'So'
        ss[nm] = {'baseline_err_T': response(tab, F, 'bx38', km, kr, T_)['baseline_err'],
                  'ref_share_of_S_untextured_T': float(ex(F['none'][kr])[T_].sum() / max(ex(F['none']['S'])[T_].sum(), 1e-30))}
        for v in TEX:
            r = response(tab, F, v, km, kr, T_)
            ss[nm][v] = {'amplitude': r['amplitude'], 'corr': r['corr'], 'textured_err': r['textured_err']}
    out['sender_split'] = ss
    return out


def strip(o):
    if isinstance(o, dict):
        return {k: strip(v) for k, v in o.items() if not k.startswith('_')}
    return o


# ------------------------------------------------------------------------------------------------ image metrics
def lum(x):
    return x.mean(-1)


def load_ref(R, v):
    d = {}
    for f in sorted(glob.glob(os.path.join(R, 'results', 'renders', f'REF_ER_CK_{v}_{TAG}_r*.npz'))):
        z = np.load(f)
        for k in ('e1', 'e2', 'h'):
            d.setdefault(k, []).append(z[k].astype(np.float64))
    d = {k: np.stack(x) for k, x in d.items()}
    d['tot'] = d['e1'] + d['e2'] + d['h']
    return d


def load_hl(R, v, model, arm):
    fns = sorted(glob.glob(os.path.join(R, 'results', 'renders', f'CK_{v}_{model}_{arm}_{TAG}_r*.npz')))
    return np.stack([np.load(f)['hl'].astype(np.float64) for f in fns]) if fns else None


def eh(h_reps, x_reps, tot, m):
    n, k = h_reps.shape[0], x_reps.shape[0]
    h = h_reps.mean(0); x = x_reps.mean(0)
    den = max(tot[m].sum(), 1e-30)
    o = {'E_H': float(np.abs(x[m] - h[m]).sum() / den), 'signed': float((x[m] - h[m]).sum() / den), 'H_share': float(h[m].sum() / den), 'pixels': int(m.sum())}
    o['noise_bound'] = float((0.5 * np.abs(x_reps[:k // 2].mean(0)[m] - x_reps[k // 2:].mean(0)[m]).sum()
                              + 0.5 * np.abs(h_reps[:n // 2].mean(0)[m] - h_reps[n // 2:].mean(0)[m]).sum()) / den)
    return o


def img_response(Rv, R0, Xv, X0, m):
    dref, darm = lum(Rv['h'].mean(0) - R0['h'].mean(0)), lum(Xv.mean(0) - X0.mean(0))
    n = Rv['h'].shape[0] // 2
    da = lum(Rv['h'][:n].mean(0) - R0['h'][:n].mean(0)); db = lum(Rv['h'][n:].mean(0) - R0['h'][n:].mean(0))
    a, b = darm[m], dref[m]
    return {'slope': slope(a, b), 'corr': corr(a, b), 'rel_err': float(np.abs(a - b).sum() / max(np.abs(b).sum(), 1e-30)),
            'ref_half_noise_rel': float(0.5 * np.abs(da[m] - db[m]).sum() / max(np.abs(b).sum(), 1e-30)), 'pixels': int(m.sum())}


def image_metrics(R, M, model, arms):
    refs = {v: load_ref(R, v) for v in VARIANTS}
    regs = {'ROI_T': M['ROI_T'], 'DIFF': M['DIFF']}
    out = {}
    for arm in arms:
        hl = {v: load_hl(R, v, model, arm) for v in VARIANTS}
        if any(x is None for x in hl.values()):
            continue
        a = {}
        for v in VARIANTS:
            tot = refs[v]['tot'].mean(0)
            for rn, m in regs.items():
                a[f'{v}|{rn}|E_H'] = eh(refs[v]['h'], hl[v], tot, m)
                if v != 'none':
                    a[f'{v}|{rn}|response'] = img_response(refs[v], refs['none'], hl[v], hl['none'], m)
        out[arm] = a
    return out


# ------------------------------------------------------------------------------------------------ Stage B
def gate_noise(tabm):
    snr = {v: tabm['C0'][f'{v}|B|T']['ref_snr_median'] for v in CHECK}
    return snr, all(s is not None and s >= 4 for s in snr.values())


def mr_ok(tm, arm='C0'):
    c = []
    for v in TEX:
        r = tm[arm][f'{v}|B|T']
        c.append(0.85 <= r['amplitude'] <= 1.15)
        if v in CHECK:
            c.append(r['corr'] is not None and r['corr'] >= 0.9)
    return all(c)


def ssr(im, eps=0.005):
    e0, e1 = im['C0']['none|ROI_T|E_H'], im['C2_srcS1']['none|ROI_T|E_H']
    d = e0['E_H'] - e1['E_H']; nb = max(e0['noise_bound'], e1['noise_bound'])
    return {'E_H_C0': e0['E_H'], 'E_H_C2_srcS1': e1['E_H'], 'diff': d, 'share': d / max(e0['E_H'], 1e-30), 'noise_bound': nb,
            'pass': bool(d >= max(2 * nb, eps) and d / max(e0['E_H'], 1e-30) >= 0.30)}


def stageB(R):
    pre = jl(os.path.join(R, 'results', 'pre', 'pre.json'))
    F = fref(R); M = masks(R)
    out = {'pre_status': pre['status'], 'table': {}, 'image': {}}
    for sd in SEEDS:
        tab = dict(np.load(os.path.join(R, 'results', 'tables', f'FZ_own_s{sd}.npz')))
        out['table'][f's{sd}'] = table_metrics(tab, F, M, ARMS_B)
        out['image'][f's{sd}'] = image_metrics(R, M, f'own_s{sd}', ARMS_B)
    snr, nok = gate_noise(out['table']['s8601'])
    out['noise'] = {'ref_snr_median_T_B': snr, 'pass': nok}
    out['MR_frozen'] = {f's{sd}': mr_ok(out['table'][f's{sd}']) for sd in SEEDS}
    out['MR_frozen']['all'] = all(out['MR_frozen'][f's{sd}'] for sd in SEEDS)
    out['SSR_frozen'] = {f's{sd}': ssr(out['image'][f's{sd}']) for sd in SEEDS}
    if pre['status'] != 'PASS':
        g = pre['status']
    elif not nok:
        g = 'STOP_NOISE'
    elif out['MR_frozen']['all']:
        g = 'SKIP_C'
    else:
        g = 'RUN_C'
    out['GATE'] = g
    rho = {f's{sd}': {v: out['table'][f's{sd}']['C0'][f'{v}|B|T']['_rho'] for v in TEX} for sd in SEEDS}
    jd(strip(out), os.path.join(R, 'results', 'an', 'stageB.json'))
    jd(rho, os.path.join(R, 'results', 'an', 'stageB_rho.json'))
    print('GATE', g, flush=True)
    return g


# ------------------------------------------------------------------------------------------------ Stage C
def pop_metrics(R, otag):
    z = np.load(os.path.join(R, 'results', 'pop', f'{otag}.npz'))
    o = {}
    for comp in ('B', 'S', 'G'):
        rr = np.log(np.maximum(z[f'ck|r{comp}'], 1e-30) / np.maximum(z[f'orig|r{comp}'], 1e-30))
        ra = np.log(np.maximum(z[f'ck|r{comp}a'], 1e-30) / np.maximum(z[f'orig|r{comp}a'], 1e-30))
        rb = np.log(np.maximum(z[f'ck|r{comp}b'], 1e-30) / np.maximum(z[f'orig|r{comp}b'], 1e-30))
        rm = np.log(np.maximum(z[f'ck|m{comp}'], 1e-30) / np.maximum(z[f'orig|m{comp}'], 1e-30))
        valid = (z[f'ck|r{comp}'] > 0) & (z[f'orig|r{comp}'] > 0) & (z[f'ck|m{comp}'] > 0) & (z[f'orig|m{comp}'] > 0)
        q = valid & (np.abs(rr) > 4 * 0.5 * np.abs(ra - rb))
        for nm, c in (('lambert', 0), ('S1', 1)):
            sel = q & (z['ck|cls'] == c)
            o[f'{comp}|{nm}'] = {'amplitude': slope(rm[sel], rr[sel]), 'corr': corr(rm[sel], rr[sel]), 'patches': int(sel.sum())}
    o['rel_l1'] = jl(os.path.join(R, 'results', 'pop', f'{otag}.json'))['rel_l1']
    return o


def stageC(R, sd):
    sd = int(sd)
    F = fref(R); M = masks(R)
    o = {'seed': sd, 'models': {}}
    fit = {}
    for c in 'AB':
        j = jl(os.path.join(R, 'results', 'models', f'{c}_s{sd}.json'))
        f1 = j['nonfinite_steps'] == 0
        f2 = j['val_orig_rel_l1_final'] <= 1.10 * j['val_orig_rel_l1_init']
        f3 = j['L_sum_last200'] <= 1.5 * j['L_sum_first200']
        fit[c] = {'F1': f1, 'F2': f2, 'F3': f3, 'val_init': j['val_orig_rel_l1_init'], 'val_final': j['val_orig_rel_l1_final'],
                  'L_sum_first200': j['L_sum_first200'], 'L_sum_last200': j['L_sum_last200'], 'init_sha256': j['init_sha256'], 'train_s': j['train_s'],
                  'params': j['params']}
        tab = dict(np.load(os.path.join(R, 'results', 'tables', f'CD_{c}_s{sd}.npz')))
        o['models'][c] = {'table': table_metrics(tab, F, M, ['C0', 'C2_rcv', 'C2_srcS1']), 'image': image_metrics(R, M, f'{c}_s{sd}', ['C0', 'C2_srcS1']),
                          'pop': pop_metrics(R, f'P_{c}_s{sd}')}
    o['pop_init'] = pop_metrics(R, f'P_own_s{sd}')
    o['T_INIT'] = {'pass': fit['A']['init_sha256'] == fit['B']['init_sha256']}
    o['FIT'] = {'per_candidate': fit, 'pass': bool(o['T_INIT']['pass'] and all(fit[c]['F1'] and fit[c]['F2'] and fit[c]['F3'] for c in 'AB'))}
    tA, tB = o['models']['A']['table']['C0'], o['models']['B']['table']['C0']
    mr = {}
    for v in ('bx38', 'bx38i', 'U03'):
        aA, aB = tA[f'{v}|B|T']['amplitude'], tB[f'{v}|B|T']['amplitude']
        dA, dB = abs(1 - aA), abs(1 - aB)
        mr[v] = {'a_A': aA, 'a_B': aB, 'improvement': dA - dB, 'pass': bool(dB <= 0.5 * dA and dA - dB >= 0.15)}
    pA, pB = o['models']['A']['pop']['B|lambert']['amplitude'], o['models']['B']['pop']['B|lambert']['amplitude']
    mr['V_ck_lambert'] = {'a_A': pA, 'a_B': pB, 'pass': bool(abs(1 - pB) < abs(1 - pA))}
    o['MR_C'] = {'cells': mr, 'pass': all(x['pass'] for x in mr.values()),
                 'no_improvement': all(mr[v]['improvement'] < 0.05 for v in ('bx38', 'bx38i', 'U03'))}
    sr = {}
    for v in ('bx38', 'bx38i'):
        eA, eB = o['models']['A']['image']['C0'][f'{v}|ROI_T|E_H'], o['models']['B']['image']['C0'][f'{v}|ROI_T|E_H']
        nb = max(eA['noise_bound'], eB['noise_bound'])
        sr[v] = {'E_H_A': eA['E_H'], 'E_H_B': eB['E_H'], 'noise_bound': nb,
                 'pass': bool(eB['E_H'] <= 0.8 * eA['E_H'] and eA['E_H'] - eB['E_H'] >= max(2 * nb, 0.005))}
    o['SR_C'] = {'cells': sr, 'pass': all(x['pass'] for x in sr.values())}
    o['SSR_C'] = ssr(o['models']['B']['image'])
    o['run_next_seed'] = bool(sd == 8601 and o['FIT']['pass'] and (o['MR_C']['pass'] or o['SR_C']['pass']))
    jd(strip(o), os.path.join(R, 'results', 'an', f'stageC_s{sd}.json'))
    print('STAGEC', sd, 'FIT', o['FIT']['pass'], 'RUN_NEXT', o['run_next_seed'], flush=True)


# ------------------------------------------------------------------------------------------------ final labels
def labels(B, Cs):
    """H-M / H-A labels by the frozen mapping (protocol labels)."""
    ssr_f = [B['SSR_frozen'][f's{sd}']['pass'] for sd in SEEDS]
    if B['GATE'] not in ('RUN_C', 'SKIP_C'):
        return {'H-M': 'INCONCLUSIVE', 'H-A': 'INCONCLUSIVE', 'reason': B['GATE']}
    if B['GATE'] == 'SKIP_C':
        hm = 'WEAKENED'
        ha = 'SUPPORTED' if all(ssr_f) else ('WEAKENED' if not any(ssr_f) else 'INCONCLUSIVE')
        return {'H-M': hm, 'H-A': ha, 'reason': 'MR_frozen'}
    fit = all(c['FIT']['pass'] for c in Cs)
    mr = fit and all(c['MR_C']['pass'] for c in Cs)
    sr = fit and all(c['SR_C']['pass'] for c in Cs)
    ssrc = all(c['SSR_C']['pass'] for c in Cs)
    noimp = fit and all(c['MR_C']['no_improvement'] for c in Cs)
    hm = 'SUPPORTED' if mr else ('WEAKENED' if noimp else 'INCONCLUSIVE')
    if all(ssr_f) and (not mr or ssrc):
        ha = 'SUPPORTED'
    elif (not any(ssr_f)) or (mr and sr and not ssrc):
        ha = 'WEAKENED'
    else:
        ha = 'INCONCLUSIVE'
    return {'H-M': hm, 'H-A': ha, 'FIT': fit, 'MR_C': mr, 'SR_C': sr, 'SSR_C': ssrc, 'SSR_frozen': ssr_f,
            'smoke_fit_status': 'FIT' if fit else 'INCONCLUSIVE_SMOKE_FIT'}


def final(R, outd):
    B = jl(os.path.join(R, 'results', 'an', 'stageB.json'))
    Cs = [jl(f) for f in sorted(glob.glob(os.path.join(R, 'results', 'an', 'stageC_s*.json')))]
    L = labels(B, Cs)
    jd({'labels': L, 'gate': B['GATE'], 'stageC_seeds': [c['seed'] for c in Cs]}, os.path.join(R, 'results', 'an', 'final.json'))
    import viz94
    files = viz94.figures(R, outd, B, Cs, L)
    man = []
    for f in files:
        man.append({'file': os.path.basename(f), 'sha256': hashlib.sha256(open(f, 'rb').read()).hexdigest()})
    jd({'worklog': 94, 'files': man, 'source_root': R}, os.path.join(outd, 'manifest.json'))
    print('FINAL', json.dumps(L), flush=True)


if __name__ == '__main__':
    a = [x for x in sys.argv[1:] if not x.startswith('--')]
    if a[0] == 'stageB':
        stageB(a[1])
    elif a[0] == 'stageC':
        stageC(a[1], a[2])
    elif a[0] == 'final':
        final(a[1], a[2])
