"""WL86A Stage E: causal attribution table from the diagnostic evidence (numpy only; rules = protocol/ecro86a.json).

  python analyze86a.py <root86> <out>
"""
import glob, json, math, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'ecro86'))
import scene86 as S  # noqa: E402
P = json.load(open(os.path.join(HERE, 'protocol', 'ecro86a.json')))
NB = S.NB
SEEDS = P['seeds']
VARS = ['TRUE', 'ANG_PANEL', 'ANG_SPHERE', 'ANG_BROAD', 'ENC_PANEL', 'ENC_ALL', 'ENC16_ALL']
SHAPES = ['floor', 'ceiling', 'wall_back', 'wall_left', 'wall_right', 'panel', 'sphere', 'boxA', 'boxB']


def region_sums(root, cfg):
    e = cfg.split('_')[0]
    mz = np.load(os.path.join(root, 'results', 'masks', f'{e}.npz'))
    fns = sorted(glob.glob(os.path.join(root, 'results', 'renders', f'REF_ER_{cfg}_r*.npz')))
    tot = 0; h = 0
    for fn in fns:
        z = np.load(fn); tot = tot + (z['e1'].astype(np.float64) + z['e2'] + z['h']) / len(fns); h = h + z['h'].astype(np.float64) / len(fns)
    return mz, tot, h


def main():
    root, out = sys.argv[1], sys.argv[2]
    R = os.path.join(out, 'results')
    res = {'cases': {}, 'eh': {}, 'validation': {}, 'beams': {}, 'geometry': {}}
    cases = [tuple(c) for c in P['cases']['primary']] + [tuple(c) for c in P['cases']['controls']]
    for cfg in P['configs']:
        mz, tot, h = region_sums(root, cfg)
        Wz = np.load(os.path.join(out, 'stageB', f'importance_{cfg}.npz'))
        z = np.load(os.path.join(root, 'results', 'fields', f'F_{cfg}.npz'))
        Bo = 0.5 * (z['fa'].astype(np.float64) + z['fb'])
        D = {}
        for v in VARS:
            zz = np.load(os.path.join(R, 'fields', f'D_{v}_{cfg}.npz'))
            D[v] = (zz['fa'].astype(np.float64), zz['fb'].astype(np.float64))
        T = 0.5 * (D['TRUE'][0] + D['TRUE'][1])
        L = {sd: np.load(os.path.join(root, 'results', 'renders', f'EC_{sd}_ecro_{cfg}_r0.npz'))['B'].astype(np.float64) for sd in SEEDS}
        Mz = {sd: np.load(os.path.join(R, 'fields', f'M_{sd}_{cfg}.npz')) for sd in SEEDS}
        G = S.Geom(S.FX[cfg.split('_')[0]][cfg.split('_')[1]])
        fw = (G.PAREA[:, None] * ((np.arange(NB) // S.NPHI + 0.5) / S.NZ)[None])[..., None]
        res['validation'][cfg] = {'true_vs_stored_flux_ratio': float((T * fw).sum() / (Bo * fw).sum()), 'true_vs_stored_rel_l1': float(np.abs(T - Bo).sum() / Bo.sum()),
                                  'true_half_noise_rel_l1': float(0.5 * np.abs(D['TRUE'][0] - D['TRUE'][1]).sum() / T.sum())}
        for (c, rk) in cases:
            if c != cfg:
                continue
            W = Wz[rk].astype(np.float64).reshape(G.N, NB, 3)
            m = mz[rk]
            ts = tot[m].sum()
            Pf = lambda X, B: float((W * (X - B)).sum() / ts)
            pcf = lambda X, B: (W * (X - B)).sum((1, 2)) / ts
            row = {'pixels': int(m.sum()), 'H_share': float(h[m].sum() / ts), 'learned': {}, 'diag': {}}
            for sd in SEEDS:
                row['learned'][sd] = {'P': Pf(L[sd], Bo), 'P_hq_source': float((W * (Mz[sd]['Bhq'] - Bo)).sum() / ts), 'P_16_source': float((W * (Mz[sd]['B16'] - Bo)).sum() / ts)}
                row['learned'][sd]['estimator_effect'] = row['learned'][sd]['P_hq_source'] - row['learned'][sd]['P_16_source']
            Pl = float(np.mean([row['learned'][sd]['P'] for sd in SEEDS]))
            pcl = np.mean([pcf(L[sd], Bo) for sd in SEEDS], 0)
            row['learned_mean_P'] = Pl
            row['true_vs_stored_P'] = Pf(T, Bo); row['true_noise_P'] = float((W * (D['TRUE'][0] - D['TRUE'][1])).sum() * 0.5 / ts)
            for v in VARS[1:]:
                X = 0.5 * (D[v][0] + D[v][1])
                pcd = pcf(X, T)
                ok = np.abs(pcl) + np.abs(pcd) > 0
                cc = float(np.corrcoef(pcd[ok], pcl[ok])[0, 1]) if ok.sum() > 2 else None
                sl = float((pcd[ok] * pcl[ok]).sum() / max((pcd[ok] ** 2).sum(), 1e-30))
                noise = float(abs((W * (D[v][0] - D[v][1])).sum()) * 0.5 / ts)
                row['diag'][v] = {'P': Pf(X, T), 'noise': noise, 'explained_fraction': Pf(X, T) / Pl if abs(Pl) > 1e-9 else None,
                                  'patch_corr_with_learned': cc, 'patch_slope_learned_on_diag': sl,
                                  'by_shape': {sn: float(pcd[G.PSID == i].sum()) for i, sn in enumerate(SHAPES)}}
            row['learned_by_shape'] = {sn: float(pcl[G.PSID == i].sum()) for i, sn in enumerate(SHAPES)}
            res['cases'][f'{cfg}:{rk}'] = row
        # rendered E_H of the diagnostic tables
        for v in P['render']['tables']:
            fns = sorted(glob.glob(os.path.join(R, 'renders', f'DR_{v}_{cfg}_r*.npz')))
            if not fns:
                continue
            hl = [np.load(fn)['hl'].astype(np.float64) for fn in fns]
            hm = np.mean(hl, 0)
            for rk in ('C', 'D', 'obj_panel', 'obj_sphere', 'stationary_all', 'A'):
                m = mz[rk]
                res['eh'][f'{cfg}:{rk}:{v}'] = {'E_H': float(np.abs(hm[m] - h[m]).sum() / tot[m].sum()), 'signed': float((hm[m].sum() - h[m].sum()) / tot[m].sum()),
                                                'noise': float(0.5 * np.abs(hl[0][m] - hl[-1][m]).sum() / tot[m].sum())}
        # geometry inputs: box poses are recoverable from the patch tables alone (centre = mean of the face-patch centroids)
        gg = S.FX[cfg.split('_')[0]][cfg.split('_')[1]]
        res['geometry'][cfg] = {b: float(np.linalg.norm(G.PC[G.PSID == sid].mean(0) - np.array(gg[b]['c']))) for b, sid in (('A', S.SID_BOXA), ('B', S.SID_BOXB))}
    # ---- beams: identical-encoding pairs
    bj = json.load(open(os.path.join(R, 'beams', 'beams.json')))
    Wc = np.load(os.path.join(out, 'stageB', 'importance_E2_T1.npz'))['C'].astype(np.float64).reshape(-1, NB, 3)
    for pname, d in bj.items():
        ld = lambda b: np.load(os.path.join(R, 'beams', f'{pname}_{b}.npz'))
        o = {'pid': d['pid'], 'bin': d['bin'], 'pairs': {}}
        for a, b in (('polar_a', 'polar_b'), ('azim_a', 'azim_b'), ('half_a', 'half_b'), ('centre', 'other_bin')):
            if a not in d['beams']:
                continue
            za, zb = ld(a), ld(b)
            A_, B_ = 0.5 * (za['fa'].astype(np.float64) + za['fb']), 0.5 * (zb['fa'].astype(np.float64) + zb['fb'])
            den = 0.5 * (A_.sum() + B_.sum())
            dist = float(np.abs(A_ - B_).sum() / den)
            noise = float(0.5 * (np.abs(za['fa'].astype(np.float64) - za['fb']).sum() + np.abs(zb['fa'].astype(np.float64) - zb['fb']).sum()) * 0.5 / den)
            pr = {'response_distance': dist, 'noise': noise, 'encoding_max_abs_diff': float(np.abs(za['s'] - zb['s']).max()), 'same_bin': d['beams'][a]['bin'] == d['beams'][b]['bin'],
                  'flux_ratio_b_over_a': float(B_.sum() / A_.sum()),
                  'importance_C_diff_over_mean': float((Wc * (A_ - B_)).sum() / max(0.5 * (Wc * (A_ + B_)).sum(), 1e-30))}
            for sd in SEEDS:
                Ma = np.load(os.path.join(R, 'beams', f'M_{sd}_{pname}_{a}.npy')); Mb = np.load(os.path.join(R, 'beams', f'M_{sd}_{pname}_{b}.npy'))
                pr[f'{sd}_model_max_abs_diff'] = float(np.abs(Ma - Mb).max())
                pr[f'{sd}_model_vs_truth_a'] = float(np.abs(Ma - A_).sum() / A_.sum()); pr[f'{sd}_model_vs_truth_b'] = float(np.abs(Mb - B_).sum() / B_.sum())
            pr['counterexample'] = bool(pr['same_bin'] and pr['encoding_max_abs_diff'] == 0.0 and dist >= 0.2 and dist >= 4 * noise)
            o['pairs'][f'{a}|{b}'] = pr
        res['beams'][pname] = o
    # ---- classification by the predeclared rules
    prim = [f'{c}:{r}' for c, r in P['cases']['primary']]
    cl = {}
    ok = all(abs(res['validation'][c]['true_vs_stored_flux_ratio'] - 1) <= 0.01 for c in P['configs']) and \
        all(abs(res['cases'][k]['true_vs_stored_P']) <= max(4 * abs(res['cases'][k]['true_noise_P']), 2e-3) for k in res['cases'])
    cl['validation'] = bool(ok)
    est = [abs(np.mean([res['cases'][k]['learned'][sd]['estimator_effect'] for sd in SEEDS])) / max(abs(res['cases'][k]['learned_mean_P']), 1e-30) for k in prim]
    cl['source_estimator'] = {'effect_over_learned': est, 'label': 'SUPPORTED' if sum(v >= 0.25 for v in est) >= 2 else ('WEAKENED' if all(v < 0.10 for v in est) else 'INDETERMINATE')}
    ef = [res['cases'][k]['diag']['ENC_ALL']['explained_fraction'] for k in prim]
    sharp_cx = any(p['counterexample'] for kk, p in res['beams']['panel_S1']['pairs'].items() if not kk.startswith('centre'))
    n_dom = sum(v is not None and v >= 0.5 for v in ef)
    cl['source_angular_encoding'] = {'explained_fraction_ENC_ALL': ef, 'sharp_counterexample': bool(sharp_cx),
                                     'label': 'SUPPORTED' if (n_dom >= 2 and sharp_cx) else ('WEAKENED' if all(v is not None and v < 0.2 for v in ef) else ('CONTRIBUTING' if any(v is not None and v >= 0.2 for v in ef) else 'INDETERMINATE'))}
    sp = []
    for k in prim:
        d = res['cases'][k]['diag']; e_ = d['ENC_ALL']['P']
        sp.append({'ANG_PANEL': d['ANG_PANEL']['P'] / e_ if abs(e_) > 1e-9 else None, 'ANG_SPHERE': d['ANG_SPHERE']['P'] / e_ if abs(e_) > 1e-9 else None,
                   'ANG_BROAD': d['ANG_BROAD']['P'] / e_ if abs(e_) > 1e-9 else None, 'ENC_PANEL': d['ENC_PANEL']['P'] / e_ if abs(e_) > 1e-9 else None})
    cl['sender_specificity'] = {'shares_of_ENC_ALL': sp, 'sharp_sender': bool(all(s['ANG_PANEL'] is not None and s['ANG_PANEL'] >= 2 / 3 and abs(s['ANG_SPHERE']) < 0.2 and abs(s['ANG_BROAD']) < 0.2 for s in sp))}
    pa = [res['cases'][k]['diag']['ENC_ALL']['patch_corr_with_learned'] for k in prim]
    cl['learned_mapping'] = {'patch_corr': pa, 'label': 'WEAKENED' if (sum(c is not None and c >= 0.7 for c in pa) >= 2 and n_dom >= 2) else ('SUPPORTED' if all(v is not None and v < 0.2 for v in ef) else 'INDETERMINATE')}
    cl['geometry_inputs_determine_pose_max_err_m'] = max(max(v.values()) for v in res['geometry'].values())
    cx = {pn: {kk: p['counterexample'] for kk, p in o['pairs'].items()} for pn, o in res['beams'].items()}
    cl['counterexamples'] = cx
    eh_enc = {k: v for k, v in res['eh'].items() if k.endswith(':ENC_ALL') and (':C:' in k)}
    cl['ENC_ALL_region_C_E_H'] = eh_enc
    cl['intervention'] = 'NOT ADMISSIBLE (cause in the frozen input contract): STOP at attribution' if cl['source_angular_encoding']['label'] == 'SUPPORTED' else 'see worklog'
    res['classification'] = cl
    json.dump(res, open(os.path.join(R, 'attribution.json'), 'w'), indent=1)
    r4 = lambda v: None if v is None else round(v, 4)
    for k in res['cases']:
        row = res['cases'][k]
        print(k, 'learned P', {sd: r4(row['learned'][sd]['P']) for sd in SEEDS}, 'est', {sd: r4(row['learned'][sd]['estimator_effect']) for sd in SEEDS}, 'TRUE-stored', r4(row['true_vs_stored_P']), '+-', r4(row['true_noise_P']))
        print('    ', {v: (r4(d['P']), r4(d['noise']), r4(d['explained_fraction']), r4(d['patch_corr_with_learned'])) for v, d in row['diag'].items()})
    for k, v in res['eh'].items():
        if ':C:' in k or ':obj_panel:' in k:
            print('  EH', k, {a: r4(b) for a, b in v.items()})
    for pn, o in res['beams'].items():
        print(pn, {kk: (r4(p['response_distance']), r4(p['noise']), p['encoding_max_abs_diff'], p['counterexample'], r4(p['importance_C_diff_over_mean'])) for kk, p in o['pairs'].items()})
    print(json.dumps({k: v for k, v in cl.items() if k not in ('counterexamples', 'ENC_ALL_region_C_E_H')}))


if __name__ == '__main__':
    main()
