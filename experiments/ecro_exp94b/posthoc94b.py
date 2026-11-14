"""WL94B POST HOC (labelled; after the STOP_INVALID_INTERVENTION audit gate; CPU only, no training, no new render).
Face-level coverage of panel-mediated transport in the 96 training items vs the CK2 panel-facing face, and the arithmetic exposure
bounds of item-level reallocation. Uses only the frozen audit aggregates (results/audit/tgt/*.npz) and CKF_none.

  python posthoc94b.py <R> <R86>  -> results/an/posthoc94b.json
"""
import json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import b94 as B  # noqa: E402


def main(R, R86):
    import scene86 as S
    MT, C94, AN = B.MT, B.C94, B.AN
    files = C94.sel_files(R, 'train'); names = [r['file'] for r in files]
    G0 = S.Geom(json.loads(str(np.load(os.path.join(R86, 'data', 'train', names[0]))['geom'])))
    faces = {r: MT.rect_patches(G0, r) for r in B.BOXA + B.BOXB}
    zf = np.load(os.path.join(R, 'results', 'fields', 'CKF_none.npz'))
    Xr = {p: 0.5 * (zf[f'{p}_a'].astype(np.float64) + zf[f'{p}_b'].astype(np.float64)) for p in B.PARTS}
    fr = sum(Xr.values()); den = fr.mean(1, keepdims=True) + 0.05 * fr.mean()
    nr = {p: (Xr[p] / den).sum((1, 2)) for p in B.PARTS}
    Tp = B.AN.masks(R)['T_panel_face']

    def shares(nm, idx):
        e2p, e2o, e3p, e3o = (nm[i][idx].sum() for i in range(4))
        return {'e2p_of_e2': e2p / max(e2p + e2o, 1e-30), 'e3p_of_e3': e3p / max(e3p + e3o, 1e-30),
                'pm_of_face': (e2p + e3p) / max(e2p + e2o + e3p + e3o, 1e-30)}
    ck = shares([nr[p] for p in B.PARTS], Tp)
    rows = []; allface = []
    for nm_ in names:
        a = np.load(os.path.join(R, 'results', 'audit', 'tgt', nm_))
        nm = a['nm'].sum(1)
        fs = {r: shares(nm, idx) for r, idx in faces.items()}
        best = max(fs, key=lambda r: fs[r]['pm_of_face'])
        rows.append({'file': nm_, 'best_face': int(best), **{f'best_{k}': float(v) for k, v in fs[best].items()},
                     'boxA_best_pm_of_face': float(max(fs[r]['pm_of_face'] for r in B.BOXA))})
        for r, idx in faces.items():
            allface.append((fs[r]['pm_of_face'], float(nm[:, idx].sum())))
    pm_best = np.array([r['best_pm_of_face'] for r in rows]); pmA = np.array([r['boxA_best_pm_of_face'] for r in rows])
    af = np.array(allface)
    au = MT.jl(os.path.join(R, 'results', 'audit', 'audit.json'))
    PM = np.array([r['tgt']['PM'] for r in au['rows']])
    cmean = np.mean([v['C'] for v in au['gates']['G_EXPOSURE']['per_seed'].values()])
    q = np.sort(PM)[::-1]
    out = {'label': 'POST HOC (after the audit stop; descriptive; no threshold, allocation or outcome is changed)',
           'ck2_panel_face': ck,
           'items_best_box_face': {'median_pm_of_face': float(np.median(pm_best)), 'p90': float(np.quantile(pm_best, 0.9)), 'max': float(pm_best.max()),
                                   'n_reaching_ck2_pm_of_face': int((pm_best >= ck['pm_of_face']).sum()),
                                   'n_reaching_half_ck2': int((pm_best >= 0.5 * ck['pm_of_face']).sum())},
           'items_best_boxA_face': {'median': float(np.median(pmA)), 'max': float(pmA.max()), 'n_reaching_ck2': int((pmA >= ck['pm_of_face']).sum())},
           'box_face_mass_share_on_faces_reaching_half_ck2': float(af[af[:, 0] >= 0.5 * ck['pm_of_face'], 1].sum() / af[:, 1].sum()),
           'exposure_bounds_vs_C_mean': {'protocol_D_sigma0.5_top_quartile': float(np.mean([v['ratio'] for v in au['gates']['G_EXPOSURE']['per_seed'].values()])),
                                         'all_steps_on_top_quartile': float(q[:24].mean() / cmean), 'all_steps_on_top_item': float(q[0] / cmean)},
           'rows': rows}
    B.jd(out, os.path.join(R, 'results', 'an', 'posthoc94b.json'))
    print('POSTHOC done', flush=True)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
    os._exit(0)
