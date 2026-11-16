"""WL97 audit track E2: analysis of the independent delivery chain (numpy only).

  python e2_analyze.py <results dir>     reads ref_boxA.npz, wl95a_deliv_boxA.npz, flight_*.npz, aref_main.npz -> analysis.json (+ printed tables)

Metrics (own definitions, 54 box-A patches, exitance summed over channels):
  ratio = sum(arm) / sum(ref);  abs rel L1 = sum|arm - ref| / sum(ref);  corr = Pearson over patches.
"""
import json, math, os, sys
import numpy as np

RD = sys.argv[1]
NB = 64
DOM = 2 * math.pi / NB
ZB = ((np.arange(NB) // 16) + 0.5) / 4
W_T = ZB * DOM                                         # 4 rings x 16 sectors: cos-weighted solid angle, sums to pi
W_94 = ((np.arange(NB) // 8) + 0.5) / 8 * DOM          # the 8 x 8 assumption (an94.W as coded), sums to pi
W_U = np.full(NB, math.pi / NB)                        # uniform (valid for a Lambert receiver: radiance constant over bins)
PF = np.arange(9)                                      # panel-facing face = first 9 box-A patches
RHO = 0.5
rng = np.random.default_rng(9760001)


def met(a, r):
    return {'ratio': float(a.sum() / r.sum()), 'abs_rel_l1': float(np.abs(a - r).sum() / r.sum()), 'corr': float(np.corrcoef(a, r)[0, 1])}


def boot(draws, ref, nb=2000, sub=None):
    """bootstrap over draws of the pooled ratio / abs rel L1 -> SE and 95 % interval."""
    D = len(draws)
    n = sub or D
    rr = np.zeros(nb); ll = np.zeros(nb)
    for b in range(nb):
        m = draws[rng.integers(0, D, n)].mean(0)
        rr[b] = m.sum() / ref.sum(); ll[b] = np.abs(m - ref).sum() / ref.sum()
    q = lambda x: [float(np.quantile(x, 0.025)), float(np.quantile(x, 0.975))]
    return {'ratio_se': float(rr.std()), 'ratio_ci95': q(rr), 'l1_se': float(ll.std()), 'l1_ci95': q(ll)}


def per_draw(draws, ref):
    r = draws.sum(1) / ref.sum(); l = np.abs(draws - ref).sum(1) / ref.sum()
    f = lambda x: {'mean': float(x.mean()), 'sd': float(x.std(ddof=1)), 'min': float(x.min()), 'p10': float(np.quantile(x, .1)), 'median': float(np.median(x)),
                   'p90': float(np.quantile(x, .9)), 'max': float(x.max())}
    return {'ratio': f(r), 'abs_rel_l1': f(l)}


def conv(draws, ref, ns=(1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048)):
    """mean pooled metrics over disjoint groups of n draws (does more relation sampling reduce the error?)."""
    D = len(draws)
    o = {}
    for n in ns:
        if n > D:
            break
        g = draws[:D // n * n].reshape(D // n, n, -1).mean(1)
        o[str(n)] = {'groups': int(D // n), 'ratio_mean': float((g.sum(1) / ref.sum()).mean()),
                     'abs_rel_l1_mean': float((np.abs(g - ref).sum(1) / ref.sum()).mean()),
                     'abs_rel_l1_median': float(np.median(np.abs(g - ref).sum(1) / ref.sum()))}
    return o


def main():
    A = {}
    zr = np.load(os.path.join(RD, 'ref_boxA.npz'))
    ea, eb = zr['e2_p_a'].astype(np.float64), zr['e2_p_b'].astype(np.float64)       # [54, 64, 3]
    ex = lambda t, W: (t * W[None, :, None]).sum((1, 2))
    ref = {nm: 0.5 * (ex(ea, W) + ex(eb, W)) for nm, W in (('W_T', W_T), ('W_94', W_94), ('W_U', W_U))}
    R = ref['W_T']
    # ---------------------------------------------------------------- reference: noise scale and trustworthiness
    half = {nm: float(0.5 * np.abs(ex(ea, W) - ex(eb, W)).sum() / ref[nm].sum()) for nm, W in (('W_T', W_T), ('W_94', W_94), ('W_U', W_U))}
    # bin-scatter noise: for a Lambert receiver all 64 bins x 2 halves estimate the same radiance rho E / pi (per channel)
    allb = np.concatenate([ea, eb], 1)                                              # [54, 128, 3]
    wt2 = np.concatenate([W_T, W_T]) / 2
    sd_bin = allb.sum(2).std(1, ddof=1)                                             # per-patch sd of a single (bin, half) estimate (channels summed)
    se_wt = sd_bin * np.sqrt((wt2 ** 2).sum())                                      # SE of the W_T-weighted exitance of the 2-half mean
    A['reference'] = {'total_exitance': {k: float(v.sum()) for k, v in ref.items()},
                      'half_split_noise_n0': half,
                      'weights_effect_on_ref': {'W94_vs_WT_rel_l1': float(np.abs(ref['W_94'] - R).sum() / R.sum()), 'WU_vs_WT_rel_l1': float(np.abs(ref['W_U'] - R).sum() / R.sum()),
                                                'W94_total_ratio': float(ref['W_94'].sum() / R.sum()), 'WU_total_ratio': float(ref['W_U'].sum() / R.sum())},
                      'bin_scatter': {'expected_abs_rel_l1_of_ref_mean_gaussian': float((se_wt * math.sqrt(2 / math.pi)).sum() / R.sum()),
                                      'rel_se_of_total': float(np.sqrt((se_wt ** 2).sum()) / R.sum())},
                      'ring_means_rel_to_patch_mean': [float((allb.sum(2)[:, np.r_[r * 16:(r + 1) * 16, 64 + r * 16:64 + (r + 1) * 16]].mean(1).sum()) / allb.sum(2).mean(1).sum()) for r in range(4)],
                      'panel_face_share_of_total': float(R[PF].sum() / R.sum()), 'per_patch_exitance': R.tolist()}
    fa = os.path.join(RD, 'aref_main.npz')
    if os.path.exists(fa):
        za = np.load(fa)
        est = RHO * za['E'].sum(2)                                                  # [nbatch, 54]
        nbt = len(est)
        ar = est.mean(0); ar_se = est.std(0, ddof=1) / math.sqrt(nbt)
        h1, h2 = est[:nbt // 2].mean(0), est[nbt // 2:].mean(0)
        d = R - ar
        A['area_reference'] = {'samples_per_patch': int(nbt * 2 ** int(za['log2n'])), 'total': float(ar.sum()), 'rel_se_total': float(est.sum(1).std(ddof=1) / math.sqrt(nbt) / ar.sum()),
                               'own_half_split_noise': float(0.5 * np.abs(h1 - h2).sum() / ar.sum()),
                               'expected_abs_rel_l1_own_noise': float((ar_se * math.sqrt(2 / math.pi)).sum() / ar.sum()),
                               'CKF_ref_vs_area': {k: met(v, ar) for k, v in ref.items()},
                               'CKF_ref_vs_area_panel_face': met(R[PF], ar[PF]),
                               'CKF_minus_area_z_total': float((R.sum() - ar.sum()) / math.sqrt((se_wt ** 2).sum() + (est.sum(1).std(ddof=1) / math.sqrt(nbt)) ** 2)),
                               'chi2_per_patch_mean': float(np.mean(d ** 2 / (se_wt ** 2 + ar_se ** 2))),
                               'per_patch_area': ar.tolist(), 'per_patch_area_se': ar_se.tolist()}
    else:
        ar = None
    # ---------------------------------------------------------------- deployed realization: mine vs WL95A arrays, vs references
    zd = np.load(os.path.join(RD, 'flight_deployed.npz'))
    zo = np.load(os.path.join(RD, 'wl95a_deliv_boxA.npz'))
    dep = {k: zd[k].astype(np.float64).sum(-1)[0] for k in zd.files if k != 'hits'}
    name95 = {'M64_s16': 'A4', 'M64_spool': 'M64_spool', 'q': 'q256', 'X1c': 'X1c', 'X1': 'X1'}
    A['deployed'] = {'hits': int(zd['hits'][0])}
    for k, v in dep.items():
        o = {'vs_CKF_WT': met(v, R)}
        if ar is not None:
            o['vs_area'] = met(v, ar)
        if k in name95:
            w = zo[f'dep|{name95[k]}'].astype(np.float64)
            o['wl95a'] = met(w, 0.5 * (zo['ref_a'] + zo['ref_b']))
            o['mine_vs_wl95a_array'] = {'max_abs': float(np.abs(v - w).max()), 'rel_l1': float(np.abs(v - w).sum() / w.sum())}
        A['deployed'][k] = o
    A['wl95a_ref_array_vs_mine'] = float(np.abs(0.5 * (zo['ref_a'] + zo['ref_b']) - R).max())
    # WL95A pooled arrays re-analysed with the own metric code (their 64 draws)
    A['wl95a_pooled_recomputed'] = {}
    for k in ('A4', 'M64_spool', 'q256', 'X1c', 'X1'):
        dr_ = zo[f'draws|{k}'].astype(np.float64)
        A['wl95a_pooled_recomputed'][k] = {**met(dr_.mean(0), R), **boot(dr_, R), 'per_draw': per_draw(dr_, R)}
    # ---------------------------------------------------------------- own pooled draws
    for tag in ('shared2048', 'indep2048'):
        fn = os.path.join(RD, f'flight_{tag}.npz')
        if not os.path.exists(fn):
            continue
        z = np.load(fn)
        T = {}
        arms = [k for k in z.files if k not in ('hits', 'seed0', 'mode')]
        X = {k: z[k].astype(np.float64).sum(-1) for k in arms}                      # [D, 54]
        for k in arms:
            o = {'D': int(len(X[k])), 'pooled_vs_CKF_WT': {**met(X[k].mean(0), R), **boot(X[k], R)}, 'per_draw_vs_CKF': per_draw(X[k], R),
                 'pooled_vs_CKF_WT_panel_face': met(X[k].mean(0)[PF], R[PF]),
                 'first64_vs_CKF': {**met(X[k][:64].mean(0), R), **boot(X[k], R, 1000, 64)}, 'convergence_vs_CKF': conv(X[k], R),
                 'pooled_vs_CKF_W94': met(X[k].mean(0), ref['W_94']), 'pooled_vs_CKF_WU': met(X[k].mean(0), ref['W_U'])}
            if ar is not None:
                o['pooled_vs_area'] = {**met(X[k].mean(0), ar), **boot(X[k], ar)}
                o['pooled_vs_area_panel_face'] = met(X[k].mean(0)[PF], ar[PF])
                o['convergence_vs_area'] = conv(X[k], ar)
            # error of the pooled mean per patch in units of its own SE (is the pooled arm consistent with the reference?)
            se = X[k].std(0, ddof=1) / math.sqrt(len(X[k]))
            o['pooled_rel_se_total'] = float(X[k].sum(1).std(ddof=1) / math.sqrt(len(X[k])) / X[k].mean(0).sum())
            o['expected_abs_rel_l1_from_own_noise'] = float((se * math.sqrt(2 / math.pi)).sum() / R.sum())
            T[k] = o
        # paired contrasts on the same draws (bootstrap over draws), against the area reference when available
        tgt = ar if ar is not None else R

        def contrast(k1, k2, nb=2000):
            D = len(X[k1])
            f = lambda m: (m.sum() / tgt.sum(), np.abs(m - tgt).sum() / tgt.sum())
            base = np.array(f(X[k1].mean(0))) - np.array(f(X[k2].mean(0)))
            bs = np.zeros((nb, 2))
            for b in range(nb):
                ii = rng.integers(0, D, D)
                bs[b] = np.array(f(X[k1][ii].mean(0))) - np.array(f(X[k2][ii].mean(0)))
            return {'d_ratio': float(base[0]), 'd_ratio_se': float(bs[:, 0].std()), 'd_abs_rel_l1': float(base[1]), 'd_abs_rel_l1_se': float(bs[:, 1].std())}
        T['contrasts_vs_' + ('area' if ar is not None else 'CKF')] = {
            'source_points: M64_s16 - M64_spool': contrast('M64_s16', 'M64_spool'),
            'incident_projection: M64_spool - q': contrast('M64_spool', 'q'),
            'own_M64_object: M64own_spool - M64_spool': contrast('M64own_spool', 'M64_spool'),
            'table_vs_perray_projection: q - Xproj_both': contrast('q', 'Xproj_both'),
            'outgoing_direction (position projected): Xproj_both - Xdir_patchavg': contrast('Xproj_both', 'Xdir_patchavg'),
            'outgoing_direction (position exact): Xpos_binavg - X1': contrast('Xpos_binavg', 'X1'),
            'position (direction projected): Xproj_both - Xpos_binavg': contrast('Xproj_both', 'Xpos_binavg'),
            'position (direction exact): Xdir_patchavg - X1': contrast('Xdir_patchavg', 'X1'),
            'centre_vs_patchavg: X1c - Xdir_patchavg': contrast('X1c', 'Xdir_patchavg'),
            'total: M64_s16 - X1': contrast('M64_s16', 'X1'), 'q - X1c': contrast('q', 'X1c'), 'X1c - X1': contrast('X1c', 'X1')}
        T['hits_per_draw'] = {'mean': float(z['hits'].mean()), 'min': int(z['hits'].min()), 'max': int(z['hits'].max())}
        # where does the deployed draw sit in the own per-draw distribution?
        if 'M64_s16' in X:
            pr = X['M64_s16'].sum(1) / R.sum()
            T['deployed_percentile_of_binned_ratio'] = float((pr < dep['M64_s16'].sum() / R.sum()).mean())
            px = X['X1'].sum(1) / R.sum()
            T['deployed_percentile_of_X1_ratio'] = float((px < dep['X1'].sum() / R.sum()).mean())
        # X1 pooled vs both references: independent unbiased estimates of the same quantity
        A[tag] = T
    json.dump(A, open(os.path.join(RD, 'analysis.json'), 'w'), indent=1)
    # ---------------------------------------------------------------- printed summary
    print('REFERENCE', json.dumps({k: A['reference'][k] for k in ('total_exitance', 'half_split_noise_n0', 'weights_effect_on_ref', 'bin_scatter', 'ring_means_rel_to_patch_mean', 'panel_face_share_of_total')}, indent=1))
    if ar is not None:
        print('AREA', json.dumps({k: v for k, v in A['area_reference'].items() if not k.startswith('per_patch')}, indent=1))
    print('DEPLOYED (ratio / abs rel L1 / corr vs CKF W_T | WL95A | array diff)')
    for k, o in A['deployed'].items():
        if k == 'hits':
            continue
        m = o['vs_CKF_WT']; s = f"{k:16s} {m['ratio']:.3f} / {m['abs_rel_l1']:.3f} / {m['corr']:.2f}"
        if 'vs_area' in o:
            s += f" | area {o['vs_area']['ratio']:.3f} / {o['vs_area']['abs_rel_l1']:.3f}"
        if 'wl95a' in o:
            s += f" | WL95A {o['wl95a']['ratio']:.3f} / {o['wl95a']['abs_rel_l1']:.3f} | rel L1 between arrays {o['mine_vs_wl95a_array']['rel_l1']:.2e}"
        print(s)
    print('WL95A pooled arrays recomputed')
    for k, o in A['wl95a_pooled_recomputed'].items():
        print(f"{k:10s} {o['ratio']:.3f} (se {o['ratio_se']:.3f}) / {o['abs_rel_l1']:.3f} (se {o['l1_se']:.3f}) / {o['corr']:.2f}  per-draw ratio median {o['per_draw']['ratio']['median']:.3f}"
              f" L1 median {o['per_draw']['abs_rel_l1']['median']:.3f} max {o['per_draw']['abs_rel_l1']['max']:.3f}")
    for tag in ('shared2048', 'indep2048'):
        if tag not in A:
            continue
        print('OWN DRAWS', tag, A[tag]['hits_per_draw'])
        for k, o in A[tag].items():
            if not isinstance(o, dict) or 'pooled_vs_CKF_WT' not in o:
                continue
            m = o['pooled_vs_CKF_WT']; f64 = o['first64_vs_CKF']; pd_ = o['per_draw_vs_CKF']
            s = f"{k:14s} CKF {m['ratio']:.3f} (se {m['ratio_se']:.3f}) / {m['abs_rel_l1']:.3f} (se {m['l1_se']:.3f}) / {m['corr']:.3f}"
            if 'pooled_vs_area' in o:
                a = o['pooled_vs_area']; s += f" | area {a['ratio']:.3f} (se {a['ratio_se']:.3f}) / {a['abs_rel_l1']:.3f} (se {a['l1_se']:.3f}) / {a['corr']:.3f}"
            s += f" | 64: {f64['ratio']:.3f} (se {f64['ratio_se']:.3f}) / {f64['abs_rel_l1']:.3f} (se {f64['l1_se']:.3f})"
            s += f" | per-draw ratio med {pd_['ratio']['median']:.2f} p10 {pd_['ratio']['p10']:.2f} p90 {pd_['ratio']['p90']:.2f} max {pd_['ratio']['max']:.2f}; L1 med {pd_['abs_rel_l1']['median']:.2f} p90 {pd_['abs_rel_l1']['p90']:.2f} max {pd_['abs_rel_l1']['max']:.2f}"
            print(s)
        ck = [k for k in A[tag] if k.startswith('contrasts')][0]
        print(ck)
        for k, c in A[tag][ck].items():
            print(f"   {k:70s} d_ratio {c['d_ratio']:+.3f} (se {c['d_ratio_se']:.3f})  d_L1 {c['d_abs_rel_l1']:+.3f} (se {c['d_abs_rel_l1_se']:.3f})")
        for k in ('M64_s16', 'q', 'X1c', 'X1'):
            cv = A[tag][k]['convergence_vs_area' if ar is not None else 'convergence_vs_CKF']
            print('   convergence', k, ' '.join(f"{n}:{v['ratio_mean']:.3f}/{v['abs_rel_l1_mean']:.3f}" for n, v in cv.items()))
        print('   deployed percentile (binned ratio, X1 ratio):', A[tag].get('deployed_percentile_of_binned_ratio'), A[tag].get('deployed_percentile_of_X1_ratio'))


if __name__ == '__main__':
    main()
