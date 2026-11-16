"""WL97 audit, track C2: independent recomputation of the WL95A delivery statistics from results/ecro_attr95a/deliv.npz.
numpy only; no audited metric function is imported.  Run from the repository root:  python experiments/audit97/C2/c2_wl95a_delivery.py"""
import json
import numpy as np

z = np.load('results/ecro_attr95a/deliv.npz')
an = json.load(open('results/ecro_attr95a/an.json'))
ra, rb = z['ref_a'], z['ref_b']
ref = 0.5 * (ra + rb)
BOXA = np.arange(1130, 1184)
PFACE = np.arange(1130, 1139)
ARMS = ['A0_s8601', 'A0_s8602', 'A1', 'A2', 'A3', 'A4', 'M64_spool', 'q256', 'X1c', 'X1']
out = {}


def l1(x, m):
    return float(np.abs(x[m] - ref[m]).sum() / ref[m].sum())


def ratio(x, m):
    return float(x[m].sum() / ref[m].sum())


def boot(fn, X, B=4000, seed=97):
    rg = np.random.default_rng(seed)
    D = len(X)
    v = np.array([fn(X[rg.integers(0, D, D)].mean(0)) for _ in range(B)])
    return float(v.std(ddof=1)), [float(np.quantile(v, q)) for q in (0.025, 0.5, 0.975)]


def jack(fn, X):
    D = len(X)
    loo = np.array([fn((X.sum(0) - X[i]) / (D - 1)) for i in range(D)])
    return float(np.sqrt((D - 1) / D * ((loo - loo.mean()) ** 2).sum()))


for mn, m in (('boxA', BOXA), ('panel_face', PFACE)):
    n0 = float(0.5 * np.abs(ra[m] - rb[m]).sum() / ref[m].sum())
    out[f'n0|{mn}'] = n0
    out[f'ref_ratio_half|{mn}'] = float((ra[m].sum() - rb[m].sum()) / ref[m].sum())
    for k in ARMS:
        dep = z[f'dep|{k}']
        X = z[f'draws|{k}']
        pr = np.array([ratio(x, m) for x in X])
        pl = np.array([l1(x, m) for x in X])
        r = {'dep_ratio': ratio(dep, m), 'dep_l1': l1(dep, m), 'dep_corr': float(np.corrcoef(dep[m], ref[m])[0, 1]),
             'pool_ratio': ratio(X.mean(0), m), 'pool_l1': l1(X.mean(0), m), 'pool_corr': float(np.corrcoef(X.mean(0)[m], ref[m])[0, 1]),
             'pool_ratio_se_iid': float(pr.std(ddof=1) / np.sqrt(len(pr))), 'pool_l1_se_jk': jack(lambda v: l1(v, m), X),
             'draw_ratio_q': [float(np.quantile(pr, q)) for q in (0, .1, .25, .5, .75, .9, 1)],
             'draw_ratio_mean': float(pr.mean()), 'draw_ratio_sd': float(pr.std(ddof=1)),
             'draw_ratio_skew': float(((pr - pr.mean()) ** 3).mean() / pr.std() ** 3),
             'draw_l1_q': [float(np.quantile(pl, q)) for q in (.5, .9, 1)],
             'draw_l1_mean': float(pl.mean()),
             'dep_ratio_rank_in_draws': float((pr < ratio(dep, m)).mean()),
             'dep_l1_rank_in_draws': float((pl < l1(dep, m)).mean()),
             'frac_draws_ratio_gt1': float((pr > 1).mean()),
             'lag1_autocorr_ratio': float(np.corrcoef(pr[:-1], pr[1:])[0, 1])}
        bs, bq = boot(lambda v: ratio(v, m), X)
        r['pool_ratio_boot_se'] = bs
        r['pool_ratio_boot_ci95'] = [bq[0], bq[2]]
        bs, bq = boot(lambda v: l1(v, m), X)
        r['pool_l1_boot_se'] = bs
        r['pool_l1_boot_ci95'] = [bq[0], bq[2]]
        r['pool_ratio_halves'] = [ratio(X[0::2].mean(0), m), ratio(X[1::2].mean(0), m)]
        r['pool_l1_halves'] = [l1(X[0::2].mean(0), m), l1(X[1::2].mean(0), m)]
        r['pilot16_ratio_l1'] = [ratio(X[:16].mean(0), m), l1(X[:16].mean(0), m)]
        r['median_draw_exitance_ratio'] = ratio(np.median(X, 0), m)
        o = an['delivery']
        r['orig'] = {'dep_ratio': o[f'dep|{k}|{mn}']['ratio'], 'dep_l1': o[f'dep|{k}|{mn}']['abs_rel_l1'], 'pool_ratio': o[f'pool|{k}|{mn}']['ratio'],
                     'pool_l1': o[f'pool|{k}|{mn}']['abs_rel_l1'], 'pool_l1_se': o[f'pool|{k}|{mn}']['se_abs_rel_l1']}
        r['max_abs_dev_vs_orig'] = max(abs(r['dep_ratio'] - r['orig']['dep_ratio']), abs(r['dep_l1'] - r['orig']['dep_l1']),
                                       abs(r['pool_ratio'] - r['orig']['pool_ratio']), abs(r['pool_l1'] - r['orig']['pool_l1']),
                                       abs(r['pool_l1_se_jk'] - r['orig']['pool_l1_se']))
        out[f'{k}|{mn}'] = r

m = BOXA
n0 = out['n0|boxA']


def djk(k1, k2, fn=l1):
    X1, X2 = z[f'draws|{k1}'], z[f'draws|{k2}']
    D = len(X1)
    f = lambda a, b: fn(a, m) - fn(b, m)
    loo = np.array([f((X1.sum(0) - X1[i]) / (D - 1), (X2.sum(0) - X2[i]) / (D - 1)) for i in range(D)])
    rg = np.random.default_rng(97)
    bs = []
    for _ in range(4000):
        i = rg.integers(0, D, D)
        bs.append(f(X1[i].mean(0), X2[i].mean(0)))
    bs = np.array(bs)
    return {'val': f(X1.mean(0), X2.mean(0)), 'se_jk': float(np.sqrt((D - 1) / D * ((loo - loo.mean()) ** 2).sum())), 'se_boot': float(bs.std(ddof=1)),
            'ci95_boot': [float(np.quantile(bs, .025)), float(np.quantile(bs, .975))], 'p_le_0_boot': float((bs <= 0).mean())}


E0 = out['A4|boxA']['dep_l1'] - n0
fl = {'n0': n0, 'E0': E0, 'Rs': E0 - (out['M64_spool|boxA']['pool_l1'] - n0), 'Rs_se': out['M64_spool|boxA']['pool_l1_se_jk'],
      'Ra': djk('M64_spool', 'X1c'), 'Rp': djk('X1c', 'X1'), 'Rq_inc': djk('M64_spool', 'q256'), 'Rq_out': djk('q256', 'X1c'),
      'A4_to_spool': djk('A4', 'M64_spool'), 'thr_0.25E0': 0.25 * E0, 'E_X1_pool': out['X1|boxA']['pool_l1'] - n0, 'orig': an['flight']}
fl['ratio_steps'] = {f'{a}->{b}': djk(a, b, ratio) for a, b in (('A4', 'M64_spool'), ('M64_spool', 'q256'), ('q256', 'X1c'), ('X1c', 'X1'))}
fl['E0_alt_pooledA4'] = out['A4|boxA']['pool_l1'] - n0
fl['A4_mean_single_draw_l1'] = out['A4|boxA']['draw_l1_mean']
fl['X1_mean_single_draw_l1'] = out['X1|boxA']['draw_l1_mean']
out['flight'] = fl

rk = {}
A4m = z['draws|A4'].mean(0)
D = 64
fr = lambda a, b: a[m].sum() / b[m].sum()
md = lambda a, b: np.abs(a[m] - b[m]).sum() / b[m].sum()
se = lambda loo: float(np.sqrt((D - 1) / D * ((loo - loo.mean()) ** 2).sum()))
for X in ('A1', 'A2', 'A3'):
    Xm = z[f'draws|{X}'].mean(0)
    XX, AA = z[f'draws|{X}'], z['draws|A4']
    loo_fr = np.array([fr((XX.sum(0) - XX[i]) / (D - 1), (AA.sum(0) - AA[i]) / (D - 1)) for i in range(D)])
    loo_md = np.array([md((XX.sum(0) - XX[i]) / (D - 1), (AA.sum(0) - AA[i]) / (D - 1)) for i in range(D)])
    rk[X] = {'flux_vs_A4': float(fr(Xm, A4m)), 'flux_se_jk': se(loo_fr), 'map_vs_A4': float(md(Xm, A4m)), 'map_se_jk': se(loo_md),
             'flux_vs_A4_deployed': float(fr(z[f'dep|{X}'], z['dep|A4'])), 'map_vs_A4_deployed': float(md(z[f'dep|{X}'], z['dep|A4'])),
             'flux_vs_ref_pool': ratio(Xm, m), 'l1_vs_ref_pool': l1(Xm, m),
             'orig': {k: an['rank16'][X][k] for k in ('flux_ratio_vs_A4', 'map_rel_l1_vs_A4')}}
out['rank16'] = rk

r_pk = int(np.argmax(ref[BOXA]))
out['peak'] = {'ref_peak_local_index': r_pk}
for k in ('X1', 'A4', 'A0_s8601'):
    xm = z[f'draws|{k}'].mean(0)[BOXA]
    out['peak'][k] = {'value_at_ref_peak_ratio_pool': float(xm[r_pk] / ref[BOXA][r_pk]), 'arm_peak_local': int(np.argmax(xm))}

ht = {}
for k in ('A4', 'q256', 'X1c', 'X1'):
    X = z[f'draws|{k}'][:, BOXA]
    tot = X.sum(1)
    srt = np.sort(tot)[::-1]
    ht[k] = {'top1_share': float(srt[0] / tot.sum()), 'top5_share': float(srt[:5].sum() / tot.sum()), 'cv_total': float(tot.std(ddof=1) / tot.mean()),
             'kurtosis_total': float(((tot - tot.mean()) ** 4).mean() / tot.var() ** 2),
             'pool_ratio_without_top5': float(np.sort(tot)[:-5].mean() / ref[BOXA].sum()),
             'max_patch_draw_over_ref_patch': float((X / np.maximum(ref[BOXA][None], 1e-30)).max()),
             'frac_zero_patch_draws': float((X == 0).mean())}
ht['pface_share_of_boxA_ref'] = float(ref[PFACE].sum() / ref[BOXA].sum())
out['heavy_tail'] = ht
json.dump(out, open('results/audit97/C2/wl95a_delivery_recompute.json', 'w'), indent=1)

print('n0 boxA', out['n0|boxA'], 'pface', out['n0|panel_face'], 'ref ratio half-diff', out['ref_ratio_half|boxA'])
for mn in ('boxA', 'panel_face'):
    print('==', mn)
    for k in ARMS:
        r = out[f'{k}|{mn}']
        print(f"{k:10s} dep {r['dep_ratio']:.3f}/{r['dep_l1']:.3f}/{r['dep_corr']:.2f} pool {r['pool_ratio']:.3f}(se {r['pool_ratio_se_iid']:.3f}, boot {r['pool_ratio_boot_ci95'][0]:.3f}..{r['pool_ratio_boot_ci95'][1]:.3f})"
              f" /{r['pool_l1']:.3f}(jk {r['pool_l1_se_jk']:.3f}; boot {r['pool_l1_boot_ci95'][0]:.3f}..{r['pool_l1_boot_ci95'][1]:.3f}) corr {r['pool_corr']:.2f} dev {r['max_abs_dev_vs_orig']:.1e}")
        print(f"           draw ratio q0/10/25/50/75/90/100 {np.round(r['draw_ratio_q'], 3)} sd {r['draw_ratio_sd']:.3f} skew {r['draw_ratio_skew']:.2f} dep-rank {r['dep_ratio_rank_in_draws']:.2f} l1-rank {r['dep_l1_rank_in_draws']:.2f}"
              f" >1: {r['frac_draws_ratio_gt1']:.2f} halves {np.round(r['pool_ratio_halves'], 3)} l1 halves {np.round(r['pool_l1_halves'], 3)} pilot16 {np.round(r['pilot16_ratio_l1'], 3)} med-ex {r['median_draw_exitance_ratio']:.3f} lag1 {r['lag1_autocorr_ratio']:.2f}"
              f" l1 draws med/p90/max {np.round(r['draw_l1_q'], 3)} mean {r['draw_l1_mean']:.3f}")
print(json.dumps(fl, indent=0))
print(json.dumps(rk))
print(json.dumps(out['peak']))
print(json.dumps(ht, indent=0))
