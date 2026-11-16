"""WL97 audit track E2: supplementary local tables (numpy only) -> <results dir>/summary_tables.json

  python e2_tables.py <results dir>

Adds to e2_analyze.py: (i) per-patch consistency of the CKF reference with the own area-form reference (chi2 on patches with signal),
(ii) the WL95A arrays (64 draws) scored against the area-form reference and located inside the own 2048-draw distribution of 64-draw
pools, (iii) the WL95A decomposition (E = abs rel L1 - n0; Rs / Rq / Ra / Rp) recomputed from own draws, (iv) 2 x 2 factorial
(direction x position) with bootstrap SE over draws, (v) per-draw distribution summaries, (vi) the n-draw convergence of the map error
with its noise-only expectation.
"""
import json, math, os, sys
import numpy as np

RD = sys.argv[1]
NB = 64
DOM = 2 * math.pi / NB
W_T = ((np.arange(NB) // 16) + 0.5) / 4 * DOM
RHO = 0.5
rng = np.random.default_rng(9761001)


def met(a, r):
    return {'ratio': float(a.sum() / r.sum()), 'abs_rel_l1': float(np.abs(a - r).sum() / r.sum()), 'corr': float(np.corrcoef(a, r)[0, 1])}


zr = np.load(os.path.join(RD, 'ref_boxA.npz'))
ea, eb = zr['e2_p_a'].astype(np.float64), zr['e2_p_b'].astype(np.float64)
ex = lambda t: (t * W_T[None, :, None]).sum((1, 2))
Ra_, Rb_ = ex(ea), ex(eb)
R = 0.5 * (Ra_ + Rb_)
za = np.load(os.path.join(RD, 'aref_main.npz'))
est = RHO * za['E'].sum(2)
AR = est.mean(0); AR_se = est.std(0, ddof=1) / math.sqrt(len(est))
allb = np.concatenate([ea, eb], 1).sum(2)                                   # [54, 128] single (bin, half) radiance estimates
wt2 = np.concatenate([W_T, W_T]) / 2
se_ref = allb.std(1, ddof=1) * np.sqrt((wt2 ** 2).sum())                    # SE of the W_T exitance of the two-half mean (bins as replicates)
S = {}

# (i) reference consistency
sig = AR > 1e-3 * AR.max()
zsc = (R - AR)[sig] / np.sqrt(se_ref[sig] ** 2 + AR_se[sig] ** 2)
S['reference_vs_area'] = {
    'patches_with_signal': int(sig.sum()), 'patches_zero_in_both': int(((AR == 0) & (R == 0)).sum()),
    'ratio_total': float(R.sum() / AR.sum()), 'rel_se_total_ref': float(np.sqrt((se_ref ** 2).sum()) / R.sum()),
    'abs_rel_l1': float(np.abs(R - AR).sum() / AR.sum()), 'expected_abs_rel_l1_from_ref_noise': float((se_ref * math.sqrt(2 / math.pi)).sum() / AR.sum()),
    'half_split_n0': float(0.5 * np.abs(Ra_ - Rb_).sum() / R.sum()),
    'half_a_vs_area': met(Ra_, AR), 'half_b_vs_area': met(Rb_, AR),
    'chi2_per_patch': float(np.mean(zsc ** 2)), 'z_max_abs': float(np.abs(zsc).max()), 'z_mean': float(zsc.mean()),
    'frac_abs_z_gt_2': float((np.abs(zsc) > 2).mean()),
    'top8_patches_by_exitance': [{'patch': int(1130 + i), 'area': float(AR[i]), 'ckf': float(R[i]), 'z': float((R[i] - AR[i]) / math.sqrt(se_ref[i] ** 2 + AR_se[i] ** 2))}
                                 for i in np.argsort(-AR)[:8]]}

# own draws
Z = {t: np.load(os.path.join(RD, f'flight_{t}.npz')) for t in ('shared2048', 'indep2048')}
X = {t: {k: Z[t][k].astype(np.float64).sum(-1) for k in Z[t].files if k not in ('hits', 'seed0', 'mode')} for t in Z}

# (ii) WL95A arrays against the area reference, and their place among own 64-draw pools
zo = np.load(os.path.join(RD, 'wl95a_deliv_boxA.npz'))
name95 = {'A4': 'M64_s16', 'M64_spool': 'M64_spool', 'q256': 'q', 'X1c': 'X1c', 'X1': 'X1'}
S['wl95a_arrays'] = {}
for k95, k in name95.items():
    d95 = zo[f'draws|{k95}'].astype(np.float64)
    o = {'D': int(len(d95)), 'pooled_vs_CKF': met(d95.mean(0), R), 'pooled_vs_area': met(d95.mean(0), AR),
         'ratio_se': float((d95.sum(1) / R.sum()).std(ddof=1) / math.sqrt(len(d95)))}
    # own distribution of pools of 64 draws (both own sets, disjoint groups)
    pools = np.concatenate([X[t][k].reshape(32, 64, 54).mean(1) for t in X])
    pr = pools.sum(1) / R.sum(); pl = np.abs(pools - R).sum(1) / R.sum()
    o['own_64pool_ratio'] = {'mean': float(pr.mean()), 'sd': float(pr.std(ddof=1)), 'min': float(pr.min()), 'max': float(pr.max())}
    o['own_64pool_abs_rel_l1'] = {'mean': float(pl.mean()), 'sd': float(pl.std(ddof=1)), 'min': float(pl.min()), 'max': float(pl.max())}
    o['wl95a_z_ratio_in_own_pools'] = float((o['pooled_vs_CKF']['ratio'] - pr.mean()) / pr.std(ddof=1))
    o['wl95a_quantile_l1_in_own_pools'] = float((pl < o['pooled_vs_CKF']['abs_rel_l1']).mean())
    S['wl95a_arrays'][k95] = o

# (iii) WL95A decomposition in its own terms (E = abs rel L1 - n0, CKF reference), own draws, all 4096 pooled + bootstrap
n0 = S['reference_vs_area']['half_split_n0']
ALL = {k: np.concatenate([X[t][k] for t in X]) for k in X['shared2048']}


def boot_stat(fun, nb=1000):
    D = len(ALL['X1'])
    v = np.zeros(nb)
    for b in range(nb):
        ii = rng.integers(0, D, D)
        v[b] = fun(ii)
    return float(v.std())


def L1(k, ii=None, ref=R):
    m = ALL[k].mean(0) if ii is None else ALL[k][ii].mean(0)
    return np.abs(m - ref).sum() / ref.sum()


dec = {}
for nm, (k1, k2) in {'Rs_like(M64_s16 - M64_spool)': ('M64_s16', 'M64_spool'), 'Rq_incident(M64_spool - q)': ('M64_spool', 'q'),
                     'Rq_outgoing(q - X1c)': ('q', 'X1c'), 'Ra(M64_spool - X1c)': ('M64_spool', 'X1c'), 'Rp(X1c - X1)': ('X1c', 'X1'),
                     'own: outgoing, patch-avg position (Xproj_both - Xdir_patchavg)': ('Xproj_both', 'Xdir_patchavg'),
                     'own: outgoing, exact position (Xpos_binavg - X1)': ('Xpos_binavg', 'X1'),
                     'own: position, exact direction (Xdir_patchavg - X1)': ('Xdir_patchavg', 'X1'),
                     'own: position, binned direction (Xproj_both - Xpos_binavg)': ('Xproj_both', 'Xpos_binavg'),
                     'own: centre sampling (X1c - Xdir_patchavg)': ('X1c', 'Xdir_patchavg')}.items():
    dec[nm] = {'vs_CKF': float(L1(k1) - L1(k2)), 'vs_CKF_se': boot_stat(lambda ii: L1(k1, ii) - L1(k2, ii)),
               'vs_area': float(L1(k1, None, AR) - L1(k2, None, AR)), 'vs_area_se': boot_stat(lambda ii: L1(k1, ii, AR) - L1(k2, ii, AR)),
               'd_ratio': float((ALL[k1].mean(0).sum() - ALL[k2].mean(0).sum()) / AR.sum()),
               'd_ratio_se': float(((ALL[k1] - ALL[k2]).sum(1) / AR.sum()).std(ddof=1) / math.sqrt(len(ALL[k1])))}
S['decomposition_4096_draws'] = dec
S['E_minus_n0_4096_draws'] = {k: float(L1(k) - n0) for k in ALL}
S['pooled_4096'] = {k: {'vs_CKF': met(ALL[k].mean(0), R), 'vs_area': met(ALL[k].mean(0), AR),
                        'ratio_se': float((ALL[k].sum(1) / AR.sum()).std(ddof=1) / math.sqrt(len(ALL[k]))),
                        'l1_vs_area_boot_se': boot_stat(lambda ii, k=k: L1(k, ii, AR), 400),
                        'expected_l1_from_own_noise': float((ALL[k].std(0, ddof=1) / math.sqrt(len(ALL[k])) * math.sqrt(2 / math.pi)).sum() / AR.sum())}
                    for k in ALL}

# (v) per-draw distributions (vs area reference), 4096 draws
pd_ = {}
for k in ('M64_s16', 'q', 'X1c', 'X1', 'Xdir_patchavg'):
    r = ALL[k].sum(1) / AR.sum(); l = np.abs(ALL[k] - AR).sum(1) / AR.sum()
    q = lambda x: {'mean': float(x.mean()), 'sd': float(x.std(ddof=1)), 'p05': float(np.quantile(x, .05)), 'p10': float(np.quantile(x, .1)), 'median': float(np.median(x)),
                   'p90': float(np.quantile(x, .9)), 'p99': float(np.quantile(x, .99)), 'max': float(x.max())}
    pd_[k] = {'ratio': q(r), 'abs_rel_l1': q(l), 'frac_draws_ratio_below_1': float((r < 1).mean())}
pd_['frac_draws_X1_l1_below_binned_l1_paired'] = float((np.abs(ALL['X1'] - AR).sum(1) < np.abs(ALL['M64_s16'] - AR).sum(1)).mean())
S['per_draw_vs_area'] = pd_

# (vi) convergence with n pooled draws (vs area): mean abs rel L1 over disjoint groups, and the noise-only expectation
cv = {}
for k in ('M64_s16', 'q', 'X1c', 'Xdir_patchavg', 'X1'):
    o = {}
    sd = ALL[k].std(0, ddof=1)
    for n in (1, 4, 16, 64, 256, 1024, 4096):
        g = ALL[k][:len(ALL[k]) // n * n].reshape(-1, n, 54).mean(1)
        o[str(n)] = {'rays_per_patch': 512 * n, 'ratio': float(g.sum(1).mean() / AR.sum()), 'abs_rel_l1_mean': float((np.abs(g - AR).sum(1) / AR.sum()).mean()),
                     'noise_only_gaussian': float((sd / math.sqrt(n) * math.sqrt(2 / math.pi)).sum() / AR.sum())}
    cv[k] = o
S['convergence_vs_area'] = cv

# deployed realization
zd = np.load(os.path.join(RD, 'flight_deployed.npz'))
dep = {k: zd[k].astype(np.float64).sum(-1)[0] for k in zd.files if k != 'hits'}
S['deployed'] = {k: {'vs_CKF': met(v, R), 'vs_area': met(v, AR), 'percentile_of_ratio_in_4096_draws': float((ALL[k].sum(1) < v.sum()).mean()),
                     'percentile_of_l1_in_4096_draws': float((np.abs(ALL[k] - AR).sum(1) < np.abs(v - AR).sum()).mean())} for k, v in dep.items()}
for fn in ('binres.json', 'mquad.json'):
    p = os.path.join(RD, fn)
    if os.path.exists(p):
        S[fn[:-5]] = json.load(open(p))
json.dump(S, open(os.path.join(RD, 'summary_tables.json'), 'w'), indent=1)


def pr(*a):
    print(*a)


pr('REF vs AREA', json.dumps({k: v for k, v in S['reference_vs_area'].items() if k != 'top8_patches_by_exitance'}))
for t in S['reference_vs_area']['top8_patches_by_exitance']:
    pr('   ', t)
pr('WL95A arrays (64 draws)')
for k, o in S['wl95a_arrays'].items():
    pr(f"  {k:10s} CKF {o['pooled_vs_CKF']['ratio']:.3f}/{o['pooled_vs_CKF']['abs_rel_l1']:.3f} area {o['pooled_vs_area']['ratio']:.3f}/{o['pooled_vs_area']['abs_rel_l1']:.3f}"
       f" | own 64-pools ratio {o['own_64pool_ratio']['mean']:.3f} sd {o['own_64pool_ratio']['sd']:.3f} [{o['own_64pool_ratio']['min']:.3f},{o['own_64pool_ratio']['max']:.3f}]"
       f" L1 {o['own_64pool_abs_rel_l1']['mean']:.3f} sd {o['own_64pool_abs_rel_l1']['sd']:.3f} [{o['own_64pool_abs_rel_l1']['min']:.3f},{o['own_64pool_abs_rel_l1']['max']:.3f}]"
       f" | z_ratio {o['wl95a_z_ratio_in_own_pools']:+.2f} q_L1 {o['wl95a_quantile_l1_in_own_pools']:.2f}")
pr('POOLED 4096 own draws (ratio/L1/corr vs CKF | vs area | ratio se | L1 boot se | noise-only L1)')
for k, o in S['pooled_4096'].items():
    pr(f"  {k:14s} {o['vs_CKF']['ratio']:.3f}/{o['vs_CKF']['abs_rel_l1']:.3f}/{o['vs_CKF']['corr']:.3f} | {o['vs_area']['ratio']:.3f}/{o['vs_area']['abs_rel_l1']:.3f}/{o['vs_area']['corr']:.3f}"
       f" | {o['ratio_se']:.4f} | {o['l1_vs_area_boot_se']:.4f} | {o['expected_l1_from_own_noise']:.4f}")
pr('DECOMPOSITION (d abs rel L1 vs CKF (se) | vs area (se) | d ratio (se))')
for k, o in dec.items():
    pr(f"  {k:62s} {o['vs_CKF']:+.3f} ({o['vs_CKF_se']:.3f}) | {o['vs_area']:+.3f} ({o['vs_area_se']:.3f}) | {o['d_ratio']:+.3f} ({o['d_ratio_se']:.3f})")
pr('PER DRAW vs area')
for k, o in pd_.items():
    if isinstance(o, dict):
        pr(f"  {k:14s} ratio mean {o['ratio']['mean']:.3f} sd {o['ratio']['sd']:.3f} p05 {o['ratio']['p05']:.2f} med {o['ratio']['median']:.2f} p90 {o['ratio']['p90']:.2f} p99 {o['ratio']['p99']:.2f} max {o['ratio']['max']:.2f}"
           f" | L1 mean {o['abs_rel_l1']['mean']:.3f} med {o['abs_rel_l1']['median']:.3f} p90 {o['abs_rel_l1']['p90']:.2f} max {o['abs_rel_l1']['max']:.2f} | frac ratio<1 {o['frac_draws_ratio_below_1']:.2f}")
pr('  frac draws X1 L1 < binned L1 (paired):', pd_['frac_draws_X1_l1_below_binned_l1_paired'])
pr('CONVERGENCE vs area: n draws: ratio / mean abs rel L1 / noise-only expectation')
for k, o in cv.items():
    pr(f"  {k:14s}", ' '.join(f"{n}:{v['abs_rel_l1_mean']:.3f}({v['noise_only_gaussian']:.3f})" for n, v in o.items()))
pr('DEPLOYED')
for k, o in S['deployed'].items():
    pr(f"  {k:14s} CKF {o['vs_CKF']['ratio']:.3f}/{o['vs_CKF']['abs_rel_l1']:.3f} area {o['vs_area']['ratio']:.3f}/{o['vs_area']['abs_rel_l1']:.3f} pct ratio {o['percentile_of_ratio_in_4096_draws']:.3f} pct L1 {o['percentile_of_l1_in_4096_draws']:.3f}")
