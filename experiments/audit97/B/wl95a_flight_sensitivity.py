"""WL97 audit track B: sensitivity of the WL95A FLIGHT_ATTRIBUTION verdict to (i) the 16 -> 64 draw extension and (ii) the protocol's
choice of E0 = the single deployed relation draw. Recomputed independently from results/ecro_attr95a/deliv.npz (per-draw box-A exitance
of every arm) with the frozen rule of a95.py (lines 449-475). numpy only.

  python wl95a_flight_sensitivity.py <deliv.npz> <out_json>
"""
import json, sys
import numpy as np

z = np.load(sys.argv[1])
BOXA = np.arange(1130, 1184)
ra, rb = z['ref_a'], z['ref_b']
ref = 0.5 * (ra + rb)
n0 = float(0.5 * np.abs(ra[BOXA] - rb[BOXA]).sum() / ref[BOXA].sum())


def e_abs(m):
    return float(np.abs(m[BOXA] - ref[BOXA]).sum() / ref[BOXA].sum())


def jk_se(X, f):
    D = len(X)
    loo = np.array([f((X.sum(0) - X[i]) / (D - 1)) for i in range(D)])
    return float(np.sqrt((D - 1) / D * ((loo - loo.mean()) ** 2).sum()))


def diff_se(X1, X2):
    D = len(X1)
    f = lambda m1, m2: e_abs(m1) - e_abs(m2)
    loo = np.array([f((X1.sum(0) - X1[i]) / (D - 1), (X2.sum(0) - X2[i]) / (D - 1)) for i in range(D)])
    return f(X1.mean(0), X2.mean(0)), float(np.sqrt((D - 1) / D * ((loo - loo.mean()) ** 2).sum()))


def verdict(E0, D):
    g = lambda k: z[f'draws|{k}'][:D]
    Rs = E0 - (e_abs(g('M64_spool').mean(0)) - n0)
    se_s = jk_se(g('M64_spool'), e_abs)
    Ra, se_a = diff_se(g('M64_spool'), g('X1c'))
    Rp, se_p = diff_se(g('X1c'), g('X1'))
    EX1 = e_abs(g('X1').mean(0)) - n0
    cnt = lambda r, se: bool(r >= 0.25 * E0 and r > 2 * se)
    s_ok, a_ok = cnt(Rs, se_s), cnt(Ra, se_a)
    if EX1 > 0.5 * E0:
        fa = 'INCONCLUSIVE'
    elif s_ok and a_ok:
        fa = 'MIXED'
    elif s_ok and not a_ok and Rs >= 0.5 * E0:
        fa = 'SAMPLING_DOMINANT'
    elif a_ok and not s_ok:
        fa = 'ANGULAR_PROJECTION_EVIDENCED'
    else:
        fa = 'INCONCLUSIVE'
    return {'D': D, 'E0': E0, 'Rs': Rs, 'Rs_se': se_s, 'Ra': Ra, 'Ra_se': se_a, 'Rp': Rp, 'Rp_se': se_p, 'E_X1_pool': EX1,
            'threshold_0.25E0': 0.25 * E0, 'Rs_counts': s_ok, 'Ra_counts': a_ok, 'verdict': fa}


out = {'n0': n0}
per = np.array([e_abs(x) for x in z['draws|A4']])
dep = e_abs(z['dep|A4'])
out['A4_single_draw_abs_rel_l1'] = {'deployed_86001': dep, 'draws_mean': float(per.mean()), 'median': float(np.median(per)),
                                    'min': float(per.min()), 'p10': float(np.quantile(per, 0.1)), 'p90': float(np.quantile(per, 0.9)),
                                    'draws_below_deployed': int((per < dep).sum()), 'n_draws': len(per)}
rat = np.array([x[BOXA].sum() / ref[BOXA].sum() for x in z['draws|A4']])
out['A4_flux_ratio'] = {'deployed': float(z['dep|A4'][BOXA].sum() / ref[BOXA].sum()), 'draws_mean': float(rat.mean()),
                        'draws_sd': float(rat.std(ddof=1)), 'se_of_mean': float(rat.std(ddof=1) / np.sqrt(len(rat))),
                        'draws_below_deployed': int((rat < z['dep|A4'][BOXA].sum() / ref[BOXA].sum()).sum()), 'min': float(rat.min())}
for nm in ('M64_spool', 'q256', 'X1c', 'X1'):
    r_ = np.array([x[BOXA].sum() / ref[BOXA].sum() for x in z[f'draws|{nm}']])
    out[f'{nm}_flux_ratio'] = {'draws_mean': float(r_.mean()), 'se_of_mean': float(r_.std(ddof=1) / np.sqrt(len(r_))), 'median': float(np.median(r_))}
E0_dep = dep - n0
out['frozen_rule_E0_deployed'] = {f'D{D}': verdict(E0_dep, D) for D in (16, 32, 64)}
out['alt_E0_mean_single_draw'] = {f'D{D}': verdict(float(per[:D].mean()) - n0, D) for D in (16, 64)}
out['alt_E0_median_single_draw'] = {f'D{D}': verdict(float(np.median(per[:D])) - n0, D) for D in (16, 64)}
json.dump(out, open(sys.argv[2], 'w'), indent=1)
for k, v in out.items():
    print(k, json.dumps(v) if not isinstance(v, dict) or 'D16' not in v else '')
    if isinstance(v, dict) and 'D16' in v:
        for kk, vv in v.items():
            print('   ', kk, {a: (round(b, 4) if isinstance(b, float) else b) for a, b in vv.items()})
