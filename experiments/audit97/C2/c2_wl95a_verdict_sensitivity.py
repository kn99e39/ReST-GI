"""WL97 audit C2: (1) is the deployed relation draw 86001 inside the pooled distribution; (2) how the frozen WL95A FLIGHT_ATTRIBUTION label
changes if any of the 64 pooled draws had been the deployed one; (3) pairwise pooled box-A L1-vs-reference differences (jackknife over draws).
numpy only; input results/ecro_attr95a/deliv.npz.  Run from the repository root."""
import json
import numpy as np
z = np.load('results/ecro_attr95a/deliv.npz')
ra, rb = z['ref_a'], z['ref_b']; ref = .5 * (ra + rb); B = np.arange(1130, 1184); R = ref[B]; S = R.sum()
n0 = 0.5 * np.abs(ra[B] - rb[B]).sum() / S
L1 = lambda x: np.abs(x - R).sum() / S
D = lambda k: z['draws|' + k][:, B]


def jk(fn, *arrs):
    n = arrs[0].shape[0]; tot = [a.sum(0) for a in arrs]
    loo = np.array([fn(*[(t - a[i]) / (n - 1) for t, a in zip(tot, arrs)]) for i in range(n)])
    return float(fn(*[t / n for t in tot])), float(np.sqrt((n - 1) / n * ((loo - loo.mean()) ** 2).sum()))


o = {'n0': float(n0)}
for k in ('A4', 'M64_spool', 'q256', 'X1c', 'X1', 'A0_s8601', 'A0_s8602', 'A1', 'A2', 'A3'):
    d = D(k); tot = d.sum(1) / S; dep = z['dep|' + k][B]
    o[k] = {'pool_ratio': float(d.mean(0).sum() / S), 'pool_L1': float(L1(d.mean(0))), 'dep_ratio': float(dep.sum() / S), 'dep_L1': float(L1(dep)),
            'draw_ratio_sd': float(tot.std(ddof=1)), 'pool_ratio_se': float(tot.std(ddof=1) / np.sqrt(len(tot))),
            'dep_z_in_draws': float((dep.sum() / S - tot.mean()) / tot.std(ddof=1)), 'draws_above_dep': int((tot > dep.sum() / S).sum()),
            'mean_single_draw_L1': float(np.mean([L1(x) for x in d])), 'frac_draw_L1_below_dep': float(np.mean([L1(x) < L1(dep) for x in d])),
            'dep_min_L1_distance_to_any_pool_draw': float(np.abs(z['draws|' + k] - z['dep|' + k][None]).sum(1).min() / z['dep|' + k].sum())}
Esp, se_sp = jk(lambda a: L1(a) - n0, D('M64_spool'))
Ra, se_a = jk(lambda a, b: L1(a) - L1(b), D('M64_spool'), D('X1c'))
Rp, se_p = jk(lambda a, b: L1(a) - L1(b), D('X1c'), D('X1'))
E0_dep = L1(z['dep|A4'][B]) - n0
o['frozen'] = {'E0': float(E0_dep), 'Rs': float(E0_dep - Esp), 'Rs_se': se_sp, 'Ra': Ra, 'Ra_se': se_a, 'Rp': Rp, 'Rp_se': se_p, 'thr': float(0.25 * E0_dep)}
cnt = {}; E0s = []
for d in range(64):
    E0 = L1(D('A4')[d]) - n0; Rs = E0 - Esp; E0s.append(E0)
    cs = Rs >= 0.25 * E0 and Rs > 2 * se_sp; ca = Ra >= 0.25 * E0 and Ra > 2 * se_a
    v = 'SAMPLING_DOMINANT' if (cs and Rs >= 0.5 * E0 and not ca) else 'ANGULAR_PROJECTION_EVIDENCED' if (ca and not cs) else 'MIXED' if (cs and ca) else 'INCONCLUSIVE'
    cnt[v] = cnt.get(v, 0) + 1
o['label_if_pool_draw_d_were_deployed'] = cnt
o['E0_over_pool_draws'] = {'median': float(np.median(E0s)), 'mean': float(np.mean(E0s)), 'frac_below_deployed_E0': float(np.mean(np.array(E0s) < E0_dep))}
o['pairs_pooled_L1_vs_ref'] = {}
for a, b in (('A1', 'A4'), ('A3', 'A4'), ('A0_s8601', 'A4'), ('A0_s8602', 'A4'), ('A0_s8601', 'A1'), ('A4', 'X1'), ('A4', 'M64_spool')):
    v, se = jk(lambda x, y: L1(x) - L1(y), D(a), D(b)); o['pairs_pooled_L1_vs_ref'][f'{a}-{b}'] = {'d': v, 'se_jk': se, 'z': v / se}
print(json.dumps(o, indent=1))
json.dump(o, open('results/audit97/C2/wl95a_verdict_sensitivity.json', 'w'), indent=1)
