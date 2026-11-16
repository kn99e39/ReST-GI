"""WL97 J6: decisive re-evaluation of noise-margin and weight-dependent decisions from STORED values only (numpy, read-only)."""
import json, math, numpy as np
J = lambda f: json.load(open(f, encoding='utf-8'))
out = {}
# (a) an94.W coded vs true weight ratios (true layout: bin = iz*16 + iphi, 4 rings in cos(theta))
NB = 64
Wc = ((np.arange(NB) // 8) + 0.5) / 8 * (2 * np.pi / NB)
Wt = ((np.arange(NB) // 16) + 0.5) / 4 * (2 * np.pi / NB)
out['weight_ratio_coded_over_true'] = {f'ring{iz}': [round(float(Wc[iz*16]/Wt[iz*16]), 4), round(float(Wc[iz*16+8]/Wt[iz*16+8]), 4)] for iz in range(4)}
out['weight_sums'] = [float(Wc.sum()), float(Wt.sum())]
# (b) paired statistics from C1 recomputation
P = J('results/audit97/C1/images_recompute.json')['pairs']
se_max = max(v['se_d_bound'] for v in P.values())
out['paired_se_max'] = se_max
B = J('results/ecro_mat94/stageB.json')['SSR_frozen']
ssr = {}
for s, v in B.items():
    paired_tol = max(2 * se_max, 0.005)
    ssr[s] = {'diff': v['diff'], 'share': v['share'], 'frozen_tol': max(2 * v['noise_bound'], 0.005), 'pass_frozen': v['pass'],
              'paired_tol': paired_tol, 'pass_paired': bool(v['diff'] >= paired_tol and v['share'] >= 0.30),
              'half_diffs': [P[f'own_{s}_C0|own_{s}_C2_srcS1'][k] for k in ('d_half1', 'd_half2')]}
out['WL94_SSR'] = ssr
# frozen mapping (an94.labels), GATE=RUN_C, FIT false -> mr False
ssr_f = [ssr[s]['pass_paired'] for s in ('s8601', 's8602')]
out['WL94_H-A_paired'] = 'SUPPORTED' if all(ssr_f) else ('WEAKENED' if not any(ssr_f) else 'INCONCLUSIVE')
out['WL94_H-A_stored'] = J('results/ecro_mat94/final.json')['labels']['H-A']
# (c) WL94D RENDER_IMP / WL94C pairs / WL94A RET / WL95A NDT under paired tolerance (0.005 floor kept, and floor removed)
def lab(diff, tol):
    return 'better' if diff > tol else ('worse' if -diff > tol else 'tie')
r = {}
for sd in (8601, 8602):
    d = J(f'results/ecro_s94d/s_s{sd}.json')
    r[sd] = {k: {'diff': v['diff'], 'frozen': lab(v['diff'], v['tol']), 'paired_floor': lab(v['diff'], max(2*se_max, 0.005)),
                 'paired_nofloor': lab(v['diff'], 2*se_max)} for k, v in d['pairs'].items()}
    r[sd]['RENDER_IMP'] = {t: r[sd]['T_vs_P'][t] == 'better' and r[sd]['T_vs_R'][t] == 'better' for t in ('frozen', 'paired_floor', 'paired_nofloor')}
out['WL94D_pairs'] = r
c = {}
for sd in (8601, 8602):
    d = J(f'results/ecro_q94c/q_s{sd}.json')
    c[sd] = {k: {'diff': v['diff_y_minus_x'], 'frozen': lab(v['diff_y_minus_x'], v['tol']), 'paired_floor': lab(v['diff_y_minus_x'], max(2*se_max, 0.005)),
                 'paired_nofloor': lab(v['diff_y_minus_x'], 2*se_max)} for k, v in d['pairs'].items()}
out['WL94C_pairs'] = c
a = J('results/ecro_attr95a/an.json')['components']
def ndt(tolf):
    comp = {k: dict(v, tol=tolf(v)) for k, v in a.items()}
    allc = lambda f: all(f(x) for x in comp.values())
    if allc(lambda x: x['dG'] > max(x['dS'], x['dF']) + x['tol']): return 'HIGHER_ORDER_TRANSPORT'
    if allc(lambda x: x['dS'] > max(x['dF'], x['dG']) + x['tol']): return 'LEARNED_SENDER'
    if allc(lambda x: x['dF'] >= max(x['dS'], x['dG']) - x['tol'] and x['dF'] > x['tol']): return 'DIRECTIONAL_CARRIER(if FA allows)'
    return 'UNDETERMINED'
out['WL95A_NDT'] = {'stored': J('results/ecro_attr95a/an.json')['verdicts']['NEXT_DESIGN_TARGET'], 'frozen_recomputed': ndt(lambda v: v['tol']),
                    'paired_floor_0.005': ndt(lambda v: 0.005), 'paired_nofloor_2se': ndt(lambda v: 2 * 3e-4)}
# (d) WL94D G_PHYS: implied correlation of the deterministic M64 flight with the noise-free label
g = J('results/ecro_s94d/audit.json')['gates']['G_PHYS']
rti = g['label_reliability_corr_256_vs_64']  # = sqrt(r_t r_i), r = S/(S+N), 64 spp noise = 4x 256 spp noise
n = (-5 + math.sqrt(25 + 16 * (1 / rti**2 - 1))) / 8  # solves (1+n)(1+4n) = 1/rti^2
r_t, r_i = 1 / (1 + n), 1 / (1 + 4 * n)
out['WL94D_G_PHYS'] = {'stored': {k: g[k] for k in ('pooled_ratio', 'corr_exact_vs_label', 'label_reliability_corr_256_vs_64', 'corr_exact_vs_independent_label')},
                       'implied_corr_vs_noisefree_label_from_256': g['corr_exact_vs_label'] / math.sqrt(r_t),
                       'implied_corr_vs_noisefree_label_from_64': g['corr_exact_vs_independent_label'] / math.sqrt(r_i),
                       'rule_passes_any_X_with_true_corr_above': math.sqrt(r_i),
                       'ck2_boxA': g['ck2_flight_floor_diagnostic']['boxA']}
json.dump(out, open('results/audit97/J6/j6_decisive.json', 'w', newline='\n'), indent=1, default=float)
print(json.dumps({k: out[k] for k in ('weight_ratio_coded_over_true', 'WL94_H-A_paired', 'WL94_H-A_stored', 'WL95A_NDT')}))
print({sd: out['WL94D_pairs'][sd]['RENDER_IMP'] for sd in out['WL94D_pairs']})
print({sd: {k: (v['frozen'], v['paired_floor'], v['paired_nofloor']) for k, v in out['WL94C_pairs'][sd].items()} for sd in out['WL94C_pairs']})
print({k: (round(v,4) if isinstance(v, float) else '') for k, v in out['WL94D_G_PHYS'].items()})
print(out['WL94_SSR'])
