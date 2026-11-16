"""WL97 audit track E3: convergence ladder, model-vs-oracle (noise-free) metrics, comparison with WL95A's stored pooled arms, reference
noise tests (numpy only).   python conv97.py <RES> <path to results/ecro_attr95a/deliv.npz>   -> RES/conv_models.json, RES/wl95a_pooled_boxA.npz"""
import json, math, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ana97

RES, WL95A = sys.argv[1], sys.argv[2]
ea, eb, se, a, b = ana97.ref(RES)
ra, rb = ea.sum(1), eb.sum(1); r = 0.5 * (ra + rb); S = r.sum()
ld = lambda t: {k[2:]: 0.5 * v.sum(1) for k, v in np.load(f'{RES}/deliv_{t}.npz').items() if k.startswith('E_')}
TAGS = ('s16r12', 's8r12', 's4r12', 's2r12', 's4r6', 's4r24', 's4r12c', 's4r12b', 's16r12hv', 's8r12o1', 's8r12o2')
D = {t: ld(t) for t in TAGS}
best = D['s4r24']; sig = best['exact'] > 0.01 * best['exact'].max()
out = {'patches_above_1pct_of_peak': int(sig.sum()), 'convergence_vs_s4r24': {}, 'table_variants_vs_s4r12': {}}
for t in TAGS:
    out['convergence_vs_s4r24'][t] = {m: {'total': float(x.sum()), 'max_patch_rel': float(np.abs(x[sig] / (best[m] if m in best else D['s8r12'][m])[sig] - 1).max()),
                                          'l1_rel': float(np.abs(x - (best[m] if m in best else D['s8r12'][m])).sum() / (best[m] if m in best else D['s8r12'][m]).sum())}
                                      for m, x in D[t].items()}
for t in ('s4r12c', 's4r12b'):
    out['table_variants_vs_s4r12'][t] = {m: {'total_rel': float(D[t][m].sum() / D['s4r12'][m].sum() - 1), 'max_patch_rel': float(np.abs(D[t][m][sig] / D['s4r12'][m][sig] - 1).max())}
                                         for m in ('qbin', 'm64')}
mm = {}
for m, x in list(best.items()) + [('bindir', D['s8r12']['bindir'])]:
    e0 = best['exact'] if m != 'bindir' else D['s8r12']['exact']
    mm[m] = {'ratio': float(x.sum() / e0.sum()), 'abs_rel_l1': float(np.abs(x - e0).sum() / e0.sum()), 'corr': float(np.corrcoef(x, e0)[0, 1]),
             'faces_-x_-y_+z': [float(x[9 * f:9 * f + 9].sum() / e0[9 * f:9 * f + 9].sum()) for f in (0, 2, 5)],
             'l1_after_energy_rescale': float(np.abs(x * e0.sum() / x.sum() - e0).sum() / e0.sum()), 'peak_patch': int(np.argmax(x))}
out['model_vs_exact_oracle'] = mm
z = np.load(WL95A); B = slice(1130, 1184)
assert np.abs(0.5 * (z['ref_a'][B] + z['ref_b'][B]) - r).max() < 1e-12
pool = {}; xc = {}; q = best['exact']
for mine, theirs in (('exact', 'X1'), ('anchor', 'X1c'), ('qbin', 'q256'), ('m64', 'M64_spool')):
    d = z['draws|' + theirs][:, B]; p = d.mean(0); sd = d.std(0, ddof=1) / math.sqrt(d.shape[0]); x = best[mine]; nz = sd > 0
    zz = (p - x)[nz] / sd[nz]
    l1r = np.abs(d - r[None]).sum(1) / S; tot = d.sum(1) / S; dep = z['dep|' + theirs][B].sum() / S
    pool[theirs] = p; pool[theirs + '_sd'] = d.std(0, ddof=1)
    xc[f'{mine}|{theirs}'] = {'wl95a_pooled_vs_ref': {'ratio': float(p.sum() / S), 'abs_rel_l1': float(np.abs(p - r).sum() / S), 'corr': float(np.corrcoef(p, r)[0, 1])},
                              'wl95a_pooled_vs_my_model': {'ratio': float(p.sum() / x.sum()), 'rel_l1': float(np.abs(p - x).sum() / x.sum()), 'corr': float(np.corrcoef(p, x)[0, 1]),
                                                           'rms_z': float(np.sqrt((zz ** 2).mean())), 'max_abs_z': float(np.abs(zz).max())},
                              'per_draw_l1_vs_ref': {'median': float(np.median(l1r)), 'p90': float(np.percentile(l1r, 90)), 'max': float(l1r.max())},
                              'per_draw_l1_vs_my_oracle_median': float(np.median(np.abs(d - q[None]).sum(1) / q.sum())),
                              'deployed_total_ratio': float(dep), 'fraction_of_pooled_draws_below_deployed': float((tot < dep).mean())}
np.savez(f'{RES}/wl95a_pooled_boxA.npz', ref_a=z['ref_a'][B], ref_b=z['ref_b'][B], **pool)
out['vs_wl95a'] = xc
nzs = se > 0; x128 = np.concatenate([a.sum(2), b.sum(2)], 1); w = np.concatenate([ana97.W, ana97.W]) / 2
rng = np.random.default_rng(97); l1 = []; rat = []
for _ in range(4000):
    xb = np.take_along_axis(x128, rng.integers(0, 128, size=(54, 128)), 1); rb_ = (xb * w).sum(1)
    l1.append(np.abs(rb_ - r).sum() / S); rat.append(rb_.sum() / S)
bw = ((ra - rb) ** 2 / 4)[nzs] / (se ** 2)[nzs]
out['reference_noise'] = {'n0': float(0.5 * np.abs(ra - rb).sum() / S), 'bootstrap_l1_mean': float(np.mean(l1)), 'bootstrap_l1_p95': float(np.percentile(l1, 95)),
                          'bootstrap_total_ratio_sd': float(np.std(rat)), 'half_vs_half_z2_mean': float(bw.mean()), 'half_vs_half_z2_median': float(np.median(bw)),
                          'sum_between_over_sum_within': float(((ra - rb) ** 2 / 4)[nzs].sum() / (se ** 2)[nzs].sum()),
                          'l1_exact_vs_half_a': float(np.abs(q - ra).sum() / S), 'l1_exact_vs_half_b': float(np.abs(q - rb).sum() / S),
                          'expected_l1_vs_one_half': float(np.abs(ra - rb).sum() / S / math.sqrt(2))}
mcf = f'{RES}/mc_l23b32s97001.npz'
if os.path.exists(mcf):
    m = np.load(mcf); mc = m['ex3'].sum(1); ms = m['se_patch']; nz = ms > 0; zz = (q - mc)[nz] / ms[nz]
    out['quadrature_vs_solid_angle_mc'] = {'mc_total': float(mc.sum()), 'mc_total_se': json.loads(str(m['meta']))['total_se_batch_means'], 'quad_total': float(q.sum()),
                                           'ratio': float(q.sum() / mc.sum()), 'rel_l1': float(np.abs(q - mc).sum() / mc.sum()), 'rms_z': float(np.sqrt((zz ** 2).mean())),
                                           'max_abs_z': float(np.abs(zz).max()), 'patches_3_6_51': {str(j): [float(r[j]), float(se[j]), float(q[j]), float(mc[j]), float(ms[j])] for j in (3, 6, 51)}}
json.dump(out, open(f'{RES}/conv_models.json', 'w'), indent=1)
print(json.dumps({k: out[k] for k in ('model_vs_exact_oracle', 'table_variants_vs_s4r12', 'reference_noise', 'quadrature_vs_solid_angle_mc')}, indent=None)[:3000])
