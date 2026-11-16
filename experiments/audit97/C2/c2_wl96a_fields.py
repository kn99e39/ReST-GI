"""WL97 audit C2: independent recomputation of the WL96A field metrics (results/ecro_util96a/tails.npz vs the CKF_none reference families,
copied from LabServer63 ~/rest_x94/results/fields/CKF_none.npz to results/audit97/C2/CKF_none.npz, sha256 3a6238a8a42d1db4...).
numpy only; no audited metric function is imported.  Run from the repository root."""
import json, math
import numpy as np

z = np.load('results/audit97/C2/CKF_none.npz')
t = np.load('results/ecro_util96a/tails.npz')
tj = json.load(open('results/ecro_util96a/tails.json'))['field']
T = np.load('results/ecro_util96a/T96A.npz')
NB = 64
WT = ((np.arange(NB) // 16) + 0.5) / 4 * 2 * math.pi / NB
W = WT[None, :, None]
PMAT = z['PMAT']
S1 = np.arange(1088, 1130); BOXA = np.arange(1130, 1184); PFACE = np.arange(1130, 1139)
assert (PMAT[S1] == 1).all() and (PMAT == 1).sum() == 42
RG = {'boxA': BOXA, 'panel_face': PFACE, 'S1': S1, 'nonS1': np.nonzero(PMAT != 1)[0], 'all': np.arange(len(PMAT))}
g = lambda k: z[k].astype(np.float64)
ref = {}
for f in ('e2_p', 'e2_o', 'e3_p', 'e3_o'):
    ref[f + '_a'], ref[f + '_b'] = g(f + '_a'), g(f + '_b')
    ref[f] = 0.5 * (ref[f + '_a'] + ref[f + '_b'])
for s in ('', '_a', '_b'):
    ref['e3' + s] = ref['e3_p' + s] + ref['e3_o' + s]
    ref['e2' + s] = ref['e2_p' + s] + ref['e2_o' + s]
dep = {'e3_p': t['dep_p'].astype(np.float64), 'e3_o': t['dep_o'].astype(np.float64)}
dep['e3'] = dep['e3_p'] + dep['e3_o']
pool = {'e3_p': t['pool_p'].astype(np.float64), 'e3_o': t['pool_o'].astype(np.float64)}
pool['e3'] = pool['e3_p'] + pool['e3_o']
D = pool['e3_o'].shape[0]


def met(X, fam, idx):
    R, A, B = ref[fam][idx], ref[fam + '_a'][idx], ref[fam + '_b'][idx]
    x = X[idx]
    ex = lambda q: (q * W).sum((1, 2))
    eX, eR, eA, eB = ex(x), ex(R), ex(A), ex(B)
    den_t = (np.abs(R) * W).sum(); den_e = eR.sum()
    o = {'flux_ratio': float(eX.sum() / den_e), 'exit_rel_l1': float(np.abs(eX - eR).sum() / den_e), 'exit_n0': float(0.5 * np.abs(eA - eB).sum() / den_e),
         'table_rel_l1': float((np.abs(x - R) * W).sum() / den_t), 'table_n0': float(0.5 * (np.abs(A - B) * W).sum() / den_t)}
    if den_e > 0 and eR.std() > 0:
        o['corr'] = float(np.corrcoef(eX, eR)[0, 1])
    # cross-validated squared error: sum W (x - A)(x - B) is an unbiased estimate of sum W (x - truth)^2 when the two halves carry independent noise
    # and x is independent of that noise;  noise^2 of the mean = 0.25 sum W (A - B)^2
    l2 = (R ** 2 * W).sum()
    cv = ((x - A) * (x - B) * W).sum()
    o['cv_rel_l2_bias'] = float(np.sign(cv) * math.sqrt(abs(cv) / l2)) if l2 > 0 else None
    o['rel_l2_noise_of_mean'] = float(math.sqrt((0.25 * (A - B) ** 2 * W).sum() / l2)) if l2 > 0 else None
    o['rel_l2_raw'] = float(math.sqrt(((x - R) ** 2 * W).sum() / l2)) if l2 > 0 else None
    return o


def jk(draws, fn):
    tot = draws.sum(0)
    loo = np.array([fn((tot - draws[i]) / (D - 1)) for i in range(D)])
    return float(np.sqrt((D - 1) / D * ((loo - loo.mean()) ** 2).sum()))


out = {'recompute': {}, 'max_abs_dev_vs_tails_json': 0.0}
mx = 0.0
for fam in ('e3_o', 'e3_p', 'e3'):
    for rn, idx in RG.items():
        if fam == 'e3_p' and rn == 'S1':
            continue
        a = met(dep[fam], fam, idx)
        b = met(pool[fam].mean(0), fam, idx)
        b['se_table_rel_l1'] = jk(pool[fam], lambda m: met(m, fam, idx)['table_rel_l1'])
        b['se_exit_rel_l1'] = jk(pool[fam], lambda m: met(m, fam, idx)['exit_rel_l1'])
        per = np.array([met(pool[fam][d], fam, idx)['table_rel_l1'] for d in range(D)])
        b['per_realization_table'] = {'median': float(np.median(per)), 'p90': float(np.percentile(per, 90)), 'worst': float(per.max()), 'min': float(per.min())}
        a['rank_of_deployed_among_pool_table'] = float((per < a['table_rel_l1']).mean())
        out['recompute'][f'XD|{fam}|{rn}'] = a
        out['recompute'][f'XP|{fam}|{rn}'] = b
        for arm, m in (('XD', a), ('XP', b)):
            o = tj.get(f'{arm}|{fam}|{rn}')
            if o:
                for k in ('flux_ratio', 'exit_rel_l1', 'exit_n0', 'table_rel_l1', 'table_n0'):
                    mx = max(mx, abs(m[k] - o[k]))
                if arm == 'XP':
                    mx = max(mx, abs(m['se_table_rel_l1'] - o['se_table_rel_l1']))
out['max_abs_dev_vs_tails_json'] = mx

# ---- table identity checks (render tables)
E2 = ref['e2']
chk = {'F': E2 + ref['e3'], 'XOD': E2 + ref['e3_p'] + dep['e3_o'], 'XD': E2 + dep['e3'], 'XOP': E2 + ref['e3_p'] + pool['e3_o'].mean(0), 'Z': E2,
       'Fa': E2 + ref['e3_a'], 'Fb': E2 + ref['e3_b']}
out['T96A_identity_max_rel'] = {k: float(np.abs(T[k].astype(np.float64) - v).max() / np.abs(v).max()) for k, v in chk.items()}

# ---- where does XOD differ from F at table level?  share of the W_T-weighted |XOD - F| by region, relative to the F table flux of the region and of all
d = np.abs(chk['XOD'] - chk['F'])
dn = 0.5 * np.abs(ref['e3_o_a'] - ref['e3_o_b'])              # e3_o-only reference noise of the mean (the part of n that XOD vs F can see)
dn_all = 0.5 * np.abs(ref['e3_a'] - ref['e3_b'])              # what n_img uses (Fa vs Fb: e3_p + e3_o halves)
tot_all = (chk['F'] * W).sum()
out['XOD_vs_F_table'] = {}
for rn, idx in RG.items():
    out['XOD_vs_F_table'][rn] = {'l1_over_region_F': float((d[idx] * W).sum() / (chk['F'][idx] * W).sum()), 'l1_over_all_F': float((d[idx] * W).sum() / tot_all),
                                 'e3o_noise_over_region_F': float((dn[idx] * W).sum() / (chk['F'][idx] * W).sum()),
                                 'e3_noise_over_region_F': float((dn_all[idx] * W).sum() / (chk['F'][idx] * W).sum()),
                                 'ratio_e3_noise_to_e3o_noise': float((dn_all[idx] * W).sum() / max((dn[idx] * W).sum(), 1e-30)),
                                 'tail_share_of_F': float((ref['e3'][idx] * W).sum() / (chk['F'][idx] * W).sum()),
                                 'e3o_share_of_F': float((ref['e3_o'][idx] * W).sum() / (chk['F'][idx] * W).sum())}

# ---- S1 sharp-reader excess: pooled (bias) per incident ring/bin, and its size relative to things an image would read
xp = pool['e3_o'].mean(0)
se = pool['e3_o'].std(0, ddof=1) / math.sqrt(D)
R = ref['e3_o']
s1 = {}
num = (np.abs(xp[S1] - R[S1]) * W).sum((0, 2)); den = (R[S1] * W).sum()
s1['pooled_table_l1_share_by_ring'] = [float(num[16 * i:16 * i + 16].sum() / den) for i in range(4)]
nn = (0.5 * np.abs(ref['e3_o_a'][S1] - ref['e3_o_b'][S1]) * W).sum((0, 2))
s1['n0_share_by_ring'] = [float(nn[16 * i:16 * i + 16].sum() / den) for i in range(4)]
s1['ref_flux_share_by_ring'] = [float((R[S1][:, 16 * i:16 * i + 16] * W[:, 16 * i:16 * i + 16]).sum() / den) for i in range(4)]
# signed pooled bias per ring (flux)
sg = ((xp[S1] - R[S1]) * W).sum((0, 2))
s1['pooled_signed_by_ring'] = [float(sg[16 * i:16 * i + 16].sum() / den) for i in range(4)]
# deployed
numd = (np.abs(dep['e3_o'][S1] - R[S1]) * W).sum((0, 2))
s1['deployed_table_l1_share_by_ring'] = [float(numd[16 * i:16 * i + 16].sum() / den) for i in range(4)]
# z-scores of the pooled bias vs combined SE (realization SE and 1-dof reference SE) over S1 entries
sig_ref = 0.5 * np.abs(ref['e3_o_a'][S1] - ref['e3_o_b'][S1])
s1['frac_entries_abs_dev_gt_3se_pool_only'] = float((np.abs(xp[S1] - R[S1]) > 3 * se[S1]).mean())
s1['S1_e3o_flux_over_all_e3o_flux'] = float((R[S1] * W).sum() / (R * W).sum())
s1['S1_e3o_table_excess_times_flux_share'] = None
out['S1_detail'] = s1

# ---- field-level 'headroom' under alternative noise treatments, per region (e3_o deployed and pooled)
alt = {}
for rn, idx in RG.items():
    a = out['recompute'][f'XD|e3_o|{rn}']; b = out['recompute'][f'XP|e3_o|{rn}']
    alt[rn] = {'XD_table_excess_linear': a['table_rel_l1'] - a['table_n0'], 'XD_table_excess_quadrature': float(math.sqrt(max(a['table_rel_l1'] ** 2 - a['table_n0'] ** 2, 0.0))),
               'XD_cv_rel_l2_bias': a['cv_rel_l2_bias'], 'XD_rel_l2_noise': a['rel_l2_noise_of_mean'],
               'XP_table_excess_linear': b['table_rel_l1'] - b['table_n0'], 'XP_table_excess_quadrature': float(math.sqrt(max(b['table_rel_l1'] ** 2 - b['table_n0'] ** 2, 0.0))),
               'XP_cv_rel_l2_bias': b['cv_rel_l2_bias'], 'XP_se_table': b['se_table_rel_l1'],
               'secondary_rule_bound_max(2SE,n0/2)': max(2 * b['se_table_rel_l1'], a['table_n0'] / 2)}
out['e3_o_headroom_alternatives'] = alt
json.dump(out, open('results/audit97/C2/wl96a_fields_recompute.json', 'w'), indent=1)
print('max abs dev vs tails.json', mx)
print('T96A identity', out['T96A_identity_max_rel'])
for k, v in out['recompute'].items():
    print(f"{k:22s} flux {v['flux_ratio']:.4f} table {v['table_rel_l1']:.4f} (n0 {v['table_n0']:.4f}) exit {v['exit_rel_l1']:.4f} (n0 {v['exit_n0']:.4f}) corr {v.get('corr', float('nan')):.4f}"
          f" | cvL2bias {v['cv_rel_l2_bias']:+.4f} noiseL2 {v['rel_l2_noise_of_mean']:.4f} rawL2 {v['rel_l2_raw']:.4f}" + (f" | se_t {v['se_table_rel_l1']:.4f} per {v['per_realization_table']}" if 'se_table_rel_l1' in v else f" | dep-rank {v['rank_of_deployed_among_pool_table']:.2f}"))
print(json.dumps(out['XOD_vs_F_table'], indent=1))
print(json.dumps(out['S1_detail'], indent=1))
print(json.dumps(out['e3_o_headroom_alternatives'], indent=1))
