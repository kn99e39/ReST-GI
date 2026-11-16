"""WL97 audit C2: statistical consistency checks on the WL95A delivery evidence.
(1) noise-free deterministic oracle (copy of results/wl96_calc/scratch/d3/toy96.py 'validate', output E_exact_h5.npy in results/audit97/C2/toy_copy)
    vs the path-traced reference and vs pooled X1 / A4: are the residuals consistent with noise alone?
(2) how the frozen FLIGHT_ATTRIBUTION rule behaves if another relation draw had been the deployed one (single-draw artifact of E0 / Rs).
(3) heavy-tail robustness of Ra / Rp (leave-worst-draws-out).
(4) learned sender vs M64 against the REFERENCE (paired over draws): is the learned-sender delivery error 'avoidable' by the exact-bin sender?
numpy only; run from the repository root."""
import json
import numpy as np

z = np.load('results/ecro_attr95a/deliv.npz')
BOXA = np.arange(1130, 1184)
ra, rb = z['ref_a'][BOXA], z['ref_b'][BOXA]
ref = 0.5 * (ra + rb)
orc = 0.5 * np.load('results/audit97/C2/toy_copy/E_exact_h5.npy').sum(1)       # box A Lambert albedo 0.5 (toy96 metrics convention)
out = {}
act = orc > 1e-9 * orc.max()
out['active_patches'] = int(act.sum())

# ---- (1) oracle vs reference
sig_ref = 0.5 * np.abs(ra - rb)                    # 1-dof estimate of the SE of the reference mean per patch
out['oracle_vs_ref'] = {'ratio': float(orc.sum() / ref.sum()), 'rel_l1': float(np.abs(orc - ref).sum() / ref.sum()), 'n0': float(sig_ref.sum() / ref.sum()),
                        'total_diff_over_ref_total_se': float((ref.sum() - orc.sum()) / (0.5 * abs(ra.sum() - rb.sum()))),
                        'ref_total_half_diff_rel': float(0.5 * (ra.sum() - rb.sum()) / ref.sum()),
                        # half a vs oracle and half b vs oracle separately: if the oracle is the truth, each half has L1 ~ sqrt(2) x the mean's
                        'l1_half_a': float(np.abs(orc - ra).sum() / orc.sum()), 'l1_half_b': float(np.abs(orc - rb).sum() / orc.sum()),
                        'ratio_half_a': float(ra.sum() / orc.sum()), 'ratio_half_b': float(rb.sum() / orc.sum()),
                        # cross-validated squared bias: (ra - orc)(rb - orc) has expectation bias^2 per patch (noise independent between halves)
                        'cv_bias2_sum_over_orc2': float(((ra - orc) * (rb - orc)).sum() / (orc ** 2).sum()),
                        'noise2_sum_over_orc2': float((0.25 * (ra - rb) ** 2).sum() / (orc ** 2).sum())}
# ---- pooled arms vs the noise-free oracle
arms = ['A0_s8601', 'A0_s8602', 'A1', 'A2', 'A3', 'A4', 'M64_spool', 'q256', 'X1c', 'X1']
D = 64
for k in arms:
    X = z[f'draws|{k}'][:, BOXA]
    m = X.mean(0); se = X.std(0, ddof=1) / np.sqrt(D)
    tot = X.sum(1)
    ok = act & (se > 0)
    zz = (m[ok] - orc[ok]) / se[ok]
    # independent halves of the draws: cross-validated squared bias (noise cancels in expectation)
    h1, h2 = X[0::2].mean(0), X[1::2].mean(0)
    out[f'{k}_vs_oracle'] = {'ratio': float(m.sum() / orc.sum()), 'ratio_se': float(tot.std(ddof=1) / np.sqrt(D) / orc.sum()),
                             'ratio_z': float((m.sum() - orc.sum()) / (tot.std(ddof=1) / np.sqrt(D))),
                             'rel_l1': float(np.abs(m - orc).sum() / orc.sum()),
                             'expected_l1_if_unbiased': float((se * np.sqrt(2 / np.pi)).sum() / orc.sum()),
                             'chi2_per_patch': float((zz ** 2).mean()), 'n_patch': int(ok.sum()), 'max_abs_z': float(np.abs(zz).max()),
                             'cv_bias_l2_rel': float(np.sign(((h1 - orc) * (h2 - orc)).sum()) * np.sqrt(abs(((h1 - orc) * (h2 - orc)).sum()) / (orc ** 2).sum())),
                             'dep_ratio_vs_oracle': float(z[f'dep|{k}'][BOXA].sum() / orc.sum()), 'dep_l1_vs_oracle': float(np.abs(z[f'dep|{k}'][BOXA] - orc).sum() / orc.sum()),
                             'single_draw_l1_vs_oracle_mean': float(np.mean([np.abs(x - orc).sum() / orc.sum() for x in X]))}

# ---- (2) frozen FLIGHT_ATTRIBUTION rule with every pooled draw taken as the 'deployed' relation (A4 on that draw)
l1 = lambda x: float(np.abs(x - ref).sum() / ref.sum())
n0 = float(sig_ref.sum() / ref.sum())
jk = lambda f, *Xs: float(np.sqrt((D - 1) / D * (((lo := np.array([f(*[(X.sum(0) - X[i]) / (D - 1) for X in Xs]) for i in range(D)])) - lo.mean()) ** 2).sum()))
XA4, XSP, XC, XX = (z[f'draws|{k}'][:, BOXA] for k in ('A4', 'M64_spool', 'X1c', 'X1'))
E_sp = l1(XSP.mean(0)) - n0
se_s = jk(l1, XSP)
Ra = l1(XSP.mean(0)) - l1(XC.mean(0)); se_a = jk(lambda a, b: l1(a) - l1(b), XSP, XC)
EX1 = l1(XX.mean(0)) - n0
verd = {}
rows = []
for d in range(-1, D):
    dep = z['dep|A4'][BOXA] if d < 0 else XA4[d]
    E0 = l1(dep) - n0
    Rs = E0 - E_sp
    s_ok = Rs >= 0.25 * E0 and Rs > 2 * se_s
    a_ok = Ra >= 0.25 * E0 and Ra > 2 * se_a
    if EX1 > 0.5 * E0:
        v = 'INCONCLUSIVE'
    elif s_ok and a_ok:
        v = 'MIXED'
    elif s_ok and not a_ok and Rs >= 0.5 * E0:
        v = 'SAMPLING_DOMINANT'
    elif a_ok and not s_ok:
        v = 'ANGULAR_PROJECTION_EVIDENCED'
    else:
        v = 'INCONCLUSIVE'
    if d < 0:
        out['rule_on_actual_deployed'] = {'E0': E0, 'Rs': Rs, 'Ra': Ra, 'verdict': v}
    else:
        verd[v] = verd.get(v, 0) + 1
        rows.append((E0, Rs, v))
out['rule_if_other_draw_deployed'] = {'counts_of_64': verd, 'E0_median': float(np.median([r[0] for r in rows])), 'E0_mean': float(np.mean([r[0] for r in rows])),
                                      'Rs_median': float(np.median([r[1] for r in rows])), 'Rs_share_of_E0_median': float(np.median([r[1] / r[0] for r in rows])),
                                      'Ra': Ra, 'Ra_se': se_a, 'E_spool_pool': E_sp, 'se_s': se_s}
# expectation-level decomposition: expected single-draw L1 of the binned sender vs pooled (sampling), pooled binned vs pooled exact (projection)
out['expectation_decomposition_L1'] = {'mean_single_draw_A4': float(np.mean([l1(x) for x in XA4])), 'pooled_A4': l1(XA4.mean(0)), 'pooled_X1': l1(XX.mean(0)),
                                       'mean_single_draw_X1': float(np.mean([l1(x) for x in XX])), 'n0': n0}

# ---- (3) heavy-tail robustness of Ra and Rp: drop the k draws with the largest X1c box-A total (from all arms, paired)
tot_c = XC.sum(1)
rob = {}
for kdrop in (0, 1, 2, 4):
    keep = np.argsort(tot_c)[:D - kdrop]
    f = lambda X: l1(X[keep].mean(0))
    rob[f'drop{kdrop}'] = {'Ra': f(XSP) - f(XC), 'Rp': f(XC) - f(XX), 'L1_X1c': f(XC), 'L1_X1': f(XX), 'ratio_X1c': float(XC[keep].mean(0).sum() / ref.sum()),
                           'ratio_X1': float(XX[keep].mean(0).sum() / ref.sum())}
# Rp against the noise-free oracle and split halves
rob['Rp_vs_oracle'] = float(np.abs(XC.mean(0) - orc).sum() / orc.sum() - np.abs(XX.mean(0) - orc).sum() / orc.sum())
rob['Rp_halves_vs_ref'] = [l1(XC[i::2].mean(0)) - l1(XX[i::2].mean(0)) for i in (0, 1)]
rob['Ra_halves_vs_ref'] = [l1(XSP[i::2].mean(0)) - l1(XC[i::2].mean(0)) for i in (0, 1)]
rob['Rp_cv_bias_l2'] = {'X1c': out['X1c_vs_oracle']['cv_bias_l2_rel'], 'X1': out['X1_vs_oracle']['cv_bias_l2_rel']}
out['heavy_tail_robustness'] = rob

# ---- (4) learned sender vs exact-bin M64 against the reference, paired over the 64 common relation draws
pair = {}
for a, b in (('A0_s8601', 'A4'), ('A0_s8602', 'A4'), ('A1', 'A4'), ('A3', 'A4'), ('A0_s8601', 'A1'), ('A0_s8601', 'X1'), ('A4', 'X1'), ('A0_s8601', 'M64_spool')):
    Xa, Xb = z[f'draws|{a}'][:, BOXA], z[f'draws|{b}'][:, BOXA]
    f = lambda u, v: l1(u) - l1(v)
    val = f(Xa.mean(0), Xb.mean(0)); se = jk(f, Xa, Xb)
    # also vs the oracle
    fo = lambda u, v: float(np.abs(u - orc).sum() / orc.sum() - np.abs(v - orc).sum() / orc.sum())
    pair[f'{a}-{b}'] = {'dL1_pooled_vs_ref': val, 'se_jk': se, 'z': val / se if se > 0 else None, 'dL1_pooled_vs_oracle': fo(Xa.mean(0), Xb.mean(0)),
                        'dL1_deployed_vs_ref': l1(z[f'dep|{a}'][BOXA]) - l1(z[f'dep|{b}'][BOXA]),
                        'frac_draws_a_better_single_draw': float(np.mean([l1(Xa[i]) < l1(Xb[i]) for i in range(D)]))}
out['paired_vs_reference'] = pair
json.dump(out, open('results/audit97/C2/wl95a_consistency.json', 'w'), indent=1)
print(json.dumps(out, indent=1))
