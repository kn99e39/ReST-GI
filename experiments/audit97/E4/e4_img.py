"""WL97 audit, track E4: image-level analysis (numpy only; own statistics).

  A  stored WL96A renders: paired statistics / decision recomputed with own code; mask and statistic sensitivity; F vs REF h (debiased)
  B  audit re-renders (new camera seeds, audit tables): decision replication, noise components, independent-reference replication,
     receiver-class decomposition of the explicit e3_o error, effect-size conversions

Usage: python e4_img.py <ORIG renders dir> <masks94.npz> <AUDIT renders dir> <out.json> [seed0=97600] [nrep=4]
"""
import json, math, os, sys
import numpy as np

ORIG, MASKS, MINE, OUT = sys.argv[1:5]
SEED0 = int(sys.argv[5]) if len(sys.argv) > 5 else 97600
NREP = int(sys.argv[6]) if len(sys.argv) > 6 else 4
ARMS = ['F', 'Fa', 'Fb', 'Z', 'XD', 'XP', 'XOD', 'XOP', 'XSD', 'XSP', 'G8601', 'G8602']
M_IMG = 0.005


def lo(key, n=4):
    return np.stack([np.load(os.path.join(ORIG, f'CK_none_{key}_X_CK2_T1_r{r}.npz'))['hl'].astype(np.float64) for r in range(n)])


def lm(key, seed0=SEED0, n=NREP):
    return np.stack([np.load(os.path.join(MINE, f'A_{key}_s{seed0 + r}.npz'))['hl'].astype(np.float64) for r in range(n)])


ref = {k: np.stack([np.load(os.path.join(ORIG, f'REF_ER_CK_none_CK2_T1_r{r}.npz'))[k].astype(np.float64) for r in range(4)]) for k in ('e1', 'e2', 'h')}
ref['tot'] = ref['e1'] + ref['e2'] + ref['h']
tot = ref['tot'].mean(0)
mk = np.load(MASKS)
sid = mk['sid']
MS = {'ROI_T': mk['ROI_T'], 'DIFF': mk['DIFF'], 'WHOLE': np.ones_like(mk['ROI_T']), 'FRAME': mk['frame'], 'PANEL': mk['panel'],
      'PANEL_NOT_ROI': mk['panel'] & ~mk['ROI_T'], 'BOXA_PIX': sid == 7, 'FLOOR_PIX': sid == 0, 'SPHERE_PIX': sid == 6, 'WALL_PIX': (sid >= 1) & (sid <= 4),
      'BOXB_PIX': sid == 8}
R = {'mask_pixels': {k: int(v.sum()) for k, v in MS.items()}, 'mask_tot_share_of_image': {k: float(tot[v].sum() / tot.sum()) for k, v in MS.items()}}


def stats(d, m, den=None):
    """d [k, H, W, 3] per-rep paired differences (or a single mean image [H, W, 3]); own statistics inside mask m."""
    dm = d.mean(0) if d.ndim == 4 else d
    den = tot[m].sum() if den is None else den
    x = dm[m]                                                    # [n, 3]
    o = {'L1': float(np.abs(x).sum() / den), 'signed': float(x.sum() / den), 'L1_lum': float(np.abs(x.sum(1)).sum() / den),
         'rms': float(math.sqrt((x ** 2).sum() / (tot[m] ** 2).sum()))}
    tp = tot[m].sum(1)
    rel = np.abs(x).sum(1) / np.maximum(tp, 1e-30)
    ok = tp > 1e-6 * tp.max() if tp.size else tp > 0
    if ok.any():
        o.update({'pp_rel_mean': float(rel[ok].mean()), 'pp_rel_median': float(np.median(rel[ok])), 'pp_rel_p95': float(np.percentile(rel[ok], 95)),
                  'pp_rel_p99': float(np.percentile(rel[ok], 99)), 'pp_rel_max': float(rel[ok].max()),
                  'frac_pix_gt_0.5%': float((rel[ok] > 0.005).mean()), 'frac_pix_gt_1%': float((rel[ok] > 0.01).mean()), 'frac_pix_gt_2%': float((rel[ok] > 0.02).mean())})
    if d.ndim == 4 and d.shape[0] >= 2:
        k = d.shape[0]
        o['nb'] = float(0.5 * np.abs(d[:k // 2].mean(0)[m] - d[k // 2:].mean(0)[m]).sum() / den)
    return o


def blockL1(img, m, b):
    """L1 of block sums (b x b pixels, zero outside the mask), per channel, relative to the mask total."""
    x = img * m[..., None]
    H, W = m.shape
    xb = x.reshape(H // b, b, W // b, b, 3).sum((1, 3))
    return float(np.abs(xb).sum() / tot[m].sum())


# ================================================================================================ A: stored renders
X = {k: lo(k) for k in ARMS}
A = {}
for mn, m in MS.items():
    if m.sum() == 0:
        continue
    for k in ARMS:
        A[f'{k}|{mn}'] = stats(X[k] - X['F'], m)
    A[f'n_img(Fa vs Fb)|{mn}'] = 0.5 * stats(X['Fa'] - X['Fb'], m)['L1']
R['A_stored_paired'] = A
x = A['XOD|ROI_T']; n_img = A['n_img(Fa vs Fb)|ROI_T']
H_ = x['L1'] - n_img
R['A_decision_recomputed'] = {'dI_XOD': x['L1'], 'n_img': n_img, 'nb': x['nb'], 'H': H_, 'U': H_ + 2 * x['nb'], 'Lo': H_ - 2 * x['nb'], 'M_img': M_IMG,
                              'outcome': 'S1_NO_HEADROOM' if H_ + 2 * x['nb'] < M_IMG else 'other', 'U_without_noise_subtraction': x['L1'] + 2 * x['nb']}


def eh(hr, xr, m):
    n, k = hr.shape[0], xr.shape[0]
    h, xx = hr.mean(0), xr.mean(0)
    den = tot[m].sum()
    d = (xx - h)[m]
    o = {'E_H': float(np.abs(d).sum() / den), 'signed': float(d.sum() / den),
         'noise_bound_orig_def': float((0.5 * np.abs(xr[:k // 2].mean(0)[m] - xr[k // 2:].mean(0)[m]).sum() + 0.5 * np.abs(hr[:n // 2].mean(0)[m] - hr[n // 2:].mean(0)[m]).sum()) / den),
         'noise_x_only': float(0.5 * np.abs(xr[:k // 2].mean(0)[m] - xr[k // 2:].mean(0)[m]).sum() / den),
         'noise_h_only': float(0.5 * np.abs(hr[:n // 2].mean(0)[m] - hr[n // 2:].mean(0)[m]).sum() / den)}
    # null expectation of E_H if x and h had the same mean: per pixel-channel Gaussian, E|N(0, s)| = s sqrt(2 / pi)
    vx = xr.var(0, ddof=1)[m] / k; vh = hr.var(0, ddof=1)[m] / n
    o['E_H_null_expectation'] = float(np.sqrt((vx + vh) * 2 / math.pi).sum() / den)
    # signed total with its standard error (rep to rep)
    sx = np.array([xr[i][m].sum() for i in range(k)]); sh = np.array([hr[i][m].sum() for i in range(n)])
    o['signed_se'] = float(math.sqrt(sx.var(ddof=1) / k + sh.var(ddof=1) / n) / den)
    o['ratio_x_over_h'] = float(sx.mean() / sh.mean())
    # debiased relative L2 (per pixel-channel) against the rms of the total
    d2 = float((d ** 2).sum()); v = float((vx + vh).sum()); t2 = float((tot[m] ** 2).sum()); h2 = float((h[m] ** 2).sum())
    o['rel_l2_raw_vs_tot'] = math.sqrt(d2 / t2); o['rel_l2_noise_vs_tot'] = math.sqrt(v / t2)
    o['rel_l2_debiased_vs_tot'] = math.copysign(math.sqrt(abs(d2 - v) / t2), d2 - v)
    o['rel_l2_debiased_vs_h'] = math.copysign(math.sqrt(abs(d2 - v) / h2), d2 - v)
    # block-aggregated L1 (noise shrinks with the block size, a smooth bias does not)
    for b in (4, 16, 64):
        dd = xx - h
        nbk = 0.5 * blockL1(xr[:k // 2].mean(0) - xr[k // 2:].mean(0), m, b) + 0.5 * blockL1(hr[:n // 2].mean(0) - hr[n // 2:].mean(0), m, b)
        o[f'E_H_block{b}'] = blockL1(dd, m, b); o[f'noise_bound_block{b}'] = nbk
    return o


R['A_F_vs_REF_h'] = {mn: eh(ref['h'], X['F'], m) for mn, m in MS.items() if m.sum() > 0}
R['A_arm_vs_REF_h_ROI'] = {k: eh(ref['h'], X[k], MS['ROI_T']) for k in ('F', 'Z', 'XD', 'XOD', 'XOP', 'G8601', 'G8602')}
R['A_ref_shares_ROI'] = {k: float(ref[k].mean(0)[MS['ROI_T']].sum() / tot[MS['ROI_T']].sum()) for k in ('e1', 'e2', 'h')}

# ================================================================================================ B: audit re-renders
have = lambda k: all(os.path.exists(os.path.join(MINE, f'A_{k}_s{SEED0 + r}.npz')) for r in range(NREP))
B = {}
if os.path.exists(os.path.join(MINE, 'A_F_s86600.npz')):
    a = np.load(os.path.join(MINE, 'A_F_s86600.npz'))['hl'].astype(np.float64)
    B['identity_F_seed86600_vs_stored_r0'] = {'max_abs_over_mean': float(np.abs(a - X['F'][0]).max() / X['F'][0].mean()), 'rel_l1': float(np.abs(a - X['F'][0]).sum() / X['F'][0].sum())}
keys = [k for k in ('F', 'Fa', 'Fb', 'XOD', 'dOab', 'XD', 'XOP', 'Z', 'Pc', 'Fn', 'Pn', 'XODn', 'D_S1', 'D_floor', 'D_sphere', 'dOn') if have(k)]
Y = {k: lm(k) for k in keys}
B['keys'] = keys
if all(k in Y for k in ('F', 'Fa', 'Fb', 'XOD')):
    B['linearity_Fa_plus_Fb_minus_2F_rel'] = float(np.abs(Y['Fa'] + Y['Fb'] - 2 * Y['F']).sum() / (2 * Y['F']).sum())
    for mn, m in MS.items():
        if m.sum() == 0:
            continue
        for k in ('Fa', 'Fb', 'XOD', 'XD', 'XOP', 'Z'):
            if k in Y:
                B[f'{k}|{mn}'] = stats(Y[k] - Y['F'], m)
        B[f'n_img(Fa vs Fb)|{mn}'] = 0.5 * stats(Y['Fa'] - Y['Fb'], m)['L1']
        if 'dOab' in Y:
            B[f'n_img_o(e3_o halves only)|{mn}'] = 0.5 * stats(Y['dOab'], m)['L1']
            B[f'n_img_p(e3_p halves only)|{mn}'] = 0.5 * stats(Y['Fa'] - Y['Fb'] - Y['dOab'], m)['L1']
    x = B['XOD|ROI_T']; n_img = B['n_img(Fa vs Fb)|ROI_T']
    dec = {'dI_XOD': x['L1'], 'signed': x['signed'], 'n_img': n_img, 'nb': x['nb'], 'H': x['L1'] - n_img, 'U': x['L1'] - n_img + 2 * x['nb'], 'M_img': M_IMG}
    if 'dOab' in Y:
        no = B['n_img_o(e3_o halves only)|ROI_T']
        dec.update({'n_img_o': no, 'H_matched(dI - n_img_o)': x['L1'] - no, 'U_matched': x['L1'] - no + 2 * x['nb']})
    dec['U_without_noise_subtraction'] = x['L1'] + 2 * x['nb']
    dec['outcome_all_variants_below_M'] = bool(max(dec['U'], dec.get('U_matched', -1), dec['U_without_noise_subtraction']) < M_IMG)
    B['decision_new_seeds'] = dec
    # per-rep (single 4,096-spp frame) values of the decisive statistic
    B['per_rep_dI_XOD_ROI'] = [stats((Y['XOD'] - Y['F'])[r], MS['ROI_T'])['L1'] for r in range(NREP)]
    B['stored_per_rep_dI_XOD_ROI'] = [stats((X['XOD'] - X['F'])[r], MS['ROI_T'])['L1'] for r in range(4)]
    # F (audit seeds) vs REF h, and 8 reps pooled
    B['F_newseeds_vs_REF_h'] = {mn: eh(ref['h'], Y['F'], m) for mn, m in MS.items() if m.sum() > 0 and mn in ('ROI_T', 'DIFF', 'WHOLE', 'PANEL', 'FLOOR_PIX', 'BOXA_PIX')}
    F8 = np.concatenate([X['F'], Y['F']])
    B['F_8reps_vs_REF_h'] = {mn: eh(ref['h'], F8, m) for mn, m in MS.items() if m.sum() > 0 and mn in ('ROI_T', 'DIFF', 'WHOLE', 'PANEL', 'FLOOR_PIX', 'BOXA_PIX')}
if all(k in Y for k in ('F', 'Pc', 'Z', 'XOD')):
    sh = {}
    for mn, m in MS.items():
        if m.sum() == 0:
            continue
        den = tot[m].sum()
        e3o = (Y['F'] - Y['Pc']).mean(0); e3p = (Y['Pc'] - Y['Z']).mean(0); e2 = Y['Z'].mean(0)
        s_o = float(np.abs(e3o[m]).sum() / den)
        sh[mn] = {'table_e2_share': float(e2[m].sum() / den), 'e3_p_share': float(e3p[m].sum() / den), 'e3_o_share': s_o,
                  'uniform_relative_e3_o_error_equal_to_M_img': M_IMG / max(s_o, 1e-30),
                  'explicit_e3_o_error_as_fraction_of_e3_o_read (dI / share)': B[f'XOD|{mn}']['L1'] / max(s_o, 1e-30),
                  'explicit_e3_o_signed_fraction_of_e3_o_read': B[f'XOD|{mn}']['signed'] / max(s_o, 1e-30)}
    B['shares_and_effect_sizes'] = sh
if all(k in Y for k in ('F', 'Pc', 'Fn', 'Pn', 'XOD', 'XODn')):
    e3o_c = (Y['F'] - Y['Pc']); e3o_n = (Y['Fn'] - Y['Pn']); xo_c = (Y['XOD'] - Y['Pc']); xo_n = (Y['XODn'] - Y['Pn'])
    e3o_6 = (2 * e3o_c + 4 * e3o_n) / 6
    ind = {}
    for mn, m in MS.items():
        if m.sum() == 0:
            continue
        den = tot[m].sum()
        d = {'ref_ckf_vs_ref_audit (e3_o as read)': stats(e3o_c - e3o_n, m)['L1'],
             'ref_ckf_vs_ref_audit signed': stats(e3o_c - e3o_n, m)['signed'],
             'ref (e2 + e3_p) ckf vs audit': stats(Y['Pc'] - Y['Pn'], m)['L1'],
             'F ckf vs F audit (whole table)': stats(Y['F'] - Y['Fn'], m)['L1'],
             'explicit(prefix ckf) vs ref ckf': stats(xo_c - e3o_c, m)['L1'], 'explicit(prefix ckf) vs ref audit': stats(xo_c - e3o_n, m)['L1'],
             'explicit(prefix audit) vs ref audit': stats(xo_n - e3o_n, m)['L1'], 'explicit(prefix audit) vs ref ckf': stats(xo_n - e3o_c, m)['L1'],
             'explicit(prefix ckf) vs ref 6 halves': stats(xo_c - e3o_6, m)['L1'], 'explicit(prefix ckf) vs ref 6 halves signed': stats(xo_c - e3o_6, m)['signed'],
             'explicit(prefix ckf) vs explicit(prefix audit)': stats(xo_c - xo_n, m)['L1']}
        # debiased L2 of the explicit readout error with two independent references
        a_ = (xo_c - e3o_c).mean(0)[m]; b_ = (xo_c - e3o_n).mean(0)[m]
        s = float((a_ * b_).sum()); t2 = float((tot[m] ** 2).sum())
        d['explicit rel L2 debiased (cross product of two independent references) vs rms tot'] = math.copysign(math.sqrt(abs(s) / t2), s)
        d['explicit rel L2 raw vs ref ckf'] = math.sqrt(float((a_ ** 2).sum()) / t2)
        d['ref ckf vs ref audit rel L2'] = math.sqrt(float(((e3o_c - e3o_n).mean(0)[m] ** 2).sum()) / t2)
        if 'dOn' in Y:
            d['n_img_o audit (4-half mean)'] = 0.5 * stats(Y['dOn'], m)['L1']
        if 'dOab' in Y:
            d['n_img_o ckf (2-half mean)'] = 0.5 * stats(Y['dOab'], m)['L1']
        ind[mn] = d
    B['independent_reference'] = ind
if all(k in Y for k in ('F', 'XOD', 'D_S1', 'D_floor', 'D_sphere')):
    rc = {}
    Dall = Y['XOD'] - Y['F']
    parts = {'S1_receivers': Y['D_S1'], 'floor_receivers': Y['D_floor'], 'sphere_receivers': Y['D_sphere'], 'lambert_receivers': Dall - Y['D_S1'] - Y['D_floor'] - Y['D_sphere']}
    for mn, m in MS.items():
        if m.sum() == 0:
            continue
        rc[mn] = {pn: {'L1': stats(pv, m)['L1'], 'signed': stats(pv, m)['signed']} for pn, pv in parts.items()}
        rc[mn]['total'] = {'L1': stats(Dall, m)['L1'], 'signed': stats(Dall, m)['signed']}
    B['receiver_class_decomposition (explicit e3_o - reference ckf, as read by the camera)'] = rc
R['B_audit_renders'] = B
json.dump(R, open(OUT, 'w'), indent=1)
print('E4IMG DONE', OUT, flush=True)
