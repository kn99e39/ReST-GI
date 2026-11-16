"""WL97 audit C2 (server, CPU numpy): independent recomputation of the WL96A Stage-1 image decision and its sensitivity to the mask, to the
noise term and to the minimum-effect threshold.  Reads ~/rest_x96a/results (read-only) and the audit renders in <R>/results/renders
(C2F / C2Fp / C2Foa / C2Fob, one rep each, seed 86600).  Usage: python c2_img96a_server.py <R (own)> <out.json>"""
import glob, json, os, sys
import numpy as np

RA = os.path.expanduser('~/rest_x96a/results')
R, OUT = sys.argv[1], sys.argv[2]
M = dict(np.load(os.path.join(RA, 'pre', 'masks94.npz')))
ARMS = ['F', 'Fa', 'Fb', 'Z', 'XD', 'XP', 'XOD', 'XOP', 'XSD', 'XSP', 'G8601', 'G8602']
X = {a: np.stack([np.load(os.path.join(RA, 'renders', f'CK_none_{a}_X_CK2_T1_r{r}.npz'))['hl'].astype(np.float64) for r in range(4)]) for a in ARMS}
ref = [np.load(f) for f in sorted(glob.glob(os.path.join(RA, 'renders', 'REF_ER_CK_none_CK2_T1_r*.npz')))]
h_reps = np.stack([z['h'].astype(np.float64) for z in ref])
tot = np.stack([z['e1'].astype(np.float64) + z['e2'].astype(np.float64) + z['h'].astype(np.float64) for z in ref]).mean(0)
h = h_reps.mean(0)
o = {'mask_info': {k: [list(v.shape), str(v.dtype), int(v.sum()) if v.dtype == bool else None] for k, v in M.items()}}
# masks: the two frozen ones plus audit alternatives
masks = {'ROI_T': M['ROI_T'], 'DIFF': M['DIFF']}
valid = tot.sum(-1) > 0
masks['ALL_VALID'] = valid
for k in ('panel', 'frame', 'vf_boxA'):
    if k in M and M[k].shape == M['ROI_T'].shape and M[k].dtype == bool:
        masks[k] = M[k]
if 'sid' in M and M['sid'].shape == M['ROI_T'].shape:
    sid = M['sid']
    o['sid_values'] = {int(v): int((sid == v).sum()) for v in np.unique(sid)}
    for v in np.unique(sid):
        if (sid == v).sum() > 500:
            masks[f'sid{int(v)}'] = sid == v
masks['NOT_ROI_T'] = valid & ~M['ROI_T']
res = {}
for mn, m in masks.items():
    den = tot[m].sum()
    if den <= 0:
        continue
    r = {'pixels': int(m.sum()), 'tot_share_of_image': float(den / tot[valid].sum()), 'H_share': float(h[m].sum() / den)}
    F = X['F']
    for a in ARMS:
        d = X[a] - F
        dmn = d.mean(0)
        e = {'dI': float(np.abs(dmn[m]).sum() / den), 'signed': float(dmn[m].sum() / den),
             'nb': float(0.5 * np.abs(d[:2].mean(0)[m] - d[2:].mean(0)[m]).sum() / den),
             'dI_rep': [float(np.abs(d[i][m]).sum() / den) for i in range(4)],
             'E_H': float(np.abs(X[a].mean(0)[m] - h[m]).sum() / den),
             'dI_max_pixel_over_mean_tot': float(np.abs(dmn[m]).max() / (den / (3 * m.sum()))),
             'dI_l2': float(np.sqrt((dmn[m] ** 2).sum() / (tot[m] ** 2).sum()))}
        r[a] = e
    dab = X['Fa'] - X['Fb']
    r['n_img'] = float(0.5 * np.abs(dab.mean(0)[m]).sum() / den)
    r['H'] = r['XOD']['dI'] - r['n_img']
    r['U_frozen'] = r['H'] + 2 * r['XOD']['nb']
    r['U_raw_no_debias'] = r['XOD']['dI'] + 2 * r['XOD']['nb']
    # correlation between the XOD - F image and the reference-tail noise image Fa - Fb: if XOD's difference from F is (mostly) F's own noise,
    # (XOD - F) is NOT correlated with (Fa - Fb) [mean vs difference of halves are orthogonal]; reported for completeness
    r['corr_XODmF_with_FamFb'] = float(np.corrcoef((X['XOD'] - F).mean(0)[m].ravel(), dab.mean(0)[m].ravel())[0, 1])
    # is |XOD - F| what is expected if XOD were the truth?  expectation of 0.5|Fa - Fb| equals E|F - truth| -> ratio dI / n_img ~ 1 under H0
    r['dI_XOD_over_n_img'] = r['XOD']['dI'] / max(r['n_img'], 1e-30)
    res[mn] = r
# audit renders (one rep, seed 86600 = rep 0 of every arm)
aud = {}
fn = lambda k: os.path.join(R, 'results', 'renders', f'CK_none_C2{k}_X_CK2_T1_r0.npz')
if all(os.path.exists(fn(k)) for k in ('F', 'Fp', 'Foa', 'Fob')):
    C = {k: np.load(fn(k))['hl'].astype(np.float64) for k in ('F', 'Fp', 'Foa', 'Fob')}
    aud['readout_identity_F_max_abs_over_image_mean'] = float(np.abs(C['F'] - X['F'][0]).max() / X['F'][0].mean())
    for mn, m in masks.items():
        den = tot[m].sum()
        if den <= 0:
            continue
        a = {'e3o_image_share_of_tot': float((C['F'] - C['Fp'])[m].sum() / den),            # image of the resolvent domain (reference e3_o)
             'tail_image_share_of_tot': float((X['F'][0] - X['Z'][0])[m].sum() / den),
             'n_img_e3o_only': float(0.5 * np.abs((C['Foa'] - C['Fob'])[m]).sum() / den),     # e3_o-only reference noise (rep 0)
             'n_img_frozen_rep0': float(0.5 * np.abs((X['Fa'][0] - X['Fb'][0])[m]).sum() / den),
             'dI_XOD_rep0': float(np.abs((X['XOD'][0] - X['F'][0])[m]).sum() / den),
             'dI_XOP_rep0': float(np.abs((X['XOP'][0] - X['F'][0])[m]).sum() / den),
             'dI_XD_rep0': float(np.abs((X['XD'][0] - X['F'][0])[m]).sum() / den)}
        # cross-validated squared difference: (XOD - Foa)(XOD - Fob) estimates |XOD - truth_table_image|^2 without the reference-noise floor
        xa, xb = (X['XOD'][0] - C['Foa'])[m], (X['XOD'][0] - C['Fob'])[m]
        cv = (xa * xb).sum()
        a['cv_rel_l2_XOD_vs_truth'] = float(np.sign(cv) * np.sqrt(abs(cv) / ((C['F'] - C['Fp'])[m] ** 2).sum()))      # relative to the e3_o image
        a['rel_l2_noise_e3o'] = float(np.sqrt((0.25 * (C['Foa'] - C['Fob'])[m] ** 2).sum() / ((C['F'] - C['Fp'])[m] ** 2).sum()))
        xa, xb = (X['XOP'][0] - C['Foa'])[m], (X['XOP'][0] - C['Fob'])[m]
        cv = (xa * xb).sum()
        a['cv_rel_l2_XOP_vs_truth'] = float(np.sign(cv) * np.sqrt(abs(cv) / ((C['F'] - C['Fp'])[m] ** 2).sum()))
        a['dI_XOD_over_e3o_image'] = a['dI_XOD_rep0'] / max(a['e3o_image_share_of_tot'], 1e-30)
        a['signed_XOD_over_e3o_image'] = float((X['XOD'][0] - X['F'][0])[m].sum() / max((C['F'] - C['Fp'])[m].sum(), 1e-30))
        a['signed_XOP_over_e3o_image'] = float((X['XOP'][0] - X['F'][0])[m].sum() / max((C['F'] - C['Fp'])[m].sum(), 1e-30))
        aud[mn] = a
o['masks'] = res; o['audit_renders'] = aud
json.dump(o, open(OUT, 'w'), indent=1)
print(json.dumps(o['mask_info']))
for mn, r in res.items():
    print(f"== {mn}: px {r['pixels']} tot-share {r['tot_share_of_image']:.3f} H-share {r['H_share']:.3f} n_img {r['n_img']:.5f} H {r['H']:+.5f} U_frozen {r['U_frozen']:+.5f} U_raw {r['U_raw_no_debias']:.5f} dI/n {r['dI_XOD_over_n_img']:.2f}")
    for a in ARMS:
        e = r[a]
        print(f"   {a:6s} dI {e['dI']:.5f} signed {e['signed']:+.5f} nb {e['nb']:.5f} E_H {e['E_H']:.4f} maxpix {e['dI_max_pixel_over_mean_tot']:.3f} l2 {e['dI_l2']:.5f}")
print(json.dumps(aud, indent=1))
