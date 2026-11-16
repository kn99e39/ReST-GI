"""WL97 audit C2 (server, CPU numpy only): independent recomputation of the WL95A image-level E_H table and of the noise of arm-vs-arm
E_H differences.  Reads ~/rest_x95a/results (read-only); writes only to the output path given as argv[1]."""
import glob, json, os, sys
import numpy as np

R = os.path.expanduser('~/rest_x95a/results')
out_fn = sys.argv[1]
M = dict(np.load(os.path.join(R, 'pre', 'masks94.npz')))
roi = M['ROI_T']
ref = [np.load(f) for f in sorted(glob.glob(os.path.join(R, 'renders', 'REF_ER_CK_none_CK2_T1_r*.npz')))]
h_reps = np.stack([z['h'].astype(np.float64) for z in ref])
tot = np.stack([(z['e1'].astype(np.float64) + z['e2'].astype(np.float64) + z['h'].astype(np.float64)) for z in ref]).mean(0)
h = h_reps.mean(0)
den = tot[roi].sum()
arms = sorted({os.path.basename(f)[len('CK_none_'):-len('_X_CK2_T1_r0.npz')] for f in glob.glob(os.path.join(R, 'renders', 'CK_none_*_X_CK2_T1_r0.npz'))})
X = {a: np.stack([np.load(os.path.join(R, 'renders', f'CK_none_{a}_X_CK2_T1_r{r}.npz'))['hl'].astype(np.float64) for r in (0, 1)]) for a in arms}
o = {'mask_keys': sorted(M), 'roi_pixels': int(roi.sum()), 'img_shape': list(h.shape), 'n_ref_reps': len(ref), 'arms': {}}
n_ref = h_reps.shape[0]
ref_half = 0.5 * np.abs(h_reps[:n_ref // 2].mean(0)[roi] - h_reps[n_ref // 2:].mean(0)[roi]).sum() / den
o['ref_half_L1'] = float(ref_half)
o['ref_signed_half'] = float(0.5 * (h_reps[:n_ref // 2].mean(0)[roi] - h_reps[n_ref // 2:].mean(0)[roi]).sum() / den)
o['H_share'] = float(h[roi].sum() / den)
fl = X['floor']
for a in arms:
    x = X[a]
    xm = x.mean(0)
    r = {'E_H': float(np.abs(xm[roi] - h[roi]).sum() / den), 'signed': float((xm[roi] - h[roi]).sum() / den),
         'arm_half_L1': float(0.5 * np.abs(x[0][roi] - x[1][roi]).sum() / den),
         'noise_bound': float(0.5 * np.abs(x[0][roi] - x[1][roi]).sum() / den + ref_half),
         # per-rep E_H against the full reference mean
         'E_H_rep': [float(np.abs(x[i][roi] - h[roi]).sum() / den) for i in (0, 1)],
         'signed_rep': [float((x[i][roi] - h[roi]).sum() / den) for i in (0, 1)],
         # arm - floor (common render seeds): effect size of the table difference on the image
         'L1_vs_floor': float(np.abs(xm[roi] - fl.mean(0)[roi]).sum() / den),
         'L1_vs_floor_rep': [float(np.abs(x[i][roi] - fl[i][roi]).sum() / den) for i in (0, 1)],
         'signed_vs_floor_rep': [float((x[i][roi] - fl[i][roi]).sum() / den) for i in (0, 1)],
         # noise of the arm - floor difference image: half difference between the two reps of the difference
         'diff_half_L1': float(0.5 * np.abs((x[0] - fl[0])[roi] - (x[1] - fl[1])[roi]).sum() / den),
         # E_H(arm) - E_H(floor) per rep (paired; reference identical)
         'dE_H_vs_floor_rep': [float((np.abs(x[i][roi] - h[roi]).sum() - np.abs(fl[i][roi] - h[roi]).sum()) / den) for i in (0, 1)],
         'dE_H_vs_floor': float((np.abs(xm[roi] - h[roi]).sum() - np.abs(fl.mean(0)[roi] - h[roi]).sum()) / den),
         # cross-rep correlation of arm noise with floor noise (common random numbers?)
         'corr_noise_with_floor': float(np.corrcoef((x[0] - x[1])[roi].ravel(), (fl[0] - fl[1])[roi].ravel())[0, 1])}
    # E_H against each half of the reference: sensitivity of E_H differences to the reference noise realisation
    r['E_H_vs_ref_half'] = [float(np.abs(xm[roi] - h_reps[:n_ref // 2].mean(0)[roi]).sum() / den), float(np.abs(xm[roi] - h_reps[n_ref // 2:].mean(0)[roi]).sum() / den)]
    # block-aggregated error (8 x 8 pixel blocks inside ROI): pixel noise averages out, bias stays
    hh, ww = roi.shape[:2]
    if roi.ndim == 2 and xm.ndim == 3:
        d = (xm - h).sum(-1) * roi
        t = tot.sum(-1) * roi
        B = 8
        hb, wb = hh // B * B, ww // B * B
        db = d[:hb, :wb].reshape(hb // B, B, wb // B, B).sum((1, 3))
        fd = ((fl.mean(0) - h).sum(-1) * roi)[:hb, :wb].reshape(hb // B, B, wb // B, B).sum((1, 3))
        r['E_H_block8'] = float(np.abs(db).sum() / t[:hb, :wb].sum())
        r['E_H_block8_floor'] = float(np.abs(fd).sum() / t[:hb, :wb].sum())
    o['arms'][a] = r
o['roi_ndim'] = int(roi.ndim)
o['hl_ndim'] = int(h.ndim)
json.dump(o, open(out_fn, 'w'), indent=1)
print(json.dumps({k: v for k, v in o.items() if k != 'arms'}))
for a in arms:
    r = o['arms'][a]
    print(f"{a:16s} E_H {r['E_H']:.4f} signed {r['signed']:+.4f} nb {r['noise_bound']:.4f} | reps {r['E_H_rep'][0]:.4f} {r['E_H_rep'][1]:.4f} | dE_H vs floor {r['dE_H_vs_floor']:+.4f} reps {r['dE_H_vs_floor_rep'][0]:+.4f} {r['dE_H_vs_floor_rep'][1]:+.4f}"
          f" | L1 vs floor {r['L1_vs_floor']:.4f} (diff half-noise {r['diff_half_L1']:.4f}) signed vs floor reps {r['signed_vs_floor_rep'][0]:+.4f} {r['signed_vs_floor_rep'][1]:+.4f} | corr-noise {r['corr_noise_with_floor']:.2f}"
          f" | blk8 {r.get('E_H_block8', float('nan')):.4f}")
