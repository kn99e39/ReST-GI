"""WL99 diagnostic (after Stage 0; not a gate): where is ROI_sharp? fraction of ROI_sharp pixels within 2 px of the body-mask / self-ROI
boundary (geometric silhouettes), and Q3 recomputed on the interior part only (descriptive)."""
import json, os, sys
import numpy as np
import wl99
S0 = sys.argv[1]
q = json.load(open(os.path.join(S0, 'qual99.json')))['workloads']


def dilate(m, r):
    out = m.copy()
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            out |= np.roll(np.roll(m, dy, 0), dx, 1)
    return out


res = {}
for wid, v in q.items():
    if v['type'] == 'C':
        continue
    spp = v['spp_per_half']
    for st in ('T0', 'T1'):
        A = dict(np.load(os.path.join(S0, 'refs', f'{wid}_{st}_A_{spp}.npz'))); B = dict(np.load(os.path.join(S0, 'refs', f'{wid}_{st}_B_{spp}.npz')))
        R = {k: (A[k].astype(np.float64) + B[k]) / 2 for k in A}
        body, roi, sharp = wl99.masks(R)
        edge = dilate(~body, 2) | dilate(~(R['mb'] >= 0.99) , 2)
        near = sharp & edge
        inner = sharp & ~edge
        sd = wl99.lum(R['Sd'])
        rng, gmax, mean = wl99.window_stats(sd, 3)
        f = lambda m: (float(np.median(rng[m] / np.maximum(gmax[m], 1e-30))), float(np.median(rng[m] / np.maximum(mean[m], 1e-30)))) if m.sum() > 10 else (None, None)
        # noise contribution to the gradient: same statistic on the half difference
        dn = wl99.lum(A['Sd'] - B['Sd']) / 2
        gy, gx = np.gradient(dn); gn = np.hypot(gx, gy)
        gy, gx = np.gradient(sd); gs = np.hypot(gx, gy)
        res[f'{wid}/{st}'] = {'n_sharp': int(sharp.sum()), 'frac_sharp_near_silhouette': float(near.sum() / max(sharp.sum(), 1)),
                              'all': f(sharp), 'interior': f(inner), 'noise_grad_over_signal_grad_median_sharp': float(np.median(gn[sharp] / np.maximum(gs[sharp], 1e-30)))}
        print(wid, st, json.dumps(res[f'{wid}/{st}']))
json.dump(res, open(os.path.join(S0, 'diag_q3.json'), 'w'), indent=1)
