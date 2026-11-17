"""WL102 scoring (numpy + scikit-image only; no import of either method). Metric definitions replicate the TransGI reproduction's
`transgi/metrics.py` (LDR = clip(x, 0, 1)^(1/2.2): RMSE / L1 / SSIM (skimage, channel_axis=-1, data_range=1); HDR RMSE, MAE,
rel-MSE = mean((e^2 mean over rgb) / (lum_ref^2 + 1e-2)), bias) plus ReST-GI conventions (linear HDR PSNR with the reference
maximum as peak, relative RMSE = RMSE / mean(ref), relative L1 = sum|e| / sum(ref)).

  python metrics102.py <ref65.npz> <ref7.npz> <ecro_eval_dir> <transgi_gi.npy> <masks.npz> <out.json>
"""
import json, sys
import numpy as np
from skimage.metrics import structural_similarity

ref65, ref7, ed, tg_path, mpath, out = sys.argv[1:7]
R65, R7 = np.load(ref65), np.load(ref7)
mk = np.load(mpath)
MASKS = {k[5:]: mk[k] for k in mk.files if k.startswith('mask_')}


def ldr(x):
    return np.clip(np.asarray(x, np.float64), 0, 1) ** (1 / 2.2)


def metrics(x, r, mask=None):
    x = np.asarray(x, np.float64); r = np.asarray(r, np.float64)
    a, b = ldr(x), ldr(r)
    if mask is not None:
        xs, rs, e = x[mask], r[mask], (a - b)[mask]
    else:
        xs, rs, e = x, r, a - b
    eh = xs - rs
    lum = rs.mean(-1)
    o = {'ldr_rmse': float(np.sqrt((e ** 2).mean())), 'ldr_l1': float(np.abs(e).mean()),
         'hdr_rmse': float(np.sqrt((eh ** 2).mean())), 'hdr_mae': float(np.abs(eh).mean()),
         'hdr_relmse': float(((eh ** 2).mean(-1) / (lum ** 2 + 1e-2)).mean()), 'bias_rel': float(eh.mean() / max(rs.mean(), 1e-12)),
         'mean_est': float(xs.mean()), 'mean_ref': float(rs.mean()),
         'rel_rmse': float(np.sqrt((eh ** 2).mean()) / max(rs.mean(), 1e-12)), 'rel_l1': float(np.abs(eh).sum() / max(rs.sum(), 1e-12)),
         'psnr_linear': float(10 * np.log10(rs.max() ** 2 / max((eh ** 2).mean(), 1e-30)))}
    if mask is None:
        o['ldr_ssim'] = float(structural_similarity(a, b, channel_axis=-1, data_range=1.0))
    return o


def score(x, r):
    o = {'full_frame': metrics(x, r)}
    for k, m in MASKS.items():
        if m.sum():
            o[k] = metrics(x, r, m)
    return o


def mean_halves(prefix, keys):
    hs = [np.load(f'{ed}/{prefix}_h{i}.npz') for i in (0, 1)]
    return {k: 0.5 * (hs[0][k] + hs[1][k]) for k in keys}, {k: float(np.sqrt(((hs[0][k] - hs[1][k]) ** 2).mean()) / 2) for k in keys}


e4 = np.load(f'{ed}/ecro_spp4.npz')
e512, n512 = mean_halves('ecro_spp512', ('e1', 'e2', 'hl'))
p6, np6 = mean_halves('phys_d6_spp1024', ('e1', 'e2', 'h'))
tg = np.load(tg_path)
DI = R65['di']
est = {
    'TransGI_4spp': tg,
    'ECRO_4spp': e4['e2'] + e4['hl'],
    'ECRO_converged_1024spp': e512['e2'] + e512['hl'],
    'PhysD6_2048spp (ECRO with an exact table; = Ref-7 estimator)': p6['e2'] + p6['h'],
}
res = {'reference_noise_gi': {'ref65': float(np.sqrt(((R65['gi_A'] - R65['gi_B']) ** 2).mean()) / 2),
                              'ref7': float(np.sqrt(((R7['gi_A'] - R7['gi_B']) ** 2).mean()) / 2)},
       'estimate_noise': {'ECRO_converged_halves': n512, 'PhysD6_halves': np6},
       'GI_vs_Ref65': {k: score(v, R65['gi']) for k, v in est.items()},
       'GI_vs_Ref7': {k: score(v, R7['gi']) for k, v in est.items()},
       'GIplusRefDI_vs_Ref65': {k: score(v + DI, R65['full']) for k, v in est.items()},
       'ECRO_own_full_vs_Ref65_full': {'ECRO_4spp': score(e4['e1'] + e4['e2'] + e4['hl'], R65['full']),
                                       'ECRO_converged_1024spp': score(e512['e1'] + e512['e2'] + e512['hl'], R65['full'])},
       'truncation_Ref7_vs_Ref65': {'GI': score(R7['gi'], R65['gi']), 'FULL': score(R7['full'], R65['full'])},
       'ecro_table_only_hl_vs_physical_h': score(e512['hl'], p6['h']),
       'ecro_e2_vs_physical_e2': score(e512['e2'], p6['e2'])}
json.dump(res, open(out, 'w'), indent=1)
ff = lambda d: {k: round(v['full_frame']['ldr_rmse'], 4) for k, v in d.items()}
print('GI vs Ref65 LDR RMSE', ff(res['GI_vs_Ref65'])); print('GI vs Ref7 LDR RMSE', ff(res['GI_vs_Ref7']))
print('truncation GI LDR RMSE', round(res['truncation_Ref7_vs_Ref65']['GI']['full_frame']['ldr_rmse'], 4),
      'table-only hl vs h rel_l1', round(res['ecro_table_only_hl_vs_physical_h']['full_frame']['rel_l1'], 4))
