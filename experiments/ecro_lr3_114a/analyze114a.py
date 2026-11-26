"""WL114A LR3 frozen analysis: GI scores (metrics102 definitions) vs the qualified full-depth reference Ref-65 and the 6-event
Ref-7, depth / representation / learned decomposition, operating points, controls, and the predeclared H1-LR3 verdict.

  python analyze114a.py ROOT
"""
import json, sys
from pathlib import Path
import numpy as np
from skimage.metrics import structural_similarity

HERE = Path(__file__).resolve().parent
REFD = Path(r'C:\Users\dna10\TransGI-Reproduction-data\runs\living_room_3\reference')
MASKS = Path(r'C:\Projects\ReST-GI\results\ecro_fd114\primary\stageA\lr3\lr3_masks.npz')


def ldr(x):
    return np.clip(np.asarray(x, np.float64), 0, 1) ** (1 / 2.2)


def metrics(x, r, mask=None):
    """metrics102.metrics (TransGI reproduction definitions + ReST-GI conventions), copied unchanged."""
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


def score(x, r, masks):
    o = {'full_frame': metrics(x, r)}
    for k, m in masks.items():
        o[k] = metrics(x, r, m)
    return o


def halves(ev, tag, keys):
    hs = [np.load(ev / f'{tag}_h{i}.npz') for i in (0, 1)]
    return {k: 0.5 * (hs[0][k] + hs[1][k]) for k in keys}, {k: float(np.sqrt(((hs[0][k] - hs[1][k]) ** 2).mean()) / 2) for k in keys}


def main(root):
    root = Path(root); ev = root / 'eval'; an = root / 'analysis'; an.mkdir(parents=True, exist_ok=True)
    R65, R7 = np.load(REFD / 'static_V0.npz'), np.load(REFD / 'static_V0_d7.npz')
    mk = np.load(MASKS); masks = {k[5:]: mk[k] for k in mk.files if k.startswith('mask_') and mk[k].sum() >= 500}
    meta = json.loads((ev / 'eval_meta.json').read_text())
    spp = 16 if any('spp16' in p.name for p in ev.glob('*.npz')) else 512
    est = {}; noise = {}
    for tag in ('arm_A', 'arm_B', 'oracle_D6', 'oracle_F'):
        h, n = halves(ev, f'{tag}_spp{spp}', ('e2', 'hl'))
        est[tag] = h['e2'] + h['hl']; noise[tag] = n
    q = root.parent / 'qual'
    a2 = np.load(q / 'a2_halves.npz')
    est['phys_D6_local'] = 0.5 * (a2['e2_0'] + a2['h_0'] + a2['e2_1'] + a2['h_1'])
    dd = [np.load(q / f'deep96_h{i}.npz') for i in (0, 1)]
    est['phys_deep96_local'] = 0.5 * sum(sum(d[n] for n in ('e2', 'h', 'c7_12', 'c13_24', 'c25_48', 'c49_96') if n in d.files) for d in dd)
    for a in ('A', 'B'):
        z = np.load(ev / f'arm_{a}_spp4.npz'); est[f'arm_{a}_4spp'] = z['e2'] + z['hl']
    est['Ref7'] = R7['gi']
    res = {'vs_Ref65': {k: score(v, R65['gi'], masks) for k, v in est.items()},
           'vs_Ref7': {k: score(v, R7['gi'], masks) for k, v in est.items() if k != 'Ref7'},
           'learned_vs_own_oracle': {'A_vs_oracle_D6': metrics(est['arm_A'], est['oracle_D6']), 'B_vs_oracle_F': metrics(est['arm_B'], est['oracle_F'])},
           'readout_noise': noise, 'meta': meta}
    # table-level (point lights): whole rel L1 / val half-noise of the arm's own target
    tn = {}
    for a in ('A', 'B'):
        t = json.loads((root / f'model_{a}' / 'training.json').read_text())
        tn[a] = {'val_whole_rel_l1': t['validation']['whole_rel_l1'], 'collapsed': bool(t['validation']['whole_rel_l1'] >= 0.99), 'seed': t['seed']}
    res['training'] = tn
    s65 = res['vs_Ref65']; lA = res['learned_vs_own_oracle']['A_vs_oracle_D6']['ldr_rmse']; lB = res['learned_vs_own_oracle']['B_vs_oracle_F']['ldr_rmse']
    rA, rB = s65['arm_A']['full_frame']['ldr_rmse'], s65['arm_B']['full_frame']['ldr_rmse']; bB = s65['arm_B']['full_frame']['bias_rel']
    crit = {'rmse_B_le_half_A': bool(rB <= 0.5 * rA), 'abs_bias_B_le_5pct': bool(abs(bB) <= 0.05), 'learned_B_le_1p5_A_plus_0005': bool(lB <= 1.5 * lA + 0.005),
            'no_collapse': bool(not tn['A']['collapsed'] and not tn['B']['collapsed'])}
    if not crit['no_collapse']:
        v = 'NOT EVALUABLE (collapse)'
    elif all(crit.values()):
        v = 'SUPPORTED'
    elif rB >= 0.9 * rA or abs(bB) >= 0.15 or lB >= 2 * lA + 0.01:
        v = 'NOT SUPPORTED'
    else:
        v = 'INCONCLUSIVE'
    res['H1_LR3'] = {'criteria': crit, 'gi_ldr_rmse_vs_ref65': {'A': rA, 'B': rB}, 'bias_B': bB, 'learned_ldr_rmse': {'A': lA, 'B': lB}, 'verdict': v}
    json.dump(res, open(an / 'lr3_metrics.json', 'w'), indent=1)
    np.savez_compressed(an / 'lr3_maps.npz', ref65_gi=R65['gi'], ref7_gi=R7['gi'], **{k: v.astype(np.float32) for k, v in est.items()})
    print('H1_LR3', json.dumps(res['H1_LR3']), flush=True)


if __name__ == '__main__':
    main(sys.argv[1])
