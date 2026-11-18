"""WL104 analysis: metrics, error decomposition, error maps (one absolute shared scale per worklog, square-root display), gallery.

  python analyze104.py <results_root> <reviewer_dir>
<results_root>/refs/<image>/, eval/<image>/, models/<tag>/training.json, gates_pre.json, gates_refs.json, chain.json
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

import common104 as C

ROOT = Path(sys.argv[1]).resolve(); OUT = Path(sys.argv[2]).resolve(); OUT.mkdir(parents=True, exist_ok=True)
LUMA = np.array([0.2126, 0.7152, 0.0722])
lum = lambda x: x @ LUMA


def rel_l1(a, b, m):
    return float(np.abs(a - b)[m].sum() / max(np.abs(b)[m].sum(), 1e-30))


def rel_rmse(a, b, m):
    return float(np.sqrt(((a - b)[m] ** 2).mean() / max((b[m] ** 2).mean(), 1e-30)))


def load_masks(image):
    import rt104 as RT
    tag, light, _, _ = C.image_spec(image)
    RT.register(tag)
    obj, ndv = RT.R81.masks(RT.scene(tag, 'lay', light=light, with_sensor=False))
    return obj, ndv


MASKS, M_ALL, IMG = {}, {}, {}
for image in C.IMAGES:
    d = ROOT / 'refs' / image; e = ROOT / 'eval' / image
    h = lambda arm: [np.load(d / f'{arm}_h{i}.npz')['mean'].astype(np.float64) for i in (0, 1)]
    R = {a: h(a) for a in ('LAY65', 'LAY7', 'ER7', 'ER65', 'LAY4')}
    ptc = [np.load(d / f'PTC_h{i}.npz') for i in (0, 1)]
    ecro4 = np.load(e / 'ecro_spp4.npy').astype(np.float64)               # [n, 3 (e1, e2, hl), H, W, 3]
    ecro64 = np.load(e / 'ecro_spp64.npy').astype(np.float64)
    ecroc = np.load(e / 'ecro_spp1024_halves.npy').astype(np.float64)
    er64 = [np.load(d / f'ER65_64_r{q}.npz')['mean'].astype(np.float64) for q in range(8)]
    tag = image.rsplit('_L', 1)[0]
    w84_er = np.load(C.W83REN / f'{image}_ER_L64.npy').astype(np.float64)
    w84_ref = np.load(C.W84REF / f'{image}_ref84.npy').astype(np.float64)
    w84_reps = [np.load(C.W84REP / f'{image}_ER_L_rep{q}.npy').astype(np.float64) for q in range(8)]
    obj, ndv = load_masks(image)
    ref = 0.5 * (R['LAY65'][0] + R['LAY65'][1])
    lr = lum(ref)
    hl = obj & (lr >= 3 * np.median(lr[obj]))
    gz = obj & (ndv <= 0.25)
    full = np.ones_like(obj)
    MASKS[image] = {'full': full, 'obj': obj, 'highlight': hl, 'grazing': gz}
    IMG[image] = {'ref': ref, 'ecro4': ecro4, 'ecro64': ecro64, 'ecroc': ecroc, 'er64': er64, 'R': R, 'ptc': ptc,
                  'w84_er': w84_er, 'w84_ref': w84_ref, 'w84_reps': w84_reps}

metrics = {'images': {}, 'masks': {}}
for image in C.IMAGES:
    I = IMG[image]; Mk = MASKS[image]; ref = I['ref']; R = I['R']
    metrics['masks'][image] = {k: int(v.sum()) for k, v in Mk.items()}
    mean2 = lambda x: 0.5 * (x[0] + x[1])
    arms = {
        'ECRO_4spp': [f.sum(0) for f in I['ecro4']],
        'ECRO_64spp': [f.sum(0) for f in I['ecro64']],
        'ECRO_converged_2x1024': [mean2([f.sum(0) for f in I['ecroc']])],
        'LearnedER_PT_full_64spp': I['er64'],
        'LearnedER_PT_full_converged': [mean2(R['ER65'])],
        'LearnedER_PT_6event_converged': [mean2(R['ER7'])],
        'Layered_PT_6event_converged': [mean2(R['LAY7'])],
    }
    am = {}
    for a, imgs in arms.items():
        rows = [{f'rel_l1_{k}': rel_l1(x, ref, m) for k, m in Mk.items()} | {'rel_rmse_full': rel_rmse(x, ref, Mk['full']),
                'rel_rmse_obj': rel_rmse(x, ref, Mk['obj']), 'mean_ratio_obj': float(lum(x)[Mk['obj']].mean() / lum(ref)[Mk['obj']].mean()),
                'mean_ratio_full': float(lum(x).mean() / lum(ref).mean())} for x in imgs]
        am[a] = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]} | ({'sd_over_seeds': {k: float(np.std([r[k] for r in rows], ddof=1)) for k in rows[0]},
                                                                               'n': len(rows)} if len(rows) > 1 else {'n': 1})
    # decomposition (converged; relative L1 against the second argument's own magnitude)
    ec = mean2([f.sum(0) for f in I['ecroc']])
    ec_low = mean2([f[0] + f[1] for f in I['ecroc']]); ec_hl = mean2([f[2] for f in I['ecroc']])
    p = I['ptc']; pt_low = mean2([z['e1'].astype(np.float64) + z['e2'] for z in p]); pt_h = mean2([z['h'].astype(np.float64) for z in p])
    dec = {}
    for k, m in Mk.items():
        dec[k] = {'representation_ECRO_vs_LearnedER6': rel_l1(ec, mean2(R['ER7']), m),
                  'HIGH_only_learned_vs_physical_E3to6': rel_l1(ec_hl, pt_h, m),
                  'LOW_check_ECRO_vs_tracer_E1E2': rel_l1(ec_low, pt_low, m),
                  'material_LearnedER6_vs_Layered6': rel_l1(mean2(R['ER7']), mean2(R['LAY7']), m),
                  'truncation_Layered6_vs_Layered65': rel_l1(mean2(R['LAY7']), ref, m),
                  'material_full_LearnedER65_vs_Layered65': rel_l1(mean2(R['ER65']), ref, m),
                  'total_ECRO_vs_Layered65': rel_l1(ec, ref, m)}
    noise = {f'{a}_half_rel_l1_obj': rel_l1(R[a][0], R[a][1], Mk['obj']) for a in R}
    noise['PTC_h_half_rel_l1_obj'] = rel_l1(p[0]['h'].astype(np.float64), p[1]['h'].astype(np.float64), Mk['obj'])
    noise['ECRO_conv_half_rel_l1_obj'] = rel_l1(I['ecroc'][0].sum(0), I['ecroc'][1].sum(0), Mk['obj'])
    # WL84 context at depth 4 (analytic sphere): stored WL83 ER_L@64 and the 8 WL84 replicates vs the WL84 65,536-spp reference
    hl84 = Mk['obj'] & (lum(I['w84_ref']) >= 3 * np.median(lum(I['w84_ref'])[Mk['obj']]))
    w84 = {'ER_L64_stored_obj': rel_l1(I['w84_er'], I['w84_ref'], Mk['obj']), 'ER_L64_stored_highlight': rel_l1(I['w84_er'], I['w84_ref'], hl84),
           'ER_L64_replicates_obj_mean': float(np.mean([rel_l1(x, I['w84_ref'], Mk['obj']) for x in I['w84_reps']])),
           'ER_L64_replicates_highlight_mean': float(np.mean([rel_l1(x, I['w84_ref'], hl84) for x in I['w84_reps']])),
           'note': 'depth 4 (max 3 scattering events), analytic sphere and rectangles; obj mask from the mesh scene (center rays)'}
    ev = json.loads((ROOT / 'eval' / image / 'eval.json').read_text(encoding='utf-8'))
    rt = json.loads((ROOT / 'refs' / image / 'refs_timing.json').read_text(encoding='utf-8'))
    timing = {'ECRO_table_s': ev['timing']['source_s'] + ev['timing']['factors_s'] + ev['timing']['apply_s'], 'ECRO_parts': {k: ev['timing'][k] for k in ('source_s', 'factors_s', 'apply_s')},
              'ECRO_frame_4spp_wall_median_s': ev['timing']['frame_4spp_wall_s']['median_warm'], 'ECRO_frame_64spp_wall_median_s': ev['timing']['frame_64spp_wall_s']['median_warm'],
              'LearnedER_PT_full_64spp_wall_median_s': float(np.median([rt[f'ER65_64_r{q}']['wall_s'] for q in range(1, 8)])),
              'LearnedER_PT_full_64spp_gpu_ms_median': float(np.median([rt[f'ER65_64_r{q}']['gpu_ms'] for q in range(1, 8)]))}
    metrics['images'][image] = {'arms_vs_Layered65': am, 'decomposition_rel_l1': dec, 'reference_noise': noise, 'wl84_context_depth4': w84,
                                'table_level_eval_lights': ev['table_level_eval_lights_L1_L2'], 'timing_local_desktop': timing}
for tag in sorted({i.rsplit('_L', 1)[0] for i in C.IMAGES}):
    tj = ROOT / 'models' / tag / 'training.json'
    if tj.exists():
        t = json.loads(tj.read_text(encoding='utf-8'))
        metrics.setdefault('training', {})[tag] = {k: t[k] for k in ('seed', 'steps', 'train_s', 'validation', 'N', 'checkpoint_sha256', 'peak_mib')} | {'log': t['log']}
for f in ('gates_pre.json', 'gates_refs.json', 'chain.json'):
    if (ROOT / f).exists():
        metrics[f[:-5]] = json.loads((ROOT / f).read_text(encoding='utf-8'))
C.write_json(OUT / 'metrics.json', metrics)

# ------------------------------------------------------------------------------------------------ figures
import matplotlib  # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

plt.rcParams.update({'font.size': 8, 'figure.facecolor': '#fcfcfb', 'axes.facecolor': '#fcfcfb', 'axes.edgecolor': '#52514e'})
SEQ = LinearSegmentedColormap.from_list('seq', ['#fcfcfb', '#cde2fb', '#86b6ef', '#3987e5', '#1c5cab', '#0d366b'])
DIV = LinearSegmentedColormap.from_list('div', ['#1c5cab', '#86b6ef', '#f0efec', '#f3a07f', '#c4471a'])
EXPO = 1.0 / float(np.percentile(np.concatenate([lum(IMG[i]['ref']).ravel() for i in C.IMAGES]), 99.5))
tone = lambda x: np.clip((np.clip(EXPO * x, 0, None) / (1 + np.clip(EXPO * x, 0, None))) ** (1 / 2.2), 0, 1)
COLS = [('ECRO 4 spp (seed 0)', lambda I: I['ecro4'][0].sum(0), 'ref'), ('ECRO 64 spp (seed 0)', lambda I: I['ecro64'][0].sum(0), 'ref'),
        ('ECRO converged 2x1024', lambda I: 0.5 * (I['ecroc'][0].sum(0) + I['ecroc'][1].sum(0)), 'ref'),
        ('Learned-ER PT full depth, 64 spp', lambda I: I['er64'][0], 'ref'),
        ('Learned-ER PT full depth, converged', lambda I: 0.5 * (I['R']['ER65'][0] + I['R']['ER65'][1]), 'ref'),
        ('WL84 ER_L 64 spp (depth 4) vs WL84 ref', lambda I: I['w84_er'], 'w84_ref')]
errs = {i: [lum(f(IMG[i]) - IMG[i][r]) for _, f, r in COLS] for i in C.IMAGES}
DECS = [('representation: ECRO conv - Learned-ER 6-event', lambda I: 0.5 * (I['ecroc'][0].sum(0) + I['ecroc'][1].sum(0)) - 0.5 * (I['R']['ER7'][0] + I['R']['ER7'][1])),
        ('HIGH only: learned hl - physical E3..6', lambda I: 0.5 * (I['ecroc'][0][2] + I['ecroc'][1][2]) - 0.5 * (I['ptc'][0]['h'] + I['ptc'][1]['h'])),
        ('material: Learned-ER 6-event - layered 6-event', lambda I: 0.5 * (I['R']['ER7'][0] + I['R']['ER7'][1]) - 0.5 * (I['R']['LAY7'][0] + I['R']['LAY7'][1])),
        ('truncation: layered 6-event - layered full', lambda I: 0.5 * (I['R']['LAY7'][0] + I['R']['LAY7'][1]) - I['ref'])]
decs = {i: [lum(f(IMG[i])) for _, f in DECS] for i in C.IMAGES}
M = float(max(max(np.abs(e).max() for e in v) for v in list(errs.values()) + list(decs.values())))
M0 = M / 1000.0
disp = lambda x: np.sign(x) * np.log1p(np.abs(x) / M0) / np.log1p(M / M0)
SCALES = {'error_scale_max_linear_hdr_luminance': M, 'mapping': 'sign(x) log(1 + |x| / m0) / log(1 + M / m0), m0 = M / 1000; one scale for every error / difference map of WL104',
          'display_exposure': EXPO, 'tone': '(e x / (1 + e x))^(1/2.2), display only'}


def cbar(fig, im, axes, signed):
    vals = M * np.array([0, 0.001, 0.01, 0.1, 1.0])
    vals = np.concatenate([-vals[::-1][:-1], vals]) if signed else vals
    cb = fig.colorbar(im, ax=axes, shrink=0.8, ticks=disp(vals))
    cb.ax.set_yticklabels([f'{v:.3g}' for v in vals])
    cb.set_label(('signed ' if signed else 'absolute ') + 'error, linear HDR luminance', fontsize=7)


for image in C.IMAGES:
    I = IMG[image]
    fig, ax = plt.subplots(3, len(COLS) + 1, figsize=(2.6 * (len(COLS) + 1), 8.2), constrained_layout=True)
    ax[0, 0].imshow(tone(I['ref'])); ax[0, 0].set_title('reference: layered PT full depth\n2 x 32768 spp', fontsize=7)
    ax[1, 0].axis('off'); ax[2, 0].axis('off')
    for j, (nm, f, r) in enumerate(COLS, 1):
        x = f(I); e = errs[image][j - 1]
        ax[0, j].imshow(tone(x)); ax[0, j].set_title(nm, fontsize=7)
        im1 = ax[1, j].imshow(disp(np.abs(e)), cmap=SEQ, vmin=0, vmax=1)
        im2 = ax[2, j].imshow(disp(e), cmap=DIV, vmin=-1, vmax=1)
        o = MASKS[image]['obj']
        ax[1, j].set_title(f'|err| obj rel L1 {rel_l1(x, I[r], o):.3f}', fontsize=7)
    for a in ax.ravel():
        a.set_xticks([]); a.set_yticks([])
    cbar(fig, im1, ax[1, 1:].tolist(), False); cbar(fig, im2, ax[2, 1:].tolist(), True)
    fig.suptitle(f'WL104 {image}: images (row 1), absolute (row 2) and signed (row 3) luminance error; one error scale for the whole worklog', fontsize=9)
    fig.savefig(OUT / f'error_maps_{image}.png', dpi=85); plt.close(fig)

fig, ax = plt.subplots(len(C.IMAGES), len(DECS), figsize=(3.0 * len(DECS), 3.0 * len(C.IMAGES)), constrained_layout=True)
for i, image in enumerate(C.IMAGES):
    for j, (nm, _) in enumerate(DECS):
        e = decs[image][j]
        im = ax[i, j].imshow(disp(e), cmap=DIV, vmin=-1, vmax=1)
        ax[i, j].set_title(f'{image}\n{nm}', fontsize=7); ax[i, j].set_xticks([]); ax[i, j].set_yticks([])
cbar(fig, im, ax.ravel().tolist(), True)
fig.suptitle('Error decomposition (converged images; signed luminance, same scale as the error maps)', fontsize=9)
fig.savefig(OUT / 'decomposition_maps.png', dpi=85); plt.close(fig)

# bar chart: object and highlight relative L1 vs the layered full-depth reference
arms = ['ECRO_4spp', 'ECRO_64spp', 'ECRO_converged_2x1024', 'LearnedER_PT_full_64spp', 'LearnedER_PT_full_converged', 'Layered_PT_6event_converged']
cols = ['#2a78d6', '#86b6ef', '#1c5cab', '#eb6834', '#c4471a', '#52514e']
fig, ax = plt.subplots(1, 2, figsize=(13, 4.2), constrained_layout=True)
for a_, key, ttl in zip(ax, ('rel_l1_obj', 'rel_l1_highlight'), ('objects (sphere + cube)', 'highlight ROI')):
    x = np.arange(len(C.IMAGES))
    for k, (arm, col) in enumerate(zip(arms, cols)):
        v = [metrics['images'][i]['arms_vs_Layered65'][arm][key] for i in C.IMAGES]
        a_.bar(x + (k - 2.5) * 0.13, v, 0.12, color=col, label=arm.replace('_', ' '), edgecolor='#fcfcfb', linewidth=1.5)
    a_.set_xticks(x); a_.set_xticklabels(C.IMAGES); a_.set_title(f'relative L1 vs layered full-depth reference: {ttl}', fontsize=9)
    a_.grid(axis='y', alpha=0.25)
ax[0].legend(fontsize=7, frameon=False)
fig.savefig(OUT / 'error_bars.png', dpi=90); plt.close(fig)

if 'training' in metrics:
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6), constrained_layout=True)
    for a_, key in zip(ax, ('whole_rel_l1', 'single_rel_l1', 'global_rel_l1')):
        for (tag, t), col in zip(metrics['training'].items(), ('#2a78d6', '#eb6834', '#1baf7a')):
            pts = [(r['step'], r['val'][key]) for r in t['log'] if 'val' in r]
            a_.plot(*zip(*pts), 'o-', color=col, lw=2, ms=5, label=tag)
        a_.set_title(f'validation {key}', fontsize=9); a_.set_xlabel('step'); a_.grid(alpha=0.25)
    ax[0].legend(fontsize=7, frameon=False)
    fig.savefig(OUT / 'training_validation.png', dpi=90); plt.close(fig)

files = sorted(p.name for p in OUT.glob('*.png'))
C.write_json(OUT / 'evaluation_manifest.json', {'id': 'WL104', 'display': SCALES, 'files': {f: C.sha(OUT / f) for f in files + ['metrics.json']}})
html = ['<!doctype html><meta charset="utf-8"><title>WL104 ECRO on the WL84 scene</title><body style="font-family:sans-serif;background:#fcfcfb">',
        '<h1>WL104 — ECRO trained and rendered on the WL84 scene (learned-ER LOW, D = 6)</h1>',
        f'<p>One error scale for every map: max {M:.4g} linear-HDR luminance, logarithmic display with floor M/1000. Numbers: metrics.json.</p>']
html += [f'<h3>{f}</h3><img src="{f}" style="max-width:100%">' for f in files]
(OUT / 'index.html').write_text('\n'.join(html), encoding='utf-8')
print('ANALYSIS_COMPLETE', json.dumps({i: metrics['images'][i]['arms_vs_Layered65']['ECRO_converged_2x1024']['rel_l1_obj'] for i in C.IMAGES}), flush=True)
os._exit(0)
