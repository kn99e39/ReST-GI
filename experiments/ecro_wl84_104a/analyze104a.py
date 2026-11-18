"""WL104A analysis: A2 gate, P1 / S1 / D1 / D2 / C1 / C2 comparisons, verdict, figures, reviewer gallery.
  <venv-cu128 python> analyze104a.py <raw_root> <reviewer_dir>
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

import core104a as K
import stats104a as S

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'ecro_wl84_104'))
import common104 as C  # noqa: E402

ROOT = Path(sys.argv[1]).resolve(); OUT = Path(sys.argv[2]).resolve(); OUT.mkdir(parents=True, exist_ok=True)
E104 = C.ROOT / 'evidence'
LUMA = K.LUMA
lum = lambda x: x @ LUMA
status = json.loads((ROOT / 'status.json').read_text(encoding='utf-8'))
mz = np.load(ROOT / 'masks.npz')
MASKS = {im: {k: mz[f'{im}|{k}'] for k in K.ROIS} for im in C.IMAGES}
KEYS = ['ecro_e1', 'ecro_e2', 'ecro_hl', 'orcA_hl', 'orcB_hl', 'er_e1', 'er_e2', 'er_h', 'lay_e1', 'lay_e2', 'lay_h']


def rel_l1(a, b, m):
    return float(np.abs(a - b)[m].sum() / max(np.abs(b)[m].sum(), 1e-30))


res = {'images': {}, 'status': status}
VIS = {}
gate_a2 = True
for i, image in enumerate(C.IMAGES):
    z = K.load(ROOT / 'rep' / image, KEYS)
    R = z['er_h'].shape[0]
    er_tot = z['er_e1'] + z['er_e2'] + z['er_h']; lay_tot = z['lay_e1'] + z['lay_e2'] + z['lay_h']
    ec_tot = z['ecro_e1'] + z['ecro_e2'] + z['ecro_hl']
    M = MASKS[image]
    out = {'R': R}
    # A2: tracer totals vs WL104 Mitsuba references (6 events)
    a2 = {}
    for name, tr, ref in (('learnedER', er_tot, 'ER7'), ('layered', lay_tot, 'LAY7')):
        h = [np.load(E104 / 'refs' / image / f'{ref}_h{q}.npz')['mean'].astype(np.float64) for q in (0, 1)]
        rm = 0.5 * (h[0] + h[1]); tm = tr.mean(0)
        t1, t2 = tr[:R // 2].mean(0), tr[R // 2:].mean(0)
        o = M['objects']
        lim = 1.5 * 0.5 * float(np.hypot(rel_l1(t1, t2, o), rel_l1(h[0], h[1], o)))
        g = {'obj_mean_ratio': float(lum(tm)[o].mean() / lum(rm)[o].mean()), 'full_mean_ratio': float(lum(tm).mean() / lum(rm).mean()),
             'obj_rel_l1': rel_l1(tm, rm, o), 'limit': lim}
        g['pass'] = bool(abs(g['obj_mean_ratio'] - 1) <= 0.01 and abs(g['full_mean_ratio'] - 1) <= 0.01 and g['obj_rel_l1'] <= lim)
        a2[name] = g; gate_a2 &= g['pass']
    out['A2'] = a2
    comps = {}
    for roi in K.ROIS:
        m = M[roi]
        c = {'P1_ECRO_vs_LearnedER': K.compare(z['ecro_hl'], z['er_h'], er_tot, z['er_h'], m),
             'D1_Field_vs_LearnedER': K.compare(z['orcA_hl'], z['er_h'], er_tot, z['er_h'], m, XB=z['orcB_hl']),
             'D2_ECRO_vs_Field': None,
             'S1_LearnedER_vs_Layered_HIGH': K.compare(z['er_h'], z['lay_h'], lay_tot, z['lay_h'], m),
             'S1_LearnedER_vs_Layered_TOTAL': K.compare(er_tot, lay_tot, lay_tot, lay_tot, m),
             'C1_LOW_ECRO_vs_LearnedER': K.compare(z['ecro_e1'] + z['ecro_e2'], z['er_e1'] + z['er_e2'], er_tot, z['er_e1'] + z['er_e2'], m),
             'ECRO_TOTAL_vs_Layered_TOTAL': K.compare(ec_tot, lay_tot, lay_tot, lay_tot, m)}
        # D2: ECRO minus field readout; A / B tables give independent table noise (A x B stable mean square)
        fa, fb = z['orcA_hl'], z['orcB_hl']
        fm = 0.5 * (fa + fb)
        Rr = R
        num = (lambda a: (a[:, m] @ LUMA).sum(1))
        eT = S.ratio_jk(num(z['ecro_hl']) - num(fm), num(er_tot))
        dA = (num(z['ecro_hl']) - num(fa)).sum() / num(er_tot).sum(); dB = (num(z['ecro_hl']) - num(fb)).sum() / num(er_tot).sum()
        se = float(np.hypot(eT['se'], abs(dA - dB) / 2)); hh = S.t975(Rr - 1) * se
        eT.update(se=se, lo=eT['point'] - hh, hi=eT['point'] + hh, table_noise_se=abs(dA - dB) / 2)
        eH = S.ratio_jk(num(z['ecro_hl']) - num(fm), num(z['er_h']))
        ms = S.stable_ms_jk((z['ecro_hl'] - fa)[:, m].reshape(Rr, -1), (z['ecro_hl'] - fb)[:, m].reshape(Rr, -1))
        rms = S.rel_rms(ms, float((er_tot.mean(0)[m] ** 2).mean()))
        Dm = (z['ecro_hl'] - fm).mean(0)[m]
        d2 = {'energy_bias_vs_total': eT, 'energy_bias_vs_high': eH, 'rel_rms_vs_total': rms,
              'descriptive': {'rel_l1_vs_ref_high': float(np.abs(Dm).sum() / z['er_h'].mean(0)[m].sum())}}
        c['D2_ECRO_vs_Field'] = d2
        for k, v in c.items():
            v['significance'] = K.significance(v)
        comps[roi] = c
    out['comparisons'] = comps
    # C2 truncation from WL104 halves: full depth minus 6 events, relative to TOTAL (two half pairings)
    tr = {}
    for a, b in (('ER65', 'ER7'), ('LAY65', 'LAY7')):
        h = {x: [np.load(E104 / 'refs' / image / f'{x}_h{q}.npz')['mean'].astype(np.float64) for q in (0, 1)] for x in (a, b)}
        tr[f'{a}_minus_{b}'] = {roi: [float((lum(h[a][q]) - lum(h[b][1 - q]))[M[roi]].sum() / lum(h[b][1 - q])[M[roi]].sum()) for q in (0, 1)] for roi in K.ROIS}
    out['C2_truncation_rel_total'] = tr
    # table level: learned table vs field table (mean of A and B), energy-weighted
    Bl = np.load(E104 / 'eval' / image / 'table.npz')['B'].astype(np.float64)
    Fa = np.load(ROOT / 'field' / f'{image}_A.npz')['F'].astype(np.float64); Fb = np.load(ROOT / 'field' / f'{image}_B.npz')['F'].astype(np.float64)
    Fm = 0.5 * (Fa + Fb)
    out['table_level'] = {'rel_l1_learned_vs_field': float(np.abs(Bl - Fm).sum() / Fm.sum()), 'rel_l1_fieldA_vs_fieldB_half': float(0.5 * np.abs(Fa - Fb).sum() / Fm.sum()),
                          'energy_ratio_learned_over_field': float(Bl.sum() / Fm.sum())}
    res['images'][image] = out
    VIS[image] = {'lay_tot': lay_tot.mean(0), 'er_tot': er_tot.mean(0), 'ec_tot': ec_tot.mean(0), 'lay_h': z['lay_h'].mean(0), 'er_h': z['er_h'].mean(0),
                  'ec_h': z['ecro_hl'].mean(0), 'f_h': 0.5 * (z['orcA_hl'] + z['orcB_hl']).mean(0),
                  'noise_P1': 0.5 * ((z['ecro_hl'] - z['er_h'])[:R // 2].mean(0) - (z['ecro_hl'] - z['er_h'])[R // 2:].mean(0))}
    del z
    print('analysed', image, flush=True)

# ------------------------------------------------------------------------------------------------ verdict
units = [(im, roi) for im in C.IMAGES for roi in K.PRIMARY]
sig = {f'{im}|{roi}': res['images'][im]['comparisons'][roi]['P1_ECRO_vs_LearnedER']['significance'] for im, roi in units}
A_ok = status.get('A0', {}).get('pass') and status.get('A1', {}).get('pass') and gate_a2
if not A_ok:
    verdict = 'INCONCLUSIVE (gate failure)'
elif any(s['energy_sig'] or s['rms_sig'] for s in sig.values()):
    verdict = 'NOT SUPPORTED'
elif all(s['energy_below'] and s['rms_below'] for s in sig.values()):
    verdict = 'SUPPORTED'
else:
    verdict = 'INCONCLUSIVE'
diag = {}
if verdict == 'NOT SUPPORTED':
    for k, s in sig.items():
        if s['energy_sig'] or s['rms_sig']:
            im, roi = k.split('|')
            c = res['images'][im]['comparisons'][roi]
            diag[k] = {'failing': {'energy': s['energy_sig'], 'rms': s['rms_sig']},
                       'D1_projection_significant': {'energy': c['D1_Field_vs_LearnedER']['significance']['energy_sig'], 'rms': c['D1_Field_vs_LearnedER']['significance']['rms_sig']},
                       'D2_learned_approx_significant': {'energy': c['D2_ECRO_vs_Field']['significance']['energy_sig'], 'rms': c['D2_ECRO_vs_Field']['significance']['rms_sig']}}
prec = {k: {'energy_halfwidth_ok': s['energy_halfwidth'] <= K.TAU_E / 4, 'rms_halfwidth_ok': s['rms_halfwidth'] <= K.TAU_S / 4} for k, s in sig.items()}
res.update(verdict=verdict, gates_ok=bool(A_ok), A2_pass=bool(gate_a2), P1_significance=sig, precision=prec, diagnosis=diag)


def clean(o):
    if isinstance(o, dict):
        return {str(a): clean(b) for a, b in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return o


C.write_json(OUT / 'metrics.json', clean(res))

# ------------------------------------------------------------------------------------------------ figures
import matplotlib  # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

plt.rcParams.update({'font.size': 8, 'figure.facecolor': '#fcfcfb', 'axes.facecolor': '#fcfcfb', 'axes.edgecolor': '#52514e'})
SEQ = LinearSegmentedColormap.from_list('seq', ['#fcfcfb', '#cde2fb', '#86b6ef', '#3987e5', '#1c5cab', '#0d366b'])
DIV = LinearSegmentedColormap.from_list('div', ['#1c5cab', '#86b6ef', '#f0efec', '#f3a07f', '#c4471a'])
eT = 1.0 / float(np.percentile(np.concatenate([lum(VIS[i]['lay_tot']).ravel() for i in C.IMAGES]), 99.5))
eH = 1.0 / float(np.percentile(np.concatenate([lum(VIS[i]['er_h']).ravel() for i in C.IMAGES]), 99.5))
tone = lambda x, e: np.clip((np.clip(e * x, 0, None) / (1 + np.clip(e * x, 0, None))) ** (1 / 2.2), 0, 1)
DIFFS = [('ECRO HIGH - learned-ER HIGH (P1)', lambda V: V['ec_h'] - V['er_h']), ('field HIGH - learned-ER HIGH (D1)', lambda V: V['f_h'] - V['er_h']),
         ('ECRO HIGH - field HIGH (D2)', lambda V: V['ec_h'] - V['f_h']), ('learned-ER HIGH - layered HIGH (S1)', lambda V: V['er_h'] - V['lay_h']),
         ('P1 noise: half-split / 2', lambda V: V['noise_P1'])]
dmaps = {im: [lum(f(VIS[im])) for _, f in DIFFS] for im in C.IMAGES}
MX = float(max(np.abs(d).max() for v in dmaps.values() for d in v))
M0 = MX / 1000.0
disp = lambda x: np.sign(x) * np.log1p(np.abs(x) / M0) / np.log1p(MX / M0)


def cbar(fig, im, axes, signed):
    vals = MX * np.array([0, 0.001, 0.01, 0.1, 1.0])
    vals = np.concatenate([-vals[::-1][:-1], vals]) if signed else vals
    cb = fig.colorbar(im, ax=axes, shrink=0.8, ticks=disp(vals)); cb.ax.set_yticklabels([f'{v:.3g}' for v in vals])
    cb.set_label(('signed' if signed else 'absolute') + ' HIGH difference, linear HDR luminance', fontsize=7)


for image in C.IMAGES:
    V = VIS[image]
    fig, ax = plt.subplots(4, 5, figsize=(16, 13.5), constrained_layout=True)
    for j, (k, nm) in enumerate((('lay_tot', 'layered TOTAL'), ('er_tot', 'learned-ER TOTAL'), ('ec_tot', 'ECRO TOTAL'))):
        ax[0, j].imshow(tone(V[k], eT)); ax[0, j].set_title(nm + ' (replicate mean)', fontsize=8)
    ax[0, 3].axis('off'); ax[0, 4].axis('off')
    for j, (k, nm) in enumerate((('lay_h', 'layered HIGH E3-6'), ('er_h', 'learned-ER HIGH E3-6'), ('ec_h', 'ECRO HIGH (learned table)'), ('f_h', 'field-table HIGH (A+B)/2'))):
        ax[1, j].imshow(tone(V[k], eH)); ax[1, j].set_title(nm, fontsize=8)
    ax[1, 4].axis('off')
    for j, (nm, _) in enumerate(DIFFS):
        d = dmaps[image][j]
        im2 = ax[2, j].imshow(disp(d), cmap=DIV, vmin=-1, vmax=1); ax[2, j].set_title(nm, fontsize=7)
        im3 = ax[3, j].imshow(disp(np.abs(d)), cmap=SEQ, vmin=0, vmax=1); ax[3, j].set_title('|' + nm.split(' (')[0] + '|', fontsize=7)
    for a in ax.ravel():
        a.set_xticks([]); a.set_yticks([])
    cbar(fig, im2, ax[2].tolist(), True); cbar(fig, im3, ax[3].tolist(), False)
    fig.suptitle(f'WL104A {image}: images (row 1, TOTAL exposure), HIGH components (row 2, HIGH exposure), signed / absolute HIGH differences (rows 3-4, one scale for WL104A)', fontsize=9)
    fig.savefig(OUT / f'high_maps_{image}.png', dpi=80); plt.close(fig)

# crops: sphere_lower and floor_near of every image, P1 and S1 signed maps
fig, ax = plt.subplots(3, 4, figsize=(14, 10), constrained_layout=True)
for i, image in enumerate(C.IMAGES):
    ys, xs = np.nonzero(MASKS[image]['sphere_lower'] | MASKS[image]['floor_near'])
    y0, y1, x0, x1 = max(ys.min() - 8, 0), min(ys.max() + 8, 255), max(xs.min() - 8, 0), min(xs.max() + 8, 255)
    V = VIS[image]
    ax[i, 0].imshow(tone(V['er_tot'], eT)[y0:y1, x0:x1]); ax[i, 0].set_title(f'{image} learned-ER TOTAL', fontsize=7)
    for j, (nm, f) in enumerate([DIFFS[0], DIFFS[3], DIFFS[4]], 1):
        im = ax[i, j].imshow(disp(lum(f(V)))[y0:y1, x0:x1], cmap=DIV, vmin=-1, vmax=1); ax[i, j].set_title(nm, fontsize=7)
        b = MASKS[image]['sphere_lower'][y0:y1, x0:x1] | MASKS[image]['floor_near'][y0:y1, x0:x1]
        ax[i, j].contour(b, levels=[0.5], colors='#52514e', linewidths=0.5)
for a in ax.ravel():
    a.set_xticks([]); a.set_yticks([])
cbar(fig, im, ax[:, 1:].ravel().tolist(), True)
fig.suptitle('Interreflection crops (sphere_lower + floor_near outlined): signed HIGH differences, same scale as the HIGH maps', fontsize=9)
fig.savefig(OUT / 'interreflection_crops.png', dpi=85); plt.close(fig)

# interval charts: energy bias (vs TOTAL) and relative RMS, primary ROIs, per comparison (small multiples, one axis each)
CMPS = [('P1_ECRO_vs_LearnedER', '#2a78d6'), ('D1_Field_vs_LearnedER', '#1baf7a'), ('D2_ECRO_vs_Field', '#eda100'), ('S1_LearnedER_vs_Layered_HIGH', '#eb6834')]
fig, ax = plt.subplots(2, 3, figsize=(15, 7.5), constrained_layout=True)
for col, roi in enumerate(K.PRIMARY):
    for row, key, tau in ((0, 'energy_bias_vs_total', K.TAU_E), (1, 'rel_rms_vs_total', K.TAU_S)):
        a = ax[row, col]
        for k, (cmpn, colr) in enumerate(CMPS):
            for i, image in enumerate(C.IMAGES):
                c = res['images'][image]['comparisons'][roi][cmpn][key]
                x = i + (k - 1.5) * 0.18
                a.plot([x, x], [c['lo'], c['hi']], color=colr, lw=2)
                a.plot(x, c['point'], 'o', color=colr, ms=6, label=cmpn if i == 0 and col == 0 and row == 0 else None)
        for t in ((-tau, tau) if row == 0 else (tau,)):
            a.axhline(t, color='#52514e', ls='--', lw=1)
        a.axhline(0, color='#52514e', lw=0.6)
        a.set_xticks(range(3)); a.set_xticklabels([im.replace('test83_', '') for im in C.IMAGES])
        a.set_title(f'{roi}: ' + ('signed HIGH energy bias / ROI TOTAL' if row == 0 else 'stable relative RMS of HIGH difference'), fontsize=8)
        a.grid(axis='y', alpha=0.25)
ax[0, 0].legend(fontsize=7, frameon=False)
fig.suptitle(f'WL104A comparisons with 95 % intervals; dashed = thresholds (tau_E {K.TAU_E}, tau_S {K.TAU_S}). Verdict: {verdict}', fontsize=9)
fig.savefig(OUT / 'intervals.png', dpi=90); plt.close(fig)

files = sorted(p.name for p in OUT.glob('*.png'))
C.write_json(OUT / 'evaluation_manifest.json', {'id': 'WL104A', 'display': {'difference_scale_max': MX, 'm0': M0, 'mapping': 'sign(x) log(1+|x|/m0)/log(1+M/m0)',
                                                                         'tone_exposure_total': eT, 'tone_exposure_high': eH},
                                                'files': {f: C.sha(OUT / f) for f in files + ['metrics.json']}, 'raw_status_sha256': C.sha(ROOT / 'status.json')})
html = ['<!doctype html><meta charset="utf-8"><title>WL104A ECRO HIGH fidelity</title><body style="font-family:sans-serif;background:#fcfcfb">',
        f'<h1>WL104A — ECRO HIGH vs its learned-ER source (verdict: {verdict})</h1><p>Numbers: metrics.json. One difference scale (max {MX:.4g}), logarithmic display.</p>']
html += [f'<h3>{f}</h3><img src="{f}" style="max-width:100%">' for f in files]
(OUT / 'index.html').write_text('\n'.join(html), encoding='utf-8')
print('ANALYSIS_COMPLETE', verdict, flush=True)
os._exit(0)
