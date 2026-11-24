"""WL111 reviewer figures. python figs111.py EVIDENCE_DIR LR3_DIR SUMMARY_JSON OUT_DIR

Display transform for every image: LDR = clip(x * 1, 0, 1)^(1/2.2) (exposure 1; same as WL102 / TransGI).
Error maps: per-pixel relative luminance error (x - ref) / mean(ref luminance), shared diverging scale per figure.
Colors (dataviz reference palette, categorical slots in fixed order): ECRO blue, PT orange, PTDN aqua; single-point
external methods carry direct labels and distinct markers.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

EV, LR, SUMF, OUT = (Path(a) for a in sys.argv[1:5])
OUT.mkdir(parents=True, exist_ok=True)
S = json.loads(SUMF.read_text())
COL = {'ECRO': '#2a78d6', 'PT': '#eb6834', 'PTDN': '#1baf7a', 'TransGI': '#6250d6', 'ECRO_hist': '#2a78d6'}
NAME = {'ECRO': 'ReST-GI (ECRO, canonical)', 'PT': 'Path tracing (full depth)', 'PTDN': 'Path tracing + OptiX denoiser'}
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'
LUMA = np.array([0.2126, 0.7152, 0.0722])
plt.rcParams.update({'font.size': 9, 'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2, 'ytick.color': INK2,
                     'axes.spines.top': False, 'axes.spines.right': False, 'figure.facecolor': '#fcfcfb', 'axes.facecolor': '#fcfcfb'})
TITLE = {'S0_V0': 'W1 SSS cfg0 - V0', 'S0_H1': 'W1 SSS cfg0 - held-out H1', 'S0_H2': 'W1 SSS cfg0 - held-out H2',
         'S1_V0': 'W3 after rigid box edit (T1) - V0', 'SL_V0': 'W5 after light change - V0', 'N0_V0': 'W2 near-mirror panel - V0'}


def ldr(x):
    return np.clip(x, 0, 1) ** (1 / 2.2)


def frontier_ax(ax, key, show_legend):
    tab = S['fixtures'][key]
    for arm in ('PT', 'PTDN', 'ECRO'):
        ks = sorted([k for k in tab if k.split('_')[0] == arm and k.count('_') == 1 and tab[k]['ms'] is not None],
                    key=lambda k: int(k.split('_')[1]))
        if not ks:
            continue
        t = [tab[k]['ms'] for k in ks]
        e = [tab[k]['E']['E'] for k in ks]
        lo = [tab[k]['E']['E'] - tab[k]['E']['lo'] for k in ks]
        hi = [tab[k]['E']['hi'] - tab[k]['E']['E'] for k in ks]
        ax.errorbar(t, e, yerr=[lo, hi], color=COL[arm], lw=2, marker='o', ms=5, capsize=0, label=NAME[arm], zorder=3)
        for k, tt, ee in zip(ks, t, e):
            spp = k.split('_')[1]
            if spp in ('1', '4', '16', '64', '256'):
                ax.annotate(spp, (tt, ee), textcoords='offset points', xytext=(4, 3), fontsize=7, color=INK2)
    if 'ECRO_CONV_4096' in tab:
        ax.axhline(tab['ECRO_CONV_4096']['E']['E'], color=COL['ECRO'], lw=1, ls=':', zorder=2)
        ax.annotate('ECRO converged (bias floor)', (0.99, tab['ECRO_CONV_4096']['E']['E']), xycoords=('axes fraction', 'data'),
                    ha='right', va='bottom', fontsize=7, color=INK2)
    for b, lab in ((10, 'ideal 10 ms'), (15, 'desired 15'), (20, 'acceptable 20')):
        ax.axvline(b, color=GRID, lw=1, zorder=1)
        ax.annotate(lab, (b, 1), xycoords=('data', 'axes fraction'), rotation=90, va='top', ha='right', fontsize=6.5, color=INK2)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(True, which='major', color=GRID, lw=0.6)
    ax.set_xlabel('Median frame time on RTX 5080, 512 x 512 (ms)')
    ax.set_ylabel('Relative HDR RMSE vs full-depth reference')
    ax.set_title(TITLE.get(key, key), fontsize=9.5, color=INK, loc='left')
    if show_legend:
        ax.legend(frameon=False, fontsize=7.5, loc='lower left')


keys = [k for k in ('S0_V0', 'S0_H1', 'S0_H2', 'S1_V0', 'SL_V0', 'N0_V0') if k in S['fixtures']]
fig, axs = plt.subplots(2, 3, figsize=(15, 8.6))
for i, (ax, k) in enumerate(zip(axs.ravel(), keys)):
    frontier_ax(ax, k, i == 0)
for ax in axs.ravel()[len(keys):]:
    ax.axis('off')
fig.suptitle('WL111 quality-performance frontiers, ECRO fixtures (lower-left is better; numbers = spp; error bars = 95 % bootstrap over replicates)',
             fontsize=10, color=INK, x=0.01, ha='left')
fig.tight_layout()
fig.savefig(OUT / 'frontier_fixtures.png', dpi=150)
plt.close(fig)
for k in keys:
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    frontier_ax(ax, k, True)
    fig.tight_layout()
    fig.savefig(OUT / f'frontier_{k}.png', dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- image grids with shared scale
def grid(key, panels, fname, crop=None):
    z = np.load(EV / 'quality' / f'{key}_images.npz')
    ref = z['ref'].astype(np.float64)
    mL = float((ref @ LUMA).mean())
    sel = [(lab, k) for lab, k in panels if k in z.files or k == 'ref']
    n = len(sel)
    fig, axs = plt.subplots(2, n, figsize=(2.6 * n, 5.6))
    lim = 0.5
    for j, (lab, k) in enumerate(sel):
        img = ref if k == 'ref' else z[k].astype(np.float64)
        if crop:
            (y0, y1), (x0, x1) = crop
            img_c, ref_c = img[y0:y1, x0:x1], ref[y0:y1, x0:x1]
        else:
            img_c, ref_c = img, ref
        axs[0, j].imshow(ldr(img_c), interpolation='nearest')
        axs[0, j].set_title(lab, fontsize=8, color=INK)
        err = ((img_c - ref_c) @ LUMA) / mL
        im = axs[1, j].imshow(err, cmap='RdBu_r', vmin=-lim, vmax=lim, interpolation='nearest')
        if k != 'ref':
            axs[1, j].set_title(f'signed rel. lum. error', fontsize=7, color=INK2)
        for a in axs[:, j]:
            a.set_xticks([])
            a.set_yticks([])
    cb = fig.colorbar(im, ax=axs[1, :].tolist(), fraction=0.02, pad=0.01)
    cb.set_label('(x - ref) / mean ref luminance', fontsize=7)
    fig.suptitle(f'{TITLE.get(key, key)} - LDR clip(x,0,1)^(1/2.2), exposure 1; bottom: shared scale +-{lim}', fontsize=9, x=0.01, ha='left')
    fig.savefig(OUT / fname, dpi=130, bbox_inches='tight')
    plt.close(fig)


panels = [('Reference (full depth)', 'ref'), ('D6 reference (diag.)', 'ref_d6'), ('ECRO 1 spp', 'ECRO_1'), ('ECRO 4 spp', 'ECRO_4'),
          ('ECRO 16 spp', 'ECRO_16'), ('ECRO converged', 'ECRO_CONV_4096'), ('PT 4 spp', 'PT_4'), ('PT 16 spp', 'PT_16'),
          ('PT+OptiX 1 spp', 'PTDN_1'), ('PT+OptiX 4 spp', 'PTDN_4')]
for k in keys:
    grid(k, panels, f'images_{k}.png')
# contact crops (SSS V0): rows 300-470, cols 150-370 hold the near floor / sphere contact in the 512^2 V0 frame
if 'S0_V0' in keys:
    grid('S0_V0', panels, 'contact_crop_S0_V0.png', crop=((330, 470), (170, 350)))

# ---------------------------------------------------------------- SSS radial profile
fig, axs = plt.subplots(1, 3, figsize=(14, 4.2), sharey=True)
for ax, k in zip(axs, ('S0_V0', 'S0_H1', 'S0_H2')):
    if k not in S['fixtures']:
        ax.axis('off')
        continue
    tab = S['fixtures'][k]
    for lab, key, col, ls in (('ECRO 16 spp', 'ECRO_16', COL['ECRO'], '-'), ('ECRO converged', 'ECRO_CONV_4096', COL['ECRO'], ':'),
                              ('PT 64 spp', 'PT_64', COL['PT'], '-'), ('PT+OptiX 4 spp', 'PTDN_4', COL['PTDN'], '-'),
                              ('PT+OptiX 16 spp', 'PTDN_16', COL['PTDN'], '--')):
        if key not in tab or 'radial' not in tab[key]:
            continue
        rings = tab[key]['radial']
        names = sorted(rings, key=lambda s: float(s.split('_')[1]))
        x = [0.5 * (float(s.split('_')[1]) + float(s.split('_')[2])) for s in names]
        y = [rings[s]['mean'] for s in names]
        lo = [rings[s]['mean'] - rings[s]['lo'] if rings[s]['lo'] == rings[s]['lo'] else 0 for s in names]
        hi = [rings[s]['hi'] - rings[s]['mean'] if rings[s]['hi'] == rings[s]['hi'] else 0 for s in names]
        ax.errorbar(x, y, yerr=[lo, hi], color=col, ls=ls, lw=2, marker='o', ms=4, label=lab)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.axvspan(0, 0.18, color=GRID, alpha=0.6, lw=0)
    ax.annotate('near ROI (< 0.18 m)', (0.005, 0.97), xycoords=('data', 'axes fraction'), fontsize=7, color=INK2, va='top')
    ax.set_xlabel('Floor distance from the sphere support point (m)')
    ax.set_title(TITLE[k], fontsize=9, loc='left', color=INK)
    ax.grid(True, color=GRID, lw=0.6)
axs[0].set_ylabel('Signed relative luminance error vs full-depth ref')
axs[0].legend(frameon=False, fontsize=7.5)
fig.suptitle('SSS near-contact radial profile (mean over replicates, 95 % CI); V0 uses the front floor, H1/H2 all azimuths', fontsize=9.5, x=0.01, ha='left')
fig.tight_layout()
fig.savefig(OUT / 'sss_radial_profile.png', dpi=150)
plt.close(fig)

# ---------------------------------------------------------------- dynamic latency
if S.get('dynamic'):
    names = list(S['dynamic'])
    arms = ['ECRO_sel', 'ECRO_full', 'PT', 'PTDN']
    lab = {'ECRO_sel': 'ECRO selective update', 'ECRO_full': 'ECRO full relation rebuild', 'PT': 'PT (BVH update + 1 spp)',
           'PTDN': 'PT + OptiX'}
    col = {'ECRO_sel': COL['ECRO'], 'ECRO_full': '#9dc3f0', 'PT': COL['PT'], 'PTDN': COL['PTDN']}
    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    w = 0.18
    for i, a in enumerate(arms):
        med = [S['dynamic'][n]['edit_to_frame'][a]['median'] for n in names]
        p95 = [S['dynamic'][n]['edit_to_frame'][a]['p95'] for n in names]
        xs = np.arange(len(names)) + (i - 1.5) * w
        ax.bar(xs, med, width=w - 0.02, color=col[a], label=lab[a])
        ax.scatter(xs, p95, color=INK, s=10, marker='_', zorder=3)
        for x_, m in zip(xs, med):
            ax.annotate(f'{m:.1f}', (x_, m), ha='center', va='bottom', fontsize=7, color=INK)
    ax.set_xticks(np.arange(len(names)))
    ax.set_xticklabels(['Rigid box edit (S0 -> S1)' if 'EDIT' in n else 'Light change (S0 -> SL)' for n in names])
    ax.set_ylabel('Edit -> first 512^2 x 1 spp frame (ms)')
    ax.legend(frameon=False, fontsize=7.5, ncol=2)
    ax.grid(True, axis='y', color=GRID, lw=0.6)
    ax.set_title('Edit-to-first-frame latency on RTX 5080 (bars = median, ticks = p95)', fontsize=9.5, loc='left')
    fig.tight_layout()
    fig.savefig(OUT / 'dynamic_edit_latency.png', dpi=150)
    plt.close(fig)

# ---------------------------------------------------------------- LR3 frontier
L3 = S.get('lr3', {})
if any(k.startswith('PT_LR3') for k in L3):
    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    for arm, c, lab in (('PT_LR3', COL['PT'], 'Path tracing (Mitsuba path, depth 65)'), ('PTDN_LR3', COL['PTDN'], 'Path tracing + OptiX')):
        ks = sorted([k for k in L3 if k.startswith(arm + '_') and 'ms' in L3[k]], key=lambda k: int(k.rsplit('_', 1)[1]))
        if ks:
            ax.plot([L3[k]['ms'] for k in ks], [L3[k]['ldr_rmse']['mean'] for k in ks], color=c, lw=2, marker='o', ms=5, label=lab)
            for k in ks:
                ax.annotate(k.rsplit('_', 1)[1], (L3[k]['ms'], L3[k]['ldr_rmse']['mean']), textcoords='offset points', xytext=(4, 3), fontsize=7, color=INK2)
    for tag, d in L3.get('_transgi_scored', {}).items():
        ms = None
        for bk, b in L3.get('_transgi_bench', {}).items():
            if '640x360' in bk:
                ms = b['frame_ms']['median']
        if ms is None:
            continue
        mk = {'wl111_L40_adopted': ('D', 'TransGI L40, adopted eval options*'), 'wl111_L40_fastmatched': ('s', 'TransGI L40, fast-frame settings'),
              'wl111_wl102_repro': ('^', 'TransGI WL102 model (repro check)')}.get(tag, ('x', tag))
        ax.scatter([ms], [d['vs_ref65']['ldr_rmse']], marker=mk[0], s=50, color=COL['TransGI'], zorder=4)
        ax.annotate(mk[1], (ms, d['vs_ref65']['ldr_rmse']), textcoords='offset points', xytext=(6, -3), fontsize=7, color=INK)
    ax.scatter([27.3], [0.170], marker='v', s=50, color=COL['ECRO'], zorder=4)
    ax.annotate('ECRO 4 spp (WL102, RTX 3090, historical)', (27.3, 0.170), textcoords='offset points', xytext=(6, -3), fontsize=7, color=INK)
    ax.axhline(0.0963, color=COL['ECRO'], ls=':', lw=1)
    ax.annotate('6-event truncation floor (Ref-7 vs Ref-65)', (0.99, 0.0963), xycoords=('axes fraction', 'data'), ha='right', va='bottom', fontsize=7, color=INK2)
    for b in (10, 15, 20):
        ax.axvline(b, color=GRID, lw=1)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_xlabel('Median frame time, 640 x 360 (ms); RTX 5080 unless labelled')
    ax.set_ylabel('GI LDR RMSE vs Ref-65')
    ax.set_title('W4 Living Room 3 static V0 - GI frontier (* timing measured on the fast frame)', fontsize=9.5, loc='left')
    ax.legend(frameon=False, fontsize=7.5, loc='upper right')
    fig.tight_layout()
    fig.savefig(OUT / 'frontier_LR3.png', dpi=150)
    plt.close(fig)
    if (LR / 'lr3_pt_images.npz').exists():
        z = np.load(LR / 'lr3_pt_images.npz')
        ref = z['ref65_gi']
        tg = {}
        for tag in ('wl111_L40_adopted', 'wl111_L40_fastmatched'):
            f = LR / f'transgi_{tag}.npz'
            if f.exists():
                zz = np.load(f)
                tg[tag] = zz['gi'] if 'gi' in zz.files else None
        pan = [('Ref-65 GI', ref), ('Ref-7 GI (6 events)', z['ref7_gi']), ('PT 4 spp GI', z.get('PT_4')), ('PT 64 spp GI', z.get('PT_64')),
               ('PT+OptiX 4 spp GI', z.get('PTDN_4'))] + [(f'TransGI {t[6:]}', v) for t, v in tg.items() if v is not None]
        pan = [p for p in pan if p[1] is not None]
        fig, axs = plt.subplots(2, len(pan), figsize=(3.2 * len(pan), 4.2))
        mL = float((ref @ LUMA).mean())
        for j, (lab, img) in enumerate(pan):
            axs[0, j].imshow(ldr(img))
            axs[0, j].set_title(lab, fontsize=8)
            im = axs[1, j].imshow(((img - ref) @ LUMA) / mL, cmap='RdBu_r', vmin=-1, vmax=1)
            for a in axs[:, j]:
                a.set_xticks([])
                a.set_yticks([])
        fig.colorbar(im, ax=axs[1, :].tolist(), fraction=0.02, pad=0.01).set_label('(x - ref) / mean ref GI luminance', fontsize=7)
        fig.suptitle('W4 LR3 GI images, LDR clip(x,0,1)^(1/2.2); bottom: shared signed error scale +-1', fontsize=9, x=0.01, ha='left')
        fig.savefig(OUT / 'images_LR3.png', dpi=130, bbox_inches='tight')
        plt.close(fig)
print('FIGURES DONE', sorted(p.name for p in OUT.glob('*.png')))
