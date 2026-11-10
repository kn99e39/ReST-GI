"""WL84 reviewer visuals: old (WL83, 4096 spp) vs new (WL84, 65536 spp) layered reference, the stored WL83 candidate renders and their
absolute error maps against the new reference (one shared sqrt scale), time-to-quality curves and replicate highlight-error distributions.

  python viz84.py <out>
"""
import json, math, os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
P84 = json.load(open(os.path.join(HERE, 'protocol', 'compiled84.json')))
if os.environ.get('W84_REF_DIR'):
    P84['paths']['ref_dir'] = os.environ['W84_REF_DIR']
if os.environ.get('W84_DEV'):                       # dev probe only: reduced budgets
    P84['reference']['chunks_per_half'] = [256, 256, 512]
    P84['replicates']['n'] = 2
    P84['economics']['reps'] = 1
    P84['economics']['ladder_ref'] = [64, 256]
    P84['economics']['ladder_closure'] = [16, 64]
W83 = os.path.expanduser('~/rest_compiled83/results/render')
IMAGES = P84['images']
SHOW = ['ER_L', 'E_L', 'ER_DF', 'C0_DF']
COL = {'REF': '#0b0b0b', 'ER_L': '#1baf7a', 'E_L': '#eb6834', 'C0_DF': '#2a78d6'}
tm = lambda x: np.clip(x / (1 + x), 0, 1) ** (1 / 2.2)


def main():
    out = sys.argv[1]
    vz = os.path.join(out, 'viz'); os.makedirs(vz, exist_ok=True)
    RD = P84['paths']['ref_dir']
    ev = json.load(open(os.path.join(out, 'eval84.json')))['rows']
    ec = json.load(open(os.path.join(out, 'econ', 'econ84.json')))
    files = []
    imgs = {}
    emax = 0.0
    for im in IMAGES:
        new = np.load(os.path.join(RD, f'{im}_ref84.npy')); old = np.load(os.path.join(W83, f'{im}_refhq.npy'))
        cand = {a: np.load(os.path.join(W83, f'{im}_{a}64.npy')) for a in SHOW}
        imgs[im] = (new, old, cand)
        emax = max([emax, float(np.abs(old - new).mean(2).max())] + [float(np.abs(c - new).mean(2).max()) for c in cand.values()])
    ncol = 3 + 2 * len(SHOW)
    fig, axs = plt.subplots(len(IMAGES), ncol, figsize=(2.0 * ncol, 2.3 * len(IMAGES)), squeeze=False)
    for r, im in enumerate(IMAGES):
        new, old, cand = imgs[im]
        axs[r, 0].imshow(tm(new)); axs[r, 1].imshow(tm(old))
        e = axs[r, 2].imshow(np.sqrt(np.abs(old - new).mean(2)), cmap='viridis', vmin=0, vmax=math.sqrt(emax))
        for c, a in enumerate(SHOW):
            axs[r, 3 + c].imshow(tm(cand[a]))
            axs[r, 3 + len(SHOW) + c].imshow(np.sqrt(np.abs(cand[a] - new).mean(2)), cmap='viridis', vmin=0, vmax=math.sqrt(emax))
            v = ev[im]['arms'][a]
            axs[r, 3 + c].set_title(f'{a} 64 spp' + (f"\n{ev[im]['ER_L_verdict'] if a == 'ER_L' else (ev[im]['E_L_verdict'] if a == 'E_L' else '')}" if a in ('ER_L', 'E_L') else ''), fontsize=7)
            axs[r, 3 + len(SHOW) + c].set_title(f'|{a} - REF84| err {v["err"]:.3f} hl {v["hl_err"]:.3f}', fontsize=6)
        axs[r, 0].set_title(f'REF84 65536 spp (unc {ev[im]["ref84_uncertainty"]:.4f})', fontsize=7)
        axs[r, 1].set_title(f'REF83 4096 spp (unc {ev[im]["ref83_uncertainty"]:.4f})', fontsize=7)
        axs[r, 2].set_title(f'|REF83 - REF84| ({ev[im]["old_vs_new_ref_rel_l1"]:.3f})', fontsize=7)
        axs[r, 0].text(-0.05, 0.5, im, transform=axs[r, 0].transAxes, rotation=90, va='center', ha='right', fontsize=7)
        for a_ in axs[r]:
            a_.axis('off')
    fig.colorbar(e, ax=axs[:, -1], fraction=0.02, label=f'sqrt(mean_c |diff|), shared scale, max {emax:.3g}')
    fig.suptitle('WL84: new independent layered reference vs WL83 reference and the stored WL83 64-spp candidate renders (tone map x/(1+x), gamma 2.2)', fontsize=9)
    fn = os.path.join(vz, 'references_and_candidates.png'); fig.savefig(fn, dpi=95); plt.close(fig); files.append(fn)
    # time-to-quality
    fig, axs = plt.subplots(1, len(IMAGES), figsize=(5 * len(IMAGES), 3.8), squeeze=False)
    for i, im in enumerate(IMAGES):
        ax = axs[0, i]
        for a, pts in ec['images'][im]['curves'].items():
            ax.loglog([p['gpu_ms'] for p in pts], [p['err'] for p in pts], marker='o', ms=4, color=COL.get(a, '#6250d6'), label=a)
            for p in pts:
                ax.annotate(str(p['spp']), (p['gpu_ms'], p['err']), fontsize=6, xytext=(2, 2), textcoords='offset points')
        ax.set_xlabel('GPU time per image (ms, uncontended, kernel history)'); ax.set_ylabel('object relative L1 vs REF84')
        ax.set_title(im, fontsize=8); ax.legend(fontsize=7, frameon=False)
    fig.suptitle('WL84 time-to-quality (labels: spp)', fontsize=9); fig.tight_layout()
    fn = os.path.join(vz, 'time_to_quality.png'); fig.savefig(fn, dpi=110); plt.close(fig); files.append(fn)
    # replicate highlight-error distributions
    fig, ax = plt.subplots(figsize=(8, 3.6))
    pos = 0; ticks = []
    for im in IMAGES:
        for a in ['ER_L', 'E_L']:
            rr = ev[im]['arms'][a].get('replicates')
            if rr:
                ax.scatter(np.full(len(rr['hl_err']), pos), rr['hl_err'], color=COL[a], s=14)
                ax.scatter([pos], [ev[im]['arms'][a]['hl_err']], marker='x', color='k', s=30)
                ticks.append((pos, f'{im}\n{a}')); pos += 1
    ax.axhline(0.10, color='k', ls=':', lw=1, label='highlight threshold 0.10')
    ax.set_xticks([p for p, _ in ticks]); ax.set_xticklabels([t for _, t in ticks], fontsize=6)
    ax.set_ylabel('highlight-ROI relative L1 vs REF84'); ax.legend(fontsize=7, frameon=False)
    ax.set_title('64-spp candidate replicates (dots, new seeds) and the stored WL83 render (x)', fontsize=8)
    fig.tight_layout(); fn = os.path.join(vz, 'replicate_highlight_error.png'); fig.savefig(fn, dpi=110); plt.close(fig); files.append(fn)
    json.dump({'files': files, 'error_scale_max': emax}, open(os.path.join(vz, 'viz_manifest.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
