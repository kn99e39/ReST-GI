"""WL114A reviewer figures (reads analysis outputs only): LR3 GI comparisons and shared-scale absolute error maps vs Ref-65,
region scores, manifest with hashes.

  python fig114a.py ROOT OUTDIR
"""
import hashlib, json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

NAMES = {'Ref7': 'physical reference, 6 events (Ref-7)', 'arm_A': 'learned ECRO, depth-6 targets', 'arm_B': 'learned ECRO, full-depth targets',
         'oracle_D6': 'physical oracle table, depth 6', 'oracle_F': 'physical oracle table, full depth', 'phys_deep96_local': 'local physical, 96 events',
         'arm_A_4spp': 'learned depth-6, 4 spp frame', 'arm_B_4spp': 'learned full-depth, 4 spp frame'}


def ldr(x):
    return np.clip(x, 0, 1) ** (1 / 2.2)


def main(root, outdir):
    root = Path(root); out = Path(outdir); out.mkdir(parents=True, exist_ok=True); files = []
    m = np.load(root / 'analysis' / 'lr3_maps.npz'); res = json.loads((root / 'analysis' / 'lr3_metrics.json').read_text())

    def save(fig, name):
        fig.savefig(out / name, dpi=110, bbox_inches='tight'); plt.close(fig); files.append(name)
    keys = ['Ref7', 'oracle_D6', 'oracle_F', 'arm_A', 'arm_B', 'phys_deep96_local']
    ref = m['ref65_gi']
    fig, ax = plt.subplots(2, 4, figsize=(22, 7))
    ax = ax.ravel(); ax[0].imshow(ldr(ref)); ax[0].set_title('Living Room 3 GI: full-depth reference Ref-65', fontsize=9); ax[0].axis('off')
    for i, k in enumerate(keys):
        ax[i + 1].imshow(ldr(m[k])); s = res['vs_Ref65'][k]['full_frame']
        ax[i + 1].set_title(f"{NAMES[k]}\nGI LDR RMSE {s['ldr_rmse']:.4f}, HDR bias {100 * s['bias_rel']:+.1f} %", fontsize=8); ax[i + 1].axis('off')
    ax[-1].axis('off')
    save(fig, 'living_room_3_gi_learned_depth6_vs_full_depth_vs_ref65.png')
    errs = {k: np.abs(ldr(m[k]) - ldr(ref)).mean(-1) for k in keys + ['arm_A_4spp', 'arm_B_4spp']}
    vmax = max(float(e.max()) for e in errs.values())
    fig, ax = plt.subplots(2, 4, figsize=(22, 7)); ax = ax.ravel()
    for i, (k, e) in enumerate(errs.items()):
        im = ax[i].imshow(np.sqrt(e / vmax), cmap='magma', vmin=0, vmax=1); ax[i].set_title(f'|LDR error| vs Ref-65: {NAMES[k]}', fontsize=8); ax[i].axis('off')
    cb = fig.colorbar(im, ax=ax.tolist(), fraction=0.015); t = np.array([0, .25, .5, .75, 1]); cb.set_ticks(t); cb.set_ticklabels([f'{x * x * vmax:.3f}' for x in t])
    cb.set_label('absolute LDR GI error (shared scale, sqrt display)')
    save(fig, 'living_room_3_gi_absolute_error_maps_vs_ref65_shared_scale.png')
    regs = [r for r in res['vs_Ref65']['arm_A'] if r != 'full_frame']
    fig, ax = plt.subplots(figsize=(10, 4)); w = 0.18
    for j, k in enumerate(['Ref7', 'arm_A', 'arm_B', 'oracle_F']):
        ax.bar(np.arange(len(regs) + 1) + j * w, [res['vs_Ref65'][k]['full_frame']['ldr_rmse']] + [res['vs_Ref65'][k][r]['ldr_rmse'] for r in regs], w, label=NAMES[k])
    ax.set_xticks(np.arange(len(regs) + 1) + 1.5 * w); ax.set_xticklabels(['full frame'] + regs); ax.set_ylabel('GI LDR RMSE vs Ref-65'); ax.legend(fontsize=7)
    ax.set_title('Living Room 3: GI error by material region (TransGI 4 spp context: 0.0775 full frame, WL102)', fontsize=9)
    save(fig, 'living_room_3_gi_error_by_region.png')
    json.dump({'files': {f: hashlib.sha256((out / f).read_bytes()).hexdigest() for f in files}, 'error_scale_ldr': vmax, 'source': str(root)},
              open(out / 'manifest.json', 'w'), indent=1)
    print('FIGURES', len(files))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
