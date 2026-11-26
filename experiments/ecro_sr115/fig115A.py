"""WL115 Stage A reviewer figures (SELECTION set).  python fig115A.py ROOT OUT_DIR
Absolute shared scales: one signed relative-error scale for every R1L window map; one HDR scale for every residual map;
sqrt display with a real-unit colour bar. Readable file names: <scene>_<view>_<what vs what>_<plot kind>.png"""
import json, math, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common115 as C  # noqa: E402

REPS = [('PC4', 'dense 4x4 piecewise-constant (U)'), ('BL4', 'bilinear from the same 4x4 values'), ('PC8', '8x8 piecewise-constant (U)'),
        ('BL8', 'bilinear from 8x8 values'), ('R32', '4x4 + 32x32 on contact neighbourhood (diagnostic)'), ('Q4', '4x4 camera-conditioned (diagnostic)'),
        ('Q8', '8x8 camera-conditioned (diagnostic)'), ('Q32R', '4x4 + 32x32 camera-conditioned contact (diagnostic)')]
NAMES = {'F0_V0': 'F0 contact, view V0', 'F1_W0': 'F1 contact, view W0', 'F2_W0': 'F2 1 cm gap, view W0', 'F3_W0': 'F3 contact, view W0',
         'SEQ_A_lift_3_V0': 'SEQ_A gap 50 mm, view V0', 'SEQ_C_independent_1_W0': 'SEQ_C gap 30 mm, view W0',
         'SEQ_C_independent_3_W0': 'SEQ_C gap 80 mm, view W0', 'SEQ_B_slide_2_V0': 'SEQ_B x 0.70 contact, view V0'}


def fname(tag):
    return NAMES.get(tag, tag).replace(',', '').replace(' ', '_')


def tonemap(x, s=1.0):
    return np.clip((x * s) ** (1 / 2.2), 0, 1)


def crop_box(r, mask, pad=40):
    ys, xs = np.nonzero(mask & (r < 0.35))
    if len(ys) == 0:
        return slice(0, 512), slice(0, 512)
    y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad, 512); x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad, 512)
    return slice(y0, y1), slice(x0, x1)


def main(root, out):
    root, out = Path(root), Path(out); out.mkdir(parents=True, exist_ok=True)
    M = json.loads((root / 'analysis' / 'stageA_metrics.json').read_text())
    S = json.loads((root / 'analysis' / 'stageA_summary.json').read_text())
    made = []
    # ---------------- 1. summary: failing R1L windows / innermost ring / lifted near bias per representation
    reps = [a for a, _ in REPS if a in S['reps']]
    fig, ax = plt.subplots(1, 4, figsize=(19, 4.2))
    ax[0].bar(reps, [S['reps'][a]['R1L_failing_total'] for a in reps], color='#4a6fa5'); ax[0].set_yscale('log'); ax[0].set_title('R1L failing windows, all contact-valid\nselection views (log)')
    ax[1].bar(reps, [100 * S['reps'][a]['R1L_max_abs_e_median'] for a in reps], color='#4a6fa5'); ax[1].axhline(3, color='r', ls='--'); ax[1].set_title('median over views of max |window error| (%)')
    ax[2].bar(reps, [100 * (S['reps'][a]['ring0_mean'] or 0) for a in reps], color='#c06c3e'); ax[2].axhline(3, color='r', ls='--'); ax[2].axhline(-3, color='r', ls='--'); ax[2].set_title('innermost ring (0-10 cm) mean bias (%)')
    ax[3].bar(reps, [100 * (S['reps'][a]['near_bias_lifted_mean'] or 0) for a in reps], color='#c06c3e'); ax[3].axhline(3, color='r', ls='--'); ax[3].set_title('lifted 30-100 mm states: mean near bias (%)')
    for a_ in ax:
        a_.tick_params(axis='x', rotation=45)
    fig.suptitle('WL115 Stage A (selection set, full-depth reference): oracle representations compared; Q* and R* are diagnostics only')
    fig.tight_layout(); p = out / 'stageA_selection_oracle_representations_summary_bars.png'; fig.savefig(p, dpi=110); plt.close(fig); made.append(p)
    # ---------------- 2. R1L window-error maps on a shared scale, contact crops
    tags = [t for t in NAMES if (root / 'analysis' / f'mapsA_{t}.npz').exists()]
    vmax = 0.15
    for tag in tags:
        z = np.load(root / 'analysis' / f'mapsA_{tag}.npz')
        r = z['r']; sl = crop_box(r, z['centres'])
        show = [a for a, _ in REPS if f'e_{a}' in z.files]
        fig, ax = plt.subplots(2, len(show) + 1, figsize=(3.1 * (len(show) + 1), 6.4))
        ax[0, 0].imshow(tonemap(z['ref_full'][sl])); ax[0, 0].set_title('reference (full depth)')
        ax[1, 0].imshow(tonemap(z['low'][sl] + z['hl_PC4'][sl])); ax[1, 0].set_title('PC4 render')
        for i, a in enumerate(show):
            e = np.where(z['centres'], z[f'e_{a}'], np.nan)[sl]
            im = ax[0, i + 1].imshow(e, cmap='RdBu_r', norm=TwoSlopeNorm(0, -vmax, vmax)); ax[0, i + 1].set_title(f'{a}: window error')
            f = z[f'fail_{a}'][sl]
            ax[1, i + 1].imshow(tonemap(z['ref_full'][sl]) * 0.5); ax[1, i + 1].imshow(np.ma.masked_where(~f, f), cmap='autumn', alpha=1)
            ax[1, i + 1].set_title(f'{a}: R1L failing ({int(z[f"fail_{a}"].sum())})')
        for a_ in ax.ravel():
            a_.axis('off')
        fig.colorbar(im, ax=ax[0, :].tolist(), fraction=0.02, label='relative luminance error of the 5x5 window (method - ref) / ref')
        fig.suptitle(f'{NAMES[tag]}: R1L 5x5-window relative error by oracle representation (shared scale +/-{int(100 * vmax)} %), failing windows in red')
        p = out / f'{fname(tag)}_R1L_window_error_by_representation_contact_crop.png'; fig.savefig(p, dpi=100); plt.close(fig); made.append(p)
    # ---------------- 3. radial ring profiles (all contact-valid selection views) per representation
    edges = [0, 0.10, 0.14, 0.18, 0.22, 0.26, 0.30, 0.35, 0.40, 0.50]
    mid = [(a + b) / 2 for a, b in zip(edges, edges[1:])]
    fig, ax = plt.subplots(1, 4, figsize=(18, 4), sharey=True)
    for k, a in enumerate(('PC4', 'BL4', 'PC8', 'R32')):
        for tag, v in M.items():
            if v['contact_valid'] and a in v['reps']:
                b = [x['b'] for x in v['reps'][a]['R4_rings']]
                ax[k].plot(mid, [100 * x for x in b], color='#c06c3e' if v['lifted'] else '#4a6fa5', alpha=0.5, lw=1)
        ax[k].axhspan(-3, 3, color='0.9'); ax[k].set_title(f'{a}'); ax[k].set_xlabel('ring centre: floor distance to support (m)')
    ax[0].set_ylabel('ring luminance bias vs full-depth reference (%)')
    fig.suptitle('Radial (R4) profiles, selection views: blue = contact / small gap, orange = lifted 30-100 mm; grey band = 3 %')
    fig.tight_layout(); p = out / 'stageA_selection_radial_ring_profiles_by_representation.png'; fig.savefig(p, dpi=110); plt.close(fig); made.append(p)
    # ---------------- 4. within-patch field and event-2 query distribution (capture views)
    for tag in tags:
        fp = root / 'analysis' / f'fieldA_{tag}.npz'
        if not fp.exists():
            continue
        z = np.load(fp)
        sph = z['is_sphere']
        fig, ax = plt.subplots(2, 2, figsize=(13, 8))
        for col, (sel, xk, xl) in enumerate(((sph, 'theta_deg', 'sphere polar angle from the bottom pole (deg)'), (~sph, 'r_floor', 'floor distance to support point (m)'))):
            x = z[xk][sel].ravel(); w = z['w'][sel].ravel(); ok = w > 0
            bins = np.linspace(0, 60 if col == 0 else 0.45, 31)
            idx = np.digitize(x, bins) - 1
            def wavg(v):
                v = v[sel].ravel()
                return np.array([np.sum((w * v)[(idx == i) & ok]) / max(np.sum(w[(idx == i) & ok]), 1e-30) if ((idx == i) & ok).any() else np.nan for i in range(len(bins) - 1)])
            xc = 0.5 * (bins[1:] + bins[:-1])
            for k_, lab, st in (('Lq', 'camera-query HIGH (physical, query-weighted)', 'k-'), ('U32', 'U 32x32', 'g--'), ('U8', 'U 8x8 (PC8)', 'b-.'), ('U4', 'U 4x4 (PC4)', 'r:')):
                ax[0, col].plot(xc, wavg(z[k_]), st, label=lab, lw=1.6)
            ax[0, col].set_xlabel(xl); ax[0, col].set_ylabel('outgoing HIGH luminance (query-bin weighted)'); ax[0, col].legend(fontsize=8)
            ax[1, col].bar(xc, [np.sum(w[(idx == i) & ok]) for i in range(len(bins) - 1)], width=bins[1] - bins[0], color='0.5')
            ax[1, col].set_xlabel(xl); ax[1, col].set_ylabel('event-2 query weight (prefix throughput, luminance)')
        ax[0, 0].set_title('sphere lowest-ring patches'); ax[0, 1].set_title('floor patches within 0.30 m')
        fig.suptitle(f'{NAMES[tag]}: within-patch full-depth HIGH vs representation, and event-2 query distribution (contact neighbourhood)')
        fig.tight_layout(); p = out / f'{fname(tag)}_within_patch_HIGH_field_and_event2_query_distribution.png'; fig.savefig(p, dpi=110); plt.close(fig); made.append(p)
    print('\n'.join(str(p) for p in made))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
