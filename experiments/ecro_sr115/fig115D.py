"""WL115 Stage D / E / G reviewer figures.  python fig115D.py ROOT OUT_DIR
Learned vs oracle (static, independent views), renders + shared-scale signed HDR residuals for selected views, dynamic sequences
(near bias per state incl. stale tables, R7 tracking) and the cost breakdown. Readable names; one residual scale for all maps."""
import json, math, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

HERE = Path(__file__).resolve().parent
for d in (HERE, HERE.parent / 'ecro_fd114', HERE.parent / 'ecro_sss113'):
    sys.path.insert(0, str(d))
import common115 as C  # noqa: E402

ARMS = ('PC4_O', 'CAR_O', 'B1', 'CAR1', 'B2', 'CAR2')
COL = {'PC4_O': '#d9a07a', 'CAR_O': '#8fb3d9', 'B1': '#c06c3e', 'CAR1': '#2f5f9e', 'B2': '#a85a30', 'CAR2': '#1f4477'}
SHOW = {'N5_W0': 'N5 fresh contact (floor-patch corner), W0', 'N8_W0': 'N8 fresh gap 45 mm, W0', 'SEQ_E_fresh_5_W0': 'SEQ_E fresh gap 70 mm, W0',
        'SEQ_E_fresh_0_W0': 'SEQ_E fresh contact, W0', 'N4_W0': 'N4 held-out gap 6 cm, W0', 'SEQ_D_independent_4_W0': 'SEQ_D held-out gap 100 mm, W0'}


def tm(x):
    return np.clip(np.clip(x, 0, None) ** (1 / 2.2), 0, 1)


def fn(s):
    return s.replace(',', '').replace(' ', '_').replace('(', '').replace(')', '')


def main(root, out):
    root, out = Path(root), Path(out); out.mkdir(parents=True, exist_ok=True)
    D = json.loads((root / 'analysis' / 'stageD_metrics.json').read_text())
    E = json.loads((root / 'analysis' / 'stageE_metrics.json').read_text())
    made = []
    # 1 static summary per class
    fig, ax = plt.subplots(1, 4, figsize=(19, 4.3))
    for cls_i, cls in enumerate(('HELD_SEEN', 'FRESH')):
        rows = [r for r in D.values() if r['cls'] == cls and r['contact_valid']]
        for k, (lab, f, sc) in enumerate((('R1L failing windows', lambda r, a: r['arms'][a]['R1L']['failing'], 1),
                                          ('innermost ring mean |bias| (%)', lambda r, a: abs(r['arms'][a]['R4_rings'][0]['b']) if not math.isnan(r['arms'][a]['R4_rings'][0]['b']) else np.nan, 100),
                                          ('lifted (>= 30 mm) near |bias| (%)', lambda r, a: abs(r['arms'][a]['b_near']) if r['lifted'] else np.nan, 100),
                                          ('views passing the full 3 % gate', lambda r, a: float(r['arms'][a]['all_pass']), 1))):
            vals = [(np.nansum if k in (0, 3) else np.nanmean)([f(r, a) for r in rows if a in r['arms']]) * sc for a in ARMS]
            ax[k].bar(np.arange(len(ARMS)) + 0.4 * cls_i, vals, 0.4, color=[COL[a] for a in ARMS], alpha=1.0 if cls_i else 0.55,
                      label='fresh (solid)' if cls_i else 'held-out (light)')
            ax[k].set_xticks(np.arange(len(ARMS)) + 0.2); ax[k].set_xticklabels(ARMS, rotation=30); ax[k].set_title(lab)
    for k in (1, 2):
        ax[k].axhline(3, color='r', ls='--', lw=1)
    ax[0].legend()
    fig.suptitle('WL115 Stage D: learned CAR (two seeds) vs matched dense B (WL114) and the oracles; independent views, full-depth reference')
    fig.tight_layout(); p = out / 'stageD_independent_learned_CAR_vs_dense_B_and_oracles_summary_bars.png'; fig.savefig(p, dpi=110); plt.close(fig); made.append(p)
    # 2 renders + signed residual maps (one shared HDR scale)
    sel = [t for t in SHOW if (root / 'analysis' / f'mapsD_{t}.npz').exists()]
    vmax = 0.0
    for t in sel:
        z = np.load(root / 'analysis' / f'mapsD_{t}.npz')
        for a in ('B1', 'CAR1'):
            vmax = max(vmax, float(np.percentile(np.abs((z[f'full_{a}'] - z['ref_full']).mean(-1))[z['mask_floor']], 99.9)))
    for t in sel:
        z = np.load(root / 'analysis' / f'mapsD_{t}.npz')
        ys, xs = np.nonzero(z['mask_near']) if z['mask_near'].any() else np.nonzero(z['mask_floor'])
        cy, cx = int(np.median(ys)), int(np.median(xs)); sl = (slice(max(cy - 150, 0), cy + 110), slice(max(cx - 170, 0), cx + 170))
        arms = [a for a in ('B1', 'CAR1', 'CAR_O') if f'full_{a}' in z.files]
        fig, ax = plt.subplots(2, len(arms) + 1, figsize=(4 * (len(arms) + 1), 7.5))
        ax[0, 0].imshow(tm(z['ref_full'][sl])); ax[0, 0].set_title('reference (full depth)'); ax[1, 0].axis('off')
        for i, a in enumerate(arms):
            ax[0, i + 1].imshow(tm(z[f'full_{a}'][sl])); ax[0, i + 1].set_title(f'{a} render')
            d = (z[f'full_{a}'] - z['ref_full']).mean(-1)
            im = ax[1, i + 1].imshow(np.sign(d[sl]) * np.sqrt(np.abs(d[sl])), cmap='RdBu_r', norm=TwoSlopeNorm(0, -math.sqrt(vmax), math.sqrt(vmax)))
            ax[1, i + 1].set_title(f'{a} - reference (signed sqrt HDR)')
        for a_ in ax.ravel():
            a_.set_xticks([]); a_.set_yticks([])
        cb = fig.colorbar(im, ax=ax[1, 1:].tolist(), fraction=0.03)
        ticks = np.array([-vmax, -vmax / 4, 0, vmax / 4, vmax]); cb.set_ticks(np.sign(ticks) * np.sqrt(np.abs(ticks))); cb.set_ticklabels([f'{x:+.3f}' for x in ticks])
        cb.set_label('RGB-mean HDR residual (shared scale, sqrt display)')
        fig.suptitle(f'{SHOW[t]}: learned dense B1 vs learned CAR1 vs CAR oracle')
        p = out / f'{fn(SHOW[t])}_learned_B1_vs_CAR1_vs_oracle_renders_and_residuals.png'; fig.savefig(p, dpi=100); plt.close(fig); made.append(p)
    # 3 dynamic sequences: near bias per state and R7
    for sname in ('SEQ_E_fresh', 'SEQ_D_independent', 'SEQ_A_lift', 'SEQ_C_independent'):
        if sname not in E:
            continue
        q = C.sequences()[sname]; rows = E[sname]['state_rows']
        fig, ax = plt.subplots(1, len(q['views']), figsize=(7 * len(q['views']), 4.2), squeeze=False)
        for k, vn in enumerate(q['views']):
            for a, st_ in (('B1', '-'), ('CAR1', '-'), ('CAR2', '-'), ('CAR_O', ':'), ('stale_CAR1', '--'), ('stale_B1', '--')):
                ys = [rows.get(f'{t}_{vn}', {}).get(a, [None])[0] for t in range(len(q['states']))]
                xs = [t for t, y in enumerate(ys) if y is not None]
                if xs:
                    ax[0, k].plot(xs, [100 * ys[t] for t in xs], st_, marker='o', label=a, color=COL.get(a.replace('stale_', ''), 'k'), alpha=0.6 if 'stale' in a else 1)
            ax[0, k].axhspan(-3, 3, color='0.92'); ax[0, k].set_xticks(range(len(q['states']))); ax[0, k].set_xticklabels(q['label'], rotation=40, fontsize=7)
            ax[0, k].set_ylabel('near-region bias vs full-depth reference (%)'); ax[0, k].set_title(f'view {vn}')
        ax[0, 0].legend(fontsize=8)
        tr = E[sname]['tracking_R7']
        txt = '; '.join(f'{a}: ' + ', '.join(f'{vn} slope {x["slope"]:.2f}' for vn, x in tr.get(a, {}).items()) for a in ('B1', 'CAR1', 'CAR2', 'CAR_O') if tr.get(a))
        fig.suptitle(f'{sname}: same weights over all states; dashed = stale previous-state table. R7 {txt}', fontsize=9)
        fig.tight_layout(); p = out / f'{sname}_dynamic_near_bias_per_state_learned_CAR_vs_B_and_stale.png'; fig.savefig(p, dpi=105); plt.close(fig); made.append(p)
    # 4 cost
    cp = root / 'analysis' / 'cost.json'
    if cp.exists():
        c = json.loads(cp.read_text()); A = c['arms']
        names = [a for a in ('OB1', 'B1', 'CAR1') if a in A]
        fig, ax = plt.subplots(1, 3, figsize=(16, 4))
        ax[0].bar(names, [A[a]['generator']['median_ms'] for a in names], color='#4a6fa5'); ax[0].set_title('generator latency (ms, median)')
        ax[1].bar(names, [A[a]['table_mib'] for a in names], color='#4a6fa5'); ax[1].set_title('HIGH table storage (MiB)')
        ax[2].bar(names, [A[a]['edit_to_table']['median_ms'] for a in names], color='#c06c3e', label='edit -> new table')
        ax[2].bar(names, [A[a]['frame_512_1spp']['median_ms'] for a in names], color='#4a6fa5', alpha=0.7, label='512^2 1-spp frame')
        ax[2].legend(); ax[2].set_title('ms (experiment-local Dr.Jit 0.4.6 runtime)')
        fig.suptitle('WL115 Stage G: cost of coarse (OB1), dense 4x4 (B1) and CAR-16 (CAR1) learned HIGH, local RTX 5080')
        fig.tight_layout(); p = out / 'stageG_cost_generator_storage_frame_edit_latency.png'; fig.savefig(p, dpi=110); plt.close(fig); made.append(p)
    print('\n'.join(map(str, made)))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
