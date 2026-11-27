"""WL115 Stage B reviewer figures (HELD_SEEN + FRESH, oracle level).  python fig115B.py ROOT OUT_DIR
Shared scales: one signed window-error scale (+/-15 %) and one sqrt HDR display for every map; readable file names."""
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
import metrics113 as MX  # noqa: E402
import analyzeB115 as AB  # noqa: E402
import analyze115A as AA  # noqa: E402
import stageB115 as SB  # noqa: E402

VIEWS = {'N5_W0': 'N5 fresh contact on a floor-patch corner, view W0', 'N6_W0': 'N6 fresh contact at a patch centre, view W0',
         'N7_W0': 'N7 fresh gap 8 mm, view W0', 'N8_W0': 'N8 fresh gap 45 mm, view W0', 'SEQ_E_fresh_5_W0': 'SEQ_E fresh gap 70 mm, view W0',
         'N4_W0': 'N4 held-out gap 6 cm, view W0', 'F4_W0': 'F4 held-out contact, view W0'}


def tonemap(x):
    return np.clip(x, 0, None) ** (1 / 2.2)


def fname(s):
    return s.replace(',', '').replace(' ', '_')


def main(root, out):
    root, out = Path(root), Path(out); out.mkdir(parents=True, exist_ok=True)
    M = json.loads((root / 'analysis' / 'stageB_metrics.json').read_text())
    Dd = json.loads((root / 'analysis' / 'stageB_decision.json').read_text())
    made = []
    # 1 summary bars by class
    fig, ax = plt.subplots(1, 4, figsize=(18, 4))
    S = Dd['summary']
    for k, (key, lab, sc) in enumerate((('R1L_failing_total', 'R1L failing windows', 1), ('R1L_max_abs_e_median', 'median max |window error| (%)', 100),
                                         ('ring0_abs_mean', 'innermost ring mean |bias| (%)', 100), ('lifted_near_abs_mean', 'lifted 30-100 mm near |bias| (%)', 100))):
        xs = np.arange(3)
        for i, (a, col) in enumerate((('PC4_O', '#c06c3e'), ('CAR_O', '#4a6fa5'))):
            ax[k].bar(xs + 0.38 * i, [(S[c][a][key] or 0) * sc for c in ('HELD_SEEN', 'FRESH', 'ALL')], 0.38, color=col, label=a)
        ax[k].set_xticks(xs + 0.19); ax[k].set_xticklabels(['held-out', 'fresh', 'all']); ax[k].set_title(lab)
        if k in (1, 2, 3):
            ax[k].axhline(3, color='r', ls='--', lw=1)
    ax[0].legend()
    fig.suptitle('WL115 Stage B: independent qualification of the CAR-16 oracle vs the dense 4x4 oracle (full-depth reference, contact-valid views)')
    fig.tight_layout(); p = out / 'stageB_independent_oracle_CAR16_vs_dense4x4_summary_bars.png'; fig.savefig(p, dpi=110); plt.close(fig); made.append(p)
    # 2 radial profiles
    edges = [0, 0.10, 0.14, 0.18, 0.22, 0.26, 0.30, 0.35, 0.40, 0.50]; mid = [(a + b) / 2 for a, b in zip(edges, edges[1:])]
    fig, ax = plt.subplots(1, 2, figsize=(13, 4), sharey=True)
    for k, a in enumerate(('PC4_O', 'CAR_O')):
        for tag, v in M.items():
            if v['contact_valid']:
                ax[k].plot(mid, [100 * x['b'] for x in v['arms'][a]['R4_rings']], color='#c06c3e' if v['lifted'] else '#4a6fa5', alpha=.5, lw=1,
                           ls='-' if v['cls'] == 'FRESH' else '--')
        ax[k].axhspan(-3, 3, color='0.9'); ax[k].set_title(a); ax[k].set_xlabel('ring centre: floor distance to support (m)')
    ax[0].set_ylabel('ring luminance bias vs full-depth reference (%)')
    fig.suptitle('Radial profiles (held-out dashed, fresh solid; orange = lifted >= 30 mm): CAR-16 removes the inner peak; a broad ~+2-3 % remains (angular, Stage A2)')
    fig.tight_layout(); p = out / 'stageB_independent_radial_ring_profiles_dense4x4_vs_CAR16.png'; fig.savefig(p, dpi=110); plt.close(fig); made.append(p)
    # 3 window-error maps, reference and renders for selected views (shared scales)
    V = SB.eval_views(); vmax = 0.15
    for tag, title in VIEWS.items():
        v = V[tag]
        st, gb = AB.load_refs(root, tag, v)
        ex = [dict(np.load(root / 'stageB' / 'streams' / f'stream_{tag}_{j}.npz')) for j in range(2)]
        out_, g3, reg, Rf, Mc, kap = AB.view_gate(st, gb, v, ['PC4_O', 'CAR_O'], ex)
        cen = g3['win_floor'] | g3['win_sphere_low'] | g3['silhouette']
        r = np.linalg.norm(gb['p'][..., :2] - np.asarray(v['g']['S']['c'][:2]), axis=-1)
        ys, xs = np.nonzero(cen & (r < 0.35)); sl = (slice(max(ys.min() - 40, 0), min(ys.max() + 40, 512)), slice(max(xs.min() - 40, 0), min(xs.max() + 40, 512)))
        fig, ax = plt.subplots(2, 3, figsize=(13, 8))
        ax[0, 0].imshow(tonemap(0.5 * (Rf[0] + Rf[1]))[sl]); ax[0, 0].set_title('reference (full depth)')
        for i, a in enumerate(('PC4_O', 'CAR_O')):
            ax[0, i + 1].imshow(tonemap(0.5 * (Mc[a][0] + Mc[a][1]))[sl]); ax[0, i + 1].set_title(f'{a} render')
            e, se = AA.window_map(Mc[a], Rf, kap); loc, fail = AA.r1l_location(e, se, cen, g3, r)
            im = ax[1, i + 1].imshow(np.where(cen, e, np.nan)[sl], cmap='RdBu_r', norm=TwoSlopeNorm(0, -vmax, vmax))
            ax[1, i + 1].contour(fail[sl], levels=[0.5], colors='k', linewidths=0.6)
            ax[1, i + 1].set_title(f'{a}: R1L failing {int(fail.sum())}')
        ax[1, 0].axis('off')
        for a_ in ax.ravel():
            a_.set_xticks([]); a_.set_yticks([])
        fig.colorbar(im, ax=ax[1, 1:].tolist(), fraction=0.03, label='(method - ref) / ref, 5x5 window luminance')
        fig.suptitle(f'{title}: oracle renders and R1L window error (shared +/-15 % scale)')
        p = out / f'{fname(title)}_oracle_dense4x4_vs_CAR16_render_and_window_error.png'; fig.savefig(p, dpi=100); plt.close(fig); made.append(p)
    # 4 support (activation w) of CAR-16 for SEQ_E states: floor cells (top view) and sphere cells (phi x z)
    q = C.sequences()['SEQ_E_fresh']
    fig, ax = plt.subplots(2, 4, figsize=(18, 8))
    for k, t in enumerate((2, 3, 4, 5)):
        g = q['states'][t]; dg = C.gdigest(g)
        z = np.load(root / 'stageB' / 'con' / f'car_{dg}.npz'); w = z['w']
        import tracer115 as T
        G = T.S.Geom(g)
        cx, cy = g['S']['c'][:2]
        for p in np.flatnonzero(G.PSID == 0):
            for s in range(16):
                o = G.PO[p] + (s % 4) / 4 * G.PE1[p] + (s // 4) / 4 * G.PE2[p]
                if abs(o[0] - cx) < 0.6 and abs(o[1] - cy) < 0.6:
                    ax[0, k].add_patch(plt.Rectangle((o[0], o[1]), 0.0625, 0.0625, color=plt.cm.viridis(w[p, s]), lw=0))
        ax[0, k].add_patch(plt.Circle((cx, cy), 0.4, fill=False, color='w', ls='--'))
        ax[0, k].set_xlim(cx - 0.6, cx + 0.6); ax[0, k].set_ylim(cy - 0.6, cy + 0.6); ax[0, k].set_aspect('equal')
        ax[0, k].set_title(f'floor cells, {q["label"][t]}'); ax[0, k].set_facecolor('k')
        sp = np.flatnonzero(G.PSID == T.S.SID_SPHERE)
        img = np.zeros((4 * 4, 8 * 4))
        for p in sp:
            ip, iz = G.PSPH[p].astype(int)
            for s in range(16):
                img[iz * 4 + s // 4, ip * 4 + s % 4] = w[p, s]
        ax[1, k].imshow(img, origin='lower', cmap='viridis', vmin=0, vmax=1, extent=[-180, 180, -1, 1], aspect='auto')
        ax[1, k].set_xlabel('sphere azimuth (deg)'); ax[1, k].set_ylabel('sphere z / R (bottom = -1)'); ax[1, k].set_title('sphere cells (activation w)')
    fig.suptitle('CAR-16 geometric support: activation w per 4x4 cell (0 = dense 4x4, 1 = full 16x16 refinement), recomputed from geometry at each SEQ_E state')
    fig.tight_layout(); p = out / 'SEQ_E_fresh_CAR16_support_activation_floor_and_sphere_by_gap.png'; fig.savefig(p, dpi=100); plt.close(fig); made.append(p)
    print('\n'.join(map(str, made)))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
