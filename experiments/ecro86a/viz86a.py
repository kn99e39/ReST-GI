"""WL86A reviewer visuals (no new metric). Shared scales per figure, stated in the colour bars.

  python viz86a.py <root86> <out>
"""
import glob, json, math, os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'ecro86'))
import scene86 as S  # noqa: E402
C = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#6250d6', '#e34948', '#52514e']
plt.rcParams.update({'font.size': 8, 'axes.edgecolor': '#52514e', 'axes.grid': True, 'grid.color': '#e4e3df', 'grid.linewidth': 0.5})
SHAPES = ['floor', 'ceiling', 'wall_back', 'wall_left', 'wall_right', 'panel', 'sphere', 'boxA', 'boxB']
NB = S.NB


def grid(v, base, nu, nv):
    return v[base:base + nu * nv].reshape(nv, nu)[::-1]


def main():
    root, out = sys.argv[1], sys.argv[2]
    R = os.path.join(out, 'results'); vz = os.path.join(R, 'viz'); os.makedirs(vz, exist_ok=True)
    A = json.load(open(os.path.join(R, 'attribution.json')))
    files = []

    def save(fig, name):
        fig.savefig(os.path.join(vz, name), dpi=100); plt.close(fig); files.append(name)
    # 1. signed rendered error by sender shape: learned vs encoding-equivalent field
    keys = [k for k in A['cases'] if k in ('E2_T1:C', 'E2_T0:C', 'E1_T1:obj_panel', 'E2_T1:D')]
    fig, axs = plt.subplots(1, len(keys), figsize=(5 * len(keys), 4))
    for a, k in zip(np.ravel(axs), keys):
        row = A['cases'][k]
        x = np.arange(len(SHAPES)); w = 0.2
        series = [('learned (mean of 2 seeds)', row['learned_by_shape'])] + [(v, row['diag'][v]['by_shape']) for v in ('ENC_ALL', 'ANG_PANEL', 'ANG_BROAD')]
        for i, (n, d) in enumerate(series):
            a.bar(x + (i - 1.5) * w, [d[s] for s in SHAPES], width=w * 0.92, color=C[i], label=n)
        a.axhline(0, color='#0b0b0b', lw=0.6); a.set_xticks(x); a.set_xticklabels(SHAPES, rotation=30, ha='right')
        a.set_title(f'{k}: signed error of the higher-order term by SENDER shape (of total radiance)'); a.legend(frameon=False, fontsize=7)
    fig.tight_layout(); save(fig, 'error_by_sender.png')
    # 2. patch maps (E2_T1): relative table-level deviation on the back wall and the floor
    cfg = 'E2_T1'
    T = np.load(os.path.join(R, 'fields', f'D_TRUE_{cfg}.npz')); T = 0.5 * (T['fa'].astype(np.float64) + T['fb'])
    tabs = {'learned 8601': np.load(os.path.join(root, 'results', 'renders', f'EC_ecro_s8601_ecro_{cfg}_r0.npz'))['B'].astype(np.float64),
            'learned 8602': np.load(os.path.join(root, 'results', 'renders', f'EC_ecro_s8602_ecro_{cfg}_r0.npz'))['B'].astype(np.float64)}
    for v in ('ENC_ALL', 'ANG_PANEL', 'ANG_BROAD'):
        z = np.load(os.path.join(R, 'fields', f'D_{v}_{cfg}.npz')); tabs[v] = 0.5 * (z['fa'].astype(np.float64) + z['fb'])
    W = np.load(os.path.join(out, 'stageB', f'importance_{cfg}.npz'))['C'].astype(np.float64).reshape(-1, NB, 3)
    lev = lambda B: B.sum((1, 2)) / np.maximum(T.sum((1, 2)), 1e-30) - 1.0
    fig, axs = plt.subplots(2, len(tabs) + 1, figsize=(3.6 * (len(tabs) + 1), 6.4))
    imp = (W * T).sum((1, 2)); imp /= imp.sum()
    for r, (nm, base, nu, nv) in enumerate([('back wall', 512, 16, 12), ('floor', 0, 16, 16)]):
        im0 = axs[r, 0].imshow(grid(imp, base, nu, nv), cmap='magma'); axs[r, 0].set_title(f'{nm}: importance for region C (share of what the panel / sphere read)')
        plt.colorbar(im0, ax=axs[r, 0], fraction=0.046)
        for i, (n, B) in enumerate(tabs.items()):
            im = axs[r, i + 1].imshow(grid(lev(B), base, nu, nv), cmap='RdBu_r', vmin=-0.3, vmax=0.3); axs[r, i + 1].set_title(f'{nm}: {n}, table level / truth - 1')
        plt.colorbar(im, ax=axs[r, len(tabs)], fraction=0.046, label='relative deviation, shared +-0.3')
    for a in np.ravel(axs):
        a.grid(False); a.set_xticks([]); a.set_yticks([])
    fig.suptitle('WL86A E2 T1: patch-level deviation of the higher-order table from the path-traced truth (panel is on the left wall; x increases to the right)', fontsize=9)
    fig.tight_layout(); save(fig, 'patch_maps_E2_T1.png')
    # 3. beams: identical encodings, different responses (panel patch)
    bd = os.path.join(R, 'beams')
    fig, axs = plt.subplots(2, 4, figsize=(15, 6.4))
    ld = lambda n: (lambda z: 0.5 * (z['fa'].astype(np.float64) + z['fb']))(np.load(os.path.join(bd, n)))
    pa, pb = ld('panel_S1_polar_a.npz'), ld('panel_S1_polar_b.npz')
    mm = np.load(os.path.join(bd, 'M_ecro_s8601_panel_S1_polar_a.npy')).astype(np.float64)
    vmax = max(x.sum((1, 2)).max() for x in (pa, pb, mm))
    for r, (nm, base, nu, nv) in enumerate([('back wall', 512, 16, 12), ('floor', 0, 16, 16)]):
        for i, (n, B) in enumerate([('beam a (path traced)', pa), ('beam b, same bin, same encoding (path traced)', pb), ('frozen operator (identical for a and b)', mm)]):
            im = axs[r, i].imshow(grid(B.sum((1, 2)), base, nu, nv), cmap='viridis', vmin=0, vmax=vmax); axs[r, i].set_title(f'{nm}: {n}')
        plt.colorbar(im, ax=axs[r, 2], fraction=0.046, label='table sum over bins and RGB, shared')
        d = grid((pa - pb).sum((1, 2)), base, nu, nv); m = np.abs(d).max()
        im = axs[r, 3].imshow(d, cmap='RdBu_r', vmin=-m, vmax=m); axs[r, 3].set_title(f'{nm}: a - b'); plt.colorbar(im, ax=axs[r, 3], fraction=0.046)
    for a in np.ravel(axs):
        a.grid(False); a.set_xticks([]); a.set_yticks([])
    fig.suptitle('WL86A: two collimated beams on one panel patch with directions in the same incident bin (identical encodings): higher-order response per receiving patch', fontsize=9)
    fig.tight_layout(); save(fig, 'beam_pair_panel.png')
    # 4. rendered error of the higher-order term (E2_T1): learned vs encoding-equivalent source vs truth table
    fns = sorted(glob.glob(os.path.join(root, 'results', 'renders', f'REF_ER_{cfg}_r*.npz')))
    h = np.mean([np.load(f)['h'].astype(np.float64) for f in fns], 0)
    ims = [('learned 8601', np.mean([np.load(f)['hl'].astype(np.float64) for f in sorted(glob.glob(os.path.join(root, 'results', 'renders', f'EC_ecro_s8601_ecro_{cfg}_r*.npz')))], 0))]
    for v in ('ENC_ALL', 'ANG_PANEL', 'TRUE'):
        ff = sorted(glob.glob(os.path.join(R, 'renders', f'DR_{v}_{cfg}_r*.npz')))
        if ff:
            ims.append((f'{v} table', np.mean([np.load(f)['hl'].astype(np.float64) for f in ff], 0)))
    mx = max(np.abs(x - h).mean(-1).max() for _, x in ims)
    fig, axs = plt.subplots(1, len(ims), figsize=(5 * len(ims), 5))
    for a, (n, x) in zip(np.ravel(axs), ims):
        d = (x - h).mean(-1)
        im = a.imshow(np.sign(d) * np.sqrt(np.abs(d)), cmap='RdBu_r', vmin=-math.sqrt(mx), vmax=math.sqrt(mx)); a.set_title(f'{cfg}: {n} - REF-ER higher-order term'); a.axis('off')
    plt.colorbar(im, ax=np.ravel(axs)[-1], fraction=0.046, label=f'signed sqrt, shared, max {mx:.3g}')
    fig.tight_layout(); save(fig, 'rendered_error_E2_T1.png')
    json.dump({'files': files}, open(os.path.join(vz, 'viz_manifest.json'), 'w'), indent=1)
    print('viz', files)


if __name__ == '__main__':
    main()
