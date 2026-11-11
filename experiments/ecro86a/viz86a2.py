"""WL86A visuals for amendments 1-2 (no new metric).

  python viz86a2.py <root86> <out>
"""
import json, math, os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'ecro86'))
import scene86 as S  # noqa: E402
import model86 as M  # noqa: E402
C = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#6250d6', '#e34948', '#52514e']
plt.rcParams.update({'font.size': 8, 'axes.edgecolor': '#52514e', 'axes.grid': True, 'grid.color': '#e4e3df', 'grid.linewidth': 0.5})
NB, K = S.NB, S.K


def grid(v, base, nu, nv):
    return v[base:base + nu * nv].reshape(nv, nu)[::-1]


def main():
    root, out = sys.argv[1], sys.argv[2]
    R = os.path.join(out, 'results'); vz = os.path.join(R, 'viz')
    A2 = json.load(open(os.path.join(R, 'attribution2.json'))); A0 = json.load(open(os.path.join(R, 'attribution.json')))
    files = json.load(open(os.path.join(vz, 'viz_manifest.json')))['files']
    # 1. what the region reads, by transport family: truth vs the frozen operator's terms
    keys = ['E2_T1:C', 'E2_T0:C', 'E1_T1:obj_panel', 'E2_T1:D']
    fig, axs = plt.subplots(1, 4, figsize=(18, 4))
    for a, k in zip(axs, keys):
        row = A2['terms'][k]
        t = row['truth_share_read']
        a.bar([0], [t['e2_p']], color=C[1], label='2 events, panel sender (light -> panel -> patch)')
        a.bar([0], [t['e2_o']], bottom=[t['e2_p']], color=C[0], label='2 events, other senders')
        a.bar([0], [t['e3_p'] + t['e3_o']], bottom=[t['e2_p'] + t['e2_o']], color=C[2], label='>= 3 events')
        for i, sd in enumerate(row['seeds']):
            m = row['seeds'][sd]['model_share_read']
            tot = 1.0 + row['seeds'][sd]['P_total'] / max(A0['cases'][k]['H_share'], 1e-9)       # model reading relative to the truth
            a.bar([i + 1], [m['single_panel'] * tot], color=C[1])
            a.bar([i + 1], [(m['single'] - m['single_panel']) * tot], bottom=[m['single_panel'] * tot], color=C[0])
            a.bar([i + 1], [m['glob'] * tot], bottom=[m['single'] * tot], color=C[2])
        a.axhline(1.0, color='#0b0b0b', lw=0.6, ls=':')
        a.set_xticks([0, 1, 2]); a.set_xticklabels(['path-traced truth', 'frozen 8601\n(single flight | global)', 'frozen 8602'])
        a.set_title(f'{k}: higher-order reading of the region by family (truth = 1)')
    axs[0].legend(frameon=False, fontsize=7, loc='lower left')
    fig.tight_layout(); fig.savefig(os.path.join(vz, 'family_reading.png'), dpi=100); plt.close(fig); files.append('family_reading.png')
    # 2. the light -> panel -> Lambert patch family on the back wall (E2 T1): truth, state rays with exact / binned sender, learned single flight
    cfg = 'E2_T1'
    G = S.Geom(S.FX['E2']['T1'])
    z = np.load(os.path.join(R, 'fields', f'D3_{cfg}.npz'))
    rw = M.ray_weights().astype(np.float64)
    hit = z['hit'].astype(np.int64); dbin = z['dbin'].astype(np.int64)
    kk = np.arange(G.N * K) % K; pid = np.arange(G.N * K) // K
    hp = (hit >= 0) & (G.PMAT[np.clip(hit, 0, None)] == 1)
    Eb = np.zeros((G.N, 3)); Er = np.zeros((G.N, 3))
    np.add.at(Eb, pid[hp], z['q'].astype(np.float64)[hit[hp], dbin[hp]] * rw[kk[hp]][:, None])
    np.add.at(Er, pid[z['ray_idx']], z['ray_L'].astype(np.float64) * rw[kk[z['ray_idx']]][:, None])
    rho = 0.7 / math.pi
    sp = np.load(os.path.join(R, 'fields', f'D_SPLIT_{cfg}.npz'))
    T = 0.5 * (sp['e2_p_a'].astype(np.float64) + sp['e2_p_b']).mean(1).mean(1)
    Ls = np.load(os.path.join(R, 'fields', f'M2_ecro_s8601_{cfg}.npz'))['single_panel'].astype(np.float64).mean(1).mean(1)
    ims = [('path-traced truth', T), ('deployed 512 state rays, exact panel radiance', Er.mean(1) * rho), ('deployed state rays, panel radiance per outgoing bin', Eb.mean(1) * rho),
           ('frozen operator 8601, single flight from panel senders', Ls), ('panel-hit rays per patch (of 512)', hp.reshape(G.N, K).sum(1).astype(np.float64))]
    vmax = max(grid(x, 512, 16, 12).max() for _, x in ims[:4])
    fig, axs = plt.subplots(1, 5, figsize=(21, 3.6))
    for i, (n, x) in enumerate(ims):
        im = axs[i].imshow(grid(x, 512, 16, 12), cmap='viridis' if i < 4 else 'magma', vmin=0, vmax=vmax if i < 4 else None)
        axs[i].set_title(n, fontsize=7); axs[i].set_xticks([]); axs[i].set_yticks([]); axs[i].grid(False)
        if i in (3, 4):
            plt.colorbar(im, ax=axs[i], fraction=0.046, label='radiance, shared (first four)' if i == 3 else 'rays')
    fig.suptitle('WL86A E2 T1, back wall (Lambert): radiance of the light -> panel -> patch family per receiving patch (the panel is on the left wall)', fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(vz, 'panel_flight_back_wall.png'), dpi=100); plt.close(fig); files.append('panel_flight_back_wall.png')
    json.dump({'files': files}, open(os.path.join(vz, 'viz_manifest.json'), 'w'), indent=1)
    print('viz2', files)


if __name__ == '__main__':
    main()
