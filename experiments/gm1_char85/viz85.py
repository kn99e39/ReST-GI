"""WL85 reviewer visuals (no new metric): census distributions, reference images, signed-change maps, depth and class accounting, economics.

  python viz85.py <root>
Error / change maps use one shared scale per figure (stated in the colour bar). Palette: categorical blue / orange / aqua / yellow / violet.
"""
import json, math, os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
P = json.load(open(os.path.join(HERE, 'protocol', 'char85.json')))
INK = '#0b0b0b'; C = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#6250d6', '#e34948']
plt.rcParams.update({'font.size': 8, 'axes.edgecolor': '#52514e', 'axes.grid': True, 'grid.color': '#e4e3df', 'grid.linewidth': 0.5})
tm = lambda x: np.clip(x / (1 + x), 0, 1) ** (1 / 2.2)


def main():
    root = sys.argv[1]
    R = os.path.join(root, 'results'); out = os.path.join(R, 'viz'); os.makedirs(out, exist_ok=True)
    files = []
    save = lambda fig, name: (fig.savefig(os.path.join(out, name), dpi=110), plt.close(fig), files.append(name))
    cen = json.load(open(os.path.join(R, 'census', 'census.json')))
    fx = json.load(open(os.path.join(R, 'census', 'fixture.json')))
    # 1. census: albedo-weighted r50 distribution and class shares vs incidence
    fig, axs = plt.subplots(1, 3, figsize=(15, 3.8))
    for i, pop in enumerate(['test83_iid', 'test83_hard', 'test83_block', 'real83']):
        h = cen[pop]['theta_45']['albedo_weighted_r50_hist']
        e = np.array(h['log10_edges'])
        axs[0].step(10 ** e[:-1], h['mass'], where='post', color=C[i], label=pop)
    for x in (0.15, 0.45):
        axs[0].axvline(x, color=INK, ls=':', lw=1)
    axs[0].set_xscale('log'); axs[0].set_xlabel('lobe angular radius r50 (rad) at 45 deg incidence; dotted: S | M | B thresholds'); axs[0].set_ylabel('share of closure albedo per bin (population mean)')
    axs[0].legend(frameon=False, fontsize=7)
    th = cen['test83_iid']['incidence_deg']
    for j, k in enumerate('SMB'):
        for i, pop in enumerate(['test83_iid', 'test83_hard']):
            q = np.array([cen[pop][f'theta_{t}']['class_energy_share_q10_q50_q90'][k] for t in th])
            axs[1].plot(th, q[:, 1], color=C[j], ls='-' if i == 0 else '--', marker='o', ms=3, label=f'{k} {pop}')
            if i == 0:
                axs[1].fill_between(th, q[:, 0], q[:, 2], color=C[j], alpha=0.12)
    axs[1].set_xlabel('incidence (deg)'); axs[1].set_ylabel('energy share by class (median; band = q10-q90, test83_iid)'); axs[1].legend(frameon=False, fontsize=6, ncol=3)
    names = ['B', 'S1', 'S2']
    w = 0.25
    for i, nme in enumerate(names):
        m = fx['materials'][nme]
        a = np.array(m['lobe_albedo_by_incidence'])[:, th.index(45)]
        r = np.array(m['lobe_r50_at_45deg'])
        axs[2].scatter(r, a / a.sum(), color=C[i], s=30, label=f'{nme} ({m["set"]} #{m["index"]})')
    for x in (0.15, 0.45):
        axs[2].axvline(x, color=INK, ls=':', lw=1)
    axs[2].set_xscale('log'); axs[2].set_yscale('log'); axs[2].set_ylim(1e-4, 1.5); axs[2].set_xlabel('lobe r50 (rad) at 45 deg'); axs[2].set_ylabel('lobe share of closure albedo (fixture materials)')
    axs[2].legend(frameon=False, fontsize=7)
    fig.suptitle('WL85 Stage 0: frozen learned ER closures - fitted lobe angular support and energy', fontsize=9); fig.tight_layout(); save(fig, 'census.png')
    if not os.path.exists(os.path.join(R, 'maps.npz')):
        json.dump({'files': files}, open(os.path.join(out, 'viz_manifest.json'), 'w'), indent=1); return
    M = np.load(os.path.join(R, 'maps.npz'))
    an = json.load(open(os.path.join(R, 'analysis.json')))
    Y = lambda x: x.mean(-1)
    # 2. references and ER - P error (shared scale)
    emax = max(np.abs(M['ER_T0'] - M['P_T0']).mean(-1).max(), np.abs(M['ER_T1'] - M['P_T1']).mean(-1).max())
    fig, axs = plt.subplots(2, 4, figsize=(15, 7.4))
    for r, cfg in enumerate(['T0', 'T1']):
        axs[r, 0].imshow(tm(M[f'P_{cfg}'])); axs[r, 0].set_title(f'REF-P {cfg} (layered, 65536 spp)')
        axs[r, 1].imshow(tm(M[f'ER_{cfg}'])); axs[r, 1].set_title(f'REF-ER {cfg} (frozen ER, 65536 spp)')
        im = axs[r, 2].imshow(np.sqrt(np.abs(M[f'ER_{cfg}'] - M[f'P_{cfg}']).mean(-1)), cmap='viridis', vmin=0, vmax=math.sqrt(emax)); axs[r, 2].set_title(f'|REF-ER - REF-P| {cfg}')
        plt.colorbar(im, ax=axs[r, 2], fraction=0.046, label=f'sqrt(mean_c |err|), shared, max {emax:.3g}')
    reg = np.zeros(M['mask_frame'].shape + (3,))
    for i, k in enumerate(['enabled_reflection', 'shadow_new', 'shadow_gone', 'reflection_static', 'mover']):
        reg[M['mask_' + k]] = matplotlib.colors.to_rgb(C[i])
    axs[0, 3].imshow(reg); axs[0, 3].set_title('regions (geometric): ' + ', '.join(f'{k}' for k in ['enabled', 'shadow_new', 'shadow_gone', 'refl_static', 'mover']), fontsize=6)
    k3 = M['ER_depth1'][..., 2:].sum(-1)
    im = axs[1, 3].imshow(np.sqrt(k3), cmap='magma'); axs[1, 3].set_title('REF-ER T1: contribution of >= 3 scattering events (sqrt)'); plt.colorbar(im, ax=axs[1, 3], fraction=0.046)
    for a in axs.ravel():
        a.axis('off')
    fig.suptitle('WL85 references (tone map x/(1+x), gamma 2.2)', fontsize=9); fig.tight_layout(); save(fig, 'references.png')
    # 3. signed change maps (shared symmetric scale)
    dmax = max(np.abs(M['P_d']).max(), np.abs(M['ER_d']).max())
    sg = lambda x: np.sign(x) * np.sqrt(np.abs(x))
    fig, axs = plt.subplots(1, 4, figsize=(16, 4))
    for i, (k, t) in enumerate([('P_d', 'REF-P signed change T1 - T0'), ('ER_d', 'REF-ER signed change T1 - T0')]):
        im = axs[i].imshow(sg(M[k]), cmap='RdBu_r', vmin=-math.sqrt(dmax), vmax=math.sqrt(dmax)); axs[i].set_title(t)
    plt.colorbar(im, ax=axs[1], fraction=0.046, label=f'signed sqrt, shared, max |change| {dmax:.3g}')
    axs[2].imshow(sg(M['ER_created']), cmap='RdBu_r', vmin=-math.sqrt(dmax), vmax=math.sqrt(dmax)); axs[2].set_title('REF-ER: T1 dirty (created / changed paths)')
    axs[3].imshow(sg(-M['ER_destroyed']), cmap='RdBu_r', vmin=-math.sqrt(dmax), vmax=math.sqrt(dmax)); axs[3].set_title('REF-ER: - T0 dirty (destroyed / changed paths)')
    for a in axs:
        a.axis('off')
    fig.tight_layout(); save(fig, 'signed_change.png')
    # 4. depth shares and class accounting per region
    regs = [r for r in ['enabled_reflection', 'shadow_new', 'shadow_gone', 'reflection_static', 'stationary_rest', 'stationary_all', 'sphere', 'floor', 'wall'] if 'ER' in an['regions'].get(r, {})]
    fig, axs = plt.subplots(1, 2, figsize=(15, 4))
    x = np.arange(len(regs))
    for k in range(6):
        axs[0].bar(x + (k - 2.5) * 0.13, [an['regions'][r]['P']['depth_share_T1'][k] for r in regs], width=0.12, color=C[k], label=f'{k + 1} events')
        axs[1].bar(x + (k - 2.5) * 0.13, [an['regions'][r]['P']['depth_abs_change_share'][k] for r in regs], width=0.12, color=C[k])
    for a, t in zip(axs, ['REF-P: share of T1 radiance by number of scattering events', 'REF-P: share of |signed change| by number of scattering events']):
        a.set_xticks(x); a.set_xticklabels(regs, rotation=25, ha='right', fontsize=7); a.set_title(t)
    axs[0].legend(frameon=False, fontsize=7, ncol=6); fig.tight_layout(); save(fig, 'depth_shares.png')
    cells = an['regions']['stationary_all']['ER']['class']['cells']
    fig, axs = plt.subplots(2, 1, figsize=(15, 7))
    for i, r in enumerate(regs[:6]):
        c = an['regions'][r]['ER']['class']
        xx = np.arange(len(cells))
        axs[0].bar(xx + (i - 2.5) * 0.14, c['T1_share'], width=0.13, color=C[i], label=r)
        axs[1].bar(xx + (i - 2.5) * 0.14, c['abs_change_share'], width=0.13, color=C[i])
        axs[1].scatter(xx + (i - 2.5) * 0.14, c['abs_change_noise_share'], color=INK, s=6, zorder=3)
    for a, t in zip(axs, ['REF-ER: share of T1 radiance by class cell (k1: direct by receiver class; a<-b: receiver class a, sender class b, >= 2 events)',
                          'REF-ER: share of |signed change| by class cell (dots: split-half noise floor of the same share)']):
        a.set_xticks(np.arange(len(cells))); a.set_xticklabels(cells); a.set_title(t)
    axs[0].legend(frameon=False, fontsize=7, ncol=6); fig.tight_layout(); save(fig, 'class_accounting.png')
    # 5. economics
    ep = os.path.join(R, 'econ', 'econ85.json')
    if os.path.exists(ep):
        ec = json.load(open(ep))
        fig, axs = plt.subplots(1, 3, figsize=(16, 4))
        L = ec['ladder']
        for a, ck, lab in [(axs[0], 'gpu_ms', 'kernel GPU time (ms)'), (axs[1], 'wall_s', 'wall-clock (s), warm')]:
            for i, ek in enumerate(['err_stationary_all', 'err_enabled_reflection', 'err_frame']):
                a.loglog([r[ck] for r in L], [r[ek] for r in L], 'o', ms=3, color=C[i], label=ek[4:])
                a.loglog([m[ck] for m in ec['ladder_median']], [m[ek] for m in ec['ladder_median']], '-', color=C[i], lw=1)
            a.axhline(ec['g_stationary'], color=INK, ls=':', lw=1, label='g = ER vs P gap (stationary)')
            a.set_xlabel(lab); a.set_ylabel('relative L1 vs pooled REF-ER (T1)'); a.legend(frameon=False, fontsize=7)
        m = ec['ladder_median']
        sp = [r['spp'] for r in m]
        axs[2].loglog(sp, [r['trace_s'] for r in m], 'o-', color=C[0], label='Python tracing')
        axs[2].loglog(sp, [r['eval_s'] for r in m], 'o-', color=C[1], label='GPU execution (wall)')
        axs[2].loglog(sp, [r['gpu_ms'] / 1e3 for r in m], 'o-', color=C[2], label='kernel GPU time')
        axs[2].loglog(sp, [r['wall_s'] for r in m], 'o-', color=INK, label='total wall')
        axs[2].set_xlabel('samples per pixel'); axs[2].set_ylabel('seconds per frame (512 x 512)'); axs[2].legend(frameon=False, fontsize=7)
        fig.suptitle('WL85 Stage 4: measured REF-ER time-to-quality (uncontended RTX 3080 Ti)', fontsize=9); fig.tight_layout(); save(fig, 'economics.png')
    json.dump({'files': files}, open(os.path.join(out, 'viz_manifest.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
