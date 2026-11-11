"""WL86 reviewer visuals (no new metric). Every error / change map uses one shared scale per figure, stated in its colour bar.

  python viz86.py <root>
Figures are produced for whatever stages exist (the chain may stop at a gate).
"""
import glob, json, math, os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
P = json.load(open(os.path.join(HERE, 'protocol', 'ecro86.json')))
INK = '#0b0b0b'; C = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#6250d6', '#e34948', '#52514e']
plt.rcParams.update({'font.size': 8, 'axes.edgecolor': '#52514e', 'axes.grid': True, 'grid.color': '#e4e3df', 'grid.linewidth': 0.5})
tm = lambda x: np.clip(x / (1 + x), 0, 1) ** (1 / 2.2)
sg = lambda x: np.sign(x) * np.sqrt(np.abs(x))


def load(R, pat):
    fns = sorted(glob.glob(os.path.join(R, 'renders', pat + '_r*.npz')))
    if not fns:
        return None
    acc = {}
    for fn in fns:
        z = np.load(fn)
        for k in z.files:
            if k != 'B':
                acc[k] = acc.get(k, 0) + z[k].astype(np.float64) / len(fns)
    if 'tot' not in acc:
        acc['tot'] = acc['e1'] + acc['e2'] + (acc['h'] if 'h' in acc else acc['hl'])
    return acc


def main():
    root = sys.argv[1]
    R = os.path.join(root, 'results'); out = os.path.join(R, 'viz'); os.makedirs(out, exist_ok=True)
    files = []

    def save(fig, name):
        fig.savefig(os.path.join(out, name), dpi=100); plt.close(fig); files.append(name)
    off = lambda axs: [a.axis('off') for a in np.ravel(axs)]
    seeds = [f'ecro_s{s}' for s in P['training']['seeds']]
    # 1. references and regions
    refs = {}
    for e in ('E1', 'E2'):
        for t in ('T0', 'T1'):
            refs[f'{e}_{t}'] = (load(R, f'REF_ER_{e}_{t}'), load(R, f'REF_P_{e}_{t}'))
    if all(v[0] is not None and v[1] is not None for v in refs.values()):
        emax = max(np.abs(a['tot'] - b['tot']).mean(-1).max() for a, b in refs.values())
        fig, axs = plt.subplots(4, 5, figsize=(20, 16))
        for r, (k, (er, p)) in enumerate(refs.items()):
            e = k.split('_')[0]
            mz = np.load(os.path.join(R, 'masks', f'{e}.npz'))
            axs[r, 0].imshow(tm(p['tot'])); axs[r, 0].set_title(f'REF-P {k} (layered physical)')
            axs[r, 1].imshow(tm(er['tot'])); axs[r, 1].set_title(f'REF-ER {k} (frozen ER)')
            im = axs[r, 2].imshow(np.sqrt(np.abs(er['tot'] - p['tot']).mean(-1)), cmap='viridis', vmin=0, vmax=math.sqrt(emax)); axs[r, 2].set_title('|REF-ER - REF-P|')
            plt.colorbar(im, ax=axs[r, 2], fraction=0.046, label=f'sqrt(mean_c |err|), shared, max {emax:.3g}')
            sh = er['h'].mean(-1) / np.maximum(er['tot'].mean(-1), 1e-9)
            im = axs[r, 3].imshow(sh, cmap='magma', vmin=0, vmax=1); axs[r, 3].set_title('share of >= 3 scattering events (H) in REF-ER'); plt.colorbar(im, ax=axs[r, 3], fraction=0.046)
            reg = np.zeros(mz['frame'].shape + (3,))
            for i, name in enumerate(['A', 'C', 'D', 'E', 'B_other', 'mover']):
                reg[mz[name]] = matplotlib.colors.to_rgb(C[i])
            axs[r, 4].imshow(reg); axs[r, 4].set_title('regions (geometric): A blue, C orange, D aqua, E yellow, other violet, mover red', fontsize=7)
        off(axs); fig.suptitle('WL86 references (tone map x/(1+x), gamma 2.2) and prospective regions', fontsize=10); fig.tight_layout(); save(fig, 'references.png')
        # 2. signed change
        fig, axs = plt.subplots(2, 4, figsize=(20, 10))
        dm = max(np.abs(refs[f'{e}_T1'][0]['tot'] - refs[f'{e}_T0'][0]['tot']).mean(-1).max() for e in ('E1', 'E2'))
        hm = max(np.abs(refs[f'{e}_T1'][0]['h'] - refs[f'{e}_T0'][0]['h']).mean(-1).max() for e in ('E1', 'E2'))
        for r, e in enumerate(('E1', 'E2')):
            for c, (name, i) in enumerate([('REF-P', 1), ('REF-ER', 0)]):
                d = (refs[f'{e}_T1'][i]['tot'] - refs[f'{e}_T0'][i]['tot']).mean(-1)
                im = axs[r, c].imshow(sg(d), cmap='RdBu_r', vmin=-math.sqrt(dm), vmax=math.sqrt(dm)); axs[r, c].set_title(f'{e} {name} signed change T1 - T0')
            plt.colorbar(im, ax=axs[r, 1], fraction=0.046, label=f'signed sqrt, shared, max |change| {dm:.3g}')
            for c, (name, i) in enumerate([('REF-P', 1), ('REF-ER', 0)]):
                d = (refs[f'{e}_T1'][i]['h'] - refs[f'{e}_T0'][i]['h']).mean(-1)
                im = axs[r, 2 + c].imshow(sg(d), cmap='RdBu_r', vmin=-math.sqrt(hm), vmax=math.sqrt(hm)); axs[r, 2 + c].set_title(f'{e} {name} higher-order change (events >= 3)')
            plt.colorbar(im, ax=axs[r, 3], fraction=0.046, label=f'signed sqrt, shared, max {hm:.3g}')
        off(axs); fig.tight_layout(); save(fig, 'signed_change.png')
    # 3. higher-order term: reference, oracle projection, ECRO, STALE
    arms = [('oracle projection', 'ORA_{c}'), (f'ECRO {seeds[0]}', 'EC_' + seeds[0] + '_ecro_{c}'), (f'ECRO {seeds[1]}', 'EC_' + seeds[1] + '_ecro_{c}'),
            ('STALE', 'EC_' + seeds[0] + '_stale_{c}'), ('no global term', 'EC_' + seeds[0] + '_nog_{c}')]
    for e in ('E1', 'E2'):
        c = f'{e}_T1'
        if refs.get(c, (None,))[0] is None:
            continue
        h = refs[c][0]['h']
        got = [(n, load(R, p.format(c=c))) for n, p in arms]
        got = [(n, g) for n, g in got if g is not None]
        if not got:
            continue
        emax = max(np.abs(g['hl'] - h).mean(-1).max() for _, g in got)
        fig, axs = plt.subplots(2, len(got) + 1, figsize=(4.2 * (len(got) + 1), 8.6))
        axs[0, 0].imshow(tm(3 * h)); axs[0, 0].set_title(f'{c} REF-ER higher-order term h (x3)')
        axs[1, 0].imshow(tm(refs[c][0]['tot'])); axs[1, 0].set_title('REF-ER total')
        for i, (n, g) in enumerate(got):
            axs[0, i + 1].imshow(tm(3 * g['hl'])); axs[0, i + 1].set_title(f'{n}: learned / looked-up term (x3)')
            im = axs[1, i + 1].imshow(np.sqrt(np.abs(g['hl'] - h).mean(-1)), cmap='viridis', vmin=0, vmax=math.sqrt(emax)); axs[1, i + 1].set_title(f'|{n} - h|')
        plt.colorbar(im, ax=axs[1, len(got)], fraction=0.046, label=f'sqrt(mean_c |err|), shared, max {emax:.3g}')
        off(axs); fig.tight_layout(); save(fig, f'higher_order_{e}.png')
        # edit response of the higher-order term
        c0 = f'{e}_T0'
        h0 = refs[c0][0]['h']
        dh = (h - h0).mean(-1)
        rows = [('REF-ER', dh)]
        for n, p in [arms[0], arms[1]]:
            a1, a0 = load(R, p.format(c=c)), load(R, p.format(c=c0))
            if a1 is not None and a0 is not None:
                rows.append((n, (a1['hl'] - a0['hl']).mean(-1)))
        st, e0 = load(R, arms[3][1].format(c=c)), load(R, arms[1][1].format(c=c0))
        if st is not None and e0 is not None:
            rows.append(('STALE (T0 state, current source)', (st['hl'] - e0['hl']).mean(-1)))
        mx = max(np.abs(d).max() for _, d in rows)
        fig, axs = plt.subplots(1, len(rows), figsize=(5 * len(rows), 5))
        for a, (n, d) in zip(np.ravel(axs), rows):
            im = a.imshow(sg(d), cmap='RdBu_r', vmin=-math.sqrt(mx), vmax=math.sqrt(mx)); a.set_title(f'{e}: change of the higher-order term, {n}')
        plt.colorbar(im, ax=np.ravel(axs)[-1], fraction=0.046, label=f'signed sqrt, shared, max {mx:.3g}')
        off(axs); fig.tight_layout(); save(fig, f'higher_order_change_{e}.png')
    # 4. unseen sources
    got = [(ls, load(R, f'REF_ER_E1_T1_{ls}'), load(R, f'EC_{seeds[0]}_ecro_E1_T1_{ls}')) for ls in ('LMIX', 'LSPOT')]
    got = [g for g in got if g[1] is not None and g[2] is not None]
    if got:
        emax = max(np.abs(g[2]['hl'] - g[1]['h']).mean(-1).max() for g in got)
        fig, axs = plt.subplots(len(got), 4, figsize=(18, 4.6 * len(got)))
        axs = np.atleast_2d(axs)
        for r, (ls, rf, ec) in enumerate(got):
            axs[r, 0].imshow(tm(rf['tot'])); axs[r, 0].set_title(f'{ls}: REF-ER total (unseen source)')
            axs[r, 1].imshow(tm(ec['e1'] + ec['e2'] + ec['hl'])); axs[r, 1].set_title(f'{ls}: ECRO total')
            axs[r, 2].imshow(tm(3 * rf['h'])); axs[r, 2].set_title('REF-ER h (x3)')
            im = axs[r, 3].imshow(np.sqrt(np.abs(ec['hl'] - rf['h']).mean(-1)), cmap='viridis', vmin=0, vmax=math.sqrt(emax)); axs[r, 3].set_title('|ECRO term - h|')
            plt.colorbar(im, ax=axs[r, 3], fraction=0.046, label=f'sqrt(mean_c |err|), shared, max {emax:.3g}')
        off(axs); fig.tight_layout(); save(fig, 'unseen_sources.png')
    # 5. E_H by region
    fj, sj = os.path.join(R, 'final.json'), os.path.join(R, 'stage1.json')
    if os.path.exists(sj):
        s1 = json.load(open(sj)); fn = json.load(open(fj)) if os.path.exists(fj) else None
        regs = ['stationary_all', 'A', 'C', 'D', 'E', 'obj_floor', 'obj_walls', 'obj_ceiling', 'obj_panel', 'obj_sphere']
        fig, axs = plt.subplots(1, 2, figsize=(18, 4.4))
        for a, e in zip(axs, ('E1', 'E2')):
            series = [('LOW only (= H share)', lambda k: s1[e]['oracle']['T1'].get(k, {}).get('H_share')), ('oracle projection', lambda k: s1[e]['oracle']['T1'].get(k, {}).get('E_H'))]
            if fn:
                for sd in seeds:
                    series.append((f'ECRO {sd}', lambda k, sd=sd: fn['seeds'][sd]['fidelity'][e]['T1'].get(k, {}).get('E_H')))
                series.append(('STALE', lambda k: fn['seeds'][seeds[0]]['stale'][e]['E_H'].get(k, {}).get('E_H')))
                series.append(('no global', lambda k: fn['seeds'][seeds[0]]['ablation'][e]['nog'].get(k, {}).get('E_H')))
                series.append(('no single flight', lambda k: fn['seeds'][seeds[0]]['ablation'][e]['nos'].get(k, {}).get('E_H')))
            x = np.arange(len(regs)); w = 0.8 / len(series)
            for i, (n, f) in enumerate(series):
                a.bar(x + (i - len(series) / 2 + 0.5) * w, [f(k) or 0 for k in regs], width=w * 0.92, color=C[i % len(C)], label=n)
            a.set_yscale('log'); a.set_xticks(x); a.set_xticklabels(regs, rotation=20, ha='right'); a.set_title(f'{e} T1: higher-order error relative to total radiance (E_H), by region')
            a.legend(frameon=False, fontsize=7, ncol=4)
        fig.tight_layout(); save(fig, 'error_by_region.png')
    # 6. training
    logs = [(sd, os.path.join(R, 'models', f'{sd}.json')) for sd in seeds]
    logs = [(sd, json.load(open(p))) for sd, p in logs if os.path.exists(p)]
    if logs:
        fig, ax = plt.subplots(1, 1, figsize=(7, 4))
        for i, (sd, j) in enumerate(logs):
            ax.semilogy([r['step'] for r in j['log']], [r['loss'] for r in j['log']], color=C[i], label=f'{sd} training loss')
            v = [(r['step'], r['val']) for r in j['log'] if 'val' in r]
            ax.semilogy([a for a, _ in v], [b for _, b in v], 'o--', color=C[i], ms=3, label=f'{sd} validation field rel. L1')
        ax.set_xlabel('step'); ax.legend(frameon=False, fontsize=7); fig.tight_layout(); save(fig, 'training.png')
    # 7. economics
    ep = os.path.join(R, 'econ', 'econ86.json')
    if os.path.exists(ep):
        ec = json.load(open(ep))
        fig, axs = plt.subplots(2, 3, figsize=(18, 9))
        for r, e in enumerate(('E1', 'E2')):
            L = ec[e]['ladder']
            for a, ck, lab in [(axs[r, 0], 'gpu_ms', 'kernel GPU time per frame (ms)'), (axs[r, 1], 'wall_s', 'wall-clock per frame (s), warm')]:
                for i, arm in enumerate(('pt', 'ecro')):
                    a.loglog([x[ck] for x in L[arm]], [x['err_stationary_all'] for x in L[arm]], 'o', ms=3, color=C[i])
                    md = ec[e]['ladder_median'][arm]
                    a.loglog([m[ck] for m in md], [m['err_stationary_all'] for m in md], '-', color=C[i], lw=1.2, label={'pt': 'full path tracing (frozen ER)', 'ecro': 'ECRO (LOW + learned table)'}[arm])
                a.axhline(ec[e]['g_stationary'], color=INK, ls=':', lw=1, label='g = REF-ER vs REF-P gap')
                a.set_xlabel(lab); a.set_ylabel(f'{e} T1: relative L1 vs pooled REF-ER (stationary pixels)'); a.legend(frameon=False, fontsize=7)
            em = ec[e]['edit_median']
            keys = ['geometry_update_s', 'state_selective_s', 'state_full_s', 'source_s', 'inference_s', 'transfer_s']
            axs[r, 2].bar(range(len(keys)), [1e3 * em[k] for k in keys], color=C[0])
            axs[r, 2].set_xticks(range(len(keys))); axs[r, 2].set_xticklabels([k[:-2] for k in keys], rotation=20, ha='right'); axs[r, 2].set_ylabel(f'{e}: ECRO per-edit cost (ms wall, median)')
        fig.suptitle('WL86 Stage 4: measured economics (uncontended GPU)', fontsize=10); fig.tight_layout(); save(fig, 'economics.png')
    json.dump({'files': files}, open(os.path.join(out, 'viz_manifest.json'), 'w'), indent=1)
    print('viz', files)


if __name__ == '__main__':
    main()
