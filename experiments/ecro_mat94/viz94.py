"""WL94 figures (called by an94.final). Error / response maps use ONE absolute scale per quantity across every figure of the worklog
(signed sqrt mapping with a real-unit colour bar); images share one exposure (untextured reference median)."""
import glob, os
import numpy as np
import an94 as AN

VLAB = {'none': 'untextured (box A 0.5)', 'U07': 'box A all 0.7 (vocabulary state)', 'U03': 'box A all 0.3 (new value)',
        'bx75': 'checker 0.7 / 0.5', 'bx38': 'checker 0.3 / 0.8', 'bx38i': 'inverted checker 0.8 / 0.3'}


def tm(x, key):
    y = x * key
    return np.clip(y / (1 + y), 0, 1) ** (1 / 2.2)


def sq(ax, x, vmax, plt, title=None):
    im = ax.imshow(np.sign(x) * np.sqrt(np.abs(x) / vmax), cmap='RdBu_r', vmin=-1, vmax=1)
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    t = np.array([-1, -0.5, -0.25, 0, 0.25, 0.5, 1])
    cb.set_ticks(t); cb.set_ticklabels([f'{np.sign(v) * v * v * vmax:.3g}' for v in t]); cb.ax.tick_params(labelsize=6)
    ax.set_xticks([]); ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=7)


def crop_of(M, pad=12):
    m = M['panel'] | (M['sid'] == 7)
    ys, xs = np.nonzero(m)
    return slice(max(ys.min() - pad, 0), ys.max() + pad), slice(max(xs.min() - pad, 0), xs.max() + pad)


def figures(R, outd, B, Cs, L):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    os.makedirs(outd, exist_ok=True)
    files = []
    M = AN.masks(R)
    refs = {v: AN.load_ref(R, v) for v in AN.VARIANTS}
    key = 0.5 / float(np.median(AN.lum(refs['none']['tot'].mean(0))[AN.lum(refs['none']['tot'].mean(0)) > 0]))
    cy, cx = crop_of(M)
    models = [('own_s8601', 'frozen own s8601'), ('own_s8602', 'frozen own s8602')]
    for c in Cs:
        models += [(f'A_s{c["seed"]}', f'smoke A s{c["seed"]} (fixed)'), (f'B_s{c["seed"]}', f'smoke B s{c["seed"]} (diverse)')]
    hl = {}
    for mt, _ in models:
        for v in AN.VARIANTS:
            for a in ('C0', 'C2_src', 'C2_rcv', 'C2_both', 'C2_srcS1'):
                x = AN.load_hl(R, v, mt, a)
                if x is not None:
                    hl[(mt, v, a)] = x.mean(0)
    href = {v: refs[v]['h'].mean(0) for v in AN.VARIANTS}
    # one absolute scale per quantity (99.9th percentile over every map shown in the worklog, inside the crop)
    errs = [np.abs(AN.lum(x - href[v]))[cy, cx] for (mt, v, a), x in hl.items()]
    E_MAX = float(np.quantile(np.concatenate([e.ravel() for e in errs]), 0.999)) if errs else 1.0
    dres = [np.abs(AN.lum(href[v] - href['none']))[cy, cx] for v in AN.TEX]
    D_MAX = float(np.quantile(np.concatenate([e.ravel() for e in dres]), 0.999))

    # ---- fig 0: references of every variant (full frame + higher-order close-up)
    fig, ax = plt.subplots(3, 6, figsize=(24, 12))
    for j, v in enumerate(AN.VARIANTS):
        t = refs[v]['tot'].mean(0)
        ax[0, j].imshow(tm(t, key)); ax[0, j].contour(M['ROI_T'], [0.5], colors='c', linewidths=0.6)
        ax[0, j].set_title(f'{v}: {VLAB[v]}\nREF full frame (cyan ROI_T)', fontsize=8)
        ax[1, j].imshow(tm(href[v][cy, cx], key)); ax[1, j].set_title('REF higher-order h (events >= 3), close-up', fontsize=8)
        if v == 'none':
            ax[2, j].imshow(np.where(M['ROI_T'][..., None], [0, 1, 1], np.where(M['DIFF'][..., None], [1, 0.6, 0], tm(t, key) * 0.4)))
            ax[2, j].set_title('regions: cyan ROI_T (sharp reader), orange DIFF (diffuse reader)', fontsize=8)
        else:
            sq(ax[2, j], AN.lum(href[v] - href['none'])[cy, cx], D_MAX, plt, 'REF texture response h(v) - h(none)')
    for a_ in ax[:2].ravel():
        a_.set_xticks([]); a_.set_yticks([])
    fig.suptitle('WL94 Stage B physical references on CK2 (REF-ER 4 x 8,192 spp, common random numbers; one exposure)', fontsize=11)
    fn = os.path.join(outd, 'fig0_references.png'); fig.savefig(fn, dpi=80, bbox_inches='tight'); plt.close(fig); files.append(fn)

    # ---- fig 1: Stage B table amplitudes
    fig, ax = plt.subplots(2, 3, figsize=(20, 9))
    for i, sd in enumerate(AN.SEEDS):
        tb = B['table'][f's{sd}']
        for j, comp in enumerate(('B', 'S', 'G')):
            arms = AN.ARMS_B if comp != 'G' else ['C0']
            w = 0.8 / len(arms)
            for k, a in enumerate(arms):
                vals = [tb[a][f'{v}|{comp}|T']['amplitude'] for v in AN.TEX]
                ax[i, j].bar(np.arange(len(AN.TEX)) + k * w, vals, w, label=a)
            ax[i, j].axhline(1, color='k', lw=0.8); ax[i, j].set_xticks(np.arange(len(AN.TEX)) + 0.4 - w / 2); ax[i, j].set_xticklabels(AN.TEX)
            ax[i, j].set_title(f'own s{sd}: component {comp} log-response amplitude over box-A patches (1 = reference)', fontsize=8)
            ax[i, j].legend(fontsize=6)
    fn = os.path.join(outd, 'fig1_stageB_table_amplitude.png'); fig.savefig(fn, dpi=80, bbox_inches='tight'); plt.close(fig); files.append(fn)

    # ---- fig 2: direction / face / sender / receivers (s8601, s8602)
    fig, ax = plt.subplots(2, 4, figsize=(24, 9))
    for i, sd in enumerate(AN.SEEDS):
        tb = B['table'][f's{sd}']
        for k, a in enumerate(AN.ARMS_B):
            ax[i, 0].plot(AN.VARIANTS, [tb[a][f'{v}|binerr']['panel_facing'] for v in AN.VARIANTS], 'o-', label=f'{a} panel-facing bins')
            ax[i, 0].plot(AN.VARIANTS, [tb[a][f'{v}|binerr']['other'] for v in AN.VARIANTS], 'x--', label=f'{a} other bins')
        ax[i, 0].set_title(f's{sd}: B patch-bin rel L1 at box-A patches', fontsize=8); ax[i, 0].legend(fontsize=5)
        for k, a in enumerate(('C0', 'C2_rcv')):
            ax[i, 1].plot(AN.TEX, [tb[a][f'{v}|B|T_panel_face']['amplitude'] for v in AN.TEX], 'o-', label=f'{a} panel face')
            ax[i, 1].plot(AN.TEX, [tb[a][f'{v}|B|T_other_faces']['amplitude'] for v in AN.TEX], 'x--', label=f'{a} other faces')
            ax[i, 1].plot(AN.TEX, [tb[a][f'{v}|B|panel_facing_bins']['amplitude'] for v in AN.TEX], 's:', label=f'{a} panel-facing bins')
        ax[i, 1].axhline(1, color='k', lw=0.8); ax[i, 1].set_title(f's{sd}: B amplitude by face / bin group', fontsize=8); ax[i, 1].legend(fontsize=5)
        ss = tb['sender_split']
        nm = list(ss)
        ax[i, 2].bar(nm, [ss[n]['baseline_err_T'] for n in nm]); ax[i, 2].set_title(f's{sd}: untextured single-flight error by sender (vs e2_p / e2_o)', fontsize=8)
        ax[i, 2].tick_params(axis='x', labelsize=6, rotation=20)
        for a in ('C0', 'C2_rcv', 'C2_srcS1'):
            ax[i, 3].plot(AN.TEX, [tb[a][f'{v}|receivers|Q_S1']['amplitude'] for v in AN.TEX], 'o-', label=f'{a} S1 receivers')
            ax[i, 3].plot(AN.TEX, [tb[a][f'{v}|receivers|Q_nonS1']['amplitude'] for v in AN.TEX], 'x--', label=f'{a} non-S1 receivers')
        ax[i, 3].axhline(1, color='k', lw=0.8); ax[i, 3].set_title(f's{sd}: response of OTHER patches to the box material (box as sender)', fontsize=8)
        ax[i, 3].legend(fontsize=5)
    fn = os.path.join(outd, 'fig2_stageB_direction_sender_receivers.png'); fig.savefig(fn, dpi=80, bbox_inches='tight'); plt.close(fig); files.append(fn)

    # ---- fig 3: image sheets per variant (frozen arms, and candidates when present)
    for v in AN.VARIANTS:
        cols = [(mt, a) for mt, _ in models for a in ('C0', 'C2_srcS1', 'C2_rcv') if (mt, v, a) in hl]
        if not cols:
            continue
        fig, ax = plt.subplots(3, len(cols) + 1, figsize=(3.3 * (len(cols) + 1), 10))
        ax[0, 0].imshow(tm(href[v][cy, cx], key)); ax[0, 0].set_title(f'REF h  [{v}]', fontsize=7)
        ax[1, 0].axis('off')
        if v != 'none':
            sq(ax[2, 0], AN.lum(href[v] - href['none'])[cy, cx], D_MAX, plt, 'REF response')
        else:
            ax[2, 0].axis('off')
        for j, (mt, a) in enumerate(cols, 1):
            x = hl[(mt, v, a)]
            ax[0, j].imshow(tm(x[cy, cx], key)); ax[0, j].set_title(f'{mt} {a}: hl', fontsize=7)
            sq(ax[1, j], AN.lum(x - href[v])[cy, cx], E_MAX, plt, 'signed error hl - h')
            if v != 'none' and (mt, 'none', a) in hl:
                sq(ax[2, j], AN.lum(x - hl[(mt, 'none', a)])[cy, cx], D_MAX, plt, 'texture response')
            else:
                ax[2, j].axis('off')
        for a_ in ax[0]:
            a_.set_xticks([]); a_.set_yticks([])
        fig.suptitle(f'WL94 CK2 {v}: {VLAB[v]} - higher-order term close-up; shared absolute scales (error max {E_MAX:.3g}, response max {D_MAX:.3g})', fontsize=10)
        fn = os.path.join(outd, f'fig3_images_{v}.png'); fig.savefig(fn, dpi=75, bbox_inches='tight'); plt.close(fig); files.append(fn)

    # ---- fig 4 / 5: Stage C
    for c in Cs:
        sd = c['seed']
        fig, ax = plt.subplots(1, 4, figsize=(22, 4.5))
        for cand in 'AB':
            j = AN.jl(os.path.join(R, 'results', 'models', f'{cand}_s{sd}.json'))
            st = [r['step'] for r in j['log']]
            for k, nm in enumerate(('L_sum', 'L_single', 'L_global')):
                ax[k].plot(st, [r[nm] for r in j['log']], label=cand); ax[k].set_title(f's{sd} {nm}', fontsize=8); ax[k].set_yscale('log')
        f = c['FIT']['per_candidate']
        ax[3].bar(['A init', 'A final', 'B init', 'B final'], [f['A']['val_init'], f['A']['val_final'], f['B']['val_init'], f['B']['val_final']])
        ax[3].set_title(f'V_orig whole rel L1 (F2 limit 1.10 x init); FIT {c["FIT"]["pass"]}', fontsize=8)
        for a_ in ax[:3]:
            a_.legend(fontsize=7)
        fn = os.path.join(outd, f'fig4_stageC_training_s{sd}.png'); fig.savefig(fn, dpi=80, bbox_inches='tight'); plt.close(fig); files.append(fn)
        fig, ax = plt.subplots(1, 3, figsize=(20, 5))
        for k, comp in enumerate(('B', 'S', 'G')):
            arm = 'C0'
            fz = B['table'][f's{sd}']['C0']
            ax[k].plot(AN.TEX, [fz[f'{v}|{comp}|T']['amplitude'] for v in AN.TEX], 'k^-', label='frozen own (init)')
            for cand, mk in (('A', 'o-'), ('B', 's-')):
                tb = c['models'][cand]['table'][arm]
                ax[k].plot(AN.TEX, [tb[f'{v}|{comp}|T']['amplitude'] for v in AN.TEX], mk, label=f'smoke {cand}')
            ax[k].axhline(1, color='k', lw=0.8); ax[k].set_title(f's{sd}: CK2 box-A {comp} amplitude (C0)', fontsize=8); ax[k].legend(fontsize=7)
        pop = ' | '.join(f"{nm}: init {c['pop_init'][f'B|{nm}']['amplitude']:.2f} A {c['models']['A']['pop'][f'B|{nm}']['amplitude']:.2f} B {c['models']['B']['pop'][f'B|{nm}']['amplitude']:.2f}"
                         for nm in ('lambert', 'S1'))
        fig.suptitle(f'V_ck population B amplitude ({pop})', fontsize=9)
        fn = os.path.join(outd, f'fig5_stageC_amplitude_s{sd}.png'); fig.savefig(fn, dpi=80, bbox_inches='tight'); plt.close(fig); files.append(fn)
    return files
