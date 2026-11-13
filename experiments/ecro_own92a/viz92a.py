"""WL92A reviewer visuals. python viz92a.py <R> <R86> <out>
Error / difference maps share ONE absolute scale per figure (sqrt colour map, colour bar in radiance units)."""
import hashlib, json, os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', 'ecro_attr92'))
import analyze92 as AN  # noqa: E402
Y = lambda x: x.mean(-1)
COL = {'ecro': '#7a52a8', 'sum': '#93a1a1', 'own': '#2aa198'}


def col(n):
    return COL[n.split('_')[0]]


def sq_cbar(fig, im, ax, vmax, signed, label):
    cb = fig.colorbar(im, ax=ax, fraction=0.015)
    tk = np.array([-1, -0.25, 0, 0.25, 1.0]) if signed else np.array([0, 0.05, 0.25, 1.0])
    cb.set_ticks(np.sign(tk) * np.sqrt(np.abs(tk))); cb.set_ticklabels([f'{v * vmax:.2g}' for v in tk]); cb.set_label(label)


def main():
    R, R86, out = sys.argv[1], sys.argv[2], sys.argv[3]
    os.makedirs(out, exist_ok=True)
    F = json.load(open(os.path.join(R, 'results', 'final92a.json')))
    Ms = F['models']; NM = list(Ms)
    made = []
    lvl = F.get('thresholds', {}).get('EH_region_max', 0.06)
    # ---- fig1: region-C and stationary E_H per model and condition
    conds = [(e, t) for e in ('E1', 'E2', 'E4') for t in ('T0', 'T1')]
    fig, ax = plt.subplots(1, 2, figsize=(15, 4.4))
    w = 0.8 / max(len(NM), 1)
    for j, (k, a_) in enumerate((('C', ax[0]), ('stationary_all', ax[1]))):
        for i, n in enumerate(NM):
            v = [Ms[n]['fidelity'].get(e, {}).get(t, {}).get(k, {}).get('E_H', np.nan) for e, t in conds]
            a_.bar(np.arange(len(conds)) + i * w, v, w, color=col(n), alpha=0.55 + 0.45 * (i % 2), label=n)
        a_.axhline(lvl if k == 'C' else 0.03, color='r', ls='--', lw=0.8)
        a_.set_xticks(np.arange(len(conds)) + 0.4); a_.set_xticklabels([f'{e} {t}' + (' (held-out)' if e == 'E4' else '') for e, t in conds], fontsize=7)
        a_.set_title(f'E_H in {"region C (sharp readers)" if k == "C" else "stationary pixels"}; dashed: WL86 rule', fontsize=9); a_.grid(alpha=0.25, axis='y')
    ax[0].legend(fontsize=7, ncol=3)
    fig.tight_layout(); fn = 'fig1_region_C_and_stationary_EH.png'; fig.savefig(os.path.join(out, fn), dpi=130); plt.close(fig); made.append(fn)
    # ---- fig2: component attribution in region C
    tags = [t for t in ('E2_T1', 'E2_T0', 'E4_T1', 'E4_T0') if any(t in Ms[n]['attribution'] for n in NM)]
    fig, ax = plt.subplots(1, len(tags), figsize=(5.2 * len(tags), 4.0), squeeze=False)
    for i, tag in enumerate(tags):
        rows = []
        tr = None
        for n in NM:
            a = Ms[n]['attribution'].get(tag, {}).get('C')
            if not a:
                continue
            tr = a['truth_panel_family_share']
            rows.append((n, a['panel_family_delivered'], a['single_minus_L2_over_h'], a['global_minus_L3_over_h']))
        yy = np.arange(len(rows))[::-1]
        for k, (lab, c_) in enumerate((('panel family delivered (1 = truth)', '#d6392a'), ('single - L2 (of h)', '#e6a02a'), ('global - L3+ (of h)', '#2a78d6'))):
            ax[0, i].barh(yy + (k - 1) * 0.27, [r[1 + k] for r in rows], 0.27, color=c_, label=lab)
        ax[0, i].axvline(0, color='k', lw=0.6); ax[0, i].axvline(1, color='k', ls=':', lw=0.6)
        ax[0, i].set_yticks(yy); ax[0, i].set_yticklabels([r[0] for r in rows], fontsize=7)
        ax[0, i].set_title(f'{tag} region C (truth panel-family share {tr:.2f})' if tr is not None else tag, fontsize=8); ax[0, i].grid(alpha=0.25, axis='x')
        if i == 0:
            ax[0, i].legend(fontsize=6, loc='lower right')
    fig.tight_layout(); fn = 'fig2_component_attribution_C.png'; fig.savefig(os.path.join(out, fn), dpi=130); plt.close(fig); made.append(fn)
    # ---- fig3: training curves
    cur = F.get('training_curves') or {}
    fig, ax = plt.subplots(1, 3, figsize=(15, 3.8))
    for n, lg in cur.items():
        if not lg:
            continue
        st = [r['step'] for r in lg]
        for a_, k in zip(ax, ('L_sum', 'L_single', 'L_global')):
            a_.semilogy(st, [r[k] for r in lg], color=col(n), ls='-' if n.endswith('8601') else '--', label=n)
    for a_, k in zip(ax, ('whole-table term (both objectives)', 'single-flight vs L2 (supervised by OWN only)', 'global vs L3+ (supervised by OWN only)')):
        a_.set_title(k, fontsize=9); a_.set_xlabel('step'); a_.grid(alpha=0.25)
    ax[0].legend(fontsize=7)
    fig.tight_layout(); fn = 'fig3_training_curves.png'; fig.savefig(os.path.join(out, fn), dpi=130); plt.close(fig); made.append(fn)
    # ---- fig4 / fig5: held-out E4 sharp readers and signed edit response
    if 'E4' in Ms[NM[0]]['fidelity']:
        M = AN.mask_of(R, R86, 'E4_T1')
        ref1 = AN.ref_of(R, R86, 'E4_T1'); ref0 = AN.ref_of(R, R86, 'E4_T0')
        h1 = Y(ref1['h'].mean(0)); h0 = Y(ref0['h'].mean(0))
        m = M['C'] | M['obj_panel']
        ys, xs = np.nonzero(m)
        sl = (slice(max(ys.min() - 20, 0), ys.max() + 20), slice(max(xs.min() - 20, 0), xs.max() + 20)) if len(ys) else (slice(None), slice(None))
        show = [n for n in NM if n.endswith('8601')]
        imgs = [(n, Y(AN.hl(R, f'I92A_{n}_E4_T1')['hl'].mean(0))) for n in show] + [(k, Y(AN.hl(R, f'I92A_{k}_E4_T1')['hl'].mean(0))) for k in ('ORACLE', 'X_EXP') if AN.hl(R, f'I92A_{k}_E4_T1') is not None]
        vmax = float(np.percentile(h1[sl], 99.5)); emax = max(float(np.abs(x - h1)[sl].max()) for _, x in imgs)
        fig, ax = plt.subplots(2, len(imgs) + 1, figsize=(3.0 * (len(imgs) + 1), 6.2), squeeze=False)
        ax[0, 0].imshow(np.sqrt(np.clip(h1[sl] / vmax, 0, 1)), cmap='gray'); ax[0, 0].set_title('REF-ER higher-order, E4 T1', fontsize=8); ax[1, 0].axis('off')
        for j, (lab, x) in enumerate(imgs):
            ax[0, j + 1].imshow(np.sqrt(np.clip(x[sl] / vmax, 0, 1)), cmap='gray'); ax[0, j + 1].set_title(lab, fontsize=8)
            d = (x - h1)[sl]
            im = ax[1, j + 1].imshow(np.sign(d) * np.sqrt(np.abs(d) / emax), cmap='RdBu_r', vmin=-1, vmax=1); ax[1, j + 1].set_title('signed error', fontsize=8)
        for a_ in ax.ravel():
            a_.set_xticks([]); a_.set_yticks([])
        sq_cbar(fig, im, ax[1, 1:], emax, True, 'radiance (sqrt scale)')
        fig.suptitle('held-out E4 T1: sharp readers (panel / region C crop); error scale shared', fontsize=9)
        fn = 'fig4_heldout_E4_sharp_maps.png'; fig.savefig(os.path.join(out, fn), dpi=130, bbox_inches='tight'); plt.close(fig); made.append(fn)
        dref = h1 - h0
        ds = [(n, Y(AN.hl(R, f'I92A_{n}_E4_T1')['hl'].mean(0)) - Y(AN.hl(R, f'I92A_{n}_E4_T0')['hl'].mean(0))) for n in show]
        vm = max([float(np.abs(dref).max())] + [float(np.abs(d).max()) for _, d in ds])
        fig, ax = plt.subplots(1, len(ds) + 1, figsize=(3.6 * (len(ds) + 1), 3.9), squeeze=False)
        for j, (lab, d) in enumerate([('REF-ER', dref)] + ds):
            im = ax[0, j].imshow(np.sign(d) * np.sqrt(np.abs(d) / vm), cmap='RdBu_r', vmin=-1, vmax=1); ax[0, j].set_title(f'{lab}: T1 - T0', fontsize=8)
            ax[0, j].contour(M['C'], levels=[0.5], colors='k', linewidths=0.4); ax[0, j].set_xticks([]); ax[0, j].set_yticks([])
        sq_cbar(fig, im, ax[0, :], vm, True, 'radiance (sqrt scale)')
        fig.suptitle('held-out E4: signed change of the higher-order term (contour: region C); scale shared', fontsize=9)
        fn = 'fig5_heldout_E4_signed_response.png'; fig.savefig(os.path.join(out, fn), dpi=130, bbox_inches='tight'); plt.close(fig); made.append(fn)
    # ---- fig6: learned panel local matrix
    L = F.get('local')
    if L and 'E4_T1' in L:
        fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
        for i, k in enumerate(('peak', 'fro')):
            ax[i].bar(range(len(NM)), [L['E4_T1'][n]['1'][k] for n in NM], color=[col(n) for n in NM])
            ax[i].set_xticks(range(len(NM))); ax[i].set_xticklabels(NM, fontsize=7, rotation=20)
            ax[i].set_title('panel a c^T: mirror-bin flux retention (1 = frozen ER)' if k == 'peak' else 'panel a c^T: relative error (flux norm)', fontsize=9); ax[i].grid(alpha=0.25, axis='y')
        fig.tight_layout(); fn = 'fig6_panel_local_matrix.png'; fig.savefig(os.path.join(out, fn), dpi=130); plt.close(fig); made.append(fn)
    # ---- fig7: sources
    tags = sorted({t for n in NM for t in Ms[n]['sources']})
    if tags:
        fig, ax = plt.subplots(figsize=(1.8 * len(tags) + 3, 3.8))
        for i, n in enumerate(NM):
            ax.bar(np.arange(len(tags)) + i * w, [Ms[n]['sources'].get(t, {}).get('stationary_all', {}).get('E_H', np.nan) for t in tags], w, color=col(n), alpha=0.55 + 0.45 * (i % 2), label=n)
        ax.axhline(0.04, color='r', ls='--', lw=0.8); ax.set_xticks(np.arange(len(tags)) + 0.4); ax.set_xticklabels(tags, fontsize=8)
        ax.set_title('unseen sources: stationary E_H (rule F4 0.04)', fontsize=9); ax.legend(fontsize=7, ncol=3); ax.grid(alpha=0.25, axis='y')
        fig.tight_layout(); fn = 'fig7_unseen_sources.png'; fig.savefig(os.path.join(out, fn), dpi=130); plt.close(fig); made.append(fn)
    man = {'worklog': '92A', 'source_root': R, 'files': {f: hashlib.sha256(open(os.path.join(out, f), 'rb').read()).hexdigest() for f in made},
           'provenance': 'experiments/ecro_own92a/viz92a.py from results/ecro_own92a (final92a.json, renders) and the WL86 / WL92A REF-ER renders and masks'}
    json.dump(man, open(os.path.join(out, 'manifest.json'), 'w'), indent=1)
    print('VIZ', made, flush=True)


if __name__ == '__main__':
    main()
