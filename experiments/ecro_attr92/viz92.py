"""WL92 reviewer visuals. python viz92.py <R> <R86> <out>
Error and difference maps share ONE absolute scale per figure (sqrt colour map, colour bar in radiance units)."""
import glob, hashlib, json, os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import analyze92 as AN  # noqa: E402
A = AN.A
Y = lambda x: x.mean(-1)
COL = {'ORA2': '#444444', 'L_RS': '#1b7837', 'L_EXP': '#2a78d6', 'L_ENCQ': '#6baed6', 'L_ENCB': '#e6862a', 'L_R16': '#d6392a', 'P_EXP': '#2a78d6', 'P_ENCB': '#e6862a', 'P_R16': '#d6392a', 'X_EXP': '#888888'}


def crop(M):
    m = M['obj_panel'] | M['C']
    ys, xs = np.nonzero(m)
    if not len(ys):
        return slice(None), slice(None)
    return slice(max(ys.min() - 20, 0), ys.max() + 20), slice(max(xs.min() - 20, 0), xs.max() + 20)


def main():
    R, R86, out = sys.argv[1], sys.argv[2], sys.argv[3]
    os.makedirs(out, exist_ok=True)
    F = AN.jl(os.path.join(R, 'results', 'final92.json'))
    C = F['conditions']
    made = []
    lvl = AN.TH['EH_region_max']
    # ---- fig1: ladder of the 2-event term (+ path-traced L3+), frozen operators, fits: region C
    tags = [t for t in AN.HIST if t in C]
    fig, ax = plt.subplots(1, len(tags), figsize=(6.2 * len(tags), 4.6), squeeze=False)
    for i, tag in enumerate(tags):
        r = C[tag]['rendered']
        names, vals, cols = [], [], []
        for k, lab in (('ORA2', 'path-traced L2'), ('L_RS', 'ray-exact (rays only)'), ('L_EXP', 'exact q, 64 bin'), ('L_ENCQ', 'q from encoding'), ('L_ENCB', 'ECRO form, R 64'), ('L_R16', 'ECRO form, R 16 (NMF)'),
                       ('P_EXP', 'pooled: exact q'), ('P_ENCB', 'pooled: R 64'), ('P_R16', 'pooled: R 16')):
            if k in r and 'C' in r[k]:
                names.append(lab); vals.append(r[k]['C']['E_H']); cols.append(COL[k])
        for sd in AN.SEEDS:
            for k, lab in ((f'SL_{sd}', 'learned single + L3 truth'), (f'LG_{sd}', 'L2 truth + learned global'), (f'B_{sd}', 'frozen ECRO')):
                if k in r and 'C' in r[k]:
                    names.append(f'{lab} ({sd[-4:]})'); vals.append(r[k]['C']['E_H']); cols.append('#7a52a8' if k.startswith('B_') else '#b39bd0')
        for v, row in F['fits'].get(tag, {}).items():
            for key, lab in (('SL', 'single + L3 truth'), ('B', 'table')):
                if key in row['rendered'] and 'C' in row['rendered'][key]:
                    names.append(f'fit {v}: {lab}'); vals.append(row['rendered'][key]['C']['E_H']); cols.append('#2aa198' if v.startswith('OWN') else '#93a1a1')
        yy = np.arange(len(names))[::-1]
        ax[0, i].barh(yy, vals, color=cols)
        ax[0, i].set_yticks(yy); ax[0, i].set_yticklabels(names, fontsize=7)
        ax[0, i].axvline(lvl, color='r', ls='--', lw=0.8)
        ax[0, i].set_title(f'{tag}: E_H in region C (rule {lvl})', fontsize=9); ax[0, i].grid(alpha=0.25, axis='x')
    fig.tight_layout(); fn = 'fig1_ladder_region_C.png'; fig.savefig(os.path.join(out, fn), dpi=130); plt.close(fig); made.append(fn)
    # ---- fig2: local operator of the sharp panel
    nmz = os.path.join(R, 'results', 'nmf.npz')
    if os.path.exists(nmz) and F.get('local'):
        z = np.load(nmz); Mq = z['Mq']; Cn, An = z['Cn'], z['An']
        fig, ax = plt.subplots(1, 4, figsize=(18, 4.2))
        for m, (nm, col) in enumerate((('B_floor', '#2a78d6'), ('S1_panel', '#d6392a'), ('S2_sphere', '#1b7837'))):
            sv = np.array(F['local']['materials'][nm]['1']['singular_values'])
            ax[0].semilogy(np.arange(1, len(sv) + 1), sv / sv[0], color=col, label=nm)
        ax[0].axvline(16, color='k', ls=':', lw=0.8); ax[0].set_ylim(1e-4, 1.2); ax[0].legend(fontsize=7); ax[0].set_title('singular values of M64 (green channel), rank 16 marked', fontsize=9); ax[0].grid(alpha=0.25)
        V = Mq[1, :, :, 1]; Vn = Cn[1, 1] @ An[1, 1].T
        vm = float(V.max())
        for a_, X, t in ((ax[1], V, 'S1 panel: M64 (incident bin -> outgoing bin)'), (ax[2], Vn, 'rank-16 NMF'), (ax[3], np.abs(Vn - V), '|difference|')):
            im = a_.imshow(np.sqrt(np.clip(X, 0, None) / vm), cmap='magma', vmin=0, vmax=1); a_.set_title(t, fontsize=9); a_.set_xlabel('outgoing bin'); a_.set_ylabel('incident bin')
        cb = fig.colorbar(im, ax=ax[1:], fraction=0.02); tk = np.array([0, 0.05, 0.2, 0.5, 1.0]); cb.set_ticks(np.sqrt(tk)); cb.set_ticklabels([f'{v * vm:.2g}' for v in tk]); cb.set_label('sr^-1 (sqrt scale)')
        fn = 'fig2_local_operator_panel.png'; fig.savefig(os.path.join(out, fn), dpi=130, bbox_inches='tight'); plt.close(fig); made.append(fn)
    # ---- fig3: higher-order term on the sharp readers and its error (one shared scale), per condition
    for tag in tags:
        M = AN.mask_of(R, R86, tag); ref = AN.ref_of(R, R86, tag)
        h = Y(ref['h'].mean(0))
        ys, xs = crop(M)
        cols = [('ORA2', 'path-traced L2 + L3'), ('L_EXP', 'exact q through rays'), ('L_ENCB', 'ECRO form R 64'), ('L_R16', 'ECRO form R 16'), ('SL_ecro_s8601', 'learned single + L3 truth'), ('B_ecro_s8601', 'frozen ECRO')]
        imgs = [(lab, Y(AN.hl(R, f'X92_{k}_{tag}')['hl'].mean(0))) for k, lab in cols if AN.hl(R, f'X92_{k}_{tag}') is not None]
        if not imgs:
            continue
        vmax = float(np.percentile(h[ys, xs], 99.5)); emax = max(float(np.abs(x - h)[ys, xs].max()) for _, x in imgs)
        fig, ax = plt.subplots(2, len(imgs) + 1, figsize=(3.0 * (len(imgs) + 1), 6.4), squeeze=False)
        ax[0, 0].imshow(np.sqrt(np.clip(h[ys, xs] / vmax, 0, 1)), cmap='gray'); ax[0, 0].set_title('REF-ER higher-order term', fontsize=8); ax[1, 0].axis('off')
        for j, (lab, x) in enumerate(imgs):
            ax[0, j + 1].imshow(np.sqrt(np.clip(x[ys, xs] / vmax, 0, 1)), cmap='gray'); ax[0, j + 1].set_title(lab, fontsize=8)
            d = (x - h)[ys, xs]
            im = ax[1, j + 1].imshow(np.sign(d) * np.sqrt(np.abs(d) / emax), cmap='RdBu_r', vmin=-1, vmax=1); ax[1, j + 1].set_title('signed error', fontsize=8)
        for a_ in ax.ravel():
            a_.set_xticks([]); a_.set_yticks([])
        cb = fig.colorbar(im, ax=ax[1, 1:], fraction=0.015); tk = np.array([-1, -0.25, 0, 0.25, 1.0]); cb.set_ticks(np.sign(tk) * np.sqrt(np.abs(tk))); cb.set_ticklabels([f'{v * emax:.2g}' for v in tk]); cb.set_label('radiance (sqrt scale)')
        fig.suptitle(f'{tag}: sharp readers (panel / region C crop); error scale shared by all columns', fontsize=9)
        fn = f'fig3_sharp_reader_maps_{tag}.png'; fig.savefig(os.path.join(out, fn), dpi=130, bbox_inches='tight'); plt.close(fig); made.append(fn)
    # ---- fig4: component attribution in region C (fractions of the higher-order reading)
    fig, ax = plt.subplots(1, len(tags), figsize=(6.0 * len(tags), 4.2), squeeze=False)
    for i, tag in enumerate(tags):
        fam = C[tag]['frozen'].get('C')
        if not fam:
            continue
        ts = fam['truth_share']
        rows = [('truth', ts['L2_via_panel'], ts['L2_other'], ts['L3plus'])]
        for sd in AN.SEEDS:
            m = fam['models'][sd]
            sp = m['panel_family_delivered'] * ts['L2_via_panel']
            s_all = m['single_minus_L2_over_h'] + ts['L2_via_panel'] + ts['L2_other']
            rows.append((f'frozen {sd[-4:]}', sp, s_all - sp, m['global_minus_L3_over_h'] + ts['L3plus']))
        for v, row in F['fits'].get(tag, {}).items():
            cr = row['final_eval']['C_read']
            if cr:
                sp = cr['panel_family_delivered'] * ts['L2_via_panel']; s_all = cr['single_minus_L2_over_h'] + ts['L2_via_panel'] + ts['L2_other']
                rows.append((f'fit {v}', sp, s_all - sp, cr['global_minus_L3_over_h'] + ts['L3plus']))
        yy = np.arange(len(rows))[::-1]
        a_ = np.array([[r[1], r[2], r[3]] for r in rows])
        ax[0, i].barh(yy, a_[:, 0], color='#d6392a', label='single flight via panel senders (truth: 2 events via panel)')
        ax[0, i].barh(yy, a_[:, 1], left=a_[:, 0], color='#e6a02a', label='single flight, other senders (truth: 2 events, other)')
        ax[0, i].barh(yy, a_[:, 2], left=a_[:, 0] + a_[:, 1], color='#2a78d6', label='global term (truth: 3-5 events)')
        ax[0, i].axvline(1.0, color='k', ls=':', lw=0.8)
        ax[0, i].set_yticks(yy); ax[0, i].set_yticklabels([r[0] for r in rows], fontsize=7); ax[0, i].set_title(f'{tag}: what region C reads, by term (1 = path-traced higher-order reading)', fontsize=8)
        ax[0, i].grid(alpha=0.25, axis='x')
        if i == 0:
            fig.legend(*ax[0, i].get_legend_handles_labels(), fontsize=7, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.06, 1, 1)); fn = 'fig4_component_attribution.png'; fig.savefig(os.path.join(out, fn), dpi=130); plt.close(fig); made.append(fn)
    # ---- fig5: fit curves and SUM-objective identifiability
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.4))
    for tag in tags:
        for fn_ in sorted(glob.glob(os.path.join(R, 'results', 'fits', f'fit_{tag}_*.json'))):
            r = AN.jl(fn_)
            ax[0].semilogy([c['step'] for c in r['curve']], [c['L_sum'] for c in r['curve']], '-' if r['objective'] == 'SUM' else '--', lw=0.9, label=f"{tag} {r['variant']}")
        idn = F['identifiability'].get(tag)
        if idn:
            for k, v in idn['L_sum_val'].items():
                if k in idn['panel_family_delivered']:
                    ax[1].plot(idn['panel_family_delivered'][k], v, 'o' if k.startswith('frozen') else 's', ms=5)
                    ax[1].annotate(f'{tag[:5]} {k}', (idn['panel_family_delivered'][k], v), fontsize=5)
            for k, v in idn['OWN_points'].items():
                if v['panel_family_delivered'] is not None:
                    ax[1].plot(v['panel_family_delivered'], v['L_sum_val'], '^', ms=5, color='#2aa198'); ax[1].annotate(f'{tag[:5]} {k}', (v['panel_family_delivered'], v['L_sum_val']), fontsize=5)
    ax[0].set_xlabel('step'); ax[0].set_ylabel('SUM objective (fit lights)'); ax[0].legend(fontsize=5, ncol=3); ax[0].grid(alpha=0.25); ax[0].set_title('fixed-geometry free-factor fits (dashed: OWN objective)', fontsize=9)
    ax[1].set_xlabel('share of the panel-sender 2-event family delivered by the single flight (region C)'); ax[1].set_ylabel('SUM objective, validation lights'); ax[1].grid(alpha=0.25)
    ax[1].set_title('same objective value, different allocation?', fontsize=9)
    fig.tight_layout(); fn = 'fig5_fits_identifiability.png'; fig.savefig(os.path.join(out, fn), dpi=130); plt.close(fig); made.append(fn)
    # ---- fig6: T0 / T1 signed response of the higher-order term (E2), one shared scale
    if 'E2_T0' in C and 'E2_T1' in C:
        M = AN.mask_of(R, R86, 'E2_T1')
        r0, r1 = AN.ref_of(R, R86, 'E2_T0'), AN.ref_of(R, R86, 'E2_T1')
        dref = Y(r1['h'].mean(0)) - Y(r0['h'].mean(0))
        cols = [('ORA2', 'path-traced table'), ('L_ENCB', 'ECRO form R 64'), ('L_R16', 'ECRO form R 16'), ('B_ecro_s8601', 'frozen ECRO')]
        names = AN.SEEDS
        if F.get('intervention'):
            cols += [(None, n) for n in F['intervention']['effect']['own']]
        ds = []
        for k, lab in cols:
            a0 = AN.hl(R, f'X92_{k}_E2_T0') if k else AN.hl(R, f'I92_{lab}_E2_T0'); a1 = AN.hl(R, f'X92_{k}_E2_T1') if k else AN.hl(R, f'I92_{lab}_E2_T1')
            if a0 is not None and a1 is not None:
                ds.append((lab, Y(a1['hl'].mean(0)) - Y(a0['hl'].mean(0))))
        vm = max(float(np.abs(dref).max()), max((float(np.abs(d).max()) for _, d in ds), default=0.0))
        fig, ax = plt.subplots(1, len(ds) + 1, figsize=(3.6 * (len(ds) + 1), 3.9), squeeze=False)
        for j, (lab, d) in enumerate([('REF-ER', dref)] + ds):
            im = ax[0, j].imshow(np.sign(d) * np.sqrt(np.abs(d) / vm), cmap='RdBu_r', vmin=-1, vmax=1); ax[0, j].set_title(f'{lab}: T1 - T0', fontsize=8); ax[0, j].set_xticks([]); ax[0, j].set_yticks([])
            ax[0, j].contour(M['C'], levels=[0.5], colors='k', linewidths=0.4)
        cb = fig.colorbar(im, ax=ax[0, :], fraction=0.012); tk = np.array([-1, -0.25, 0, 0.25, 1.0]); cb.set_ticks(np.sign(tk) * np.sqrt(np.abs(tk))); cb.set_ticklabels([f'{v * vm:.2g}' for v in tk]); cb.set_label('radiance (sqrt scale)')
        fig.suptitle('E2: signed change of the higher-order term (contour: region C); scale shared', fontsize=9)
        fn = 'fig6_signed_response_E2.png'; fig.savefig(os.path.join(out, fn), dpi=130, bbox_inches='tight'); plt.close(fig); made.append(fn)
    # ---- fig7: intervention (only if it ran)
    iv = F.get('intervention')
    if iv:
        ce = iv['effect']['C_E_H']
        keys = sorted({k for v in ce.values() for k in v if v[k] is not None})
        fig, ax = plt.subplots(figsize=(1.6 * len(keys) + 3, 4.2))
        w = 0.8 / max(len(ce), 1)
        for i, (n, v) in enumerate(ce.items()):
            ax.bar(np.arange(len(keys)) + i * w, [v.get(k) or 0 for k in keys], w, label=n, color='#2aa198' if n.startswith('own') else '#93a1a1' if n.startswith('ctl') else '#7a52a8', alpha=0.6 + 0.4 * (i % 2))
        ax.axhline(lvl, color='r', ls='--', lw=0.8); ax.set_xticks(np.arange(len(keys)) + 0.4); ax.set_xticklabels(keys, fontsize=8); ax.set_ylabel('E_H in region C'); ax.legend(fontsize=7); ax.grid(alpha=0.25, axis='y')
        ax.set_title('original contract (frozen, ctl) against component ownership (own)', fontsize=9)
        fig.tight_layout(); fn = 'fig7_intervention_region_C.png'; fig.savefig(os.path.join(out, fn), dpi=130); plt.close(fig); made.append(fn)
    man = {'worklog': 92, 'source_root': R, 'files': {f: hashlib.sha256(open(os.path.join(out, f), 'rb').read()).hexdigest() for f in made},
           'provenance': 'experiments/ecro_attr92/viz92.py from results/ecro_attr92 (final92.json, renders, nmf.npz, fit logs) and the WL86 REF-ER renders / masks'}
    json.dump(man, open(os.path.join(out, 'manifest.json'), 'w'), indent=1)
    print('VIZ', made, flush=True)


if __name__ == '__main__':
    main()
