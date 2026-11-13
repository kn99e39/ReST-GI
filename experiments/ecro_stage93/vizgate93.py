"""WL93 identifiability-gate visual (stop path). python vizgate93.py <results_dir> <out>"""
import hashlib, json, os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    R, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    I = json.load(open(os.path.join(R, 'ident.json'))); D = json.load(open(os.path.join(R, 'posthoc_ident.json')))
    made = []
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.6))
    # (a) frozen gate: irreducible vs 0.5 x learned, per seed and set
    labs, irr, thr = [], [], []
    for sd, v in I['val'].items():
        labs.append(f'val s{sd}'); irr.append(v['irreducible_panel_rel_l1']); thr.append(0.5 * v['own_learned_panel_rel_l1'])
    for tag, v in I['E2'].items():
        labs.append(f'{tag}'); irr.append(v['irreducible_panel_rel_l1']); thr.append(0.5 * min(v['own_learned_panel_rel_l1'].values()))
    x = np.arange(len(labs))
    ax[0].bar(x - 0.2, irr, 0.4, color='#2a78d6', label='exact frozen-ER through deployed encoding vs q')
    ax[0].bar(x + 0.2, thr, 0.4, color='#bbbbbb', label='gate bound = 0.5 x WL92A OWN learned Bq vs q')
    for i in range(len(labs)):
        ax[0].text(i - 0.2, irr[i] + 0.01, 'FAIL' if irr[i] > thr[i] else 'pass', ha='center', fontsize=7, color='r' if irr[i] > thr[i] else 'k')
    ax[0].set_xticks(x); ax[0].set_xticklabels(labs, fontsize=8); ax[0].legend(fontsize=7); ax[0].grid(alpha=0.25, axis='y')
    ax[0].set_title('frozen identifiability gate (panel patches, pooled rel L1)', fontsize=9)
    # (b) post hoc per geometry
    r = D['per_geometry']
    e = np.array([g['enc'] for g in r]); o = np.array([0.5 * (g['own8601'] + g['own8602']) for g in r]); qn = np.array([g['q_vs_q2'] for g in r]); w = np.array([g['panel_q_sum'] for g in r])
    sc = ax[1].scatter(o, e, s=8 + 400 * w / w.sum(), c=qn, cmap='viridis', vmin=0, vmax=1, edgecolor='k', linewidth=0.3)
    lim = [0, max(1.05, float(o.max()) + 0.05)]
    ax[1].plot(lim, [0.5 * v for v in lim], 'r--', lw=0.8, label='irreducible = 0.5 x learned')
    for g, ee, oo in zip(r, e, o):
        if ee > 0.6:
            ax[1].annotate(f"g{g['idx']:05d}", (oo, ee), fontsize=7)
    ax[1].set_xlabel('WL92A OWN Bq vs q (mean of seeds)'); ax[1].set_ylabel('enc_q vs q (irreducible term)')
    fig.colorbar(sc, ax=ax[1], fraction=0.04).set_label('q vs independent q (target MC noise)')
    ax[1].legend(fontsize=7); ax[1].grid(alpha=0.25)
    ax[1].set_title('POST HOC: per validation geometry (marker area = share of pooled panel q)', fontsize=9)
    # (c) post hoc: irreducible term vs target noise
    ax[2].scatter(qn, e, s=8 + 400 * w / w.sum(), color='#2a78d6', edgecolor='k', linewidth=0.3)
    ax[2].set_xlabel('q vs independent q (rel L1, panel)'); ax[2].set_ylabel('enc_q vs q (rel L1, panel)'); ax[2].grid(alpha=0.25)
    ax[2].set_title(f"POST HOC: irreducible term vs target noise (corr {np.corrcoef(e, qn)[0, 1]:.2f})", fontsize=9)
    fig.tight_layout(); fn = 'fig1_identifiability_gate.png'; fig.savefig(os.path.join(out, fn), dpi=130); plt.close(fig); made.append(fn)
    man = {'worklog': '93', 'files': {f: hashlib.sha256(open(os.path.join(out, f), 'rb').read()).hexdigest() for f in made},
           'provenance': 'experiments/ecro_stage93/vizgate93.py from results/ecro_stage93/ident.json (frozen gate) and posthoc_ident.json (post hoc descriptive); chain stopped at the gate, no training / renders exist'}
    json.dump(man, open(os.path.join(out, 'manifest.json'), 'w'), indent=1)
    print('VIZ', made)


if __name__ == '__main__':
    main()
