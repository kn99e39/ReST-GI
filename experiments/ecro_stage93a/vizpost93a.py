"""WL93A post hoc visual: per-geometry dM (TQ - T0) vs the B5 target noise of the geometry. python vizpost93a.py <results_dir> <out>"""
import hashlib, json, os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    R, out = sys.argv[1], sys.argv[2]
    D = json.load(open(os.path.join(R, 'posthoc93a.json')))
    rows = D['groups']['val']['per_geometry']
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.3))
    for k, s in enumerate((8601, 8602)):
        n = np.array([r['N_B5'] for r in rows]); d = np.array([r[f'M_TQ_s{s}'] - r[f'M_T0_s{s}'] for r in rows])
        dl = np.array([r[f'l1_TQ_s{s}'] - r[f'l1_T0_s{s}'] for r in rows])
        ax[0].scatter(n, d, s=14, label=f's{s}', alpha=0.8)
        ax[1].scatter(n, dl, s=14, label=f's{s}', alpha=0.8)
        for r, x, y, z in zip(rows, n, d, dl):
            if x > 0.1:
                ax[0].annotate(r['id'][:6], (x, y), fontsize=7); ax[1].annotate(r['id'][:6], (x, z), fontsize=7)
    for a_, t in zip(ax, ('per geometry dM = M(TQ) - M(T0) (debiased rel L2)', 'per geometry d relative L1 (TQ - T0)')):
        a_.axhline(0, color='k', lw=0.6); a_.axvline(0.10, color='r', ls=':', lw=0.8, label='high-confidence bound (B5 noise 0.10)')
        a_.set_xscale('log'); a_.set_xlabel('B5 single-realization target noise of the geometry'); a_.grid(alpha=0.25); a_.legend(fontsize=7)
        a_.set_title('POST HOC, validation: ' + t, fontsize=9)
    fig.tight_layout(); fn = 'fig4_posthoc_per_geometry_dM.png'; fig.savefig(os.path.join(out, fn), dpi=130); plt.close(fig)
    mf = os.path.join(out, 'manifest.json'); man = json.load(open(mf))
    man['files'][fn] = hashlib.sha256(open(os.path.join(out, fn), 'rb').read()).hexdigest()
    man['provenance'] += '; fig4 (POST HOC, labelled) from experiments/ecro_stage93a/vizpost93a.py on posthoc93a.json'
    json.dump(man, open(mf, 'w'), indent=1)
    print('VIZ', fn)


if __name__ == '__main__':
    main()
