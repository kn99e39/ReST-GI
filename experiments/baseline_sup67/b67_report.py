"""WL67 report-only exports (no sampling/fitting): quality-vs-update-latency figures, contention screening, file manifest.

python b67_report.py <server root ~/rest_bsup67> <export dir>
"""
import hashlib, json, re, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

LADDER = (1, 4, 16, 64, 256, 1024)


def contention(root):
    """Foreign GPU processes (not this batch's root) seen by the 15 s sampler during each stage interval."""
    steps = [l.split() for l in open(root / 'launcher/steps.log') if ' start ' in l or ' end ' in l]
    iv = {}
    for s in steps:
        iv.setdefault(s[2], {})[s[1]] = s[0]
    foreign = {}
    for l in open(root / 'launcher/gpu_samples.log'):
        f = l.split(maxsplit=3)
        if len(f) < 3 or f[1] == 'tick': continue
        if 'rest_bsup67' in l or 'baseline_sup67' in l or 'fnto_sel56' in l: continue
        for name, se in iv.items():
            if 'start' in se and se['start'] <= f[0] <= se.get('end', '9999'):
                foreign.setdefault(name, []).append(l.strip()[:220])
    return dict(intervals=iv, foreign_samples={k: dict(n=len(v), first=v[:3]) for k, v in foreign.items()})


def main():
    root, ex = Path(sys.argv[1]), Path(sys.argv[2]); ex.mkdir(parents=True, exist_ok=True)
    w1 = json.load(open(root / 'out/analysis/w1_analysis.json')); w2 = json.load(open(root / 'out/analysis/w2_analysis.json'))
    con = contention(root); json.dump(con, open(ex / 'contention.json', 'w'), indent=1)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    # W1: dent, max |group bias| and passes vs edit-to-output latency (four primary views)
    for r, mk in ((0, 'o'), (1, 's')):
        xs = [np.median(w1['cycles_timing'][f'dent|{s}']['update_seconds']) for s in LADDER]
        ys = [w1['cycles'][f'dent|{s}|r{r}']['M2_max_abs_bias'] * 100 for s in LADDER]
        ax[0].plot(xs, ys, mk + '-', color='#3b6ea5', label=f'Cycles Full Recompute (r{r})' if True else None)
        for s, x, y in zip(LADDER, xs, ys):
            if r == 0: ax[0].annotate(f'{s}', (x, y), textcoords='offset points', xytext=(3, 3), fontsize=7)
    t = w1['fnto_timing']
    for arm, key, col in (('F', 'dent_F_r0', '#b5462f'), ('S', 'dent_S_r0', '#d98a2b'), ('T (stale, bin)', 'dent_T_r0', '#777777')):
        k = 'dent|' + arm[0] + ('|bin' if arm.startswith('T') else '|nee_ball')
        y = max(w1['fnto'][k][r]['M2_max_abs_bias'] for r in ('r0', 'r1')) * 100
        ax[0].scatter([t[key]['total_update_seconds']], [y], color=col, zorder=3, label=f'ReST-GI oracle {arm}')
    ax[0].axhline(3, ls=':', color='k', lw=0.8); ax[0].set_xscale('log'); ax[0].set_yscale('log')
    ax[0].set_xlabel('edit-to-output latency, 4 face-on views (s, LabServer69 RTX 3090)'); ax[0].set_ylabel('max |group bias| over 12 primary tests (%)')
    ax[0].set_title('W1 two-slab medium, dent'); ax[0].legend(fontsize=7)
    # W2: interaction-ROI display MAE vs latency
    for r, mk in ((0, 'o'), (1, 's')):
        xs = [w2['timing'][f'T3|{s}']['median'] for s in LADDER]; ys = [w2['scores'][f'PT|{s}|r{r}']['refit_T3_interaction_error'] for s in LADDER]
        ax[1].plot(xs, ys, mk + '-', color='#3b6ea5', label=f'Mitsuba PT Full Recompute (r{r})')
    inh = w2['inherited']
    pts = {'RNA frozen (0 s update, invalid)': ('RNA_frozen', None), '8DNA frozen (0 s update, invalid)': ('8DNA_frozen_WL22', None),
           'RNA refit (WL27, RTX 5080)': ('RNA_refit', inh['rebuild_rna_full_schedule_s']), '8DNA refit (WL27, RTX 5080)': ('8DNA_refit', inh['rebuild_8dna_full_schedule_s'])}
    for lab, (arm, sec) in pts.items():
        y = w2['scores'][arm]['refit_T3_interaction_error']
        if sec is None: ax[1].axhline(y, ls='--', lw=0.8, color='#b5462f' if 'RNA' in lab else '#6a8d3a', label=lab)
        else: ax[1].scatter([sec], [y], color='#b5462f' if 'RNA' in lab else '#6a8d3a', marker='D', label=lab)
    ax[1].set_xscale('log'); ax[1].set_xlabel('edit-to-output latency (s)'); ax[1].set_ylabel('interaction-ROI display MAE vs GT (T3)')
    ax[1].set_title('W2 teaset common light, T0 -> T3'); ax[1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(ex / '01_quality_vs_latency.png', dpi=150); plt.close(fig)
    man = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(ex.iterdir()) if p.is_file() and p.name != 'manifest.json'}
    json.dump(dict(source_root=str(root), generator='experiments/baseline_sup67/b67_report.py', files=man), open(ex / 'manifest.json', 'w'), indent=1)
    print(json.dumps(con['foreign_samples'], indent=1))


if __name__ == '__main__':
    main()
