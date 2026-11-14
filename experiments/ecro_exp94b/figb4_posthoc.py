import json, sys, numpy as np, matplotlib, hashlib, os
matplotlib.use('Agg'); import matplotlib.pyplot as plt
R = sys.argv[1]; outd = sys.argv[2]
p = json.load(open(os.path.join(R, 'results/an/posthoc94b.json')))
ck = p['ck2_panel_face']
best = np.array([r['best_pm_of_face'] for r in p['rows']]); bA = np.array([r['boxA_best_pm_of_face'] for r in p['rows']])
fig, ax = plt.subplots(1, 2, figsize=(15, 4.8))
bins = np.linspace(0, 1, 26)
ax[0].hist(best, bins, alpha=0.6, label='best box face (box A or B)'); ax[0].hist(bA, bins, alpha=0.6, label='best box-A face')
ax[0].axvline(ck['pm_of_face'], color='r', ls='--', label=f"CK2 panel-facing face ({ck['pm_of_face']:.2f})")
ax[0].axvline(0.5 * ck['pm_of_face'], color='r', ls=':', label='half of CK2')
ax[0].set_xlabel('panel-mediated share (e2_p + e3_p) of the loss-normalised face target'); ax[0].set_ylabel('training items (of 96)')
ax[0].set_title('POST HOC: per training item, the box face with the largest panel-mediated share', fontsize=9); ax[0].legend(fontsize=8)
eb = p['exposure_bounds_vs_C_mean']
k = ['protocol D\n(top quartile 50 %)', 'all steps on\ntop quartile', 'all steps on\none top item']
ax[1].bar(k, [eb['protocol_D_sigma0.5_top_quartile'], eb['all_steps_on_top_quartile'], eb['all_steps_on_top_item']], color=['C0', 'gray', 'gray'])
ax[1].axhline(1.25, color='r', ls='--', label='G_EXPOSURE gate 1.25x'); ax[1].axhline(1, color='k', lw=0.8)
ax[1].set_ylabel('panel-mediated loss-mass exposure / C (uniform)'); ax[1].legend(fontsize=8)
ax[1].set_title('item-level reallocation bounds (arithmetic from the audit; only the first was predeclared)', fontsize=9)
fig.suptitle('WL94B post hoc face-level coverage (labelled post hoc; no training; frozen audit aggregates)', fontsize=10)
fn = os.path.join(outd, 'figB4_posthoc_face_coverage.png'); fig.savefig(fn, dpi=80, bbox_inches='tight')
print(hashlib.sha256(open(fn, 'rb').read()).hexdigest())
