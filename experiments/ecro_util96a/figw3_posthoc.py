"""WL96A post-hoc figure fix (figure only; no metric or verdict changes): figW3 lifecycle bars with ONE colour per component across both
panels (the chain's figW3 assigned colours per panel, so the same colour meant different components). Input: results/ecro_util96a/cost_s1.0.json.
  python figw3_posthoc.py <cost_s1.0.json> <out.png>"""
import json, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

c = json.load(open(sys.argv[1]))
arms = list(c['arms'])
keys = ['geometry_update', 'relation_update', 'relation_coefficients', 'generator', 'source_q_table', 'prefix_order', 'tail_orders',
        'lambda_apply', 'table_handover', 'render']
col = {k: plt.cm.tab10(i) for i, k in enumerate(keys)}
fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
for a, kind in zip(ax, ('edit', 'source')):
    bottom = np.zeros(len(arms))
    for k in keys:
        v = np.array([c['arms'][ar][kind]['parts'].get(k, 0.0) * 1e3 for ar in arms])
        if v.any():
            a.bar(arms, v, bottom=bottom, label=k, color=col[k]); bottom += v
    for i, ar in enumerate(arms):
        a.text(i, bottom[i] + 1, f"{c['arms'][ar][kind]['wall_s'] * 1e3:.1f}", ha='center', fontsize=8)
    a.set_ylabel('ms (median of warm events)')
    a.set_title(f"CK2 {kind} -> frame (N {c['N']}, {c['res']}^2, {c['frame_spp']} spp)", fontsize=9)
h, l = ax[0].get_legend_handles_labels()
fig.legend(h, l, fontsize=7, loc='center right')
fig.tight_layout(rect=(0, 0, 0.84, 1))
fig.savefig(sys.argv[2], dpi=130)
