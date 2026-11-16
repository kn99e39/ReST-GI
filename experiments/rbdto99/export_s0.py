"""WL99 Stage 0 reviewer export: reference renders (tot, Sd) of every candidate at T0 / T1 (+ stale / co / T1x where present),
identical display mapping for all (Reinhard on 2^ev, gamma 2.2, one ev for all), plus a contact sheet and the sharp-ROI masks.
  python export_s0.py <stage0_dir> <out_dir>
"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import wl99

S0, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)
q = json.load(open(os.path.join(S0, 'qual99.json')))
EV = 0.0


def tone(x):
    y = np.clip(x * 2.0 ** EV, 0, None)
    return np.clip(y / (1 + y), 0, 1) ** (1 / 2.2)


rows = []
for wid, v in q['workloads'].items():
    spp = v['spp_per_half']
    for st in ('T0', 'T1'):
        A = dict(np.load(os.path.join(S0, 'refs', f'{wid}_{st}_A_{spp}.npz'))); B = dict(np.load(os.path.join(S0, 'refs', f'{wid}_{st}_B_{spp}.npz')))
        R = {k: (A[k].astype(np.float64) + B[k]) / 2 for k in A}
        for key in ('tot', 'Sd'):
            plt.imsave(os.path.join(OUT, f'{wid}_{st}_{key}.png'), tone(R[key]))
        body, roi, sharp = wl99.masks(R)
        m = np.zeros(R['tot'].shape); m[..., 0] = roi * 0.5; m[..., 1] = sharp
        plt.imsave(os.path.join(OUT, f'{wid}_{st}_masks.png'), np.clip(0.6 * tone(R['Sd']) + 0.4 * m, 0, 1))
        rows.append((wid, st, R))
n = len(q['workloads'])
fig, ax = plt.subplots(n, 4, figsize=(12, 3 * n))
for i, wid in enumerate(q['workloads']):
    v = q['workloads'][wid]
    for j, (st, key) in enumerate((('T0', 'tot'), ('T0', 'Sd'), ('T1', 'tot'), ('T1', 'Sd'))):
        R = [r for r in rows if r[0] == wid and r[1] == st][0][2]
        ax[i, j].imshow(tone(R[key])); ax[i, j].set_xticks([]); ax[i, j].set_yticks([])
        if i == 0:
            ax[i, j].set_title(f'{st} {key}')
    failed = [g for g, ok in v['gates'].items() if not ok]
    ax[i, 0].set_ylabel(f"{wid}\n{'ADMITTED' if v['admitted'] else 'not admitted: ' + ','.join(failed)}", fontsize=8)
fig.suptitle('WL99 Stage 0 references (identical display mapping; Sd = body self-transport of direct light)', fontsize=10)
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'figS0_contact_sheet.png'), dpi=70)
json.dump({'display': 'Reinhard x / (1 + x) of 2^0 * radiance, gamma 2.2, no per-image normalisation', 'source': 'reference halves mean',
           'qual99': {k: {'admitted': v['admitted'], 'gates': v['gates']} for k, v in q['workloads'].items()},
           'required_set': q['required_set'], 'verdict': q['stage0_verdict']}, open(os.path.join(OUT, 'manifest_stage0.json'), 'w'), indent=1)
print('ok', len(rows))
