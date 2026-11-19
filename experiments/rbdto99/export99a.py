"""WL99A reviewer export: D1 reproduction check, v1 / v2 resolution tables, figures (Sd, v2 high-pass residual, T1-change localisation).
  python export99a.py <wl99_stage0_dir> <wl99a_dir> <out_dir>
"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import wl99, q3v2

S0, WA, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(OUT, exist_ok=True)
qa = json.load(open(os.path.join(WA, 'qual99a.json')))
q1 = json.load(open(os.path.join(S0, 'qual99.json')))['workloads']
IND = {w['id']: w for w in json.load(open(os.path.join(wl99.HERE, 'protocol', 'workloads99a_indep.json')))['workloads']}
ALL = {w['id']: w for w in wl99.WL}; ALL.update(IND)


def load(wid, st, spp, res, store):
    p = (lambda h: os.path.join(store, 'refs', f'{wid}_{st}_{h}_{spp}.npz')) if res == 384 else (lambda h: os.path.join(WA, 'refs', f'{wid}_{st}_{h}_{spp}_r{res}.npz'))
    A, B = dict(np.load(p('A'))), dict(np.load(p('B')))
    return A, B, {k: (A[k].astype(np.float64) + B[k]) / 2 for k in A}


# D1: WL99 v1 Q3 spatial reproduced from the stored references (development set)
d1 = {}
for wid, c in qa['candidates'].items():
    if c['set'] != 'development' or c['type'] == 'C':
        continue
    for st in ('T0', 'T1'):
        new = c[f'{st}_384']['v1']; old = q1[wid]['Q3_spatial'][st]
        d1[f'{wid}/{st}'] = {'old': [old['edge_width_px'], old['contrast']], 'new': [new['edge_width_px'], new['contrast']],
                             'equal': bool(abs(old['edge_width_px'] - new['edge_width_px']) < 1e-9 and abs(old['contrast'] - new['contrast']) < 1e-9)}
rows = []
for wid, c in qa['candidates'].items():
    if c['type'] == 'C':
        continue
    for st in ('T0', 'T1'):
        r3, r7 = c[f'{st}_384'], c[f'{st}_768']
        rows.append({'id': wid, 'set': c['set'], 'type': c['type'], 'state': st,
                     'v1_w384': r3['v1']['edge_width_px'], 'v1_w768': r7['v1']['edge_width_px'], 'v1_c384': r3['v1']['contrast'], 'v1_c768': r7['v1']['contrast'],
                     'v1_w384_halfA': r3['v1_halfA']['edge_width_px'],
                     'v2_C384': r3['v2']['C'], 'v2_se384': r3['v2']['se'], 'v2_C768': r7['v2']['C'], 'v2_se768': r7['v2']['se'],
                     'v2_C1536': c.get(f'{st}_1536', {}).get('v2', {}).get('C'), 'conv_384_768': c[f'{st}_conv_384_768'],
                     'footprint_mm_384': r3['footprint_mm'], 'v1_admitted': c.get('v1_admitted'), 'v2_admitted': c.get('v2_admitted'),
                     'v2_state': c['Q3v2'][st]})
summ = {'D1_reproduction': d1, 'D1_all_equal': all(v['equal'] for v in d1.values()), 'rows': rows,
        'required_set_v2': qa['required_set_v2'], 'verdict': qa['verdict']}
json.dump(summ, open(os.path.join(OUT, 'wl99a_summary.json'), 'w'), indent=1, default=float)

# figure: per candidate T0 Sd (384), v2 high-pass residual (768, same physical scale), |Sd_T1 - Sd_stale| (384)
ids = [k for k, c in qa['candidates'].items() if c['type'] != 'C']
fig, ax = plt.subplots(len(ids), 3, figsize=(9, 2.6 * len(ids)))
tone = lambda x: np.clip(x / (1 + x), 0, 1) ** (1 / 2.2)
for i, wid in enumerate(ids):
    c = qa['candidates'][wid]
    store = S0 if c['set'] == 'development' else WA
    spp = c['spp_384']
    A, B, R = load(wid, 'T0', spp, 384, store)
    A7, B7, R7 = load(wid, 'T0', 8192, 768, None)
    body7 = R7['mb'] >= 0.99
    s7 = q3v2.LAMBDA_C / np.sqrt(12) / (c['T0_768']['footprint_mm'] / 1e3)
    hp = q3v2.highpass(R7['Sd'].mean(-1), body7, s7) * body7
    sd_mean = max(float(R7['Sd'].mean(-1)[body7].mean()), 1e-12)
    st_spp = 16384
    try:
        As, Bs, Rs = load(wid, 'stale', st_spp, 384, store)
        A1, B1, R1 = load(wid, 'T1', spp, 384, store)
        dm = np.abs(R1['Sd'] - Rs['Sd']).mean(-1)
    except FileNotFoundError:
        dm = np.zeros(R['Sd'].shape[:2])
    ax[i, 0].imshow(tone(R['Sd'])); ax[i, 1].imshow(hp / sd_mean, cmap='RdBu_r', vmin=-0.2, vmax=0.2)
    ax[i, 2].imshow(np.sqrt(dm / max(dm.max(), 1e-12)), cmap='magma', vmin=0, vmax=1)
    for a in ax[i]:
        a.set_xticks([]); a.set_yticks([])
    q = c['Q3v2']
    ax[i, 0].set_ylabel(f"{wid}\nC={q['T0']['C']:.3f}/{q['T1']['C']:.3f}\n{'v2 ADMITTED' if c['v2_admitted'] else 'v2 not admitted'}", fontsize=7)
    if i == 0:
        ax[i, 0].set_title('T0 Sd (384)', fontsize=8); ax[i, 1].set_title('2-cm high-pass of Sd / mean (768, +-0.2)', fontsize=8)
        ax[i, 2].set_title('|Sd_T1 - Sd_stale| (sqrt, per-row max)', fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(OUT, 'fig99A_structure.png'), dpi=60)

# resolution sensitivity scatter
fig, ax = plt.subplots(1, 3, figsize=(13, 4))
w3 = [r['v1_w384'] for r in rows]; w7 = [r['v1_w768'] for r in rows]
ax[0].scatter(w3, w7, s=12); ax[0].plot([0, 4.5], [0, 9], 'g--', label='resolved feature (width x2)'); ax[0].plot([0, 4.5], [0, 4.5], 'k:', label='pixel-locked (width x1)')
ax[0].axhline(2, color='r', lw=0.6); ax[0].axvline(2, color='r', lw=0.6); ax[0].set_xlabel('WL99 edge width at 384 (px)'); ax[0].set_ylabel('at 768 (px)'); ax[0].legend(fontsize=7)
c3 = [r['v1_c384'] for r in rows]; c7 = [r['v1_c768'] for r in rows]
ax[1].scatter(c3, c7, s=12); ax[1].plot([0, 2.2], [0, 2.2], 'k:'); ax[1].axhline(0.3, color='r', lw=0.6); ax[1].axvline(0.3, color='r', lw=0.6)
ax[1].set_xlabel('WL99 contrast at 384'); ax[1].set_ylabel('at 768')
C3 = [r['v2_C384'] for r in rows]; C7 = [r['v2_C768'] for r in rows]
ax[2].errorbar(C3, C7, xerr=[r['v2_se384'] for r in rows], yerr=[r['v2_se768'] for r in rows], fmt='o', ms=3, lw=0.5)
ax[2].plot([0, 0.16], [0, 0.16], 'k:'); ax[2].axhline(0.05, color='r', lw=0.6); ax[2].axvline(0.05, color='r', lw=0.6)
ax[2].set_xlabel('Q3 v2 C at 384'); ax[2].set_ylabel('at 768')
fig.suptitle('WL99A resolution sensitivity (all non-null candidates, T0 and T1)')
fig.tight_layout(); fig.savefig(os.path.join(OUT, 'fig99A_resolution.png'), dpi=80)
print('D1 all equal', summ['D1_all_equal'], len(rows))
