"""WL101 analysis: timing, correctness, quality, scaling -> <root>/analysis101.json + reviewer figures in evaluation/101.
  python analyze101.py <evidence_root> <evaluation_dir>
"""
import glob, hashlib, json, os, sys
from pathlib import Path
import numpy as np
from scipy.ndimage import uniform_filter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import seq101 as Q  # noqa: E402

O = Path(sys.argv[1]); EV = Path(sys.argv[2]); EV.mkdir(parents=True, exist_ok=True)
P = Q.P
FR = ['512x1', '512x4', '1024x1']
WLS = ['SEL', 'V_SSS_TOUR', 'V_NEAR_TOUR', 'V_SSS_ANIM']
TOL = {'state_frac': 1e-4, 'src': 1e-5, 'tab_l1': 1e-4, 'tab_max': 1e-2, 'frame': 1e-4}
out = {'timing': {}, 'correctness': {}, 'quality': {}, 'scaling': {}}
J = lambda p: json.load(open(p))


# ------------------------------------------------------------------------------------------------ timing + correctness
def load(wl, arm, r):
    p = O / 'timing' / f'{wl}_{arm}_{r}.json'
    return J(p) if p.exists() else None


for wl in WLS:
    T = {}
    for arm in ['A_sel', 'A_full', 'B_sel', 'B_full', 'B_eager']:
        for r in ['r1', 'r2']:
            x = load(wl, arm, r)
            if x is None:
                continue
            s = x['summary']
            T[f'{arm}_{r}'] = {'dynamic_frame_ms': s['dynamic_frame_ms'], 'parts_median_ms': {k: v['median'] for k, v in s['parts_ms'].items()},
                               'parts_p99_ms': {k: v['p99'] for k, v in s['parts_ms'].items()},
                               'drjit_gpu_median_ms': {k: v['median'] for k, v in s['drjit_gpu_ms'].items()},
                               'cuda_event_median_ms': {k: v['median'] for k, v in s.get('cuda_event_ms', {}).items()},
                               'static_frame_ms': {k: v['median'] for k, v in x['static_frame_ms'].items()},
                               'static_frame_p99_ms': {k: v['p99'] for k, v in x['static_frame_ms'].items()},
                               'dirty_rays_median': s['dirty_rays']['median'], 'state_rays': x['state_rays'], 'N': x['N'], 'memory': x['memory'],
                               'snapshot_before': x['snapshot_before'], 'snapshot_after': x['snapshot_after'], 'recordings': x.get('recordings'),
                               'versions': x['versions'], 'checkpoint_sha256': x['checkpoint_sha256']}
            if arm in ('A_sel', 'B_sel'):
                ev = [e for e in x['events'] if not e['warm']]
                c = {'n_events': len(ev),
                     'selective_equals_full_all': all(e.get('selective_equals_full', True) for e in ev),
                     'selective_equals_full_checked': sum('selective_equals_full' in e for e in ev),
                     'revisit_digest_equal_all': all(e['revisit_digest_equal'] for e in ev if e['revisit_digest_equal'] is not None),
                     'revisits': sum(e['revisit_digest_equal'] is not None for e in ev)}
                if arm == 'B_sel':
                    xs = [e for e in ev if 'xstack_hit_mismatch' in e]
                    c['xstack_events'] = len(xs)
                    if xs:
                        c['xstack_hit_mismatch_max_frac'] = max(e['xstack_hit_mismatch'] for e in xs) / x['state_rays']
                        c['xstack_dbin_mismatch_max'] = max(e['xstack_dbin_mismatch'] for e in xs)
                        c['xstack_src_rel_max'] = max(e['xstack_src_rel_max'] for e in xs)
                        c['xstack_table_rel_l1_max'] = max(e['xstack_table_rel_l1'] for e in xs)
                        c['xstack_table_rel_max_max'] = max(e['xstack_table_rel_max'] for e in xs)
                        fr = [v for e in xs for k in ('xstack_frame1_mean_rel', 'xstack_frame4_mean_rel') if k in e for v in e[k].values()]
                        c['xstack_frame_mean_rel_max'] = max(fr) if fr else None
                        c['xstack_frame_checks'] = len(fr)
                        rv = [e['replay_vs_eager_max_abs'] for e in xs if 'replay_vs_eager_max_abs' in e]
                        c['replay_vs_eager_max_abs'] = max(rv) if rv else None
                        c['table_changed_when_dirty_all'] = all(e.get('table_changed_vs_previous', True) for e in ev if e['dirty_rays'] > 0)
                        c['PASS'] = bool(c['selective_equals_full_all'] and c['revisit_digest_equal_all'] and c['xstack_hit_mismatch_max_frac'] <= TOL['state_frac']
                                         and c['xstack_src_rel_max'] <= TOL['src'] and c['xstack_table_rel_l1_max'] <= TOL['tab_l1']
                                         and c['xstack_table_rel_max_max'] <= TOL['tab_max'] and (c['xstack_frame_mean_rel_max'] or 0) <= TOL['frame']
                                         and (c['replay_vs_eager_max_abs'] in (None, 0.0) or c['replay_vs_eager_max_abs'] < 1e-3))
                out['correctness'][f'{wl}_{arm}_{r}'] = c
    if T:
        # pooled over processes r1 + r2 (event-level)
        pooled = {}
        for arm in ['A_sel', 'B_sel']:
            xs = [load(wl, arm, r) for r in ('r1', 'r2')]
            xs = [x for x in xs if x]
            if not xs:
                continue
            evs = [e for x in xs for e in x['events'] if not e['warm']]
            pooled[arm] = {f: Q.stats([sum(v for k, v in e['parts_ms'].items() if k != 'frame') + (e['parts_ms']['frame'] if f == '512x1' else e['secondary_frame_ms'][f])
                                       for e in evs]) for f in FR}
            pooled[arm]['edit_no_frame'] = Q.stats([sum(v for k, v in e['parts_ms'].items() if k != 'frame') for e in evs])
            pooled[arm]['independent_total_512x1'] = Q.stats([e['total_ms'] for e in evs])
        if 'A_sel' in pooled and 'B_sel' in pooled:
            pooled['speedup_median'] = {f: pooled['A_sel'][f]['median'] / pooled['B_sel'][f]['median'] for f in FR + ['edit_no_frame']}
            pooled['speedup_p99'] = {f: pooled['A_sel'][f]['p99'] / pooled['B_sel'][f]['p99'] for f in FR}
        T['pooled'] = pooled
        out['timing'][wl] = T

# ------------------------------------------------------------------------------------------------ scaling
for f in sorted(glob.glob(str(O / 'scaling' / '*.json'))):
    x = J(f)
    out['scaling'][Path(f).stem] = {'N': x['N'], 'state_rays': x['state_rays'], 'arm': x['arm'], 'scale': x['scale'],
                                    'dynamic_512x1': x['summary']['dynamic_frame_ms']['512x1'],
                                    'parts_median_ms': {k: v['median'] for k, v in x['summary']['parts_ms'].items()},
                                    'dirty_rays_median': x['summary']['dirty_rays']['median'], 'memory': x['memory'],
                                    'snapshot_before': x['snapshot_before']}
for arm in ['A_sel', 'B_sel', 'B_full']:
    p = O / 'timing' / f'V_SSS_TOUR_{arm}_r1.json'
    if p.exists():
        x = J(p)
        out['scaling'][f'V_SSS_TOUR_{arm}_s1.0'] = {'N': x['N'], 'state_rays': x['state_rays'], 'arm': arm, 'scale': 1.0,
                                                   'dynamic_512x1': x['summary']['dynamic_frame_ms']['512x1'],
                                                   'parts_median_ms': {k: v['median'] for k, v in x['summary']['parts_ms'].items()},
                                                   'dirty_rays_median': x['summary']['dirty_rays']['median'], 'memory': x['memory']}


# ------------------------------------------------------------------------------------------------ quality
def lum(x):
    return x.mean(-1)


def rrmse(p, r, m=None):
    if m is not None:
        p, r = p[m], r[m]
    return float(np.sqrt(((p - r) ** 2).mean()) / np.sqrt((r ** 2).mean()))


qimg = {}
for wl in ['V_SSS_TOUR', 'V_NEAR_TOUR']:
    case, states = Q.tour_states(wl)
    d = O / 'quality' / case
    if not (d / f'PRED_B_{states[-1][2]}.npz').exists():
        continue
    roi_sid = 6 if case == 'sss' else 5
    ref = {}
    for g, L, sid, prev in states:
        a, b = np.load(d / f'REF_{sid}_0.npz'), np.load(d / f'REF_{sid}_1.npz')
        ref[sid] = {'tot': 0.5 * (a['e1'] + a['e2'] + a['h'] + b['e1'] + b['e2'] + b['h']).astype(np.float64),
                    'totA': (a['e1'] + a['e2'] + a['h']).astype(np.float64), 'totB': (b['e1'] + b['e2'] + b['h']).astype(np.float64),
                    'h': 0.5 * (a['h'] + b['h']).astype(np.float64), 'hA': a['h'].astype(np.float64), 'hB': b['h'].astype(np.float64)}
    rows = []
    for g, L, sid, prev in states:
        R_ = ref[sid]; Pv = ref[prev]
        roi = np.load(d / f'MAP_{sid}.npz')['sid'] == roi_sid
        dh = R_['h'] - Pv['h']
        Rh = float(np.abs(dh[roi]).sum() / R_['h'][roi].sum())
        Rn = float((0.5 * np.abs(R_['hA'] - R_['hB'])[roi].sum() + 0.5 * np.abs(Pv['hA'] - Pv['hB'])[roi].sum()) / R_['h'][roi].sum())
        v = lum(0.25 * (R_['hA'] - R_['hB']) ** 2) + lum(0.25 * (Pv['hA'] - Pv['hB']) ** 2)
        chg = np.abs(uniform_filter(lum(dh), 3)) > 3 * np.sqrt(uniform_filter(v, 7) / 9)
        row = {'state': sid, 'prev': prev, 'roi_pixels': int(roi.sum()), 'h_change_ratio_roi': Rh, 'h_change_noise_roi': Rn,
               'qualifies_material_h_change': bool(Rh >= 0.05 and Rn <= 0.5 * Rh), 'h_change_region_pixels': int(chg.sum()),
               'ref_noise_rrmse_full': rrmse(0.5 * R_['totA'], 0.5 * R_['totB']) / np.sqrt(2) * 1.0,
               'ref_noise_rrmse_roi': float(np.sqrt((0.25 * (R_['totA'] - R_['totB']) ** 2)[roi].mean()) / np.sqrt((R_['tot'][roi] ** 2).mean())),
               'h_share_roi': float(R_['h'][roi].sum() / R_['tot'][roi].sum())}
        imgs = {}
        for arm, fn in (('A', f'PRED_A_{sid}.npz'), ('B', f'PRED_B_{sid}.npz'), ('STALE', f'PRED_STALE_{sid}.npz')):
            z = np.load(d / fn)
            tot = (z['e1'] + z['e2'] + z['hl']).astype(np.float64); hl = z['hl'].astype(np.float64)
            imgs[arm] = (tot, hl)
            den_full = R_['tot'].sum(); den_roi = R_['tot'][roi].sum(); den_chg = R_['tot'][chg].sum() if chg.any() else np.nan
            row[arm] = {'rrmse_full': rrmse(tot, R_['tot']), 'rrmse_roi': rrmse(tot, R_['tot'], roi),
                        'EH_full': float(np.abs(hl - R_['h']).sum() / den_full), 'EH_roi': float(np.abs(hl - R_['h'])[roi].sum() / den_roi),
                        'EH_hchange': float(np.abs(hl - R_['h'])[chg].sum() / den_chg) if chg.any() else None,
                        'h_rel_err_roi': float(np.abs(hl - R_['h'])[roi].sum() / R_['h'][roi].sum())}
        row['A_vs_B_tot_rel_l1'] = float(np.abs(imgs['A'][0] - imgs['B'][0]).sum() / imgs['A'][0].sum())
        row['A_vs_B_hl_rel_l1'] = float(np.abs(imgs['A'][1] - imgs['B'][1]).sum() / imgs['A'][1].sum())
        tabA, tabB = np.load(d / f'PRED_A_{sid}.npz')['B'], np.load(d / f'PRED_B_{sid}.npz')['B']
        row['table_A_vs_B_rel_l1'] = float(np.abs(tabA - tabB).sum() / tabA.sum())
        row['table_B_vs_STALE_rel_l1'] = float(np.abs(np.load(d / f'PRED_STALE_{sid}.npz')['B'] - tabB).sum() / tabB.sum())
        rows.append(row)
        qimg[(case, sid)] = (R_, Pv, imgs, roi, chg, dh)
    out['quality'][wl] = rows

json.dump(out, open(O / 'analysis101.json', 'w'), indent=1)

# ------------------------------------------------------------------------------------------------ figures
COL = {'geometry_update': '#4e79a7', 'relation_update': '#f28e2b', 'lights': '#bab0ac', 'source': '#59a14f', 'generator_apply': '#e15759',
       'table_upload': '#76b7b2', 'table_handoff': '#76b7b2', 'frame': '#b07aa1'}
figs = []
if out['timing']:
    fig, axs = plt.subplots(1, len(out['timing']), figsize=(4.2 * len(out['timing']), 4.6), sharey=True)
    axs = np.atleast_1d(axs)
    for ax, (wl, T) in zip(axs, out['timing'].items()):
        bars = [k for k in ['A_sel_r1', 'A_full_r1', 'B_eager_r1', 'B_full_r1', 'B_sel_r1'] if k in T]
        for i, k in enumerate(bars):
            y = 0
            for part, v in T[k]['parts_median_ms'].items():
                ax.bar(i, v, bottom=y, color=COL.get(part, 'k'), label=part if (wl == list(out['timing'])[0] and i == 0) or part == 'table_handoff' and i == len(bars) - 1 and wl == list(out['timing'])[0] else None)
                y += v
            ax.text(i, y + 1, f'{y:.1f}', ha='center', fontsize=8)
        for t, ls in ((20, '--'), (15, '-.'), (10, ':')):
            ax.axhline(t, color='k', ls=ls, lw=0.8)
        ax.set_xticks(range(len(bars))); ax.set_xticklabels([b.replace('_r1', '') for b in bars], rotation=30, fontsize=8)
        ax.set_title(wl, fontsize=10); ax.set_yscale('log'); ax.set_ylim(0.5, 2000)
    axs[0].set_ylabel('median ms per dynamic frame (512² × 1 spp), log'); axs[0].legend(fontsize=7, loc='upper right')
    fig.suptitle('WL101 dynamic-frame breakdown (part medians; lines: 20 / 15 / 10 ms)', fontsize=11)
    fig.tight_layout(); fig.savefig(EV / 'timing_breakdown.png', dpi=130); plt.close(fig); figs.append('timing_breakdown.png')
    # per-event latency
    fig, axs = plt.subplots(1, len(out['timing']), figsize=(4.2 * len(out['timing']), 3.8), sharey=True)
    axs = np.atleast_1d(axs)
    for ax, wl in zip(axs, out['timing']):
        for arm, c in (('A_sel', '#e15759'), ('B_sel', '#4e79a7')):
            for r, m in (('r1', 'o'), ('r2', 's')):
                x = load(wl, arm, r)
                if x:
                    ev = [e for e in x['events'] if not e['warm']]
                    ax.plot([e['total_ms'] for e in ev], m + '-', color=c, ms=3, lw=0.8, label=f'{arm} {r}')
        for t, ls in ((20, '--'), (15, '-.'), (10, ':')):
            ax.axhline(t, color='k', ls=ls, lw=0.8)
        ax.set_yscale('log'); ax.set_title(wl, fontsize=10); ax.set_xlabel('timed event')
    axs[0].set_ylabel('edit -> frame complete, ms (log)'); axs[0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(EV / 'latency_per_event.png', dpi=130); plt.close(fig); figs.append('latency_per_event.png')
if out['scaling']:
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for arm, c in (('A_sel', '#e15759'), ('B_sel', '#4e79a7'), ('B_full', '#59a14f')):
        pts = sorted((v['N'], v['dynamic_512x1']['median'], v) for k, v in out['scaling'].items() if v['arm'] == arm)
        if pts:
            ax[0].plot([p[0] for p in pts], [p[1] for p in pts], 'o-', color=c, label=arm)
            if arm != 'A_sel':
                for part, ls in (('generator_apply', '-'), ('relation_update', '--'), ('geometry_update', ':'), ('frame', '-.')):
                    ax[1].plot([p[0] for p in pts], [p[2]['parts_median_ms'][part] for p in pts], ls, marker='o', color=c, ms=3, label=f'{arm} {part}')
    for a in ax:
        a.set_xscale('log'); a.set_yscale('log'); a.set_xlabel('patches N')
    for t, ls in ((20, '--'), (15, '-.'), (10, ':')):
        ax[0].axhline(t, color='k', ls=ls, lw=0.8)
    ax[0].set_ylabel('median dynamic frame ms (512² × 1 spp)'); ax[0].legend(fontsize=8); ax[1].legend(fontsize=6); ax[1].set_ylabel('part median ms')
    fig.suptitle('Scaling with patch refinement (timing only, same checkpoint; quality not evaluated)', fontsize=10)
    fig.tight_layout(); fig.savefig(EV / 'scaling.png', dpi=130); plt.close(fig); figs.append('scaling.png')


def tm(x, ex):
    return np.clip((x / ex) ** (1 / 2.2), 0, 1)


for wl, rows in out['quality'].items():
    case = 'sss' if 'SSS' in wl else 'near'
    keys = [k for k in qimg if k[0] == case]
    ex = np.percentile(np.concatenate([qimg[k][0]['tot'].ravel() for k in keys]), 99.5)
    emax = np.percentile(np.concatenate([np.abs(qimg[k][2]['STALE'][1] - qimg[k][0]['h']).mean(-1).ravel() for k in keys]), 99.5)
    hmax = np.percentile(np.concatenate([qimg[k][0]['h'].ravel() for k in keys]), 99.5)
    sel = sorted(rows, key=lambda r: -r['h_change_ratio_roi'])[:3]
    fig, axs = plt.subplots(len(sel), 7, figsize=(24, 3.6 * len(sel)))
    axs = np.atleast_2d(axs)
    for ax_r, row in zip(axs, sel):
        R_, Pv, imgs, roi, chg, dh = qimg[(case, row['state'])]
        panels = [(tm(R_['tot'], ex), f'reference T (state {row["state"]})'), (tm(imgs['B'][0], ex), 'B_sel prediction (= A)'),
                  (np.sqrt(np.abs(dh).mean(-1) / hmax), f'|h_ref(T) - h_ref(prev {row["prev"]})| (sqrt)'),
                  (tm(R_['h'], hmax), 'reference higher-order h'), (tm(imgs['B'][1], hmax), 'B_sel learned hl'),
                  (np.sqrt(np.clip(np.abs(imgs['B'][1] - R_['h']).mean(-1) / emax, 0, 1)), f'|hl - h| B_sel (sqrt, max {emax:.3g})'),
                  (np.sqrt(np.clip(np.abs(imgs['STALE'][1] - R_['h']).mean(-1) / emax, 0, 1)), '|hl - h| STALE (same scale)')]
        for a, (im, t) in zip(ax_r, panels):
            a.imshow(im, cmap=None if im.ndim == 3 else 'inferno', vmin=0, vmax=1); a.set_title(t, fontsize=8); a.axis('off')
            if im.ndim == 2:
                a.contour(chg, levels=[0.5], colors='cyan', linewidths=0.3)
        ax_r[0].set_ylabel(f'R_h {row["h_change_ratio_roi"]:.3f}')
    fig.suptitle(f'{wl}: three largest higher-order changes. Error maps share one absolute scale per scene (sqrt display); cyan = h-change region', fontsize=10)
    fig.tight_layout(); fn = f'quality_{case}.png'; fig.savefig(EV / fn, dpi=110); plt.close(fig); figs.append(fn)

man = {'files': {f: hashlib.sha256((EV / f).read_bytes()).hexdigest() for f in figs},
       'analysis_json_sha256': hashlib.sha256((O / 'analysis101.json').read_bytes()).hexdigest(), 'source': str(O)}
json.dump(man, open(EV / 'manifest101.json', 'w'), indent=1)
print(json.dumps({wl: T.get('pooled', {}).get('speedup_median') for wl, T in out['timing'].items()}, indent=1))
print(json.dumps({k: v.get('PASS') for k, v in out['correctness'].items()}))
