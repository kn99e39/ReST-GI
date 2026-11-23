"""WL110 analysis: aggregate the live runs, verification and same-day WL101 baseline into evaluation/110/.

  python analyze110.py <run110 out_root> <repo>/evaluation/110
Copies only reviewer-facing artifacts (screenshots, recordings, per-run summaries, verify.json, aggregated tables, figures) and
records their source path and sha256 in manifest110.json. Raw frames.jsonl stay in the run root (their hashes are recorded).
"""
import hashlib, json, shutil, sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

SRC, DST = Path(sys.argv[1]), Path(sys.argv[2])
DST.mkdir(parents=True, exist_ok=True)
INK, INK2, GRID, SURF = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'
SERIES = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948']   # fixed categorical order
plt.rcParams.update({'font.size': 10, 'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2, 'ytick.color': INK2,
                     'axes.facecolor': SURF, 'figure.facecolor': SURF, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.8,
                     'axes.spines.top': False, 'axes.spines.right': False})


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def frames(name):
    return [json.loads(l) for l in open(SRC / 'runs' / name / 'frames.jsonl')]


runs = sorted(p.name for p in (SRC / 'runs').iterdir() if (p / 'summary.json').exists())
S = {n: json.loads((SRC / 'runs' / n / 'summary.json').read_text()) for n in runs}
table = []
for n in runs:
    s = S[n]
    rec = SRC / 'rec' / n / 'capture.json'
    cap = json.loads(rec.read_text()) if rec.exists() else None
    table.append({
        'run': n, 'case': s['args']['case'], 'mode': s['args']['mode'], 'present': s['args']['present'], 'fps_cap': s['args']['fps_cap'],
        'recorder': cap is not None, 'frames_measured': s['frames_measured'],
        'presented_fps': s['presented_fps'],
        'present_interval_ms_median': s['present_interval_ms']['median'], 'present_interval_ms_p95': s['present_interval_ms']['p95'],
        'present_interval_ms_max': s['present_interval_ms']['max'],
        'compute_ms_median': s['compute_ms']['median'], 'compute_ms_p95': s['compute_ms']['p95'],
        'present_ms_median': s['present_ms']['median'], 'present_ms_p95': s['present_ms']['p95'],
        'parts_ms_median': {k: v['median'] for k, v in s['parts_ms'].items()},
        'missed_deadlines': s['deadline']['missed'], 'animation_lag_s': s['deadline']['animation_lag_s'],
        'torch_peak_mib': s['memory']['torch_peak_mib'], 'device_used_delta_mib': s['memory']['device_used_delta_mib'],
        'integrity': s['integrity'],
        'screen_change_rate_hz': cap['screen_change_rate_hz'] if cap else None, 'recorder_busy_fraction': cap['recorder_busy_fraction'] if cap else None,
        'foreign_gpu_before': [a for a in s['gpu_before'].get('compute_apps', []) if str(s['gpu_before'].get('own_pid')) not in a],
    })
wl101 = {}
for p in sorted((SRC / 'wl101').glob('*.json')):
    d = json.loads(p.read_text())
    wl101[p.stem] = {'dynamic_frame_512x1_ms': d['summary']['dynamic_frame_ms']['512x1'], 'parts_ms_median': {k: v['median'] for k, v in d['summary']['parts_ms'].items()},
                     'selective_equals_full': sum(1 for e in d['events'] if e.get('selective_equals_full')),
                     'checked': sum(1 for e in d['events'] if 'selective_equals_full' in e), 'checkpoint_sha256': d['checkpoint_sha256']}
arch = Path.home() / 'rest_dyn101/results/ecro_dyn101/evidence/timing'
wl101_hist = {r: json.loads((arch / f'V_SSS_ANIM_B_sel_{r}.json').read_text())['summary']['dynamic_frame_ms']['512x1'] for r in ('r1', 'r2')}
ver = json.loads((SRC / 'verify' / 'verify.json').read_text()) if (SRC / 'verify' / 'verify.json').exists() else None
agg = {'batch': 'WL110', 'runs': table, 'wl101_same_day_rerun': wl101, 'wl101_historical_2026_11_17': wl101_hist,
       'pmon_logs': {p.stem: sha(p) for p in sorted((SRC / 'logs').glob('*.pmon'))}}
(DST / 'runs110.json').write_text(json.dumps(agg, indent=1))

# ---- figure 1: per-frame time series, primary run
prim = 'sss_raw_cap30_r1'
if prim in S:
    F = frames(prim)
    k = [f['k'] for f in F]
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.plot(k[1:], [f['loop_ms'] for f in F[1:]], color=SERIES[1], lw=1.2, label='present-to-present interval')
    ax.plot(k, [f['compute_ms'] for f in F], color=SERIES[0], lw=1.2, label='compute: edit → display-ready image')
    ax.plot(k, [f['compute_ms'] + f['present_ms'] for f in F], color=SERIES[2], lw=1.0, label='compute + Tk/X11 present')
    ax.axhline(1000 / 30, color=INK2, lw=0.8, ls='--'); ax.text(k[-1], 1000 / 30 + 0.6, '33.3 ms (30 FPS slot)', ha='right', color=INK2, fontsize=9)
    ax.axvspan(0, S[prim]['warmup_frames'], color=GRID, alpha=0.6, lw=0); ax.text(2, 2, 'warm-up\n(excluded)', color=INK2, fontsize=8)
    ax.set_xlabel('frame'); ax.set_ylabel('ms'); ax.set_ylim(0, max(60, np.percentile([f['loop_ms'] or 0 for f in F], 99.5) * 1.1))
    ax.set_title(f"WL110 live loop · SSS raw 512² 1 spp · 30 FPS cap · {S[prim]['presented_fps']:.2f} presented FPS", loc='left', color=INK)
    ax.legend(loc='upper right', frameon=False)
    fig.tight_layout(); fig.savefig(DST / 'fig01_frame_timeline_sss_raw_cap30.png', dpi=130); plt.close(fig)

# ---- figure 2: presented FPS per run (single series) with p95 interval annotation
order = [n for n in runs]
fig, ax = plt.subplots(figsize=(10, 0.42 * len(order) + 1.2))
y = np.arange(len(order))
fps = [S[n]['presented_fps'] for n in order]
ax.barh(y, fps, color=SERIES[0], height=0.6)
for i, n in enumerate(order):
    s = S[n]
    ax.text(fps[i] + 0.5, i, f"{fps[i]:.1f} FPS · interval med {s['present_interval_ms']['median']:.1f} / p95 {s['present_interval_ms']['p95']:.1f} ms · missed {s['deadline']['missed']}",
            va='center', color=INK, fontsize=8.5)
ax.set_yticks(y); ax.set_yticklabels(order); ax.invert_yaxis()
ax.axvline(30, color=INK2, lw=0.8, ls='--')
ax.set_xlim(0, max(fps) * 1.9); ax.set_xlabel('measured presented frames per second (headless rows: loop rate without presentation)')
ax.set_title('WL110 · presented FPS per run (600 measured frames each)', loc='left', color=INK)
fig.tight_layout(); fig.savefig(DST / 'fig02_presented_fps_per_run.png', dpi=130); plt.close(fig)

# ---- figure 3: median frame composition per run (stacked, fixed category order)
cats = [('geometry', 'geometry update'), ('relation', 'relation update'), ('source', 'source encoding'), ('neural', 'ECRO neural response'),
        ('render', 'ray render'), ('recon', 'guide + reconstruction'), ('other', 'hand-off, compose, tonemap, readback'), ('present', 'Tk/X11 present')]
fig, ax = plt.subplots(figsize=(10, 0.42 * len(order) + 1.8))
left = np.zeros(len(order))
for ci, (key, lab) in enumerate(cats):
    vals = []
    for n in order:
        p = S[n]['parts_ms']
        m = lambda q: p[q]['median'] if q in p else 0.0
        if key == 'recon':
            v = m('guide') + m('reconstruct')
        elif key == 'other':
            v = m('handoff') + m('compose') + m('display')
        elif key == 'present':
            v = S[n]['present_ms']['median']
        else:
            v = m(key)
        vals.append(v)
    ax.barh(y, vals, left=left, color=SERIES[ci], height=0.6, label=lab, edgecolor=SURF, linewidth=1.5)
    left += np.array(vals)
ax.set_yticks(y); ax.set_yticklabels(order); ax.invert_yaxis()
ax.axvline(1000 / 30, color=INK2, lw=0.8, ls='--')
ax.set_xlabel('median ms per frame (synchronised parts)')
ax.legend(ncol=4, loc='upper center', bbox_to_anchor=(0.5, -0.18 if len(order) > 6 else -0.25), frameon=False, fontsize=8.5)
ax.set_title('WL110 · median frame composition per run', loc='left', color=INK)
fig.tight_layout(); fig.savefig(DST / 'fig03_frame_composition_per_run.png', dpi=130); plt.close(fig)

# ---- copy reviewer artifacts
man = {'batch': 'WL110', 'source_root': str(SRC), 'files': {}}


def put(src, name):
    shutil.copy2(src, DST / name)
    man['files'][name] = {'source': str(src), 'sha256': sha(DST / name)}


for n in runs:
    put(SRC / 'runs' / n / 'summary.json', f'summary_{n}.json')
    man['files'][f'summary_{n}.json']['frames_jsonl_sha256'] = sha(SRC / 'runs' / n / 'frames.jsonl')
    if (SRC / 'runs' / n / 'last_frame.png').exists():
        put(SRC / 'runs' / n / 'last_frame.png', f'last_frame_{n}.png')
for d in sorted((SRC / 'rec').iterdir()):
    for f, nm in (('screenshot.png', 'screenshot'), ('recording.webp', 'recording'), ('recording_preview.gif', 'recording_preview'), ('capture.json', 'capture')):
        if (d / f).exists():
            put(d / f, f'{nm}_{d.name}{Path(f).suffix}')
if ver:
    put(SRC / 'verify' / 'verify.json', 'verify.json')
for p in sorted((SRC / 'wl101').glob('*.json')):
    put(p, p.name)
put(SRC / 'status.txt', 'status.txt')
for f in ('runs110.json', 'fig01_frame_timeline_sss_raw_cap30.png', 'fig02_presented_fps_per_run.png', 'fig03_frame_composition_per_run.png'):
    if (DST / f).exists():
        man['files'][f] = {'source': 'generated by analyze110.py', 'sha256': sha(DST / f)}
man['analyze110_sha256'] = sha(__file__)
(DST / 'manifest110.json').write_text(json.dumps(man, indent=1))
print('ANALYSIS DONE', len(man['files']), 'files')
