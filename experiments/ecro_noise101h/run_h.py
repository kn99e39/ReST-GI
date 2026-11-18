"""WL101H evidence run (local RTX 5080). Order: G0 inputs -> geometry ROIs -> G1 HIGH state -> G2/G3 replays -> J normal sweep
(+ G4, G5 frame-level) -> PF split triples (+ G4, G6) -> HQ -> Q quadrature (conditional) -> timing -> manifest.
Stops (exit 2) on any G0-G6 failure. Never relaxes a threshold.
  <venv-cu128 python> run_h.py <raw_root>
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

import common_h as CH

P = CH.P
RAW = Path(sys.argv[1]).resolve()
RAW.mkdir(parents=True, exist_ok=True)
LOG = open(RAW / 'run.log', 'a', encoding='utf-8')
SMOKE = os.environ.get('WL101H_SMOKE') == '1'          # pre-freeze pipeline smoke: tiny counts, all seeds offset by +5e6
OFF = 5_000_000 if SMOKE else 0
CFG = {'smoke': SMOKE, 'replicates': 4 if SMOKE else P['arms']['J_normal']['replicates'], 'hq_spp': 64 if SMOKE else 4096,
       'q_spp': 64 if SMOKE else 1024, 'timing_iters': 8 if SMOKE else 36, 'seed_offset': OFF}
MAN = {'id': P['id'], 'started': time.strftime('%Y-%m-%d %H:%M:%S'), 'gates': {}, 'stages': {}, 'config': CFG}


def log(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True); LOG.write(s + '\n'); LOG.flush()


def save_manifest():
    CH.write_json(RAW / 'manifest.json', MAN)


def gate(name, ok, detail):
    MAN['gates'][name] = {'pass': bool(ok), 'detail': detail}
    save_manifest()
    log(('GATE PASS ' if ok else 'GATE FAIL ') + name, json.dumps(detail)[:600])
    if not ok:
        MAN['stopped_at'] = name
        save_manifest()
        log('STOP: gate failure, no threshold relaxation')
        LOG.close()
        os._exit(2)


def git(*args):
    return subprocess.check_output(['git', '-C', str(CH.REPO), *args], text=True).strip()


# ---------------------------------------------------------------------------------------------------------------- G0
MAN['commit'] = git('rev-parse', 'HEAD')
MAN['branch'] = git('rev-parse', '--abbrev-ref', 'HEAD')
dirty = git('status', '--porcelain', '--', 'experiments/ecro_noise101h')
MAN['experiment_dir_clean'] = dirty == ''
MAN['protocol_sha256'] = CH.sha(CH.PROTOCOL_PATH)
MAN['code_sha256'] = {f: CH.sha(CH.HERE / f) for f in ('common_h.py', 'render_h.py', 'stats_h.py', 'run_h.py', 'analyze_h.py', 'tests_h.py')
                      if (CH.HERE / f).exists()}
bad = {k: CH.sha(CH.REPO / k) for k, v in P['frozen_inputs_sha256'].items() if CH.sha(CH.REPO / k) != v}
gate('G0_inputs', not bad and (MAN['experiment_dir_clean'] or SMOKE), {'mismatch': bad, 'experiment_dir_clean': MAN['experiment_dir_clean'],
                                                           'commit': MAN['commit']})

import torch  # noqa: E402
import render_h as RH  # noqa: E402

MAN['versions'] = RH.versions()
MAN['driver'] = subprocess.check_output(['nvidia-smi', '--query-gpu=driver_version,name,memory.total', '--format=csv,noheader'], text=True).strip()
cam = RH.camera()
RES = 512
rt, s0, B0 = RH.build(split=True)
Bd = RH.R.t2d(B0.view(-1))
B0_np = B0.cpu().numpy().copy()
np.save(RAW / 'table_B.npy', B0_np)
MAN['table_sha256'] = CH.sha(RAW / 'table_B.npy')

# ---------------------------------------------------------------------------------------------------------------- ROIs
cov = RH.coverage(rt, cam, RES, P['rois']['subpixel_grid'])
np.savez_compressed(RAW / 'coverage.npz', **cov)
sid_c, _ = RH.gbuffer(rt, cam, RES, 1)
with np.load(CH.REND100 / 'map_0.npz') as g:
    msid, mok = g['sid'], g['ok']
masks = CH.all_masks(CH.REND100 / 'map_0.npz', cov)
counts = {k: int(masks[k].sum()) for k in P['rois']['expected_counts']}
np.savez_compressed(RAW / 'masks.npz', **masks)
gate('ROI_geometry', counts == P['rois']['expected_counts'] and int((np.where(mok, msid, -1) != sid_c[..., 0]).sum()) == 0,
     {'counts': counts, 'own_vs_map0_sid_mismatch': int((np.where(mok, msid, -1) != sid_c[..., 0]).sum())})
PRIMARY = P['rois']['primary']


def rel_l1(a, b, m=None):
    if m is not None:
        a, b = a[m], b[m]
    return float(np.abs(a.astype(np.float64) - b).sum() / max(np.abs(b.astype(np.float64)).sum(), 1e-300))


# ---------------------------------------------------------------------------------------------------------------- G1
arch = np.load(CH.REND100 / 'table_0.npz')['B']
rep = []
for _ in range(3):
    s2 = rt.source(); B2, _ = rt.respond(s2)
    rep.append({'source_bitwise': bool(torch.equal(s0, s2)), 'table_bitwise': bool(torch.equal(B0, B2))})
g1 = {'table_vs_archive_rel_l1': rel_l1(B0_np, arch), 'repeats': rep}
gate('G1_high_state', g1['table_vs_archive_rel_l1'] <= 1e-4 and all(r['source_bitwise'] and r['table_bitwise'] for r in rep), g1)


def comps_from_sums(sums, spp):
    """[9, P] sums -> dict of [H, W, 3] float64 per-pixel means for e1, e2, hl, low, high, full."""
    x = (np.asarray(sums, np.float64) / spp).reshape(3, 3, RES, RES).transpose(0, 2, 3, 1)
    return {'e1': x[0], 'e2': x[1], 'hl': x[2], 'low': x[0] + x[1], 'high': x[2], 'full': x[0] + x[1] + x[2]}


def roi_rel(a, b, keys_a, keys_b):
    out = {}
    for ka, kb in zip(keys_a, keys_b):
        out[ka] = {'full_frame': rel_l1(a[ka], b[kb])}
        for r in PRIMARY:
            out[ka][r] = rel_l1(a[ka], b[kb], masks[r])
    return out


# ---------------------------------------------------------------------------------------------------------------- G2 / G3
(RAW / 'replay').mkdir(exist_ok=True)
t = time.perf_counter()
pred_s = RH.R.render_hq(rt, Bd, cam, RES, 4096, P['arms']['replays']['prediction']['seed'])
np.save(RAW / 'replay' / 'pred_100903.npy', pred_s)
c = comps_from_sums(pred_s, 4096)
with np.load(CH.REND100 / 'PRED_0.npz') as z:
    pr = {k: z[k].astype(np.float64) for k in z}
pr.update(low=pr['e1'] + pr['e2'], high=pr['hl'], full=pr['e1'] + pr['e2'] + pr['hl'])
g2 = roi_rel(c, pr, ['full', 'low', 'high', 'e1', 'e2'], ['full', 'low', 'high', 'e1', 'e2'])
worst2 = max(v for d in g2.values() for v in d.values())
MAN['stages']['replay_pred_s'] = time.perf_counter() - t
gate('G2_prediction_replay', worst2 <= 1e-4, {'worst_rel_l1': worst2, 'detail': g2})

g3 = {}
for half, seed in enumerate(P['arms']['replays']['paired_low']['seeds']):
    sm = RH.R.render_hq(rt, Bd, cam, RES, 4096, seed)
    np.save(RAW / 'replay' / f'paired_{seed}.npy', sm)
    c = comps_from_sums(sm, 4096)
    with np.load(CH.REND100 / f'REF_0_{half}.npz') as z:
        rf = {k: z[k].astype(np.float64) for k in z}
    rf['low'] = rf['e1'] + rf['e2']
    g3[str(seed)] = roi_rel(c, rf, ['e1', 'e2', 'low'], ['e1', 'e2', 'low'])
worst3 = max(v for h in g3.values() for d in h.values() for v in d.values())
gate('G3_paired_low', worst3 <= 1e-4, {'worst_rel_l1': worst3, 'detail': g3})

# ---------------------------------------------------------------------------------------------------------------- J arm
JA = P['arms']['J_normal']
(RAW / 'joint').mkdir(exist_ok=True)
g5 = {}
for spp in JA['spp']:
    R_ = CFG['replicates']
    ch = np.zeros((R_, 9, RES * RES), np.float32)
    co = np.zeros((R_, 3, RES * RES), np.float32)
    t = time.perf_counter()
    worst = 0.0
    for r in range(R_):
        out = rt.frame(Bd, cam, RES, spp, OFF + 10183000 + 100 * spp + r)
        ch[r] = RH.to_np(out)
        co[r] = RH.compose_contract(out, spp)
        ref64 = ch[r].astype(np.float64).reshape(3, 3, -1).sum(0) / spp
        worst = max(worst, float(np.abs(co[r] - ref64).sum() / np.abs(ref64).sum()))
    MAN['stages'][f'joint_{spp}_s'] = time.perf_counter() - t
    rep0 = RH.to_np(rt.frame(Bd, cam, RES, spp, OFF + 10183000 + 100 * spp))
    gate(f'G4_determinism_joint_{spp}', np.array_equal(rep0, ch[0]), {'spp': spp})
    fin = bool(np.isfinite(ch).all() and np.isfinite(co).all() and ch.min() >= 0)
    g5[spp] = worst
    gate(f'G5_frame_recomposition_{spp}', worst <= 1e-6 and fin, {'worst_rel_l1': worst, 'finite_nonneg': fin})
    np.save(RAW / 'joint' / f'spp{spp}_ch.npy', ch)
    np.save(RAW / 'joint' / f'spp{spp}_comp.npy', co)
    del ch, co
    log('joint done', spp)

# ---------------------------------------------------------------------------------------------------------------- PF arm
PA = P['arms']['PF_split']
(RAW / 'pf').mkdir(exist_ok=True)
for i, spp in enumerate(PA['spp']):
    R_ = CFG['replicates']
    arr = np.zeros((R_, 3, 9, RES * RES), np.float32)
    t = time.perf_counter()
    bad = []
    for r in range(R_):
        b = OFF + 10185000 + 1000 * i + 10 * r
        trip = ((b, b + 1, b + 2), (b, b + 1, b + 3), (b, b + 4, b + 5))
        for j, (sj, sp, ss) in enumerate(trip):
            arr[r, j] = RH.to_np(rt.split_frame(Bd, cam, RES, spp, sj, sp, ss))
        if not (np.array_equal(arr[r, 0, 0:3], arr[r, 1, 0:3]) and np.array_equal(arr[r, 0, 6:9], arr[r, 1, 6:9])):
            bad.append(r)
    MAN['stages'][f'pf_{spp}_s'] = time.perf_counter() - t
    fin = bool(np.isfinite(arr).all() and arr.min() >= 0)
    gate(f'G6_split_structure_{spp}', not bad and fin, {'spp': spp, 'non_identical_triples': bad, 'finite_nonneg': fin})
    b = OFF + 10185000 + 1000 * i
    rep0 = RH.to_np(rt.split_frame(Bd, cam, RES, spp, b, b + 1, b + 2))
    gate(f'G4_determinism_pf_{spp}', np.array_equal(rep0, arr[0, 0]), {'spp': spp})
    np.save(RAW / 'pf' / f'spp{spp}.npy', arr)
    del arr
    log('pf done', spp)

# ---------------------------------------------------------------------------------------------------------------- HQ arm
(RAW / 'hq').mkdir(exist_ok=True)
t = time.perf_counter()
for seed in P['arms']['HQ_normal']['seeds']:
    np.save(RAW / 'hq' / f'hq_{seed}.npy', RH.R.render_hq(rt, Bd, cam, RES, CFG['hq_spp'], OFF + seed))
MAN['stages']['hq_s'] = time.perf_counter() - t
log('hq done')

# ---------------------------------------------------------------------------------------------------------------- Q arm (conditional)
QA = P['arms']['Q_quadrature']
(RAW / 'quad').mkdir(exist_ok=True)
PT = RH.PT
US0, UP0 = list(rt.US), list(rt.UP)
N, K, PSRC = rt.N, RH.R.K, RH.R.PSRC
qseed = OFF + 10189401
base = RH.R.render_hq(rt, Bd, cam, RES, CFG['q_spp'], qseed)
np.save(RAW / 'quad' / 'base.npy', base)
qrep = {}
for kind, key in (('source', 'source_offset_redraws'), ('state', 'state_offset_redraws')):
    for sd in QA[key]['rng_seeds']:
        g = np.random.default_rng(sd + OFF)
        if kind == 'source':
            u = g.random((2, 1, PSRC), np.float32)
            rt.UP = [PT.fl(np.broadcast_to(u[k], (N, PSRC))) for k in range(2)]
            RH.dr.eval(rt.UP)
        else:
            u = g.random((4, 1, K), np.float32)
            rt.US = [PT.fl(np.broadcast_to(u[k], (N, K))) for k in range(4)]
            RH.dr.eval(rt.US)
            rt.rebuild_state()
        sq = rt.source(); Bq, _ = rt.respond(sq)
        Bq_np = Bq.cpu().numpy().copy()
        hq = RH.R.render_hq(rt, RH.R.t2d(Bq.view(-1)), cam, RES, CFG['q_spp'], qseed)
        crn = bool(np.array_equal(hq[0:6], base[0:6]))
        np.save(RAW / 'quad' / f'{kind}_{sd}_hl.npy', hq[6:9]); np.save(RAW / 'quad' / f'{kind}_{sd}_B.npy', Bq_np)
        qrep[f'{kind}_{sd}'] = {'table_rel_l1': rel_l1(Bq_np, B0_np), 'table_rel_l2': float(np.linalg.norm(Bq_np - B0_np) / np.linalg.norm(B0_np)),
                                'low_crn_bitwise': crn}
        rt.US, rt.UP = list(US0), list(UP0)
        rt.rebuild_state()
s_r = rt.source(); B_r, _ = rt.respond(s_r)
MAN['stages']['quad'] = qrep
gate('Q_restore_and_crn', bool(torch.equal(B_r, B0)) and all(v['low_crn_bitwise'] for v in qrep.values()),
     {'restored_bitwise': bool(torch.equal(B_r, B0)), 'crn': {k: v['low_crn_bitwise'] for k, v in qrep.items()}})
Bd = RH.R.t2d(B0.view(-1))

# ---------------------------------------------------------------------------------------------------------------- timing
def gpu_util():
    try:
        return subprocess.check_output(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used', '--format=csv,noheader'], text=True).strip()
    except Exception as e:  # noqa: BLE001
        return str(e)


tim = {'gpu_before': gpu_util()}
for spp in (1, 4, 16):
    rec = {k: [] for k in ('frame', 'frame_readback', 'readback', 'compose', 'split_frame_readback')}
    for it in range(CFG['timing_iters']):
        sd = OFF + 10189500 + 100 * spp + it
        RH.R.sync(); t0 = time.perf_counter(); out = rt.frame(Bd, cam, RES, spp, sd); RH.R.sync(); t1 = time.perf_counter()
        x = RH.to_np(out); t2 = time.perf_counter()
        y = RH.compose_contract(out, spp); RH.R.sync(); t3 = time.perf_counter()
        z = RH.to_np(rt.split_frame(Bd, cam, RES, spp, sd, sd + 1, sd + 2)); t4 = time.perf_counter()
        if it >= 4:
            rec['frame'].append((t1 - t0) * 1e3); rec['frame_readback'].append((t2 - t0) * 1e3); rec['readback'].append((t2 - t1) * 1e3)
            rec['compose'].append((t3 - t2) * 1e3); rec['split_frame_readback'].append((t4 - t3) * 1e3)
    tim[spp] = {k: {'median_ms': float(np.median(v)), 'p25_ms': float(np.percentile(v, 25)), 'p75_ms': float(np.percentile(v, 75)), 'n': len(v)}
                for k, v in rec.items()}
tim['gpu_after'] = gpu_util()
MAN['stages']['timing'] = tim
B_end = rt.respond(rt.source())[0]
gate('G1_table_unchanged_after_frames', bool(torch.equal(B_end, B0)), {})
MAN['finished'] = time.strftime('%Y-%m-%d %H:%M:%S')
MAN['raw_sha256'] = {str(p.relative_to(RAW)).replace('\\', '/'): CH.sha(p) for p in sorted(RAW.rglob('*.np[yz]'))}
save_manifest()
log('RUN COMPLETE')
LOG.close()
os._exit(0)
