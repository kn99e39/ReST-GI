"""WL101 arm A (frozen stack) timing + equivalence dumps.
  python run_a101.py <workload> <A_sel|A_full> <out_stem> [--scale=s] [--nodump]
Writes <out_stem>.json (timings) and, for A_sel validation workloads, <out_stem>_dump.npz (state digests, tables, same-seed frames).
"""
import hashlib, json, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import seq101 as Q  # noqa: E402

wl, arm, stem = sys.argv[1], sys.argv[2], sys.argv[3]
opt = {a.split('=')[0]: a.split('=')[1] for a in sys.argv[4:] if '=' in a}
scale = float(opt.get('--scale', 1.0))
dump = arm == 'A_sel' and '--nodump' not in sys.argv and scale == 1.0
import common101 as K  # noqa: E402
Q.refine(K.S, scale)
torch, mi, dr = K.torch, K.mi, K.dr
snap0 = Q.gpu_snapshot()
case, start, ev = Q.workload(wl)
F = [(f['res'], f['spp']) for f in [Q.P['frame']['primary']] + Q.P['frame']['secondary']]
if scale != 1.0:
    F = F[:1]
free0 = torch.cuda.mem_get_info()[0]
model, ck = K.load_model(case)
A = K.FrozenECRO(case, start[0], model)
dr.set_flag(dr.JitFlag.KernelHistory, True)
cams = {r: K.cam_of(case, r) for r in {r for r, _ in F}}
rec = {'workload': wl, 'arm': arm, 'case': case, 'scale': scale, 'N': int(A.ctx.N), 'state_rays': int(A.ctx.N * K.S.K), 'checkpoint_sha256': ck,
       'frozen_sources': K.frozen_hashes(), 'versions': {'mitsuba': mi.__version__, 'drjit': dr.__version__, 'torch': torch.__version__},
       'gpu_name': torch.cuda.get_device_name(), 'snapshot_before': snap0, 'events': []}
D = {}
digests = {}
for i, (g, L, sid) in enumerate(ev):
    warm = i < Q.WARM
    T = K.Timer()
    ce = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
    K.sync(); t0 = time.perf_counter()
    d = A.edit(g, L, T, full_state=(arm == 'A_full'))
    res, spp = F[0]
    A.frame(d['lt'], d['Bd'], cams[res], res, spp, 9000 + i, T)
    K.sync(); total = (time.perf_counter() - t0) * 1e3
    e = {'i': i, 'state': sid, 'warm': warm, 'total_ms': total, 'parts_ms': dict(T.t), 'drjit_gpu_ms': dict(T.g), 'dirty_rays': d['dirty_rays'],
         'moved': d['moved'], 'secondary_frame_ms': {}}
    for (r2, s2) in F[1:]:
        T2 = K.Timer()
        A.frame(d['lt'], d['Bd'], cams[r2], r2, s2, 9100 + i, T2)
        e['secondary_frame_ms'][f'{r2}x{s2}'] = T2.t['frame']
    # ---- untimed checks / dumps
    dg = hashlib.sha256(A.st.hit.tobytes() + A.st.dbin.tobytes()).hexdigest()[:16]
    e['state_digest'] = dg
    e['revisit_digest_equal'] = (digests[sid] == dg) if sid in digests else None
    digests.setdefault(sid, dg)
    if arm == 'A_sel' and not warm and scale == 1.0:
        stf = K.EC.build_state(A.ctx)
        e['selective_equals_full'] = bool(np.array_equal(stf.hit, A.st.hit) and np.array_equal(stf.dbin, A.st.dbin))
    if dump and not warm:
        D[f'e{i}_B'] = d['B']; D[f'e{i}_hit'] = A.st.hit; D[f'e{i}_dbin'] = A.st.dbin; D[f'e{i}_src'] = d['s']
        if i < Q.WARM + 4:
            for s3 in (1, 4):
                acc, _ = A.frame(d['lt'], d['Bd'], cams[512], 512, s3, 7000 + i, K.Timer())
                for k, v in acc.items():
                    D[f'e{i}_f{s3}_{k}'] = v.astype(np.float32)
    rec['events'].append(e)
    if i % 8 == 0 or i == len(ev) - 1:
        print(wl, arm, i, sid, round(total, 2), {k: round(v, 2) for k, v in T.t.items()}, d['dirty_rays'], flush=True)
# ---- static frames (no edit)
rec['static_frame_ms'] = {}
for (r2, s2) in F:
    ts = []
    for j in range(Q.WARM + 24):
        T2 = K.Timer()
        A.frame(d['lt'], d['Bd'], cams[r2], r2, s2, 9500 + j, T2)
        if j >= Q.WARM:
            ts.append(T2.t['frame'])
    rec['static_frame_ms'][f'{r2}x{s2}'] = Q.stats(ts)
rec['memory'] = {'torch_peak_mib': torch.cuda.max_memory_allocated() / 2 ** 20, 'device_used_delta_mib': (free0 - torch.cuda.mem_get_info()[0]) / 2 ** 20,
                 'state_bytes': int(A.st.hit.nbytes + A.st.dbin.nbytes + A.st.seg.nbytes)}
rec['snapshot_after'] = Q.gpu_snapshot()
tim = [e for e in rec['events'] if not e['warm']]
rec['summary'] = {'total_ms': Q.stats([e['total_ms'] for e in tim]),
                  'parts_ms': {k: Q.stats([e['parts_ms'][k] for e in tim]) for k in tim[0]['parts_ms']},
                  'drjit_gpu_ms': {k: Q.stats([e['drjit_gpu_ms'][k] for e in tim]) for k in tim[0]['drjit_gpu_ms']},
                  'dirty_rays': Q.stats([e['dirty_rays'] for e in tim]),
                  'dynamic_frame_ms': {f'{r}x{s}': Q.stats([sum(v for k, v in e['parts_ms'].items() if k != 'frame') +
                                                            (e['parts_ms']['frame'] if (r, s) == F[0] else e['secondary_frame_ms'][f'{r}x{s}'])
                                                            for e in tim]) for (r, s) in F}}
os.makedirs(os.path.dirname(stem) or '.', exist_ok=True)
json.dump(rec, open(stem + '.json', 'w'), indent=1)
if dump:
    D['meta'] = json.dumps({'workload': wl, 'case': case, 'checkpoint_sha256': ck})
    np.savez_compressed(stem + '_dump.npz', **D)
print('DONE', wl, arm, json.dumps({k: round(v['median'], 2) for k, v in rec['summary']['dynamic_frame_ms'].items()}), flush=True)
K.os._exit(0)
