"""WL101 arm B (optimized stack) timing + correctness checks.
  python run_b101.py <workload> <B_sel|B_full|B_eager> <out_stem> [--scale=s] [--cmp=<A dump npz>]
"""
import hashlib, json, os, sys, time
import numpy as np

arm = sys.argv[2]
os.environ['FREEZE'] = '0' if arm == 'B_eager' else '1'
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import seq101 as Q  # noqa: E402

wl, stem = sys.argv[1], sys.argv[3]
opt = {a.split('=')[0]: a.split('=')[1] for a in sys.argv[4:] if '=' in a}
scale = float(opt.get('--scale', 1.0))
import rt101 as R  # noqa: E402
Q.refine(R.S, scale)
torch, mi, dr = R.torch, R.mi, R.dr
snap0 = Q.gpu_snapshot()
case, start, ev = Q.workload(wl)
F = [(f['res'], f['spp']) for f in [Q.P['frame']['primary']] + Q.P['frame']['secondary']]
if scale != 1.0:
    F = F[:1]
checks = arm == 'B_sel' and scale == 1.0
Z = np.load(opt['--cmp']) if '--cmp' in opt else None
free0 = torch.cuda.mem_get_info()[0]
model = R.load_model(case)
cams = {r: dict(R.C100.P['datasets'][case]['camera'], res=r) for r in {r for r, _ in F}}
t_init = time.perf_counter()
rt = R.RT(start[0], case, model, start[1])
R.sync(); t_init = (time.perf_counter() - t_init) * 1e3
dr.set_flag(dr.JitFlag.KernelHistory, True)
rec = {'workload': wl, 'arm': arm, 'case': case, 'scale': scale, 'N': int(rt.N), 'state_rays': int(rt.N * R.K), 'freeze': R.FREEZE,
       'checkpoint_sha256': hashlib.sha256(open(R.ROOT100 / case / 'model.pt', 'rb').read()).hexdigest(),
       'versions': {'mitsuba': mi.__version__, 'drjit': dr.__version__, 'torch': torch.__version__}, 'gpu_name': torch.cuda.get_device_name(),
       'init_ms': t_init, 'snapshot_before': snap0, 'events': []}


class T:
    def __init__(self):
        self.t = {}; self.g = {}; self.cu = {}

    def __call__(self, name, fn):
        R.sync(); dr.kernel_history()
        a = torch.cuda.Event(enable_timing=True); b = torch.cuda.Event(enable_timing=True)
        t0 = time.perf_counter(); a.record()
        r = fn()
        b.record(); R.sync()
        self.t[name] = (time.perf_counter() - t0) * 1e3
        self.g[name] = sum(float(h.get('execution_time', 0.0)) for h in dr.kernel_history())
        self.cu[name] = a.elapsed_time(b)
        return r


def rel(a, b):
    a = np.asarray(a, np.float64); b = np.asarray(b, np.float64)
    return float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-30))


digests = {}
prevB = None
Bd = None
for i, (g, L, sid) in enumerate(ev):
    warm = i < Q.WARM
    tm = T()
    R.sync(); t0 = time.perf_counter()
    moved, g_old = tm('geometry_update', lambda: rt.set_geom(g))
    if arm == 'B_full':
        nd = tm('relation_update', lambda: rt.rebuild_state())
    else:
        nd = tm('relation_update', lambda: rt.update_state(g_old, moved))
    tm('lights', lambda: rt.set_light(L))
    s = tm('source', lambda: rt.source())
    B, _ = tm('generator_apply', lambda: rt.respond(s))
    Bd = tm('table_handoff', lambda: R.t2d(B.view(-1)))
    res, spp = F[0]
    tm('frame', lambda: rt.frame(Bd, cams[res], res, spp, 9000 + i))
    R.sync(); total = (time.perf_counter() - t0) * 1e3
    e = {'i': i, 'state': sid, 'warm': warm, 'total_ms': total, 'parts_ms': dict(tm.t), 'drjit_gpu_ms': dict(tm.g), 'cuda_event_ms': dict(tm.cu),
         'dirty_rays': int(nd), 'moved': moved, 'secondary_frame_ms': {}}
    for (r2, s2) in F[1:]:
        t2 = T()
        t2('frame', lambda: rt.frame(Bd, cams[r2], r2, s2, 9100 + i))
        e['secondary_frame_ms'][f'{r2}x{s2}'] = t2.t['frame']
    # ---- untimed checks
    hb = rt.hit.cpu().numpy().astype(np.int16); db = rt.dbin.cpu().numpy().astype(np.uint8)
    dg = hashlib.sha256(hb.tobytes() + db.tobytes()).hexdigest()[:16]
    e['state_digest'] = dg
    e['revisit_digest_equal'] = (digests[sid] == dg) if sid in digests else None
    digests.setdefault(sid, dg)
    Bn = B.cpu().numpy()
    if prevB is not None:
        e['table_changed_vs_previous'] = bool(not np.array_equal(Bn, prevB))
    prevB = Bn
    if checks and not warm:
        h0, d0, s0 = rt.hit.clone(), rt.dbin.clone(), rt.seg.clone()
        rt.rebuild_state()
        e['selective_equals_full'] = bool(torch.equal(h0, rt.hit) and torch.equal(d0, rt.dbin))
        rt.hit, rt.dbin, rt.seg = h0, d0, s0
        if Z is not None and f'e{i}_B' in Z.files:
            e['xstack_hit_mismatch'] = int((hb != Z[f'e{i}_hit']).sum())
            e['xstack_dbin_mismatch'] = int((db != Z[f'e{i}_dbin']).sum())
            e['xstack_src_rel_max'] = rel(s[0].cpu().numpy(), Z[f'e{i}_src'])
            Ba = Z[f'e{i}_B']
            e['xstack_table_rel_l1'] = float(np.abs(Bn - Ba).sum() / Ba.sum()); e['xstack_table_rel_max'] = rel(Bn, Ba)
            for s3 in (1, 4):
                keys = sorted(k for k in Z.files if k.startswith(f'e{i}_f{s3}_'))
                if not keys:
                    continue
                f = rt.frame(Bd, cams[512], 512, s3, 7000 + i)
                names = sorted(k.split('_', 2)[2] for k in keys)
                fr = {}
                for nm, v in zip(names, f):
                    a = Z[f'e{i}_f{s3}_{nm}']; b = np.asarray(v)
                    fr[nm] = float(np.abs(a - b).mean() / max(np.abs(a).mean(), 1e-30))
                e[f'xstack_frame{s3}_mean_rel'] = fr
                if s3 == 1 and R.FREEZE:            # replay vs eager re-trace of the same program
                    fe = rt._frame_prog(rt.scene, rt.tabs(), rt.light_args(), Bd, dr.opaque(mi.UInt64, (7000 + i) * 1000003), (dict(cams[512]), 512, 1))
                    dr.eval(fe)
                    e['replay_vs_eager_max_abs'] = float(max(np.abs(np.asarray(x) - np.asarray(y)).max() for x, y in zip(f, fe)))
    rec['events'].append(e)
    if i % 8 == 0 or i == len(ev) - 1:
        print(wl, arm, i, sid, round(total, 2), {k: round(v, 2) for k, v in tm.t.items()}, nd, flush=True)
rec['static_frame_ms'] = {}
for (r2, s2) in F:
    ts = []
    for j in range(Q.WARM + 24):
        t2 = T()
        t2('frame', lambda: rt.frame(Bd, cams[r2], r2, s2, 9500 + j))
        if j >= Q.WARM:
            ts.append(t2.t['frame'])
    rec['static_frame_ms'][f'{r2}x{s2}'] = Q.stats(ts)
rec['recordings'] = {'state': getattr(rt._f_state, 'n_recordings', None), 'source': getattr(rt._f_source, 'n_recordings', None),
                     'frames': {f'{k[0]}x{k[1]}': getattr(v, 'n_recordings', None) for k, v in rt._frames.items()}}
rec['memory'] = {'torch_peak_mib': torch.cuda.max_memory_allocated() / 2 ** 20, 'device_used_delta_mib': (free0 - torch.cuda.mem_get_info()[0]) / 2 ** 20,
                 'state_bytes_resident': int(rt.hit.numel() * 4 + rt.dbin.numel() * 4 + rt.seg.numel() * 4)}
rec['snapshot_after'] = Q.gpu_snapshot()
tim = [e for e in rec['events'] if not e['warm']]
rec['summary'] = {'total_ms': Q.stats([e['total_ms'] for e in tim]),
                  'parts_ms': {k: Q.stats([e['parts_ms'][k] for e in tim]) for k in tim[0]['parts_ms']},
                  'drjit_gpu_ms': {k: Q.stats([e['drjit_gpu_ms'][k] for e in tim]) for k in tim[0]['drjit_gpu_ms']},
                  'cuda_event_ms': {k: Q.stats([e['cuda_event_ms'][k] for e in tim]) for k in tim[0]['cuda_event_ms']},
                  'dirty_rays': Q.stats([e['dirty_rays'] for e in tim]),
                  'dynamic_frame_ms': {f'{r}x{s}': Q.stats([sum(v for k, v in e['parts_ms'].items() if k != 'frame') +
                                                            (e['parts_ms']['frame'] if (r, s) == F[0] else e['secondary_frame_ms'][f'{r}x{s}'])
                                                            for e in tim]) for (r, s) in F}}
os.makedirs(os.path.dirname(stem) or '.', exist_ok=True)
json.dump(rec, open(stem + '.json', 'w'), indent=1)
print('DONE', wl, arm, json.dumps({k: round(v['median'], 2) for k, v in rec['summary']['dynamic_frame_ms'].items()}), flush=True)
os._exit(0)
