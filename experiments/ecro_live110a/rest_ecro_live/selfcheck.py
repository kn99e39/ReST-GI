"""Cross-backend self-check: replay the WL101 V_SSS_ANIM event sequence (48 events after 4 warm-up) through the WL110 engine on
the selected backend and compare with the CUDA reference recorded on LabServer69 (reference/wl110a_reference.json + .npz,
tools/make_reference.py).

Predeclared tolerances (the WL101 cross-stack tolerances; fixed before any LLVM / macOS result):
  table B          relative L1 <= 1e-4 on the stored events
  composed frame   mean relative |difference| <= 1e-4 (same seed, 128^2 x 1 spp) on the stored events
Reported, not gated: state-digest equality per event (ray-hit ties may differ between OptiX and Embree), dirty-ray counts,
table / image sums on every event, per-event compute time.
"""
import json, time
from pathlib import Path

REF = Path(__file__).resolve().parent / 'reference'
TOL = {'table_rel_l1': 1e-4, 'frame_mean_rel': 1e-4}


def replay(be, L, eng, events, keep):
    import numpy as np
    rows, kept = [], {}
    for i, (g, light, sid) in enumerate(events):
        P = L.Parts()
        L.sync(); t0 = time.perf_counter()
        r = eng.step(g, 9000 + i, P)
        ms = (time.perf_counter() - t0) * 1e3
        pr = eng.proof(r)
        rows.append({'i': i, 'pose': sid, 'digest': eng.state_digest(), 'dirty': r['dirty'], 'table_sum': pr['table_sum'],
                     'table_abs_sum': float(r['B'].abs().double().sum()), 'hdr_mean': float(r['hdr'].double().mean()), 'compute_ms': ms})
        if i in keep:
            kept[i] = {'B': r['B'].detach().cpu().numpy().astype(np.float32), 'hdr': r['hdr'].detach().cpu().numpy().astype(np.float32)}
    return rows, kept


def run(args):
    import numpy as np
    from . import runtime, backend
    from .app import build_engine
    root = runtime.find_root()
    auth = runtime.authenticate(root)
    runtime.add_import_paths(root)
    be = backend.configure(args.backend, args.generator_device, freeze=args.freeze)
    import seq101 as Q
    ref = json.loads((REF / 'wl110a_reference.json').read_text())
    arr = np.load(REF / 'wl110a_reference.npz')
    case, start, events = Q.workload('V_SSS_ANIM')
    L, eng = build_engine(be, case, ref['res'], 1, 6.0, args.relation)
    assert L.A.geom_hash(start[0]) == L.A.geom_hash(eng.anim.pose(0.0)[0])
    keep = ref['stored_events']
    rows, kept = replay(be, L, eng, events, keep)
    per = []
    for i in keep:
        Bc, Hc = arr[f'B_{i}'], arr[f'hdr_{i}']
        Bx, Hx = kept[i]['B'], kept[i]['hdr']
        per.append({'event': i, 'table_rel_l1': float(np.abs(Bx - Bc).sum() / np.abs(Bc).sum()),
                    'frame_mean_rel': float(np.abs(Hx - Hc).mean() / np.abs(Hc).mean()),
                    'frame_pixels_differing': int((np.abs(Hx - Hc) > 1e-6 * max(1.0, float(np.abs(Hc).max()))).any(0).sum())})
    R0 = ref['rows']
    res = {
        'backend': be.name, 'variant': be.variant, 'generator_device': be.generator_device,
        'generator_check': getattr(eng, 'generator_check', None), 'relation': eng.relation_mode, 'info': be.info, 'runtime': auth,
        'tolerances': TOL, 'stored_events': per,
        'pass': all(p['table_rel_l1'] <= TOL['table_rel_l1'] and p['frame_mean_rel'] <= TOL['frame_mean_rel'] for p in per),
        'state_digest_equal': sum(a['digest'] == b['digest'] for a, b in zip(rows, R0)),
        'state_digest_equal_archived_wl101': sum(a['digest'] == b['wl101_digest'] for a, b in zip(rows, R0)),
        'dirty_equal': sum(a['dirty'] == b['dirty'] for a, b in zip(rows, R0)), 'events': len(rows),
        'table_sum_max_rel': max(abs(a['table_sum'] - b['table_sum']) / abs(b['table_sum']) for a, b in zip(rows, R0)),
        'hdr_mean_max_rel': max(abs(a['hdr_mean'] - b['hdr_mean']) / abs(b['hdr_mean']) for a, b in zip(rows, R0)),
        'compute_ms_median_after_warmup': float(np.median([r['compute_ms'] for r in rows[Q.WARM:]])),
        'reference': {k: ref[k] for k in ('created', 'backend', 'gpu', 'commit')},
    }
    out = args.out or 'selfcheck.json'
    Path(out).write_text(json.dumps({'result': res, 'rows': rows}, indent=1))
    print('SELFCHECK', 'PASS' if res['pass'] else 'FAIL', json.dumps({k: res[k] for k in (
        'backend', 'generator_device', 'relation', 'stored_events', 'state_digest_equal', 'state_digest_equal_archived_wl101', 'dirty_equal',
        'table_sum_max_rel', 'hdr_mean_max_rel', 'compute_ms_median_after_warmup')}), flush=True)
    return res
