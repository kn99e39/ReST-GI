"""Record the CUDA reference for the cross-backend self-check (run once on the CUDA machine, LabServer69):
  python tools/make_reference.py            -> rest_ecro_live/reference/wl110a_reference.{json,npz}
Same engine, same V_SSS_ANIM sequence and seeds as selfcheck.py, at 128^2 x 1 spp; the archived WL101 state digests are attached.
"""
import json, os, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from rest_ecro_live import runtime, backend  # noqa: E402

RES, KEEP = 128, [4, 20, 40]


def main():
    import numpy as np
    root = runtime.find_root()
    auth = runtime.authenticate(root)
    assert auth['ok'], auth
    runtime.add_import_paths(root)
    be = backend.configure('cuda', 'cuda')
    from rest_ecro_live.app import build_engine
    from rest_ecro_live.selfcheck import replay, REF
    import seq101 as Q
    case, start, events = Q.workload('V_SSS_ANIM')
    L, eng = build_engine(be, case, RES, 1, 6.0)
    rows, kept = replay(be, L, eng, events, KEEP)
    arch = Path(os.environ.get('WL101_EVIDENCE', str(Path.home() / 'rest_dyn101/results/ecro_dyn101/evidence'))) / 'timing' / 'V_SSS_ANIM_B_sel_r1.json'
    A = json.loads(arch.read_text())['events']
    for r, a in zip(rows, A):
        assert r['pose'] == a['state']
        r['wl101_digest'] = a['state_digest']
    REF.mkdir(exist_ok=True)
    np.savez_compressed(REF / 'wl110a_reference.npz', **{f'B_{i}': kept[i]['B'] for i in KEEP}, **{f'hdr_{i}': kept[i]['hdr'] for i in KEEP})
    meta = {'created': time.strftime('%Y-%m-%d %H:%M:%S %Z'), 'backend': be.name, 'gpu': be.info['device_name'], 'info': be.info,
            'commit': subprocess.run(['git', '-C', str(HERE), 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip(),
            'res': RES, 'spp': 1, 'seeds': '9000 + event index', 'stored_events': KEEP, 'workload': 'WL101 V_SSS_ANIM (seq101)',
            'digest_equal_archived_wl101': sum(r['digest'] == r['wl101_digest'] for r in rows), 'rows': rows}
    (REF / 'wl110a_reference.json').write_text(json.dumps(meta, indent=1))
    print('REFERENCE', meta['digest_equal_archived_wl101'], '/', len(rows), 'digests equal archived WL101', flush=True)
    os._exit(0)


if __name__ == '__main__':
    main()
