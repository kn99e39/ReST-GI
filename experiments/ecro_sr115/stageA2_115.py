"""WL115 Stage A2 (post hoc mechanism diagnostic; SELECTION capture views only): angular sub-bin refinement of the U measure on the
diagnostic contact neighbourhood at 16 x 16 spatial cells (Q16R ~ Q32R in Stage A, so 16 x 16 isolates the angular part).

  python stageA2_115.py con ROOT       16 x 16 x (64 bins x 4 x 4 sub-bins) deep U construction per capture-view geometry (n = 256)
  python stageA2_115.py streams ROOT   readout of R16n (sub-bins pooled: nested control), R16A2, R16A4 (U elsewhere: PC4) and
                                       Q16RU (camera-conditioned 16 x 16 on the contact patches, U PC4 elsewhere) with the WL114 seeds
"""
import json, os, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common115 as C  # noqa: E402
import tracer115 as T  # noqa: E402
import tracerA2_115 as A2  # noqa: E402
import stageA115 as SA  # noqa: E402

PT, dr = T.PT, T.dr
SPP = C.dv(4096, 64)
NA2 = C.dv(256, 2)
SEED_A2 = 11513000


def con(root):
    d = root / 'stageA2' / 'con'; d.mkdir(parents=True, exist_ok=True)
    V = SA.views()
    for i, tag in enumerate(SA.CAPTURE_VIEWS):
        key, dg, g, l, cam, seeds = V[tag]
        p = d / f'ca_{dg}.npz'
        if p.exists():
            continue
        ctx = A2.CtxA2(g, 'C')
        pids = T.contact_patches(ctx.G, g['S']['c'][:2])
        acc, diag = ctx.construct_ka(PT.Lights(l), NA2, SEED_A2 + 100 * i + C.SO, 23, 16, 4, pids, b=C.dv(2, 1), verbose=True)
        np.savez_compressed(p, meta=json.dumps({'tag': tag, 'digest': dg, 'seed': SEED_A2 + 100 * i + C.SO, 'n': NA2, 'K': 16, 'Ka': 4, 'diag': diag}),
                            pids=pids, w=(acc['w0'] + acc['w1']).astype(np.float32),
                            **{f'HF{c}': (acc[f'HF0{c}'] + acc[f'HF1{c}']).astype(np.float32) for c in range(3)})
        print('CONA2', tag, round(diag['seconds'], 1), 'mismatch', diag['mismatch'] / max(diag['counted'], 1), flush=True)
        del ctx, acc


def streams(root):
    d = root / 'stageA2' / 'streams'; d.mkdir(parents=True, exist_ok=True)
    V = SA.views()
    for tag in SA.CAPTURE_VIEWS:
        key, dg, g, l, cam, seeds = V[tag]
        todo = [(j, s) for j, s in enumerate(seeds) if not (d / f'stream_{tag}_{j}.npz').exists()]
        if not todo:
            continue
        with np.load(root / 'stageA2' / 'con' / f'ca_{dg}.npz') as z:
            pids = z['pids']; w = z['w'].astype(np.float64); L = np.stack([z[f'HF{c}'] for c in range(3)], -1).astype(np.float64)
        O = SA.oracle_tables(root, dg)
        with np.load(root / 'stageA' / 'capture' / f'cap_{tag}.npz') as zc:
            cap = {k: zc[k] for k in zc.files}
        Q = SA.query_tables(cap, O, O)
        reps = {}
        for Ka in (1, 2, 4):
            Tk = A2.table_ka(A2.pool_ang(w, 4, Ka), A2.pool_ang(L, 4, Ka), 16)
            reps[f'R16A{Ka}' if Ka > 1 else 'R16n'] = A2.pca_hybrid(O['PC4'], 4, Tk, 16, Ka, pids)
        reps['Q16RU'] = T.pc_hybrid(O['PC4'], 4, Q['Q16'], 16, pids)
        ctx = A2.CtxA2(g, 'C'); light = PT.Lights(l)
        for j, s in todo:
            t0 = time.perf_counter()
            img = ctx.readout_multi(light, dict(cam, res=C.RES), C.RES, SPP, s, reps, b=64)
            rec = {'ro_e1': img['e1'].astype(np.float32), 'ro_e2': img['e2'].astype(np.float32)}
            for nm in reps:
                rec[f'hl|{nm}'] = img[nm].astype(np.float32)
            np.savez_compressed(d / f'stream_{tag}_{j}.npz', meta=json.dumps({'tag': tag, 'seed': s, 'reps': list(reps), 'seconds': time.perf_counter() - t0}), **rec)
            print('STREAMA2', tag, j, round(time.perf_counter() - t0, 1), flush=True)
        del ctx


if __name__ == '__main__':
    cmd, root = sys.argv[1:3]
    {'con': con, 'streams': streams}[cmd](Path(root))
    sys.stdout.flush(); os._exit(0)
