"""WL115 Stage B - independent qualification of the frozen candidate oracle CAR-16 (HELD_SEEN + FRESH configurations only).

  python stageB115.py con ROOT [KEY ...]      per unique evaluation geometry: CAR-16 support, 16 x 16 deep U construction on the support
                                              patches (n = 1,024); FRESH geometries also get the dense 4 x 4 U base (n = 2,048)
  python stageB115.py streams ROOT [TAG ...]  readout of PC4_O (dense 4 x 4 oracle) and CAR_O (candidate oracle) with the reference seeds
Dense base: HELD_SEEN = the WL114 S4_F oracle (replica mean), FRESH = the new 4 x 4 construction.
"""
import json, os, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / 'ecro_fd114'))
import common115 as C  # noqa: E402
import tracer115 as T  # noqa: E402
import candidate115 as CA  # noqa: E402
import ref115 as RF  # noqa: E402

PT, dr = T.PT, T.dr
SPP = C.dv(4096, 64)
NFINE = C.dv(1024, 2)
NDENSE = C.dv(2048, 2)
SEED_FINE, SEED_DENSE = 11516000, 11517000


def eval_views():
    """tag -> dict(key, cls, geometry, light, camera, seeds, ref_dir)"""
    import eval114 as EV
    out = {}
    for key, (g, l, vl, _) in EV.states().items():
        if key in C.HELD_FIX or any(key.startswith(s + '_') for s in C.HELD_SEQ):
            for vn, cam, seeds in vl:
                out[f'{key}_{vn}'] = {'key': key, 'cls': 'HELD_SEEN', 'g': g, 'l': l, 'cam': cam, 'seeds': seeds, 'ref': C.WL114_ROOT / 'stageB'}
    for tag, (key, g, l, cam, seeds) in RF.fresh_views().items():
        out[tag] = {'key': key, 'cls': 'FRESH', 'g': g, 'l': l, 'cam': cam, 'seeds': seeds, 'ref': None}
    return out


def geoms():
    seen = {}
    for tag, v in eval_views().items():
        d = C.gdigest(v['g'])
        if d not in seen:
            seen[d] = (v['key'], v['cls'], v['g'], v['l'])
    return sorted(((k, d, c, g, l) for d, (k, c, g, l) in seen.items()), key=lambda x: x[0])


def con(root, only=None):
    d = root / 'stageB' / 'con'; d.mkdir(parents=True, exist_ok=True)
    for i, (key, dg, cls, g, l) in enumerate(geoms()):
        if only and not any(key.startswith(o) for o in only):
            continue
        p = d / f'car_{dg}.npz'
        if p.exists():
            continue
        ctx = T.Ctx115(g, 'C'); light = PT.Lights(l)
        t0 = time.perf_counter()
        w, pids, m = CA.support(ctx)
        acc, dgF = ctx.construct_k(light, NFINE, SEED_FINE + 100 * i + C.SO, 23, 16, pids=pids, b=C.dv(6, 1))
        rec = {'w': w, 'm': m, 'pids': pids, 'f_w': (acc['w0'] + acc['w1']).astype(np.float32),
               **{f'f_HF{c}': (acc[f'HF0{c}'] + acc[f'HF1{c}']).astype(np.float32) for c in range(3)}}
        meta = {'key': key, 'cls': cls, 'digest': dg, 'seed_fine': SEED_FINE + 100 * i + C.SO, 'n_fine': NFINE, 'support_cells': int((w > 0).sum()),
                'support_patches': int(len(pids)), 'diag_fine': dgF}
        if cls == 'FRESH':
            accD, dgD = ctx.construct_k(light, NDENSE, SEED_DENSE + 100 * i + C.SO, 23, 4, b=C.dv(8, 1))
            rec.update({'d_w': (accD['w0'] + accD['w1']).astype(np.float32), **{f'd_HF{c}': (accD[f'HF0{c}'] + accD[f'HF1{c}']).astype(np.float32) for c in range(3)}})
            meta.update({'seed_dense': SEED_DENSE + 100 * i + C.SO, 'n_dense': NDENSE, 'diag_dense': dgD})
        meta['seconds'] = time.perf_counter() - t0
        tmp = p.with_name(p.stem + '.tmp.npz')
        np.savez_compressed(tmp, meta=json.dumps(meta), **rec); os.replace(tmp, p)
        print('CONB', key, cls, meta['support_cells'], meta['support_patches'], round(meta['seconds'], 1),
              'mismatch', dgF['mismatch'] / max(dgF['counted'], 1), flush=True)
        del ctx, acc


def tables(root, v):
    """(dense T4 [N,16,NB,3], CAR-16 inputs) of the view's geometry."""
    dg = C.gdigest(v['g'])
    with np.load(root / 'stageB' / 'con' / f'car_{dg}.npz') as z:
        r = {k: z[k] for k in z.files}
    if v['cls'] == 'HELD_SEEN':
        import eval114 as EV
        T4 = EV.oracle_dense(C.WL114_ROOT, v['key'])['S4_F']
    else:
        T4 = T.table_from_sums(r['d_w'].astype(np.float64), np.stack([r[f'd_HF{c}'] for c in range(3)], -1).astype(np.float64), 4)
    T16 = T.table_from_sums(r['f_w'].astype(np.float64), np.stack([r[f'f_HF{c}'] for c in range(3)], -1).astype(np.float64), 16)
    return T4, T16, r['w'], r['pids']


def streams(root, only=None):
    d = root / 'stageB' / 'streams'; d.mkdir(parents=True, exist_ok=True)
    V = eval_views()
    cur = None; ctx = None
    for tag, v in V.items():
        if only and not any(tag.startswith(o) for o in only):
            continue
        todo = [(j, s) for j, s in enumerate(v['seeds']) if not (d / f'stream_{tag}_{j}.npz').exists()]
        if not todo:
            continue
        if cur != v['key']:
            ctx = T.Ctx115(v['g'], 'C'); cur = v['key']
            T4, T16, w, pids = tables(root, v)
            reps = {'PC4_O': T.pc_uniform(T4, 4), 'CAR_O': CA.rep(T4, T16, w, pids)}
        light = PT.Lights(v['l'])
        for j, s in todo:
            t0 = time.perf_counter()
            img = ctx.readout_multi(light, dict(v['cam'], res=C.RES), C.RES, SPP, s, reps, b=64)
            rec = {'ro_e1': img['e1'].astype(np.float32), 'ro_e2': img['e2'].astype(np.float32), **{f'hl|{n}': img[n].astype(np.float32) for n in reps}}
            tmp = d / f'stream_{tag}_{j}.tmp.npz'
            np.savez_compressed(tmp, meta=json.dumps({'tag': tag, 'seed': s, 'reps': list(reps), 'seconds': time.perf_counter() - t0}), **rec)
            os.replace(tmp, d / f'stream_{tag}_{j}.npz')
            print('STREAMB', tag, j, round(time.perf_counter() - t0, 1), flush=True)


if __name__ == '__main__':
    cmd, root, *rest = sys.argv[1:]
    {'con': con, 'streams': streams}[cmd](Path(root), rest or None)
    sys.stdout.flush(); os._exit(0)
