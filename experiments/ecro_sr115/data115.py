"""WL115 Stage C dataset D115: the CAR-16 refinement targets for the D114 configurations (D114 itself, the dense / coarse full-depth
targets, is reused unchanged).

Per configuration (D113 / D114 train_config, same geometry and lights):
  support w [N,16], m [N,16], pids (patches with an active cell) from candidate115.support (geometry only),
  child descriptors cdesc [P,256,5] = geomdesc115.vhs_k(16) near / logd / esc / prox(all) + prox_other(16, ell 0.0625),
  per light: 16 x 16 deep U construction on pids (n = 32 per child cell and bin; val: a second half with seed + 8) ->
      c16_f2 (camera event 3) and c16_fhF (camera events 4..24) U means [L,P,256,NB,3] float16, c16_m (counted) [L,P,256,NB].
  python data115.py ROOT SPLIT START STOP
"""
import json, os, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common115 as C  # noqa: E402
import tracer115 as T  # noqa: E402
import candidate115 as CA  # noqa: E402
import geomdesc115 as GD  # noqa: E402

PT = T.PT
SEED0 = {'train': 11518000, 'val': 11518500}


def targets(acc):
    w = acc['w0'] + acc['w1']
    f2 = np.stack([acc[f'F20{c}'] + acc[f'F21{c}'] for c in range(3)], -1)
    hf = np.stack([acc[f'HF0{c}'] + acc[f'HF1{c}'] for c in range(3)], -1)
    den = np.maximum(w, 1e-30)[..., None]
    return {'c16_f2': (f2 / den).astype(np.float16), 'c16_fhF': ((hf - f2) / den).astype(np.float16), 'c16_m': w > 0}


def main(root, split, start, stop):
    root = Path(root)
    n = C.dv(32 if split == 'train' else 64, 4)
    out = root / 'stageC' / 'd115' / split; out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    for i in range(int(start), int(stop)):
        fn = out / f'g{i:05d}.npz'
        if fn.exists():
            continue
        cfg = C.C114.train_config(split, i)
        ctx = T.Ctx115(cfg['geom'], 'C')
        w, pids, m = CA.support(ctx)
        cdesc = np.concatenate([GD.vhs_k(ctx, 16, pids), GD.prox_other(ctx, 16, pids=pids, ells=(CA.ELL,))], -1)
        per = []
        for li, ls in enumerate(cfg['lights']):
            light = PT.Lights(ls)
            seed = (SEED0[split] + C.SO) * 100 + i * 16 + li
            acc, _ = ctx.construct_k(light, n, seed, 23, 16, pids=pids, b=C.dv(8, 1), keep_f2=True)
            t = targets(acc)
            if split == 'val':
                acc, _ = ctx.construct_k(light, n, seed + 8, 23, 16, pids=pids, b=C.dv(8, 1), keep_f2=True)
                t.update({k + 'b': v for k, v in targets(acc).items()})
            per.append(t)
        data = {'config_hash': C.C114.digest(cfg), 'w': w, 'm': m, 'pids': pids, 'cdesc': cdesc.astype(np.float32)}
        for k in per[0]:
            data[k] = np.stack([p[k] for p in per])
        tmp = fn.with_name(fn.stem + '.tmp.npz')
        np.savez_compressed(tmp, **data); os.replace(tmp, fn)
        del ctx
        print('GEN115', split, i, len(pids), int((w > 0).sum()), round(time.perf_counter() - t0, 1), flush=True)
    (out / f'done_{start}_{stop}').write_text('DONE')


if __name__ == '__main__':
    main(*sys.argv[1:])
    sys.stdout.flush(); os._exit(0)
