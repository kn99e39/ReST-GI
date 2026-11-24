"""WL112 Stage C dataset: camera-independent construction-v3 targets (C semantics) on the D112 distribution.

  python data112.py ROOT SPLIT START STOP     (geometry index range; resumable, atomic per geometry)
Per geometry file: patch tables, state rays, source encoding s [L,N,NB,3], and per light:
  coarse U (f2, fh, mask), coarse G (f2, fh, mask), 2x2 G (f2, fh float16, mask), 2x2 U (f2, fh float16, mask).
Validation geometries store two independent halves (suffix 'b').
"""
import json, os, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common112 as C  # noqa: E402
import tracer112 as T  # noqa: E402

PT = T.PT
N, NB, NS = 1270, T.NB, T.NSUB


def targets(acc):
    out = {}
    for meas in ('U', 'G'):
        w = acc[f'{meas}_w'].reshape(N, NS, NB)
        f2 = np.stack([acc[f'{meas}_f2{c}'] for c in range(3)], -1).reshape(N, NS, NB, 3)
        fh = np.stack([acc[f'{meas}_fh{c}'] for c in range(3)], -1).reshape(N, NS, NB, 3)
        wc = w.sum(1)
        out[f'{meas}c_f2'] = (f2.sum(1) / np.maximum(wc, 1e-30)[..., None]).astype(np.float32)
        out[f'{meas}c_fh'] = (fh.sum(1) / np.maximum(wc, 1e-30)[..., None]).astype(np.float32)
        out[f'{meas}c_m'] = wc > 0
        out[f'{meas}s_f2'] = (f2 / np.maximum(w, 1e-30)[..., None]).astype(np.float16)
        out[f'{meas}s_fh'] = (fh / np.maximum(w, 1e-30)[..., None]).astype(np.float16)
        out[f'{meas}s_m'] = w > 0
    return out


def main(root, split, start, stop):
    root = Path(root); spec = C.P['stage_C']['splits'][split]
    out = root / 'stageC' / 'data' / split; out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    for i in range(int(start), int(stop)):
        fn = out / f'g{i:05d}.npz'
        if fn.exists():
            continue
        cfg = C.train_config(split, i)
        ctx = T.Ctx112(cfg['geom'], 'C')
        hit, bn, _ = ctx.state_rays()
        rec = {k: [] for k in ('s',)}
        per = []
        for li, ls in enumerate(cfg['lights']):
            light = PT.Lights(ls)
            rec['s'].append(ctx.source(light))
            seed = (spec['seed'] + C.SO) * 100 + i * 16 + li
            n = C.dv(spec.get('proposals_per_subcell', spec.get('proposals_per_subcell_per_half')), 8)
            t = targets(ctx.construct(light, n, seed, b=32))
            if split == 'val':
                tb = targets(ctx.construct(light, n, seed + 8, b=32))
                t.update({k + 'b': v for k, v in tb.items()})
            per.append(t)
        g = ctx.G
        data = {'config': json.dumps(cfg), 'config_hash': C.digest(cfg),
                'pc': g.PC.astype(np.float32), 'pn': g.PN.astype(np.float32), 'pt': g.PT.astype(np.float32),
                'parea': g.PAREA.astype(np.float32), 'pmat': g.PMAT.astype(np.int16), 'hit': hit, 'dbin': bn,
                's': np.stack(rec['s'])}
        for k in per[0]:
            data[k] = np.stack([p[k] for p in per])
        assert all(np.isfinite(v.astype(np.float32)).all() for k, v in data.items() if isinstance(v, np.ndarray) and v.dtype.kind == 'f'), 'STOP_NONFINITE'
        tmp = fn.with_name(fn.stem + '.tmp.npz')
        np.savez_compressed(tmp, **data); os.replace(tmp, fn)
        del ctx
        print('GEN', split, i, round(time.perf_counter() - t0, 1), flush=True)
    (out / f'done_{start}_{stop}').write_text('DONE')


if __name__ == '__main__':
    main(*sys.argv[1:])
    sys.stdout.flush(); os._exit(0)
