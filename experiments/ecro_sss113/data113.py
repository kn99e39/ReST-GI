"""WL113 Stage C dataset (D113): construction v3 at 4x4 (C semantics, depth 6), coarse targets by exact pooling of the same
samples (both arms), refined 4x4 targets on the supported patches of the selected representation, VHS descriptors.

  python data113.py ROOT SPLIT START STOP
"""
import json, os, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common113 as C  # noqa: E402
import tracer113 as T  # noqa: E402

PT = T.PT
N, NS, NB = 1270, 16, 64


def targets(acc, sup):
    w = acc['U_w'].reshape(N, NS, NB)
    f2 = np.stack([acc[f'U_f2{c}'] for c in range(3)], -1).reshape(N, NS, NB, 3)
    fh = np.stack([acc[f'U_fh{c}'] for c in range(3)], -1).reshape(N, NS, NB, 3)
    wc = w.sum(1)
    out = {'c_f2': (f2.sum(1) / np.maximum(wc, 1e-30)[..., None]).astype(np.float32),
           'c_fh': (fh.sum(1) / np.maximum(wc, 1e-30)[..., None]).astype(np.float32), 'c_m': wc > 0}
    ws = w[sup]
    out['s_f2'] = (f2[sup] / np.maximum(ws, 1e-30)[..., None]).astype(np.float16)
    out['s_fh'] = (fh[sup] / np.maximum(ws, 1e-30)[..., None]).astype(np.float16)
    out['s_m'] = ws > 0
    return out


def main(root, split, start, stop):
    root = Path(root)
    dec = json.loads((root / 'stageB' / 'decision.json').read_text())
    n = C.dv(32 if split == 'train' else 64, 8)
    seed0 = {'train': 11320100, 'val': 11320200}[split] + C.SO
    out = root / 'stageC' / 'data' / split; out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    for i in range(int(start), int(stop)):
        fn = out / f'g{i:05d}.npz'
        if fn.exists():
            continue
        cfg = C.train_config(split, i)
        ctx = T.Ctx113(cfg['geom'], 'C')
        hit, bn, _ = ctx.state_rays()
        desc = ctx.vhs()
        sup = T.support_from_desc(desc, ctx.G) if dec['selected'] == 'L4' else np.ones(N, bool)
        rec = {'s': []}; per = []
        for li, ls in enumerate(cfg['lights']):
            light = PT.Lights(ls)
            rec['s'].append(ctx.source(light))
            seed = seed0 * 100 + i * 16 + li
            t = targets(ctx.construct4(light, n, seed, b=16), sup)
            if split == 'val':
                tb = targets(ctx.construct4(light, n, seed + 8, b=16), sup)
                t.update({k + 'b': v for k, v in tb.items()})
            per.append(t)
        g = ctx.G
        data = {'config': json.dumps(cfg), 'config_hash': C.digest(cfg), 'pc': g.PC.astype(np.float32), 'pn': g.PN.astype(np.float32),
                'pt': g.PT.astype(np.float32), 'parea': g.PAREA.astype(np.float32), 'pmat': g.PMAT.astype(np.int16), 'hit': hit, 'dbin': bn,
                's': np.stack(rec['s']), 'desc': desc, 'support': sup}
        for k in per[0]:
            data[k] = np.stack([p[k] for p in per])
        tmp = fn.with_name(fn.stem + '.tmp.npz')
        np.savez_compressed(tmp, **data); os.replace(tmp, fn)
        del ctx
        print('GEN', split, i, int(sup.sum()), round(time.perf_counter() - t0, 1), flush=True)
    (out / f'done_{start}_{stop}').write_text('DONE')


if __name__ == '__main__':
    main(*sys.argv[1:])
    sys.stdout.flush(); os._exit(0)
