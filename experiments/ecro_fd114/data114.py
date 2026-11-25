"""WL114 Stage C dataset D114: the D113 configurations and construction seeds / launch partition (b = 16), deep construction to
probe event K*-1; D6 targets (c_f2, c_fh, s_f2, s_fh) and full-depth targets (c_fhF = fh + deep buckets <= K*, s_fhF) from the SAME
samples, masks, VHS descriptors. Authentication of the D6 part against the D113 files.

  python data114.py ROOT SPLIT START STOP
  python data114.py ROOT auth
"""
import json, os, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common114 as C  # noqa: E402
import tracer114 as T  # noqa: E402

PT = T.PT
N, NS, NB = 1270, 16, 64
D113 = C.REPO / 'results' / 'ecro_sss113' / 'primary' / 'stageC' / 'data'


def targets(acc, K):
    w = acc['U_w'].reshape(N, NS, NB)
    g = lambda nm: np.stack([acc.get(f'U_{nm}{c}', np.zeros(N * NS * NB)) for c in range(3)], -1).reshape(N, NS, NB, 3)
    f2, fh = g('f2'), g('fh')
    fhF = fh + sum((g(nm) for nm in T.deep_through(K)), np.zeros_like(fh))
    wc = w.sum(1)
    out = {'c_m': wc > 0, 's_m': w > 0}
    for nm, v in (('f2', f2), ('fh', fh), ('fhF', fhF)):
        out[f'c_{nm}'] = (v.sum(1) / np.maximum(wc, 1e-30)[..., None]).astype(np.float32)
        out[f's_{nm}'] = (v / np.maximum(w, 1e-30)[..., None]).astype(np.float16)
    out['sq6'] = np.stack([acc[f'U_sq6{c}'] for c in range(3)], -1).reshape(N, NS, NB, 3).sum(1).astype(np.float32)
    out['sqA'] = np.stack([acc[f'U_sqA{c}'] for c in range(3)], -1).reshape(N, NS, NB, 3).sum(1).astype(np.float32)
    return out


def main(root, split, start, stop):
    root = Path(root)
    K = json.loads((root / 'analysis' / 'stageA_qualification.json').read_text())['SSS']['K_star']
    n = C.dv(32 if split == 'train' else 64, 8)
    seed0 = {'train': 11320100, 'val': 11320200}[split] + C.SO
    out = root / 'stageC' / 'data' / split; out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    for i in range(int(start), int(stop)):
        fn = out / f'g{i:05d}.npz'
        if fn.exists():
            continue
        cfg = C.train_config(split, i)
        ctx = T.Ctx114(cfg['geom'], 'C')
        hit, bn, _ = ctx.state_rays()
        desc = ctx.vhs()
        sup = np.ones(N, bool)
        rec = {'s': []}; per = []
        for li, ls in enumerate(cfg['lights']):
            light = PT.Lights(ls)
            rec['s'].append(ctx.source(light))
            seed = seed0 * 100 + i * 16 + li
            acc, _ = ctx.construct_deep(light, n, seed, K - 1, b=16)
            t = targets(acc, K)
            if split == 'val':
                acc, _ = ctx.construct_deep(light, n, seed + 8, K - 1, b=16)
                tb = targets(acc, K)
                t.update({k + 'b': v for k, v in tb.items() if not k.startswith('sq')})
            per.append(t)
        g = ctx.G
        data = {'config': json.dumps(cfg), 'config_hash': C.digest(cfg), 'pc': g.PC.astype(np.float32), 'pn': g.PN.astype(np.float32),
                'pt': g.PT.astype(np.float32), 'parea': g.PAREA.astype(np.float32), 'pmat': g.PMAT.astype(np.int16), 'hit': hit, 'dbin': bn,
                's': np.stack(rec['s']), 'desc': desc, 'support': sup, 'K_star': K}
        for k in per[0]:
            data[k] = np.stack([p[k] for p in per])
        tmp = fn.with_name(fn.stem + '.tmp.npz')
        np.savez_compressed(tmp, **data); os.replace(tmp, fn)
        del ctx
        print('GEN', split, i, round(time.perf_counter() - t0, 1), flush=True)
    (out / f'done_{start}_{stop}').write_text('DONE')


def auth(root):
    """D6 part of D114 vs D113 (rel L1 per file on c_f2 + c_fh and s_f2 + s_fh); identical configuration hashes."""
    root = Path(root); res = {'files': 0, 'max_rel_c': 0., 'max_rel_s': 0., 'config_mismatch': 0, 'missing_d113': 0}
    for split in ('train', 'val'):
        for f in sorted((root / 'stageC' / 'data' / split).glob('g*.npz')):
            if '.tmp' in f.name:
                continue
            g = D113 / split / f.name
            if not g.exists():
                res['missing_d113'] += 1; continue
            a, b = np.load(f), np.load(g)
            res['config_mismatch'] += int(str(a['config_hash']) != str(b['config_hash']))
            ca = a['c_f2'] + a['c_fh']; cb = b['c_f2'] + b['c_fh']
            sa = a['s_f2'].astype(np.float32) + a['s_fh'].astype(np.float32); sb = b['s_f2'].astype(np.float32) + b['s_fh'].astype(np.float32)
            res['max_rel_c'] = max(res['max_rel_c'], float(np.abs(ca - cb).sum() / max(np.abs(cb).sum(), 1e-30)))
            res['max_rel_s'] = max(res['max_rel_s'], float(np.abs(sa - sb).sum() / max(np.abs(sb).sum(), 1e-30)))
            res['files'] += 1
    res['pass'] = bool(res['files'] == C.dv(208, 0) or C.DEV) and res['missing_d113'] == 0 and res['config_mismatch'] == 0 and res['max_rel_c'] <= 1e-5 and res['max_rel_s'] <= 1e-5
    C.jd(res, root / 'stageC' / 'd114_authentication.json')
    print('AUTH', json.dumps(res), flush=True)


if __name__ == '__main__':
    if sys.argv[2] == 'auth':
        auth(sys.argv[1])
    else:
        main(*sys.argv[1:])
    sys.stdout.flush(); os._exit(0)
