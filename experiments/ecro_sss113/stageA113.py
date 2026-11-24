"""WL113 Stage A: focused identity checks, 4x4 constructions (F0-F3, 4 replicas), decorrelated-RNG control (F0, F1),
synthetic projection check, VHS descriptors. Resumable; atomic outputs.

  python stageA113.py ROOT
"""
import json, os, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common113 as C  # noqa: E402
import tracer113 as T  # noqa: E402
sys.path.insert(0, str(HERE.parent / 'ecro_sss112'))

PT, mi, dr = T.PT, T.mi, T.dr
FIXI = {'F0': 0, 'F1': 1, 'F2': 2, 'F3': 3}


from concurrent.futures import ThreadPoolExecutor
_W = ThreadPoolExecutor(max_workers=2); _P = []


def save(acc, path, meta):
    arrs = {k: np.asarray(v, np.float32) for k, v in acc.items()}

    def _w():
        tmp = path.with_name(path.stem + '.tmp.npz')
        np.savez_compressed(tmp, **arrs, meta=json.dumps(meta)); os.replace(tmp, path)
    _P.append(_W.submit(_w))                     # compression overlaps the next construction


def main(root, fixes=None):
    root = Path(root); out = root / 'stageA'; out.mkdir(parents=True, exist_ok=True)
    fixes = fixes or list(FIXI)
    so = C.SO
    if 'F0' in fixes and not (out / 'identity.json').exists():
        import stageA112 as A112
        g0, l0 = C.fixture('F0')
        ident = A112.identity(g0, PT.Lights(l0))                     # A1 / A2 of WL112 (tracer and pinning identity)
        ctx = T.Ctx113(g0, 'C')
        cam = dict(C.V0_CAM, res=128)
        a, _, _, _ = ctx.cam_pass(PT.Lights(l0), cam, 128, 32, 11306002 + so, 'cam', b=32)
        f, _ = ctx.full_pass(PT.Lights(l0), cam, 128, 32, 11306002 + so, b=32)
        ident['deep_events1to6_rel_l1'] = float(sum(np.abs(a[k] - f[k]).sum() for k in ('e1', 'e2', 'h')) / sum(np.abs(a[k]).sum() for k in ('e1', 'e2', 'h')))
        C.jd(ident, out / 'identity.json'); print('IDENTITY', json.dumps(ident), flush=True)
        del ctx
    n = C.dv(1024, 32); R = C.dv(4, 2)
    for fix, fi in FIXI.items():
        if fix not in fixes:
            continue
        geom, lc = C.fixture(fix); light = PT.Lights(lc)
        ctx = T.Ctx113(geom, 'C')
        if not (out / f'vhs_{fix}.npy').exists():
            np.save(out / f'vhs_{fix}.npy', ctx.vhs())
        jobs = [(f'{fix}_r{r}', 11301000 + 100 * fi + r + so, False) for r in range(R)]
        if fix in ('F0', 'F1'):
            jobs += [(f'{fix}_dec_r{r}', 11302000 + 100 * fi + r + so, True) for r in range(R)]
        for name, seed, dec in jobs:
            p = out / f'{name}.npz'
            if p.exists():
                continue
            t0 = time.perf_counter()
            acc = ctx.construct4(light, n, seed, b=16, decorrelate=dec)
            dt = time.perf_counter() - t0
            save(acc, p, {'fixture': fix, 'seed': seed, 'proposals_per_subcell': n, 'decorrelated': dec, 'seconds': dt})
            print('CONSTRUCT', name, round(dt, 1), {k: float(acc[k].sum()) for k in ('U_w', 'buried', 'singular', 'mismatch', 'bad_origin', 'bad_floor')}, flush=True)
        if fix == 'F0' and not (out / 'synthetic.npz').exists():
            def L(p, w, nn):
                v = 1. + 0.5 * p.x + 0.3 * p.y + 0.4 * dr.dot(w, nn) + 0.2 * w.x
                return mi.Color3f(v, v, v)
            acc = ctx.construct4(light, C.dv(256, 32), 11305001 + so, b=8, synthetic=L)
            save(acc, out / 'synthetic.npz', {'seed': 11305001 + so})
            print('SYNTHETIC done', flush=True)
        del ctx
    for f in _P:
        f.result()
    (out / f'stageA_{"_".join(fixes)}.done').write_text('DONE')


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2:] or None)
    sys.stdout.flush(); os._exit(0)
