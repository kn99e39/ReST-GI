"""WL115 full-depth physical references of the FRESH configurations (N5-N8 views W0 / W2; SEQ_E_fresh states x views W0 / W2):
two 4,096-spp deterministic deep passes to K* = 24 (tracer114.deep_pass, the WL114 reference estimator), g-buffer and silhouette
coverage. Candidate-independent; rendered before the Stage B freeze but NOT read by any analysis until the freeze is committed
(sha256 of every file recorded in ref_manifest.json at render time).

  python ref115.py ROOT [TAG_PREFIX ...]
"""
import json, os, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common115 as C  # noqa: E402
import tracer115 as T  # noqa: E402

PT, dr = T.PT, T.dr
SPP = C.dv(4096, 64)


def fresh_views():
    """tag -> (state key, geometry, light, camera, [seed j0, seed j1])"""
    out = {}
    for fi, f in enumerate(C.FRESH_SUPPORT):
        g, l = C.fresh_fixture(f)
        for vi, (vn, cam) in enumerate(C.fresh_views(f).items()):
            out[f'{f}_{vn}'] = (f, g, l, cam, [11530000 + 100 * fi + 10 * vi + j + C.SO for j in range(2)])
    q = C.seq_E(); cams = C.seq_views(q)
    for t, g in enumerate(q['states']):
        for vi, vn in enumerate(q['views']):
            out[f'SEQ_E_fresh_{t}_{vn}'] = (f'SEQ_E_fresh_{t}', g, q['light'], cams[vn], [11535000 + 10 * t + 2 * vi + j + C.SO for j in range(2)])
    return out


def main(root, only=None):
    d = root / 'refs'; d.mkdir(parents=True, exist_ok=True)
    V = fresh_views()
    cur = None; ctx = None
    for tag, (key, g, l, cam, seeds) in V.items():
        if only and not any(tag.startswith(o) for o in only):
            continue
        cam = dict(cam, res=C.RES)
        if cur != key:
            ctx = T.Ctx115(g, 'C'); cur = key
        light = PT.Lights(l)
        gp = d / f'gbuffer_{tag}.npz'
        if not gp.exists():
            gb = ctx.gbuffer(cam, C.RES); gb['coverage'] = T.T113.silhouette_coverage(ctx, cam)
            np.savez_compressed(gp, **gb)
        for j, s in enumerate(seeds):
            p = d / f'ref_{tag}_{j}.npz'
            if p.exists():
                continue
            t0 = time.perf_counter()
            img, dg = ctx.deep_pass(light, cam, C.RES, SPP, s, 24, b=64)
            tmp = p.with_name(p.stem + '.tmp.npz')
            np.savez_compressed(tmp, meta=json.dumps({'seed': s, 'spp': SPP, 'K_star': 24, 'seconds': time.perf_counter() - t0, 'diag': dg}),
                                **{f'phys_{k}': v.astype(np.float32) for k, v in img.items()})
            os.replace(tmp, p)
            print('REF', tag, j, round(time.perf_counter() - t0, 1), flush=True)
    man = {p.name: C.sha(p) for p in sorted(d.glob('*.npz')) if '.tmp' not in p.name}
    C.jd(man, d / 'ref_manifest.json')


if __name__ == '__main__':
    main(Path(sys.argv[1]), sys.argv[2:] or None)
    sys.stdout.flush(); os._exit(0)
