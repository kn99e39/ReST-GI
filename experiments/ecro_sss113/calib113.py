"""WL113 pre-freeze support calibration on training-distribution geometries (camera-independent, table level; no camera,
ROI or reference image). For each geometry: two independent 4x4 U constructions (512 proposals per sub-cell each); debiased
within-patch RMS deviation of sub-cell mean radiance D(p) from the cross product of the halves; VHS support at delta;
recall of patches with D(p) >= 3 %."""
import json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / 'ecro_sss112'))
import tracer113 as T  # noqa: E402
import common112 as C  # noqa: E402

PT = T.PT
N, NS, NB = 1270, 16, 64


def sub_means(acc):
    w = acc['U_w'].reshape(N, NS, NB)
    L = sum(T.LUM[c] * (acc[f'U_f2{c}'] + acc[f'U_fh{c}']) for c in range(3)).reshape(N, NS, NB)
    # bin-equal mean per sub-cell over bins with samples; weights = accepted count
    m = (L / np.maximum(w, 1)).mean(2)
    return m, w.sum(2)


def main(out_path, seed_base, geoms):
    res = []
    for gi, cfg in enumerate(geoms):
        ctx = T.Ctx113(cfg['geom'], 'C')
        light = PT.Lights(cfg['lights'][0])
        a = ctx.construct4(light, 512, seed_base + 10 * gi, b=8); b = ctx.construct4(light, 512, seed_base + 10 * gi + 1, b=8)
        ma, wa = sub_means(a); mb, wb = sub_means(b)
        ws = wa + wb
        mpa = (ma * ws).sum(1) / np.maximum(ws.sum(1), 1); mpb = (mb * ws).sum(1) / np.maximum(ws.sum(1), 1)
        cross = ((ma - mpa[:, None]) * (mb - mpb[:, None]) * ws).sum(1) / np.maximum(ws.sum(1), 1)
        D = np.sqrt(np.maximum(cross, 0)) / np.maximum(0.5 * (mpa + mpb), 1e-9)
        desc = ctx.vhs(); rng = desc[..., 0].max(1) - desc[..., 0].min(1)
        g = ctx.G
        row = {'index': cfg['index'], 'sphere': cfg['geom']['S']['c']}
        mat = D >= 0.03
        for d in (0.05, 0.1, 0.15, 0.2):
            sup = rng >= d
            row[f'delta{d}'] = {'support_frac': float(sup.mean()), 'recall': float((sup & mat).sum() / max(mat.sum(), 1)),
                                'missed_patches': [[int(p), int(g.PSID[p]), round(float(D[p]), 3)] for p in np.flatnonzero(mat & ~sup)][:20]}
        row['material_frac'] = float(mat.mean()); row['material_by_shape'] = {int(s): int(mat[g.PSID == s].sum()) for s in range(9)}
        row['D_top'] = [[int(p), int(g.PSID[p]), round(float(D[p]), 3), round(float(rng[p]), 3)] for p in np.argsort(-D)[:12]]
        res.append(row); print(json.dumps(row)[:1500], flush=True)
        del ctx
    C.jd(res, out_path)


if __name__ == '__main__':
    import os
    sys.path.insert(0, str(HERE))
    import common113 as C3
    geoms = [C3.train_config('train', i) for i in range(4)]
    main(C.REPO / 'results/ecro_sss113/dev/calib113.json', 11398000, geoms)
    sys.stdout.flush(); os._exit(0)
