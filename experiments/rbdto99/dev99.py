"""WL99 DEVELOPMENT looks (not evidence): render candidate configurations at low spp, report tag shares on body pixels, save PNGs.

  python dev99.py <out_dir> <cand_json> [spp] [res]
"""
import json, os, sys, time
import numpy as np
import pt99

out_dir, cand = sys.argv[1], json.load(open(sys.argv[2]))
spp = int(sys.argv[3]) if len(sys.argv) > 3 else 64
res = int(sys.argv[4]) if len(sys.argv) > 4 else 384
os.makedirs(out_dir, exist_ok=True)


def tone(x, ev):
    y = np.clip(x * 2.0 ** ev, 0, None)
    return (np.clip(y / (1 + y), 0, 1) ** (1 / 2.2) * 255).astype(np.uint8)


def save(path, img):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.imsave(path, img)


rows = []
ctx = None
for c in cand:
    t0 = time.perf_counter()
    if ctx is None:
        ctx = pt99.Ctx(c)
    else:
        ctx.set_cfg(c)
    acc, Tm = ctx.render(c['camera'], res, spp, seed=99001)
    r = pt99.pack(acc, spp)
    mb = r['mb'] > 0.5
    shares = {t: float(r[t][mb].sum() / max(r['tot'][mb].sum(), 1e-30)) for t in pt99.TAGS if t in r}
    row = {'name': c['name'], 'body_px': int(mb.sum()), 'shares_body_px': shares,
           'self_mean_body': float(r['self'][mb].mean()) if mb.any() else 0.0, 'gpu_ms': Tm['gpu_ms'], 'wall_s': time.perf_counter() - t0}
    rows.append(row)
    print(json.dumps(row), flush=True)
    for key in ('tot', 'self', 'S3', 'B1'):
        if key in r:
            save(os.path.join(out_dir, f"{c['name']}_{key}.png"), tone(r[key].reshape(res, res, 3), c.get('ev', 0)))
    np.savez_compressed(os.path.join(out_dir, f"{c['name']}.npz"), **{k: v.astype(np.float32) for k, v in r.items()})
json.dump(rows, open(os.path.join(out_dir, 'dev_rows.json'), 'w'), indent=1)
