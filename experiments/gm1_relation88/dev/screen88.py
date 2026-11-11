"""WL88 dev: geometry-only screening of candidate workloads (mask pixel counts; no radiance).  python screen88.py <root> <W>..."""
import json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
import fx88  # noqa: E402
import masks88  # noqa: E402
S = fx88.S
root = sys.argv[1]
for w in sys.argv[2:]:
    t0 = time.perf_counter()
    ex = fx88.FX[w]
    gok = bool(S.geom_ok(ex['T0']) and S.geom_ok(ex['T1']))
    m, aux = masks88.compute(w, res=256, nph=4_000_000)
    cnt = {k: int(v.sum()) * 4 for k, v in m.items()}
    print(w, 'geom_ok', gok, round(time.perf_counter() - t0, 1), 's', json.dumps(cnt), flush=True)
    import numpy as np
    os.makedirs(os.path.join(root, 'screen'), exist_ok=True)
    np.savez_compressed(os.path.join(root, 'screen', f'{w}.npz'), **m, **aux)
os._exit(0)
