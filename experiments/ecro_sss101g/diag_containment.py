"""Pre-freeze development diagnostic: classify unexplained visibility misses and back-facing receivers."""
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
HERE = Path(__file__).resolve().parent
sys.argv = sys.argv[:2]
import run_generalization as RG
import gpu101g as GG
from oracle_field import ExposedOracle

root = Path(sys.argv[1]).resolve()
reader, learned, *_ = RG.F101.inputs(root)
R, rt, dr, mi = reader.R, reader.rt, reader.dr, reader.mi
oracle = ExposedOracle(R, rt)
slots, ray, hits, rng, sd, info = oracle.probes(32, 10203101, 0)
ids = dr.compress(info["valid"])
g = lambda x: dr.gather(type(x), x, ids)
hit, r, req = g(hits), g(ray), dr.gather(R.U, slots, ids)
w = -r.d
y = oracle.scene.ray_intersect(hit.spawn_ray(w))
front = y.is_valid() & (y.to_local(-w).z > 1e-6)
opening, seam = GG.room_exit(hit.p, w, dr, mi, R)
bad = ~y.is_valid() & ~opening & ~seam
back = y.is_valid() & ~front
P = lambda v: np.stack([np.asarray(v[k]) for k in range(3)], 1)
p, wn, sl, b, bk = P(hit.p), P(w), np.asarray(req), np.asarray(bad), np.asarray(back)
psid = rt.inc.G.PSID
print("unexplained", b.sum())
for k in np.flatnonzero(b)[:31]:
    print(int(sl[k]), int(psid[sl[k] // 64]), np.round(p[k], 5), np.round(wn[k], 5))
ysid = np.asarray(rt.shape_id(y))
print("back receivers by (source sid -> receiver sid):")
pairs, counts = np.unique(np.stack([psid[sl[bk] // 64], ysid[bk]], 1), axis=0, return_counts=True)
for pr, c in zip(pairs, counts):
    print(pr, c)
