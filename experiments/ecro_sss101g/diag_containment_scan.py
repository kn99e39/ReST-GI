"""Read-only replay of construction visibility segments (geometry only) to locate unexplained misses."""
from pathlib import Path
import json, sys
sys.argv = sys.argv[:4]
import numpy as np
import run_generalization as RG
import gpu101g as GG
from oracle_field import ExposedOracle

reader, *_ = RG.F101.inputs(Path(sys.argv[1]).resolve())
seed = int(sys.argv[2])
ORIGIN = sys.argv[3] if len(sys.argv) > 3 else "hit"
R, rt, dr, mi = reader.R, reader.rt, reader.dr, reader.mi
oracle = GG.amended_oracle(ExposedOracle, 8e-6)(R, rt)
psid = rt.inc.G.PSID
P = lambda v: np.stack([np.asarray(v[k]) for k in range(3)], 1)
total = 0
counts = {}
for launch in range(128):
    slots, ray, hits, rng, sd, info = oracle.probes(32, seed, launch)
    ids = dr.compress(info["valid"])
    g = lambda x: dr.gather(type(x), x, ids)
    hit, r, req = g(hits), g(ray), dr.gather(R.U, slots, ids)
    w = -r.d
    spawn = hit.spawn_ray(w)
    y = oracle.scene.ray_intersect(spawn)
    opening, seam = GG.room_exit(spawn.o if ORIGIN == "spawn" else hit.p, w, dr, mi, R)
    tangent = (dr.gather(R.U, oracle.ids, req // 64) == 6) & (hit.p.z < 1e-3) & (w.z < 0)
    miss = ~y.is_valid()
    for key, mask in (("opening", miss & opening), ("seam", miss & ~opening & seam),
                      ("tangency", miss & ~opening & ~seam & tangent)):
        counts[key] = counts.get(key, 0) + int(dr.count(mask)[0])
    bad = miss & ~opening & ~seam & ~tangent
    idx = np.flatnonzero(np.asarray(bad))
    for k in idx:
        s = int(np.asarray(req)[k])
        rec = {"launch": launch, "slot": s, "pid": s // 64, "bin": s % 64, "sid": int(psid[s // 64]),
               "p": P(hit.p)[k].tolist(), "n": P(hit.n)[k].tolist(), "w": P(w)[k].tolist(),
               "spawn_o": P(spawn.o)[k].tolist()}
        o, d = np.array(rec["p"], np.float64), np.array(rec["w"], np.float64)
        ts = []
        for ax, (lo, hi) in enumerate(((-2, 2), (-2, 2), (0, 3))):
            for b in (lo, hi):
                if d[ax] != 0:
                    t = (b - o[ax]) / d[ax]
                    if t > 0:
                        ts.append((t, ax, b))
        t, ax, b = min(ts)
        rec["exit_point"] = (o + t * d).tolist()
        rec["exit_plane"] = ["xyz"[ax], b]
        print(json.dumps(rec), flush=True)
        total += 1
print("origin", ORIGIN, "unexplained", total, json.dumps(counts))
