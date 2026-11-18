"""Read-only geometry probe: where do the room walls' front edges (next to the open y=-2 face) actually end?"""
from pathlib import Path
import json, sys
sys.argv = sys.argv[:2]
import numpy as np
import run_generalization as RG

reader, *_ = RG.F101.inputs(Path(sys.argv[1]).resolve())
R, rt, dr, mi = reader.R, reader.rt, reader.dr, reader.mi
offsets = np.array([0, 1e-5, 1e-4, 3e-4, 5e-4, 8e-4, 1e-3, 1.2e-3, 1.5e-3, 2e-3, 3e-3, 1e-2], np.float64)
out = {}
for name, d, origin_of in (("right wall x=2", (1, 0, 0), lambda y, z: (1.5, y, z)),
                           ("left wall x=-2", (-1, 0, 0), lambda y, z: (-1.0, y, z)),
                           ("floor z=0", (0, 0, -1), lambda y, z: (0.9, y, 0.5)),
                           ("ceiling z=3", (0, 0, 1), lambda y, z: (0.9, y, 2.5))):
    rows = []
    for off in offsets:
        y = -2.0 + off
        zs = np.linspace(0.3, 2.7, 64)
        o = np.array([origin_of(y, z) for z in zs], np.float32)
        ray = mi.Ray3f(mi.Point3f(*[R.F(o[:, k]) for k in range(3)]), mi.Vector3f(*[R.F(np.full(64, float(c), np.float32)) for c in d]))
        si = rt.scene.ray_intersect(ray)
        hit = np.asarray(si.is_valid())
        sid = np.asarray(rt.shape_id(si))
        rows.append({"distance_from_open_face_m": float(off), "hit_fraction": float(hit.mean()),
                     "hit_shapes": sorted(set(sid[hit].tolist()))})
    out[name] = rows
params = rt.params
for k in ("r4.to_world", "r3.to_world", "r0.to_world"):
    try:
        out[k] = str(params[k])
    except Exception as e:
        out[k] = repr(e)
print(json.dumps(out, indent=1))
