"""Read-only retrace of the two unexplained visibility misses and controlled variants."""
from pathlib import Path
import json, sys
sys.argv = sys.argv[:2]
import numpy as np
import run_generalization as RG

reader, *_ = RG.F101.inputs(Path(sys.argv[1]).resolve())
R, rt, dr, mi = reader.R, reader.rt, reader.dr, reader.mi
cases = [([1.3755288124084473, 1.0336922407150269, 0.32518693804740906], [0.20150798559188843, -0.9789317846298218, 0.03297034278512001]),
         ([1.3833842277526855, 1.0075345039367676, 0.28638482093811035], [0.15546424686908722, -0.7582852840423584, 0.6331143975257874])]
def trace(o, d):
    ray = mi.Ray3f(mi.Point3f(*[R.F([float(x)]) for x in o]), mi.Vector3f(*[R.F([float(x)]) for x in d]))
    si = rt.scene.ray_intersect(ray)
    return {"hit": bool(np.asarray(si.is_valid())[0]), "shape": int(np.asarray(rt.shape_id(si))[0]),
            "t": float(np.asarray(si.t)[0]), "p": [float(np.asarray(si.p[k])[0]) for k in range(3)]}
out = []
for o, d in cases:
    o, d = np.array(o), np.array(d) / np.linalg.norm(d)
    row = {"exact": trace(o, d)}
    row["origin_advanced_1cm"] = trace(o + 0.01 * d, d)
    row["origin_advanced_1m"] = trace(o + 1.0 * d, d)
    t = (2.0 - o[0]) / d[0]
    wall = o + t * d
    row["analytic_wall_point"] = wall.tolist()
    for dy in (-2e-3, -1e-3, 0.0, 1e-3, 2e-3, 1e-2):
        target = wall + np.array([0, dy, 0])
        dd = (target - o) / np.linalg.norm(target - o)
        row[f"aim_wall_dy_{dy}"] = trace(o, dd)
    row["reverse_from_wall"] = trace(wall - 1e-3 * d, -d)
    out.append(row)
print(json.dumps(out, indent=1))
