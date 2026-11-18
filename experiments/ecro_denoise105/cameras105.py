"""WL105 camera sets (geometry only; numpy).

Evaluation (held out, never used in training or model selection): V0 (WL101A-C camera) and the five WL101G
held-out views H1-H5 (defined from geometry in WL101G; no table, model or reconstructor has read them).
Training / validation: random views of the same room, rejected if near-duplicate of an evaluation view.
"""
import math

import numpy as np

TARGET_H = np.array([0.9, 0.4, 0.35])
HELD_OUT = [("H1", -45, 2.2, 18), ("H2", 35, 1.75, 20), ("H3", 165, 1.45, 22), ("H4", -150, 1.6, 22), ("H5", 0, 1.6, 50)]
V0 = {"origin": [0.9, -2.4, 1.05], "target": [0.9, 0.4, 0.55], "fov_deg": 32}


def eval_cameras():
    cams = {"V0": dict(V0)}
    for vid, az, d, el in HELD_OUT:
        a, e = math.radians(az), math.radians(el)
        o = TARGET_H + np.array([d * math.sin(a), -d * math.cos(a), d * math.tan(e)])
        cams[vid] = {"origin": [float(x) for x in o], "target": TARGET_H.tolist(), "fov_deg": 32}
    return cams


def _dir(c):
    d = np.asarray(c["target"], float) - np.asarray(c["origin"], float)
    return d / np.linalg.norm(d)


def _inside_solid(o, geom):
    if np.linalg.norm(o - np.array([0.9, 0.4, 0.4])) < 0.45:
        return True
    for b in ("A", "B"):
        c, h, yaw = np.asarray(geom[b]["c"]), np.asarray(geom[b]["h"]), geom[b]["yaw"]
        ax = np.array([[math.cos(yaw), math.sin(yaw), 0], [-math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]])
        if np.all(np.abs(ax @ (o - c)) < h + 0.05):
            return True
    return False


def random_cameras(n, seed, geom, forbidden):
    """Random views: 60% aimed near the sphere, 40% anywhere in the room. Origin inside the room (margin 0.1 m)
    or just outside the open y = -2 face. Rejected if within 0.75 m AND 15 degrees of any forbidden view."""
    rs = np.random.default_rng(seed)
    out = []
    while len(out) < n:
        if rs.random() < 0.6:
            t = TARGET_H + rs.uniform([-0.6, -0.6, -0.3], [0.6, 0.6, 0.4])
        else:
            t = rs.uniform([-1.6, -1.2, 0.1], [1.7, 1.7, 1.6])
        az, el, dist = rs.uniform(0, 2 * math.pi), math.radians(rs.uniform(8, 55)), rs.uniform(1.2, 3.4)
        o = t + dist * np.array([math.cos(el) * math.sin(az), -math.cos(el) * math.cos(az), math.sin(el)])
        inside = (-1.9 < o[0] < 1.9) and (-1.9 < o[1] < 1.9) and (0.1 < o[2] < 2.9)
        outside_front = (-1.9 < o[0] < 1.9) and (-3.2 < o[1] < -2.05) and (0.1 < o[2] < 2.9)
        if not (inside or outside_front) or _inside_solid(o, geom):
            continue
        cam = {"origin": o.tolist(), "target": t.tolist(), "fov_deg": float(rs.uniform(28, 50))}
        dup = any(np.linalg.norm(o - np.asarray(f["origin"])) < 0.75 and
                  math.degrees(math.acos(np.clip(_dir(cam) @ _dir(f), -1, 1))) < 15 for f in forbidden)
        if dup:
            continue
        out.append(cam)
    return out
