"""WL101G views and region masks (numpy only)."""
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "ecro_sss101a"))
from common import P as A, masks  # noqa: E402

PROTOCOL = HERE / "protocol/generalization.json"
P = json.loads(PROTOCOL.read_text())
EDGES = [0, 0.10, 0.14, 0.18, 0.22, 0.26, 0.30, 0.35, 0.40, 0.50]
VIEWS = ["V0"] + [v["id"] for v in P["views"]["held_out"]]


def view_cameras(base):
    cams = {"V0": dict(base)}
    target = np.array([0.9, 0.4, 0.35])
    for v in P["views"]["held_out"]:
        az, el, d = np.radians(v["azimuth_deg"]), np.radians(v["elevation_deg"]), v["horizontal_m"]
        origin = target + np.array([d * np.sin(az), -d * np.cos(az), d * np.tan(el)])
        cams[v["id"]] = {"origin": [float(x) for x in origin], "target": target.tolist(),
                         "fov_deg": base["fov_deg"], "res": base["res"]}
    return cams


def view_regions(gpath, v0):
    """WL101A historical masks plus near/flank and radial bands on one view's center-ray gbuffer."""
    reg = masks(gpath)
    with np.load(gpath) as m:
        pos = m["p"]
    center = np.asarray(A["sphere"]["center"])
    r = np.linalg.norm(pos[..., :2] - center[:2], axis=-1)
    floor = reg["floor"]
    reg["near_all"] = floor & (r < .18)
    reg["flank_all"] = floor & (r >= .26) & (r < .40)
    base = floor & (pos[..., 1] <= center[1]) if v0 else floor
    reg["near"] = base & (r < .18)
    reg["flank"] = base & (r >= .26) & (r < .40)
    if v0:
        assert reg["near"].sum() == 1318 and reg["flank"].sum() == 6042
        assert np.all(reg["contact_shadow"][reg["near"]])
    bands = [base & (r >= a) & (r < b) for a, b in zip(EDGES, EDGES[1:])]
    return reg, bands


def contact_valid(reg):
    return bool(reg["near"].sum() >= 200 and reg["flank"].sum() >= 1000)
