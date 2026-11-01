"""WL69 8DNA adapter (protocol xbench69_v1): the C1 slab as a native 8DNA volumetric asset, and the C1 evaluation scenes.

Unit convention (protocol X1, fixed before results): the T0 slab is centred at the origin and scaled so that its bounding-box
diagonal is 2.0 units (8DNA samples incident rays on the unit sphere, PathSamplingDataset.reload, so the asset must lie inside it;
official volumetric assets: diagonal 2.10-2.40, centred). sigma_t = 1000 m^-1 / s per unit; every vertex and camera follows the
same similarity, so radiance is unchanged. The boundary is index-matched (Mitsuba null BSDF), as in the WL53-55 physics.

Training scene (get_scene, imported by the unchanged 8DNA train.py as scenes.xbslab): structure of the official volumetric scenes
(perspective camera, envmap 'directional.exr' background used only by 8DNA's validation render, asset in shapegroup + instance,
prbvolpath). Evaluation scene (eval_scene): the asset instance at the T0 / T1 pose, the WL55 Lambert floor (rho 0.8), the WL55 sun
(directional, E = 1) and the WL55 orthographic cameras of that state, all in the same units.
"""
import math
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "fnto_k55")); sys.path.insert(0, str(HERE.parent / "phys_ref53")); sys.path.insert(0, str(HERE.parent / "fnto_c54"))
import k55_fixtures as KF  # noqa: E402
import p53_fixtures as FX  # noqa: E402

DIAG = 2.0
STATES = {"T0": "floor", "T1": "floor_c4"}


def _dom(sd):
    return [(m, V) for n, k, m, V in sd["comps"] if k == "medium"][0]


def similarity():
    """(s units per metre, centre c in metres) from the T0 slab."""
    m, V = _dom(KF.scene_def("floor")); c = 0.5 * (V.min(0) + V.max(0))
    return DIAG / float(np.linalg.norm(V.max(0) - V.min(0))), c


def write_ply(path, V, F):
    path = Path(path)
    with open(path, "wb") as f:
        f.write((f"ply\nformat binary_little_endian 1.0\nelement vertex {len(V)}\nproperty float x\nproperty float y\nproperty float z\n"
                 f"element face {len(F)}\nproperty list uchar int vertex_indices\nend_header\n").encode())
        f.write(np.asarray(V, "<f4").tobytes())
        rec = np.zeros(len(F), dtype=[("n", "u1"), ("i", "<i4", 3)]); rec["n"] = 3; rec["i"] = F; f.write(rec.tobytes())
    return str(path)


def asset_mesh():
    """Canonical (T0, scaled, centred) slab vertices / faces."""
    s, c = similarity(); m, V = _dom(KF.scene_def("floor")); return s * (V - c), m["F"]


def pose_translation(state):
    """Rigid placement of the canonical asset in state's scene (units): T1 = T0 + s * (centre_T1 - centre_T0)."""
    s, c = similarity(); m, V = _dom(KF.scene_def(STATES[state])); c1 = 0.5 * (V.min(0) + V.max(0))
    Vt0 = s * (_dom(KF.scene_def("floor"))[1] - c); Vt = s * (V - c)
    t = s * (c1 - c); assert np.abs(Vt - (Vt0 + t)).max() < 1e-5, "state is not a pure translation of T0"
    return t


def medium(s):
    return {"type": "homogeneous", "albedo": {"type": "rgb", "value": list(map(float, FX.ALBEDO))}, "sigma_t": 1.0, "scale": FX.SIGMA_T / s,
            "phase": {"type": "isotropic"}}


def get_scene(res, theta=70, phi=0, r=3.5, tgt=(0, 0, 0), ply_dir=None):
    import mitsuba as mi
    s, _ = similarity(); V, F = asset_mesh(); d = Path(ply_dir or os.environ.get("XB69_PLY_DIR", HERE)); d.mkdir(parents=True, exist_ok=True)
    ply = write_ply(d / "xb69_slab.ply", V, F)
    th, ph = math.radians(theta), math.radians(phi); src = np.array([math.sin(th) * math.cos(ph), math.cos(th), math.sin(th) * math.sin(ph)]) * r
    tex = Path(os.environ.get("XB69_TEXTURES", Path.cwd() / "scenes" / "textures")) / "directional.exr"
    return {
        "type": "scene",
        "camera": {"type": "perspective", "fov": 35, "to_world": mi.ScalarTransform4f.look_at(origin=list(src), target=list(tgt), up=[0, 1, 0]),
                   "film": {"type": "hdrfilm", "width": res, "height": res, "filter": {"type": "box"}}},
        "background": {"type": "envmap", "filename": str(tex), "scale": 15, "to_world": mi.ScalarTransform4f.rotate([1, 0, 0], -90)},
        "group0": {"type": "shapegroup", "slab": {"type": "ply", "filename": ply, "face_normals": True, "bsdf": {"type": "null"}, "interior": medium(s)}},
        "instance0": {"type": "instance", "shapegroup": {"type": "ref", "id": "group0"}},
        "integrator": {"type": "prbvolpath", "max_depth": -1, "rr_depth": 5, "hide_emitters": False},
    }


def eval_scene(state, geo, ply_dir):
    """Mitsuba scene dict for 8DNA inference in state 'T0' / 'T1'; geo = WL55 ccam json of that state (metres)."""
    import mitsuba as mi
    s, c = similarity(); V, F = asset_mesh(); d = Path(ply_dir); d.mkdir(parents=True, exist_ok=True)
    sd = KF.scene_def(STATES[state]); fl = [(m, Vf) for n, k, m, Vf in sd["comps"] if k == "lambert"][0]
    t = pose_translation(state)
    out = {"type": "scene",
           "group0": {"type": "shapegroup", "slab": {"type": "ply", "filename": write_ply(d / f"slab_{state}.ply", V, F), "face_normals": True,
                                                     "bsdf": {"type": "null"}, "interior": medium(s)}},
           "instance0": {"type": "instance", "shapegroup": {"type": "ref", "id": "group0"}, "to_world": mi.ScalarTransform4f.translate(list(map(float, t)))},
           "floor": {"type": "ply", "filename": write_ply(d / f"floor_{state}.ply", s * (fl[1] - c), fl[0]["F"]), "face_normals": True,
                     "bsdf": {"type": "diffuse", "reflectance": {"type": "rgb", "value": [FX.FLOOR_RHO] * 3}}}}
    for i, (sv, E) in enumerate(sd["lights"]):
        dv = np.asarray(sv, float); out[f"sun{i}"] = {"type": "directional", "direction": list(map(float, dv / np.linalg.norm(dv))),
                                                      "irradiance": {"type": "rgb", "value": [float(E)] * 3}}
    for cam, g in geo.items():
        o = s * (np.asarray(g["origin"]) - c); w = s * g["width"]
        T = mi.ScalarTransform4f.look_at(origin=list(map(float, o)), target=list(map(float, o + np.asarray(g["fwd"]))), up=list(map(float, g["up"])))
        out[cam] = {"type": "orthographic", "to_world": T @ mi.ScalarTransform4f.scale([w / 2, w / 2, 1.0]), "near_clip": 1e-6, "far_clip": 1e4,
                    "film": {"type": "hdrfilm", "width": g["res"], "height": g["res"], "rfilter": {"type": "box"}, "pixel_format": "rgb",
                             "component_format": "float32"}}
    return out
