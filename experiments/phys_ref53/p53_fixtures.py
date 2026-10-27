"""WL53 fixtures: media, canonical geometry, poses / deformations, lights and cameras (protocol phys_ref53).

Every scene is built deterministically from these definitions, so the render process (Mitsuba llvm_ad_rgb) and the
oracle process (TR / C, Mitsuba cuda intersector) see bit-identical vertices.
"""
import math

import numpy as np

import p53_geom as G

SIGMA_T = 1000.0                  # m^-1 (mean free path 1 mm), achromatic
ALBEDO = (0.999, 0.99, 0.9)       # single-scattering albedo R / G / B
G_PHASE = 0.0                     # isotropic
H = 0.01                          # voxel / port size 1 cm
SUB = 4                           # 4 x 4 sub-cells x 2 triangles per port
E_SUN = 1.0
FLOOR_RHO = 0.8
DOUT, DIN = 64, int(__import__("os").environ.get("P53_DIN", 16))   # out 8 x 8, in 4 x 4 bins (amendment 3 diagnostic: P53_DIN=64)
S50, C50 = math.sin(math.radians(50)), math.cos(math.radians(50))
DIRS = dict(top=(0.0, 0.0, 1.0), bot=(0.0, 0.0, -1.0), otop=(0.0, S50, C50), obot=(0.0, -S50, -C50),
            ocorner=(math.sin(math.radians(60)), 0.0, math.cos(math.radians(60))))
R_POSE1 = G.rot((1.0, 2.0, 3.0), 37.0)


def slab_mesh():
    return G.voxel_boundary(np.ones((10, 10, 1), bool), (H, H, H), (-0.05, -0.05, -0.005), sub=SUB)


def lblock_mesh():
    occ = np.zeros((10, 10, 2), bool); occ[:, :, 0] = True; occ[:5, :, 1] = True
    return G.voxel_boundary(occ, (H, H, H), (-0.05, -0.05, -0.01), sub=SUB)


def floor_mesh(z=0.0, n=30):
    return G.open_grid(n, n, H, (-n * H / 2, -n * H / 2, z), sub=1)


def merge(meshes_and_V):
    """Merge (mesh, Vcur) pairs into one mesh (ports renumbered, rest vertices concatenated)."""
    V0, V1, F, tp, keys = [], [], [], [], ("port_center", "port_normal", "port_tangent", "port_area", "port_face", "port_voxel")
    acc = {k: [] for k in keys}; ov = op = 0
    for m, Vc in meshes_and_V:
        V0.append(m["V"]); V1.append(Vc); F.append(m["F"] + ov); tp.append(m["tri_port"] + op)
        for k in keys:
            acc[k].append(m[k])
        ov += len(m["V"]); op += len(m["port_area"])
    out = dict(V=np.concatenate(V0), F=np.concatenate(F), tri_port=np.concatenate(tp), **{k: np.concatenate(acc[k]) for k in keys})
    return out, np.concatenate(V1)


def place_above(V, z_floor=0.0, clearance=0.01, xy=(0.0, 0.0)):
    V = V.copy(); V[:, 2] += z_floor + clearance - V[:, 2].min(); V[:, 0] += xy[0]; V[:, 1] += xy[1]; return V


# ------------------------------------------------------------------ domain geometries (rest mesh, current vertices)


def domain_geometry(name):
    """name -> (mesh, Vcur in the domain's canonical placement before scene placement)."""
    if name == "slab":
        m = slab_mesh(); return m, m["V"].copy()
    if name == "lblock":
        m = lblock_mesh(); return m, m["V"].copy()
    if name == "lblock_pose1":
        m = lblock_mesh(); return m, G.map_rigid(m["V"], R_POSE1, (0.0, 0.0, 0.0))
    if name == "lblock_far":
        m = lblock_mesh(); return m, G.map_rigid(m["V"], R_POSE1, (10.0, -7.0, 3.0))
    if name == "bend30":
        m = slab_mesh(); return m, G.map_bend(m["V"], 30.0, 0.1)
    if name == "bend90":
        m = slab_mesh(); return m, G.map_bend(m["V"], 90.0, 0.1)
    if name == "dent":
        m = slab_mesh(); return m, G.map_dent(m["V"], 0.005, 0.02, H)
    if name.startswith("pair"):
        gap = {"pair4": 0.004, "pair8": 0.008}[name]; m = slab_mesh(); off = (H + gap) / 2
        return merge([(m, m["V"] + [0, 0, -off]), (m, m["V"] + [0, 0, off])])
    raise KeyError(name)


# ------------------------------------------------------------------ scenes


def _cam(dir_key, target, dist, width, res, origin=None, up=None):
    d = np.asarray(DIRS[dir_key], float)
    o = np.asarray(target, float) + dist * d if origin is None else np.asarray(origin, float)
    if up is None:
        up = (0.0, 1.0, 0.0) if abs(d[1]) < 0.9 and abs(d[2]) > 0.1 else (1.0, 0.0, 0.0)
        if abs(d[2]) < 0.1:
            up = (0.0, 0.0, 1.0)
    return dict(dir=tuple(d), origin=tuple(o), target=tuple(np.asarray(target, float)), up=tuple(up), width=width, res=res)


def scene_def(name):
    """Scene: comps (name, kind, mesh, Vcur, extra), lights [(s_travel, E)], cams {id: cam}, plate (bool), tri masks."""
    sun_s = lambda deg, az=0.0: (math.sin(math.radians(deg)) * math.cos(math.radians(az)), math.sin(math.radians(deg)) * math.sin(math.radians(az)), -math.cos(math.radians(deg)))  # noqa: E731
    slab, Vs = domain_geometry("slab")
    if name.startswith("iso_"):
        light = {"iso_L1": (0.0, 0.0, -1.0), "iso_L2": sun_s(50.0), "iso_L3": (0.0, 0.0, -1.0)}[name]
        cams = {k: _cam(k, (0, 0, 0), 0.1, 0.11, 220) for k in ("top", "bot", "otop", "obot")}
        sc = dict(comps=[("slab", "medium", slab, Vs)], lights=[(light, E_SUN)], cams=cams, plate=(name == "iso_L3"))
        if name == "iso_L3":
            fr = G.tri_frames(slab["V"], Vs, slab["F"], slab["tri_port"], slab["port_tangent"])
            c = fr["centroid"]; sc["sun_mask"] = {"slab": (np.abs(c[:, 0]) < 0.01) & (np.abs(c[:, 1]) < 0.01) & (fr["n"][:, 2] > 0.5)}
        return sc
    if name in ("gap4", "gap8", "gap4_below"):
        gap = 0.004 if name.startswith("gap4") else 0.008; off = (H + gap) / 2
        light = sun_s(30.0) if name != "gap4_below" else (0.0, 0.0, 1.0)
        cams = {k: _cam(k, (0, 0, 0), 0.1, 0.11, 220) for k in ("top", "bot", "otop", "obot")}
        cams["gapdown"] = _cam("top", (0, 0, -off), 0, 0.11, 220, origin=(0, 0, 0.0))
        cams["gapup"] = _cam("bot", (0, 0, off), 0, 0.11, 220, origin=(0, 0, 0.0))
        return dict(comps=[("slab_lo", "medium", slab, Vs + [0, 0, -off]), ("slab_hi", "medium", slab, Vs + [0, 0, off])],
                    lights=[(light, E_SUN)], cams=cams, plate=False, gap=gap)
    floor = floor_mesh(0.0)
    geo = {"floor": "slab", "lblock": "lblock", "lblock_pose1": "lblock_pose1", "bend30": "bend30", "bend90": "bend90", "dent": "dent"}
    if name in geo:
        m, Vc = domain_geometry(geo[name]); Vc = place_above(Vc)
        ctr = 0.5 * (Vc.min(0) + Vc.max(0)); zc = float(ctr[2])
        light = {"floor": sun_s(40.0, 20.0), "lblock": (-math.sin(math.radians(45)), 0.15, -math.cos(math.radians(45))),
                 "lblock_pose1": (-math.sin(math.radians(45)), 0.15, -math.cos(math.radians(45)))}.get(name, sun_s(35.0, 20.0))
        cams = {"top": _cam("top", ctr, 0.2, 0.31, 310), "otop": _cam("otop", ctr, 0.2, 0.16, 320),
                "underup": _cam("bot", ctr, 0, 0.13, 260, origin=(ctr[0], ctr[1], 0.004)),
                "underdown": _cam("top", (ctr[0], ctr[1], 0.0), 0, 0.13, 260, origin=(ctr[0], ctr[1], 0.004))}
        if name.startswith("lblock"):
            cams["ocorner"] = _cam("ocorner", ctr, 0.2, 0.16, 320)
        return dict(comps=[("dom", "medium", m, Vc), ("floor", "lambert", floor, floor["V"].copy())], lights=[(light, E_SUN)], cams=cams,
                    plate=False, zc=zc)
    raise KeyError(name)


def plate_mesh(z=0.5, half=0.5, hole=0.01):
    """Black occluder at height z with a square hole |x|, |y| < hole (normal -z so it faces the slab)."""
    rects = [(-half, -hole, -half, half), (hole, half, -half, half), (-hole, hole, -half, -hole), (-hole, hole, hole, half)]
    V, F = [], []
    for x0, x1, y0, y1 in rects:
        k = len(V); V += [(x0, y0, z), (x0, y1, z), (x1, y1, z), (x1, y0, z)]; F += [(k, k + 1, k + 2), (k, k + 2, k + 3)]
    return np.array(V, float), np.array(F)
