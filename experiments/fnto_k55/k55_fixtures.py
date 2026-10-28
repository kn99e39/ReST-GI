"""WL55 scene definitions shared by the oracle side (GPU) and the Cycles renders (protocol fnto_k55). No renderer import.

Every scene is a dict(comps=[(name, kind, mesh, Vcur)], lights=[(s_travel, E)], cams={id: cam}, emit={comp: profile}) with kind in
{medium, lambert, black, emitter}. Meshes are the WL53 meshes (1 cm ports, 4 x 4 sub-cells x 2 triangles) built by p53_fixtures, so
port numbering and group labels equal WL53's. Cameras render at 0.5 mm pixels (WL54 A3: 0.5 mm is converged for the medium groups).

  gap4 / gap8 / floor / bend90 / dent    WL53 scenes (+ obotx camera on the gap scenes)
  kf_<gap4|gap8>_<U|X|S>                 known-field single-flight fixtures: slab_lo medium; slab_hi black, its bottom face a one-sided
                                         Lambertian emitter of radiance 1 (U) or exp(-(x + 0.05) / 0.005) (X); S: no emitter, the sun
  gap6s                                  HELD-OUT: 6 mm gap, slab_hi shifted (3.7, -2.1) mm, sun zenith 35 deg azimuth 25 deg
  floor_c4                               Stage E: the floor scene's slab at 4 mm clearance, shifted (2.3, 1.7) mm
"""
import math

import numpy as np

import k55_common as KC  # noqa: F401  (registers the obotx camera direction)
import p53_fixtures as FX
import c54_fixtures as CF

H = FX.H
PX = 0.0005
LAMBDA_X = 0.005
GAP = {"gap4": 0.004, "gap6s": 0.006, "gap8": 0.008}


def sun_s(deg, az=0.0):
    return (math.sin(math.radians(deg)) * math.cos(math.radians(az)), math.sin(math.radians(deg)) * math.sin(math.radians(az)), -math.cos(math.radians(deg)))


def with_px(cams, px=PX):
    out = {}
    for k, c in cams.items():
        c = dict(c); c["res"] = int(round(c["width"] / px)); out[k] = c
    return out


def profile(name):
    """Emitter radiance as a function of world positions X (N, 3) (numpy or torch)."""
    if name == "U":
        return lambda X: X[:, 0] * 0 + 1.0
    if name == "X":
        def f(X):
            import torch
            lib = torch if isinstance(X, torch.Tensor) else np
            return lib.exp(-(X[:, 0] + 0.05) / LAMBDA_X)
        return f
    raise KeyError(name)


def submesh(m, port_mask):
    """Restrict a WL53 mesh to the ports in port_mask (ports renumbered in their original order; vertices kept)."""
    keep_p = np.nonzero(port_mask)[0]; newid = -np.ones(len(port_mask), int); newid[keep_p] = np.arange(len(keep_p))
    tk = port_mask[m["tri_port"]]
    out = dict(V=m["V"], F=m["F"][tk], tri_port=newid[m["tri_port"][tk]])
    for k in ("port_center", "port_normal", "port_tangent", "port_area", "port_face", "port_voxel"):
        out[k] = m[k][keep_p]
    return out


def gap_cams(off_lo, off_hi):
    cams = {k: FX._cam(k, (0, 0, 0), 0.1, 0.11, 220) for k in ("top", "bot", "otop", "obot", "obotx")}
    cams["gapdown"] = FX._cam("top", (0, 0, -off_lo), 0, 0.11, 220, origin=(0, 0, 0.0))
    cams["gapup"] = FX._cam("bot", (0, 0, off_hi), 0, 0.11, 220, origin=(0, 0, 0.0))
    return cams


def scene_def(name):
    slab = FX.slab_mesh(); Vs = slab["V"]
    if name in ("gap4", "gap8", "gap6s"):
        gap = GAP[name]; off = (H + gap) / 2; sh = (0.0037, -0.0021) if name == "gap6s" else (0.0, 0.0)
        light = sun_s(35.0, 25.0) if name == "gap6s" else sun_s(30.0)
        comps = [("slab_lo", "medium", slab, Vs + [0, 0, -off]), ("slab_hi", "medium", slab, Vs + [sh[0], sh[1], off])]
        return dict(comps=comps, lights=[(light, FX.E_SUN)], cams=with_px(gap_cams(off, off)), emit={}, gap=gap)
    if name.startswith("kf_"):
        _, g, field = name.split("_"); gap = GAP[g]; off = (H + gap) / 2
        bot = slab["port_face"] == 4
        comps = [("slab_lo", "medium", slab, Vs + [0, 0, -off]),
                 ("hi_bot", "black" if field == "S" else "emitter", submesh(slab, bot), Vs + [0, 0, off]),
                 ("hi_rest", "black", submesh(slab, ~bot), Vs + [0, 0, off])]
        cams = {k: c for k, c in gap_cams(off, off).items() if k in ("gapdown", "obot", "obotx", "bot")}
        lights = [(sun_s(30.0), FX.E_SUN)] if field == "S" else []
        return dict(comps=comps, lights=lights, cams=with_px(cams), emit={} if field == "S" else {"hi_bot": field}, gap=gap)
    if name in ("floor", "bend90", "dent", "bend30", "lblock"):
        sd = FX.scene_def(name); sd["cams"] = with_px(sd["cams"]); sd["emit"] = {}; return sd
    if name == "floor_c4":
        sd = FX.scene_def("floor"); m, V = FX.domain_geometry("slab"); Vc = FX.place_above(V, 0.0, 0.004, xy=(0.0023, 0.0017))
        sd["comps"][0] = ("dom", "medium", m, Vc); ctr = 0.5 * (Vc.min(0) + Vc.max(0))
        sd["cams"] = with_px({"top": FX._cam("top", ctr, 0.2, 0.31, 310), "otop": FX._cam("otop", ctr, 0.2, 0.16, 320),
                              "underup": FX._cam("bot", ctr, 0, 0.13, 260, origin=(ctr[0], ctr[1], 0.002)),
                              "underdown": FX._cam("top", (ctr[0], ctr[1], 0.0), 0, 0.13, 260, origin=(ctr[0], ctr[1], 0.002))})
        sd["emit"] = {}; return sd
    if name == "iso_L2":
        sd = FX.scene_def("iso_L2"); sd["cams"] = with_px(sd["cams"]); sd["emit"] = {}; return sd
    if name == "floor_lam":
        sd = CF.scene_def("floor_lam"); sd["cams"] = with_px(sd["cams"]); sd["emit"] = {}; return sd
    raise KeyError(name)


def medium_ports(sd):
    """bool per global coarse port: True for medium components (the ruled population of the K-only fixtures)."""
    return np.concatenate([np.full(len(m["port_area"]), k == "medium") for n, k, m, V in sd["comps"]])
