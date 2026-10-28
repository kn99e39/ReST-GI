"""WL54 Lambert fixtures (protocol fnto_c54): exact local K (opaque Lambert, rho = 0.8), WL53 geometry conventions, and any
sub-port refinement s (s x s cells per 1 cm port edge, one unknown per cell).

Fixtures (all grey Lambert, rho 0.8, collimated sun E = 1 as in WL53):
  gap4 / gap8 / gap30   two closed 10 x 10 x 1 cm boxes, 4 / 8 / 30 mm air gap, sun 30 deg           (WL53 gap scenes, Lambert K)
  floor_lam             box 1 cm above a 30 x 30 cm Lambert floor, sun 40 deg az 20                  (WL53 amendment 4 control)
  lblock_lam            L-block (concave) 1 cm above the floor                                       (WL53 lblock, Lambert K)
  rot_lam               floor_lam with the box rotated 17 deg about z and shifted (3.7, -2.1) mm: the shadow edges and the box
                        edges no longer coincide with the floor-port grid (controlled sub-port visibility change)
Coarse meshes use the WL53 tessellation (sub = 4); fine meshes use sub = 1 (one cell = one unknown = 2 triangles).
"""
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "phys_ref53"))
import p53_fixtures as FX  # noqa: E402
import p53_geom as G  # noqa: E402

H = FX.H
NF = 30                                   # floor cells per edge (30 cm)
SCENES = ("gap4", "gap8", "gap30", "floor_lam", "lblock_lam", "rot_lam")
GAP = {"gap4": 0.004, "gap8": 0.008, "gap30": 0.030}
ROT_DEG, ROT_T = 17.0, (0.0037, -0.0021)
SLAB_OCC = np.ones((10, 10, 1), bool)
LB_OCC = np.zeros((10, 10, 2), bool); LB_OCC[:, :, 0] = True; LB_OCC[:5, :, 1] = True
SLAB_ORG, LB_ORG = (-0.05, -0.05, -0.005), (-0.05, -0.05, -0.01)


def sun_s(deg, az=0.0):
    return (math.sin(math.radians(deg)) * math.cos(math.radians(az)), math.sin(math.radians(deg)) * math.sin(math.radians(az)), -math.cos(math.radians(deg)))


def light_of(name):
    if name in GAP:
        return sun_s(30.0)
    if name == "lblock_lam":
        return (-math.sin(math.radians(45)), 0.15, -math.cos(math.radians(45)))
    return sun_s(40.0, 20.0)


def box_mesh(occ, org, s, coarse=False):
    """s = 1 and coarse=True gives the WL53 mesh (sub = 4); otherwise a sub = 1 mesh of voxel size H / s."""
    if coarse:
        return G.voxel_boundary(occ, (H, H, H), org, sub=FX.SUB)
    return G.voxel_boundary(np.kron(occ, np.ones((s, s, s), bool)), (H / s,) * 3, org, sub=1)


def floor_mesh(s, coarse=False):
    if coarse:
        return FX.floor_mesh(0.0, NF)
    return G.open_grid(NF * s, NF * s, H / s, (-NF * H / 2, -NF * H / 2, 0.0), sub=1)


def _rotz(V, deg, t):
    a = math.radians(deg); R = np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1.0]])
    return V @ R.T + np.array([t[0], t[1], 0.0])


def _transform(name, kind, V):
    """Placement of the box vertices in the scene (box kinds only)."""
    if name in GAP:
        return None
    Vc = FX.place_above(V, 0.0, 0.01)
    return _rotz(Vc, ROT_DEG, ROT_T) if name == "rot_lam" else Vc


def comps_at(name, s, coarse=False):
    """List of (comp name, mesh, Vcur) at refinement s (coarse=True: the WL53 1 cm ports, sub-4 tessellation)."""
    out = []
    if name in GAP:
        off = (H + GAP[name]) / 2
        m = box_mesh(SLAB_OCC, SLAB_ORG, s, coarse)
        out += [("slab_lo", m, m["V"] + [0, 0, -off]), ("slab_hi", m, m["V"] + [0, 0, off])]
        return out
    occ, org = (LB_OCC, LB_ORG) if name == "lblock_lam" else (SLAB_OCC, SLAB_ORG)
    m = box_mesh(occ, org, s, coarse)
    out.append(("dom", m, _transform(name, "box", m["V"].copy())))
    fm = floor_mesh(s, coarse); out.append(("floor", fm, fm["V"].copy()))
    return out


def parents(name, s):
    """For every fine port (concatenated over comps, SceneSet order) the global coarse port id (WL53 1 cm port numbering)."""
    cc = comps_at(name, 1, coarse=True); cf = comps_at(name, s); par = []; off = 0
    for (n, mc, _), (_, mf, _) in zip(cc, cf):
        if "port_face" in mc and n != "floor":
            key = {(int(v[0]), int(v[1]), int(v[2]), int(f)): i for i, (v, f) in enumerate(zip(mc["port_voxel"], mc["port_face"]))}
            p = [key[(int(v[0]) // s, int(v[1]) // s, int(v[2]) // s, int(f))] for v, f in zip(mf["port_voxel"], mf["port_face"])]
        else:
            nfine = NF * s; idx = np.arange(len(mf["port_area"])); p = ((idx // nfine) // s) * NF + ((idx % nfine) // s)
        par.append(np.asarray(p) + off); off += len(mc["port_area"])
    return np.concatenate(par)


def coarse_labels(name):
    """Port group label (component:face) per coarse port, as WL53 p53_report.port_groups."""
    face = {0: "-x", 1: "+x", 2: "-y", 3: "+y", 4: "bot", 5: "top"}; lab = []
    for n, m, _ in comps_at(name, 1, coarse=True):
        lab += [f"{n}:{face[int(f)]}" if n != "floor" else "floor" for f in m["port_face"]]
    return np.array(lab)


def scene_def(name):
    """WL53-style scene definition (coarse meshes) for the Cycles reference and the production C0."""
    comps = [(n, "lambert", m, V) for n, m, V in comps_at(name, 1, coarse=True)]
    sd = dict(comps=comps, lights=[(light_of(name), FX.E_SUN)], plate=False)
    if name in GAP:
        gap = GAP[name]; off = (H + gap) / 2
        cams = {k: FX._cam(k, (0, 0, 0), 0.1, 0.11, 220) for k in ("top", "bot", "otop", "obot")}
        cams["gapdown"] = FX._cam("top", (0, 0, -off), 0, 0.11, 220, origin=(0, 0, 0.0))
        cams["gapup"] = FX._cam("bot", (0, 0, off), 0, 0.11, 220, origin=(0, 0, 0.0))
    else:
        Vc = comps[0][3]; ctr = 0.5 * (Vc.min(0) + Vc.max(0))
        if name == "rot_lam":
            ctr = np.array([0.0, 0.0, ctr[2]])                     # same camera frame as floor_lam (box centred about its rotation axis)
        cams = {"top": FX._cam("top", ctr, 0.2, 0.31, 310), "otop": FX._cam("otop", ctr, 0.2, 0.16, 320),
                "underup": FX._cam("bot", ctr, 0, 0.13, 260, origin=(ctr[0], ctr[1], 0.004)),
                "underdown": FX._cam("top", (ctr[0], ctr[1], 0.0), 0, 0.13, 260, origin=(ctr[0], ctr[1], 0.004))}
        if name == "lblock_lam":
            cams["ocorner"] = FX._cam("ocorner", ctr, 0.2, 0.16, 320)
    sd["cams"] = cams
    return sd
