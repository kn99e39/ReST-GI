"""WL44 A6 - pixel light-visibility rule vs the renderer's geometry (diagnostic; no training input is changed here).

Step 'mi' (8dna26 venv, Mitsuba scalar_rgb on the CPU): for a random sample of Lego pixels of a held-out probe and every
light of the configuration (facing by the camera-facing shading normal, spot cone as r1_prep), visibility under
  r0       the r1_prep rule: origin p + 1e-4 n, any hit before the light occludes
  tmin03   hits closer than 0.3 mm to the origin are ignored
  tmin1    hits closer than 1 mm are ignored
  off1     origin p + 1e-3 n
Step 'bpy' (RNA venv, Blender BVH on the evaluated Stage A scene incl. masks / box, origin p + 2e-4 n): the renderer
geometry proxy for the same (pixel, light) pairs, then agreement of each rule with it, overall and split by the r0 bit.

    python k44_visrule.py mi <i> [n=600]
    RNA_DIR=<rna repo> python k44_visrule.py bpy -- <i>        -> results/fnto_lego_k44/A6_visrule_<i>.json
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "fnto_lego_r1")); sys.path.insert(0, str(HERE.parent / "fnto_lego")); sys.path.insert(0, str(HERE))
import r1_common as C  # noqa: E402

WORK = Path.home() / "rest_fnto_r1/k44_visrule"


def pairs(i, n):
    cfg = {c["i"]: c for c in json.loads((C.OUT1 / "stagea/holdout/configs.json").read_text())}[i]
    d = C.OUT1 / "stagea/prep_holdout"; start = np.load(d / "start.npy"); s, e = start[i], start[i + 1]
    pos = np.load(d / "px_pos.npy", mmap_mode="r")[s:e].astype(np.float64); nrm = np.load(d / "px_nrm.npy", mmap_mode="r")[s:e].astype(np.float64)
    vb = np.load(d / "px_vbits.npy", mmap_mode="r")[s:e]; dom = np.load(d / "px_dom.npy", mmap_mode="r")[s:e]
    sel = np.random.default_rng(i).choice(np.flatnonzero(dom == 0), n, replace=False)
    out = []
    for k in sel:
        p = pos[k]; nn = nrm[k] / np.linalg.norm(nrm[k])
        for jl, L in enumerate(cfg["lights"]):
            if L["type"] == "sun":
                w = np.asarray(L["dir"], np.float64); w /= np.linalg.norm(w); dist = 1e4
            else:
                v = np.asarray(L["pos"], np.float64) - p; dist = float(np.linalg.norm(v)); w = v / dist
                if L["type"] == "spot" and float(-w @ np.asarray(L["axis"])) < L["cos_half"]:
                    continue
            if nn @ w <= 0:
                continue
            out.append((int(k), jl, p, nn, w, dist, bool((int(vb[k]) >> jl) & 1)))
    return cfg, out


def run_mi(i, n):
    import mitsuba as mi
    mi.set_variant("scalar_rgb")
    cfg, P = pairs(i, n)
    m = np.load(C.MESH_A); bV = np.load(C.BOX_T0)

    def mesh(name, V, F):
        me = mi.Mesh(name, V.shape[0], F.shape[0], has_vertex_normals=False); p = mi.traverse(me)
        p["vertex_positions"] = type(p["vertex_positions"])(np.ascontiguousarray(V.astype(np.float32).ravel()))
        p["faces"] = type(p["faces"])(np.ascontiguousarray(F.astype(np.uint32).ravel())); p.update(); return me
    shapes = {"type": "scene", "A": mesh("A", m["V"], m["F"])}
    if cfg["box_offset"] is not None:
        shapes["B"] = mesh("B", bV["V"] + np.asarray(cfg["box_offset"]), bV["F"])
    for k, s in enumerate(cfg["masks"]):
        shapes[f"M{k}"] = {"type": "sphere", "center": list(map(float, s["c"])), "radius": float(s["r"])}
    sc = mi.load_dict(shapes)

    def vis(o, w, dist, tmin):
        o = o + tmin * w
        r = mi.Ray3f(mi.Point3f(*o), mi.Vector3f(*w)); r.maxt = float(max(dist - 1e-4 - tmin, 1e-6))
        return not sc.ray_test(r)
    rows = []
    for k, jl, p, nn, w, dist, bit in P:
        rows.append([k, jl, bit, vis(p + 1e-4 * nn, w, dist, 0.0), vis(p + 1e-4 * nn, w, dist, 3e-4), vis(p + 1e-4 * nn, w, dist, 1e-3),
                     vis(p + 1e-3 * nn, w, dist, 0.0)])
    WORK.mkdir(parents=True, exist_ok=True)
    np.save(WORK / f"mi_{i}.npy", np.array(rows, np.int64)); print("mi", i, len(rows), flush=True)


def run_bpy(i):
    sys.path.insert(0, os.environ.get("RNA_DIR", str(HERE.parents[1] / "external/relightable-neural-assets")))
    import bpy
    from mathutils import Vector
    import r1_stagea_render as SR
    rows = np.load(WORK / f"mi_{i}.npy"); cfg, P = pairs(i, len(np.unique(rows[:, 0])))
    key = {(k, jl): (p, nn, w, dist) for k, jl, p, nn, w, dist, _ in P}
    scn = SR.Scene(64, 1); scn.apply(dict(cfg)); dg = bpy.context.evaluated_depsgraph_get(); sc = bpy.context.scene
    bl = []
    for r in rows:
        p, nn, w, dist = key[(int(r[0]), int(r[1]))]
        hit, _, _, _, obj, _ = sc.ray_cast(dg, Vector(p + 2e-4 * nn), Vector(w), distance=dist - 1e-4)
        if hit and obj is not None and obj.name.startswith("B_") and cfg["box_offset"] is None:
            hit = False
        bl.append(not hit)
    bl = np.array(bl)
    names = ["r0", "tmin03", "tmin1", "off1"]
    rec = {"i": i, "family": cfg["family"], "pairs": int(len(rows)), "blender_visible_frac": float(bl.mean()), "agreement_with_blender": {}}
    for c, nm in enumerate(names):
        v = rows[:, 3 + c].astype(bool)
        rec["agreement_with_blender"][nm] = {"all": float((v == bl).mean()), "false_occluded": float((~v & bl).mean()), "false_visible": float((v & ~bl).mean())}
    rec["r0_equals_prep_bits"] = float((rows[:, 3].astype(bool) == rows[:, 2].astype(bool)).mean())
    out = C.ROOT / "results/fnto_lego_k44"; out.mkdir(parents=True, exist_ok=True)
    (out / f"A6_visrule_{i}.json").write_text(json.dumps(rec, indent=1)); print(json.dumps(rec)); sys.stdout.flush(); os._exit(0)


if __name__ == "__main__":
    if sys.argv[1] == "mi":
        run_mi(int(sys.argv[2]), int(sys.argv[3]) if len(sys.argv) > 3 else 600)
    else:
        a = sys.argv[sys.argv.index("--") + 1:]; run_bpy(int(a[0]))
