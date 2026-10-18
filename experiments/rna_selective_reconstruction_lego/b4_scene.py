"""Architecture Batch 4 (worklog 37): Diffuse Interaction Variant A scene builder (RNA venv bpy; read-only on the official blend).

    python b4_scene.py <official Lego_rough_aovs_normalized.blend> <out dir>

Object A = the official RNA Lego (geometry, materials, transform unchanged; the qualified L16 partition stays valid).
Object B = an axis-aligned box (0.20 x 0.15 x 0.25 m) in a saturated red version of the official Lego plastic shader
(the 'Gray' node graph copied, only the Diffuse BSDF colour changed), floating just above the baseplate in the
free space under A's raised bucket, 3.6 mm (T1) / 12 mm (T0) above the highest plate point under its footprint.  The edit is a rigid translation of B by DELTA = (+0.44, 0, 0) (left -> right).
Lighting, camera protocol, render settings and AOVs are the upstream generator's; nothing else is added.

Variants (all saved as separate .blend files; the generator opens them unchanged):
  T0_s<k>  : B at T0_CENTRE                     (A at origin)
  T1_s<k>  : B at T0_CENTRE + DELTA             (A at origin)
  T1b_s<k> : the T1 arrangement translated by -DELTA (A at -DELTA, B at T0_CENTRE): the B-frame view of T1. The scene
             is lit only by the generator's directional sun (no world emission, checked here), so a rigid translation of
             the whole arrangement leaves B's radiance unchanged, and B's pixels correspond to the T0 views exactly.
  k = Cycles seed (0 = the official blend's seed, used for training data; 1, 2 = independent noise for references).
"""

import json
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix

B_SIZE = (0.20, 0.15, 0.25)
B_BOTTOM_Z = -0.186                 # plate top under the footprint: -0.198 (T0), -0.1897 (T1) (lego_mesh.npz); one height for both
T0_CENTRE = (-0.22, -0.56, B_BOTTOM_Z + B_SIZE[2] / 2)
DELTA = (0.44, 0.0, 0.0)
B_RED = (0.75, 0.03, 0.02, 1.0)     # linear diffuse albedo of B (saturated red)
VARIANTS = {"T0": ((0, 0, 0), (0, 0, 0)), "T1": ((0, 0, 0), DELTA), "T1b": (tuple(-d for d in DELTA), (0, 0, 0))}
SEEDS = {"T0": (0, 1, 2), "T1": (0, 1, 2), "T1b": (1, 2)}


def build(official, a_off, b_off, seed):
    bpy.ops.wm.open_mainfile(filepath=official)
    sc = bpy.context.scene
    lego = bpy.data.objects["Lego"]
    lego.location = a_off
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0, 0, 0))
    b = bpy.context.object
    b.name = "B_red_box"
    b.data.transform(Matrix.Diagonal((*B_SIZE, 1.0)))
    b.location = tuple(T0_CENTRE[i] + b_off[i] for i in range(3))
    mat = bpy.data.materials["Gray"].copy()
    mat.name = "B_Red"
    diffuse = [n for n in mat.node_tree.nodes if n.bl_idname == "ShaderNodeBsdfDiffuse"]
    assert len(diffuse) == 1
    diffuse[0].inputs["Color"].default_value = B_RED
    b.data.materials.append(mat)
    sc.cycles.seed = seed
    sc.cycles.use_animated_seed = False
    bpy.context.view_layer.update()
    return sc, lego, b, mat


def main():
    official, out = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:3]
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    rec = {"B_size": B_SIZE, "B_bottom_z": B_BOTTOM_Z, "T0_centre": T0_CENTRE, "delta": DELTA, "B_red_linear": B_RED, "variants": {}}
    for name, (a_off, b_off) in VARIANTS.items():
        for seed in SEEDS[name]:
            sc, lego, b, mat = build(official, a_off, b_off, seed)
            f = out / f"lego_b4_{name}_s{seed}.blend"
            bpy.ops.wm.save_as_mainfile(filepath=str(f), copy=True)
            rec["variants"][f"{name}_s{seed}"] = {"file": f.name, "lego_location": list(lego.location), "B_location": list(b.location), "cycles_seed": seed}
    # checks on the T0/T1 arrangements (A at origin): world emission, intersections, B mesh export (canonical = T0 placement)
    sc, lego, b, mat = build(official, (0, 0, 0), (0, 0, 0), 0)
    w = sc.world
    rec["world"] = {"nodes": [n.bl_idname for n in w.node_tree.nodes],
                    "links": [f"{l.from_node.bl_idname}.{l.from_socket.name}->{l.to_node.bl_idname}.{l.to_socket.name}" for l in w.node_tree.links]}
    out_node = [n for n in w.node_tree.nodes if n.bl_idname == "ShaderNodeOutputWorld"]
    rec["world"]["output_surface_linked"] = bool(out_node and out_node[0].inputs["Surface"].is_linked)
    rec["world"]["upstream_removes_node"] = "Background (blender_utils.disable_scene_lights)"
    me = b.data
    V = np.array([b.matrix_world @ v.co for v in me.vertices], np.float64)
    tris = []
    for p in me.polygons:
        vs = list(p.vertices)
        for i in range(1, len(vs) - 1):
            tris.append([vs[0], vs[i], vs[i + 1]])
    F = np.array(tris, np.int64)
    nrm = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    cen = V[F].mean(1); outward = ((cen - V.mean(0)) * nrm).sum(1) > 0
    np.savez(out / "B_box_T0.npz", V=V, F=F)
    rec["B_mesh"] = {"triangles": len(F), "area": float(0.5 * np.linalg.norm(nrm, axis=1).sum()), "normals_outward": bool(outward.all()),
                     "aabb_T0": [V.min(0).tolist(), V.max(0).tolist()]}
    mixes = {n.name: {i.name: (float(i.default_value) if not i.is_linked and i.name == "Fac" else ("linked" if i.is_linked else None)) for i in n.inputs}
             for n in mat.node_tree.nodes if n.bl_idname == "ShaderNodeMixShader"}
    rec["B_material"] = {"copied_from": "Gray", "diffuse_color": B_RED, "mix_nodes": mixes,
                         "glossy_roughness": [float(n.inputs["Roughness"].default_value) for n in mat.node_tree.nodes if n.bl_idname == "ShaderNodeBsdfGlossy"],
                         "aov_outputs": sorted(n.name for n in mat.node_tree.nodes if n.bl_idname == "ShaderNodeOutputAOV")}
    (out / "scene.json").write_text(json.dumps(rec, indent=1))
    print("done", json.dumps(rec["world"]), json.dumps(rec["B_mesh"]))


if __name__ == "__main__":
    main()
