"""WL49 Gate 0 - direct Cycles measurement of the Lego Yellow first-event exit radiance (bpy 3.5.1, docker on LabServer69).

lego_b4_T0_s1.blend, lights stripped, one sun exactly as the first_r1 references (strength E_SUN, angle 0, held-out T0
view 6 direction), Cycles CPU. DIFFUSE bake (pass filter DIRECT + COLOR; Cycles 3.x merges subsurface into the diffuse
passes; the Yellow glossy 2 % is excluded) into a POINT-domain float colour attribute of the Lego object -> per-vertex
exit radiance (isotropic exit). Writes vertex radiance, world-space triangle areas / vertices / material flags.

    python g0_exitance_bake.py -- <out dir> <spp> <seed> [view=6]
"""
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "fnto_lego_r1")); sys.path.insert(0, str(HERE.parent / "fnto_lego"))
import bpy  # noqa: E402
from mathutils import Vector  # noqa: E402

import r1_common as C  # noqa: E402
from r1_stagea_render import strip_lights  # noqa: E402


def main():
    a = sys.argv[sys.argv.index("--") + 1:]; out = Path(a[0]); spp = int(a[1]); seed = int(a[2]); view = int(a[3]) if len(a) > 3 else 6
    out.mkdir(parents=True, exist_ok=True); t0 = time.time()
    bpy.ops.wm.open_mainfile(filepath=str(C.B4 / "scene/lego_b4_T0_s1.blend")); sc = bpy.context.scene; sc.frame_current = 1; strip_lights(sc)
    views = json.loads((C.ROOT.parent / "views_r1.json").read_text()) if (C.ROOT.parent / "views_r1.json").exists() else json.loads(Path.home().joinpath("rest_fnto_r1/views_r1.json").read_text())
    d = bpy.data.lights.new("g0_sun", "SUN"); d.energy = C.E_SUN; d.angle = 0.0
    sun = bpy.data.objects.new("g0_sun", d); sc.collection.objects.link(sun); sun.rotation_mode = "QUATERNION"
    sun.rotation_quaternion = Vector((0, 0, 1)).rotation_difference(Vector(views["heldout_T0"][view]["sun"]))
    lego = [o for o in sc.objects if o.type == "MESH" and any(s.material and s.material.name == "Yellow" for s in o.material_slots)]
    assert len(lego) == 1, [o.name for o in lego]
    ob = lego[0]; me = ob.data
    for o in sc.objects:
        o.select_set(False)
    ob.select_set(True); bpy.context.view_layer.objects.active = ob
    attr = me.color_attributes.new("g0_bake", "FLOAT_COLOR", "POINT"); me.color_attributes.active_color = attr
    r = sc.render; r.engine = "CYCLES"; c = sc.cycles; c.device = "CPU"; c.samples = spp; c.seed = seed; c.use_adaptive_sampling = False
    c.use_denoising = False; c.sample_clamp_direct = 0.0; c.sample_clamp_indirect = 0.0; c.use_light_tree = False
    c.max_bounces = 128; c.diffuse_bounces = 128; c.glossy_bounces = 128; c.transmission_bounces = 128; c.transparent_max_bounces = 128
    sc.render.bake.target = "VERTEX_COLORS"
    bpy.context.view_layer.update(); tb = time.time()
    bpy.ops.object.bake(type="DIFFUSE", pass_filter={"DIRECT", "COLOR"}, target="VERTEX_COLORS")
    t_bake = time.time() - tb
    col = np.empty(len(me.vertices) * 4, np.float32); me.color_attributes["g0_bake"].data.foreach_get("color", col); col = col.reshape(-1, 4)[:, :3]
    me.calc_loop_triangles(); nt = len(me.loop_triangles)
    tv = np.empty(nt * 3, np.int64); me.loop_triangles.foreach_get("vertices", tv); tv = tv.reshape(-1, 3)
    tm = np.empty(nt, np.int64); me.loop_triangles.foreach_get("material_index", tm)
    co = np.empty(len(me.vertices) * 3, np.float32); me.vertices.foreach_get("co", co); co = co.reshape(-1, 3).astype(np.float64)
    M = np.array(ob.matrix_world); cw = co @ M[:3, :3].T + M[:3, 3]
    area = 0.5 * np.linalg.norm(np.cross(cw[tv[:, 1]] - cw[tv[:, 0]], cw[tv[:, 2]] - cw[tv[:, 0]]), axis=1)
    yslots = [i for i, s in enumerate(ob.material_slots) if s.material and s.material.name == "Yellow"]
    tri_yel = np.isin(tm, yslots)
    Lt = col[tv].mean(1)                                                              # [nt, 3] mean vertex radiance per triangle
    rec = {"blender": bpy.app.version_string, "object": ob.name, "verts": len(me.vertices), "tris": nt, "spp": spp, "seed": seed, "view": view,
           "sun_dir": views["heldout_T0"][view]["sun"], "E_SUN": C.E_SUN, "t_bake_s": t_bake, "t_total_s": time.time() - t0,
           "yellow_area_m2": float(area[tri_yel].sum()), "total_area_m2": float(area.sum()),
           "yellow_exit_flux_RGB": (math.pi * (Lt[tri_yel] * area[tri_yel, None]).sum(0)).tolist(),
           "all_exit_flux_RGB": (math.pi * (Lt * area[:, None]).sum(0)).tolist(), "finite": bool(np.isfinite(col).all()), "max_radiance": float(col.max())}
    np.savez_compressed(out / f"bake_spp{spp}_seed{seed}.npz", col=col.astype(np.float32), tv=tv.astype(np.int32), tri_yel=tri_yel, area=area.astype(np.float32), cw=cw.astype(np.float32))
    (out / f"bake_spp{spp}_seed{seed}.json").write_text(json.dumps(rec, indent=1)); print(json.dumps(rec))


main()
