"""WL51 Gate 2 - Cycles path-semantics control for the Yellow Burley BSSRDF (protocol fnto_owner51, gate2).

Two parallel closed slabs 0.4 x 0.4 m, 4 mm thick, 4 mm air gap (bottom z in [0, 0.004], top z in [0.008, 0.012]) with the
blend's Yellow material (copied; displacement link removed, nothing else changed); black world; sun angle 0, E_SUN.
  J_below  one object, sun travelling +z (only the bottom slab's bottom face is lit)
  S_below  identical vertices as two objects (changed reference condition: object scope only)
  J_above  one object, sun from above at 45 deg (descriptive: camera vs diffuse-ancestor Yellow on a uniformly lit thin slab)
  L_above  as J_above with a Lambert diffuse of albedo 0.98 * Yellow colour instead of Yellow (measurement-chain control:
           no ancestor dependence, so E_meas / E_pred must be 1)
Measurements
  top    orthographic camera above (camera vertex on Yellow): L_top over the slab, 128^2
  pdiff  white Lambert patch (10 cm, h above the top face, facing down, invisible to shadow / diffuse / glossy /
         transmission rays) seen by an upward orthographic camera: E_meas = pi L_patch; Yellow has a DIFFUSE ancestor
  pmirr  the same patch as a white GGX roughness-0 mirror: L_mirror; Yellow has a GLOSSY / SINGULAR ancestor
  gap    S_below only: orthographic camera in the gap looking down (camera vertex on the bottom slab's top face)
E_pred (pdiff) assumes the camera-seen L_top is what a diffuse-ancestor ray sees: sum L_top cos cos / d^2 dA, averaged over
the measured patch footprint. Also: per-material mesh areas of the Lego / box (object scale applied).

    python o51_slab_render.py -- <out dir> [spp_top=1024] [spp_patch=4096] [smoke]
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

W = 0.4; ZB = (0.0, 0.004); ZT = (0.008, 0.012); H_PATCH = 0.1; P_HALF = 0.05
RES_TOP = 128; RES_P = 32


def cuboid(z0, z1, half, off):
    v = [(-half, -half, z0), (half, -half, z0), (half, half, z0), (-half, half, z0), (-half, -half, z1), (half, -half, z1), (half, half, z1), (-half, half, z1)]
    f = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]                         # outward normals
    return [tuple(x) for x in v], [tuple(i + off for i in q) for q in f]


def mesh_obj(name, parts, mat):
    V, F = [], []
    for z0, z1 in parts:
        v, f = cuboid(z0, z1, W / 2, len(V)); V += v; F += f
    me = bpy.data.meshes.new(name); me.from_pydata(V, [], F); me.update(); ob = bpy.data.objects.new(name, me); bpy.context.scene.collection.objects.link(ob)
    me.materials.append(mat); return ob


def quad(name, z, half, mat, flip=False):
    v = [(-half, -half, z), (half, -half, z), (half, half, z), (-half, half, z)]
    f = [(0, 3, 2, 1)] if flip else [(0, 1, 2, 3)]
    me = bpy.data.meshes.new(name); me.from_pydata(v, [], f); me.update(); ob = bpy.data.objects.new(name, me); bpy.context.scene.collection.objects.link(ob)
    me.materials.append(mat); return ob


def ortho(name, loc, rot, scale, clip=1e-4):
    cd = bpy.data.cameras.new(name); cd.type = "ORTHO"; cd.ortho_scale = scale; cd.clip_start = clip; cd.clip_end = 100.0
    ob = bpy.data.objects.new(name, cd); bpy.context.scene.collection.objects.link(ob); ob.location = Vector(loc); ob.rotation_euler = rot; return ob


def render(sc, cam, res, spp, seed, path):
    sc.camera = cam; r = sc.render; r.resolution_x = r.resolution_y = res; r.resolution_percentage = 100
    r.image_settings.file_format = "OPEN_EXR"; r.image_settings.color_depth = "32"; r.filepath = str(path)
    c = sc.cycles; c.samples = spp; c.seed = seed; bpy.context.view_layer.update(); bpy.ops.render.render(write_still=True)
    img = bpy.data.images.load(str(path) + ".exr"); a = np.empty(len(img.pixels), np.float32); img.pixels.foreach_get(a)
    out = a.reshape(res, res, 4)[..., :3].copy(); bpy.data.images.remove(img); Path(str(path) + ".exr").unlink(); return out


def mesh_areas():
    rec = {}
    for ob in bpy.data.objects:
        if ob.type != "MESH":
            continue
        me = ob.data; me.calc_loop_triangles(); nt = len(me.loop_triangles)
        co = np.empty(len(me.vertices) * 3, np.float64); me.vertices.foreach_get("co", co); co = co.reshape(-1, 3)
        Mw = np.array(ob.matrix_world); co = co @ Mw[:3, :3].T + Mw[:3, 3]
        tv = np.empty(nt * 3, np.int64); me.loop_triangles.foreach_get("vertices", tv); tv = tv.reshape(-1, 3)
        mi = np.empty(nt, np.int64); me.loop_triangles.foreach_get("material_index", mi)
        a = 0.5 * np.linalg.norm(np.cross(co[tv[:, 1]] - co[tv[:, 0]], co[tv[:, 2]] - co[tv[:, 0]]), axis=1)
        names = [s_.material.name if s_.material else "-" for s_ in ob.material_slots] or ["-"]
        rec[ob.name] = {names[k]: float(a[mi == k].sum()) for k in range(len(names))}
    return rec


def e_pred(Ltop, ztop):
    px = W / RES_TOP; xs = (np.arange(RES_TOP) + 0.5) * px - W / 2; X, Yg = np.meshgrid(xs, xs)
    q = (np.arange(4) + 0.5) / 4 * (P_HALF) - P_HALF / 2                                          # 4 x 4 points over the central 5 cm of the patch
    acc = np.zeros(3)
    for qx in q:
        for qy in q:
            d2 = (X - qx) ** 2 + (Yg - qy) ** 2 + H_PATCH ** 2; ct = H_PATCH / np.sqrt(d2)
            acc += (Ltop * (ct * ct / d2)[..., None] * px * px).reshape(-1, 3).sum(0)
    return acc / 16


def main():
    a = sys.argv[sys.argv.index("--") + 1:]; out = Path(a[0]); out.mkdir(parents=True, exist_ok=True); t0 = time.time()
    spp_top = int(a[1]) if len(a) > 1 else 1024; spp_p = int(a[2]) if len(a) > 2 else 4096; smoke = len(a) > 3 and a[3] == "smoke"
    bpy.ops.wm.open_mainfile(filepath=str(C.B4 / "scene/lego_b4_T0_s1.blend")); sc = bpy.context.scene; sc.frame_current = 1
    rec = {"blender": bpy.app.version_string, "spp_top": spp_top, "spp_patch": spp_p, "E_SUN": C.E_SUN, "geometry": {"W": W, "ZB": ZB, "ZT": ZT, "h_patch": H_PATCH, "patch": 2 * P_HALF},
           "mesh_areas_m2": mesh_areas(), "runs": {}}
    strip_lights(sc); sc.use_nodes = False
    for ob in list(sc.objects):
        ob.hide_render = True
    y = bpy.data.materials["Yellow"].copy(); y.name = "Yellow_wl51"
    for l in list(y.node_tree.links):
        if l.to_socket.name == "Displacement" and l.to_node.bl_idname == "ShaderNodeOutputMaterial":
            y.node_tree.links.remove(l)
    wm = bpy.data.materials.new("white51"); wm.use_nodes = True; nt = wm.node_tree; nt.nodes.clear(); o_ = nt.nodes.new("ShaderNodeOutputMaterial")
    df = nt.nodes.new("ShaderNodeBsdfDiffuse"); df.inputs["Color"].default_value = (1, 1, 1, 1); nt.links.new(df.outputs[0], o_.inputs["Surface"])
    mm = bpy.data.materials.new("mirror51"); mm.use_nodes = True; nt = mm.node_tree; nt.nodes.clear(); o_ = nt.nodes.new("ShaderNodeOutputMaterial")
    gl = nt.nodes.new("ShaderNodeBsdfGlossy"); gl.distribution = "GGX"; gl.inputs["Color"].default_value = (1, 1, 1, 1); gl.inputs["Roughness"].default_value = 0.0
    nt.links.new(gl.outputs[0], o_.inputs["Surface"])
    d = bpy.data.lights.new("s51", "SUN"); d.energy = C.E_SUN; d.angle = 0.0; sun = bpy.data.objects.new("s51", d); sc.collection.objects.link(sun)
    r = sc.render; r.engine = "CYCLES"; r.film_transparent = False; c = sc.cycles; c.device = "CPU"; c.use_adaptive_sampling = False; c.use_denoising = False
    c.sample_clamp_direct = 0.0; c.sample_clamp_indirect = 0.0; c.pixel_filter_type = "GAUSSIAN"; c.filter_width = 0.01; c.use_light_tree = False
    c.max_bounces = 128; c.diffuse_bounces = 128; c.glossy_bounces = 128; c.transmission_bounces = 128
    w = bpy.data.worlds.new("w51"); w.use_nodes = True; w.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.0; sc.world = w
    ztop = ZT[1]; ctop = ortho("c_top", (0, 0, 1.0), (0, 0, 0), W); cup = ortho("c_up", (0, 0, ztop + H_PATCH / 2), (math.pi, 0, 0), 2 * P_HALF)
    cgap = ortho("c_gap", (0, 0, 0.5 * (ZB[1] + ZT[0])), (0, 0, 0), W)
    if smoke:
        spp_top, spp_p = 16, 32
    lam = bpy.data.materials.new("lambert51"); lam.use_nodes = True; nt = lam.node_tree; nt.nodes.clear(); o_ = nt.nodes.new("ShaderNodeOutputMaterial")
    ld = nt.nodes.new("ShaderNodeBsdfDiffuse"); col = [n for n in y.node_tree.nodes if n.bl_idname == "ShaderNodeSubsurfaceScattering"][0].inputs["Color"].default_value
    ld.inputs["Color"].default_value = (0.98 * col[0], 0.98 * col[1], 0.98 * col[2], 1); nt.links.new(ld.outputs[0], o_.inputs["Surface"])
    configs = {"J_below": (True, (math.pi, 0, 0), y), "S_below": (False, (math.pi, 0, 0), y), "J_above": (True, (math.radians(45), 0, 0), y),
               "L_above": (True, (math.radians(45), 0, 0), lam)}
    for cfg, (joined, srot, smat) in configs.items():
        sun.rotation_euler = srot                                                                     # default sun travels -z; pi about x -> +z
        objs = [mesh_obj("slab", [ZB, ZT], smat)] if joined else [mesh_obj("slab_b", [ZB], smat), mesh_obj("slab_t", [ZT], smat)]
        rr = {"objects": [o.name for o in objs], "sun_rotation_euler": list(srot)}; rec["runs"][cfg] = rr
        for seed in (5101, 5102):
            m = {}
            Lt = render(sc, ctop, RES_TOP, spp_top, seed, out / f"top_{cfg}_{seed}"); np.save(out / f"Ltop_{cfg}_{seed}.npy", Lt)
            c0, c1 = RES_TOP // 2 - 16, RES_TOP // 2 + 16; m["L_top_center_RGB"] = Lt[c0:c1, c0:c1].reshape(-1, 3).mean(0).tolist()
            m["L_top_mean_RGB"] = Lt.reshape(-1, 3).mean(0).tolist()
            for kind, mat in (("pdiff", wm), ("pmirr", mm)):
                pt = quad(f"patch_{kind}", ztop + H_PATCH, P_HALF, mat, flip=True)
                pt.visible_shadow = False; pt.visible_diffuse = False; pt.visible_glossy = False; pt.visible_transmission = False
                Lp = render(sc, cup, RES_P, spp_p, seed + 7, out / f"{kind}_{cfg}_{seed}"); np.save(out / f"L{kind}_{cfg}_{seed}.npy", Lp)
                cen = Lp[8:24, 8:24].reshape(-1, 3).mean(0)
                if kind == "pdiff":
                    Ep = e_pred(Lt, ztop); m["pdiff"] = {"E_meas_RGB": (math.pi * cen).tolist(), "E_pred_RGB": Ep.tolist(), "ratio_RGB": (math.pi * cen / np.maximum(Ep, 1e-30)).tolist()}
                else:
                    ref = Lt[RES_TOP // 2 - 8:RES_TOP // 2 + 8, RES_TOP // 2 - 8:RES_TOP // 2 + 8].reshape(-1, 3).mean(0)          # central 5 cm = the patch's central 16 px
                    ref5 = Lt[RES_TOP // 2 - 16:RES_TOP // 2 + 16, RES_TOP // 2 - 16:RES_TOP // 2 + 16].reshape(-1, 3).mean(0)      # central 10 cm
                    m["pmirr"] = {"L_mirror_RGB": cen.tolist(), "L_top_same_footprint_RGB": ref.tolist(), "ratio_RGB": (cen / np.maximum(ref, 1e-30)).tolist(), "L_top_10cm_RGB": ref5.tolist()}
                bpy.data.objects.remove(pt)
            if not joined:
                Lg = render(sc, cgap, RES_TOP, spp_top, seed + 11, out / f"gap_{cfg}_{seed}"); np.save(out / f"Lgap_{cfg}_{seed}.npy", Lg)
                m["L_gap_center_RGB"] = Lg[c0:c1, c0:c1].reshape(-1, 3).mean(0).tolist()
            m["wall_s"] = time.time() - t0; rr[str(seed)] = m; print(cfg, seed, json.dumps(m), flush=True)
            (out / "slab.json").write_text(json.dumps(rec, indent=1))
        for o in objs:
            bpy.data.objects.remove(o)
    rec["E_sun_over_pi"] = C.E_SUN / math.pi; rec["wall_s"] = time.time() - t0; (out / "slab.json").write_text(json.dumps(rec, indent=1)); print(json.dumps(rec, indent=1))


main()
