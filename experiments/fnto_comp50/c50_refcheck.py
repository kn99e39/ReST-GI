"""WL50 amendment 2 - Cycles reference semantics: does an INDIRECT ray see the Yellow (Burley) exit radiance a camera sees?

Scene (bpy 3.5.1, CPU): the blend's Yellow material (displacement link removed, as worklog 47) on a 1 x 1 m plane at z = 0
(normal +z), a zenith sun (strength E_SUN, angle 0), black world, every other object hidden. A white Lambertian (rho 1)
10 x 10 cm patch at z = h facing down, invisible to shadow / diffuse / glossy rays (it neither shadows the plane nor
reflects back).
  R1  orthographic camera from +z, patch hidden: camera-seen Yellow radiance L_cam(x) over the plane (128 x 128).
  R2  orthographic camera at z = h / 2 looking up at the patch: L_patch = E_ind / pi (rho 1; full GI, max bounces 128).
Prediction (isotropic exit assumed, numerical over R1 pixels): E_pred = sum L_cam cos(t_y) cos(t_r) / d^2 dA.
Ratio E_meas / E_pred = 1 -> indirect rays see the camera radiance. Two heights, replicate seeds.

    python c50_refcheck.py -- <out dir> [spp=1024] [sss_only]
'sss_only' (amendment 2b): the Mix Shader factor is set to 0 (pure Burley BSSRDF, no 2 % glossy) - the first run's
orthographic top camera sat in the sun's mirror direction everywhere, so its L_cam included the glossy highlight.
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


def quad(name, z, half, mat, flip=False):
    v = [(-half, -half, z), (half, -half, z), (half, half, z), (-half, half, z)]
    f = [(0, 3, 2, 1)] if flip else [(0, 1, 2, 3)]
    me = bpy.data.meshes.new(name); me.from_pydata(v, [], f); me.update(); ob = bpy.data.objects.new(name, me); bpy.context.scene.collection.objects.link(ob)
    me.materials.append(mat); return ob


def render(sc, cam, res, spp, seed, path):
    sc.camera = cam; r = sc.render; r.resolution_x = r.resolution_y = res; r.resolution_percentage = 100
    r.image_settings.file_format = "OPEN_EXR"; r.image_settings.color_depth = "32"; r.filepath = str(path)
    c = sc.cycles; c.samples = spp; c.seed = seed; bpy.context.view_layer.update(); bpy.ops.render.render(write_still=True)
    img = bpy.data.images.load(str(path) + ".exr"); a = np.empty(len(img.pixels), np.float32); img.pixels.foreach_get(a)
    out = a.reshape(res, res, 4)[..., :3].copy(); bpy.data.images.remove(img); Path(str(path) + ".exr").unlink(); return out


def main():
    a = sys.argv[sys.argv.index("--") + 1:]; out = Path(a[0]); out.mkdir(parents=True, exist_ok=True); spp = int(a[1]) if len(a) > 1 else 1024; t0 = time.time()
    sss_only = len(a) > 2 and a[2] == "sss_only"
    bpy.ops.wm.open_mainfile(filepath=str(C.B4 / "scene/lego_b4_T0_s1.blend")); sc = bpy.context.scene; sc.frame_current = 1; strip_lights(sc); sc.use_nodes = False
    for ob in list(sc.objects):
        ob.hide_render = True
    y = bpy.data.materials["Yellow"].copy(); y.name = "Yellow_wl50"
    for l in list(y.node_tree.links):
        if l.to_socket.name == "Displacement" and l.to_node.bl_idname == "ShaderNodeOutputMaterial":
            y.node_tree.links.remove(l)
    if sss_only:
        mx = [n for n in y.node_tree.nodes if n.bl_idname == "ShaderNodeMixShader"][0]; mx.inputs["Fac"].default_value = 0.0
        assert mx.inputs[1].links[0].from_node.bl_idname == "ShaderNodeSubsurfaceScattering"            # Fac 0 -> input 1 = BSSRDF
    wm = bpy.data.materials.new("white50"); wm.use_nodes = True; nt = wm.node_tree; nt.nodes.clear()
    o_ = nt.nodes.new("ShaderNodeOutputMaterial"); df = nt.nodes.new("ShaderNodeBsdfDiffuse"); df.inputs["Color"].default_value = (1, 1, 1, 1); nt.links.new(df.outputs[0], o_.inputs["Surface"])
    quad("yplane", 0.0, 0.5, y)
    d = bpy.data.lights.new("s50", "SUN"); d.energy = C.E_SUN; d.angle = 0.0; sun = bpy.data.objects.new("s50", d); sc.collection.objects.link(sun)
    r = sc.render; r.engine = "CYCLES"; r.film_transparent = False; c = sc.cycles; c.device = "CPU"; c.use_adaptive_sampling = False; c.use_denoising = False
    c.sample_clamp_direct = 0.0; c.sample_clamp_indirect = 0.0; c.pixel_filter_type = "GAUSSIAN"; c.filter_width = 0.01; c.use_light_tree = False
    c.max_bounces = 128; c.diffuse_bounces = 128; c.glossy_bounces = 128; c.transmission_bounces = 128
    w = bpy.data.worlds.new("w50"); w.use_nodes = True; w.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.0; sc.world = w
    cd = bpy.data.cameras.new("c_top"); cd.type = "ORTHO"; cd.ortho_scale = 1.0; ctop = bpy.data.objects.new("c_top", cd); sc.collection.objects.link(ctop)
    ctop.location = Vector((0, 0, 2.0)); ctop.rotation_euler = (0, 0, 0)
    rec = {"blender": bpy.app.version_string, "spp": spp, "sss_only": sss_only, "E_SUN": C.E_SUN, "runs": []}
    for seed in (5021, 5121):
        L1 = render(sc, ctop, 128, max(256, spp // 2), seed, out / f"r1_{seed}")                       # camera-seen Yellow radiance
        rec.setdefault("L_cam_center_mean_RGB", []).append(L1[48:80, 48:80].reshape(-1, 3).mean(0).tolist())
        np.save(out / f"Lcam_{seed}.npy", L1)
        px = 1.0 / 128; xs = (np.arange(128) + 0.5) * px - 0.5
        X, Yg = np.meshgrid(xs, xs)
        for hgt in (0.3, 0.6):
            pt = quad(f"patch{hgt}", hgt, 0.05, wm, flip=True)                                         # normal -z
            pt.visible_shadow = False; pt.visible_diffuse = False; pt.visible_glossy = False; pt.visible_transmission = False
            cdp = bpy.data.cameras.new(f"c_up{hgt}"); cdp.type = "ORTHO"; cdp.ortho_scale = 0.1
            cup = bpy.data.objects.new(f"c_up{hgt}", cdp); sc.collection.objects.link(cup); cup.location = Vector((0, 0, hgt / 2)); cup.rotation_euler = (math.pi, 0, 0)
            L2 = render(sc, cup, 32, spp, seed + 7, out / f"r2_{seed}_{hgt}")
            E_meas = math.pi * L2[8:24, 8:24].reshape(-1, 3).mean(0)
            d2 = X ** 2 + Yg ** 2 + hgt ** 2; ct = hgt / np.sqrt(d2)                                   # cos at plane and at patch (parallel planes)
            E_pred = ((L1[::-1] if False else L1) * (ct * ct / d2)[..., None] * px * px).reshape(-1, 3).sum(0)
            rec["runs"].append({"seed": seed, "h": hgt, "E_meas_RGB": E_meas.tolist(), "E_pred_RGB": E_pred.tolist(), "ratio_RGB": (E_meas / E_pred).tolist()})
            print(json.dumps(rec["runs"][-1]), flush=True)
            bpy.data.objects.remove(pt); bpy.data.objects.remove(cup)
    rec["wall_s"] = time.time() - t0; (out / "refcheck.json").write_text(json.dumps(rec, indent=1))


main()
