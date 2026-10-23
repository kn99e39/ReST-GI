"""Worklog 42: renderer-side verification of audit findings F4 (internal reflection / TIR) and F5 (clamping, filter,
inherited settings). WSL RNA venv bpy 3.5.1, CPU. Read-only on every scene file (nothing is saved).

    python r1_render_checks.py -- <out dir>

settings : every Cycles / film / view setting the B4 blends carry that can change the radiometry of a reference.
clamp    : held-out T0 views 6 and 15 (their camera and sun), full GI, 256^2, 2048 spp, identical seed: the generator's
           clamp (direct 20, indirect = blend value) vs no clamp (0 / 0). Same seed -> the difference is the clamp bias.
super    : two-sun superposition with identical seeds, with and without clamp: rel L1 of both - (sun1 + sun2).
glass    : a 4 cm glass cube lit through one face, viewed through another: first-event setting (Batch 6 Stage A) vs
           'intrinsic-all' (all bounces, but every non-camera ray that reaches a surface from outside, i.e. after a free-
           space segment, is absorbed via Light Path / Backfacing masking) vs full GI; and the Lego glass parts.
"""

import json
import math
import sys
import time
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix, Vector

sys.path.insert(0, "/mnt/c/Projects/ReST-GI/external/relightable-neural-assets")
sys.path.insert(0, "/mnt/c/Projects/ReST-GI/experiments/fnto_lego")
from utils import exr  # noqa: E402

import fnto_common as C  # noqa: E402
from fnto_stagea_render import first_event, set_sun, strip_lights  # noqa: E402

RES = 256


def settings(path):
    bpy.ops.wm.open_mainfile(filepath=str(path)); sc = bpy.context.scene; c = sc.cycles; r = sc.render; v = sc.view_settings
    keys = ["samples", "sample_clamp_direct", "sample_clamp_indirect", "max_bounces", "diffuse_bounces", "glossy_bounces",
            "transmission_bounces", "volume_bounces", "transparent_max_bounces", "caustics_reflective", "caustics_refractive",
            "blur_glossy", "pixel_filter_type", "filter_width", "film_exposure", "light_sampling_threshold", "use_light_tree",
            "use_denoising", "seed", "sample_offset", "scrambling_distance", "min_light_bounces", "min_transparent_bounces"]
    out = {k: getattr(c, k) for k in keys if hasattr(c, k)}
    out.update({"view_transform": v.view_transform, "look": v.look, "exposure": v.exposure, "gamma": v.gamma,
                "display_device": sc.display_settings.display_device, "film_transparent": r.film_transparent,
                "file_format": r.image_settings.file_format, "color_depth": r.image_settings.color_depth})
    return {k: (float(x) if isinstance(x, float) else x) for k, x in out.items()}


def setup(blend, spp, clamp, res=RES):
    bpy.ops.wm.open_mainfile(filepath=str(blend)); sc = bpy.context.scene; sc.frame_current = 1; strip_lights(sc)
    r = sc.render; r.engine = "CYCLES"; r.resolution_x = res; r.resolution_y = res; r.resolution_percentage = 100
    r.image_settings.file_format = "OPEN_EXR"; r.image_settings.color_depth = "32"; r.image_settings.color_mode = "RGBA"
    r.film_transparent = True; r.filepath = "temp_chk"; sc.use_nodes = False
    c = sc.cycles; c.samples = spp; c.use_adaptive_sampling = False; c.use_denoising = False; c.device = "CPU"
    c.use_light_tree = False; sc.world.cycles.query_filename = ""; c.seed = 7; c.max_bounces = 32
    if clamp == "generator":
        c.sample_clamp_direct = 20                                     # sample_clamp_indirect: the blend's own value
    else:
        c.sample_clamp_direct = 0.0; c.sample_clamp_indirect = 0.0
    cd = bpy.data.cameras.new("chk_cam"); cd.type = "PERSP"; cd.lens_unit = "FOV"; cd.angle = math.radians(45)
    cam = bpy.data.objects.new("chk_cam", cd); sc.collection.objects.link(cam); sc.camera = cam
    suns = []
    for k in range(2):
        d = bpy.data.lights.new(f"chk_sun{k}", "SUN"); d.energy = C.E_SUN; d.angle = 0.0
        o = bpy.data.objects.new(f"chk_sun{k}", d); sc.collection.objects.link(o); suns.append(o)
    return sc, cam, suns


def render():
    bpy.ops.render.render(write_still=True)
    x = exr.read("temp_chk.exr"); return x.reshape(-1, x.shape[-1])


def ldr(x):
    x = np.clip(x, 0, None); return (x / (1 + x)) ** (1 / 2.2)


def clamp_and_super(out):
    import h5py
    f = h5py.File(C.B4 / "data/heldout_T0_s1_val.h5", "r"); cams = json.loads((C.B4 / "data/heldout_T0_s1_val.json").read_text())["cameras"]
    rec = {}
    for v in (6, 15):
        L = np.nan_to_num(f["light_dir"][v].astype(np.float32)); mk = np.nan_to_num(f["alpha"][v, :, 0].astype(np.float32)) > 0
        w = L[mk].mean(0); w /= np.linalg.norm(w)
        imgs = {}
        for clamp in ("generator", "none"):
            sc, cam, suns = setup(C.B4 / "scene/lego_b4_T0_s1.blend", 2048, clamp)
            loc, rot, sca = Matrix(cams[v]["mtx"]).decompose(); cam.location = loc; cam.rotation_euler = rot.to_euler(); cam.scale = sca
            set_sun(suns[0], w); suns[1].hide_render = True; bpy.context.view_layer.update()
            t = time.time(); imgs[clamp] = render(); rec.setdefault(f"view{v}", {})[f"seconds_{clamp}"] = time.time() - t
            rec[f"view{v}"]["sample_clamp_indirect_used"] = float(sc.cycles.sample_clamp_indirect)
        a, b = imgs["generator"][:, :3], imgs["none"][:, :3]; m = imgs["none"][:, 3] > 0
        rec[f"view{v}"].update({"energy_ratio_clamped_over_unclamped": float(a[m].sum() / b[m].sum()),
                                "display_psnr_clamped_vs_unclamped": float(10 * np.log10(1 / ((ldr(a[m]) - ldr(b[m])) ** 2).mean())),
                                "pixels_changed_gt_1pct": float((np.abs(a[m] - b[m]).sum(1) > 0.01 * np.maximum(b[m].sum(1), 1e-3)).mean())})
    # two-sun superposition, identical seeds, first view's camera, sun 2 from the opposite azimuth
    v = 6; L = np.nan_to_num(f["light_dir"][v].astype(np.float32)); mk = np.nan_to_num(f["alpha"][v, :, 0].astype(np.float32)) > 0
    w1 = L[mk].mean(0); w1 /= np.linalg.norm(w1); w2 = np.array([-w1[0], -w1[1], w1[2]])
    for clamp in ("generator", "none"):
        sc, cam, suns = setup(C.B4 / "scene/lego_b4_T0_s1.blend", 1024, clamp)
        loc, rot, sca = Matrix(cams[v]["mtx"]).decompose(); cam.location = loc; cam.rotation_euler = rot.to_euler(); cam.scale = sca
        out_ = {}
        for name, (a1, a2) in {"s1": (True, False), "s2": (False, True), "both": (True, True)}.items():
            set_sun(suns[0], w1); set_sun(suns[1], w2); suns[0].hide_render = not a1; suns[1].hide_render = not a2
            bpy.context.view_layer.update(); out_[name] = render()[:, :3]
        d = out_["both"] - out_["s1"] - out_["s2"]
        rec[f"superposition_{clamp}"] = {"rel_L1": float(np.abs(d).sum() / np.abs(out_["both"]).sum()),
                                         "rel_signed_energy": float(d.sum() / out_["both"].sum())}
    return rec


def mask_intrinsic(sc):
    """Absorb every non-camera ray that reaches a surface from OUTSIDE (front face): only camera hits, BSSRDF events and
    paths that stay inside a dielectric (back-face hits on Glass / Refraction materials) survive -> intrinsic-all with
    unlimited internal bounces. Opaque and SSS materials pass camera rays only (open shells hit from behind via free space
    must not count as 'inside')."""
    for m in bpy.data.materials:
        if not m.use_nodes or m.name.startswith("chk_"):
            continue
        nt = m.node_tree; outn = [n for n in nt.nodes if n.bl_idname == "ShaderNodeOutputMaterial"]
        if not outn or not outn[0].inputs["Surface"].is_linked:
            continue
        o = outn[0]; src = o.inputs["Surface"].links[0].from_socket
        lp = nt.nodes.new("ShaderNodeLightPath"); geo = nt.nodes.new("ShaderNodeNewGeometry")
        mx = nt.nodes.new("ShaderNodeMath"); mx.operation = "MAXIMUM"
        nt.links.new(lp.outputs["Is Camera Ray"], mx.inputs[0])
        dielectric = any(n.bl_idname in ("ShaderNodeBsdfGlass", "ShaderNodeBsdfRefraction") for n in nt.nodes)
        if dielectric:                                  # inside a dielectric solid: a back-face hit continues the path
            nt.links.new(geo.outputs["Backfacing"], mx.inputs[1])
        else:
            mx.inputs[1].default_value = 0.0
        blk = nt.nodes.new("ShaderNodeEmission"); blk.inputs["Strength"].default_value = 0.0
        mix = nt.nodes.new("ShaderNodeMixShader")
        nt.links.new(mx.outputs[0], mix.inputs["Fac"]); nt.links.new(blk.outputs[0], mix.inputs[1]); nt.links.new(src, mix.inputs[2])
        nt.links.new(mix.outputs[0], o.inputs["Surface"])


def glass(out):
    rec = {}
    for case in ("cube", "lego"):
        res_ = {}
        for mode in ("first", "intrinsic_all", "full"):
            if case == "cube":
                bpy.ops.wm.open_mainfile(filepath=str(C.B4 / "scene/lego_b4_T0_s1.blend")); sc = bpy.context.scene; strip_lights(sc)
                for o in list(sc.objects):
                    if o.type == "MESH":
                        bpy.data.objects.remove(o, do_unlink=True)
                bpy.ops.mesh.primitive_cube_add(size=0.04, location=(0, 0, 0.02)); g = bpy.context.object; g.data.materials.append(bpy.data.materials["transparent"])
                bpy.ops.mesh.primitive_plane_add(size=1.0, location=(0, 0, -0.001)); pl = bpy.context.object; pl.data.materials.append(bpy.data.materials["Gray"])
                cam_pos, sun_dir = (0.12, -0.12, 0.10), (-0.3, 0.6, 0.74)
            else:
                bpy.ops.wm.open_mainfile(filepath=str(C.B4 / "scene/lego_b4_T0_s1.blend")); sc = bpy.context.scene; strip_lights(sc)
                bpy.data.objects.remove(bpy.data.objects["B_red_box"], do_unlink=True)
                cam_pos, sun_dir = (-1.9, -2.0, 1.1), (0.45, -0.55, 0.70)
            r = sc.render; r.engine = "CYCLES"; r.resolution_x = RES; r.resolution_y = RES; r.film_transparent = True
            r.image_settings.file_format = "OPEN_EXR"; r.image_settings.color_depth = "32"; r.image_settings.color_mode = "RGBA"; r.filepath = "temp_chk"; sc.use_nodes = False
            c = sc.cycles; c.samples = 1024; c.use_adaptive_sampling = False; c.use_denoising = False; c.device = "CPU"; c.use_light_tree = False
            sc.world.cycles.query_filename = ""; c.sample_clamp_direct = 0; c.sample_clamp_indirect = 0; c.seed = 3
            if mode == "first":
                first_event(c)
            else:
                c.max_bounces = 128; c.diffuse_bounces = 128; c.glossy_bounces = 128; c.transmission_bounces = 128; c.transparent_max_bounces = 128
                if mode == "intrinsic_all":
                    mask_intrinsic(sc)
            d = bpy.data.lights.new("chk_s", "SUN"); d.energy = C.E_SUN; d.angle = 0.0; s = bpy.data.objects.new("chk_s", d); sc.collection.objects.link(s)
            set_sun(s, sun_dir)
            cd = bpy.data.cameras.new("chk_cam"); cd.lens_unit = "FOV"; cd.angle = math.radians(30 if case == "cube" else 45)
            cam = bpy.data.objects.new("chk_cam", cd); sc.collection.objects.link(cam); sc.camera = cam
            dv = (Vector((0, 0, 0.02)) - Vector(cam_pos)).normalized(); cam.location = cam_pos; cam.rotation_mode = "QUATERNION"; cam.rotation_quaternion = dv.to_track_quat("-Z", "Y")
            bpy.context.view_layer.update()
            x = render(); res_[mode] = x
        if case == "cube":
            h = RES // 2; w_ = 12
            reg = lambda x: x.reshape(RES, RES, -1)[h - w_:h + w_, h - w_:h + w_, :3].mean((0, 1))  # noqa: E731
            rec["cube_centre"] = {m: reg(x).round(5).tolist() for m, x in res_.items()}
        else:
            rec["lego_total"] = {m: x[:, :3].sum(0).round(2).tolist() for m, x in res_.items()}
    return rec


def main():
    a = sys.argv[sys.argv.index("--") + 1:]; out = Path(a[0]); out.mkdir(parents=True, exist_ok=True)
    rec = {"settings": {n: settings(C.B4 / f"scene/{n}") for n in ("lego_b4_T0_s0.blend", "lego_b4_T0_s1.blend", "lego_b4_T1_s1.blend")}}
    (out / "render_checks.json").write_text(json.dumps(rec, indent=1))
    rec["glass"] = glass(out); (out / "render_checks.json").write_text(json.dumps(rec, indent=1))
    rec["clamp"] = clamp_and_super(out); (out / "render_checks.json").write_text(json.dumps(rec, indent=1))
    print(json.dumps(rec, indent=1))


if __name__ == "__main__":
    main()
