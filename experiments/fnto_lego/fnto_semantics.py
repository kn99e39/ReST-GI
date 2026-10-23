"""FNTO Batch 6 (worklog 41): reference-renderer semantics study (WSL RNA venv bpy 3.5.1, the custom Cycles build).

    python fnto_semantics.py <out dir> [--gpu]

Establishes, before any FNTO training, how the reference renderer (Blender 3.5.1 Cycles, the RNA-generator settings)
associates incident and outgoing scattering events with geometry, as required by the Batch 6 brief §5:

  bounce : which Cycles light-path limits isolate the *first intrinsic event* (one surface interface + material interior,
           no editable free-space segment after the light-to-surface segment) on the isolated official Lego.
  burley : whether Christensen-Burley SSS transport crosses (a) disconnected components of one mesh object, (b) separate
           objects, (c) material boundaries inside one object, (d) an occluder that blocks only shadow rays.
  mask   : whether a camera-invisible, shadow-only occluder changes nothing but direct-light visibility.

Every render: the RNA generator's light (sun, energy 5, angle 0, world Background removed, scene lights hidden),
sample_clamp_direct 20, no denoising, film transparent. No reference image is changed or overwritten.
"""

import json
import math
import sys
import time
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix, Vector

ROOT = Path("/mnt/c/Projects/ReST-GI")
BLEND = ROOT / "results/rna_selective_reconstruction_lego/scene/lego_b4_T0_s0.blend"
RES = 256


def load_exr(path):
    img = bpy.data.images.load(str(path), check_existing=False)
    w, h = img.size
    px = np.array(img.pixels[:], np.float32).reshape(h, w, img.channels)[::-1]
    bpy.data.images.remove(img)
    return px


def setup_render(sc, spp, gpu, out):
    r = sc.render
    r.engine = "CYCLES"; r.resolution_x = RES; r.resolution_y = RES; r.resolution_percentage = 100
    r.image_settings.file_format = "OPEN_EXR"; r.image_settings.color_depth = "32"; r.image_settings.color_mode = "RGBA"
    r.film_transparent = True; r.filepath = str(out)
    c = sc.cycles
    c.samples = spp; c.use_adaptive_sampling = False; c.use_denoising = False; c.sample_clamp_direct = 20
    c.use_light_tree = False
    sc.world.cycles.query_filename = ""
    sc.use_nodes = False                      # no compositor file outputs in this study
    if gpu:
        p = bpy.context.preferences.addons["cycles"].preferences
        p.compute_device_type = "CUDA"; p.get_devices()
        for d in p.devices:
            d.use = d.type == "CUDA"
        c.device = "GPU"
    else:
        c.device = "CPU"


def bounces(sc, mode):
    c = sc.cycles
    if mode == "full":                        # the generator's override (max 32) on top of the blend's per-type limits
        c.max_bounces = 32
    elif mode == "b0":
        c.max_bounces = 0
    elif mode == "first":                     # first intrinsic event: no diffuse/glossy continuation, one glass traversal
        c.max_bounces = 1; c.diffuse_bounces = 0; c.glossy_bounces = 0; c.transmission_bounces = 1
        c.volume_bounces = 0; c.transparent_max_bounces = 0
    elif mode == "first2":
        c.max_bounces = 2; c.diffuse_bounces = 0; c.glossy_bounces = 0; c.transmission_bounces = 2
        c.volume_bounces = 0; c.transparent_max_bounces = 0
    else:
        raise ValueError(mode)


def add_sun(direction, energy=5.0):
    d = Vector(direction).normalized()
    data = bpy.data.lights.new("FNTO_sun", "SUN"); data.energy = energy; data.angle = 0.0
    o = bpy.data.objects.new("FNTO_sun", data); bpy.context.scene.collection.objects.link(o)
    o.rotation_mode = "QUATERNION"; o.rotation_quaternion = Vector((0, 0, 1)).rotation_difference(d)   # +Z -> toward light
    return o


def add_camera(pos, target, fov=45):
    cd = bpy.data.cameras.new("FNTO_cam"); cd.lens_unit = "FOV"; cd.angle = math.radians(fov)
    o = bpy.data.objects.new("FNTO_cam", cd); bpy.context.scene.collection.objects.link(o)
    d = (Vector(target) - Vector(pos)).normalized()
    o.location = pos; o.rotation_mode = "QUATERNION"; o.rotation_quaternion = d.to_track_quat("-Z", "Y")
    bpy.context.scene.camera = o
    return o


def strip_lights(sc):
    bg = sc.world.node_tree.nodes.get("Background")
    if bg is not None:
        sc.world.node_tree.nodes.remove(bg)
    for o in sc.objects:
        if o.type in ("LIGHT", "CAMERA"):
            o.hide_render = True


def shadow_only(o):
    o.visible_camera = False; o.visible_diffuse = False; o.visible_glossy = False
    o.visible_transmission = False; o.visible_volume_scatter = False; o.visible_shadow = True


def render(sc, path):
    t = time.time(); sc.render.filepath = str(path); bpy.ops.render.render(write_still=True)
    return load_exr(str(path) + ".exr" if not str(path).endswith(".exr") else path), time.time() - t


def bounce_study(out, gpu, spp):
    bpy.ops.wm.open_mainfile(filepath=str(BLEND))
    sc = bpy.context.scene
    rec = {"blend_per_type": {k: getattr(sc.cycles, k) for k in ("max_bounces", "diffuse_bounces", "glossy_bounces", "transmission_bounces",
                                                                "volume_bounces", "transparent_max_bounces", "caustics_reflective",
                                                                "caustics_refractive", "blur_glossy")}}
    bpy.data.objects.remove(bpy.data.objects["B_red_box"], do_unlink=True)       # isolated Lego domain
    strip_lights(sc)
    add_sun((0.45, -0.55, 0.70)); add_camera((-1.9, -2.0, 1.1), (0, 0, 0.1))
    setup_render(sc, spp, gpu, out / "x")
    lego = bpy.data.objects["Lego"]
    mats = [s.material.name for s in lego.material_slots]
    rec["materials"] = mats
    # material index buffer: render a flat emission-coded pass via object/material index AOV is not in the blend; instead
    # per-material masks come from replacing each material by a black holdout one at a time is too slow -> use a
    # 'material id' render: every material emission = its index code, no lights, 1 spp, no bounces.
    imgs = {}
    for mode in ("full", "b0", "first", "first2"):
        bounces(sc, mode)
        img, sec = render(sc, out / f"bounce_{mode}")
        imgs[mode] = img; rec[f"seconds_{mode}"] = sec
    # material-id pass
    saved = {}
    for i, s in enumerate(lego.material_slots):
        m = s.material; saved[i] = m
        e = bpy.data.materials.new(f"id_{i}"); e.use_nodes = True; nt = e.node_tree; nt.nodes.clear()
        em = nt.nodes.new("ShaderNodeEmission"); em.inputs["Color"].default_value = (i + 1, 0, 0, 1); em.inputs["Strength"].default_value = 1.0
        o_ = nt.nodes.new("ShaderNodeOutputMaterial"); nt.links.new(em.outputs[0], o_.inputs["Surface"]); s.material = e
    sc.cycles.samples = 1; sc.cycles.filter_width = 0.01; bounces(sc, "b0")
    idimg, _ = render(sc, out / "bounce_matid")
    mid = np.rint(idimg[..., 0]).astype(int) - 1
    for i, m in saved.items():
        lego.material_slots[i].material = m
    rows = {}
    for i, name in enumerate(mats):
        mk = (mid == i) & (idimg[..., 3] > 0.5)
        if mk.sum() < 20:
            continue
        rows[name] = {"px": int(mk.sum())}
        for mode, img in imgs.items():
            rows[name][mode] = img[..., :3][mk].mean(0).round(5).tolist()
    rec["per_material_mean_rgb"] = rows
    np.savez_compressed(out / "bounce_study.npz", matid=mid.astype(np.int8), **{k: v.astype(np.float16) for k, v in imgs.items()})
    return rec


def slab(name, x0, mat, size=(0.3, 0.3, 0.02)):
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(x0, 0, size[2] / 2))
    o = bpy.context.object; o.name = name
    o.data.transform(Matrix.Diagonal((*size, 1.0)))
    o.data.materials.append(mat)
    return o


def burley_study(out, gpu, spp):
    """Synthetic: two 30 x 30 x 2 cm slabs, 2 cm gap; the sun lights slab 1 only (shadow-only blocker over slab 2)."""
    rec = {}
    cases = {
        "one_object_both_yellow": ("join", "Yellow", "Yellow"),
        "two_objects_both_yellow": ("separate", "Yellow", "Yellow"),
        "one_object_lit_gray_dark_yellow": ("join", "Gray", "Yellow"),
        "one_object_lit_yellow_dark_gray": ("join", "Yellow", "Gray"),
    }
    for case, (how, m1, m2) in cases.items():
        bpy.ops.wm.open_mainfile(filepath=str(BLEND))
        sc = bpy.context.scene; strip_lights(sc)
        for o in list(sc.objects):
            if o.type == "MESH":
                bpy.data.objects.remove(o, do_unlink=True)
        m1, m2 = bpy.data.materials[m1], bpy.data.materials[m2]
        s1 = slab("S1", -0.16, m1); s2 = slab("S2", 0.16, m2)
        if how == "join":
            bpy.ops.object.select_all(action="DESELECT"); s1.select_set(True); s2.select_set(True)
            bpy.context.view_layer.objects.active = s1; bpy.ops.object.join()
        bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0.16, 0, 0.5))
        blk = bpy.context.object; blk.name = "blocker"; blk.data.transform(Matrix.Diagonal((0.34, 0.34, 0.02, 1.0)))
        shadow_only(blk)
        add_sun((0, 0, 1)); add_camera((0, 0, 1.2), (0, 0, 0), fov=40)
        setup_render(sc, spp, gpu, out / "x")
        row = {}
        for mode in ("b0", "first", "full"):
            bounces(sc, mode)
            img, sec = render(sc, out / f"burley_{case}_{mode}")
            h = img.shape[0]; xs = (np.arange(h) + 0.5) / h * 2 - 1   # image x in [-1, 1]
            half = np.tan(math.radians(20)) * 1.2                     # half-width of the view at z = 0 in metres
            X = xs[None, :] * half; Y = -xs[:, None] * half
            m_1 = (np.abs(X + 0.16) < 0.10) & (np.abs(Y) < 0.10)
            m_2 = (np.abs(X - 0.16) < 0.10) & (np.abs(Y) < 0.10)
            row[mode] = {"lit_slab_mean": img[..., :3][m_1].mean(0).round(5).tolist(),
                         "dark_slab_mean": img[..., :3][m_2].mean(0).round(5).tolist(), "s": sec}
        rec[case] = row
    return rec


def main():
    args = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    out = Path(args[0]); out.mkdir(parents=True, exist_ok=True)
    gpu = "--gpu" in args
    spp = 256
    rec = {"bpy": bpy.app.version_string, "gpu": gpu, "spp": spp, "res": RES}
    t = time.time(); rec["burley"] = burley_study(out, gpu, spp); rec["burley_s"] = time.time() - t
    (out / "semantics.json").write_text(json.dumps(rec, indent=1))
    t = time.time(); rec["bounce"] = bounce_study(out, gpu, spp); rec["bounce_s"] = time.time() - t
    (out / "semantics.json").write_text(json.dumps(rec, indent=1))
    print(json.dumps(rec, indent=1))


if __name__ == "__main__" and "--glass" not in sys.argv:
    main()


def glass_study(out, spp):
    """Does a bounce-limited render see *through* glass to another lit surface (a free-space C path = contamination),
    and does it keep sunlight transmitted through the glass solid (an intrinsic K path)?"""
    rec = {}
    for case in ("plane_behind", "sun_behind"):
        bpy.ops.wm.open_mainfile(filepath=str(BLEND))
        sc = bpy.context.scene; strip_lights(sc)
        for o in list(sc.objects):
            if o.type == "MESH":
                bpy.data.objects.remove(o, do_unlink=True)
        g = slab("G", 0.0, bpy.data.materials["transparent"], size=(0.2, 0.2, 0.02))
        g.location = (0, 0, 0.3)
        if case == "plane_behind":
            p = slab("P", 0.0, bpy.data.materials["Gray"], size=(1.2, 1.2, 0.02))
            p.location = (0, 0, -0.01)
            add_sun((0.55, 0.0, 0.835))                  # lights the plane; the glass shadow falls off-centre
        else:
            add_sun((0.0, 0.0, -1.0))                    # sun below: light enters the glass bottom face, exits the top
        add_camera((0, 0, 1.5), (0, 0, 0), fov=30)
        setup_render(sc, spp, False, out / "x")
        row = {}
        for mode in ("b0", "first", "first2", "full"):
            bounces(sc, mode)
            img, sec = render(sc, out / f"glass_{case}_{mode}")
            h = img.shape[0]; c = h // 2; w = int(h * 0.12)
            row[mode] = {"glass_centre_mean": img[c - w:c + w, c - w:c + w, :3].mean((0, 1)).round(6).tolist(),
                         "outside_mean": img[5:25, 5:25, :3].mean((0, 1)).round(6).tolist()}
        rec[case] = row
    return rec


if __name__ == "__main__" and "--glass" in sys.argv:
    o_ = Path(sys.argv[sys.argv.index("--") + 1])
    r_ = glass_study(o_, 256)
    (o_ / "semantics_glass.json").write_text(json.dumps(r_, indent=1)); print(json.dumps(r_, indent=1))
