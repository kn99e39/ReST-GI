"""WL46 KF regime 'emit46': Blender oracle of free-space transfer of a known DIRECTIONAL outgoing field.

As worklog 45's emit45 'emitall' (every Lego / box surface, Yellow included, emits for non-camera rays; non-Yellow camera
rays see diffuse white + emission ('emit') or emission only ('emitonly'); Yellow camera rays keep the real material),
but the emitted radiance depends on the outgoing direction:
    L_e(x, w) = f_s(x) g_kappa(|n_true(x) . w|),   g_kappa(c) = (kappa + 2) / 2 c^kappa
(w = Cycles' Geometry 'Incoming' = toward the ray origin; True Normal, no bump; |.| = the side facing the ray, matching
the relation table's facing-side port). The cosine-weighted flux of g_kappa equals the isotropic field's for every
kappa. f_s = emit45's smooth analytic field. E_in(pixel) = pi (P_emit - P_emitonly).

    RNA_DIR=<rna repo> python emit46_render.py -- <emit|emitonly> <view> <spp> <kappa> <out dir>   (env D46_SEED_OFFSET)
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "fnto_diag45/track_b")); sys.path.insert(0, str(HERE.parent / "fnto_lego_r1")); sys.path.insert(0, str(HERE.parent / "fnto_lego"))
sys.path.insert(0, os.environ.get("RNA_DIR", str(HERE.parents[1] / "external/relightable-neural-assets")))
import bpy  # noqa: E402
from mathutils import Matrix  # noqa: E402
from utils import exr  # noqa: E402

import r1_common as C  # noqa: E402
from r1_stagea_render import strip_lights  # noqa: E402
from emit45_render import field_nodes  # noqa: E402


def directional(nt, kappa):
    geo = nt.nodes.new("ShaderNodeNewGeometry"); dot = nt.nodes.new("ShaderNodeVectorMath"); dot.operation = "DOT_PRODUCT"
    nt.links.new(geo.outputs["True Normal"], dot.inputs[0]); nt.links.new(geo.outputs["Incoming"], dot.inputs[1])
    ab = nt.nodes.new("ShaderNodeMath"); ab.operation = "ABSOLUTE"; nt.links.new(dot.outputs["Value"], ab.inputs[0])
    pw = nt.nodes.new("ShaderNodeMath"); pw.operation = "POWER"; nt.links.new(ab.outputs[0], pw.inputs[0]); pw.inputs[1].default_value = float(kappa)
    sc = nt.nodes.new("ShaderNodeMath"); sc.operation = "MULTIPLY"; nt.links.new(pw.outputs[0], sc.inputs[0]); sc.inputs[1].default_value = (kappa + 2) / 2.0
    fs = field_nodes(nt); mul = nt.nodes.new("ShaderNodeMath"); mul.operation = "MULTIPLY"
    nt.links.new(fs, mul.inputs[0]); nt.links.new(sc.outputs[0], mul.inputs[1])
    return mul.outputs[0]


def rewrite(mode, kappa):
    for mat in bpy.data.materials:
        if not mat.use_nodes:
            continue
        nt = mat.node_tree; outn = [n for n in nt.nodes if n.bl_idname == "ShaderNodeOutputMaterial"]
        if not outn or not outn[0].inputs["Surface"].is_linked:
            continue
        o = outn[0]; src = o.inputs["Surface"].links[0].from_socket; yellow = mat.name == "Yellow"
        lp = nt.nodes.new("ShaderNodeLightPath"); em = nt.nodes.new("ShaderNodeEmission"); em.inputs["Color"].default_value = (1, 1, 1, 1)
        nt.links.new(directional(nt, kappa), em.inputs["Strength"])
        if yellow:
            cam = src
        elif mode == "emitonly":
            cam = em.outputs[0]
        else:
            df = nt.nodes.new("ShaderNodeBsdfDiffuse"); df.inputs["Color"].default_value = (1, 1, 1, 1); df.inputs["Roughness"].default_value = 0.0
            add = nt.nodes.new("ShaderNodeAddShader"); nt.links.new(df.outputs[0], add.inputs[0]); nt.links.new(em.outputs[0], add.inputs[1]); cam = add.outputs[0]
        mix = nt.nodes.new("ShaderNodeMixShader")
        nt.links.new(lp.outputs["Is Camera Ray"], mix.inputs["Fac"]); nt.links.new(em.outputs[0], mix.inputs[1]); nt.links.new(cam, mix.inputs[2])
        nt.links.new(mix.outputs[0], o.inputs["Surface"])
        try:
            mat.cycles.emission_sampling = "FRONT_BACK"
        except Exception:
            pass


def main():
    a = sys.argv[sys.argv.index("--") + 1:]
    mode, v, spp, kappa, out = a[0], int(a[1]), int(a[2]), int(a[3]), Path(a[4]); out.mkdir(parents=True, exist_ok=True)
    views = json.loads((Path.home() / "rest_fnto_r1/views_r1.json").read_text())
    bpy.ops.wm.open_mainfile(filepath=str(C.B4 / "scene/lego_b4_T0_s1.blend"))
    sc = bpy.context.scene; sc.frame_current = 1; strip_lights(sc); sc.use_nodes = False
    rewrite(mode, kappa)
    view = views["heldout_T0"][v]; mtx = Matrix(view["mtx"])
    cd = bpy.data.cameras.new("d46_cam"); cd.type = "PERSP"; cd.lens_unit = "FOV"; cd.angle = 0.785398
    cam = bpy.data.objects.new("d46_cam", cd); sc.collection.objects.link(cam); sc.camera = cam
    loc, rot, sca = mtx.decompose(); cam.location = loc; cam.rotation_euler = rot.to_euler(); cam.scale = sca
    r = sc.render; res = views["res"]; r.engine = "CYCLES"; r.resolution_x = res; r.resolution_y = res; r.resolution_percentage = 100
    r.image_settings.file_format = "OPEN_EXR"; r.image_settings.color_depth = "32"; r.image_settings.color_mode = "RGBA"
    r.film_transparent = True; r.filepath = f"temp_d46_{mode}_{v}_{kappa}"
    c = sc.cycles; c.samples = spp; c.use_adaptive_sampling = False; c.use_denoising = False; c.device = "CPU"
    c.sample_clamp_direct = 0.0; c.sample_clamp_indirect = 0.0; c.pixel_filter_type = "GAUSSIAN"; c.filter_width = 0.01
    c.use_light_tree = False; sc.world.cycles.query_filename = ""
    c.max_bounces = 8; c.diffuse_bounces = 8; c.glossy_bounces = 8; c.transmission_bounces = 8; c.transparent_max_bounces = 8
    c.seed = 4600 + v + int(os.environ.get("D46_SEED_OFFSET", "0")); bpy.context.view_layer.update()
    bpy.ops.render.render(write_still=True)
    img = exr.read(r.filepath + ".exr").reshape(-1, 4); mk = np.asarray(view["mk"], np.int64)
    np.savez(out / f"{mode}_k{kappa}_v{v}.npz", rgb=img[mk, :3].astype(np.float32), spp=spp, seed=c.seed, kappa=kappa)
    print("done", mode, v, kappa, flush=True); sys.stdout.flush(); os._exit(0)


if __name__ == "__main__":
    main()
