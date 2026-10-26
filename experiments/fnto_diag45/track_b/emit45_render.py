"""WL45 Track B - regime 'emit45': Blender oracle of single-flight free-space transfer of a known outgoing field.

Scene: the held-out T0 blend (Lego + box, r1 film: GAUSSIAN 0.01, no clamp, CPU Cycles), no lights, black world.
Every material is rewritten (displacement / bump and AOV links untouched):
  non-camera rays : non-Yellow surfaces = Emission(white, strength f(x)); Yellow = Emission(0) (absorbing, not emitting)
  camera rays     : non-Yellow = Diffuse(white) + Emission f(x)   ('emit' mode)   or  Emission f(x) only ('emitonly')
                    Yellow     = its original surface shader (SSS + 2 % glossy)
so a camera pixel sees its own emission plus exactly ONE free flight from emitting surfaces (emitters have no BSDF for
non-camera rays, so no second flight). E_in(pixel) = pi (P_emit_mode - P_emitonly_mode) for non-Yellow pixels; Yellow
pixels give the real Yellow material's response to that incident field. The oracle is Blender's own BVH on the true
geometry; it does not use C's relation tables, ports or projection.
f(x) = max(0.05, 1 + 0.6 sin(2 pi x / 0.4) sin(2 pi y / 0.4) + 0.3 sin(2 pi z / 0.25))   (world coordinates, metres)

'emitall' (WL45 amendment, control for Yellow->Yellow exposed paths): as 'emit', but Yellow also emits f for non-camera
rays; its camera rays keep the real material (Cycles' Burley exit closure is built from the entry closure, the exit
point's shader graph is not re-evaluated, so the Yellow pixel stays the real SSS response).

env D45_FIELD=bump (amendment 4): f = 10 exp(-|x - c|^2 / (2 (2 cm)^2)), c = BUMP_C, a localized field.

    RNA_DIR=<rna repo> python emit45_render.py -- <mode emit|emitonly|emitall|calib> <view> <spp> <out dir>
"""
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "fnto_lego_r1")); sys.path.insert(0, str(HERE.parents[1] / "fnto_lego"))
sys.path.insert(0, os.environ.get("RNA_DIR", str(HERE.parents[2] / "external/relightable-neural-assets")))
import bpy  # noqa: E402
from mathutils import Matrix  # noqa: E402
from utils import exr  # noqa: E402

import r1_common as C  # noqa: E402
from r1_stagea_render import strip_lights  # noqa: E402


def field_numpy(p):
    if os.environ.get("D45_FIELD", "smooth") == "bump":
        return BUMP_AMP * np.exp(-((p - np.asarray(BUMP_C)) ** 2).sum(-1) / (2 * BUMP_SIGMA ** 2))
    x, y, z = p[..., 0], p[..., 1], p[..., 2]
    return np.maximum(0.05, 1 + 0.6 * np.sin(2 * np.pi * x / 0.4) * np.sin(2 * np.pi * y / 0.4) + 0.3 * np.sin(2 * np.pi * z / 0.25))


BUMP_C, BUMP_SIGMA, BUMP_AMP = (0.26708984, -0.45532227, 0.54345703), 0.02, 10.0   # WL45 amendment 4: localized field


def field_nodes(nt):
    if os.environ.get("D45_FIELD", "smooth") == "bump":                 # f = AMP exp(-|x - c|^2 / (2 sigma^2))
        geo = nt.nodes.new("ShaderNodeNewGeometry"); sub = nt.nodes.new("ShaderNodeVectorMath"); sub.operation = "SUBTRACT"
        nt.links.new(geo.outputs["Position"], sub.inputs[0]); sub.inputs[1].default_value = BUMP_C
        dot = nt.nodes.new("ShaderNodeVectorMath"); dot.operation = "DOT_PRODUCT"; nt.links.new(sub.outputs[0], dot.inputs[0]); nt.links.new(sub.outputs[0], dot.inputs[1])
        mul = nt.nodes.new("ShaderNodeMath"); mul.operation = "MULTIPLY"; nt.links.new(dot.outputs["Value"], mul.inputs[0]); mul.inputs[1].default_value = -1.0 / (2 * BUMP_SIGMA ** 2)
        ex = nt.nodes.new("ShaderNodeMath"); ex.operation = "EXPONENT"; nt.links.new(mul.outputs[0], ex.inputs[0])
        amp = nt.nodes.new("ShaderNodeMath"); amp.operation = "MULTIPLY"; nt.links.new(ex.outputs[0], amp.inputs[0]); amp.inputs[1].default_value = BUMP_AMP
        return amp.outputs[0]
    geo = nt.nodes.new("ShaderNodeNewGeometry"); sep = nt.nodes.new("ShaderNodeSeparateXYZ"); nt.links.new(geo.outputs["Position"], sep.inputs[0])

    def m(op, a, b=None):
        n = nt.nodes.new("ShaderNodeMath"); n.operation = op
        for i, v in enumerate((a, b)):
            if v is None:
                continue
            if isinstance(v, (int, float)):
                n.inputs[i].default_value = float(v)
            else:
                nt.links.new(v, n.inputs[i])
        return n.outputs[0]
    sx = m("SINE", m("MULTIPLY", sep.outputs["X"], 2 * math.pi / 0.4)); sy = m("SINE", m("MULTIPLY", sep.outputs["Y"], 2 * math.pi / 0.4))
    sz = m("SINE", m("MULTIPLY", sep.outputs["Z"], 2 * math.pi / 0.25))
    f = m("ADD", m("ADD", 1.0, m("MULTIPLY", m("MULTIPLY", sx, sy), 0.6)), m("MULTIPLY", sz, 0.3))
    return m("MAXIMUM", f, 0.05)


def rewrite(mode):
    for mat in bpy.data.materials:
        if not mat.use_nodes:
            continue
        nt = mat.node_tree; outn = [n for n in nt.nodes if n.bl_idname == "ShaderNodeOutputMaterial"]
        if not outn or not outn[0].inputs["Surface"].is_linked:
            continue
        o = outn[0]; src = o.inputs["Surface"].links[0].from_socket
        yellow = mat.name == "Yellow"
        lp = nt.nodes.new("ShaderNodeLightPath")
        em = nt.nodes.new("ShaderNodeEmission"); em.inputs["Color"].default_value = (1, 1, 1, 1)
        if yellow and mode == "emitall":                       # control: Yellow also emits f for non-camera rays
            nt.links.new(field_nodes(nt), em.inputs["Strength"]); cam = src
        elif yellow:
            em.inputs["Strength"].default_value = 0.0; cam = src
        else:
            nt.links.new(field_nodes(nt), em.inputs["Strength"])
            if mode == "emitonly":
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
    mode, v, spp, out = a[0], int(a[1]), int(a[2]), Path(a[3]); out.mkdir(parents=True, exist_ok=True)
    views = json.loads((Path.home() / "rest_fnto_r1/views_r1.json").read_text())
    bpy.ops.wm.open_mainfile(filepath=str(C.B4 / "scene/lego_b4_T0_s1.blend"))
    sc = bpy.context.scene; sc.frame_current = 1; strip_lights(sc); sc.use_nodes = False
    if mode == "calib":                                         # one emissive plane of strength 1 seen head-on
        for ob in list(sc.objects):
            ob.hide_render = True
        bpy.ops.mesh.primitive_plane_add(size=4.0, location=(0, 0, 0)); pl = bpy.context.object
        mt = bpy.data.materials.new("calib"); mt.use_nodes = True; nt = mt.node_tree; nt.nodes.clear()
        em = nt.nodes.new("ShaderNodeEmission"); em.inputs["Strength"].default_value = 1.0; o = nt.nodes.new("ShaderNodeOutputMaterial")
        nt.links.new(em.outputs[0], o.inputs["Surface"]); pl.data.materials.append(mt)
        mtx = Matrix(((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 2.0), (0, 0, 0, 1)))
    else:
        rewrite(mode); mtx = Matrix(views["heldout_T0"][v]["mtx"])
    cd = bpy.data.cameras.new("d45_cam"); cd.type = "PERSP"; cd.lens_unit = "FOV"; cd.angle = math.radians(45)
    cam = bpy.data.objects.new("d45_cam", cd); sc.collection.objects.link(cam); sc.camera = cam
    loc, rot, sca = mtx.decompose(); cam.location = loc; cam.rotation_euler = rot.to_euler(); cam.scale = sca
    r = sc.render; res = views["res"] if mode != "calib" else 64
    r.engine = "CYCLES"; r.resolution_x = res; r.resolution_y = res; r.resolution_percentage = 100
    r.image_settings.file_format = "OPEN_EXR"; r.image_settings.color_depth = "32"; r.image_settings.color_mode = "RGBA"
    r.film_transparent = True; r.filepath = f"temp_d45_{mode}_{v}"
    c = sc.cycles; c.samples = spp; c.use_adaptive_sampling = False; c.use_denoising = False; c.device = "CPU"
    c.sample_clamp_direct = 0.0; c.sample_clamp_indirect = 0.0; c.pixel_filter_type = "GAUSSIAN"; c.filter_width = 0.01
    c.use_light_tree = False; sc.world.cycles.query_filename = ""
    c.max_bounces = 8; c.diffuse_bounces = 8; c.glossy_bounces = 8; c.transmission_bounces = 8; c.transparent_max_bounces = 8
    c.seed = 4500 + v + int(os.environ.get("D45_SEED_OFFSET", "0")); bpy.context.view_layer.update()   # offset: oracle-noise replicate
    bpy.ops.render.render(write_still=True)
    img = exr.read(r.filepath + ".exr").reshape(-1, 4)
    if mode == "calib":
        rec = {"mode": mode, "pixel_radiance_of_strength_1_emitter": float(np.median(img[:, :3]))}
        (out / "calib.json").write_text(json.dumps(rec)); print(rec)
    else:
        mk = np.asarray(views["heldout_T0"][v]["mk"], np.int64)
        np.savez(out / f"{mode}_v{v}.npz", rgb=img[mk, :3].astype(np.float32), alpha=img[mk, 3].astype(np.float16), spp=spp, seed=c.seed)
        print("done", mode, v, flush=True)
    sys.stdout.flush(); os._exit(0)


if __name__ == "__main__":
    main()
