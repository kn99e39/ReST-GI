"""WL46 MS (class II specular real-renderer scene): official Lego + red box (worklog-37 T0 / T1 box move) + a glossy plane.

Plane: x = 0.85, y in [-1.0, 0.9], z in [-0.25, 0.8], facing -x; Cycles Glossy BSDF (GGX, roughness 0.2 -> alpha 0.04,
colour 0.9, no Fresnel). One sun w = normalize(0.2, -0.5, 0.84) of strength E_SUN (angle 0); black world; the blend's
other lights stripped. The plane's front is not sun-lit (n . w < 0: no caustic) and its shadow does not reach the Lego or
the box. Cameras CAMS (FOV 45 deg, res 512) see the Lego, the box and the plane with the reflected Lego / box.

  ref  full GI, r1 settings (no clamp, GAUSSIAN 0.01 point sampling, max_bounces 32), CPU, seed 4610 + view (+ D46_SEED_OFFSET)
  aov  16 spp: Position, Normal, Object index (Lego 1, box 2, plane 3), Diffuse Direct, camera direction
Outputs <out>/<mode>_<state>_v<view>.npz at all pixels of the res x res image (alpha > 0 mask stored).

    RNA_DIR=<rna repo> python ms46_render.py -- <ref|aov> <T0|T1> <view> <spp> <out dir>
"""
import math
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "fnto_lego_r1")); sys.path.insert(0, str(HERE.parent / "fnto_lego"))
sys.path.insert(0, os.environ.get("RNA_DIR", str(HERE.parents[1] / "external/relightable-neural-assets")))
import bpy  # noqa: E402
import OpenEXR  # noqa: E402
import Imath  # noqa: E402
from mathutils import Vector  # noqa: E402

import r1_common as C  # noqa: E402
from r1_stagea_render import strip_lights  # noqa: E402

PLANE = {"x": 0.85, "y": (-1.0, 0.9), "z": (-0.25, 0.8), "roughness": 0.2, "colour": 0.9}
SUN = np.array([0.2, -0.5, 0.84]) / np.linalg.norm([0.2, -0.5, 0.84])
CAMS = [((-1.6, -1.9, 1.1), (0.7, -0.3, 0.2)), ((-0.9, 1.7, 0.7), (0.8, -0.5, 0.25)), ((-0.6, -2.4, 0.5), (0.8, 0.0, 0.25))]
RES = 512


def add_plane():
    x = PLANE["x"]; (y0, y1), (z0, z1) = PLANE["y"], PLANE["z"]
    me = bpy.data.meshes.new("d46_plane")
    me.from_pydata([(x, y0, z0), (x, y0, z1), (x, y1, z1), (x, y1, z0)], [], [(0, 1, 2, 3)])        # normal -x
    me.update(); ob = bpy.data.objects.new("d46_plane", me); bpy.context.scene.collection.objects.link(ob)
    mt = bpy.data.materials.new("d46_glossy"); mt.use_nodes = True; nt = mt.node_tree; nt.nodes.clear()
    gl = nt.nodes.new("ShaderNodeBsdfGlossy"); gl.distribution = "GGX"; gl.inputs["Roughness"].default_value = PLANE["roughness"]
    gl.inputs["Color"].default_value = (PLANE["colour"],) * 3 + (1.0,)
    o = nt.nodes.new("ShaderNodeOutputMaterial"); nt.links.new(gl.outputs[0], o.inputs["Surface"]); me.materials.append(mt)
    return ob


def read_exr(path):
    f = OpenEXR.InputFile(path); dw = f.header()["dataWindow"]; w, h = dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1
    pt = Imath.PixelType(Imath.PixelType.FLOAT)
    return {ch: np.frombuffer(f.channel(ch, pt), np.float32).reshape(h * w) for ch in f.header()["channels"]}


def main():
    a = sys.argv[sys.argv.index("--") + 1:]
    mode, state, v, spp, out = a[0], a[1], int(a[2]), int(a[3]), Path(a[4]); out.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.open_mainfile(filepath=str(C.B4 / f"scene/lego_b4_{state}_s1.blend"))
    sc = bpy.context.scene; sc.frame_current = 1; strip_lights(sc); sc.use_nodes = False
    pl = add_plane()
    d = bpy.data.lights.new("d46_sun", "SUN"); d.energy = C.E_SUN; d.angle = 0.0
    sun = bpy.data.objects.new("d46_sun", d); sc.collection.objects.link(sun)
    sun.rotation_mode = "QUATERNION"; sun.rotation_quaternion = Vector((0, 0, 1)).rotation_difference(Vector(SUN.tolist()))
    pos, look = CAMS[v]
    cd = bpy.data.cameras.new("d46_cam"); cd.type = "PERSP"; cd.lens_unit = "FOV"; cd.angle = math.radians(45)
    cam = bpy.data.objects.new("d46_cam", cd); sc.collection.objects.link(cam); sc.camera = cam
    cam.location = Vector(pos); cam.rotation_mode = "QUATERNION"; cam.rotation_quaternion = (Vector(look) - Vector(pos)).to_track_quat("-Z", "Y")
    r = sc.render; r.engine = "CYCLES"; r.resolution_x = RES; r.resolution_y = RES; r.resolution_percentage = 100; r.film_transparent = True
    c = sc.cycles; c.samples = spp; c.use_adaptive_sampling = False; c.use_denoising = False; c.device = "CPU"
    c.sample_clamp_direct = 0.0; c.sample_clamp_indirect = 0.0; c.pixel_filter_type = "GAUSSIAN"; c.filter_width = 0.01
    c.use_light_tree = False; sc.world.cycles.query_filename = ""; c.max_bounces = 32; c.diffuse_bounces = 32; c.glossy_bounces = 32
    c.transmission_bounces = 32; c.transparent_max_bounces = 32; c.seed = 4610 + v + int(os.environ.get("D46_SEED_OFFSET", "0"))
    vl = sc.view_layers[0]
    if mode == "aov":
        for ob in sc.objects:
            if ob.type == "MESH":
                ob.pass_index = 3 if ob.name == "d46_plane" else (2 if ob.name == "B_red_box" else 1)
        vl.use_pass_position = True; vl.use_pass_normal = True; vl.use_pass_object_index = True; vl.use_pass_diffuse_direct = True
        r.image_settings.file_format = "OPEN_EXR_MULTILAYER"
    else:
        r.image_settings.file_format = "OPEN_EXR"; r.image_settings.color_mode = "RGBA"
    r.image_settings.color_depth = "32"; r.filepath = f"temp_d46ms_{mode}_{state}_{v}"
    bpy.context.view_layer.update(); bpy.ops.render.render(write_still=True)
    L = read_exr(r.filepath + ".exr")
    pick = lambda name, chs: np.stack([L[k] for k in sorted([k for k in L if k.split(".")[-2:-1] == [name] or k == name], key=lambda s: chs.index(s.split(".")[-1]))], 1)  # noqa: E731
    flip = lambda x: x.reshape(RES * RES, -1)  # noqa: E731   EXR row order kept (same for every mode)
    if mode == "aov":
        rec = {"pos": flip(pick("Position", "XYZ")), "nrm": flip(pick("Normal", "XYZ")), "obj": flip(pick("IndexOB", "X"))[:, 0],
               "diffdir": flip(pick("DiffDir", "RGB")), "alpha": flip(pick("Combined", "RGBA"))[:, 3]}
    else:
        rgba = np.stack([L[k] for k in ("R", "G", "B", "A")], 1) if "R" in L else pick("Combined", "RGBA")
        rec = {"rgb": flip(rgba)[:, :3], "alpha": flip(rgba)[:, 3]}
    cam_pos = np.asarray(pos, np.float64); rec["cam_pos"] = cam_pos; rec["spp"] = spp; rec["seed"] = c.seed
    np.savez_compressed(out / f"{mode}_{state}_v{v}.npz", **rec)
    print("done", mode, state, v, sorted(L.keys())[:14], flush=True); sys.stdout.flush(); os._exit(0)


if __name__ == "__main__":
    main()
