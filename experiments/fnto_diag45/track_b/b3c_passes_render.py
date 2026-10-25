"""WL45 Track B3c - full-GI render of held-out T0 view v with Cycles light passes (r1 full settings: max_bounces 32,
no clamp, GAUSSIAN 0.01; the real sun of the view), to obtain an independently rendered incident indirect field at
opaque pixels: for a diffuse closure, Cycles' 'Diffuse Indirect' pass is the indirect light before the diffuse
colour, i.e. E_ind / pi for a Lambertian receiver; 'Diffuse Color' is the closure albedo weight.
Stores at the historical pixel list: combined, DiffDir, DiffInd, DiffCol, GlossDir, GlossInd, GlossCol, TransInd.

    RNA_DIR=<rna repo> python b3c_passes_render.py -- <view> <spp> <out dir>
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
import OpenEXR  # noqa: E402
import Imath  # noqa: E402
from mathutils import Matrix, Vector  # noqa: E402

import r1_common as C  # noqa: E402
from r1_stagea_render import strip_lights  # noqa: E402

PASSES = {"DiffDir": "use_pass_diffuse_direct", "DiffInd": "use_pass_diffuse_indirect", "DiffCol": "use_pass_diffuse_color",
          "GlossDir": "use_pass_glossy_direct", "GlossInd": "use_pass_glossy_indirect", "GlossCol": "use_pass_glossy_color",
          "TransInd": "use_pass_transmission_indirect"}


def read_layer(path):
    f = OpenEXR.InputFile(path); dw = f.header()["dataWindow"]; w, h = dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1
    pt = Imath.PixelType(Imath.PixelType.FLOAT); chans = list(f.header()["channels"].keys()); out = {}
    for ch in chans:
        out[ch] = np.frombuffer(f.channel(ch, pt), np.float32).reshape(h * w)
    return out


def main():
    a = sys.argv[sys.argv.index("--") + 1:]
    v, spp, out = int(a[0]), int(a[1]), Path(a[2]); out.mkdir(parents=True, exist_ok=True)
    views = json.loads((Path.home() / "rest_fnto_r1/views_r1.json").read_text()); view = views["heldout_T0"][v]
    bpy.ops.wm.open_mainfile(filepath=str(C.B4 / "scene/lego_b4_T0_s1.blend"))
    sc = bpy.context.scene; sc.frame_current = 1; strip_lights(sc); sc.use_nodes = False
    d = bpy.data.lights.new("b3c_sun", "SUN"); d.energy = C.E_SUN; d.angle = 0.0
    sun = bpy.data.objects.new("b3c_sun", d); sc.collection.objects.link(sun)
    sun.rotation_mode = "QUATERNION"; sun.rotation_quaternion = Vector((0, 0, 1)).rotation_difference(Vector(view["sun"]))
    cd = bpy.data.cameras.new("b3c_cam"); cd.type = "PERSP"; cd.lens_unit = "FOV"; cd.angle = math.radians(45)
    cam = bpy.data.objects.new("b3c_cam", cd); sc.collection.objects.link(cam); sc.camera = cam
    loc, rot, sca = Matrix(view["mtx"]).decompose(); cam.location = loc; cam.rotation_euler = rot.to_euler(); cam.scale = sca
    r = sc.render; res = views["res"]; r.engine = "CYCLES"; r.resolution_x = res; r.resolution_y = res; r.resolution_percentage = 100
    r.image_settings.file_format = "OPEN_EXR_MULTILAYER"; r.image_settings.color_depth = "32"; r.film_transparent = True; r.filepath = f"temp_b3c_{v}"
    vl = sc.view_layers[0]
    for k, attr in PASSES.items():
        setattr(vl, attr, True)
    c = sc.cycles; c.samples = spp; c.use_adaptive_sampling = False; c.use_denoising = False; c.device = "CPU"
    c.sample_clamp_direct = 0.0; c.sample_clamp_indirect = 0.0; c.pixel_filter_type = "GAUSSIAN"; c.filter_width = 0.01
    c.use_light_tree = False; sc.world.cycles.query_filename = ""; c.max_bounces = 32; c.seed = 4510 + v
    bpy.context.view_layer.update(); bpy.ops.render.render(write_still=True)
    L = read_layer(r.filepath + ".exr"); mk = np.asarray(view["mk"], np.int64)
    rec = {}
    for name in ["Combined"] + list(PASSES):
        cs = [k for k in L if k.split(".")[-2:-1] == [name] or (k.count(".") >= 1 and k.split(".")[-2] == name)]
        rgb = [k for k in sorted(cs) if k.endswith((".R", ".G", ".B"))]
        if len(rgb) == 3:
            rec[name] = np.stack([L[k] for k in sorted(rgb, key=lambda s: "RGB".index(s[-1]))], 1)[mk].astype(np.float32)
    np.savez(out / f"passes_v{v}.npz", **rec, spp=spp, seed=c.seed)
    print("done", v, sorted(rec.keys()), sorted(L.keys())[:12], flush=True); sys.stdout.flush(); os._exit(0)


if __name__ == "__main__":
    main()
