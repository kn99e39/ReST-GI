"""WL42 reference regime r1 for the worklog-37 held-out views (LabServer63 bpy 3.5.1, CPU). Versioned separately; the
historical references (clamp_direct 20, clamp_indirect 5, point-sampled) are not touched.

    RNA_DIR=<rna repo> python r1_refs_render.py -- <out dir> <views.json> <set> <mode: full | first> [--spp N]

views.json (made on the PC from the historical held-out H5 / JSON): per set, per view the camera matrix, the single sun
direction and the alpha > 0 pixel indices (so r1 pixels align with the historical inputs and regions).
  full  : full GI, max_bounces 32 (the generator's override; per-type 128 from the blend), NO clamp, GAUSSIAN 0.01 px filter
          (point samples, as the historical held-out renders; amendment 2 - BOX 0.01 was a 1 px box), 4096 spp, Cycles seed = view index
  first : intrinsic-all (free-space hits absorbed; glass internal paths kept), same filter / clamp, 1024 spp
Output <out>/<set>_<mode>.npz: v<k>_rgb at the historical alpha > 0 pixels, and the r1 alpha agreement.
"""

import json
import math
import os
import sys
import time
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix, Vector

HERE = Path(__file__).resolve().parent
sys.path.insert(0, os.environ.get("RNA_DIR", str(HERE.parents[1] / "external/relightable-neural-assets")))
sys.path.insert(0, str(HERE.parent / "fnto_lego")); sys.path.insert(0, str(HERE))
from utils import exr  # noqa: E402

import r1_common as C  # noqa: E402
from r1_stagea_render import mask_intrinsic, strip_lights  # noqa: E402


def main():
    a = sys.argv[sys.argv.index("--") + 1:]
    out = Path(a[0]); views = json.loads(Path(a[1]).read_text()); set_name = a[2]; mode = a[3]
    spp = int(a[a.index("--spp") + 1]) if "--spp" in a else (4096 if mode == "full" else 1024)
    state = set_name.split("_")[1]
    bpy.ops.wm.open_mainfile(filepath=str(C.B4 / f"scene/lego_b4_{state}_s1.blend"))
    sc = bpy.context.scene; sc.frame_current = 1; strip_lights(sc); sc.use_nodes = False
    if mode == "first":
        mask_intrinsic()
    d = bpy.data.lights.new("r1_sun", "SUN"); d.energy = C.E_SUN; d.angle = 0.0
    sun = bpy.data.objects.new("r1_sun", d); sc.collection.objects.link(sun)
    cd = bpy.data.cameras.new("r1_cam"); cd.type = "PERSP"; cd.lens_unit = "FOV"; cd.angle = math.radians(45)
    cam = bpy.data.objects.new("r1_cam", cd); sc.collection.objects.link(cam); sc.camera = cam
    r = sc.render; res = views["res"]
    r.engine = "CYCLES"; r.resolution_x = res; r.resolution_y = res; r.resolution_percentage = 100
    r.image_settings.file_format = "OPEN_EXR"; r.image_settings.color_depth = "32"; r.image_settings.color_mode = "RGBA"
    r.film_transparent = True; r.filepath = f"temp_{set_name}_{mode}"
    c = sc.cycles; c.samples = spp; c.use_adaptive_sampling = False; c.use_denoising = False; c.device = "CPU"
    c.sample_clamp_direct = 0.0; c.sample_clamp_indirect = 0.0; c.pixel_filter_type = "GAUSSIAN"; c.filter_width = 0.01  # amendment 2 (r1_filter_check)
    c.use_light_tree = False; sc.world.cycles.query_filename = ""
    if mode == "full":
        c.max_bounces = 32
    else:
        c.max_bounces = 128; c.diffuse_bounces = 128; c.glossy_bounces = 128; c.transmission_bounces = 128; c.transparent_max_bounces = 128
    store, rec = {}, {"set": set_name, "mode": mode, "spp": spp, "clamp": [0, 0], "filter": "GAUSSIAN 0.01 (amendment 2)", "views": []}
    out.mkdir(parents=True, exist_ok=True)
    for v, view in enumerate(views[set_name]):
        loc, rot, sca = Matrix(view["mtx"]).decompose(); cam.location = loc; cam.rotation_euler = rot.to_euler(); cam.scale = sca
        sun.rotation_mode = "QUATERNION"; sun.rotation_quaternion = Vector((0, 0, 1)).rotation_difference(Vector(view["sun"]))
        c.seed = v; bpy.context.view_layer.update()
        t = time.time(); bpy.ops.render.render(write_still=True)
        img = exr.read(r.filepath + ".exr"); img = img.reshape(-1, img.shape[-1])
        mk = np.asarray(view["mk"], np.int64); a_ = img[:, 3] > 0
        store[f"v{v}_rgb"] = img[mk, :3].astype(np.float32); store[f"v{v}_mk"] = mk.astype(np.int32)
        mkb = np.zeros(len(img), bool); mkb[mk] = True
        rec["views"].append({"v": v, "seconds": time.time() - t, "alpha_agree": float((a_ == mkb).mean())})
        print(json.dumps(rec["views"][-1]), flush=True)
        np.savez_compressed(out / f"{set_name}_{mode}.npz", **store); (out / f"{set_name}_{mode}.json").write_text(json.dumps(rec, indent=1))


if __name__ == "__main__":
    main()
