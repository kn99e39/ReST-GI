"""WL42 - is a Cycles (bpy 3.5.1) film with pixel_filter_type BOX, filter_width 0.01 point-sampled? (LabServer63, CPU)

Renders held-out T0 view 0 (r1 reference setup, full GI, 64 spp) with ONE filter setting per process (in this bpy build
integrator / film settings changed after the first render of a process are not re-synced: a later samples = 1 still
renders 64/64), saves the RGBA, and in 'compare' mode reports per setting the fraction of fractional-alpha pixels and the
RGB difference between settings. A point-sampled film has binary alpha (up to transparency); a pixel-wide filter has
fractional alpha on silhouette pixels.

    RNA_DIR=<rna repo> python r1_filter_check.py -- <views.json> <out dir> render <BOX|GAUSSIAN> <width>
    python r1_filter_check.py -- <views.json> <out dir> compare
"""
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def render(views, out, ft, w):
    import bpy
    from mathutils import Matrix, Vector
    sys.path.insert(0, os.environ.get("RNA_DIR", str(HERE.parents[1] / "external/relightable-neural-assets")))
    sys.path.insert(0, str(HERE.parent / "fnto_lego")); sys.path.insert(0, str(HERE))
    from utils import exr
    import r1_common as C
    from r1_stagea_render import strip_lights
    view = views["heldout_T0"][0]
    bpy.ops.wm.open_mainfile(filepath=str(C.B4 / "scene/lego_b4_T0_s1.blend"))
    sc = bpy.context.scene; sc.frame_current = 1; strip_lights(sc); sc.use_nodes = False
    d = bpy.data.lights.new("r1_sun", "SUN"); d.energy = C.E_SUN; d.angle = 0.0
    sun = bpy.data.objects.new("r1_sun", d); sc.collection.objects.link(sun)
    sun.rotation_mode = "QUATERNION"; sun.rotation_quaternion = Vector((0, 0, 1)).rotation_difference(Vector(view["sun"]))
    cd = bpy.data.cameras.new("r1_cam"); cd.type = "PERSP"; cd.lens_unit = "FOV"; cd.angle = math.radians(45)
    cam = bpy.data.objects.new("r1_cam", cd); sc.collection.objects.link(cam); sc.camera = cam
    loc, rot, sca = Matrix(view["mtx"]).decompose(); cam.location = loc; cam.rotation_euler = rot.to_euler(); cam.scale = sca
    r = sc.render; res = views["res"]
    r.engine = "CYCLES"; r.resolution_x = res; r.resolution_y = res; r.resolution_percentage = 100
    r.image_settings.file_format = "OPEN_EXR"; r.image_settings.color_depth = "32"; r.image_settings.color_mode = "RGBA"
    r.film_transparent = True; r.filepath = f"temp_filter_{ft}_{w}"
    c = sc.cycles; c.samples = 64; c.use_adaptive_sampling = False; c.use_denoising = False; c.device = "CPU"; c.max_bounces = 32
    c.sample_clamp_direct = 0.0; c.sample_clamp_indirect = 0.0; c.use_light_tree = False; sc.world.cycles.query_filename = ""
    c.pixel_filter_type = ft; c.filter_width = w; c.seed = 0; bpy.context.view_layer.update()
    bpy.ops.render.render(write_still=True)
    img = exr.read(r.filepath + ".exr").reshape(-1, 4)
    np.save(out / f"filter_{ft}_{w}.npy", img.astype(np.float32))
    (out / f"filter_{ft}_{w}.json").write_text(json.dumps({"filter": ft, "width_readback": float(c.filter_width), "persistent_data": bool(r.use_persistent_data),
                                                           "blender": bpy.app.version_string}))


def compare(out):
    rec = {}; imgs = {p.stem: np.load(p) for p in sorted(out.glob("filter_*.npy"))}
    ref = imgs.get("filter_BOX_0.01")
    for k, im in imgs.items():
        al = im[:, 3]
        rec[k] = {"frac_alpha_fractional": float(((al > 1e-4) & (al < 1 - 1e-4)).mean()), "frac_alpha_positive": float((al > 1e-4).mean()),
                  **json.loads((out / f"{k}.json").read_text())}
        if ref is not None and k != "filter_BOX_0.01":
            m = (al > 1e-4) | (ref[:, 3] > 1e-4)
            rec[k]["vs_BOX_0.01_mean_abs_rgb"] = float(np.abs(im[m, :3] - ref[m, :3]).mean()); rec[k]["vs_BOX_0.01_mean_abs_alpha"] = float(np.abs(al[m] - ref[m, 3]).mean())
    (out / "filter_check.json").write_text(json.dumps(rec, indent=1)); print(json.dumps(rec, indent=1))


if __name__ == "__main__":
    a = sys.argv[sys.argv.index("--") + 1:]
    out = Path(a[1]); out.mkdir(parents=True, exist_ok=True)
    if a[2] == "render":
        render(json.loads(Path(a[0]).read_text()), out, a[3], float(a[4]))
    else:
        compare(out)
