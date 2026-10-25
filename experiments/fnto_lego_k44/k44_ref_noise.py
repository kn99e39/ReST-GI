"""WL44 - noise floor of the r1 held-out references: re-render ONE view of a set in mode first / full with a different
Cycles seed (fresh process; settings before the first render), store rgb at the historical pixel list, and compare
with refs/<set>_<mode>.npz. Single-reference floor = pair PSNR + 3.01 dB.

    RNA_DIR=<rna repo> python k44_ref_noise.py -- <set> <mode> <view> <out dir>
    python k44_ref_noise.py cmp <out dir>        -> results/fnto_lego_k44/A_ref_noise.json
"""
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "fnto_lego_r1")); sys.path.insert(0, str(HERE.parent / "fnto_lego"))


def render(set_name, mode, v, out):
    sys.path.insert(0, os.environ.get("RNA_DIR", str(HERE.parents[1] / "external/relightable-neural-assets")))
    import bpy
    from mathutils import Matrix, Vector
    from utils import exr
    import r1_common as C
    from r1_stagea_render import mask_intrinsic, strip_lights
    views = json.loads((Path.home() / "rest_fnto_r1/views_r1.json").read_text()); view = views[set_name][v]
    spp = 4096 if mode == "full" else 1024
    bpy.ops.wm.open_mainfile(filepath=str(C.B4 / f"scene/lego_b4_{set_name.split('_')[1]}_s1.blend"))
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
    r.film_transparent = True; r.filepath = f"temp_k44_{set_name}_{mode}_{v}"
    c = sc.cycles; c.samples = spp; c.use_adaptive_sampling = False; c.use_denoising = False; c.device = "CPU"
    c.sample_clamp_direct = 0.0; c.sample_clamp_indirect = 0.0; c.pixel_filter_type = "GAUSSIAN"; c.filter_width = 0.01
    c.use_light_tree = False; sc.world.cycles.query_filename = ""
    if mode == "full":
        c.max_bounces = 32
    else:
        c.max_bounces = 128; c.diffuse_bounces = 128; c.glossy_bounces = 128; c.transmission_bounces = 128; c.transparent_max_bounces = 128
    loc, rot, sca = Matrix(view["mtx"]).decompose(); cam.location = loc; cam.rotation_euler = rot.to_euler(); cam.scale = sca
    sun.rotation_mode = "QUATERNION"; sun.rotation_quaternion = Vector((0, 0, 1)).rotation_difference(Vector(view["sun"]))
    c.seed = 104729 + v; bpy.context.view_layer.update()
    bpy.ops.render.render(write_still=True)
    img = exr.read(r.filepath + ".exr").reshape(-1, 4); mk = np.asarray(view["mk"], np.int64)
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / f"{set_name}_{mode}_v{v}.npz", rgb=img[mk, :3].astype(np.float32), seed=c.seed)
    print("done", flush=True); sys.stdout.flush(); os._exit(0)


def cmp(out):
    import k44_common as K
    rec = {}
    for f in sorted(out.glob("heldout_*_v*.npz")):
        s, mode, v = f.stem.rsplit("_", 2)[0], f.stem.rsplit("_", 2)[1], int(f.stem.rsplit("_", 2)[2][1:])
        a = np.load(K.C.OUT1 / f"refs/{s}_{mode}.npz")[f"v{v}_rgb"].astype(np.float64); b = np.load(f)["rgb"].astype(np.float64)
        px = np.load(K.C.OUT1 / f"eval_px/{s}.npz"); val = px[f"v{v}_valid"].astype(bool)
        mat = np.where(px[f"v{v}_dom"] == 1, 8, px[f"v{v}_mat"])[val]
        se = ((K.C.ldr3(a[val]) - K.C.ldr3(b[val])) ** 2).mean(1)
        r = {"pair_psnr": float(10 * np.log10(len(se) / se.sum()))}; r["single_ref_floor_psnr"] = r["pair_psnr"] + 3.0103
        names = K.C.MATERIALS[:8] + ["box"]
        r["per_material_floor"] = {names[i]: float(10 * np.log10((mat == i).sum() / max(se[mat == i].sum(), 1e-30)) + 3.0103) for i in range(9) if (mat == i).any()}
        rec[f.stem] = r
    K.save("A_ref_noise.json", rec)


if __name__ == "__main__":
    if sys.argv[1] == "cmp":
        cmp(Path(sys.argv[2]))
    else:
        a = sys.argv[sys.argv.index("--") + 1:]
        render(a[0], a[1], int(a[2]), Path(a[3]))
