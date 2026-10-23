"""FNTO attribution references: first-intrinsic-event renders of the worklog-37 held-out views (WSL RNA venv bpy).

    python fnto_firstevent_render.py -- <out dir> <set: heldout_T0 | heldout_T1> [--spp 1024]

Same blend (the B4 state file), same camera matrices (<set>_s1_val.json), same per-view sun direction (light_dir of the
held-out H5), same resolution, clamp and light convention as the held-out references -- but light paths limited to the
first intrinsic event (fnto_semantics). These images are NOT a reference for any verdict rule: they split the held-out
reference into its first-event part (what K alone must reproduce from the exact incident sun) and the remainder (what the
free-space coupling C and the multi-bounce solve must add). Output: <out>/<set>_first.npz with per view the alpha > 0
pixel indices and RGB, plus an alignment check against the held-out H5 alpha mask.
"""

import json
import math
import sys
import time
from pathlib import Path

import bpy
import h5py
import numpy as np
from mathutils import Matrix, Vector

U = Path("/mnt/c/Projects/ReST-GI/external/relightable-neural-assets")
sys.path.insert(0, str(U)); sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import exr  # noqa: E402

import fnto_common as C  # noqa: E402
from fnto_stagea_render import first_event, set_sun, strip_lights  # noqa: E402


def main():
    a = sys.argv[sys.argv.index("--") + 1:]
    out = Path(a[0]); set_name = a[1]; spp = int(a[a.index("--spp") + 1]) if "--spp" in a else 1024
    state = set_name.split("_")[1]
    blend = C.B4 / f"scene/lego_b4_{state}_s1.blend"
    cams = json.loads((C.B4 / f"data/{set_name}_s1_val.json").read_text())["cameras"]
    f = h5py.File(C.B4 / f"data/{set_name}_s1_val.h5", "r"); res = int(f.attrs["resolution"][0])
    bpy.ops.wm.open_mainfile(filepath=str(blend))
    sc = bpy.context.scene; sc.frame_current = 1; strip_lights(sc)
    data = bpy.data.lights.new("FNTO_sun", "SUN"); data.energy = C.E_SUN; data.angle = 0.0
    sun = bpy.data.objects.new("FNTO_sun", data); sc.collection.objects.link(sun)
    cd = bpy.data.cameras.new("FNTO_cam"); cd.type = "PERSP"; cd.lens_unit = "FOV"; cd.angle = math.radians(45)
    cam = bpy.data.objects.new("FNTO_cam", cd); sc.collection.objects.link(cam); sc.camera = cam
    r = sc.render
    r.engine = "CYCLES"; r.resolution_x = res; r.resolution_y = res; r.resolution_percentage = 100
    r.image_settings.file_format = "OPEN_EXR"; r.image_settings.color_depth = "32"; r.image_settings.color_mode = "RGBA"
    r.film_transparent = True; r.filepath = "temp_first"; sc.use_nodes = False
    c = sc.cycles
    c.samples = spp; c.use_adaptive_sampling = False; c.use_denoising = False; c.sample_clamp_direct = 20
    c.use_light_tree = False; c.device = "CPU"; sc.world.cycles.query_filename = ""; c.seed = 0
    first_event(c)
    rec = {"set": set_name, "blend": str(blend), "spp": spp, "views": []}; store = {}
    for v, cm in enumerate(cams):
        loc, rot, sca = Matrix(cm["mtx"]).decompose(); cam.location = loc; cam.rotation_euler = rot.to_euler(); cam.scale = sca
        L = np.nan_to_num(f["light_dir"][v].astype(np.float32)); mk = np.nan_to_num(f["alpha"][v, :, 0].astype(np.float32)) > 0
        w = L[mk].mean(0); w /= np.linalg.norm(w)
        set_sun(sun, w); bpy.context.view_layer.update()
        t = time.time(); bpy.ops.render.render(write_still=True)
        img = exr.read("temp_first.exr"); img = img.reshape(-1, img.shape[-1])
        a_ = img[:, 3] > 0
        store[f"v{v}_mk"] = np.flatnonzero(mk).astype(np.int32); store[f"v{v}_rgb"] = img[mk, :3].astype(np.float32)
        rec["views"].append({"v": v, "seconds": time.time() - t, "alpha_px": int(a_.sum()), "h5_px": int(mk.sum()),
                             "alpha_agree": float((a_ == mk).mean()), "sun": w.tolist()})
        print(json.dumps(rec["views"][-1]), flush=True)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / f"{set_name}_first.npz", **store)
    (out / f"{set_name}_first.json").write_text(json.dumps(rec, indent=1))


if __name__ == "__main__":
    main()
