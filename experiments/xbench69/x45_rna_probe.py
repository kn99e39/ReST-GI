"""WL69 gate checks X4a / X5 (protocol xbench69_v1) for RNA on C1, run in the RNA custom-Cycles bpy (3.5.1 75b072e3ad79):

X4a  RNA's own AOV setup (scripts/setup_blender_scene: Geometry Position/Normal/Incoming via shader AOV outputs + compositor files) on
     the C1 floor scene (medium slab inside a white Transparent BSDF). Is the position AOV of pixels whose first geometric hit is the
     slab equal to that hit (within 1 mm)?  The expected first hit is computed analytically (axis-aligned slab box, floor plane).
X5   CPU cost of one RNA training view (1024^2, perspective fov 45 as BlenderRender.add_camera, hemisphere camera at radius
     2.75 x the asset bounding radius as generate_Lego, reference transport settings), timed at 4 and 32 spp per camera and projected
     linearly to the generate_Lego 1024 spp.

    bpy_native.sh x45_rna_probe.py <out_dir> <rna_repo>
"""
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "fnto_k55")); sys.path.insert(0, str(HERE.parent / "phys_ref53")); sys.path.insert(0, str(HERE.parent / "fnto_c54"))
import bpy  # noqa: E402
from mathutils import Vector  # noqa: E402

import k55_fixtures as KF  # noqa: E402
import p53_cycles as PC  # noqa: E402

OUT = Path(sys.argv[1]).resolve(); OUT.mkdir(parents=True, exist_ok=True); os.chdir(OUT)
sys.path.insert(0, str(Path(sys.argv[2]) / "scripts"))
import setup_blender_scene as RS  # noqa: E402  (RNA's unchanged AOV setup)

RES = 1024
ELEV_Z = (0.2, 0.5, 0.9)                 # hemisphere z of the probe cameras (RNA: z ~ U[0, 0.9999))
SPPS = (4, 32)


def scene():
    sd = KF.scene_def("floor")
    sc = PC.reset(SPPS[0], 69401, bounces=1024)               # WL53/55 reference transport settings (1024 bounces, no clamp)
    sc.render.threads_mode = "AUTO"; sc.render.film_transparent = True   # RNA generator settings
    for n, k, m, V in sd["comps"]:
        PC.mesh_obj(n, V, m["F"], PC.material("vol" if k == "medium" else "lam", 0.8))
    for i, (s, E) in enumerate(sd["lights"]):
        PC.sun(s, E, f"sun{i}")
    vl = bpy.context.view_layer
    vl.use_pass_diffuse_direct = True; vl.use_pass_glossy_direct = True; vl.use_pass_uv = True
    RS.add_aovs(False); RS.add_shading_aov_nodes(False); RS.add_compositor_nodes(False)
    dom = [V for n, k, m, V in sd["comps"] if k == "medium"][0]; flo = [V for n, k, m, V in sd["comps"] if k == "lambert"][0]
    return sc, sd, dom, flo


def camera(sc, z, centre, radius):
    phi = 0.7
    pos = Vector(centre) + radius * Vector((math.sqrt(1 - z * z) * math.cos(phi), math.sqrt(1 - z * z) * math.sin(phi), z))
    cd = bpy.data.cameras.new(f"c{z}"); cd.angle = math.radians(45); cd.clip_start = 1e-4; cd.clip_end = 100
    ob = bpy.data.objects.new(f"c{z}", cd); sc.collection.objects.link(ob); ob.location = pos
    ob.rotation_euler = (Vector(centre) - pos).normalized().to_track_quat("-Z", "Y").to_euler(); sc.camera = ob
    bpy.context.view_layer.update(); return ob


def first_hit(ob, dom, flo):
    """Analytic first hit of each pixel-centre ray (rows bottom-up, as bpy image pixels): slab AABB, then floor plane."""
    M = np.array(ob.matrix_world); o = M[:3, 3]; t = math.tan(ob.data.angle / 2)
    j, i = np.meshgrid(np.arange(RES), np.arange(RES)); x = (2 * (j + 0.5) / RES - 1) * t; y = (2 * (i + 0.5) / RES - 1) * t
    d = np.stack([x, y, -np.ones_like(x)], -1) @ M[:3, :3].T; d /= np.linalg.norm(d, axis=-1, keepdims=True)
    lo, hi = dom.min(0), dom.max(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        t1 = (lo - o) / d; t2 = (hi - o) / d
    tn = np.nanmax(np.minimum(t1, t2), -1); tf = np.nanmin(np.maximum(t1, t2), -1)
    slab = (tf >= tn) & (tf > 0)
    tz = (flo[:, 2].max() - o[2]) / d[..., 2]; pf = o + tz[..., None] * d
    onfloor = (tz > 0) & (np.abs(pf[..., 0]) <= np.abs(flo[:, 0]).max()) & (np.abs(pf[..., 1]) <= np.abs(flo[:, 1]).max())
    hit = np.where(slab[..., None], o + np.maximum(tn, 0)[..., None] * d, pf)
    return hit, slab, onfloor & ~slab


def load_exr(path, ch=3):
    img = bpy.data.images.load(str(path)); a = np.empty(len(img.pixels), np.float32); img.pixels.foreach_get(a)
    bpy.data.images.remove(img); return a.reshape(RES, RES, 4)[..., :ch]


def main():
    sc, sd, dom, flo = scene(); fv = np.concatenate([dom, flo]); centre = 0.5 * (fv.min(0) + fv.max(0))
    radius = 2.75 * 0.5 * float(np.linalg.norm(fv.max(0) - fv.min(0)))
    rec = dict(protocol="xbench69_v1", checks="X4a,X5", blender=bpy.app.version_string, build_hash=str(bpy.app.build_hash), threads=os.cpu_count(),
               res=RES, camera_radius_m=radius, centre=centre.tolist(), cams={})
    r = sc.render; r.resolution_x = r.resolution_y = RES; r.resolution_percentage = 100
    r.image_settings.file_format = "OPEN_EXR"; r.image_settings.color_mode = "RGBA"; r.image_settings.color_depth = "32"; r.filepath = str(OUT / "beauty")
    for z in ELEV_Z:
        ob = camera(sc, z, centre, radius); times = {}
        for spp in SPPS:
            sc.cycles.samples = spp; t0 = time.time(); bpy.ops.render.render(write_still=True); times[spp] = time.time() - t0
        pos = load_exr(OUT / "position0001.exr"); beauty = load_exr(OUT / "beauty.exr", 4)
        hit, slab, floor = first_hit(ob, dom, flo)
        err = np.linalg.norm(pos - hit, axis=-1)
        slope = (times[SPPS[1]] - times[SPPS[0]]) / (SPPS[1] - SPPS[0])
        c = dict(z=z, seconds=times, proj_1024spp_s=times[SPPS[0]] + slope * (1024 - SPPS[0]), n_slab_px=int(slab.sum()), n_floor_px=int(floor.sum()),
                 slab_px_within_1mm=float((err[slab] < 1e-3).mean()), floor_px_within_1mm=float((err[floor] < 1e-3).mean()),
                 slab_err_p50_mm=float(np.median(err[slab]) * 1e3), slab_err_p90_mm=float(np.percentile(err[slab], 90) * 1e3),
                 slab_pos_z_p50=float(np.median(pos[slab][:, 2])), slab_hit_z_p50=float(np.median(hit[slab][:, 2])),
                 alpha_mean_slab=float(beauty[..., 3][slab].mean()))
        rec["cams"][str(z)] = c; print(json.dumps(c), flush=True)
        np.save(OUT / f"pos_z{z}.npy", pos.astype(np.float32)); np.save(OUT / f"hit_z{z}.npy", hit.astype(np.float32)); np.save(OUT / f"slab_z{z}.npy", slab)
    p = [c["proj_1024spp_s"] for c in rec["cams"].values()]
    rec["proj_view_1024spp_s_mean"] = float(np.mean(p))
    rec["X4a_pass"] = bool(all(c["slab_px_within_1mm"] >= 0.99 for c in rec["cams"].values()))
    json.dump(rec, open(OUT / "x45.json", "w"), indent=1); print(json.dumps({k: rec[k] for k in ("proj_view_1024spp_s_mean", "X4a_pass")}))


main()
