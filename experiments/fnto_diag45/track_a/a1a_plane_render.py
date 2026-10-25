"""WL45 Track A1a - the Lego Yellow material's Burley scattering length, measured in Blender (not assumed).

A 3 x 1 m plane carrying the blend's own 'Yellow' material, a zenith sun (angle 0, strength 5) and a shadow-only
occluder (invisible to the camera) covering x > 0, so the shadow edge is the line x = 0. Orthographic top-down camera
over x in [-0.6, 1.0] m (1024 x 64 px, GAUSSIAN 0.01, no clamp, 4096 spp, CPU). The radiance profile across the edge
is the half-plane step response of the subsurface profile; k_a1a_fit.py fits the Burley length per channel.

    RNA_DIR=<rna repo> python a1a_plane_render.py -- <out dir> [spp=4096]
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
from mathutils import Vector  # noqa: E402
from utils import exr  # noqa: E402

import r1_common as C  # noqa: E402
from r1_stagea_render import strip_lights  # noqa: E402


def main():
    a = sys.argv[sys.argv.index("--") + 1:]
    out = Path(a[0]); out.mkdir(parents=True, exist_ok=True); spp = int(a[1]) if len(a) > 1 else 4096
    bpy.ops.wm.open_mainfile(filepath=str(C.B4 / "scene/lego_b4_T0_s1.blend"))
    sc = bpy.context.scene; sc.frame_current = 1; strip_lights(sc); sc.use_nodes = False
    yellow = bpy.data.materials["Yellow"]
    for ob in list(sc.objects):
        ob.hide_render = True
    bpy.ops.mesh.primitive_plane_add(size=1.0, location=(0.2, 0, 0)); pl = bpy.context.object; pl.scale = (3.0, 1.0, 1.0)
    bpy.ops.object.transform_apply(scale=True); pl.data.materials.append(yellow)
    bpy.ops.mesh.primitive_plane_add(size=1.0, location=(1.25, 0, 0.3)); occ = bpy.context.object; occ.scale = (2.5, 3.0, 1.0)   # covers x in [0, 2.5]
    bpy.ops.object.transform_apply(scale=True)
    occ.visible_camera = False; occ.visible_diffuse = False; occ.visible_glossy = False; occ.visible_transmission = False; occ.visible_shadow = True
    d = bpy.data.lights.new("a1a_sun", "SUN"); d.energy = C.E_SUN; d.angle = 0.0
    sun = bpy.data.objects.new("a1a_sun", d); sc.collection.objects.link(sun); sun.rotation_euler = (0, 0, 0)    # emits along -Z
    cd = bpy.data.cameras.new("a1a_cam"); cd.type = "ORTHO"; cd.ortho_scale = 1.6
    cam = bpy.data.objects.new("a1a_cam", cd); sc.collection.objects.link(cam); sc.camera = cam
    cam.location = Vector((0.2, 0.0, 1.0)); cam.rotation_euler = (0, 0, 0)
    r = sc.render; r.engine = "CYCLES"; r.resolution_x = 1024; r.resolution_y = 64; r.resolution_percentage = 100
    r.image_settings.file_format = "OPEN_EXR"; r.image_settings.color_depth = "32"; r.image_settings.color_mode = "RGBA"
    r.film_transparent = True; r.filepath = "temp_a1a"
    c = sc.cycles; c.samples = spp; c.use_adaptive_sampling = False; c.use_denoising = False; c.device = "CPU"
    c.sample_clamp_direct = 0.0; c.sample_clamp_indirect = 0.0; c.pixel_filter_type = "GAUSSIAN"; c.filter_width = 0.01
    c.use_light_tree = False; sc.world.cycles.query_filename = ""; c.max_bounces = 32; c.seed = 4501
    bpy.context.view_layer.update(); bpy.ops.render.render(write_still=True)
    img = exr.read(r.filepath + ".exr")                       # [64, 1024, 4]
    prof = img[8:56, :, :3].mean(0)                            # rows away from the strip ends
    x = 0.2 - 0.8 + (np.arange(1024) + 0.5) * 1.6 / 1024
    np.savez(out / "a1a_profile.npz", x=x, rgb=prof, spp=spp)
    ss = [n for n in yellow.node_tree.nodes if n.bl_idname == "ShaderNodeSubsurfaceScattering"][0]
    meta = {"falloff": ss.falloff, "radius": list(ss.inputs["Radius"].default_value), "scale": ss.inputs["Scale"].default_value,
            "color": list(ss.inputs["Color"].default_value), "ior": ss.inputs["IOR"].default_value, "spp": spp, "blender": bpy.app.version_string}
    (out / "a1a_meta.json").write_text(json.dumps(meta, indent=1)); print(meta); sys.stdout.flush(); os._exit(0)


if __name__ == "__main__":
    main()
