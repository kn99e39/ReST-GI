"""WL42: Cycles light radiometry calibration (server bpy 3.5.1, CPU). Read-only on scene files.

    python r1_calib.py -- <out dir>

A white Lambertian plane (Diffuse BSDF, colour 1, roughness 0) facing +z, camera straight down, light paths = direct only,
no clamp, box filter 0.01 px (point samples). Measured centre radiance L gives irradiance E = pi L.
  sun   (strength S, angle 0, from +z):   expect E = S
  point (power P W, radius 0, height d):  kappa_point = E d^2 / P   (radiant intensity per watt, W/sr per W)
  spot  (power P, radius 0, cone 40 deg, blend 0, aimed down): kappa_spot on the axis, and E at 15 deg / 25 deg off axis
"""

import json
import math
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "fnto_lego"))
import os  # noqa: E402
sys.path.insert(0, os.environ.get("RNA_DIR", str(Path(__file__).resolve().parents[2] / "external/relightable-neural-assets")))
from utils import exr  # noqa: E402


def scene(res=64):
    bpy.ops.wm.read_homefile(use_empty=True)
    sc = bpy.context.scene
    if sc.world is None:
        sc.world = bpy.data.worlds.new("W")
    sc.world.use_nodes = True
    bg = sc.world.node_tree.nodes.get("Background")
    if bg:
        sc.world.node_tree.nodes.remove(bg)
    m = bpy.data.materials.new("white"); m.use_nodes = True; nt = m.node_tree; nt.nodes.clear()
    d = nt.nodes.new("ShaderNodeBsdfDiffuse"); d.inputs["Color"].default_value = (1, 1, 1, 1); d.inputs["Roughness"].default_value = 0.0
    o = nt.nodes.new("ShaderNodeOutputMaterial"); nt.links.new(d.outputs[0], o.inputs["Surface"])
    bpy.ops.mesh.primitive_plane_add(size=20.0); bpy.context.object.data.materials.append(m)
    cd = bpy.data.cameras.new("cam"); cd.type = "ORTHO"; cd.ortho_scale = 0.2
    cam = bpy.data.objects.new("cam", cd); sc.collection.objects.link(cam); cam.location = (0, 0, 5); sc.camera = cam
    r = sc.render; r.engine = "CYCLES"; r.resolution_x = res; r.resolution_y = res; r.film_transparent = True
    r.image_settings.file_format = "OPEN_EXR"; r.image_settings.color_depth = "32"; r.filepath = "temp_calib"
    c = sc.cycles; c.device = "CPU"; c.samples = 256; c.max_bounces = 0; c.sample_clamp_direct = 0; c.sample_clamp_indirect = 0
    c.pixel_filter_type = "BOX"; c.filter_width = 0.01; c.use_denoising = False; c.use_light_tree = False
    return sc


def measure(sc):
    bpy.ops.render.render(write_still=True)
    x = exr.read("temp_calib.exr"); h = x.shape[0] // 2
    return float(x[h - 2:h + 2, h - 2:h + 2, :3].mean())


def main():
    a = sys.argv[sys.argv.index("--") + 1:]; out = Path(a[0]); out.mkdir(parents=True, exist_ok=True)
    rec = {}
    sc = scene(); d = bpy.data.lights.new("s", "SUN"); d.energy = 5.0; d.angle = 0.0
    o = bpy.data.objects.new("s", d); sc.collection.objects.link(o)
    rec["sun_strength5_E"] = math.pi * measure(sc)
    for kind in ("POINT", "SPOT"):
        for h in (0.5, 1.0):
            sc = scene(); d = bpy.data.lights.new("l", kind); d.energy = 100.0; d.shadow_soft_size = 0.0
            if kind == "SPOT":
                d.spot_size = math.radians(40); d.spot_blend = 0.0
            o = bpy.data.objects.new("l", d); sc.collection.objects.link(o); o.location = (0, 0, h)
            E = math.pi * measure(sc)
            rec[f"{kind.lower()}_P100_h{h}_E"] = E; rec[f"kappa_{kind.lower()}_h{h}"] = E * h * h / 100.0
    # spot off-axis: move the light sideways so the centre is seen at 15 / 25 deg (inside / outside the 20 deg half-cone)
    for ang in (15, 25):
        sc = scene(); d = bpy.data.lights.new("l", "SPOT"); d.energy = 100.0; d.shadow_soft_size = 0.0; d.spot_size = math.radians(40); d.spot_blend = 0.0
        o = bpy.data.objects.new("l", d); sc.collection.objects.link(o)
        h = 1.0; x = h * math.tan(math.radians(ang)); o.location = (x, 0, h)            # still pointing straight down
        E = math.pi * measure(sc); rec[f"spot_offaxis_{ang}deg_E_times_r2_over_cos3_per_P"] = E * (h * h + x * x) / (math.cos(math.radians(ang)) * 100.0) * (1 / math.cos(math.radians(ang))) ** 0
    rec["note"] = "kappa = radiant intensity (W/sr) per watt of Blender power; E at the plane = kappa P cos(theta) / r^2"
    (out / "calib.json").write_text(json.dumps(rec, indent=1)); print(json.dumps(rec, indent=1))


if __name__ == "__main__":
    main()
