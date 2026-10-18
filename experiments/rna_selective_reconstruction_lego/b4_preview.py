"""Preview render (diagnostic only, not evidence): T0/T1 from a fixed front camera under a fixed sun, standard Cycles."""
import sys
import bpy
import numpy as np
from mathutils import Vector

blend, out, cam_pos, sun_dir = sys.argv[1], sys.argv[2], [float(x) for x in sys.argv[3].split(",")], [float(x) for x in sys.argv[4].split(",")]
bpy.ops.wm.open_mainfile(filepath=blend)
sc = bpy.context.scene
for o in sc.objects:
    if o.type == "LIGHT":
        o.hide_render = True
bpy.ops.object.light_add(type="SUN"); sun = bpy.context.object; sun.data.energy = 5.0; sun.data.angle = 0.0
d = Vector(sun_dir).normalized(); sun.rotation_euler = (-d).to_track_quat("-Z", "Y").to_euler()
cam = sc.camera; cam.location = cam_pos
cam.rotation_euler = (Vector((0, 0, 0)) - Vector(cam_pos)).to_track_quat("-Z", "Y").to_euler()
cam.data.lens_unit = "FOV"; cam.data.angle = np.deg2rad(45)
sc.use_nodes = False
sc.render.engine = "CYCLES"; sc.cycles.device = "CPU"; sc.cycles.samples = 128; sc.cycles.max_bounces = 32
sc.render.resolution_x = sc.render.resolution_y = 512; sc.render.film_transparent = True
sc.render.image_settings.file_format = "PNG"; sc.render.filepath = out
sc.view_settings.view_transform = "Standard"
bpy.ops.render.render(write_still=True)
