"""WL115 (user request): build a .blend of a WL115 view from blendspec115's JSON (no rendering).
blender --background --factory-startup --python blend115_build.py -- SPEC.json OUT.blend [REF_V0.npy ...]"""
import json, math, sys
import bpy
import numpy as np
from mathutils import Matrix, Vector

argv = sys.argv[sys.argv.index('--') + 1:]
spec = json.load(open(argv[0])); out = argv[1]; refs = argv[2:]

bpy.ops.wm.read_factory_settings(use_empty=True)
sc = bpy.context.scene
sc.render.engine = 'CYCLES'
cy = sc.cycles
cy.samples = 1024
cy.use_denoising = False
cy.use_adaptive_sampling = False
for k in ('max_bounces', 'diffuse_bounces', 'glossy_bounces', 'transmission_bounces', 'volume_bounces', 'transparent_max_bounces'):
    setattr(cy, k, 64)
cy.sample_clamp_direct = 0.0; cy.sample_clamp_indirect = 0.0
cy.caustics_reflective = True; cy.caustics_refractive = True; cy.blur_glossy = 0.0
cy.pixel_filter_type = 'BOX'; cy.filter_width = 1.0
sc.render.resolution_x = sc.render.resolution_y = 512; sc.render.resolution_percentage = 100
sc.view_settings.view_transform = 'Standard'; sc.view_settings.look = 'None'; sc.view_settings.exposure = 0.0; sc.view_settings.gamma = 1.0
world = bpy.data.worlds.new('black'); world.use_nodes = True
world.node_tree.nodes['Background'].inputs['Color'].default_value = (0, 0, 0, 1)
world.node_tree.nodes['Background'].inputs['Strength'].default_value = 0.0
sc.world = world


def lambert(name, a):
    m = bpy.data.materials.new(name); m.use_nodes = True
    nt = m.node_tree; nt.nodes.clear()
    o = nt.nodes.new('ShaderNodeOutputMaterial'); d = nt.nodes.new('ShaderNodeBsdfDiffuse')
    d.inputs['Color'].default_value = (a, a, a, 1); d.inputs['Roughness'].default_value = 0.0
    nt.links.new(d.outputs[0], o.inputs['Surface'])
    return m


def layered(name, desc):
    """Approximation of the WL81 coated conductor (GGX coat alpha_i, IOR eta, thin scattering slab tau / albedo, GGX conductor alpha_b /
    reflectance r) by a Principled BSDF: metallic base colour r, roughness sqrt(alpha_b), coat weight 1, coat roughness sqrt(alpha_i),
    coat IOR eta (Blender roughness = sqrt(GGX alpha)). The thin slab (tau <= 0.043) is not represented."""
    ai, eta, tau, a0, a1, a2, r0, r1, r2, ab = desc
    m = bpy.data.materials.new(name); m.use_nodes = True
    p = m.node_tree.nodes['Principled BSDF']
    p.inputs['Base Color'].default_value = (r0, r1, r2, 1); p.inputs['Metallic'].default_value = 1.0
    p.inputs['Roughness'].default_value = math.sqrt(ab)
    p.inputs['Coat Weight'].default_value = 1.0; p.inputs['Coat Roughness'].default_value = math.sqrt(ai); p.inputs['Coat IOR'].default_value = eta
    return m


def sss_medium(name, sigma_t, alb):
    """ReST-GI SSS sphere semantics (WL100 / WL53 validated Cycles equivalent): index-matched boundary (white Transparent BSDF, no
    refraction, no Fresnel) around a homogeneous isotropic medium, sigma_s = sigma_t * albedo, sigma_a = sigma_t * (1 - albedo)."""
    m = bpy.data.materials.new(name); m.use_nodes = True
    nt = m.node_tree; nt.nodes.clear()
    o = nt.nodes.new('ShaderNodeOutputMaterial')
    tr = nt.nodes.new('ShaderNodeBsdfTransparent'); tr.inputs['Color'].default_value = (1, 1, 1, 1)
    sca = nt.nodes.new('ShaderNodeVolumeScatter'); ab = nt.nodes.new('ShaderNodeVolumeAbsorption'); add = nt.nodes.new('ShaderNodeAddShader')
    for n in (sca, ab):
        n.inputs['Color'].default_value = (alb[0], alb[1], alb[2], 1); n.inputs['Density'].default_value = sigma_t
    if 'Anisotropy' in sca.inputs:
        sca.inputs['Anisotropy'].default_value = 0.0
    nt.links.new(sca.outputs[0], add.inputs[0]); nt.links.new(ab.outputs[0], add.inputs[1])
    nt.links.new(tr.outputs[0], o.inputs['Surface']); nt.links.new(add.outputs[0], o.inputs['Volume'])
    try:
        m.cycles.homogeneous_volume = True
    except AttributeError:
        pass
    return m


mat = {}
for i in range(6):
    s = str(i)
    if s in spec['lambert_albedo']:
        mat[i] = lambert(f'lambert_{i}', spec['lambert_albedo'][s])
    else:
        mat[i] = layered(['floor_B_coated_conductor_approx', 'panel_S1_coated_conductor_approx', 'unused_S2'][i], spec['layered_desc'][s])
names = ['floor', 'ceiling', 'back_wall', 'left_wall', 'right_wall', 'panel']
for nm, r in zip(names, spec['rects']):
    c, a1, a2 = Vector(r['centre']), Vector(r['a1']), Vector(r['a2']); h1, h2 = r['h']
    vs = [c - a1 * h1 - a2 * h2, c + a1 * h1 - a2 * h2, c + a1 * h1 + a2 * h2, c - a1 * h1 + a2 * h2]   # normal = a1 x a2 (into the room)
    me = bpy.data.meshes.new(nm); me.from_pydata([tuple(v) for v in vs], [], [(0, 1, 2, 3)]); me.update()
    ob = bpy.data.objects.new(nm, me); sc.collection.objects.link(ob); ob.data.materials.append(mat[r['mat']])
for b in spec['boxes']:
    M = Matrix(b['matrix'])
    vs = [tuple(M @ Vector((x, y, z))) for z in (-1, 1) for y in (-1, 1) for x in (-1, 1)]
    faces = [(0, 2, 3, 1), (4, 5, 7, 6), (0, 1, 5, 4), (2, 6, 7, 3), (0, 4, 6, 2), (1, 3, 7, 5)]
    me = bpy.data.meshes.new(b['name']); me.from_pydata(vs, [], faces); me.update()
    me.validate(); ob = bpy.data.objects.new(b['name'], me); sc.collection.objects.link(ob); ob.data.materials.append(mat[b['mat']])
    bpy.context.view_layer.objects.active = ob
# outward box normals (from_pydata winding is not guaranteed): recompute
for ob in [o for o in sc.objects if o.name.startswith('box')]:
    bpy.context.view_layer.objects.active = ob; ob.select_set(True)
    bpy.ops.object.mode_set(mode='EDIT'); bpy.ops.mesh.select_all(action='SELECT'); bpy.ops.mesh.normals_make_consistent(inside=False)
    bpy.ops.object.mode_set(mode='OBJECT'); ob.select_set(False)
sp = spec['sphere']
bpy.ops.mesh.primitive_uv_sphere_add(segments=512, ring_count=256, radius=sp['radius'], location=sp['centre'])
s_ob = bpy.context.active_object; s_ob.name = 'sss_sphere'
bpy.ops.object.shade_smooth()
s_ob.data.materials.append(sss_medium('sss_sphere_medium', spec['sss']['sigma_t'], spec['sss']['albedo']))
for i, L in enumerate(spec['lights']):
    I = max(L['rgb'])
    ld = bpy.data.lights.new(f'point_light_{i}', 'POINT')
    ld.energy = 4 * math.pi * I                          # radiant intensity I [W/sr] (ReST-GI rgb) -> Blender point power P = 4 pi I
    ld.color = tuple(c / I for c in L['rgb'])
    ld.shadow_soft_size = 0.0
    if hasattr(ld, 'use_soft_falloff'):
        ld.use_soft_falloff = False
    lo = bpy.data.objects.new(f'point_light_{i}', ld); lo.location = L['pos']; sc.collection.objects.link(lo)
for vn, cam in spec['cameras'].items():
    o = Vector(cam['origin']); t = Vector(cam['target'])
    fwd = (t - o).normalized(); right = fwd.cross(Vector((0, 0, 1))).normalized(); up = right.cross(fwd)
    cd = bpy.data.cameras.new(f'camera_{vn}'); cd.type = 'PERSP'; cd.lens_unit = 'FOV'; cd.sensor_fit = 'AUTO'
    cd.angle = math.radians(cam['fov_deg']); cd.clip_start = 0.01; cd.clip_end = 100
    co = bpy.data.objects.new(f'camera_{vn}', cd)
    R = Matrix(((right.x, up.x, -fwd.x), (right.y, up.y, -fwd.y), (right.z, up.z, -fwd.z)))
    co.matrix_world = Matrix.Translation(o) @ R.to_4x4()
    sc.collection.objects.link(co)
    if vn == 'V0' or sc.camera is None:
        sc.camera = co
for p in refs:                                           # ReST-GI full-depth reference images (linear HDR) -> OpenEXR next to the .blend
    a = np.load(p).astype(np.float32); h, w = a.shape[:2]
    nm = p.replace(chr(92), '/').split('/')[-1].replace('.npy', '')
    im = bpy.data.images.new(nm, w, h, alpha=True, float_buffer=True)
    rgba = np.concatenate([a, np.ones((h, w, 1), np.float32)], -1)[::-1].ravel()
    im.pixels.foreach_set(rgba)
    im.filepath_raw = p.replace('.npy', '.exr'); im.file_format = 'OPEN_EXR'; im.save()
    print('EXR', im.filepath_raw)
bpy.ops.wm.save_as_mainfile(filepath=out, compress=True)
print('SAVED', out)
