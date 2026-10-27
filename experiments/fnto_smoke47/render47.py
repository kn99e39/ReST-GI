"""WL47 Stage B renderer references (Blender / Cycles 3.5.1 CPU, bpy module).

MIR: a closed black box with a rough-mirror floor and four ceiling windows (protocol stage_b_mir); outputs are port-mean
exiting radiances seen by orthographic cameras along the 8 output directions (direct boundary queries, 64 x 64 px).
SSS: the blend's own Lego 'Yellow' material on a plane with a shadow-only half-plane occluder (protocol stage_b_sss).

    RNA_DIR=<rna repo> python render47.py -- mir_smoke <out>
    RNA_DIR=<rna repo> python render47.py -- mir_basis <out> <rep 0|1>
    RNA_DIR=<rna repo> python render47.py -- mir_fields <out>
    RNA_DIR=<rna repo> python render47.py -- sss <out>
"""
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "fnto_lego_r1")); sys.path.insert(0, str(HERE.parent / "fnto_lego"))
import bpy  # noqa: E402
from mathutils import Vector  # noqa: E402

import smoke47_core as K  # noqa: E402

PM = K.PM; PS = K.PROTO["stage_b_sss"]; RES = 64; OSCALE = 1.3; HB = K.MIR_H


# ---- helpers ---------------------------------------------------------------------------------------------------------
def mat_simple(name, kind, c, rough=0.0):
    m = bpy.data.materials.new(name); m.use_nodes = True; nt = m.node_tree; nt.nodes.clear()
    o = nt.nodes.new("ShaderNodeOutputMaterial")
    d = nt.nodes.new("ShaderNodeBsdfDiffuse" if kind == "diffuse" else "ShaderNodeBsdfGlossy")
    if kind != "diffuse":
        d.distribution = "GGX"
    d.inputs["Color"].default_value = (c, c, c, 1); d.inputs["Roughness"].default_value = rough
    nt.links.new(d.outputs[0], o.inputs["Surface"]); return m


def quad(name, v, mat=None):
    me = bpy.data.meshes.new(name); me.from_pydata([tuple(map(float, p)) for p in v], [], [(0, 1, 2, 3)]); me.update()
    ob = bpy.data.objects.new(name, me); bpy.context.scene.collection.objects.link(ob)
    if mat is not None:
        me.materials.append(mat)
    return ob


def hquad(x0, x1, y0, y1, z):
    return [(x0, y0, z), (x1, y0, z), (x1, y1, z), (x0, y1, z)]


def shadow_only(ob):
    ob.visible_camera = False; ob.visible_diffuse = False; ob.visible_glossy = False
    ob.visible_transmission = False; ob.visible_volume_scatter = False; ob.visible_shadow = True


def add_sun(name, travel, strength):
    d = bpy.data.lights.new(name, "SUN"); d.energy = float(strength); d.angle = math.radians(PM["sun_angle_deg"])
    ob = bpy.data.objects.new(name, d); bpy.context.scene.collection.objects.link(ob)
    ob.rotation_mode = "QUATERNION"; ob.rotation_quaternion = Vector(tuple(travel)).to_track_quat("-Z", "Y"); return ob


def add_cam(name, wo, target, dist, scale):
    cd = bpy.data.cameras.new(name); cd.type = "ORTHO"; cd.ortho_scale = scale
    ob = bpy.data.objects.new(name, cd); bpy.context.scene.collection.objects.link(ob)
    ob.location = Vector(tuple(np.asarray(target) + dist * np.asarray(wo)))
    ob.rotation_mode = "QUATERNION"; ob.rotation_quaternion = Vector(tuple(-np.asarray(wo))).to_track_quat("-Z", "Y"); return ob


def setup(sc, rx, ry, spp, seed, bounces=4):
    r = sc.render; r.engine = "CYCLES"; r.resolution_x = rx; r.resolution_y = ry; r.resolution_percentage = 100
    r.image_settings.file_format = "OPEN_EXR"; r.image_settings.color_depth = "32"; r.image_settings.color_mode = "RGBA"
    r.film_transparent = False
    c = sc.cycles; c.samples = spp; c.use_adaptive_sampling = False; c.use_denoising = False; c.device = "CPU"
    c.sample_clamp_direct = 0.0; c.sample_clamp_indirect = 0.0; c.pixel_filter_type = "GAUSSIAN"; c.filter_width = 0.01
    c.use_light_tree = False; c.max_bounces = bounces; c.glossy_bounces = bounces; c.diffuse_bounces = bounces; c.seed = seed


def render(sc, cam, path):
    sc.camera = cam; sc.render.filepath = str(path); bpy.context.view_layer.update()
    bpy.ops.render.render(write_still=True)
    img = bpy.data.images.load(str(path) + ".exr"); a = np.empty(len(img.pixels), np.float32); img.pixels.foreach_get(a)
    out = a.reshape(sc.render.resolution_y, sc.render.resolution_x, 4)[..., :3].copy(); bpy.data.images.remove(img)
    Path(str(path) + ".exr").unlink(); return out                                         # rows bottom -> top


def pixel_hits(cam, rx, ry, z):
    """World hit points of the pixel-centre rays of an orthographic camera with the plane at height z [ry, rx, 3]."""
    M = np.array(cam.matrix_world); right, up, back, loc = M[:3, 0], M[:3, 1], M[:3, 2], M[:3, 3]
    right = right / np.linalg.norm(right); up = up / np.linalg.norm(up); f = -back / np.linalg.norm(back)
    s = cam.data.ortho_scale; px = s / max(rx, ry)
    u = ((np.arange(rx) + 0.5) - rx / 2) * px; v = ((np.arange(ry) + 0.5) - ry / 2) * px
    o = loc[None, None] + u[None, :, None] * right + v[:, None, None] * up
    t = (z - o[..., 2]) / f[2]; return o + t[..., None] * f


def win_masks(cam):
    p = pixel_hits(cam, RES, RES, HB); m = 0.005
    return [(np.abs(p[..., 0] - c) <= K.MIR_WW / 2 - m) & (np.abs(p[..., 1]) <= 0.2 - m) for c in K.MIR_WIN]


# ---- MIR domain ------------------------------------------------------------------------------------------------------
def build_mir(open_ports, irr=False):
    bpy.ops.wm.read_factory_settings(use_empty=True); sc = bpy.context.scene
    w = bpy.data.worlds.new("w47"); w.use_nodes = True; w.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.0; sc.world = w
    black = mat_simple("black", "diffuse", 0.0); mirror = mat_simple("mirror", "glossy", 0.9, 0.25); white = mat_simple("white", "diffuse", 1.0)
    X0, X1, Y0, Y1 = -0.55, 0.55, -0.2, 0.2
    quad("floor", hquad(X0, X1, Y0, Y1, 0.0), mirror)
    quad("wall_x0", [(X0, Y0, 0), (X0, Y1, 0), (X0, Y1, HB), (X0, Y0, HB)], black)
    quad("wall_x1", [(X1, Y0, 0), (X1, Y1, 0), (X1, Y1, HB), (X1, Y0, HB)], black)
    quad("wall_y0", [(X0, Y0, 0), (X1, Y0, 0), (X1, Y0, HB), (X0, Y0, HB)], black)
    quad("wall_y1", [(X0, Y1, 0), (X1, Y1, 0), (X1, Y1, HB), (X0, Y1, HB)], black)
    edges = [X0] + [e for c in K.MIR_WIN for e in (c - K.MIR_WW / 2, c + K.MIR_WW / 2)] + [X1]
    for i in range(0, len(edges), 2):
        quad(f"ceil{i}", hquad(edges[i], edges[i + 1], Y0, Y1, HB), black)
    for q, c in enumerate(K.MIR_WIN):
        if q not in open_ports:
            shadow_only(quad(f"mask{q}", hquad(c - 0.06, c + 0.06, -0.25, 0.25, HB + 0.0005)))
        elif irr:
            quad(f"irr{q}", hquad(c - K.MIR_WW / 2, c + K.MIR_WW / 2, Y0, Y1, HB), white)
    return sc


def mir_cams():
    return [add_cam(f"cam{k}", K.mir_out(t), (0, 0, HB), 3.0, OSCALE) for k, t in enumerate(K.MIR_T)]


def lum(img):
    return img.mean(-1)


def mir_field(open_ports, suns, spp, seed, tag, work):
    """suns: [(signed tan, E)]; returns port values [8 cams, 4 ports] and images [8, 64, 64, 3]."""
    sc = build_mir(open_ports)
    for i, (t, E) in enumerate(suns):
        add_sun(f"sun{i}", K.mir_travel(t), E / K.mir_cos(t))
    cams = mir_cams(); setup(sc, RES, RES, spp, seed)
    vals = np.zeros((len(cams), 4)); imgs = np.zeros((len(cams), RES, RES, 3), np.float32)
    for k, cam in enumerate(cams):
        im = render(sc, cam, work / f"{tag}_c{k}"); imgs[k] = im; L = lum(im)
        for q, m in enumerate(win_masks(cam)):
            vals[k, q] = L[m].mean()
    return vals, imgs


def mir_irr(open_ports, suns, spp, seed, work):
    sc = build_mir(open_ports, irr=True)
    for i, (t, E) in enumerate(suns):
        add_sun(f"sun{i}", K.mir_travel(t), E / K.mir_cos(t))
    cam = add_cam("top", np.array([0.0, 0, 1]), (0, 0, HB), 3.0, OSCALE); setup(sc, RES, RES, spp, seed)
    im = render(sc, cam, work / "irr"); L = lum(im); ms = win_masks(cam)
    return [float(math.pi * L[m].mean()) for m in ms], im, [int(m.sum()) for m in ms]


# ---- SSS (Lego Yellow) -----------------------------------------------------------------------------------------------
def build_sss(white=False):
    import r1_common as C
    bpy.ops.wm.open_mainfile(filepath=str(C.B4 / "scene/lego_b4_T0_s1.blend")); sc = bpy.context.scene; sc.frame_current = 1
    bg = sc.world.node_tree.nodes.get("Background")
    if bg is not None:
        sc.world.node_tree.nodes.remove(bg)
    sc.use_nodes = False
    for ob in list(sc.objects):
        ob.hide_render = True
    y = bpy.data.materials["Yellow"].copy(); y.name = "Yellow_wl47_nodisp"
    for l in list(y.node_tree.links):
        if l.to_socket.name == "Displacement" and l.to_node.bl_idname == "ShaderNodeOutputMaterial":
            y.node_tree.links.remove(l)
    mat = mat_simple("white47", "diffuse", 1.0) if white else y
    quad("plane47", hquad(-1.3, 1.7, -0.5, 0.5, 0.0), mat)
    shadow_only(quad("occ47", hquad(0.0, 2.5, -1.5, 1.5, 0.3)))
    return sc


def sss_sun(field):
    f = PS["fields"][field]; th = math.radians(f["theta_deg"])
    return add_sun(f"sun_{field}", (0.0, -math.sin(th), -math.cos(th)), f["strength"])


def sss_cams():
    return {k: add_cam(f"sss_{k}", np.array(v), (0.1, 0.0, 0.0), 2.0, 0.4) for k, v in PS["cameras"].items()}


# ---- modes -----------------------------------------------------------------------------------------------------------
def main():
    a = sys.argv[sys.argv.index("--") + 1:]; mode, out = a[0], Path(a[1]); out.mkdir(parents=True, exist_ok=True)
    work = out / "work"; work.mkdir(exist_ok=True); t0 = time.time(); R = PM["render"]
    if mode == "mir_smoke":
        vals, imgs = mir_field([0], [(0.75, 1.0)], 64, 1, "smoke", work)
        E, imi, npx = mir_irr([0], [(0.75, 1.0)], 16, 2, work)
        np.savez(out / "mir_smoke.npz", vals=vals, imgs=imgs, irr=imi)
        print(json.dumps({"vals": vals.round(5).tolist(), "E": E, "npx": npx, "s": time.time() - t0}))
    elif mode == "mir_basis":
        rep = int(a[2]); V = np.zeros((2, len(K.MIR_T), len(K.MIR_T), 4)); IM = np.zeros((2, len(K.MIR_T), len(K.MIR_T), RES, RES, 3), np.float16)
        for p in (0, 1):
            for i, t in enumerate(K.MIR_T):
                b = p * len(K.MIR_T) + i
                V[p, i], IM[p, i] = mir_field([p], [(float(t), PM["basis_E"])], R["spp_basis"], 4710 + b + 100 * rep, f"b{b}", work)
                print(f"basis rep {rep} p {p} t {t} {time.time() - t0:.0f}s", flush=True)
        np.savez(out / f"mir_basis_rep{rep}.npz", vals=V, imgs=IM, spp=R["spp_basis"], wall_s=time.time() - t0)
    elif mode == "mir_fields":
        rec = {}; i = 0; arrs = {}
        for pn, pr in PM["pairs"].items():
            for side in ("A", "B"):
                suns = [tuple(s) for s in pr[side]]
                for rep in (0, 1):
                    v, im = mir_field(pr["open"], suns, R["spp_fields"], 4730 + i + 100 * rep, f"{pn}{side}{rep}", work)
                    arrs[f"{pn}_{side}_rep{rep}_vals"] = v; arrs[f"{pn}_{side}_rep{rep}_imgs"] = im.astype(np.float16)
                E, imi, npx = mir_irr(pr["open"], suns, R["spp_irr"], 4750 + i, work)
                rec[f"{pn}_{side}"] = {"E_ports": E, "irr_pixels": npx}; arrs[f"{pn}_{side}_irr"] = imi.astype(np.float16)
                i += 1; print(f"field {pn}{side} {time.time() - t0:.0f}s", flush=True)
        np.savez(out / "mir_fields.npz", **arrs); rec["wall_s"] = time.time() - t0
        (out / "mir_fields_irr.json").write_text(json.dumps(rec, indent=1))
    elif mode == "sss":
        arrs = {}; rec = {"blender": bpy.app.version_string}
        for fi, field in enumerate(("A", "B")):
            for rep in (0, 1):
                sc = build_sss(); sss_sun(field); cams = sss_cams(); setup(sc, 64, 8, PS["render"]["spp"], 4760 + fi + 100 * rep, bounces=32)
                for k, cam in cams.items():
                    arrs[f"{field}_rep{rep}_{k}"] = render(sc, cam, work / f"sss_{field}{rep}_{k}")
                    arrs[f"x_{k}"] = pixel_hits(cam, 64, 8, 0.0)[..., 0]
                print(f"sss {field} rep {rep} {time.time() - t0:.0f}s", flush=True)
            sc = build_sss(white=True); sss_sun(field); cams = sss_cams(); setup(sc, 64, 8, PS["render"]["spp_irr"], 4770 + fi, bounces=32)
            arrs[f"{field}_irr_top"] = render(sc, cams["top"], work / f"sss_irr_{field}")
        ss = [n for n in bpy.data.materials["Yellow"].node_tree.nodes if n.bl_idname == "ShaderNodeSubsurfaceScattering"][0]
        rec.update({"falloff": ss.falloff, "radius": list(ss.inputs["Radius"].default_value), "scale": ss.inputs["Scale"].default_value,
                    "wall_s": time.time() - t0})
        np.savez(out / "sss.npz", **arrs); (out / "sss_meta.json").write_text(json.dumps(rec, indent=1))
    print(f"done {mode} {time.time() - t0:.1f}s")


main()
