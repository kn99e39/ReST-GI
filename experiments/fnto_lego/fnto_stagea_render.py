"""FNTO Stage A: isolated-domain first-intrinsic-event probe renders (WSL RNA venv bpy 3.5.1, CPU Cycles).

    python fnto_stagea_render.py -- <out dir> <split> <n images> <seed> [--res 512] [--spp 512] [--superpose]

Each image: the official Lego (domain A) and, unless removed, the red box (domain B, randomly translated, never within
0.25 m of its held-out T1 placement), lit by ONE sun of the generator's convention (energy 5, angle 0) from a random direction
(70 % upper / 30 % lower hemisphere, uniform within each), with 0..10 camera-invisible, shadow-only spheres ("masks") that create spatially
localised incident patterns. Light paths are limited to the *first intrinsic event* (fnto_semantics: max 1, diffuse 0,
glossy 0, transmission 1, transparent 0): direct light at the camera hit, Christensen-Burley SSS (renderer semantics:
entry anywhere on the same object), and one traversal of a glass solid. No free-space interreflection is rendered, so the
other domain and the masks act on these images only as occluders of the incident sunlight.

Per image, written to <out>/<split>/img_<i>.npz (alpha > 0 pixels only): pix (flat index), pos (world, fp32), nrm, cam,
col (RGB radiance), vis (sun visibility of the pixel, the generator's white-diffuse direct pass > 0); and the image
configuration in <out>/<split>/configs.json (sun direction, camera matrix, box offset or null, masks).
--superpose: renders linearity probes instead (pairs of suns alone and together; same masks / box / camera per triple).
Camera protocol: the generator's (radius 2.75, look-at origin, FOV 45 deg, up +z); 70 % upper / 30 % lower hemisphere.
"""

import json
import math
import sys
import time
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix, Vector

U = Path("/mnt/c/Projects/ReST-GI/external/relightable-neural-assets")
sys.path.insert(0, str(U)); sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import exr, ops  # noqa: E402  (RNA utilities, unchanged)

import fnto_common as C  # noqa: E402

BLEND = C.B4 / "scene/lego_b4_T0_s0.blend"
AOVS = {"pos": "position0001.exr", "nrm": "normal0001.exr", "cam": "camera_dir0001.exr"}


def strip_lights(sc):
    bg = sc.world.node_tree.nodes.get("Background")
    if bg is not None:
        sc.world.node_tree.nodes.remove(bg)
    for o in sc.objects:
        if o.type in ("LIGHT", "CAMERA"):
            o.hide_render = True


def first_event(c):
    c.max_bounces = 1; c.diffuse_bounces = 0; c.glossy_bounces = 0; c.transmission_bounces = 1
    c.volume_bounces = 0; c.transparent_max_bounces = 0


def look(cam, pos):
    m = ops.get_look_at(np.asarray(pos, np.float64), np.zeros(3), np.array([0.0, 0, 1]))
    loc, rot, sca = Matrix(m.tolist()).decompose()
    cam.location = loc; cam.rotation_euler = rot.to_euler(); cam.scale = sca
    return m


def set_sun(sun, d):
    sun.rotation_mode = "QUATERNION"; sun.rotation_quaternion = Vector((0, 0, 1)).rotation_difference(Vector(d).normalized())


def sample_config(rng, i, box_t0_centre):
    z = rng.uniform(0, 0.9999) if rng.random() < 0.7 else -rng.uniform(0, 0.9999)
    ph = rng.uniform(0, 2 * np.pi); cr = math.sqrt(max(0.0, 1 - z * z))
    cam = (np.array([cr * math.cos(ph), cr * math.sin(ph), z]) * 2.75).tolist()
    s = rng.normal(size=3); s /= np.linalg.norm(s)
    s[2] = abs(s[2]) if rng.random() < 0.7 else -abs(s[2])      # 70 % upper / 30 % lower hemisphere sun
    box = None
    if rng.random() > 0.25:
        while True:
            c = np.array([rng.uniform(-0.9, 0.9), rng.uniform(-1.2, 1.2), rng.uniform(-0.06, 0.5)])
            d = c - box_t0_centre
            if np.linalg.norm(d - C.DELTA) > 0.25:
                box = d.tolist(); break
    masks = []
    if rng.random() > 0.3:
        for _ in range(int(rng.integers(1, 11))):
            masks.append({"c": [rng.uniform(-0.7, 0.7), rng.uniform(-1.0, 1.0), rng.uniform(-0.3, 0.9)], "r": rng.uniform(0.03, 0.2)})
    return {"i": i, "sun": s.tolist(), "cam_pos": cam, "box_offset": box, "masks": masks}


class Scene:
    def __init__(self, res, spp):
        bpy.ops.wm.open_mainfile(filepath=str(BLEND))
        sc = self.sc = bpy.context.scene
        sc.frame_current = 1
        strip_lights(sc)
        data = bpy.data.lights.new("FNTO_sun", "SUN"); data.energy = C.E_SUN; data.angle = 0.0
        self.sun = bpy.data.objects.new("FNTO_sun", data); sc.collection.objects.link(self.sun)
        d2 = bpy.data.lights.new("FNTO_sun2", "SUN"); d2.energy = C.E_SUN; d2.angle = 0.0
        self.sun2 = bpy.data.objects.new("FNTO_sun2", d2); sc.collection.objects.link(self.sun2); self.sun2.hide_render = True
        cd = bpy.data.cameras.new("FNTO_cam"); cd.type = "PERSP"; cd.lens_unit = "FOV"; cd.angle = math.radians(45)
        self.cam = bpy.data.objects.new("FNTO_cam", cd); sc.collection.objects.link(self.cam); sc.camera = self.cam
        self.box = bpy.data.objects["B_red_box"]; self.box_loc0 = Vector(self.box.location)
        r = sc.render
        r.engine = "CYCLES"; r.resolution_x = res; r.resolution_y = res; r.resolution_percentage = 100
        r.image_settings.file_format = "OPEN_EXR"; r.image_settings.color_depth = "32"; r.image_settings.color_mode = "RGBA"
        r.film_transparent = True; r.filepath = "temp_beauty"; r.threads_mode = "AUTO"
        c = sc.cycles
        c.samples = spp; c.use_adaptive_sampling = False; c.use_denoising = False; c.sample_clamp_direct = 20
        c.use_light_tree = False; c.device = "CPU"; sc.world.cycles.query_filename = ""
        first_event(c)
        self.masks = []
        self.mask_mat = bpy.data.materials.new("FNTO_mask")

    def apply(self, cfg, sun2=None):
        sc = self.sc
        for o in self.masks:
            bpy.data.objects.remove(o, do_unlink=True)
        self.masks = []
        for k, m in enumerate(cfg["masks"]):
            bpy.ops.mesh.primitive_uv_sphere_add(radius=m["r"], location=m["c"], segments=32, ring_count=16)
            o = bpy.context.object; o.name = f"FNTO_mask_{k}"; o.data.materials.append(self.mask_mat)
            o.visible_camera = False; o.visible_diffuse = False; o.visible_glossy = False
            o.visible_transmission = False; o.visible_volume_scatter = False; o.visible_shadow = True
            self.masks.append(o)
        if cfg["box_offset"] is None:
            self.box.hide_render = True
        else:
            self.box.hide_render = False; self.box.location = self.box_loc0 + Vector(cfg["box_offset"])
        set_sun(self.sun, cfg["sun"])
        if sun2 is not None:
            self.sun2.hide_render = False; set_sun(self.sun2, sun2)
        else:
            self.sun2.hide_render = True
        cfg["cam_mtx"] = look(self.cam, cfg["cam_pos"]).tolist()
        bpy.context.view_layer.update()

    def render(self):
        bpy.context.scene.update_render_engine()
        bpy.ops.render.render(write_still=True)
        img = exr.read("temp_beauty.exr")
        h, w, _ = img.shape
        return img.reshape(h * w, -1)


def white_diffuse_visibility_pass(scn):
    """The generator's diffuse_direct procedure: all materials white diffuse, no bounces, 1 spp, opaque film."""
    sc = scn.sc
    for o in bpy.data.objects:
        if o.type != "MESH" or o.name.startswith("FNTO_mask"):
            continue
        for s in o.material_slots:
            m = s.material
            if m is None or m.name == "FNTO_mask":
                continue
            nt = m.node_tree; nt.nodes.clear()
            d = nt.nodes.new("ShaderNodeBsdfDiffuse"); d.inputs["Color"].default_value = (1, 1, 1, 1)
            o_ = nt.nodes.new("ShaderNodeOutputMaterial"); nt.links.new(d.outputs["BSDF"], o_.inputs["Surface"])
    sc.cycles.samples = 1; sc.cycles.max_bounces = 0
    sc.render.film_transparent = False; sc.render.image_settings.color_mode = "RGB"; sc.render.filepath = "temp_vis"
    sc.use_nodes = False


def main():
    a = sys.argv[sys.argv.index("--") + 1:]
    out = Path(a[0]); split = a[1]; n = int(a[2]); seed = int(a[3])
    res = int(a[a.index("--res") + 1]) if "--res" in a else 512
    spp = int(a[a.index("--spp") + 1]) if "--spp" in a else 512
    superpose = "--superpose" in a
    od = out / split; od.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    box_c = np.asarray(C.SCENE["T0_centre"], np.float64)
    scn = Scene(res, spp)
    cfgs, jobs = [], []
    for i in range(n):
        cfg = sample_config(rng, i, box_c)
        if superpose:
            s2 = rng.normal(size=3); s2 /= np.linalg.norm(s2)
            for part, (sa, sb) in enumerate(((cfg["sun"], None), (s2.tolist(), None), (cfg["sun"], s2.tolist()))):
                c2 = dict(cfg); c2["i"] = 3 * i + part; c2["sun"] = sa; c2["sun2"] = sb; c2["triple"] = i; c2["part"] = part
                jobs.append(c2)
        else:
            cfg["sun2"] = None; jobs.append(cfg)
    t0 = time.time(); beauty = {}
    if (od / "configs_partial.json").exists():
        cfgs = json.loads((od / "configs_partial.json").read_text())
    for cfg in jobs:
        if (od / f"img_{cfg['i']:05d}.npz").exists() or (od / f"img_{cfg['i']:05d}_beauty.npz").exists():
            continue
        scn.apply(cfg, cfg["sun2"])
        t = time.time(); rgba = scn.render(); cfg["render_s"] = time.time() - t
        a_ = rgba[:, 3]; m = a_ > 0
        aov = {}
        for k, v in AOVS.items():
            x = exr.read(v); aov[k] = x.reshape(-1, x.shape[-1])[:, :3]
        beauty[cfg["i"]] = m
        np.savez(od / f"img_{cfg['i']:05d}_beauty.npz", pix=np.flatnonzero(m).astype(np.int32), pos=aov["pos"][m].astype(np.float32),
                 nrm=aov["nrm"][m].astype(np.float16), cam=aov["cam"][m].astype(np.float16), col=rgba[m, :3].astype(np.float32),
                 alpha=a_[m].astype(np.float16))
        cfgs.append(cfg)
        print(f"beauty {cfg['i']} {cfg['render_s']:.1f}s px={int(m.sum())} total={time.time() - t0:.0f}s", flush=True)
        (od / "configs_partial.json").write_text(json.dumps(cfgs))
    white_diffuse_visibility_pass(scn)
    for cfg in jobs:
        f = od / f"img_{cfg['i']:05d}_beauty.npz"
        if not f.exists():
            continue
        scn.apply(cfg, cfg["sun2"])
        bpy.context.scene.update_render_engine(); bpy.ops.render.render(write_still=True)
        v = exr.read("temp_vis.exr"); v = v.reshape(-1, v.shape[-1])[:, :3].sum(1) > 0
        z = dict(np.load(f)); z["vis"] = v[z["pix"]].astype(np.uint8)
        np.savez(od / f"img_{cfg['i']:05d}.npz", **z); f.unlink()
        print(f"vis {cfg['i']} lit={float(z['vis'].mean()):.3f}", flush=True)
    allc = json.loads((od / "configs_partial.json").read_text()) if (od / "configs_partial.json").exists() else cfgs
    byi = {c["i"]: c for c in allc}
    (od / "configs.json").write_text(json.dumps([byi[k] for k in sorted(byi)], indent=0))
    meta = {"split": split, "n": n, "seed": seed, "res": res, "spp": spp, "superpose": superpose, "blend": str(BLEND),
            "blend_sha16": C.sha16(BLEND), "bounces": "first intrinsic event (max 1, diffuse 0, glossy 0, transmission 1)",
            "clamp_direct": 20, "seconds": time.time() - t0}
    (od / "meta.json").write_text(json.dumps(meta, indent=1)); print("done", json.dumps(meta))


if __name__ == "__main__":
    main()
