"""WL42 Stage A r1: isolated-domain INTRINSIC-ALL probe renders (LabServer63 bpy 3.5.1, CPU Cycles).

    RNA_DIR=<rna repo> python r1_stagea_render.py -- <out dir> <split> <n images> <seed> [--res 512] [--spp 512]

Differences to the Batch 6 Stage A renderer (audit findings in brackets):
  * light paths: all bounces (128), but every non-camera ray that reaches a surface from OUTSIDE -- i.e. after a free-
    space segment -- is absorbed (Light Path / Backfacing masking; dielectrics continue on back-face hits). Kept: local
    BSDF at the camera hit, Burley SSS, every path that stays inside a glass solid incl. internal reflection and TIR.
    Removed: all editable free-space transport (C)                                                  [#5 / F4]
  * no sample clamping (direct 0, indirect 0)                                                      [#8 / F5]
  * point-sampled pixels: box filter 0.01 px, like the point-sampled held-out references              [#9]
  * probe families (independent spatial / directional excitation)                                    [#2 / F2]
      sun       one sun (70 % upper / 30 % lower hemisphere) x 0-10 shadow-only mask spheres     (train 25 %)
      spot      1-3 spot lights (cone 10-40 deg, hard edge) at 4-40 cm from a random Lego surface point, aimed at it;
                the target is drawn by AREA over the whole mesh, so cavities and undersides are excited     (35 %)
      broad     6-20 point lights at random free-space positions in and around the Lego (indirect-like, multi-sided)
                                                                                                    (25 %)
      multisun  2-5 suns from random directions (both hemispheres) x masks                          (15 %)
    Light positions are rejected if inside geometry (>= 3 of 6 axis rays hit a back face first) or < 2 cm from a surface.
    The box is translated at random (never within 0.25 m of its T1 placement) or absent (25 %).
Radiometry (r1_calib): sun strength S -> irradiance S; point / spot power P -> radiant intensity P / (4 pi) W/sr.
Per image: <out>/<split>/img_<i>.npz (alpha > 0 pixels: pix, pos, nrm, cam, col) and configs.json (lights, masks, box,
camera). Visibility per light is traced later on the PC (Mitsuba), at ports and at pixel positions.
"""

import json
import math
import os
import sys
import time
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix, Vector

HERE = Path(__file__).resolve().parent
sys.path.insert(0, os.environ.get("RNA_DIR", str(HERE.parents[1] / "external/relightable-neural-assets")))
sys.path.insert(0, str(HERE.parent / "fnto_lego")); sys.path.insert(0, str(HERE))
from utils import exr, ops  # noqa: E402

import r1_common as C  # noqa: E402

BLEND = C.B4 / "scene/lego_b4_T0_s0.blend"
AOVS = {"pos": "position0001.exr", "nrm": "normal0001.exr", "cam": "camera_dir0001.exr"}
FAMILY = {"train": (("sun", .25), ("spot", .35), ("broad", .25), ("multisun", .15)),
          "holdout": (("sun", .25), ("spot", .35), ("broad", .25), ("multisun", .15))}


def strip_lights(sc):
    bg = sc.world.node_tree.nodes.get("Background")
    if bg is not None:
        sc.world.node_tree.nodes.remove(bg)
    for o in sc.objects:
        if o.type in ("LIGHT", "CAMERA"):
            o.hide_render = True


def mask_intrinsic():
    for m in bpy.data.materials:
        if not m.use_nodes:
            continue
        nt = m.node_tree; outn = [n for n in nt.nodes if n.bl_idname == "ShaderNodeOutputMaterial"]
        if not outn or not outn[0].inputs["Surface"].is_linked:
            continue
        o = outn[0]; src = o.inputs["Surface"].links[0].from_socket
        lp = nt.nodes.new("ShaderNodeLightPath"); geo = nt.nodes.new("ShaderNodeNewGeometry")
        mx = nt.nodes.new("ShaderNodeMath"); mx.operation = "MAXIMUM"; nt.links.new(lp.outputs["Is Camera Ray"], mx.inputs[0])
        if any(n.bl_idname in ("ShaderNodeBsdfGlass", "ShaderNodeBsdfRefraction") for n in nt.nodes):
            nt.links.new(geo.outputs["Backfacing"], mx.inputs[1])
        else:
            mx.inputs[1].default_value = 0.0
        blk = nt.nodes.new("ShaderNodeEmission"); blk.inputs["Strength"].default_value = 0.0
        mix = nt.nodes.new("ShaderNodeMixShader")
        nt.links.new(mx.outputs[0], mix.inputs["Fac"]); nt.links.new(blk.outputs[0], mix.inputs[1]); nt.links.new(src, mix.inputs[2])
        nt.links.new(mix.outputs[0], o.inputs["Surface"])


class Scene:
    def __init__(self, res, spp):
        bpy.ops.wm.open_mainfile(filepath=str(BLEND))
        sc = self.sc = bpy.context.scene; sc.frame_current = 1; strip_lights(sc); mask_intrinsic()
        self.suns = [self._light("SUN", f"r1_sun{k}") for k in range(5)]
        self.points = [self._light("POINT", f"r1_pt{k}") for k in range(20)]
        self.spots = [self._light("SPOT", f"r1_spot{k}") for k in range(3)]
        cd = bpy.data.cameras.new("r1_cam"); cd.type = "PERSP"; cd.lens_unit = "FOV"; cd.angle = math.radians(45)
        self.cam = bpy.data.objects.new("r1_cam", cd); sc.collection.objects.link(self.cam); sc.camera = self.cam
        self.box = bpy.data.objects["B_red_box"]; self.box_loc0 = Vector(self.box.location)
        r = sc.render; r.engine = "CYCLES"; r.resolution_x = res; r.resolution_y = res; r.resolution_percentage = 100
        r.image_settings.file_format = "OPEN_EXR"; r.image_settings.color_depth = "32"; r.image_settings.color_mode = "RGBA"
        r.film_transparent = True; r.filepath = "temp_beauty"; r.threads_mode = "AUTO"
        c = sc.cycles
        c.samples = spp; c.use_adaptive_sampling = False; c.use_denoising = False; c.sample_clamp_direct = 0.0; c.sample_clamp_indirect = 0.0
        c.pixel_filter_type = "BOX"; c.filter_width = 0.01; c.use_light_tree = False; c.device = "CPU"; sc.world.cycles.query_filename = ""
        c.max_bounces = 128; c.diffuse_bounces = 128; c.glossy_bounces = 128; c.transmission_bounces = 128; c.transparent_max_bounces = 128
        c.volume_bounces = 0; c.caustics_reflective = True; c.caustics_refractive = True
        self.masks = []; self.mask_mat = bpy.data.materials.new("r1_mask")

    def _light(self, kind, name):
        d = bpy.data.lights.new(name, kind); d.energy = 0.0
        if kind == "SUN":
            d.angle = 0.0
        else:
            d.shadow_soft_size = 0.0
        if kind == "SPOT":
            d.spot_blend = 0.0
        o = bpy.data.objects.new(name, d); bpy.context.scene.collection.objects.link(o); o.hide_render = True
        return o

    def apply(self, cfg):
        for o in self.masks:
            bpy.data.objects.remove(o, do_unlink=True)
        self.masks = []
        for k, m in enumerate(cfg["masks"]):
            bpy.ops.mesh.primitive_uv_sphere_add(radius=m["r"], location=m["c"], segments=32, ring_count=16)
            o = bpy.context.object; o.name = f"r1_mask_{k}"; o.data.materials.append(self.mask_mat)
            o.visible_camera = False; o.visible_diffuse = False; o.visible_glossy = False
            o.visible_transmission = False; o.visible_volume_scatter = False; o.visible_shadow = True
            self.masks.append(o)
        if cfg["box_offset"] is None:
            self.box.hide_render = True
        else:
            self.box.hide_render = False; self.box.location = self.box_loc0 + Vector(cfg["box_offset"])
        for o in self.suns + self.points + self.spots:
            o.hide_render = True
        ks = kp = kq = 0
        for L in cfg["lights"]:
            if L["type"] == "sun":
                o = self.suns[ks]; ks += 1; o.data.energy = L["strength"]
                o.rotation_mode = "QUATERNION"; o.rotation_quaternion = Vector((0, 0, 1)).rotation_difference(Vector(L["dir"]))
            elif L["type"] == "point":
                o = self.points[kp]; kp += 1; o.data.energy = L["power"]; o.location = L["pos"]
            else:
                o = self.spots[kq]; kq += 1; o.data.energy = L["power"]; o.data.spot_size = 2 * math.acos(L["cos_half"])
                o.location = L["pos"]; o.rotation_mode = "QUATERNION"
                o.rotation_quaternion = Vector((0, 0, -1)).rotation_difference(Vector(L["axis"]))   # spot emits along -Z
            o.hide_render = False
        m = ops.get_look_at(np.asarray(cfg["cam_pos"], np.float64), np.zeros(3), np.array([0.0, 0, 1]))
        loc, rot, sca = Matrix(m.tolist()).decompose(); self.cam.location = loc; self.cam.rotation_euler = rot.to_euler(); self.cam.scale = sca
        cfg["cam_mtx"] = m.tolist(); self.sc.cycles.seed = int(cfg["i"]); bpy.context.view_layer.update()


class Sampler:
    def __init__(self, rng):
        z = np.load(C.MESH_A); self.V = z["V"]; self.F = z["F"]; self.rng = rng
        e1 = self.V[self.F[:, 1]] - self.V[self.F[:, 0]]; e2 = self.V[self.F[:, 2]] - self.V[self.F[:, 0]]
        cr = np.cross(e1, e2); a = 0.5 * np.linalg.norm(cr, axis=1); self.n = cr / np.maximum(2 * a[:, None], 1e-12)
        self.cdf = np.cumsum(a) / a.sum(); self.box_c = np.asarray(C.SCENE["T0_centre"], np.float64)

    def surface_point(self):
        t = int(np.searchsorted(self.cdf, self.rng.random())); r1, r2 = self.rng.random(2); s = math.sqrt(r1)
        p = (1 - s) * self.V[self.F[t, 0]] + s * (1 - r2) * self.V[self.F[t, 1]] + s * r2 * self.V[self.F[t, 2]]
        n = self.n[t] * (1 if self.rng.random() < 0.5 else -1)       # either side: two-sided / inner faces are excited too
        return p, n

    def unit(self):
        v = self.rng.normal(size=3); return v / np.linalg.norm(v)


def free_space_ok(p, depsgraph, sc):
    back, near = 0, 1e9
    for d in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)):
        hit, loc, nrm, _, _, _ = sc.ray_cast(depsgraph, Vector(p), Vector(d))
        if hit:
            dist = (loc - Vector(p)).length; near = min(near, dist)
            if nrm.dot(Vector(d)) > 0:
                back += 1
    return back < 3 and near > 0.02


def sample_config(S, i, split, rng, depsgraph, sc):
    fams, w = zip(*FAMILY.get(split, FAMILY["train"])); fam = fams[int(np.searchsorted(np.cumsum(w), rng.random()))]
    z = rng.uniform(0, 0.9999) if rng.random() < 0.7 else -rng.uniform(0, 0.9999); ph = rng.uniform(0, 2 * np.pi)
    cam = (np.array([math.sqrt(1 - z * z) * math.cos(ph), math.sqrt(1 - z * z) * math.sin(ph), z]) * 2.75).tolist()
    box = None
    if rng.random() > 0.25:
        while True:
            c = np.array([rng.uniform(-0.9, 0.9), rng.uniform(-1.2, 1.2), rng.uniform(-0.06, 0.5)]); d = c - S.box_c
            if np.linalg.norm(d - C.DELTA) > 0.25:
                box = d.tolist(); break
    lights, masks = [], []

    def sun_dir(upper_frac=0.7):
        v = S.unit(); v[2] = abs(v[2]) if rng.random() < upper_frac else -abs(v[2]); return (v / np.linalg.norm(v)).tolist()

    def free_point(lo, hi):
        for _ in range(200):
            p = rng.uniform(lo, hi)
            if free_space_ok(p, depsgraph, sc):
                return p.tolist()
        return None
    if fam in ("sun", "multisun"):
        for _ in range(1 if fam == "sun" else int(rng.integers(2, 6))):
            lights.append({"type": "sun", "dir": sun_dir(0.7 if fam == "sun" else 0.5), "strength": C.E_SUN if fam == "sun" else float(rng.uniform(1, 5))})
        if rng.random() > 0.3:
            for _ in range(int(rng.integers(1, 11))):
                masks.append({"c": [rng.uniform(-0.7, 0.7), rng.uniform(-1.0, 1.0), rng.uniform(-0.3, 0.9)], "r": rng.uniform(0.03, 0.2)})
    elif fam == "spot":
        for _ in range(int(rng.integers(1, 4))):
            for _ in range(200):
                tgt, n = S.surface_point(); u = S.unit()
                u = u if u @ n > 0.2 else (u + 1.5 * n); u /= np.linalg.norm(u)
                p = tgt + u * rng.uniform(0.04, 0.4)
                if free_space_ok(p, depsgraph, sc):
                    ax = (tgt - p) / np.linalg.norm(tgt - p)
                    lights.append({"type": "spot", "pos": p.tolist(), "axis": ax.tolist(), "cos_half": math.cos(math.radians(rng.uniform(5, 20))),
                                   "power": float(rng.uniform(5, 60))}); break
    else:
        lo = np.array([-0.65, -1.0, -0.25]); hi = np.array([0.65, 1.0, 0.95])
        for _ in range(int(rng.integers(6, 21))):
            p = free_point(lo, hi)
            if p is not None:
                lights.append({"type": "point", "pos": p, "power": float(rng.uniform(2, 25))})
    return {"i": i, "family": fam, "lights": lights, "masks": masks, "box_offset": box, "cam_pos": cam}


def main():
    a = sys.argv[sys.argv.index("--") + 1:]
    out = Path(a[0]); split = a[1]; n = int(a[2]); seed = int(a[3])
    res = int(a[a.index("--res") + 1]) if "--res" in a else 512
    spp = int(a[a.index("--spp") + 1]) if "--spp" in a else 512
    od = out / split; od.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed); scn = Scene(res, spp); S = Sampler(rng)
    dg = bpy.context.evaluated_depsgraph_get()
    cfgs = json.loads((od / "configs.json").read_text()) if (od / "configs.json").exists() else []
    done = {c["i"] for c in cfgs}
    t0 = time.time()
    for i in range(n):
        cfg = sample_config(S, i, split, rng, dg, scn.sc)                       # always drawn: identical sequence on resume
        if i in done:
            continue
        scn.apply(cfg)
        t = time.time(); bpy.context.scene.update_render_engine(); bpy.ops.render.render(write_still=True); cfg["render_s"] = time.time() - t
        img = exr.read("temp_beauty.exr"); rgba = img.reshape(-1, img.shape[-1]); m = rgba[:, 3] > 0
        aov = {}
        for k, v in AOVS.items():
            x = exr.read(v); aov[k] = x.reshape(-1, x.shape[-1])[:, :3]
        np.savez(od / f"img_{i:05d}.npz", pix=np.flatnonzero(m).astype(np.int32), pos=aov["pos"][m].astype(np.float32),
                 nrm=aov["nrm"][m].astype(np.float16), cam=aov["cam"][m].astype(np.float16), col=rgba[m, :3].astype(np.float32),
                 alpha=rgba[m, 3].astype(np.float16))
        cfgs.append(cfg); (od / "configs.json").write_text(json.dumps(cfgs))
        print(f"img {i} {cfg['family']} lights={len(cfg['lights'])} {cfg['render_s']:.1f}s px={int(m.sum())} total={time.time() - t0:.0f}s", flush=True)
    meta = {"split": split, "n": n, "seed": seed, "res": res, "spp": spp, "blend": str(BLEND), "paths": "intrinsic-all (free-space hits absorbed)",
            "clamp": [0, 0], "filter": "BOX 0.01 px", "max_bounces": 128, "radiometry": "sun S -> E = S; point/spot P -> I = P/(4 pi) W/sr (r1_calib)",
            "families": FAMILY.get(split, FAMILY["train"]), "seconds": time.time() - t0}
    (od / "meta.json").write_text(json.dumps(meta, indent=1)); print("done", json.dumps(meta))


if __name__ == "__main__":
    main()
