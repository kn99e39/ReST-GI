"""WL53 PRIMARY reference renders: Cycles 3.5.1 volume path tracing (protocol phys_ref53, decision revision R1).

Why Cycles volume (decided before primary evidence, see protocol): Mitsuba 3.5.1 (float32) skips a boundary layer of
~RayEpsilon (1 + |p|) = 8.9e-5 m every time a path crosses a null interface (surface spawn offset), which at sigma_t =
1000 m^-1 removes exp(-sigma_t delta / mu0) of the near-surface response (single scattering 0.914 / 0.833 at 0 / 60 deg,
semi-infinite 0.944 / 0.878 at albedo 0.5; bias proportional to sigma_t), and the floor of that bias cannot be scaled
away. Cycles 3.5.1 volume rendering (self-intersection by primitive id, no spawn offset) matches the exact H-function to
<= 0.13 % on the same cases. Volume closures are not subject to Cycles' PATH_RAY_DIFFUSE_ANCESTOR rule (BSSRDF only).

  vol  Volume Scatter (colour = albedo, density sigma_t) + Volume Absorption (colour = albedo, density sigma_t) inside a
       white Transparent BSDF: sigma_s = sigma_t albedo, sigma_a = sigma_t (1 - albedo), index-matched.
  rw   Subsurface Scattering RANDOM_WALK (diagnostic only: a BSSRDF, so the diffuse-ancestor rule applies).
  lam  Lambert (chain control / floor), black, mirror (GGX sharp).
Image rows are bottom-up (bpy); camera geometry is saved so p53_render.py cports can classify pixels with the same rays.

    python p53_cycles.py -- <out dir> [stage ...] [smoke]
stages: chain g1a g1b g1c g1d g1e g1f trunc scenes
"""
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import bpy  # noqa: E402
from mathutils import Vector  # noqa: E402

import p53_fixtures as FX  # noqa: E402
import p53_geom as G  # noqa: E402

SEEDS = (5301, 5302)
SEEDS8 = tuple(range(5301, 5309))
LD = (0.0181, 0.0058, 0.0018)
CFG = dict(spp_scene=1024, spp_h=8192, spp_furnace=1024, spp_cam=8192, spp_fish=2048, spp_recip=2048, px=0.001)
SMOKE = dict(spp_scene=16, spp_h=256, spp_furnace=16, spp_cam=64, spp_fish=16, spp_recip=16, px=0.002)
OUT = None
LOG = None


def log(*a):
    line = time.strftime("%H:%M:%S") + " " + " ".join(map(str, a)); print(line, flush=True); LOG.write(line + "\n"); LOG.flush()


def reset(spp, seed, world=0.0, bounces=1024):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene; sc.render.engine = "CYCLES"; c = sc.cycles
    c.device = "CPU"; c.samples = spp; c.seed = seed; c.use_denoising = False; c.use_adaptive_sampling = False
    c.max_bounces = bounces; c.diffuse_bounces = bounces; c.glossy_bounces = bounces; c.transmission_bounces = bounces; c.volume_bounces = bounces
    c.transparent_max_bounces = 1024; c.sample_clamp_direct = 0; c.sample_clamp_indirect = 0; c.caustics_reflective = True
    c.caustics_refractive = True; c.blur_glossy = 0.0; c.pixel_filter_type = "GAUSSIAN"; c.filter_width = 0.01
    sc.render.film_transparent = False
    w = bpy.data.worlds.new("w"); w.use_nodes = True; bg = w.node_tree.nodes["Background"]; bg.inputs[0].default_value = (1, 1, 1, 1); bg.inputs[1].default_value = world
    sc.world = w
    sc.view_settings.view_transform = "Standard"; sc.view_settings.look = "None"; sc.view_settings.exposure = 0.0
    return sc


def material(kind, rho=0.8, albedo=None):
    alb = FX.ALBEDO if albedo is None else albedo
    m = bpy.data.materials.new(kind); m.use_nodes = True; nt = m.node_tree; nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    if kind == "vol":
        tr = nt.nodes.new("ShaderNodeBsdfTransparent"); tr.inputs[0].default_value = (1, 1, 1, 1)
        scat = nt.nodes.new("ShaderNodeVolumeScatter"); ab = nt.nodes.new("ShaderNodeVolumeAbsorption"); add = nt.nodes.new("ShaderNodeAddShader")
        for n in (scat, ab):
            n.inputs["Color"].default_value = (*np.broadcast_to(alb, (3,)), 1); n.inputs["Density"].default_value = FX.SIGMA_T
        scat.inputs["Anisotropy"].default_value = FX.G_PHASE
        nt.links.new(scat.outputs[0], add.inputs[0]); nt.links.new(ab.outputs[0], add.inputs[1])
        nt.links.new(tr.outputs[0], out.inputs["Surface"]); nt.links.new(add.outputs[0], out.inputs["Volume"])
        m.cycles.homogeneous_volume = True
    elif kind == "rw":
        s = nt.nodes.new("ShaderNodeSubsurfaceScattering"); s.falloff = "RANDOM_WALK"
        s.inputs["Color"].default_value = (*alb, 1); s.inputs["Scale"].default_value = 1.0; s.inputs["Radius"].default_value = LD
        if "IOR" in s.inputs:
            s.inputs["IOR"].default_value = 1.01
        nt.links.new(s.outputs[0], out.inputs["Surface"])
    elif kind in ("lam", "black"):
        d = nt.nodes.new("ShaderNodeBsdfDiffuse"); v = rho if kind == "lam" else 0.0; d.inputs["Color"].default_value = (v, v, v, 1)
        nt.links.new(d.outputs[0], out.inputs["Surface"])
    elif kind == "mirror":
        g = nt.nodes.new("ShaderNodeBsdfGlossy")
        try:
            g.distribution = "SHARP"
        except TypeError:
            pass
        g.inputs["Color"].default_value = (1, 1, 1, 1); g.inputs["Roughness"].default_value = 0.0
        nt.links.new(g.outputs[0], out.inputs["Surface"])
    return m


def mesh_obj(name, V, F, mat):
    me = bpy.data.meshes.new(name); me.from_pydata([tuple(map(float, v)) for v in V], [], [tuple(map(int, f)) for f in F]); me.update()
    ob = bpy.data.objects.new(name, me); bpy.context.scene.collection.objects.link(ob); me.materials.append(mat); return ob


def sun(s, E=1.0, name="sun"):
    ld = bpy.data.lights.new(name, "SUN"); ld.energy = E; ld.angle = 0.0
    ob = bpy.data.objects.new(name, ld); bpy.context.scene.collection.objects.link(ob)
    ob.rotation_euler = (-Vector(s)).normalized().to_track_quat("Z", "Y").to_euler()
    return ob


def ortho(name, cam, res):
    cd = bpy.data.cameras.new(name); cd.type = "ORTHO"; cd.ortho_scale = cam["width"]; cd.clip_start = 1e-5; cd.clip_end = 100.0
    ob = bpy.data.objects.new(name, cd); bpy.context.scene.collection.objects.link(ob)
    ob.location = Vector(cam["origin"]); fwd = Vector(cam["target"]) - Vector(cam["origin"])
    ob.rotation_euler = fwd.normalized().to_track_quat("-Z", "Y").to_euler()
    bpy.context.view_layer.update(); mw = ob.matrix_world.to_3x3()
    return ob, dict(origin=list(ob.location), right=list(mw @ Vector((1, 0, 0))), up=list(mw @ Vector((0, 1, 0))), fwd=list(mw @ Vector((0, 0, -1))),
                    width=cam["width"], res=res, dir=list(cam.get("dir", [])))


def fisheye(name, origin, look):
    cd = bpy.data.cameras.new(name); cd.type = "PANO"
    for tgt in (cd, getattr(cd, "cycles", None)):
        if tgt is not None and hasattr(tgt, "panorama_type"):
            tgt.panorama_type = "FISHEYE_EQUIDISTANT"; tgt.fisheye_fov = math.pi
    cd.clip_start = 1e-5
    ob = bpy.data.objects.new(name, cd); bpy.context.scene.collection.objects.link(ob); ob.location = Vector(origin)
    ob.rotation_euler = Vector(look).normalized().to_track_quat("-Z", "Y").to_euler(); return ob


def render(sc, cam, res, spp, seed):
    path = OUT / "tmp_render"
    sc.camera = cam; r = sc.render; r.resolution_x = r.resolution_y = res; r.resolution_percentage = 100
    r.image_settings.file_format = "OPEN_EXR"; r.image_settings.color_depth = "32"; r.filepath = str(path)
    sc.cycles.samples = spp; sc.cycles.seed = seed; bpy.context.view_layer.update(); bpy.ops.render.render(write_still=True)
    img = bpy.data.images.load(str(path) + ".exr"); a = np.empty(len(img.pixels), np.float32); img.pixels.foreach_get(a)
    out = a.reshape(res, res, 4)[..., :3].copy(); bpy.data.images.remove(img); Path(str(path) + ".exr").unlink(); return out


def fisheye_E(img):
    """Irradiance on the plane facing the view direction, from an equidistant 180-degree fisheye image."""
    res = img.shape[0]; c = (np.arange(res) + 0.5) - res / 2; X, Y = np.meshgrid(c, c); r = np.hypot(X, Y); Rr = res / 2
    th = r / Rr * (math.pi / 2); k = (math.pi / 2) / Rr
    w = np.where(r < Rr, np.cos(th) * np.sinc(th / math.pi) * k * k, 0.0)
    return (img * w[..., None]).sum((0, 1))


def res_of(cam, cfg):
    return max(8, int(round(cam["width"] / cfg["px"])))


def build(sd, mk="vol", grouped=False):
    med = [(n, m, V) for n, k, m, V in sd["comps"] if k == "medium"]
    if grouped and len(med) > 1:
        V = np.concatenate([V for _, _, V in med]); off = np.cumsum([0] + [len(V_) for _, _, V_ in med])[:-1]
        mesh_obj("media", V, np.concatenate([m["F"] + o for (_, m, _), o in zip(med, off)]), material(mk))
    else:
        for n, m, V in med:
            mesh_obj(n, V, m["F"], material(mk))
    for n, k, m, V in sd["comps"]:
        if k == "lambert":
            mesh_obj(n, V, m["F"], material("lam", FX.FLOOR_RHO))
    if sd.get("plate"):
        Vp, Fp = FX.plate_mesh(); mesh_obj("plate", Vp, Fp, material("black"))


def render_scene(tag, sd, cfg, lights=None, grouped=False, mk="vol", cams=None, bounces=1024, seeds=SEEDS, spp=None):
    geo = {}
    for seed in seeds:
        sc = reset(spp or cfg["spp_scene"], seed, bounces=bounces); build(sd, mk, grouped)
        for i, (s, E) in enumerate(sd["lights"] if lights is None else lights):
            sun(s, E, f"sun{i}")
        for cam, c in sd["cams"].items():
            if cams and cam not in cams:
                continue
            res = res_of(c, cfg); ob, g = ortho(cam, c, res); geo[cam] = g
            t0 = time.time(); img = render(sc, ob, res, spp or cfg["spp_scene"], seed)
            np.save(OUT / f"cimg_{tag}_{cam}_{seed}.npy", img.astype(np.float32))
            log(f"{tag} {cam} seed {seed} {time.time() - t0:.0f}s")
    json.dump(geo, open(OUT / f"ccam_{tag}.json", "w"), indent=1)


# ------------------------------------------------------------------ stages


def st_chain(cfg):
    """Units chain control: Lambert 0.8 under the oblique sun, analytic rho E cos / pi; fisheye irradiance of a uniform
    world of radiance 1 = pi."""
    sc = reset(64, SEEDS[0]); fl = FX.floor_mesh(0.0, 10); mesh_obj("f", fl["V"], fl["F"], material("lam")); s = FX.scene_def("iso_L2")["lights"][0][0]; sun(s, 1.0)
    ob, _ = ortho("c", dict(origin=(0, 0, 0.1), target=(0, 0, 0), width=0.05), 16); img = render(sc, ob, 16, 64, SEEDS[0])
    sc = reset(256, SEEDS[0], world=1.0); fe = fisheye("fish", (0, 0, 0), (0, 0, -1)); Ef = fisheye_E(render(sc, fe, 128, 256, SEEDS[0]))
    rec = dict(lambert_L=float(img.mean()), lambert_analytic=0.8 * -s[2] / math.pi, fisheye_E_uniform=Ef.tolist(), fisheye_analytic=math.pi)
    json.dump(rec, open(OUT / "c_chain.json", "w"), indent=1); log("chain", rec)


def st_g1a(cfg):
    res = []
    V = np.array([[-1, -1, -0.5], [1, -1, -0.5], [1, 1, -0.5], [-1, 1, -0.5], [-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]], float)
    F = np.array([[0, 3, 2, 1], [4, 5, 6, 7], [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]])
    for alb in (0.5, 0.9, 0.99, 0.999):
        for th0 in (0.0, 60.0):
            s = (math.sin(math.radians(th0)), 0.0, -math.cos(math.radians(th0)))
            for th in (0.0, 40.0, 70.0):
                d = (0.0, math.sin(math.radians(th)), math.cos(math.radians(th))); px = []
                for seed in SEEDS:
                    sc = reset(cfg["spp_h"], seed); mesh_obj("cube", V, F, material("vol", albedo=(alb, alb, alb))); sun(s, 1.0)
                    ob, _ = ortho("c", dict(origin=tuple(0.5 * np.array(d)), target=(0, 0, 0), width=0.04), 8)
                    px.append(render(sc, ob, 8, cfg["spp_h"], seed)[..., 0].ravel())
                px = np.concatenate(px); exp = G.semi_infinite_radiance(alb, math.cos(math.radians(th)), math.cos(math.radians(th0)), 1.0)
                res.append(dict(albedo=alb, theta0=th0, theta=th, cycles=float(px.mean()), sem=float(px.std() / math.sqrt(len(px))), analytic=exp))
                log("g1a", alb, th0, th, round(px.mean() / exp, 4))
    json.dump(res, open(OUT / "c_g1a.json", "w"), indent=1)


def st_g1b(cfg):
    sd = FX.scene_def("iso_L1"); _, _, m, V = sd["comps"][0]
    for tag, alb in (("alb1", (1.0, 1.0, 1.0)), ("albW", FX.ALBEDO)):
        for seed in SEEDS:
            sc = reset(cfg["spp_furnace"], seed, world=1.0); mesh_obj("slab", V, m["F"], material("vol", albedo=alb)); geo = {}
            for cam in ("top", "otop"):
                c = sd["cams"][cam]; res = res_of(c, cfg); ob, geo[cam] = ortho(cam, c, res)
                np.save(OUT / f"cimg_furnace_{tag}_{cam}_{seed}.npy", render(sc, ob, res, cfg["spp_furnace"], seed))
            json.dump(geo, open(OUT / f"ccam_furnace_{tag}.json", "w"))
        log("g1b", tag)


def st_g1c(cfg):
    sd = FX.scene_def("iso_L1")
    A = (math.sin(math.radians(30)), 0.0, -math.cos(math.radians(30))); B = (0.0, math.sin(math.radians(50)), math.cos(math.radians(50)))
    for tag, lights in (("zero", []), ("A", [(A, 1.0)]), ("B", [(B, 1.0)]), ("AB", [(A, 1.0), (B, 1.0)])):
        render_scene(f"sup_{tag}", sd, cfg, lights=lights, cams=("top", "bot"))


def st_g1d(cfg):
    """Camera-vertex irradiance at the patch centre (fisheye) vs diffuse-ancestor irradiance (pi L_patch / rho) vs mirror."""
    res = {}
    for config in ("above_single", "below_pair"):
        if config == "above_single":
            sd = FX.scene_def("iso_L1"); s = (math.sin(math.radians(45)), 0.0, -math.cos(math.radians(45))); zt = 0.005; half = 0.01
        else:
            sd = FX.scene_def("gap4_below"); s = (0.0, 0.0, 1.0); zt = 0.012; half = 0.05
        zp = zt + 0.05
        for mk in (("vol", "rw", "lam") if config == "above_single" else ("vol", "rw")):
            key = f"{config}_{mk}"; r = res.setdefault(key, dict(E_cv=[], E_da=[], L_mirror=[], L_direct=[]))
            for seed in SEEDS8:
                for patch in ("diffuse", "mirror"):
                    sc = reset(cfg["spp_cam"], seed)
                    if mk == "lam":
                        _, _, m, V = sd["comps"][0]; mesh_obj("slab", V, m["F"], material("lam"))
                    else:
                        build(sd, mk, grouped=True)
                    sun(s, 1.0)
                    Vq = np.array([(-half, -half, zp), (half, -half, zp), (half, half, zp), (-half, half, zp)])
                    mesh_obj("patch", Vq, np.array([[0, 2, 1], [0, 3, 2]]), material("lam" if patch == "diffuse" else "mirror"))   # normal -z
                    pc, _ = ortho("pcam", dict(origin=(0, 0, zt + 0.02), target=(0, 0, 1), width=0.012), 24)
                    if patch == "diffuse":
                        img = render(sc, pc, 24, cfg["spp_cam"], seed); r["E_da"].append((math.pi * img[2:22, 2:22].reshape(-1, 3).mean(0) / 0.8).tolist())
                        fe = fisheye("fish", (0, 0, zp - 1e-4), (0, 0, -1)); img = render(sc, fe, 128, cfg["spp_fish"], seed)
                        r["E_cv"].append(fisheye_E(img).tolist())
                    else:
                        img = render(sc, pc, 24, cfg["spp_cam"], seed); r["L_mirror"].append(img[2:22, 2:22].reshape(-1, 3).mean(0).tolist())
                        tc, _ = ortho("tcam", dict(origin=(0, 0, zt + 0.02), target=(0, 0, 0), width=0.012), 24)
                        img = render(sc, tc, 24, cfg["spp_cam"], seed); r["L_direct"].append(img[2:22, 2:22].reshape(-1, 3).mean(0).tolist())
            log("g1d", key, {k: np.round(np.mean(v, 0), 5).tolist() for k, v in r.items() if v})
    json.dump(res, open(OUT / "c_g1d.json", "w"), indent=1)


def st_g1e(cfg):
    for name in ("gap4", "gap4_below"):
        sd = FX.scene_def(name)
        for grouped in (True, False):
            render_scene(f"grp_{name}_vol_{'one' if grouped else 'two'}", sd, cfg, grouped=grouped)
    sd = FX.scene_def("gap4_below")
    for grouped in (True, False):                                                   # diagnostic: random-walk BSSRDF
        render_scene(f"grp_gap4_below_rw_{'one' if grouped else 'two'}", sd, cfg, grouped=grouped, mk="rw", cams=("top", "gapdown", "gapup"))


def st_g1f(cfg):
    """Reciprocity of the reference: total radiant intensity toward c under a sun travelling along s equals the intensity
    toward -s under a sun travelling along -c (I(c|s) = I(-s|-c)); orthographic camera covering the whole slab."""
    sd = FX.scene_def("iso_L1"); _, _, m, V = sd["comps"][0]; res = {}
    cases = {"R": ((math.sin(math.radians(30)), 0.0, -math.cos(math.radians(30))), (0.0, math.sin(math.radians(60)), math.cos(math.radians(60)))),
             "T": ((math.sin(math.radians(30)), 0.0, -math.cos(math.radians(30))), (0.0, math.sin(math.radians(30)), -math.cos(math.radians(30))))}
    for name, (s, c) in cases.items():
        r = {}
        for tag, (ss, cc) in (("fwd", (s, c)), ("rev", (tuple(-np.array(c)), tuple(-np.array(s))))):
            vals = []
            for seed in SEEDS8:
                sc = reset(cfg["spp_recip"], seed); mesh_obj("slab", V, m["F"], material("vol")); sun(ss, 1.0)
                w = 0.16; resx = res_of(dict(width=w), cfg); ob, _ = ortho("c", dict(origin=tuple(0.2 * np.array(cc)), target=(0, 0, 0), width=w), resx)
                img = render(sc, ob, resx, cfg["spp_recip"], seed); vals.append((img.reshape(-1, 3).sum(0) * (w / resx) ** 2).tolist())
            r[tag] = vals
        res[name] = r; log("g1f", name, np.mean(r["fwd"], 0), np.mean(r["rev"], 0))
    json.dump(res, open(OUT / "c_g1f.json", "w"), indent=1)


def st_trunc(cfg):
    """Bounce-cap control: L-block scene at 512 volume bounces (1024 is the Cycles maximum)."""
    render_scene("trunc512_lblock", FX.scene_def("lblock"), cfg, cams=("top", "underup"), bounces=512, seeds=SEEDS[:1])


def st_scenes(cfg, only=None):
    for name in (only or ("iso_L1", "iso_L2", "iso_L3", "gap4", "gap8", "floor", "lblock", "lblock_pose1", "bend30", "bend90", "dent")):
        render_scene(name, FX.scene_def(name), cfg)


def main():
    global OUT, LOG
    args = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    OUT = Path(args[0]); OUT.mkdir(parents=True, exist_ok=True); LOG = open(OUT / "cycles.log", "a")
    smoke = "smoke" in args; cfg = SMOKE if smoke else CFG
    only = [a.split("=", 1)[1].split(",") for a in args if a.startswith("only=")]
    stages = [a for a in args[1:] if a in ("chain", "g1a", "g1b", "g1c", "g1d", "g1e", "g1f", "trunc", "scenes")] or ["chain", "g1a", "g1b", "g1c", "g1d", "g1e", "g1f", "trunc", "scenes"]
    t0 = time.time()
    for st in stages:
        t1 = time.time()
        if st == "scenes":
            st_scenes(cfg, only[0] if only else None)
        else:
            globals()[f"st_{st}"](cfg)
        log(f"STAGE {st} {time.time() - t1:.0f}s")
    json.dump(dict(blender=bpy.app.version_string, cfg=cfg, seeds=SEEDS, seconds=time.time() - t0, smoke=smoke, stages=stages),
              open(OUT / f"cycles_meta_{'_'.join(stages)}.json", "w"))


if __name__ == "__main__":
    main()
