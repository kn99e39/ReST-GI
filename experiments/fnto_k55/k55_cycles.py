"""WL55 Cycles 3.5.1 references (protocol fnto_k55): WL53 settings (p53_cycles.reset: CPU, no denoising / adaptive sampling / clamping,
1024 bounces, Gaussian 0.01 point-sampled pixels) and materials (volume = Volume Scatter + Absorption inside a white Transparent BSDF),
plus WL55's one-sided Lambertian emitter (Emission, strength = radiance profile; back faces emit nothing; no BSDF, so it absorbs).

    bpy_native.sh k55_cycles.py -- <out dir> units
    bpy_native.sh k55_cycles.py -- <out dir> render <scene> <cam,...|all> <seed,...> [spp=1024] [threads=12]
Image rows are bottom-up (bpy); camera geometry is saved to ccam_<scene>.json for pixel classification."""
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "phys_ref53")); sys.path.insert(0, str(HERE.parent / "fnto_c54"))
import bpy  # noqa: E402

import p53_cycles as PC  # noqa: E402
import k55_fixtures as KF  # noqa: E402


def emitter_material(profile):
    m = bpy.data.materials.new(f"emit_{profile}"); m.use_nodes = True; nt = m.node_tree; nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial"); em = nt.nodes.new("ShaderNodeEmission"); em.inputs["Color"].default_value = (1, 1, 1, 1)
    geo = nt.nodes.new("ShaderNodeNewGeometry")
    if profile == "U":
        em.inputs["Strength"].default_value = 1.0
    elif profile == "X":
        sep = nt.nodes.new("ShaderNodeSeparateXYZ"); nt.links.new(geo.outputs["Position"], sep.inputs[0])
        a = nt.nodes.new("ShaderNodeMath"); a.operation = "ADD"; a.inputs[1].default_value = 0.05; nt.links.new(sep.outputs["X"], a.inputs[0])
        b = nt.nodes.new("ShaderNodeMath"); b.operation = "MULTIPLY"; b.inputs[1].default_value = -1.0 / KF.LAMBDA_X; nt.links.new(a.outputs[0], b.inputs[0])
        e = nt.nodes.new("ShaderNodeMath"); e.operation = "POWER"; e.inputs[0].default_value = math.e; nt.links.new(b.outputs[0], e.inputs[1])
        nt.links.new(e.outputs[0], em.inputs["Strength"])
    else:
        raise KeyError(profile)
    blk = nt.nodes.new("ShaderNodeEmission"); blk.inputs["Strength"].default_value = 0.0
    mix = nt.nodes.new("ShaderNodeMixShader"); nt.links.new(geo.outputs["Backfacing"], mix.inputs[0])
    nt.links.new(em.outputs[0], mix.inputs[1]); nt.links.new(blk.outputs[0], mix.inputs[2]); nt.links.new(mix.outputs[0], out.inputs["Surface"])
    return m


def build(sd):
    for n, k, m, V in sd["comps"]:
        if k == "medium":
            mat = PC.material("vol")
        elif k == "lambert":
            mat = PC.material("lam", 0.8)
        elif k == "black":
            mat = PC.material("black")
        else:
            mat = emitter_material(sd["emit"][n])
        PC.mesh_obj(n, V, m["F"], mat)


DEVICE = "CPU"


def setup(spp, seed, threads):
    """WL53 settings; DEVICE = CPU (bpy fork via bpy_native.sh) or CUDA (official Blender 3.5.1, user decision 2026-10-28: the stock
    CUDA build is statistically equivalent to the fork on CPU, validated by the user; every render's build / device is recorded)."""
    sc = PC.reset(spp, seed, bounces=1024); sc.render.threads_mode = "FIXED"; sc.render.threads = threads
    if DEVICE != "CPU":
        p = bpy.context.preferences.addons["cycles"].preferences; p.compute_device_type = DEVICE; p.get_devices()
        for d in p.devices:
            d.use = d.type == DEVICE
        sc.cycles.device = "GPU"
    return sc


def manifest(out, **kw):
    kw.update(blender=bpy.app.version_string, build_hash=bpy.app.build_hash.decode() if isinstance(bpy.app.build_hash, bytes) else str(bpy.app.build_hash),
              device=DEVICE, time=time.strftime("%Y-%m-%dT%H:%M:%S"))
    with open(out / "render_manifest.jsonl", "a") as f:
        f.write(json.dumps(kw) + "\n")


def st_units(out, threads):
    """Emitter calibration: (a) a camera sees an emitter of strength 1 directly -> radiance 1; (b) a Lambert 0.8 square (2 cm) 1 cm below
    a 1 m x 1 m emitter -> radiance 0.8 F (F = 0.99960 at the centre, area mean computed in the analysis); (c) the X profile seen directly."""
    rec = {}
    for case in ("direct_U", "lambert_U", "direct_X"):
        sc = setup(256, 5509, threads)
        prof = "X" if case.endswith("X") else "U"
        if case == "lambert_U":
            Ve = np.array([[-0.5, -0.5, 0.01], [-0.5, 0.5, 0.01], [0.5, 0.5, 0.01], [0.5, -0.5, 0.01]]); Fe = np.array([[0, 1, 2], [0, 2, 3]])  # normal -z
            PC.mesh_obj("e", Ve, Fe, emitter_material(prof))
            Vr = np.array([[-0.01, -0.01, 0], [0.01, -0.01, 0], [0.01, 0.01, 0], [-0.01, 0.01, 0]]); Fr = np.array([[0, 1, 2], [0, 2, 3]])         # normal +z
            PC.mesh_obj("r", Vr, Fr, PC.material("lam", 0.8))
            cam = dict(origin=(0, 0, 0.005), target=(0, 0, 0), width=0.02)
        else:
            Ve = np.array([[-0.06, -0.06, 0.0], [-0.06, 0.06, 0.0], [0.06, 0.06, 0.0], [0.06, -0.06, 0.0]]); Fe = np.array([[0, 1, 2], [0, 2, 3]])  # normal -z
            PC.mesh_obj("e", Ve, Fe, emitter_material(prof))
            cam = dict(origin=(0, 0, -0.05), target=(0, 0, 0), width=0.11)
        res = 110 if case != "lambert_U" else 40
        ob, g = PC.ortho(case, cam, res); img = PC.render(sc, ob, res, 256, 5509)
        np.save(out / f"units_{case}.npy", img.astype(np.float32)); rec[case] = dict(geo=g, mean=float(img.mean()))
        PC.log("units", case, rec[case]["mean"])
    json.dump(rec, open(out / "units.json", "w"), indent=1)


def grid_quad(half, z, n, down=True):
    """Square [-half, half]^2 at height z, n x n cells x 2 triangles, normal -z (down=True) or +z."""
    g = np.linspace(-half, half, n + 1); X, Y = np.meshgrid(g, g, indexing="ij"); V = np.stack([X.ravel(), Y.ravel(), np.full(X.size, z)], 1)
    F = []
    for i in range(n):
        for j in range(n):
            a, b, c, d = i * (n + 1) + j, (i + 1) * (n + 1) + j, (i + 1) * (n + 1) + j + 1, i * (n + 1) + j + 1
            F += [(a, d, c), (a, c, b)] if down else [(a, b, c), (a, c, d)]
    return V, np.array(F)


def st_units2(out, threads):
    """Labelled calibration extension (after units showed lambert_U 0.7934 vs analytic 0.7997): Lambert 0.8 receiver (2 cm) under
    emitters of radiance 1 -> 0.8 F; cases vary emitter size / tessellation / distance and Cycles' light tree."""
    rec = {}
    cases = [("e1m_2tri_d10", 0.5, 1, 0.01, True), ("e1m_200x200_d10", 0.5, 200, 0.01, True), ("e20cm_2tri_d10", 0.1, 1, 0.01, True),
             ("e10cm_40x40_d4", 0.05, 40, 0.004, True), ("e1m_2tri_d10_notree", 0.5, 1, 0.01, False), ("e10cm_40x40_d4_notree", 0.05, 40, 0.004, False)]
    for name, half, n, d, tree in cases:
        sc = setup(512, 5508, threads)
        if hasattr(sc.cycles, "use_light_tree"):
            sc.cycles.use_light_tree = tree
        Ve, Fe = grid_quad(half, d, n, True); PC.mesh_obj("e", Ve, Fe, emitter_material("U"))
        Vr, Fr = grid_quad(0.01, 0.0, 1, False); PC.mesh_obj("r", Vr, Fr, PC.material("lam", 0.8))
        ob, g = PC.ortho(name, dict(origin=(0, 0, d / 2), target=(0, 0, 0), width=0.02), 40); img = PC.render(sc, ob, 40, 512, 5508)
        rec[name] = dict(mean=float(img[..., 0].mean()), half=half, n=n, d=d, light_tree=tree, light_tree_attr=hasattr(sc.cycles, "use_light_tree"))
        PC.log("units2", name, rec[name]["mean"])
    json.dump(rec, open(out / "units2.json", "w"), indent=1)


def st_render(out, scene, cams, seeds, spp, threads):
    sd = KF.scene_def(scene); geo = {}
    if (out / f"ccam_{scene}.json").exists():
        geo = json.load(open(out / f"ccam_{scene}.json"))
    cams = list(sd["cams"]) if cams == ["all"] else cams
    for seed in seeds:
        todo = [c for c in cams if not (out / f"cimg_{scene}_{c}_s{seed}.npy").exists()]
        if not todo:
            continue
        sc = setup(spp, seed, threads); build(sd)
        for i, (s, E) in enumerate(sd["lights"]):
            PC.sun(s, E, f"sun{i}")
        for cam in todo:
            c = sd["cams"][cam]; ob, g = PC.ortho(cam, c, c["res"]); geo[cam] = g
            t0 = time.time(); img = PC.render(sc, ob, c["res"], spp, seed)
            dt = time.time() - t0; np.save(out / f"cimg_{scene}_{cam}_s{seed}.npy", img.astype(np.float32)); PC.log(scene, cam, seed, c["res"], f"{dt:.0f}s", DEVICE)
            manifest(out, scene=scene, cam=cam, seed=seed, res=c["res"], spp=spp, seconds=round(dt, 1))
            json.dump(geo, open(out / f"ccam_{scene}.json", "w"), indent=1)


def main():
    a = sys.argv[sys.argv.index("--") + 1:]; out = Path(a[0]); out.mkdir(parents=True, exist_ok=True)
    PC.OUT = out; PC.LOG = open(out / "cycles.log", "a")
    kv = dict(x.split("=") for x in a if "=" in x); pos = [x for x in a if "=" not in x]
    threads = int(kv.get("threads", 12)); spp = int(kv.get("spp", 1024))
    global DEVICE
    DEVICE = kv.get("device", "CPU")
    if pos[1] == "units":
        st_units(out, threads)
    elif pos[1] == "units2":
        st_units2(out, threads)
    elif pos[1] == "render":
        st_render(out, pos[2], pos[3].split(","), [int(s) for s in pos[4].split(",")], spp, threads)


if __name__ == "__main__":
    main()
