"""WL53 reference renders (Mitsuba 3.5.1 volpath, llvm_ad_rgb) and their port averages (protocol phys_ref53).

    P53_MI_VARIANT=llvm_ad_rgb python p53_render.py <stage> <out dir> [smoke]
stages: g1a (semi-infinite H-function), g1b (furnace), g1c (zero / superposition), g1d (path history), g1e (grouping),
        g1f (reciprocity), scenes (port averages of every comparison scene), all
"""
import json
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("P53_MI_VARIANT", "llvm_ad_rgb")
import numpy as np  # noqa: E402

import mitsuba as mi  # noqa: E402
import p53_compose as CP  # noqa: E402
import p53_fixtures as FX  # noqa: E402
import p53_geom as G  # noqa: E402
import p53_mi as M  # noqa: E402

assert M.VARIANT.startswith("llvm"), "renders must use llvm_ad_rgb (LabServer63 cuda film accumulation is wrong)"
SEEDS = (5301, 5302)
SEEDS8 = tuple(range(5301, 5309))   # point measurements (meters, patch cameras) are heavy-tailed: 8 seeds for a CI
CFG = dict(spp_h=16384, spp_furnace=4096, spp_scene=1024, spp_pathhist_cam=16384, spp_meter=4194304, spp_recip=4194304)
SMOKE = dict(spp_h=256, spp_furnace=64, spp_scene=32, spp_pathhist_cam=256, spp_meter=65536, spp_recip=65536)
MED = M.medium(FX.ALBEDO, FX.SIGMA_T, FX.G_PHASE)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def sensors_of(cams, which=None):
    return {k: M.ortho(c["origin"], c["target"], c["up"], c["width"], c["width"], c["res"], c["res"]) for k, c in cams.items() if which is None or k in which}


def mi_shapes(sd, grouped=False):
    sh = {}
    med = [(n, m, V) for n, k, m, V in sd["comps"] if k == "medium"]
    if grouped and len(med) > 1:
        V = np.concatenate([V for _, _, V in med]); off = np.cumsum([0] + [len(V) for _, _, V in med])[:-1]
        F = np.concatenate([m["F"] + o for (_, m, _), o in zip(med, off)])
        sh["media_grouped"] = M.medium_shape(V, F, MED, "grp")
    else:
        for n, m, V in med:
            sh[n] = M.medium_shape(V, m["F"], MED, n)
    for n, k, m, V in sd["comps"]:
        if k == "lambert":
            sh[n] = M.lambert_shape(V, m["F"], FX.FLOOR_RHO, n)
    if sd.get("plate"):
        Vp, Fp = FX.plate_mesh(); sh["plate"] = M.black_shape(Vp, Fp, "plate")
    return sh


def sceneset_of(sd):
    comps = [CP.Comp(n, k, m, V, rho=FX.FLOOR_RHO if k == "lambert" else None) for n, k, m, V in sd["comps"]]
    return CP.SceneSet(comps, FX.DOUT, FX.DIN)


def scene_ports(name, out, cfg, grouped=False, light_override=None, tag=None):
    sd = FX.scene_def(name); lights = light_override if light_override is not None else sd["lights"]
    sc = M.scene(mi_shapes(sd, grouped), [M.sun(s, E) for s, E in lights], sensors_of(sd["cams"]))
    ss = sceneset_of(sd); rec = {}
    for cam in sd["cams"]:
        rays, (Hh, Ww) = M.sensor_pixel_rays(M.sensor_by_id(sc, cam))
        pid = CP.classify(ss, rays)
        imgs = []
        for seed in SEEDS:
            t0 = time.time(); img = M.render(sc, cam, cfg["spp_scene"], seed).reshape(-1, 3); imgs.append(img)
            log(f"{tag or name} {cam} seed {seed} {time.time() - t0:.1f}s")
        means = [CP.port_means(pid, im, ss.Ptot)[0] for im in imgs]; cnt = CP.port_means(pid, imgs[0], ss.Ptot)[1]
        rec[cam] = dict(mean=np.stack(means), count=cnt)
        np.save(out / f"img_{tag or name}_{cam}.npy", np.stack(imgs).reshape(len(SEEDS), Hh, Ww, 3).astype(np.float32))
        np.save(out / f"pid_{tag or name}_{cam}.npy", pid.cpu().numpy().reshape(Hh, Ww))
    np.savez(out / f"ports_{tag or name}.npz", **{f"{c}__{k}": v for c, d in rec.items() for k, v in d.items()})
    return rec


# ------------------------------------------------------------------ G1


def g1a(out, cfg):
    res = []
    for alb in (0.5, 0.9, 0.99, 0.999):
        for th0 in (0.0, 60.0):
            s = (math.sin(math.radians(th0)), 0.0, -math.cos(math.radians(th0)))
            cams = {}
            for th in (0.0, 40.0, 70.0):
                d = np.array([0.0, math.sin(math.radians(th)), math.cos(math.radians(th))])
                cams[f"th{int(th)}"] = M.ortho(0.5 * d, (0, 0, 0), (1, 0, 0) if th > 0 else (0, 1, 0), 0.04, 0.04, 16, 16)
            cube = {"type": "cube", "to_world": mi.ScalarTransform4f.translate([0, 0, -0.25]) @ mi.ScalarTransform4f.scale([1.0, 1.0, 0.25]),
                    "bsdf": {"type": "null"}, "interior": M.medium(alb, FX.SIGMA_T)}
            sc = M.scene({"cube": cube}, [M.sun(s, 1.0)], cams)
            for th in (0.0, 40.0, 70.0):
                imgs = [M.render(sc, f"th{int(th)}", cfg["spp_h"], seed).reshape(-1, 3)[:, 0] for seed in SEEDS]
                exp = G.semi_infinite_radiance(alb, math.cos(math.radians(th)), math.cos(math.radians(th0)), 1.0)
                allpx = np.concatenate(imgs)                                        # radiance is laterally uniform: pixels are iid samples
                res.append(dict(albedo=alb, theta0=th0, theta=th, mitsuba=float(allpx.mean()), sem=float(allpx.std() / math.sqrt(len(allpx))),
                                seeds=[float(i.mean()) for i in imgs], analytic=exp))
                log("g1a", alb, th0, th, res[-1]["mitsuba"], exp)
    json.dump(res, open(out / "g1a_mitsuba.json", "w"), indent=1)


def g1b(out, cfg):
    sd = FX.scene_def("iso_L1"); res = {}
    for tag, alb in (("alb1", (1.0, 1.0, 1.0)), ("albW", FX.ALBEDO)):
        m = sd["comps"][0][2]; V = sd["comps"][0][3]
        sc = M.scene({"slab": M.medium_shape(V, m["F"], M.medium(alb, FX.SIGMA_T))}, [{"type": "constant", "radiance": M.rgb(1.0)}], sensors_of(sd["cams"], ("top", "otop")))
        ss = sceneset_of(sd); res[tag] = {}
        for cam in ("top", "otop"):
            rays, _ = M.sensor_pixel_rays(M.sensor_by_id(sc, cam)); pid = CP.classify(ss, rays)
            ms = [CP.port_means(pid, M.render(sc, cam, cfg["spp_furnace"], seed).reshape(-1, 3), ss.Ptot) for seed in SEEDS]
            res[tag][cam] = dict(mean=np.stack([m_[0] for m_ in ms]).tolist(), count=ms[0][1].tolist())
            log("g1b", tag, cam, np.stack([m_[0][ms[0][1] > 0] for m_ in ms]).mean((0, 1)))
    json.dump(res, open(out / "g1b_furnace.json", "w"))


def g1c(out, cfg):
    A = (math.sin(math.radians(30)), 0.0, -math.cos(math.radians(30)))
    B = (0.0, math.sin(math.radians(50)), math.cos(math.radians(50)))
    for tag, lights in (("zero", []), ("A", [(A, 1.0)]), ("B", [(B, 1.0)]), ("AB", [(A, 1.0), (B, 1.0)])):
        scene_ports("iso_L1", out, cfg, light_override=lights, tag=f"sup_{tag}")


def _pathhist_scene(config, grouped, kind):
    """above_single (primary): one slab, sun from above at 45 deg, 2 x 2 cm patch 5 cm above the top face (facing down).
    below_pair (secondary, WL51 J_below analogue): two slabs with a 4 mm gap lit from below, patch 5 cm above."""
    if config == "above_single":
        sd = FX.scene_def("iso_L1"); light = (math.sin(math.radians(45)), 0.0, -math.cos(math.radians(45))); zt = 0.005; half = 0.01
    else:
        sd = FX.scene_def("gap4_below"); light = (0.0, 0.0, 1.0); zt = 0.012; half = 0.05
    shapes = mi_shapes(sd, grouped); zp = zt + 0.05
    Vp, Fp = M.quad_mesh((0, 0, zp), (0, 0, -1), half)
    shapes["patch"] = (M.lambert_shape(Vp, Fp, 0.8, "patch") if kind == "diffuse" else M.mirror_shape(Vp, Fp, "patch"))
    Vm, Fm = M.quad_mesh((0, 0, zp - 1e-4), (0, 0, -1), 0.005)
    film = {"type": "hdrfilm", "width": 1, "height": 1, "rfilter": {"type": "box"}, "pixel_format": "rgb"}
    shapes["meter"] = {"type": "ply", "filename": M.tmp_ply(Vm, Fm, "meter"), "face_normals": True, "bsdf": {"type": "null"}, "sensor": {"type": "irradiancemeter", "film": film}}
    sens = {"pcam": M.ortho((0, 0, zt + 0.02), (0, 0, 1), (0, 1, 0), 0.012, 0.012, 24, 24),
            "tcam": M.ortho((0, 0, zt + 0.02), (0, 0, 0), (0, 1, 0), 0.012, 0.012, 24, 24)}
    return M.scene(shapes, [M.sun(light, 1.0)], sens)


def g1d(out, cfg):
    """Camera-vertex vs indirect-vertex consistency of the reference. E_cv: irradiancemeter (the medium is the first vertex
    seen from the meter). E_da = pi L_patch / rho (the medium is seen after a DIFFUSE bounce). Mirror: L via a perfect
    mirror (singular ancestor) vs direct L. The central 1 cm of the patch (20 x 20 px of 24 x 24) matches the meter quad."""
    res = {}
    for config, groups in (("above_single", (False,)), ("below_pair", (True, False))):
        for grouped in groups:
            g = f"{config}_{'one_shape' if grouped else 'two_shapes'}"; res[g] = {}
            sc = _pathhist_scene(config, grouped, "diffuse")
            meter = [s for s in sc.sensors() if s.id() not in ("pcam", "tcam")][0]
            Ecv = [np.array(mi.render(sc, sensor=meter, spp=cfg["spp_meter"], seed=seed)).ravel() for seed in SEEDS8]
            Lp = [M.render(sc, "pcam", cfg["spp_pathhist_cam"], seed)[2:22, 2:22].reshape(-1, 3).mean(0) for seed in SEEDS8]
            res[g]["diffuse"] = dict(E_camera_vertex=[e.tolist() for e in Ecv], E_diffuse_ancestor=[(math.pi * x / 0.8).tolist() for x in Lp])
            sc = _pathhist_scene(config, grouped, "mirror")
            Lm = [M.render(sc, "pcam", cfg["spp_pathhist_cam"], seed)[2:22, 2:22].reshape(-1, 3).mean(0) for seed in SEEDS8]
            Lt = [M.render(sc, "tcam", cfg["spp_pathhist_cam"], seed)[2:22, 2:22].reshape(-1, 3).mean(0) for seed in SEEDS8]
            res[g]["mirror"] = dict(L_via_mirror=[x.tolist() for x in Lm], L_direct=[x.tolist() for x in Lt])
            log("g1d", g, "Ecv", np.mean(Ecv, 0), "Eda", np.mean([math.pi * x / 0.8 for x in Lp], 0), "mirror", np.mean(Lm, 0), "direct", np.mean(Lt, 0))
    json.dump(res, open(out / "g1d_pathhist.json", "w"), indent=1)


def g1e(out, cfg):
    for name in ("gap4", "gap4_below"):
        for grouped in (True, False):
            scene_ports(name, out, cfg, grouped=grouped, tag=f"grp_{name}_{'one' if grouped else 'two'}")


def g1f(out, cfg):
    sd = FX.scene_def("iso_L1"); m = sd["comps"][0][2]; V = sd["comps"][0][3]
    film = {"type": "hdrfilm", "width": 1, "height": 1, "rfilter": {"type": "box"}, "pixel_format": "rgb"}
    quads = {"A": ((0.0, 0.0, 0.008), (0, 0, -1)), "B1": ((0.02, 0.0, -0.008), (0, 0, 1)), "B2": ((0.04, 0.01, -0.008), (0, 0, 1))}   # 3 mm from the faces
    res = {}
    for pair in (("A", "B1"), ("A", "B2")):
        r = {}
        for em, me in (pair, pair[::-1]):
            shapes = {"slab": M.medium_shape(V, m["F"], MED, "slab")}
            for q in pair:
                Vq, Fq = M.quad_mesh(quads[q][0], quads[q][1], 0.005)
                d = {"type": "ply", "filename": M.tmp_ply(Vq, Fq, q), "face_normals": True, "bsdf": {"type": "diffuse", "reflectance": M.rgb(0.0)}}
                if q == em:
                    d["emitter"] = {"type": "area", "radiance": M.rgb(1.0)}
                if q == me:
                    d["sensor"] = {"type": "irradiancemeter", "film": film}
                shapes[q] = d
            sc = M.scene(shapes, [], None)
            E = [np.array(mi.render(sc, sensor=sc.sensors()[0], spp=cfg["spp_recip"], seed=seed)).ravel() for seed in SEEDS8]
            r[f"{em}->{me}"] = [e.tolist() for e in E]
            log("g1f", em, "->", me, np.mean(E, 0))
        res["_".join(pair)] = r
    json.dump(res, open(out / "g1f_recip.json", "w"), indent=1)


def blender_ortho_rays(geo, offsets=((0.5, 0.5), (0.1, 0.1), (0.9, 0.1), (0.1, 0.9), (0.9, 0.9))):
    """Rays of a Blender orthographic camera, pixel order of the saved image (row 0 = bottom)."""
    import torch
    res, w = geo["res"], geo["width"]; o0 = np.asarray(geo["origin"]); R = np.asarray(geo["right"]); U = np.asarray(geo["up"]); Fw = np.asarray(geo["fwd"])
    rr, cc = np.meshgrid(np.arange(res), np.arange(res), indexing="ij"); out = []
    for ox, oy in offsets:
        x = (cc.ravel() + ox) / res * w - w / 2; y = (rr.ravel() + oy) / res * w - w / 2
        o = o0[None] + x[:, None] * R[None] + y[:, None] * U[None]
        out.append((torch.tensor(o, dtype=torch.float32, device="cuda"), torch.tensor(np.broadcast_to(Fw, o.shape).copy(), dtype=torch.float32, device="cuda")))
    return out


def _tag_scene(tag):
    if tag.startswith(("sup_", "furnace_")):
        return "iso_L1"
    if tag.startswith("grp_gap4_below"):
        return "gap4_below"
    if tag.startswith("grp_gap4"):
        return "gap4"
    if tag.startswith("trunc512_"):
        return tag.split("_", 1)[1]
    return tag


def cports(out, cfg):
    """Port averages of every Cycles image set (primary reference): <results>/cycles/ccam_<tag>.json + cimg_<tag>_<cam>_<seed>.npy
    -> <out>/cports_<tag>.npz (cam__mean (n_seeds, P, 3), cam__count)."""
    cdir = out.parent / "cycles"
    for gj in sorted(cdir.glob("ccam_*.json")):
        tag = gj.stem[5:]; geo = json.load(open(gj)); sd = FX.scene_def(_tag_scene(tag)); ss = sceneset_of(sd); rec = {}
        for cam, g in geo.items():
            files = sorted(cdir.glob(f"cimg_{tag}_{cam}_*.npy"))
            pid = CP.classify(ss, blender_ortho_rays(g))
            means = [CP.port_means(pid, np.load(f).reshape(-1, 3), ss.Ptot) for f in files]
            rec[f"{cam}__mean"] = np.stack([m[0] for m in means]); rec[f"{cam}__count"] = means[0][1]
            np.save(out / f"cpid_{tag}_{cam}.npy", pid.cpu().numpy().reshape(g["res"], g["res"]))
        np.savez(out / f"cports_{tag}.npz", **rec); log("cports", tag, len(geo))


SCENES = ("iso_L1", "iso_L2", "iso_L3")          # Mitsuba is the diagnostic second renderer (decision R1): isolated slab only


def main():
    stage, out = sys.argv[1], Path(sys.argv[2]); smoke = "smoke" in sys.argv[3:]
    cfg = dict(SMOKE if smoke else CFG); out.mkdir(parents=True, exist_ok=True)
    for a in sys.argv[3:]:
        if a.startswith("cfg."):
            k, v = a[4:].split("="); cfg[k] = int(v)
    only = [a.split("=", 1)[1].split(",") for a in sys.argv[3:] if a.startswith("only=")]
    t0 = time.time()
    stages = [stage] if stage != "all" else ["g1a", "g1b", "g1c", "g1d", "g1e", "g1f", "scenes"]
    for st in stages:
        if st == "scenes":
            for name in (only[0] if only else SCENES):
                scene_ports(name, out, cfg)
        elif st == "cports":
            cports(out, cfg)
        else:
            globals()[st](out, cfg)
    json.dump(dict(cfg=cfg, seeds=SEEDS, variant=M.VARIANT, mitsuba=mi.MI_VERSION, seconds=time.time() - t0, smoke=smoke),
              open(out / f"render_meta_{stage}.json", "w"))
    log("done", stage, time.time() - t0)
    sys.stdout.flush(); os._exit(0)


if __name__ == "__main__":
    main()
