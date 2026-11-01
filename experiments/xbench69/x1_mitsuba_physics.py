"""WL69 gate check X1 (protocol xbench69_v1): does Mitsuba 3.5.1 volpath -- the renderer that produces 8DNA's training paths and
performs its first free flight -- reproduce the C1 floor-fixture physics within the historical rule, at the 8DNA-native unit
convention (asset centred, bbox diagonal 2.0 units) and, as a discriminating control, at metres (WL53's -12 % regime)?

Mitsuba images are re-oriented to the Blender ortho pixel convention (checked ray by ray) and pass through the unchanged WL55
cycles_ports / group_bias machinery against the WL55 Cycles references (floor seeds 5501/5502), so pixel->port mapping, amended
selection and the 3 % + 3 SE rule are identical to WL55's.

    cd <repo>/experiments/fnto_k55 && P53_MI_VARIANT=cuda_ad_rgb python ../xbench69/x1_mitsuba_physics.py <out_dir>
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "fnto_k55"))
import k55_common as KCm  # noqa: E402
import k55_fixtures as KF  # noqa: E402
import p53_fixtures as FX  # noqa: E402
import p53_mi as M  # noqa: E402
import mitsuba as mi  # noqa: E402
import c54_common as CC  # noqa: E402

CYC = Path(os.environ.get("WL55_CYC_DIR", Path.home() / "rest_wl55" / "out" / "cycles_cuda"))
CAMS = ("top", "underup", "underdown")
SEEDS = (6911, 6912)
SPP = int(os.environ.get("X1_SPP", 1024))
DIAG = 2.0                                    # protocol unit_convention: asset bbox diagonal in scene units


def similarity(sd, native):
    dom = [V for n, k, m, V in sd["comps"] if k == "medium"][0]
    if not native:
        return 1.0, np.zeros(3)
    c = 0.5 * (dom.min(0) + dom.max(0)); s = DIAG / float(np.linalg.norm(dom.max(0) - dom.min(0)))
    return s, c


def build(sd, geo, s, c):
    sh = {}
    for n, k, m, V in sd["comps"]:
        Vt = s * (V - c)
        if k == "medium":
            sh[n] = M.medium_shape(Vt, m["F"], M.medium(FX.ALBEDO, FX.SIGMA_T / s, FX.G_PHASE), n)
        elif k == "lambert":
            sh[n] = M.lambert_shape(Vt, m["F"], FX.FLOOR_RHO, n)
        else:
            raise ValueError(k)
    sens = {}
    for cam in CAMS:
        g = geo[cam]; o = s * (np.asarray(g["origin"]) - c); w = s * g["width"]
        T = mi.ScalarTransform4f.look_at(origin=list(map(float, o)), target=list(map(float, o + np.asarray(g["fwd"]))), up=list(map(float, g["up"])))
        T = T @ mi.ScalarTransform4f.scale([w / 2, w / 2, 1.0])
        sens[cam] = {"type": "orthographic", "to_world": T, "near_clip": 1e-6, "far_clip": 1e4,
                     "film": {"type": "hdrfilm", "width": g["res"], "height": g["res"], "rfilter": {"type": "box"}, "pixel_format": "rgb",
                              "component_format": "float32"}}
    return M.scene(sh, [M.sun(sv, E) for sv, E in sd["lights"]], sens)


def orientation(sc, cam, g, s, c):
    """Return f(img) mapping a Mitsuba (H, W, 3) image to Blender pixel order, verified on the pixel-centre ray origins."""
    o, _ = M.sensor_pixel_rays(M.sensor_by_id(sc, cam), offsets=((0.5, 0.5),))[0][0]
    res = g["res"]; om = o.cpu().numpy().reshape(res, res, 3) / s + c
    (ob, _), = CC.blender_ortho_rays(g, ((0.5, 0.5),))
    ob = ob.cpu().numpy().reshape(res, res, 3)
    best = None
    for name, f in (("id", lambda a: a), ("ud", np.flipud), ("lr", np.fliplr), ("udlr", lambda a: a[::-1, ::-1])):
        err = float(np.abs(f(om) - ob).max())
        if best is None or err < best[1]:
            best = (name, err, f)
    assert best[1] < 1e-6 + 1e-4 * g["width"] / res, (cam, best[:2])
    return best


def main():
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
    sd = KF.scene_def("floor"); geo = json.load(open(CYC / "ccam_floor.json")); ss = KCm.classify_sceneset(sd); lab = KCm.labels_of(sd)
    ref = {cam: KCm.cycles_ports(ss, geo[cam], [np.load(f)[..., :3] for f in sorted(CYC.glob(f"cimg_floor_{cam}_s*.npy"))])["amended"] for cam in CAMS}
    res = dict(protocol="xbench69_v1", check="X1", spp=SPP, seeds=SEEDS, mitsuba=mi.__version__, variant=mi.variant(), conventions={})
    for conv, native in (("8dna_native", True), ("metres_control", False)):
        s, c = similarity(sd, native); sc = build(sd, geo, s, c)
        R = dict(scale_units_per_m=s, centre_m=c.tolist(), sigma_t_per_unit=FX.SIGMA_T / s, cams={})
        for cam in CAMS:
            name, err, f = orientation(sc, cam, geo[cam], s, c)
            imgs, secs = [], []
            for seed in SEEDS:
                t0 = time.time(); img = M.render(sc, cam, SPP, seed); secs.append(time.time() - t0)
                imgs.append(f(img.reshape(geo[cam]["res"], geo[cam]["res"], 3)))
            np.save(out / f"mi_{conv}_{cam}.npy", np.stack(imgs).astype(np.float32))
            u = KCm.cycles_ports(ss, geo[cam], imgs)["amended"]; r = ref[cam]
            per_seed = [KCm.group_bias(u["seeds"][i], r["mean"], r["sel"], lab, r["se"]) for i in range(len(SEEDS))]
            g = KCm.group_bias(u["mean"], r["mean"], r["sel"], lab, r["se"])
            for gn in g:
                so = [abs(a - b) / 2 for a, b in zip(per_seed[0][gn]["bias"], per_seed[1][gn]["bias"])]
                g[gn]["se_mitsuba"] = so; g[gn]["se_total"] = [float(np.hypot(a, b)) for a, b in zip(g[gn]["se"], so)]
                g[gn]["passed_total"] = [bool(abs(b) <= KCm.TOL + 3 * t) if rl else None for b, t, rl in zip(g[gn]["bias"], g[gn]["se_total"], g[gn]["ruled"])]
            R["cams"][cam] = dict(orientation=name, orient_err=err, seconds=secs, groups=g)
            print(time.strftime("%H:%M:%S"), conv, cam, {k: [round(100 * b, 2) for b in v["bias"]] for k, v in g.items()}, flush=True)
        ruled = [(cam, gn, ch, v["bias"][ch], v["passed_total"][ch]) for cam, cr in R["cams"].items() for gn, v in cr["groups"].items()
                 for ch in range(3) if v["ruled"][ch]]
        R["n_ruled"] = len(ruled); R["n_pass"] = sum(bool(p) for *_, p in ruled)
        R["max_abs_bias"] = max(abs(b) for _, _, _, b, _ in ruled) if ruled else None
        R["pass"] = bool(ruled) and R["n_pass"] == R["n_ruled"]
        res["conventions"][conv] = R
    KCm.dump(res, out / "x1.json")
    print(json.dumps({k: {kk: v[kk] for kk in ("n_ruled", "n_pass", "max_abs_bias", "pass")} for k, v in res["conventions"].items()}))
    sys.stdout.flush(); os._exit(0)


if __name__ == "__main__":
    main()
