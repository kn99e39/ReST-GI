"""Smoke test: Cycles 3.5.1 (bpy) on CPU vs GPU (CUDA / OptiX) for the WL53 / WL54 reference scenes -- speed and statistical agreement.
Not part of protocol fnto_c54 (a post-batch infrastructure check, 2026-10-28). Same scene builder and render settings as the references
(p53_cycles.reset: point-sampled Gaussian 0.01 px, 1024 bounces, no clamping, sun angle 0); only the compute device changes.

    python c54_gpu_smoke.py -- <out dir> <device,...>      device in CPU, CUDA, OPTIX
Cases: Lambert gap4 / gapup at 0.5 mm, 1024 spp; WL53 volume gap8 / obot at 1 mm, 256 spp. Two seeds per device and case; the first render
of a device includes kernel loading, so the reported time is the second seed. Agreement vs CPU: ratio of image means and per-pixel z of the
difference of two-seed means (variance from the seed pair).
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "phys_ref53")); sys.path.insert(0, str(HERE))
import bpy  # noqa: E402

import c54_fixtures as CF  # noqa: E402
import p53_cycles as PC  # noqa: E402
import p53_fixtures as FX  # noqa: E402

CASES = {"lam_gap4_gapup": (lambda: CF.scene_def("gap4"), "gapup", "lam", 0.0005, 1024),
         "vol_gap8_obot": (lambda: FX.scene_def("gap8"), "obot", "vol", 0.001, 256)}
SEEDS = (11, 12)


def set_device(sc, dev):
    if dev == "CPU":
        sc.cycles.device = "CPU"; return "CPU"
    p = bpy.context.preferences.addons["cycles"].preferences
    p.compute_device_type = dev; p.get_devices()
    names = []
    for d in p.devices:
        d.use = d.type == dev
        if d.use:
            names.append(d.name)
    sc.cycles.device = "GPU"
    return ",".join(names)


def main():
    a = sys.argv[sys.argv.index("--") + 1:]; out = Path(a[0]); out.mkdir(parents=True, exist_ok=True); PC.OUT = out; PC.LOG = open(out / "smoke.log", "a")
    devs = a[1].split(","); res = {}
    for case, (sdf, cam, mk, px, spp) in CASES.items():
        sd = sdf(); res[case] = {}
        for dev in devs:
            imgs, times, used = [], [], None
            for seed in SEEDS:
                sc = PC.reset(spp, seed, bounces=1024); used = set_device(sc, dev); PC.build(sd, mk, False)
                for i, (s, E) in enumerate(sd["lights"]):
                    PC.sun(s, E, f"sun{i}")
                c = sd["cams"][cam]; resn = int(round(c["width"] / px)); ob, g = PC.ortho(cam, c, resn)
                t0 = time.time(); img = PC.render(sc, ob, resn, spp, seed); times.append(time.time() - t0); imgs.append(img.astype(np.float64))
                np.save(out / f"{case}_{dev}_{seed}.npy", img.astype(np.float32))
            res[case][dev] = dict(device=used, time_first_s=times[0], time_warm_s=times[1], mean=float(np.mean(imgs)), res=resn, spp=spp)
            PC.log(case, dev, used, [round(t, 1) for t in times], res[case][dev]["mean"])
        if "CPU" in res[case]:
            c0 = [np.load(out / f"{case}_CPU_{s}.npy").astype(np.float64) for s in SEEDS]
            for dev in devs:
                if dev == "CPU":
                    continue
                g0 = [np.load(out / f"{case}_{dev}_{s}.npy").astype(np.float64) for s in SEEDS]
                mc, mg = (c0[0] + c0[1]) / 2, (g0[0] + g0[1]) / 2; var = ((c0[0] - c0[1]) ** 2 + (g0[0] - g0[1]) ** 2) / 4
                ok = var > 0; z = (mg - mc)[ok] / np.sqrt(var[ok])
                res[case][dev].update(mean_ratio_vs_cpu=float(mg.mean() / mc.mean()), mean_ratio_se=float(np.sqrt(var.sum()) / mc.sum()),
                                      z_abs_gt3_frac=float((np.abs(z) > 3).mean()), z_mean=float(z.mean()), z_std=float(z.std()),
                                      speedup_warm=res[case]["CPU"]["time_warm_s"] / res[case][dev]["time_warm_s"])
    json.dump(dict(blender=bpy.app.version_string, cases=res), open(out / "gpu_smoke.json", "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
