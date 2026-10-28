"""Compare the Cycles GPU smoke renders (c54_gpu_smoke.py) against the reference build: every device of the official Blender 3.5.1 build
(CPU / CUDA / OPTIX) vs the relightable-neural-assets bpy fork on CPU (the build behind the WL53 / WL54 references).
    python c54_gpu_compare.py <fork dir> <stock dir> <out json>
Per case and device: warm render time and speed-up vs the fork CPU, ratio of image means with its SE, and the per-pixel z of the difference
of two-seed means (variance from the seed pairs of both arms): |z| > 3 should be ~0.3 % and z_std ~1 if the two are the same estimator.
"""
import json
import sys
from pathlib import Path

import numpy as np

SEEDS = (11, 12)
fork, stock, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
F = json.load(open(fork / "gpu_smoke.json"))["cases"]; S = json.load(open(stock / "gpu_smoke.json"))["cases"]; res = {}
for case in F:
    ref = [np.load(fork / f"{case}_CPU_{s}.npy").astype(np.float64) for s in SEEDS]; mr = (ref[0] + ref[1]) / 2; vr = (ref[0] - ref[1]) ** 2 / 4
    res[case] = {"fork_CPU": dict(time_warm_s=F[case]["CPU"]["time_warm_s"], mean=F[case]["CPU"]["mean"])}
    for dev, v in S[case].items():
        x = [np.load(stock / f"{case}_{dev}_{s}.npy").astype(np.float64) for s in SEEDS]; mx = (x[0] + x[1]) / 2; var = vr + (x[0] - x[1]) ** 2 / 4
        ok = var > 0; z = (mx - mr)[ok] / np.sqrt(var[ok])
        res[case][f"stock_{dev}"] = dict(device=v["device"], time_first_s=v["time_first_s"], time_warm_s=v["time_warm_s"], speedup_vs_fork_cpu=F[case]["CPU"]["time_warm_s"] / v["time_warm_s"],
                                         mean_ratio_vs_fork_cpu=float(mx.mean() / mr.mean()), mean_ratio_se=float(np.sqrt(var.sum()) / mr.sum()),
                                         z_abs_gt3_frac=float((np.abs(z) > 3).mean()), z_std=float(z.std()), max_abs_diff=float(np.abs(mx - mr).max()))
json.dump(res, open(out, "w"), indent=1)
for case, r in res.items():
    print(case, "| fork CPU warm %.1f s" % r["fork_CPU"]["time_warm_s"])
    for k, v in r.items():
        if k.startswith("stock"):
            print(f"   {k:12s} warm {v['time_warm_s']:6.1f} s (first {v['time_first_s']:6.1f} s)  x{v['speedup_vs_fork_cpu']:5.1f} vs fork CPU | mean ratio {v['mean_ratio_vs_fork_cpu']:.5f} +- {v['mean_ratio_se']:.5f} | |z|>3 {100 * v['z_abs_gt3_frac']:.2f} %  z_std {v['z_std']:.3f}")
