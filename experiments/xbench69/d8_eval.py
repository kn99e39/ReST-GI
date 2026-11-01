"""WL69 Stage 4 arm D8 (protocol xbench69_v1 + amendment 1): released 8DNA NeuralVolPathIntegrator (unchanged; demo settings
max_depth 10, rr_depth 5, LoopRecord / VCallRecord off) rendering the C1 scene with the trained slab asset. T1 moves the asset
instance only (native rigid transform; no retraining). spp ladder x 2 realizations per state and camera; each render is timed
(scene load = the edit's update, render = per-view cost). Images are re-oriented to the Blender pixel order (checked as in X1).

    cd <8dna work dir> && python d8_eval.py <out_dir> <ckpt> <tag>
"""
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, os.getcwd()); sys.path.insert(0, str(HERE))
import torch  # noqa: E402
import mitsuba as mi  # noqa: E402
mi.set_variant("cuda_ad_rgb")
import drjit as dr  # noqa: E402
import models.integrator  # noqa: E402,F401
import d8_scene as DS  # noqa: E402

CYC = Path.home() / "rest_wl55" / "out" / "cycles_cuda"
LADDER = tuple(int(x) for x in os.environ.get("XB69_LADDER", "1,4,16,64,256,1024").split(","))
CHUNK = 16
CAMS = ("top", "underup", "underdown", "otop")


def seed_base(si, li, r):
    return 7_000_000 + 100_000 * si + 1000 * li + 500 * r          # chunk seeds base + k, k < 64: disjoint across (state, level, r)


def orient(sensor, g, s, c):
    W, H = sensor.film().size(); yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    ray, _ = sensor.sample_ray(0.0, 0.5, mi.Point2f(mi.Float(((xx.ravel() + .5) / W).astype(np.float32)), mi.Float(((yy.ravel() + .5) / H).astype(np.float32))),
                               mi.Point2f(.5, .5))
    om = np.stack([np.array(ray.o.x), np.array(ray.o.y), np.array(ray.o.z)], -1).reshape(H, W, 3) / s + c
    res, w = g["res"], g["width"]; o0, R, U = (np.asarray(g[k]) for k in ("origin", "right", "up"))
    rr, cc = np.meshgrid(np.arange(res), np.arange(res), indexing="ij")
    ob = o0 + (((cc + .5) / res * w - w / 2)[..., None] * R) + (((rr + .5) / res * w - w / 2)[..., None] * U)
    best = min(((n, float(np.abs(f(om) - ob).max()), f) for n, f in (("id", lambda a: a), ("ud", np.flipud), ("lr", np.fliplr), ("udlr", lambda a: a[::-1, ::-1]))),
               key=lambda x: x[1])
    assert best[1] < 1e-4 * w / res + 1e-6, best[:2]
    return best


def main():
    out = Path(sys.argv[1]); ckpt = sys.argv[2]; tag = sys.argv[3]; out.mkdir(parents=True, exist_ok=True)
    dr.set_flag(dr.JitFlag.LoopRecord, False); dr.set_flag(dr.JitFlag.VCallRecord, False)
    s, c = DS.similarity(); rec = dict(protocol="xbench69_v1", arm="D8", tag=tag, ckpt=ckpt, renders=[], failures=[])
    for si, (state, scene) in enumerate(DS.STATES.items()):
        # NeuralPathIntegrator.prepare_scene caches the asset instance / transform / bbox on first use (8DNA audit, WL21), so every
        # state gets its own integrator (native construction); its load time is reported as part of that state's update.
        torch.cuda.synchronize(); t0 = time.perf_counter()
        integ = mi.load_dict({"type": "neuralvolpath", "max_depth": 10, "rr_depth": 5, "device": 0, "model_type": "8dna", "asset_instance0": ckpt})
        torch.cuda.synchronize(); rec[f"t_load_model_{state}_s"] = time.perf_counter() - t0
        geo = json.load(open(CYC / f"ccam_{scene}.json")); geo = {k: geo[k] for k in CAMS}
        torch.cuda.synchronize(); t0 = time.perf_counter(); sc = mi.load_dict(DS.eval_scene(state, geo, out / "ply")); torch.cuda.synchronize()
        rec[f"t_scene_load_{state}_s"] = time.perf_counter() - t0
        sens = {x.id(): x for x in sc.sensors()}
        for cam in CAMS:
            name, err, f = orient(sens[cam], geo[cam], s, c)
            _ = mi.render(sc, sensor=sens[cam], integrator=integ, spp=1, seed=6_999_999)            # warm-up (kernel compile), excluded
            for li, spp in enumerate(LADDER):
                for r in (0, 1):
                    b = seed_base(si, li, r); n = max(1, spp // CHUNK); ch = min(spp, CHUNK)
                    try:
                        torch.cuda.synchronize(); t0 = time.perf_counter(); img = None
                        for k in range(n):
                            part = mi.render(sc, sensor=sens[cam], integrator=integ, spp=ch, seed=b + k); img = part if img is None else img + part
                        img = np.array(img, dtype=np.float32) / n; torch.cuda.synchronize(); dt = time.perf_counter() - t0
                        np.save(out / f"d8_{tag}_{state}_{cam}_spp{spp}_r{r}.npy", f(img))
                        rec["renders"].append(dict(state=state, cam=cam, spp=spp, r=r, seed_base=b, seconds=dt, orient=name, finite=bool(np.isfinite(img).all())))
                        print(time.strftime("%H:%M:%S"), state, cam, spp, r, f"{dt:.2f}s", flush=True)
                    except Exception as e:  # noqa: BLE001  (native-code failure is recorded, not hidden)
                        rec["failures"].append(dict(state=state, cam=cam, spp=spp, r=r, error=f"{type(e).__name__}: {e}", tb=traceback.format_exc()[-800:]))
                        print("FAIL", state, cam, spp, r, e, flush=True)
    rec["peak_gpu_gb"] = torch.cuda.max_memory_allocated() / 1e9
    json.dump(rec, open(out / f"d8_eval_{tag}.json", "w"), indent=1); sys.stdout.flush(); os._exit(0)


if __name__ == "__main__":
    main()
