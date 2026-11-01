"""WL69 gate check X2 (protocol xbench69_v1): the released 8DNA code (external/8dna26 @ 4a2157c, unchanged) builds its CUDA
extension on LabServer69 with a user-space GCC 11 (conda-forge, glibc 2.17 sysroot; no root) and renders a released volumetric
homogeneous-medium asset (candle) with NeuralVolPathIntegrator.

    cd <8dna26 repo> && python x2_8dna_env.py <out_dir>   (env: CC/CXX = toolchain, PATH has nvcc, LD_LIBRARY_PATH = toolchain lib)
"""
import json
import os
import sys
import time
from pathlib import Path

out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, os.getcwd())
import torch  # noqa: E402
import mitsuba as mi  # noqa: E402
mi.set_variant("cuda_ad_rgb")
import drjit as dr  # noqa: E402
rec = dict(protocol="xbench69_v1", check="X2", CC=os.environ.get("CC"), torch=torch.__version__, mitsuba=mi.__version__)
t0 = time.time()
import models.integrator  # noqa: E402,F401  (JIT-builds models/cu_ext)
rec["import_and_build_s"] = time.time() - t0
dr.set_flag(dr.JitFlag.LoopRecord, False); dr.set_flag(dr.JitFlag.VCallRecord, False)
import importlib  # noqa: E402
sd = importlib.import_module("scenes.candle").get_scene(128)
scene = mi.load_dict(sd)
integ = mi.load_dict({"type": "neuralvolpath", "max_depth": 64, "rr_depth": 5, "device": 0, "model_type": "8dna",
                      "asset_instance0": "checkpoints/candle.ckpt"})
t0 = time.time(); img = None
for s in range(4):
    im = mi.render(scene, integrator=integ, spp=4, seed=s); img = im if img is None else img + im
img = (img / 4).torch(); torch.cuda.synchronize(); rec["render_16spp_128px_s"] = time.time() - t0
rec["finite"] = bool(torch.isfinite(img).all()); rec["mean_rgb"] = img.mean((0, 1)).tolist(); rec["peak_gpu_gb"] = torch.cuda.max_memory_allocated() / 1e9
torch.save(img.cpu(), out / "candle_smoke.pt")
rec["pass"] = bool(rec["finite"] and min(rec["mean_rgb"]) > 0)
json.dump(rec, open(out / "x2.json", "w"), indent=1); print(json.dumps(rec)); sys.stdout.flush(); os._exit(0)
