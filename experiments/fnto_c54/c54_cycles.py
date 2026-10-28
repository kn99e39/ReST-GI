"""WL54 Cycles 3.5.1 references for the Lambert fixtures (protocol fnto_c54). Same renderer settings as WL53 (p53_cycles.reset:
point-sampled GAUSSIAN 0.01 px, 1024 bounces, no clamping, sun angle 0). Pixel size 0.5 mm for every camera except the two
overview cameras top / otop (1 mm, as WL53); 2048 spp; seeds 5401-5404.
Run inside restgi/bpy-runtime with the relightable-neural-assets venv:  python c54_cycles.py -- <out dir> [scene ...] [smoke]
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "phys_ref53")); sys.path.insert(0, str(HERE))
import p53_cycles as PC  # noqa: E402
import c54_fixtures as CF  # noqa: E402

SEEDS54 = (5401, 5402, 5403, 5404)
PX = {"top": 0.001, "otop": 0.001}
PX_DEFAULT = 0.0005
SPP = 2048


def render_scene54(tag, sd, smoke=False):
    geo = {}
    for seed in (SEEDS54[:1] if smoke else SEEDS54):
        spp = 16 if smoke else SPP
        sc = PC.reset(spp, seed, bounces=1024); PC.build(sd, "lam", False)
        for i, (s, E) in enumerate(sd["lights"]):
            PC.sun(s, E, f"sun{i}")
        for cam, c in sd["cams"].items():
            px = PX.get(cam, PX_DEFAULT) * (4 if smoke else 1)
            res = max(8, int(round(c["width"] / px))); ob, g = PC.ortho(cam, c, res); geo[cam] = g
            t0 = time.time(); img = PC.render(sc, ob, res, spp, seed)
            np.save(PC.OUT / f"cimg_{tag}_{cam}_{seed}.npy", img.astype(np.float32)); PC.log(f"{tag} {cam} seed {seed} res {res} {time.time() - t0:.0f}s")
    json.dump(geo, open(PC.OUT / f"ccam_{tag}.json", "w"), indent=1)


def main():
    args = sys.argv[sys.argv.index("--") + 1:]
    out = Path(args[0]); out.mkdir(parents=True, exist_ok=True)
    PC.OUT = out; PC.LOG = open(out / "cycles.log", "a")
    smoke = "smoke" in args
    for n in [a for a in args[1:] if a in CF.SCENES] or list(CF.SCENES):
        render_scene54(n + "_L", CF.scene_def(n), smoke)
    json.dump(dict(blender=PC.bpy.app.version_string, seeds=SEEDS54, spp=SPP, px=PX, px_default=PX_DEFAULT), open(out / "cycles_meta_c54.json", "w"))


if __name__ == "__main__":
    main()
