"""WL54 reference-resolution renders (amendments 1 and 3): selected cameras at finer pixel sizes with Cycles 3.5.1, WL53 reset() settings.
    python c54_cycles_res.py -- <out dir> <fixture | vol:<wl53 scene>> <cam,...> <px_mm,...> [seeds=5499,5498] [spp=2048]
A 'vol:' fixture renders the WL53 volume scene (medium K reference) with the WL53 material; otherwise the Lambert fixture of c54_fixtures."""
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "phys_ref53")); sys.path.insert(0, str(HERE))
import p53_cycles as PC  # noqa: E402
import p53_fixtures as FX  # noqa: E402
import c54_fixtures as CF  # noqa: E402


def main():
    a = sys.argv[sys.argv.index("--") + 1:]; out = Path(a[0]); out.mkdir(parents=True, exist_ok=True); PC.OUT = out; PC.LOG = open(out / "cycles.log", "a")
    name, cams, pxs = a[1], a[2].split(","), [float(x) * 1e-3 for x in a[3].split(",")]
    kv = dict(x.split("=") for x in a[4:]); seeds = [int(s) for s in kv.get("seeds", "5499").split(",")]; spp = int(kv.get("spp", 2048))
    vol = name.startswith("vol:"); sd = FX.scene_def(name[4:]) if vol else CF.scene_def(name); tag = name[4:] if vol else name; geo = {}
    for px in pxs:
        for seed in seeds:
            sc = PC.reset(spp, seed, bounces=1024); PC.build(sd, "vol" if vol else "lam", False)
            for i, (s, E) in enumerate(sd["lights"]):
                PC.sun(s, E, f"sun{i}")
            for cam in cams:
                c = sd["cams"][cam]; res = int(round(c["width"] / px)); ob, g = PC.ortho(cam, c, res); geo[f"{cam}@{px * 1e3:g}"] = g
                t0 = time.time(); img = PC.render(sc, ob, res, spp, seed); np.save(out / f"cimg_{tag}_{cam}_px{px * 1e3:g}_s{seed}.npy", img.astype(np.float32)); PC.log(tag, cam, px, seed, res, f"{time.time() - t0:.0f}s")
    json.dump(geo, open(out / f"ccam_{tag}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
