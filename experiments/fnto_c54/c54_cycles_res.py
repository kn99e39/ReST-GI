"""WL54 reference-resolution diagnostic (amendment 1, post hoc): re-render selected gap cameras at finer pixel sizes (one seed, 2048 spp) to measure
how the point-sampled port average converges with the pixel size.   python c54_cycles_res.py -- <out dir> <fixture> <cam,...> <px_mm,...>"""
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "phys_ref53")); sys.path.insert(0, str(HERE))
import p53_cycles as PC  # noqa: E402
import c54_fixtures as CF  # noqa: E402


def main():
    a = sys.argv[sys.argv.index("--") + 1:]; out = Path(a[0]); out.mkdir(parents=True, exist_ok=True); PC.OUT = out; PC.LOG = open(out / "cycles.log", "a")
    name, cams, pxs = a[1], a[2].split(","), [float(x) * 1e-3 for x in a[3].split(",")]; sd = CF.scene_def(name); geo = {}
    for px in pxs:
        sc = PC.reset(2048, 5499, bounces=1024); PC.build(sd, "lam", False)
        for i, (s, E) in enumerate(sd["lights"]):
            PC.sun(s, E, f"sun{i}")
        for cam in cams:
            c = sd["cams"][cam]; res = int(round(c["width"] / px)); ob, g = PC.ortho(cam, c, res); geo[f"{cam}@{px * 1e3:g}"] = g
            t0 = time.time(); img = PC.render(sc, ob, res, 2048, 5499); np.save(out / f"cimg_{name}_{cam}_px{px * 1e3:g}.npy", img.astype(np.float32)); PC.log(name, cam, px, res, f"{time.time() - t0:.0f}s")
    json.dump(geo, open(out / f"ccam_{name}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
