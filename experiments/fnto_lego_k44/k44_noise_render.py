"""WL44 - Monte-Carlo noise floor of the r1 Stage A probe targets (LabServer63 bpy, CPU; one process per image).

Re-renders held-out Stage A configuration <i> with the identical scene / lights / camera / film / spp as r1_stagea_render
but a different Cycles seed (set before the first render of the process; see amendment 2 of fnto_r1), and stores the
colour at the original pixel list. k44_noise_cmp.py compares it with the original image: two independent renders differ
by sqrt(2) x the per-render noise, so the single-render noise-floor PSNR = PSNR(render_a, render_b) + 3.01 dB.

    RNA_DIR=<rna repo> python k44_noise_render.py -- <i> <out dir> [spp=512] [tag=noise]
  tag 'clean' + spp 4096: a low-noise target (8x the Stage A samples) for the same configuration.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "fnto_lego_r1")); sys.path.insert(0, str(HERE.parent / "fnto_lego"))
sys.path.insert(0, os.environ.get("RNA_DIR", str(HERE.parents[1] / "external/relightable-neural-assets")))
import bpy  # noqa: E402,F401
from utils import exr  # noqa: E402

import r1_common as C  # noqa: E402
import r1_stagea_render as SR  # noqa: E402


def main():
    a = sys.argv[sys.argv.index("--") + 1:]
    i = int(a[0]); out = Path(a[1]); out.mkdir(parents=True, exist_ok=True)
    spp = int(a[2]) if len(a) > 2 else 512; tag = a[3] if len(a) > 3 else "noise"
    cfg = {c["i"]: c for c in json.loads((C.OUT1 / "stagea/holdout/configs.json").read_text())}[i]
    orig = np.load(C.OUT1 / f"stagea/holdout/img_{i:05d}.npz")
    scn = SR.Scene(512, spp); scn.apply(dict(cfg))
    scn.sc.cycles.seed = (7919 if tag == "noise" else 15013) + i                                             # differs from the original (seed = i)
    bpy.context.scene.update_render_engine(); bpy.ops.render.render(write_still=True)
    rgba = exr.read("temp_beauty.exr").reshape(-1, 4)
    np.savez(out / f"{tag}_{i:05d}.npz", pix=orig["pix"], col=rgba[orig["pix"], :3].astype(np.float32), alpha=rgba[orig["pix"], 3].astype(np.float16),
             seed=scn.sc.cycles.seed, spp=spp)
    print("done", i, flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
