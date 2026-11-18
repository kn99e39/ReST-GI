"""Export individual WL101A reviewer panels with the frozen display transfer."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analyze import tone
from common import P, components, image, sha


def export(raw, out):
    run = json.loads((raw / "run_manifest.json").read_text())
    row = next(row for row in run["frames"]
               if row["spp"] == 32 and row["replicate"] == P["sweep"]["show_replicate"])
    path = raw / row["path"]
    assert sha(path) == row["sha256"]
    frame = components(image(path))
    out.mkdir(parents=True, exist_ok=False)
    with np.load(raw / "analysis_maps.npz") as data:
        reference = data["reference"]
        converged = data["converged"]
        panels = {
            "reference.png": reference[..., 0],
            "ecro_32spp.png": frame[..., 0],
            "low_32spp.png": frame[..., 1],
            "high_32spp.png": frame[..., 2],
        }
        for name, hdr in panels.items():
            plt.imsave(out / name, tone(hdr))
        cap = P["display"]["absolute_error_max"]
        plt.imsave(out / "low_stochastic_32spp.png",
                   np.abs(data["stochastic_low_32"]).mean(axis=-1),
                   cmap="magma", vmin=0, vmax=cap)
        cap = P["display"]["signed_error_max"]
        for i, name in enumerate(("full", "low", "high")):
            residual = (converged[..., i] - reference[..., i]).mean(axis=-1)
            plt.imsave(out / f"converged_{name}_signed.png", residual, cmap="RdBu_r", vmin=-cap, vmax=cap)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    export(args.raw, args.out)
