"""Post-hoc control: isolate steady reconstruction error from frozen ECRO bias."""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from analyze import WL101A, decomposition, geometry_masks, sha


def pair(path, learned):
    with np.load(path) as data:
        return data["e1"] + data["e2"] + data["hl" if learned else "h"]


def main(raw, refs, prior, output):
    physical = np.stack([pair(refs/f"REF_0_{i}.npz", False) for i in range(2)])
    converged = np.stack([pair(prior/f"paired_{i}_4096.npz", True) for i in range(2)])
    prior_manifest = json.loads((prior/"run_manifest.json").read_text())
    for row in prior_manifest["matched"]:
        assert sha(prior/row["path"]) == row["sha256"]
    regions = geometry_masks(raw/"guide_0_A.npz")
    d = {"labels":"Post-hoc diagnostic using executed outputs; never feeds reconstruction or the frozen gates.",
         "ref_pair_sha256":[sha(refs/f"REF_0_{i}.npz") for i in range(2)],
         "ecro_pair_sha256":[sha(prior/f"paired_{i}_4096.npz") for i in range(2)],
         "steady_frame_by_spp":{}}
    ref_mean = physical.mean(0)
    ecro_mean = converged.mean(0)
    old_bias = ecro_mean-ref_mean
    cap = WL101A["display"]["signed_error_max"]
    rows, axes = plt.subplots(2,3, figsize=(11,8), layout="constrained")
    for n,spp in enumerate((1,4)):
        files = [raw/f"static_{spp}spp_r{rep}_f23.npz" for rep in range(4)]
        samples = []
        for path in files:
            with np.load(path) as frame:
                samples.append(frame["reconstructed"])
        frames = np.stack(samples)
        offset = frames.mean(0)-ecro_mean
        total = frames.mean(0)-ref_mean
        assert np.max(np.abs(total-(old_bias+offset))) <= 1e-6
        d["steady_frame_by_spp"][spp] = {
            "frame_sha256":{path.name:sha(path) for path in files},
            "pipeline_added_error_against_converged_ecro": {
                k:decomposition(frames,(converged,ecro_mean),mask) for k,mask in regions.items()},
            "physical_comparison": {
                k:decomposition(frames,(physical,ref_mean),mask) for k,mask in regions.items()},
        }
        for j,(name,image) in enumerate((
                ("Reconstructed minus physical",total),
                ("Reconstructed minus converged ECRO",offset),
                ("Converged ECRO minus physical",old_bias))):
            ax = axes[n,j]
            instance = ax.imshow(image.mean(-1), cmap="RdBu_r",vmin=-cap,vmax=cap)
            ax.set_title(f"{spp} spp · {name}",fontsize=9)
            ax.set_axis_off()
    rows.colorbar(instance, ax=axes.ravel().tolist(), shrink=0.75, label="Mean signed RGB difference · linear HDR")
    rows.suptitle("WL101B · steady-frame residual attribution · post-hoc")
    rows.savefig(output/"residual_attribution.png",dpi=150)
    plt.close(rows)
    (output/"residual_attribution.json").write_text(json.dumps(d,indent=2,allow_nan=False)+"\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("raw", "references", "wl101a_raw", "reviewer"):
        parser.add_argument(name,type=Path)
    args = parser.parse_args()
    main(args.raw,args.references,args.wl101a_raw,args.reviewer)
