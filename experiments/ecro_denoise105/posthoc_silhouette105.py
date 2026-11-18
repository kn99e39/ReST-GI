"""POST HOC (after the frozen verdict; descriptive only): silhouette residual vs sphere coverage fraction, and
edge-anchored crops at a residual scale set from R1's crop. Does not change any frozen decision.

python posthoc_silhouette105.py DATA EVAL OUT
"""
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

LUMA = np.array([0.2126, 0.7152, 0.0722])
ARMS = ("R0", "R1", "R2", "R3")
SPPS = (4, 16, 32)
BINS = np.array([0, 1 / 64, .125, .25, .375, .5, .625, .75, .875, 63 / 64, 1.0 + 1e-9])


def main(data, ev, out):
    out.mkdir(parents=True, exist_ok=True)
    eman = json.loads((ev / "eval_manifest.json").read_text())
    views, reps = eman["views"], eman["reps"]
    acc = {(a, s): np.zeros((len(BINS) - 1, 2)) for a in ARMS for s in SPPS}
    crops = {}
    for name in views:
        d = data / "eval" / name
        with np.load(d / "aux.npz") as a:
            sph = a["cover"][6]
        h = [np.load(d / f"target_half{i}.npz")["comp"].astype(np.float64) for i in range(2)]
        T = 0.5 * (h[0] + h[1]); Y = np.tensordot(LUMA, T[0:3] + T[3:6] + T[6:9], 1)
        idx = np.digitize(sph, BINS) - 1
        for s in SPPS:
            for r in range(reps):
                with np.load(ev / f"{name}_spp{s}_r{r}.npz") as f:
                    for a in ARMS:
                        e = np.tensordot(LUMA, f[a].astype(np.float64), 1) - Y
                        for b in range(len(BINS) - 1):
                            m = idx == b
                            if b in (0, len(BINS) - 2):
                                continue
                            acc[(a, s)][b] += [e[m].sum(), Y[m].sum()]
                        if r == 0 and s == 4:
                            crops[(name, a)] = np.tensordot(LUMA, f[a], 1)
        crops[(name, "target")] = Y
        crops[(name, "sph")] = sph
    res = {f"{a}_spp{s}": (acc[(a, s)][1:-1, 0] / np.maximum(acc[(a, s)][1:-1, 1], 1e-12)).tolist() for a in ARMS for s in SPPS}
    res["bins"] = BINS.tolist()
    (out / "posthoc_silhouette.json").write_text(json.dumps(res, indent=1))
    centres = 0.5 * (BINS[1:-2] + BINS[2:-1])
    fig, ax = plt.subplots(1, 3, figsize=(14, 4), layout="constrained", sharey=True)
    for a_, s in zip(ax, SPPS):
        for arm, c, mk in (("R0", "#2a78d6", "o"), ("R1", "#eb6834", "s"), ("R2", "#1baf7a", "^"), ("R3", "#6250d6", "D")):
            a_.plot(centres, res[f"{arm}_spp{s}"], mk + "-", color=c, label=arm)
        a_.axhline(0, color="#58616b", lw=.8); a_.set_title(f"{s} spp"); a_.set_xlabel("sphere coverage fraction of the pixel"); a_.grid(alpha=.25)
    ax[0].set_ylabel("signed residual / target (luminance, pooled)"); ax[0].legend()
    fig.suptitle("WL105 POST HOC · silhouette bias by coverage fraction (6 held-out views x 8 seeds)")
    fig.savefig(out / "posthoc_silhouette_profile.png", dpi=140); plt.close(fig)
    fig, axs = plt.subplots(2, 5, figsize=(16, 6.6), layout="constrained")
    for row, name in enumerate(("V0", "H2")):
        sph = crops[(name, "sph")]
        ys, xs = np.nonzero((sph > 0) & (sph < 1))
        k = np.argmin(xs)
        cy, cx = ys[k], xs[k]
        sl = np.s_[max(cy - 24, 0):cy + 24, max(cx - 8, 0):cx + 40]
        Y = crops[(name, "target")]
        cap = float(np.percentile(np.abs(crops[(name, "R1")] - Y)[sl], 99.5))
        axs[row, 0].imshow(Y[sl], cmap="gray"); axs[row, 0].set_title(f"{name} target (luminance)", fontsize=9)
        for j, a in enumerate(("R0", "R1", "R2", "R3")):
            axs[row, j + 1].imshow((crops[(name, a)] - Y)[sl], cmap="RdBu_r", vmin=-cap, vmax=cap)
            axs[row, j + 1].set_title(f"{a} signed error (±{cap:.3g})", fontsize=9)
        for a_ in axs[row]:
            a_.set_axis_off()
    fig.suptitle("WL105 POST HOC · left sphere silhouette, 4 spp replicate 0, residual scale from R1's crop")
    fig.savefig(out / "posthoc_silhouette_crops.png", dpi=140); plt.close(fig)
    print("POSTHOC DONE")


if __name__ == "__main__":
    main(*[Path(a).resolve() for a in sys.argv[1:4]])
