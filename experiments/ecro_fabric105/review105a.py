"""Package the paired Fabric attribution and visualize HIGH on a common exposure."""

import argparse
import json
from pathlib import Path
from shutil import copyfile

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from review105 import LUMA, display, sha


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("raw",type=Path)
    cli.add_argument("review",type=Path)
    args = cli.parse_args()
    args.review.mkdir(parents=True,exist_ok=False)
    for key in ("summary","crossover","public","held"):
        copyfile(args.raw/f"{key}.json",args.review/f"{key}.json")
    fig,axes = plt.subplots(2,4,figsize=(10.8,5.6),layout="constrained")
    for i,name in enumerate(("public","held")):
        with np.load(args.raw/f"{name}.npz") as z:
            mask = z["mask"]
            phys = z["physical"].mean(axis=0)
            white = max(float(np.quantile((phys@LUMA)[mask],.95)),1e-20)
            for ax,arm in zip(axes[i],("physical","area","camera","learned")):
                ax.imshow(display(z[arm].mean(axis=0),white))
        axes[i,0].set_ylabel("Public view" if i==0 else "Held-out view")
    for ax,title in zip(axes[0],("Physical HIGH","Area-averaged HIGH","Camera-weighted HIGH","Learned HIGH")):
        ax.set_title(title)
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.suptitle("Fabric HIGH at a common exposure within each view")
    fig.savefig(args.review/"high_attribution.png",dpi=160,facecolor="white")
    plt.close(fig)
    manifest = {
        "raw_sha256": {p.name:sha(p) for p in sorted(args.raw.iterdir()) if p.is_file()},
        "review_sha256": {p.name:sha(p) for p in sorted(args.review.iterdir()) if p.is_file()},
        "code_sha256": {n:sha(Path(__file__).with_name(n))
                        for n in ("attribute105a.py","attribution105a.json","review105a.py","review105.py")},
    }
    (args.review/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")


if __name__ == "__main__":
    main()
