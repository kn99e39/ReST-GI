"""Visualize matched coarse and refined Fabric HIGH output."""

import argparse
import json
from pathlib import Path
from shutil import copyfile

import matplotlib
import numpy as np
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from review105 import LUMA, display, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw",type=Path)
    parser.add_argument("review",type=Path)
    a = parser.parse_args()
    a.review.mkdir(parents=True,exist_ok=False)
    reports = ("summary.json","refinement.json","preflight.json","data.json","fit/training.json",
               "oracle_public.json","oracle_held.json","view_public.json","view_held.json")
    for report in reports:
        copyfile(a.raw/report,a.review/Path(report).name)
    fig,axes = plt.subplots(2,5,figsize=(13.1,5.75),layout="constrained")
    for i,name in enumerate(("public","held")):
        with np.load(a.raw/f"view_{name}.npz") as z:
            mask = z["mask"]
            physical = z["physical"].mean(0)
            white = max(float(np.quantile((physical@LUMA)[mask],.95)),1e-20)
            keys = ("physical","coarse_area","fine_area","coarse_learned","fine_learned")
            for ax,key in zip(axes[i],keys):
                ax.imshow(display(z[key].mean(0),white))
            low = z["low"].mean(0)
            reference = low+physical
            total_white = max(float(np.quantile((reference@LUMA)[mask],.95)),1e-20)
            Image.fromarray(display(low+z["fine_learned"].mean(0),total_white)).save(a.review/f"fabric_{name}_6cm.png")
        axes[i,0].set_ylabel("Public view" if i == 0 else "Held-out view")
    for ax,title in zip(axes[0],("Per-ray physical","18 cm physical table","6 cm physical table",
                               "18 cm trained","6 cm trained")):
        ax.set_title(title,fontsize=11)
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.suptitle("Fabric HIGH with spatial grid refinement")
    fig.supxlabel("The two physical tables use area averages. All columns share one exposure per view.",fontsize=9)
    fig.savefig(a.review/"high_refinement.png",dpi=160,facecolor="white")
    plt.close(fig)
    manifest = {"raw_sha256":{p.relative_to(a.raw).as_posix():sha(p) for p in sorted(a.raw.rglob("*")) if p.is_file()},
                "review_sha256":{p.name:sha(p) for p in sorted(a.review.iterdir()) if p.is_file()},
                "code_sha256":{n:sha(Path(__file__).with_name(n)) for n in (
                    "fabric105.py","run105.py","run105b.py","refinement105b.json","review105b.py")},
                }
    (a.review/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")


if __name__=="__main__":
    main()
