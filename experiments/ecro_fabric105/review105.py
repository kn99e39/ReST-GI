"""Display the trained Fabric render and package its measurements."""

import argparse
import hashlib
import json
from pathlib import Path
from shutil import copyfile

import matplotlib
import numpy as np
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt

LUMA = np.array([.2126,.7152,.0722])


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f,"sha256").hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def display(v, white):
    x = np.maximum(v,0.) / white
    x = x/(1+x)
    srgb = np.where(x <= .0031308,12.92*x,1.055*x**(1/2.4)-.055)
    return np.rint(np.clip(srgb,0,1)*255).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run",type=Path)
    parser.add_argument("out",type=Path)
    parser.add_argument("--render-dir",default="renders")
    args = parser.parse_args()
    run, out = args.run, args.out
    renders = run/args.render_dir
    out.mkdir(parents=True,exist_ok=False)
    copy = {"preflight.json":run/"preflight.json","training.json":run/"fit/training.json",
            "data.json":run/"data.json", "metrics.json":renders/"summary.json"}
    for name,path in copy.items():
        copyfile(path,out/name)
    fig, axes = plt.subplots(2,4,figsize=(11,5.8),layout="constrained")
    exposure = {}
    budget = read(renders/"public.json")
    for i,name in enumerate(("public","held")):
        copyfile(renders/f"{name}.json",out/f"{name}_metrics.json")
        with np.load(renders/f"{name}.npz") as z:
            mask = z["mask"]
            ref = z["reference"]
            white = max(float(np.quantile((ref@LUMA)[mask],.95)),1e-20)
            exposure[name] = white
            for j,arm in enumerate(("reference","low","ecro")):
                rgb = display(z[arm],white)
                axes[i,j].imshow(rgb)
                if arm != "low":
                    Image.fromarray(rgb).save(out/f"fabric_{name}_{arm}.png")
            den = float((ref@LUMA)[mask].mean())
            err = ((z["ecro"]-ref)@LUMA)/den
            limit = max(float(np.quantile(np.abs(err[mask]),.98)),1e-8)
            view = axes[i,3].imshow(np.where(mask,err,np.nan),cmap="RdBu_r",vmin=-limit,vmax=limit)
            fig.colorbar(view,ax=axes[i,3],shrink=.70,extend="both")
        axes[i,0].set_ylabel("Public view" if name=="public" else "Held-out view and light")
    for ax,title in zip(axes[0],("Native path trace","Explicit LOW","Trained ECRO","ECRO signed error")):
        ax.set_title(title)
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.suptitle(f"8DNA Fabric at {budget['resolution']}×{budget['resolution']} · {budget['spp_each']*len(budget['seeds']):,} spp")
    fig.supxlabel("Signed luminance difference is normalized by mean reference fabric luminance.",fontsize=9)
    fig.savefig(out/"fabric_comparison.png",dpi=160,facecolor="white")
    plt.close(fig)
    report = {"presentation_exposure_white":exposure,"raw_sha256":{
        p.relative_to(run).as_posix():sha(p) for p in sorted(run.rglob("*")) if p.is_file()},
        "review_sha256":{p.name:sha(p) for p in sorted(out.iterdir()) if p.is_file()},
        "review_code_sha256":sha(Path(__file__))}
    (out/"manifest.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")


if __name__=="__main__":
    main()
