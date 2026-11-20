"""Create reviewer artifacts from the completed, saved WL108 measurements."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common108 import ARMS, SEEDS, SPPS, read, sha, write
from common107 import LUMA


COLORS = {"R3":"#313746", "A0":"#4974a5","A1":"#21846b","B0":"#9367ab","B1":"#bc6244"}


def quality_figure(result, file):
    fig, axs = plt.subplots(1,3,figsize=(13,4.7),layout="constrained")
    for ax,roi,k,title in zip(axs,("primary","silhouette","silhouette"),
                             ("relative_rmse","relative_rmse","signed_bias"),
                             ("Primary HDR RMSE","Silhouette HDR RMSE","Silhouette signed bias")):
        for arm,color in COLORS.items():
            values = [result["pooled"][str(s)][roi]["quality"][arm][k] for s in SPPS]
            y = np.array([v["point"] for v in values])*100
            ci = np.array([v["ci95"] for v in values])*100
            ax.errorbar(SPPS,y,yerr=np.maximum(0,np.stack((y-ci[:,0],ci[:,1]-y))),fmt="o-",color=color,
                        lw=1.45,capsize=2,ms=3,label=arm)
        if k=="signed_bias":
            ax.axhline(0,lw=.8,color="grey")
        ax.set(title=title,ylabel="Percent",xlabel="Samples per pixel",xticks=SPPS)
        ax.grid(alpha=.19)
    axs[-1].legend(fontsize=8)
    fig.savefig(file,dpi=155)
    plt.close(fig)


def crop_figure(data, result, file):
    choices = tuple(v for v in ("V0","H3","H4") if v in result["views"])
    if not choices:
        choices = tuple(result["views"])
    fig, axes = plt.subplots(len(choices),6,figsize=(13.2,7.6),layout="constrained",squeeze=False)
    fig.suptitle(f"4 spp silhouette crops and errors, seed {SEEDS[0]}",fontsize=15)
    for row,v in enumerate(choices):
        with np.load(data/(v+"_maps.npz")) as z:
            ys,xs = np.where(z["silhouette"])
            middles = np.unique(ys)
            mid = middles[len(middles)//2]
            side = np.min(xs[ys==mid])
            sl = (slice(max(0,mid-15),mid+16),slice(max(0,side-20),side+21))
            target = np.einsum("c,chw->hw",LUMA,z["target"])[sl]
            images = [target]+[np.einsum("c,chw->hw",LUMA,z[f"{a}_4"])[sl]-target for a in COLORS]
        for n,(ax,img) in enumerate(zip(axes[row],images)):
            ax.imshow(img,cmap="magma" if n==0 else "RdBu_r",vmin=0 if n==0 else -.15,vmax=2.8 if n==0 else .15)
            ax.set_xticks([])
            ax.set_yticks([])
        axes[row,0].set_ylabel(v)
    for ax,title in zip(axes[0],("Reference",*(f"{a} − ref." for a in COLORS))):
        ax.set_title(title,fontsize=9)
    fig.savefig(file,dpi=155)
    plt.close(fig)


def review(raw, out):
    out.mkdir(parents=True,exist_ok=False)
    frames = {}
    for stage in ("original","confirmation_eval"):
        src = raw / stage
        if not (src/"metrics.json").is_file():
            continue
        complete = read(src/"metrics.json")
        assert complete["complete"]
        frames[stage] = {k:v for k,v in complete.items() if k not in ("views","pooled")}
        frames[stage]["pooled"] = {
            s: {k:po[k] for k in ("primary","silhouette","sss_interior","material_boundary",
                                 "inside_1","inside_2","inside_3","inside_beyond_3") if k in po}
            for s,po in complete["pooled"].items()
        }
        frames[stage]["metrics_source_sha256"] = sha(src/"metrics.json")
        quality_figure(complete,out/(stage+"_quality.png"))
        crop_figure(src,complete,out/(stage+"_crops.png"))
    assert frames
    write(out/"results108.json",frames)
    desc = raw/"descriptors/qualification.json"
    (out/"qualification108.json").write_bytes(desc.read_bytes())
    files = {str(f.relative_to(raw)):sha(f) for f in raw.rglob("*") if f.is_file()}
    write(out/"manifest108.json",{"raw_mirror":str(raw),"inputs_and_per_draw_source":files,
                                   "report_files":{f.name:sha(f) for f in out.iterdir()},
                                   "crops":{"reference_luminance":[0,2.8],"signed_luminance_error":[-.15,.15],
                                            "kernel_predictor_training_seed":SEEDS[0]}})


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("results",type=Path)
    p.add_argument("output",type=Path)
    a=p.parse_args()
    review(a.results,a.output)
