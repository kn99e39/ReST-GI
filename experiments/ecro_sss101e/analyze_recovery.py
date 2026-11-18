"""WL101E attribution using frozen physical references and paired camera seeds."""
import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/"ecro_sss101c"))
from analyze_contact import A, CAP, LUMA, P as C, make_regions, tone
from common import sha, write_json

PROTOCOL=HERE/"protocol/recovery.json"
P=json.loads(PROTOCOL.read_text())


def load_high(path,key):
    with np.load(path) as data:
        image=data[key].astype(np.float64)
    assert image.shape==(512,512,3) and np.isfinite(image).all() and image.min()>=0
    return image


def interval(grid):
    """Crossed marginal variances. Camera variation retains the physical pairing."""
    grid=np.asarray(grid,dtype=np.float64)
    n,m=grid.shape
    by_table,by_camera=grid.mean(1),grid.mean(0)
    u,v=by_table.var(ddof=1)/n,by_camera.var(ddof=1)/m
    se=np.sqrt(u+v)
    df=(u+v)**2/(u*u/(n-1)+v*v/(m-1)) if se else None
    half=float(t.ppf(.975,df)*se) if df is not None else 0.
    center=float(grid.mean())
    return {"mean":center,"ci95":[center-half,center+half],"half_width95":half,
            "table_mean_se":float(np.sqrt(u)),"paired_camera_mean_se":float(np.sqrt(v)),
            "cells":grid.tolist(),"degrees_of_freedom":float(df) if df is not None else None}


def luma(images,mask):
    return (images[...,mask,:]@LUMA).mean(-1)


def endpoint(learned,oracle,ref,baseline):
    total=interval(np.broadcast_to(learned-ref,oracle.shape))
    generated=interval(learned-oracle)
    readout=interval(oracle-ref)
    assert abs(total["mean"]-generated["mean"]-readout["mean"])<1e-12
    noise=lambda x:float(abs(x[0]-x[1])/2)
    return {"learned":float(learned.mean()),"oracle":float(oracle.mean()),"reference":float(ref.mean()),
            "total":total,"generated":generated,"readout":readout,"historical_total":baseline,
            "reference_mean_noise":noise(ref),"learned_path_noise":noise(learned),
            "oracle_path_noise":noise(oracle.mean(0)),
            "oracle_table_half_noise":noise(np.array([oracle[:2].mean(),oracle[2:].mean()])),
            "signed_fraction":{key:x["mean"]/total["mean"] if total["mean"] else None
                               for key,x in (("generated",generated),("readout",readout))},
            "fraction_ci95_using_frozen_historical_total":{
                key:[p/baseline for p in x["ci95"]] for key,x in (("generated",generated),("readout",readout))}}


def classify(endpoints):
    for endpoint_result in endpoints.values():
        if endpoint_result["total"]["ci95"][0]<=0 or any(
            endpoint_result[k]["half_width95"]>.25*abs(endpoint_result["historical_total"])
            for k in ("generated","readout")):
            return "INCONCLUSIVE"
    side=("generated","readout")
    positive={k:[x[k]["ci95"][0]>0 for x in endpoints.values()] for k in side}
    resolved={k:[x[k]["ci95"][0]>0 or x[k]["ci95"][1]<0 for x in endpoints.values()] for k in side}
    if all(positive["generated"]) and not any(resolved["readout"]):
        return "GENERATED-TABLE-SIDE"
    if all(positive["readout"]) and not any(resolved["generated"]):
        return "REPRESENTATION/READOUT-SIDE"
    if all(any(resolved[k]) for k in side):
        return "MIXED"
    return "INCONCLUSIVE"


def pixel_score(value,target,mask):
    v,y=value[mask],target[mask]
    d=v-y
    rmse=np.sqrt(np.mean(d*d))
    return {"absolute_hdr_rmse":float(rmse),"relative_hdr_rmse":float(rmse/np.sqrt(np.mean(y*y))),
            "relative_hdr_l1":float(np.abs(d).sum()/np.abs(y).sum()),"signed_rgb":d.mean(0).tolist()}


def figures(out,theta,oracle,ref,profile):
    T,G,R=theta-ref,theta-oracle,oracle-ref
    tiles=[("Physical HIGH",ref,False),("Learned HIGH",theta,False),("Physical Oracle HIGH",oracle,False),
           ("Learned − physical",T,True),("Learned − Oracle",G,True),("Oracle − physical",R,True)]
    xa,ya,xb,yb=C["regions"]["crop_xyxy"]
    for filename,sl,dimensions in (("full_comparison",np.s_[:,:],(12,8)),("contact_crops",np.s_[ya:yb,xa:xb],(12,5))):
        fig,axs=plt.subplots(2,3,figsize=dimensions,layout="constrained")
        for ax,(title,img,signed) in zip(axs.flat,tiles):
            if signed:
                scale=ax.imshow(img[sl]@LUMA,vmin=-CAP,vmax=CAP,cmap="RdBu_r",interpolation="none")
            else:
                ax.imshow(tone(img[sl]),interpolation="none")
            ax.set_title(title)
            ax.set_axis_off()
        fig.suptitle("WL101E · shared exposure and matched 4,096-spp camera streams")
        fig.colorbar(scale,ax=axs.ravel().tolist(),shrink=.72,label="Signed linear HDR luminance")
        fig.savefig(out/f"{filename}.png",dpi=150)
        plt.close(fig)

    x=[np.mean(item["range_m"]) for item in profile]
    fig,axs=plt.subplots(1,2,figsize=(11,4.3),layout="constrained")
    for axis in axs:
        axis.axvspan(0,.18,alpha=.09,color="#b25922")
        axis.axvspan(.26,.4,alpha=.10,color="#293240")
        axis.set_xlabel("Foreground floor radius (m)")
        axis.grid(alpha=.2)
    for key,color in (("reference","#22252c"),("learned","#b2532e"),("oracle","#007c8c")):
        axs[0].plot(x,[item["estimates"][key] for item in profile],".-",label=key.title(),color=color)
    for key,color,title in (("total","#22252c","Learned − physical"),
                            ("generated","#b2532e","Learned − Oracle"),
                            ("readout","#007c8c","Oracle − physical")):
        rows=[item["estimates"][key] for item in profile]
        axs[1].plot(x,[r["mean"] for r in rows],".-",label=title,color=color)
        a,b=np.asarray([r["ci95"] for r in rows]).T
        axs[1].fill_between(x,a,b,color=color,alpha=.15)
    axs[0].set_ylabel("HIGH luminance · linear HDR")
    axs[1].set_ylabel("Signed luminance residual · linear HDR")
    axs[1].axhline(0,color="#727272",lw=.8)
    for axis in axs:
        axis.legend(fontsize=8)
    fig.savefig(out/"intensity_profile.png",dpi=150)
    plt.close(fig)


def main(raw,archives,history,wl101c,out):
    run=json.loads((raw/"run_manifest.json").read_text())
    assert run["protocol_sha256"]==sha(PROTOCOL) and "stop" not in run
    historical=json.loads((history/"run_manifest.json").read_text())
    assert run["prior_manifest_sha256"]==sha(history/"run_manifest.json")
    assert historical["response_sha256"]==run["learned_table_sha256"]
    for name in ("REF_0_0.npz","REF_0_1.npz","MAP_0.npz"):
        assert sha(archives/name)==A["references"]["sha256"][name]
    regions,bands,_=make_regions(archives/"MAP_0.npz")
    ref=np.stack([load_high(archives/f"REF_0_{i}.npz","h") for i in range(2)])
    learned=[]
    for row in historical["matched"]:
        path=history/row["path"]
        assert sha(path)==row["sha256"]
        learned.append(load_high(path,"hl"))
    learned=np.stack(learned)
    oracle=np.empty((4,2,512,512,3),np.float64)
    expected={(i,j) for i in range(4) for j in range(2)}
    for row in run["readouts"]:
        i,j=row["oracle_index"],row["camera_half"]
        assert (i,j) in expected
        expected.remove((i,j))
        assert row["camera_seed"]==historical["matched"][j]["seed"]
        path=raw/row["file"]
        assert sha(path)==row["sha256"]
        oracle[i,j]=load_high(path,"hl")
    assert not expected

    mean_learned,mean_oracle,mean_ref=learned.mean(0),oracle.mean((0,1)),ref.mean(0)
    total,generated,readout=(mean_learned-mean_ref,mean_learned-mean_oracle,mean_oracle-mean_ref)
    additivity=float(np.max(np.abs(total-generated-readout)))
    assert additivity<=1e-12
    old=json.loads(wl101c.read_text())
    baseline={k:old["tests"][k]["per_component"]["high"]["difference"] for k in ("near","concentration")}
    def signals(mask):
        return tuple(luma(a,mask) for a in (learned,oracle,ref))
    near,flank=signals(regions["near"]),signals(regions["flank"])
    primary={"near":endpoint(*near,baseline["near"]),
             "concentration":endpoint(*(a-b for a,b in zip(near,flank)),baseline["concentration"])}
    report={}
    for k,mask in regions.items():
        power=np.sqrt(np.mean(mean_ref[mask]**2))
        noises={
            "reference":(ref[0]-ref[1])/2,
            "oracle_table":(oracle[:2].mean((0,1))-oracle[2:].mean((0,1)))/2,
            "oracle_camera":(oracle[:,0].mean(0)-oracle[:,1].mean(0))/2,
            "generated_camera_paired":((learned[0]-oracle[:,0].mean(0))-
                                       (learned[1]-oracle[:,1].mean(0)))/2,
        }
        report[k]={"pixels":int(mask.sum()),"signed_luminance":endpoint(*signals(mask),baseline["near"]),
                   "sum_luminance":{key:float((v[mask]@LUMA).sum()) for key,v in
                                    (("learned",mean_learned),("oracle",mean_oracle),("reference",mean_ref))},
                   "learned_vs_physical":pixel_score(mean_learned,mean_ref,mask),
                   "oracle_vs_physical":pixel_score(mean_oracle,mean_ref,mask),
                   "learned_vs_oracle":pixel_score(mean_learned,mean_oracle,mask),
                   "hdr_mse_cross_term":float(2*np.mean(generated[mask]*readout[mask])),
                   "relative_noise_rms":{key:float(np.sqrt(np.mean(v[mask]**2))/power)
                                         for key,v in noises.items()}}
    edges=C["regions"]["front_radial_edges_m"]
    profile=[{"range_m":[a,b],"pixels":int(mask.sum()),"estimates":endpoint(*signals(mask),baseline["near"])}
             for a,b,mask in zip(edges,edges[1:],bands)]
    tables=[]
    for row in run["halves"]:
        path=raw/row["file"]
        assert sha(path)==row["sha256"]
        with np.load(path) as f:
            tables.append(f["B"].astype(np.float64))
    field_half_noise=np.sqrt(np.mean(((tables[0]-tables[1])/2)**2)/
                             np.mean(((tables[0]+tables[1])/2)**2))
    verdict=classify(primary)
    result={"verdict":verdict,"render_manifest_sha256":sha(raw/"run_manifest.json"),
            "primary":primary,"regions":report,"radial_profile":profile,
            "decomposition_max_absolute":additivity,"table_half_relative_rms":float(field_half_noise),
            "near_flank_ratio":{k:float(a.mean()/b.mean())
                                for k,a,b in zip(("learned","oracle","reference"),near,flank)}}
    out.mkdir(parents=True,exist_ok=True)
    write_json(out/"metrics.json",result)
    np.savez_compressed(raw/"analysis_maps.npz",learned=mean_learned,oracle=mean_oracle,physical=mean_ref,
                        total=total,generated=generated,readout=readout)
    figures(out,mean_learned,mean_oracle,mean_ref,profile)
    print(verdict,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for label in ("raw","archives","historical","wl101c_metrics","reviewer"):
        p.add_argument(label,type=Path)
    a=p.parse_args()
    main(a.raw,a.archives,a.historical,a.wl101c_metrics,a.reviewer)
