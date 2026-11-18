"""Review WL101F with the exact WL101C HDR masks, scales and matched controls."""
import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/"ecro_sss101c"))
from analyze_contact import A, CAP, LUMA, P as C, make_regions, tone
from common import sha, write_json
sys.path.insert(0,str(HERE.parent/"ecro_sss101e"))
from analyze_recovery import interval, luma, pixel_score


LABELS={"learned":"Learned", "uniform":"Uniform U", "query":"Physical query", "Q":"Query density Q",
        "P":"Weighted P", "S":"Spatial S", "A":"Angular A", "SA":"Joint SA"}
COLORS={"physical":"#24242a", "learned":"#94969e", "uniform":"#ae4e34", "query":"#5e8569",
        "Q":"#7b61a1","P":"#007e8b","S":"#bf8600","A":"#5867a4","SA":"#be4680"}


def mean(grid):
    return grid.mean((0,1))


def normalized(weights, axes):
    total=weights.sum(axes,keepdims=True)
    return np.divide(weights,total,out=np.zeros_like(weights),where=total>0)


def usage(raw, run):
    stage=run["stages"].get("B",{})
    if "uniform_audit" not in stage or not stage["training"]:
        return {}
    with np.load(raw/stage["uniform_audit"]["file"]) as f:
        uniform=f["counts"].astype(np.float64)
    rows=[]
    for row in stage["training"]:
        with np.load(raw/row["file"]) as f:
            W=f["W"]@LUMA
            M=f["mass"]
            count=f["count"]
            fresh=f["WL"]
            old=f["L"]
            w_rgb=f["W"].sum((1,2))
        weights=W.sum((1,2))
        shared=(M.sum((1,2))>0)&(uniform.sum((1,2))>0)
        distance={}
        for key,left,right in (("uniform_vs_camera_density",uniform,M),
                               ("uniform_vs_camera_throughput",uniform,W),
                               ("camera_density_vs_throughput",M,W)):
            values={}
            for label,axis in (("spatial",2),("angular",1),("joint",None)):
                p,q=(left,right) if axis is None else (left.sum(axis),right.sum(axis))
                axes=tuple(range(1,p.ndim))
                tv=.5*np.abs(normalized(p,axes)-normalized(q,axes)).sum(axes)
                values[label]=float(np.average(tv[shared],weights=weights[shared]))
            distance[key]=values
        Wmean=np.divide(fresh.sum((1,2)),w_rgb,out=np.zeros_like(w_rgb),where=w_rgb>0)
        Mmass=M.sum((1,2))[:,None]
        Lmean=np.divide(old.sum((1,2)),Mmass,out=np.zeros_like(w_rgb),where=Mmass>0)
        delta=Wmean-Lmean
        shift=delta@LUMA
        rms=np.sqrt(np.average(np.square(shift[shared]),weights=weights[shared]))
        rows.append({"seed":row["prefix_seed"],"joint_active_subcells":int(np.count_nonzero(count)),
                     "original_active_slots":int(np.count_nonzero(weights)),
                     "total_variation_weighted_by_query_throughput":distance,
                     "sampled_uniform_parent_missing_camera_weight_fraction":float(weights[~shared].sum()/weights.sum()),
                     "within_cell_throughput_mean_shift_luma_rms":float(rms),
                     "within_cell_throughput_mean_shift_luma_weighted":float(np.average(shift[shared],weights=weights[shared])),
                     "effective_joint_raw_count_quantiles":np.quantile(count[count>0],[0,.01,.5,.95,1]).tolist()})
    return {"replicas":rows,
            "definition":"Parent-conditional histogram TV. Parents are averaged with camera throughput. The WL101E-probe histogram is a finite independent geometry sample. Wmean-Lmean = Cov_query(throughput, suffix)/E_query(throughput) by RGB within parent and then luminance."}


def plots(out, imgs, ref, profile):
    if not imgs:
        return
    priority=[name for name in ("uniform","query","P","S","A","SA","Q","learned") if name in imgs]
    colors=COLORS
    xa,ya,xb,yb=C["regions"]["crop_xyxy"]
    crop=np.s_[ya:yb,xa:xb]
    raw={**imgs,"physical":ref}
    core=[name for name in ("uniform","P","query","Q","learned") if name in imgs]
    figures=[("contact_crops.png",["physical",*core],crop,12,2.0,3),
             ("full_comparison.png",["physical",*priority[:3]],np.s_[:,:],12,3.0,4)]
    if {"S","A","SA"}<=imgs.keys():
        figures.append(("refinement_crops.png",["P","S","A","SA"],crop,14,1.8,4))
    for file,variant,sl,w,h,ncol in figures:
        ncol=min(len(variant),ncol)
        groups=[variant[i:i+ncol] for i in range(0,len(variant),ncol)]
        fig,axs=plt.subplots(2*len(groups),ncol,figsize=(w,2*h*len(groups)),layout="constrained",squeeze=False)
        for j,group in enumerate(groups):
            for col in range(ncol):
                top,bottom=axs[2*j:2*j+2,col]
                top.set_axis_off()
                bottom.set_axis_off()
                if col>=len(group):
                    continue
                label=group[col]
                top.imshow(tone(raw[label][sl]),interpolation="none")
                top.set_title(LABELS.get(label,"Physical HIGH"))
                residual=(raw[label][sl]-ref[sl])@LUMA
                scale=bottom.imshow(residual,vmin=-CAP,vmax=CAP,cmap="RdBu_r",interpolation="none")
                bottom.set_title("Signed residual")
        fig.colorbar(scale,ax=axs.ravel().tolist(),shrink=.64,label="Signed linear HDR luminance")
        fig.suptitle("WL101F · one exposure and residual scale")
        fig.savefig(out/file,dpi=145)
        plt.close(fig)

    fig,axis=plt.subplots(1,2,figsize=(12,4.6),layout="constrained")
    x=np.array([np.mean(row["range_m"]) for row in profile])
    axis[0].plot(x,[row["reference"] for row in profile],".-",label="Physical HIGH",color=colors["physical"])
    for k in priority:
        residual=np.array([row["residuals"][k]["mean"] for row in profile])
        axis[1].plot(x,residual,".-",label=LABELS[k],color=colors[k])
        axis[0].plot(x,[row["means"][k] for row in profile],".-",label=LABELS[k],color=colors[k])
        if k in ("uniform","P","SA"):
            a,b=np.asarray([row["residuals"][k]["ci95"] for row in profile]).T
            axis[1].fill_between(x,a,b,alpha=.13,color=colors[k])
    for i,ax in enumerate(axis):
        ax.axvspan(.10,.14,color="#bc7040",alpha=.11)
        ax.axvspan(.26,.40,color="#2b4260",alpha=.07)
        ax.set_xlabel("Foreground floor radius (m)")
        ax.set_ylabel(("HIGH luminance","Signed residual")[i])
        ax.grid(alpha=.22)
    axis[1].axhline(0,color="#58616b",lw=.8)
    axis[0].legend(loc="best",fontsize=8,ncol=2)
    axis[1].legend(loc="best",fontsize=8,ncol=2)
    fig.savefig(out/"radial_profile.png",dpi=150)
    plt.close(fig)


def report(raw, wl101e, refs, history, out):
    run_path=raw/"run_manifest.json"
    if not run_path.exists(): run_path=raw/"progress.json"
    run=json.loads(run_path.read_text())
    assert sha(HERE/"protocol/attribution.json")==run["protocol_sha256"]
    baseline=json.loads((wl101e/"run_manifest.json").read_text())
    assert sha(wl101e/"run_manifest.json")==run["baseline_manifest_sha256"]
    for name,rec in run.get("files",{}).items():
        assert sha(raw/name)==rec["sha256"] and (raw/name).stat().st_size==rec["bytes"]
    for filename in ("REF_0_0.npz","REF_0_1.npz","MAP_0.npz"):
        assert sha(refs/filename)==baseline["input_sha256"][filename]
    regions,bands,_=make_regions(refs/"MAP_0.npz")
    prior=json.loads((history/"run_manifest.json").read_text())
    assert sha(history/"run_manifest.json")==baseline["prior_manifest_sha256"]

    def read(path,key,digest=None):
        if digest is not None: assert sha(path)==digest
        with np.load(path) as f:
            x=f[key]
        assert x.shape==(512,512,3) and np.isfinite(x).all() and x.min()>=0
        return x

    physical=np.stack([read(refs/f"REF_0_{i}.npz","h") for i in range(2)]).astype(np.float64)
    learned=np.stack([read(history/r["path"],"hl",r["sha256"]) for r in prior["matched"]])
    uniform=np.empty((4,2,512,512,3),dtype=np.float32)
    for row in baseline["readouts"]:
        uniform[row["oracle_index"],row["camera_half"]]=read(wl101e/row["file"],"hl",row["sha256"])
    arms={"learned":np.broadcast_to(learned,uniform.shape),"uniform":uniform}
    stages=run.get("stages",{})
    streams=stages.get("A",{}).get("streams",[])
    if len(streams)==2:
        query=np.stack([read(raw/r["file"],"query") for r in streams])
        arms["query"]=np.broadcast_to(query,uniform.shape)
    for stage,names in (("B",("P","Q")),("C",("S","A","SA"))):
        streams=stages.get(stage,{}).get("streams",[])
        if len(streams)!=2:
            continue
        for name in names:
            with np.load(raw/streams[0]["file"]) as f:
                assert f[f"{name}_0"].shape==(512,512,3)
            arms[name]=np.stack([np.stack([read(raw/row["file"],f"{name}_{i}") for row in streams]) for i in range(4)])
    means={name:grid.mean((0,1),dtype=np.float64) for name,grid in arms.items()}
    ref=physical.mean(0)

    def observations(mask):
        ref_mu=luma(physical,mask)
        values={name:luma(a,mask) for name,a in arms.items()}
        return {"reference":float(ref_mu.mean()),"means":{k:float(v.mean()) for k,v in values.items()},
                "residuals":{name:interval(v-ref_mu) for name,v in values.items()}}
    detail={}
    for label,mask in regions.items():
        detail[label]={"pixels":int(mask.sum()),**observations(mask),
                       "errors":{name:pixel_score(img,ref,mask) for name,img in means.items()}}
    radial=[{"range_m":[a,b],"pixels":int(mask.sum()),**observations(mask)}
            for a,b,mask in zip(C["regions"]["front_radial_edges_m"],C["regions"]["front_radial_edges_m"][1:],bands)]
    near,flank=regions["near"],regions["flank"]
    values={k:luma(v,near)-luma(v,flank) for k,v in arms.items()}
    baseline_ref=luma(physical,near)-luma(physical,flank)
    contrast={k:interval(v-baseline_ref) for k,v in values.items()}
    if stages.get("B",{}).get("residuals") and "P" in contrast:
        pass_B=stages["B"]["residuals"]["P"]
        assert abs(contrast["P"]["mean"]-pass_B["concentration"]["mean"])<1e-10
        assert abs(detail["near"]["residuals"]["P"]["mean"]-pass_B["near"]["mean"])<1e-10
    balances={}
    if {"uniform","Q","P"}<=means.keys():
        u,q,p=[means[k] for k in ("uniform","Q","P")]
        balances["projection_rgb_max_abs"]=float(np.abs((u-ref)-((u-q)+(q-p)+(p-ref))).max())
    if {"P","S","A","SA"}<=means.keys():
        p,s,a,sa=[means[k] for k in ("P","S","A","SA")]
        balances["refinement_rgb_max_abs"]=float(np.abs((p-sa)-((p-s)+(p-a)-(p-s-a+sa))).max())
    assert all(value<=1e-12 for value in balances.values())
    out.mkdir(parents=True,exist_ok=True)
    result={"source_run_manifest_sha256":sha(run_path),
            "verdict":run.get("verdict","STOPPED"),"stop":run.get("stop"),
            "gate_stages":stages,"regions":detail,"contrast":contrast,"radial_bands":radial,
            "usage":usage(raw,run),"decomposition_max_absolute":balances}
    write_json(out/"metrics.json",result)
    np.savez_compressed(raw/"review_maps.npz",physical=ref,**means,
                        **{f"{name}_residual":rgb-ref for name,rgb in means.items()})
    plots(out,means,ref,radial)
    print(result["verdict"],flush=True)


if __name__=="__main__":
    arg=argparse.ArgumentParser(description=__doc__)
    for name in ("raw","wl101e","references","history","output"):
        arg.add_argument(name,type=Path)
    a=arg.parse_args()
    report(a.raw,a.wl101e,a.references,a.history,a.output)
