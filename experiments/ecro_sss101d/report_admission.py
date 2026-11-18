"""Summarize the failed Oracle admission. The support render is a unit marker."""
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

SHAPES=("Floor","Ceiling","Back wall","Left wall","Right wall","Panel","Sphere","Box A","Box B")


def read(path):
    return json.loads(path.read_text())


def checked_npz(directory,row):
    path=directory/row["file"]
    assert sha(path)==row["sha256"], path
    return np.load(path)


def main(raw,archive,history,out):
    pre=read(raw/"preflight/progress.json")
    support=read(raw/"support/run_manifest.json")
    membership=read(raw/"membership/run_manifest.json")
    audit=pre["preflight"]
    assert support["stop"] and not support["oracle_tables"] and not support["readouts"]
    assert pre["learned_table_sha256"]==support["learned_table_sha256"]==membership["learned_table_sha256"]
    assert pre["scene_sha256"]==support["scene_sha256"]==membership["scene_sha256"]
    assert sha(raw/"preflight"/pre["learned"]["file"])==pre["learned"]["sha256"]
    assert sha(raw/"support"/support["learned"]["file"])==support["learned"]["sha256"]
    assert sha(raw/"membership"/membership["learned"]["file"])==membership["learned"]["sha256"]
    assert membership["membership_replay"]["source_probe_sha256"]==audit["probe_packet"]["sha256"]
    map_path=archive/"MAP_0.npz"
    assert sha(map_path)==A["references"]["sha256"][map_path.name]
    regions,_,_=make_regions(map_path)
    weights, summaries = [],[]
    for row in support["support_camera_test"]:
        with checked_npz(raw/"support",row) as data:
            w=data["weight"]
        assert np.isfinite(w).all() and np.min(w)>=0 and w.shape==(512,512,3)
        assert np.count_nonzero(w.max(-1))==row["nonzero_pixels"]
        weights.append(w)
        summaries.append({"seed":row["seed"],"spp":4096,"regions":{name:{
            "pixels":int(mask.sum()),"nonzero_pixels":int(np.count_nonzero(w[mask].max(-1))),
            "mean_unit_weight_rgb":w[mask].mean(0,dtype=np.float64).tolist(),
            "mean_unit_weight_rgb_pooled":float(w[mask].mean(dtype=np.float64)),
            "max_unit_weight":float(w[mask].max())} for name,mask in regions.items()}})
    with checked_npz(raw/"membership",membership["membership_replay"]) as all_hits:
        with checked_npz(raw/"support",support["support_slot_info"]) as slots:
            assert np.array_equal(slots["valid_count"],all_hits["valid"])
            assert np.array_equal(slots["misrouted_count"],all_hits["wrong"])
            shape=slots["patch_shape"]
            centers=slots["patch_center"]
        with checked_npz(raw/"preflight",audit["probe_packet"]) as field:
            assert np.array_equal(field["valid_count"],all_hits["valid"])
            assert np.array_equal(field["misrouted_count"],all_hits["wrong"])
        valid, wrong, crossing=all_hits["valid"],all_hits["wrong"],all_hits["cross_shape_matrix"]
        assert int(wrong.sum())==audit["misrouted_samples"] and int(np.count_nonzero(wrong))==audit["slots_with_misrouting"]
        fraction=wrong/valid
        p,b=np.unravel_index(np.argmax(fraction),fraction.shape)
        metrics={"verdict":"INCONCLUSIVE","stop_stage":"0 — physical slot membership",
            "attribution":None,"stop":support["stop"],"membership":membership["membership_replay"],
            "preflight":audit,"accepted_samples":int(valid.sum()),"nominal_samples":int(valid.size*audit["audit_spp"]),
            "misrouted_sample_fraction":float(wrong.sum()/valid.sum()),
            "worst_slot":{"patch":int(p),"bin":int(b),"shape":SHAPES[shape[p]],"patch_center_xyz_m":centers[p].tolist(),
                          "wrong":int(wrong[p,b]),"accepted":int(valid[p,b])},
            "by_expected_shape":{key:{"accepted":int(valid[shape==i].sum()),"wrong":int(wrong[shape==i].sum()),
                                      "marked_slots":int(np.count_nonzero(wrong[shape==i]))}
                                 for i,key in enumerate(SHAPES)},
            "camera_support":summaries}
    out.mkdir(parents=True,exist_ok=True)
    write_json(out/"metrics.json",metrics)

    prior=read(history/"run_manifest.json")
    assert prior["response_sha256"]==support["learned_table_sha256"]
    predicted, physical = [],[]
    for half,row in enumerate(prior["matched"]):
        with checked_npz(history,{"file":row["path"],"sha256":row["sha256"]}) as z:
            predicted.append(z["hl"].astype(np.float64))
        name=f"REF_0_{half}.npz"
        with checked_npz(archive,{"file":name,"sha256":A["references"]["sha256"][name]}) as z:
            physical.append(z["h"].astype(np.float64))
    t_hat,ref=np.mean(predicted,axis=0),np.mean(physical,axis=0)
    pooled=np.mean(weights,axis=0,dtype=np.float64).mean(-1)
    xa,ya,xb,yb=C["regions"]["crop_xyxy"]
    fig,axs=plt.subplots(2,4,figsize=(15,7),layout="constrained")
    for row,sl in zip(axs,(np.s_[:,:],np.s_[ya:yb,xa:xb])):
        row[0].imshow(tone(ref[sl]),interpolation="none")
        row[1].imshow(tone(t_hat[sl]),interpolation="none")
        diff=row[2].imshow((t_hat[sl]-ref[sl])@LUMA,vmin=-CAP,vmax=CAP,cmap="RdBu_r",interpolation="none")
        support_panel=row[3].imshow(pooled[sl],vmin=0,vmax=1,cmap="inferno",interpolation="none")
        for ax,label in zip(row,("Archived physical HIGH","Archived learned HIGH","Archived signed residual",
                                  "Readout of flagged-slot unit marker")):
            ax.set_title(label,fontsize=8)
            ax.set_axis_off()
    fig.colorbar(diff,ax=axs[:,2].tolist(),shrink=.72,label="Linear HDR luminance")
    fig.colorbar(support_panel,ax=axs[:,3].tolist(),shrink=.72,label="Transport weight for unit marker")
    fig.suptitle("WL101D · failed slot-membership gate · original camera paths")
    fig.savefig(out/"camera_support.png",dpi=150)
    plt.close(fig)

    fig,ax=plt.subplots(figsize=(8,6.6),layout="constrained")
    from matplotlib.colors import LogNorm
    color=ax.imshow(np.ma.masked_equal(crossing,0),norm=LogNorm(vmin=1,vmax=10000),cmap="YlOrRd")
    for i,j in zip(*np.nonzero(crossing)):
        ax.text(j,i,f"{crossing[i,j]:,}",ha="center",va="center",
                color="white" if crossing[i,j]>1000 else "#1d2228",fontsize=8)
    ax.set_xticks(range(len(SHAPES)),labels=SHAPES,rotation=42,ha="right")
    ax.set_yticks(range(len(SHAPES)),labels=SHAPES)
    ax.set_xlabel("Actual first-hit shape")
    ax.set_ylabel("Expected probe shape")
    ax.set_title("Accepted probe samples whose first hit changed shape")
    fig.colorbar(color,ax=ax,label="Samples (log scale)")
    fig.savefig(out/"shape_membership.png",dpi=150)
    plt.close(fig)
    print("INCONCLUSIVE: first-hit support admission failed.",flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ("raw","archives","historical","reviewer"):
        parser.add_argument(key,type=Path)
    a=parser.parse_args()
    main(a.raw,a.archives,a.historical,a.reviewer)
