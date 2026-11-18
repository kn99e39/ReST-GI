"""Read-only HDR contact attribution and visual evidence for the frozen WL101C run."""
import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import convolve
from scipy.stats import t

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent/"ecro_sss101a"))
from common import P as A, masks as historical_masks, sha, write_json

P = json.loads((HERE/"protocol/contact.json").read_text())
COMPONENTS = ("full", "low", "high")
LUMA = np.array(P["intensity"]["luminance_rgb"])
LABELS = ("Full", "Physical LOW", "HIGH")
CAP = A["display"]["signed_error_max"]


def load_frame(path, physical=False, recorded=False):
    with np.load(path) as z:
        low = z["e1"].astype(np.float64) + z["e2"]
        high = z["h" if physical else "hl"].astype(np.float64)
        full = low + high
        if recorded:
            assert np.max(np.abs(z["low"]-low)) <= 2e-6
            assert np.max(np.abs(z["full"]-full)) <= 4e-6
        frame = np.stack([full, low, high])
    assert frame.shape == (3, 512, 512, 3) and np.isfinite(frame).all() and frame.min() >= 0
    return frame


def make_regions(path):
    regions = historical_masks(path)
    with np.load(path) as m:
        position = m["p"]
    center = np.asarray(A["sphere"]["center"])
    radius = np.linalg.norm(position[..., :2]-center[:2], axis=-1)
    front = regions["floor"] & (position[..., 1] <= center[1])
    near_radius = P["regions"]["near_radius_m"]
    lo, hi = P["regions"]["flank_radius_m"]
    regions["near"] = front & (radius < near_radius)
    regions["flank"] = front & (radius >= lo) & (radius < hi)
    assert regions["near"].sum() == 1318 and regions["flank"].sum() == 6042
    assert not np.any(regions["near"] & regions["flank"])
    assert np.all(regions["contact_shadow"][regions["near"]])
    edges = P["regions"]["front_radial_edges_m"]
    bands = [front & (radius >= a) & (radius < b) for a,b in zip(edges,edges[1:])]
    assert all(m.any() for m in bands)
    return regions, bands, position


def welch(x, y, family=1):
    x, y = np.asarray(x), np.asarray(y)
    estimate = x.mean()-y.mean()
    u, v = x.var(ddof=1)/len(x), y.var(ddof=1)/len(y)
    se = np.sqrt(u+v)
    df = (u+v)**2/(u*u/(len(x)-1)+v*v/(len(y)-1)) if se else None
    margin = t.ppf(0.975,df)*se if df else 0
    lower = estimate-t.ppf(1-0.05/family,df)*se if df else estimate
    return {"difference":float(estimate), "ci95":[float(estimate-margin),float(estimate+margin)],
            "one_sided_family_lower":float(lower), "degrees_of_freedom":float(df) if df else None,
            "new_mean":float(x.mean()), "reference_mean":float(y.mean()),
            "new_seed_sd":float(x.std(ddof=1)), "reference_mean_se":float(np.sqrt(v))}


def region_metrics(fresh, refs, mask):
    observed = fresh[:, :, mask]
    halves = refs[:, :, mask]
    mean, ref = observed.mean(0), halves.mean(0)
    difference = mean-ref
    half_difference = (halves[0]-halves[1])*0.5
    d0, d1 = observed[:4].mean(0)-halves[0], observed[4:].mean(0)-halves[1]
    cross = np.mean(d0*d1, axis=(1,2))
    pair_covariance = float(np.mean(d0[1]*d1[2]+d0[2]*d1[1]))
    assert abs(cross[0]-cross[1]-cross[2]-pair_covariance) <= 1e-12
    report = {}
    for i, name in enumerate(COMPONENTS):
        ref_rms = np.sqrt(np.mean(ref[i]**2))
        mean_ref_rgb = ref[i].mean(0)
        rgb = [welch(observed[:,i,:,ch].mean(1),halves[:,i,:,ch].mean(1)) for ch in range(3)]
        report[name] = {
            "ensemble_hdr_rmse":float(np.sqrt(np.mean(difference[i]**2))),
            "ensemble_relative_rmse":float(np.sqrt(np.mean(difference[i]**2))/ref_rms),
            "ensemble_relative_l1":float(np.abs(difference[i]).sum()/np.abs(ref[i]).sum()),
            "individual_hdr_rmse":np.sqrt(np.mean((observed[:,i]-ref[i])**2,axis=(1,2))).tolist(),
            "bias_rgb_hdr":difference[i].mean(0).tolist(),
            "bias_rgb_percent":(100*difference[i].mean(0)/mean_ref_rgb).tolist(),
            "bias_rgb_ci95": [x["ci95"] for x in rgb],
            "reference_mean_noise_relative_rms":float(np.sqrt(np.mean(half_difference[i]**2))/ref_rms),
            "prediction_seed_relative_sd":float(np.sqrt(np.mean(observed[:,i].var(axis=0,ddof=1)))/ref_rms),
            "stable_mse_cross":float(cross[i]),
            "stable_relative_rms_if_positive":float(np.sqrt(cross[i])/ref_rms) if cross[i]>=0 else None,
            "luma_bias":welch((observed[:,i]@LUMA).mean(1),(halves[:,i]@LUMA).mean(1)),
        }
    return {"pixels":int(mask.sum()), "components":report, "stable_mse_low_high_cross_terms":pair_covariance}


def roi_luminance(values, mask):
    return (values[:, :, mask]@LUMA).mean(-1)


def scalar_tests(fresh, refs, regions):
    near = roi_luminance(fresh,regions["near"])
    near_ref = roi_luminance(refs,regions["near"])
    flank = roi_luminance(fresh,regions["flank"])
    flank_ref = roi_luminance(refs,regions["flank"])
    checkpoints = {"near":(near,near_ref), "concentration":(near-flank,near_ref-flank_ref)}
    results = {}
    for endpoint,(pred, physical) in checkpoints.items():
        result = {"per_component":{}}
        for i, name in enumerate(COMPONENTS):
            result["per_component"][name] = welch(pred[:,i],physical[:,i],family=6)
            result["per_component"][name]["seed_minus_reference_halves"] = (
                pred[:,i,None]-physical[None,:,i]).tolist()
        result["high_minus_low"] = welch(pred[:,2]-pred[:,1],physical[:,2]-physical[:,1],family=6)
        result["low_minus_high"] = welch(pred[:,1]-pred[:,2],physical[:,1]-physical[:,2],family=6)
        result["low_plus_high"] = welch(pred[:,1]+pred[:,2],physical[:,1]+physical[:,2],family=6)
        results[endpoint] = result
    for candidate, contrast in (("high","high_minus_low"),("low","low_minus_high")):
        tests = []
        for result in results.values():
            full = result["per_component"]["full"]
            tests.extend([
                full["one_sided_family_lower"]>0,
                result[contrast]["one_sided_family_lower"]>0,
                result["low_plus_high"]["one_sided_family_lower"]>0,
            ])
            signs = np.asarray(result["per_component"][candidate]["seed_minus_reference_halves"])
            tests.append(bool((signs>0).all()))
        if all(tests):
            return "PERSISTENT " + candidate.upper(), results
    return "INCONCLUSIVE", results


def local_stats(img, mask, position):
    luma = img@LUMA
    kernel = np.ones((5,5))
    count = convolve(mask.astype(np.float64),kernel,mode="constant")
    box = convolve(np.where(mask,luma,0),kernel,mode="constant")/np.maximum(count,1)
    y, x = np.unravel_index(np.argmax(np.where(mask & (count>=10),box,-np.inf)),mask.shape)
    return {"maximum_local_mean":float(box[y,x]), "maximum_xy":[int(x),int(y)],
            "maximum_world_xyz_m":position[y,x].tolist(),
            "near_pixel_p95":float(np.percentile(luma[mask],95))}


def profile(fresh, refs, bands):
    edges = P["regions"]["front_radial_edges_m"]
    return [{"range_m":[a,b], "pixels":int(mask.sum()),
             "components":{k:welch(x,y) for k,x,y in
                zip(COMPONENTS,roi_luminance(fresh,mask).T,roi_luminance(refs,mask).T)}}
             for a,b,mask in zip(edges,edges[1:],bands)]


def tone(x):
    v = np.maximum(x,0)*A["display"]["exposure"]
    return (v/(1+v))**(1/2.2)


def plot(out, fresh, refs, old, regions, radial):
    ref, representative, mean, historical = refs.mean(0),fresh[0],fresh.mean(0),old.mean(0)
    xa,ya,xb,yb = P["regions"]["crop_xyxy"]
    crop = np.s_[ya:yb,xa:xb]
    for filename, sl, title in (("full_comparison",np.s_[:,:],"Full image"),("contact_crops",crop,"Fixed contact crop")):
        fig, axes = plt.subplots(3,5,figsize=(15,9 if filename=="full_comparison" else 6),layout="constrained")
        for i, label in enumerate(LABELS):
            tiles = [(f"{label} · physical",ref[i],False),
                     (f"{label} · seed {P['seeds'][0]}",representative[i],False),
                     (f"{label} · single-seed residual",representative[i]-ref[i],True),
                     (f"{label} · new mean residual",mean[i]-ref[i],True),
                     (f"{label} · WL101A paired residual",historical[i]-ref[i],True)]
            for ax,(column,image,signed) in zip(axes[i],tiles):
                if signed:
                    scale = ax.imshow((image@LUMA)[sl],cmap="RdBu_r",vmin=-CAP,vmax=CAP, interpolation="none")
                else:
                    ax.imshow(tone(image[sl]),interpolation="none")
                ax.set_title(column, fontsize=8)
                ax.set_axis_off()
        fig.colorbar(scale,ax=axes.ravel().tolist(),shrink=0.72,label="ECRO − physical · linear HDR luminance")
        fig.suptitle(f"WL101C · {title} · raw 4096 spp per seed · shared exposure", fontsize=13)
        fig.savefig(out/f"{filename}.png",dpi=150)
        plt.close(fig)

    fig,axes=plt.subplots(2,4,figsize=(12,5),layout="constrained")
    for ax,seed,image in zip(axes.flat,P["seeds"],fresh):
        j = ax.imshow(((image[2]-ref[2])@LUMA)[crop],cmap="RdBu_r",vmin=-CAP,vmax=CAP,
                      interpolation="none")
        ax.set_title(str(seed))
        ax.set_axis_off()
    fig.suptitle("WL101C · HIGH signed residual · each independent 4096-spp render")
    fig.colorbar(j, ax=axes.ravel().tolist(),shrink=0.75,label="Linear HDR luminance")
    fig.savefig(out/"independent_seeds.png",dpi=150)
    plt.close(fig)

    fig,axes=plt.subplots(1,2,figsize=(11,4.3),layout="constrained")
    x = [np.mean(cell["range_m"]) for cell in radial]
    for ax in axes:
        ax.axvspan(0,0.18,alpha=0.09,color="#bb5400",label="Near ROI")
        ax.axvspan(0.26,0.40,alpha=0.12,color="#888888",label="Flank")
        ax.set_xlabel("Distance from support point on the visible foreground floor (m)")
        ax.grid(alpha=0.18)
    for k, color in (("full","#24262a"),("low","#00808f"),("high","#b24929")):
        rows = [one["components"][k] for one in radial]
        axes[0].plot(x,[r["reference_mean"] for r in rows],"--",color=color,label=f"{k.title()} physical")
        axes[0].plot(x,[r["new_mean"] for r in rows],".-",color=color,label=f"{k.title()} ECRO")
        axes[1].plot(x,[r["difference"] for r in rows],".-",color=color,label=k.title())
        low, high = np.asarray([r["ci95"] for r in rows]).T
        axes[1].fill_between(x,low,high,color=color,alpha=0.15)
    axes[0].set_ylabel("Mean radiance · linear HDR luminance")
    axes[1].set_ylabel("ECRO − physical · linear HDR luminance")
    axes[1].axhline(0,color="#777777",lw=0.7)
    axes[0].legend(fontsize=7,ncol=2)
    axes[1].legend(fontsize=8)
    fig.savefig(out/"floor_intensity_profile.png",dpi=150)
    plt.close(fig)

    fig,ax=plt.subplots(figsize=(6,5),layout="constrained")
    color=tone(ref[0])*0.7
    for k,tint in (("near",np.array([1,0.46,0.15])),("flank",np.array([0.15,0.67,0.95]))):
        mask=regions[k]
        color[mask]=0.45*color[mask]+0.55*tint
    ax.imshow(color)
    ax.set_title("Fixed geometry ROIs · near (orange), flank (blue)")
    ax.set_axis_off()
    fig.savefig(out/"regions.png",dpi=140)
    plt.close(fig)


def main(raw, archives, wl101a, out):
    manifest = json.loads((raw/"run_manifest.json").read_text())
    assert manifest["protocol_sha256"] == sha(HERE/"protocol/contact.json")
    assert manifest["table_bit_identical_to_archive"] and manifest["table_preserved_after_rendering"]
    assert manifest["table_sha256"] == P["table_sha256"] and manifest["scene_sha256"] == P["scene_sha256"]
    assert manifest["checkpoint_sha256"] == A["checkpoint_sha256"]
    assert [row["seed"] for row in manifest["frames"]] == P["seeds"]
    frames = []
    for row in manifest["frames"]:
        path=raw/row["file"]
        assert row["spp"] == P["spp"] and sha(path) == row["sha256"]
        frames.append(load_frame(path,recorded=True))
    fresh=np.stack(frames)
    for name in ("REF_0_0.npz","REF_0_1.npz","MAP_0.npz"):
        assert sha(archives/name) == A["references"]["sha256"][name]
    refs=np.stack([load_frame(archives/f"REF_0_{h}.npz",physical=True) for h in (0,1)])
    prior=json.loads((wl101a/"run_manifest.json").read_text())
    assert prior["scene_sha256"]==manifest["scene_sha256"]
    assert prior["response_sha256"]==manifest["table_sha256"]
    assert prior["checkpoint_sha256"]==manifest["checkpoint_sha256"]
    assert prior["configs_sha256"]==manifest["configs_sha256"]
    assert prior["active_optix_sha256"]==manifest["environment"]["optix_sha256"]
    rows=prior["matched"]
    assert [row["seed"] for row in rows] == A["references"]["seed_by_half"]
    for row in rows:
        assert sha(wl101a/row["path"])==row["sha256"]
    old=np.stack([load_frame(wl101a/row["path"]) for row in rows])
    historic_replay=prior["historical_replay"]
    replay_path=wl101a/historic_replay["path"]
    assert sha(replay_path)==historic_replay["sha256"]
    historic_independent=load_frame(replay_path)
    regions,bands,position=make_regions(archives/"MAP_0.npz")
    verdict,tests=scalar_tests(fresh,refs,regions)
    radial=profile(fresh,refs,bands)
    roi={key:region_metrics(fresh,refs,mask) for key,mask in regions.items()}
    near, flank=regions["near"], regions["flank"]
    mean, ref = fresh.mean(0), refs.mean(0)
    maxima, concentration, prior_intensity = {}, {}, {}
    for i,k in enumerate(COMPONENTS):
        maxima[k] = {"new_mean":local_stats(mean[i],near,position), "reference_mean":local_stats(ref[i],near,position),
                     "signed_residual":local_stats(mean[i]-ref[i],near,position),
                     "fresh_individual": [local_stats(frame[i],near,position) for frame in fresh]}
        concentration[k] = {"new_near_flank_ratio":float((mean[i][near]@LUMA).mean()/(mean[i][flank]@LUMA).mean()),
                            "ref_near_flank_ratio":float((ref[i][near]@LUMA).mean()/(ref[i][flank]@LUMA).mean())}
        historic_delta=old[:,i]-refs[:,i]
        prior_intensity[k] = {"matched_near_bias_per_half":(historic_delta[:,near]@LUMA).mean(1).tolist(),
                              "matched_concentration_excess_per_half":((historic_delta[:,near]@LUMA).mean(1)-
                                                                       (historic_delta[:,flank]@LUMA).mean(1)).tolist(),
                              "independent_seed101500_near_bias":float(((historic_independent[i]-ref[i])[near]@LUMA).mean())}
    result={"verdict":verdict,"render_commit":manifest["commit"],"run_manifest_sha256":sha(raw/"run_manifest.json"),
            "protocol_sha256":manifest["protocol_sha256"],"seeds":P["seeds"],"roi":roi,"tests":tests,
            "floor_profile":radial,"local_intensity":maxima,"concentration_ratios":concentration,
            "historical_WL101A":prior_intensity,"frames":manifest["frames"]}
    out.mkdir(parents=True,exist_ok=True)
    write_json(out/"metrics.json",result)
    np.savez_compressed(raw/"analysis_maps.npz",**regions,reference=ref,new_mean=mean,
                        new_residual=mean-ref,historical_matched_residual=old.mean(0)-ref)
    plot(out,fresh,refs,old,regions,radial)
    print(verdict,flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for arg in ("raw","archives","wl101a","reviewer"):
        parser.add_argument(arg,type=Path)
    args=parser.parse_args()
    main(args.raw,args.archives,args.wl101a,args.reviewer)
