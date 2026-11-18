"""Offline reference scoring and review output for WL101B."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent/"ecro_sss101a"))
from common import P as WL101A, masks as geometry_masks  # noqa: E402

PROTOCOL = HERE/"protocol/reconstruction.json"
P = json.loads(PROTOCOL.read_text())


def sha(path):
    with path.open("rb") as stream:
        h = hashlib.sha256()
        for data in iter(lambda: stream.read(1024*1024), b""):
            h.update(data)
    return h.hexdigest()


def tone(hdr):
    x = WL101A["display"]["exposure"] * np.maximum(hdr, 0)
    return (x/(1+x))**(1/2.2)


def rmse(x):
    return float(np.sqrt(np.mean(x*x, dtype=np.float64)))


def score(img, ref, mask):
    x, y = img[mask], ref[mask]
    return {
        "hdr_rmse": rmse(x-y),
        "hdr_relative_rmse": rmse(x-y)/rmse(y),
        "hdr_relative_l1": float(np.abs(x-y).sum(dtype=np.float64)/np.abs(y).sum(dtype=np.float64)),
        "ldr_rmse": rmse(tone(x)-tone(y)),
    }


def gradient(img, ref, mask):
    # Include edges crossing the ROI boundary, including the sphere silhouette.
    horizontal = mask[:, 1:] | mask[:, :-1]
    vertical = mask[1:] | mask[:-1]
    lhs = np.concatenate(((img[:, 1:]-img[:, :-1])[horizontal],
                          (img[1:]-img[:-1])[vertical]), axis=0)
    rhs = np.concatenate(((ref[:, 1:]-ref[:, :-1])[horizontal],
                          (ref[1:]-ref[:-1])[vertical]), axis=0)
    power = rmse(rhs)
    return {
        "gradient_relative_rmse": rmse(lhs-rhs)/power,
        "gradient_rms_ratio": rmse(lhs)/power,
    }


def mean_dict(items):
    return {name: float(np.mean([x[name] for x in items])) for name in items[0]}


def refs_and_masks(path, output, run):
    frozen = json.loads((HERE/"protocol/reference_hashes.json").read_text())
    for name, digest in frozen.items():
        assert sha(path/name) == digest, name
    references, regions = {}, {}
    for geometry, light, sid in run["tour"]:
        images = []
        for half in (0, 1):
            archive = path/f"REF_{sid}_{half}.npz"
            meta = json.loads(archive.with_suffix(".json").read_text())
            assert meta["geom"] == geometry and meta["light"] == light
            assert meta["spp"] == 4096 and meta["seed"] == 101100+10*sid+half
            with np.load(archive) as packet:
                images.append(packet["e1"] + packet["e2"] + packet["h"])
        references[sid] = (np.stack(images), (images[0]+images[1])*0.5)
        regions[sid] = geometry_masks(output/f"guide_{sid}_A.npz")
    return references, regions


def read_group(raw, label, spp, rep):
    rows = json.loads((raw/f"{label}_{spp}spp_r{rep}.json").read_text())
    assert len(rows) == (8 if label=="tour" else 24)
    for row in rows:
        if "file" in row:
            path = raw/row["file"]
            assert sha(path) == row["sha256"], path
            with np.load(path) as data:
                yield row, {k: data[k] for k in data.files}


def phase_metrics(rows):
    roi_names = list(rows[0]["raw"])
    result = {}
    for arm in ("raw","reconstructed"):
        result[arm] = {}
        for roi in roi_names:
            metrics = [row[arm][roi] for row in rows]
            stats = mean_dict(metrics)
            stats["hdr_rmse_pooled"] = float(np.mean([x["hdr_rmse"]**2 for x in metrics])**0.5)
            stats["hdr_relative_rmse_pooled"] = float(np.mean([x["hdr_relative_rmse"]**2 for x in metrics])**0.5)
            result[arm][roi] = stats
    return result


def decomposition(samples, refs, mask):
    reference_halves, reference = refs
    values = samples[:, mask].astype(np.float64)
    mean = values.mean(axis=0)
    variance = np.mean(values.var(axis=0, ddof=1))
    a, b = reference_halves[:, mask].astype(np.float64)
    noise = np.mean(((a-b)*0.5)**2)
    stable = np.mean((mean-a)*(mean-b))-variance/len(values)
    power = np.mean(reference[mask].astype(np.float64)**2)
    return {
        "interreplicate_variance": float(variance),
        "stable_mse_cross_debiased": float(stable),
        "reference_mean_mse": float(noise),
        "stochastic_relative_rms": float(np.sqrt(variance/power)),
        "stable_relative_rms": float(np.sqrt(stable/power)) if stable >= 0 else None,
        "reference_mean_relative_noise": float(np.sqrt(noise/power)),
    }


def evaluate_spp(raw, spp, refs, rois, repeats):
    frame_metrics = {"cold": [], "steady": [], "tour": []}
    samples = {"cold": [], "steady": [], "tour": {sid: [] for sid in refs}}
    flicker, camera, changes = [], [], []
    preview = {"tour": [], "camera": []}
    arms = ("raw", "reconstructed")

    def scoring(data, sid):
        _, reference = refs[sid]
        result = {arm: {} for arm in arms}
        for arm in arms:
            for tag, roi in rois[sid].items():
                result[arm][tag] = score(data[arm], reference, roi)
                if tag in ("sphere_boundary", "contact_shadow"):
                    result[arm][tag].update(gradient(data[arm], reference, roi))
        return result

    for rep in range(repeats):
        static = {arm: [] for arm in arms}
        for row, data in read_group(raw, "static", spp, rep):
            t, sid = row["frame"], row["state"]
            if t == 0:
                assert np.all(data["count"] == 1)
                frame_metrics["cold"].append(scoring(data, sid))
                samples["cold"].append([data[arm] for arm in arms])
                if rep == 0:
                    preview["cold"] = data
            if t in P["evaluation"]["steady_frames"]:
                for arm in arms:
                    static[arm].append(data[arm])
                frame_metrics["steady"].append(scoring(data, sid))
            if t == P["evaluation"]["steady_frames"][-1]:
                samples["steady"].append([data[arm] for arm in arms])
                if rep == 0:
                    preview["steady"] = data
        flicker.append({
            arm: {tag: float(np.var(np.stack(static[arm])[:,mask], axis=0,
                                     dtype=np.float64, ddof=1).mean())
                  for tag, mask in rois[0].items()} for arm in arms
        })
        previous = None
        for row, data in read_group(raw, "tour", spp, rep):
            sid = row["state"]
            assert row["exact_reset"] and np.all(data["count"] == 1)
            frame_metrics["tour"].append(scoring(data, sid))
            samples["tour"][sid].append([data[arm] for arm in arms])
            if rep == 0:
                preview["tour"].append((sid, data))
            if previous is not None:
                old_sid, old = previous
                delta = refs[sid][1] - refs[old_sid][1]
                change_score = {arm: {} for arm in arms}
                for arm in arms:
                    predicted = data[arm] - old[arm]
                    for tag, mask in rois[sid].items():
                        d, p = delta[mask].astype(np.float64), predicted[mask].astype(np.float64)
                        v = np.mean(d*d)
                        change_score[arm][tag] = {
                            "signed_change_relative_rmse": rmse(p-d)/max(np.sqrt(v),1e-12),
                            "signed_change_slope": float(np.mean(p*d)/max(v,1e-24)),
                            "signed_change_error_hdr": rmse(p-d),
                        }
                changes.append({"state_from": old_sid, "state_to": sid, "replica": rep, **change_score})
            previous = sid, data
        for row, data in read_group(raw, "camera", spp, rep):
            t = row["frame"]
            if rep == 0:
                preview["camera"].append((row, data))
            if t in (16,20):
                mask = data["disoccluded"]
                assert np.any(mask)
                reuse = data["count"] > 1
                d = data["reconstructed"]-data["spatial_reset"]
                record = {
                    "frame": t, "replica": rep, "camera": row["camera"],
                    "disoccluded_pixels": int(np.sum(mask)),
                    "disoccluded_reuse": float(reuse[mask].mean()),
                    "valid_history_fraction": float(reuse.mean()),
                    "disoccluded_history_effect_rmse_hdr": rmse(d[mask]),
                    "all_pixel_history_effect_rmse_hdr": rmse(d),
                }
                if row["camera"]=="A":
                    ref = refs[0][1]
                    record["return_disoccluded_hdr_rmse"] = {
                        k: rmse(data[k][mask]-ref[mask])
                        for k in ("raw","reconstructed","spatial_reset")
                    }
                camera.append(record)

    result = {phase: phase_metrics(rows) for phase, rows in frame_metrics.items()}
    result["static_flicker_variance"] = {
        arm: {tag: float(np.mean([x[arm][tag] for x in flicker])) for tag in rois[0]}
        for arm in arms}
    result["camera_transitions"] = camera
    result["tour_signed_change"] = changes
    result["per_frame_scores"] = frame_metrics
    result["residual"] = {}
    for stage in ("cold","steady","tour"):
        by_state = samples[stage] if stage=="tour" else {0:samples[stage]}
        result["residual"][stage] = {}
        for sid, items in by_state.items():
            frames = np.stack(items)
            result["residual"][stage][sid] = {
                arm: {k: decomposition(frames[:,i], refs[sid], mask)
                      for k,mask in rois[sid].items()} for i,arm in enumerate(arms)}
    return result, preview


def aggregate_timing(timing):
    result = {}
    for spp, record in timing.items():
        result[spp] = {}
        for arm in ("raw","reconstructed"):
            row = record[arm]
            ms = [e["ms"] for e in row["events"]]
            gpu = [e["denoiser_gpu_ms"] for e in row["events"]]
            result[spp][arm] = {
                "median_ms": float(np.median(ms)),
                "p95_ms": float(np.percentile(ms,95)),
                "denoiser_gpu_median_ms": float(np.median(gpu)),
                "denoiser_gpu_p95_ms": float(np.percentile(gpu,95)),
                "foreign_gpu_pids": row["foreign_gpu_pids"],
                "peak_torch_mib": row["peak_torch_bytes"]/2**20,
                "reconstruction_resident_mib": row["reconstruction_resident_bytes"]/2**20,
            }
        camera = record["camera_only_ms"]
        result[spp]["camera_only"] = {"median_ms":float(np.median(camera)),
                                       "p95_ms":float(np.percentile(camera,95))}
    return result


def decisions(quality, timing):
    results = {}
    for spp, rows in quality.items():
        checks = {}
        for phase, reduction in (("cold",0.25),("steady",0.4),("tour",0.25)):
            for roi in ("sphere","floor","contact_shadow"):
                raw, recon = rows[phase]["raw"][roi], rows[phase]["reconstructed"][roi]
                loss_ratio = (recon["hdr_rmse_pooled"]/raw["hdr_rmse_pooled"])**2
                checks[f"{phase}_{roi}_quality"] = bool(loss_ratio <= 1-reduction and recon["ldr_rmse"] < raw["ldr_rmse"])
        for phase in ("steady","tour"):
            for roi in ("sphere_boundary","contact_shadow"):
                raw, recon = rows[phase]["raw"][roi], rows[phase]["reconstructed"][roi]
                checks[f"{phase}_{roi}_detail"] = bool(
                   recon["gradient_relative_rmse"] < raw["gradient_relative_rmse"] and
                   0.8 <= recon["gradient_rms_ratio"] <= 1.2)
        checks["static_flicker"] = all(
            rows["static_flicker_variance"]["reconstructed"][roi] <=
            0.5*rows["static_flicker_variance"]["raw"][roi] for roi in ("sphere","floor"))
        checks["tour_exact_reset"] = True
        camera = rows["camera_transitions"]
        checks["camera_history_validity"] = max(r["disoccluded_reuse"] for r in camera) <= 0.005
        checks["camera_return_quality"] = all(
            r["return_disoccluded_hdr_rmse"]["reconstructed"] <=
            1.10*r["return_disoccluded_hdr_rmse"]["spatial_reset"]
            for r in camera if r["camera"]=="A")
        checks["latency"] = (not timing[spp]["reconstructed"]["foreign_gpu_pids"] and
                             timing[spp]["reconstructed"]["median_ms"] <= P["acceptance"]["budget_ms"])
        results[spp] = {"passed":all(checks.values()),"checks":checks}
    return results


def plot(out, previews, refs, rois, quality):
    for spp, data in previews.items():
        tiles = [
            ("Physical reference", refs[0][1]),
            ("Raw · first frame", data["cold"]["raw"]),
            ("Reconstructed · first frame", data["cold"]["reconstructed"]),
            ("Raw · frame 24", data["steady"]["raw"]),
            ("Reconstructed · frame 24", data["steady"]["reconstructed"]),
        ]
        yy, xx = np.nonzero(rois[0]["sphere"])
        crop = np.s_[max(0,yy.min()-16):min(P["resolution"],yy.max()+41),
                     max(0,xx.min()-16):min(P["resolution"],xx.max()+17)]
        fig, axes = plt.subplots(2,5,figsize=(15,6),layout="constrained")
        for j,(title,img) in enumerate(tiles):
            axes[0,j].imshow(tone(img))
            axes[1,j].imshow(tone(img[crop]))
            axes[0,j].set_title(title)
            axes[0,j].set_axis_off()
            axes[1,j].set_axis_off()
        fig.suptitle(f"WL101B · SSS cfg0 · {spp} spp · common HDR exposure")
        fig.savefig(out/f"static_{spp}spp.png",dpi=145)
        plt.close(fig)
        sequence(out/f"tour_{spp}spp.gif", [
            (f"WL101 SSS tour · state {sid} · {spp} spp · history reset", [
                ("Physical reference",refs[sid][1]),("Raw",item["raw"]),
                ("Reconstructed",item["reconstructed"])])
            for sid,item in data["tour"]])
        sequence(out/f"camera_{spp}spp.gif", [
            (f"Camera {row['camera']} · frame {row['frame']+1} · {spp} spp",[
                ("Raw",item["raw"]),("Reconstructed",item["reconstructed"]),
                ("Same-frame reset",item.get("spatial_reset"))])
            for row,item in data["camera"]])
    figure, axes = plt.subplots(1,2,figsize=(10,4),layout="constrained")
    for ax,(spp,row) in zip(axes,quality.items()):
        groups = ("cold","steady","tour")
        x = np.arange(len(groups))
        for label, delta, color in (("raw",-0.17,"#5a7077"),("reconstructed",0.17,"#b85135")):
            val = [100*row[g][label]["sphere"]["hdr_relative_rmse_pooled"] for g in groups]
            ax.bar(x+delta,val,width=0.34,color=color,label=label.title())
        ax.set_xticks(x,labels=["Cold","Steady","Geometry edit"])
        ax.set_title(f"{spp} spp")
        ax.set_ylabel("Sphere relative HDR RMSE (%)")
    axes[0].legend()
    figure.savefig(out/"quality.png",dpi=150)
    plt.close(figure)


def sequence(path, items):
    frames = []
    for title, tiles in items:
        frame = Image.new("RGB",(256*len(tiles),294),"#182127")
        draw = ImageDraw.Draw(frame)
        draw.text((6,5),title,fill="white")
        for j,(label,img) in enumerate(tiles):
            if img is not None:
                pixels = np.round(tone(img)*255).astype(np.uint8)
                frame.paste(Image.fromarray(pixels).resize((256,256),Image.Resampling.LANCZOS),(j*256,38))
            draw.text((j*256+6,22),label,fill="white")
        frames.append(frame)
    frames[0].save(path,save_all=True,append_images=frames[1:],duration=400,loop=0)


def main(raw, ref_path, out):
    run = json.loads((raw/"run_manifest.json").read_text())
    assert run["protocol_sha256"] == sha(PROTOCOL)
    references, rois = refs_and_masks(ref_path, raw, run)
    quality, previews = {}, {}
    for n in P["spp"]:
        quality[str(n)], previews[str(n)] = evaluate_spp(raw,n,references,rois,P["evaluation"]["repeats"])
        print(f"Scored {n} spp.",flush=True)
    timing = aggregate_timing(json.loads((raw/"timing.json").read_text()))
    verdict = decisions(quality,timing)
    out.mkdir(parents=True,exist_ok=True)
    result = {"run_commit":run["commit"],"run_manifest_sha256":sha(raw/"run_manifest.json"),
              "verdict":verdict,"timing":timing,"quality":quality}
    (out/"metrics.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    plot(out,previews,references,rois,quality)
    print(f"Verdict {verdict}",flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw",type=Path)
    parser.add_argument("references",type=Path)
    parser.add_argument("reviewer",type=Path)
    args=parser.parse_args()
    main(args.raw,args.references,args.reviewer)
