"""Generate Amendment 1 reviewer figures and descriptive summaries from stored evidence."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from figures104d import comparison_plot, maps, read, response_plot

P = read(HERE.parent / "protocol.json")
ARMS = ("A", "B", "C", "D", "ER_DF", "ER_L")
ROIS = ("objects", "highlight_core", "highlight_ring", "grazing")


def reference_plot(ref, out):
    groups = ref["per_group"]
    names = [f"{row['theta_deg']:g}° / {row['phi_deg']:g}°" for row in groups] + ["Material mean"]
    initial = [row["original_eshape_half_noise"] for row in groups] + [ref["original_target"]["band_shape"]]
    extended = [row["amended_eshape_half_noise"] for row in groups] + [ref["amended_target"]["band_shape"]]
    fig, ax = plt.subplots(figsize=(9.7, 3.8), layout="constrained")
    x = np.arange(len(names))
    ax.axvspan(5.5, 6.5, color="#e6edf1")
    ax.bar(x - 0.16, initial, width=0.31, color="#697f91", label="4,096 walks")
    ax.bar(x + 0.16, extended, width=0.31, color="#26786a", label="16,384 walks")
    maximum = P["response"]["max_band_shape_noise"]
    ax.hlines(maximum, 5.55, 6.45, color="#ad694d", linestyle="--", label="Frozen mean limit")
    ax.set_ylabel("Shape error between reference halves / 2")
    ax.set_xticks(x, names, rotation=19, ha="right")
    ax.set_title("Independent reference precision · test83_hard_76")
    ax.margins(y=0.19)
    ax.legend(frameon=False, ncol=3)
    fig.savefig(out / "reference_precision.png", dpi=155)
    plt.close(fig)


def summarize(response, scenes, rows):
    angles = response["angles_deg"]
    stages = ["development", "held_out"]
    counts = {name: np.array([row["cohort"] == name for row in rows]) for name in stages}
    metrics = ("band_shape", "grazing_shape", "overall_albedo", "overall_el1")
    arm_summary, contrasts, failures = {}, {}, []
    for arm in ARMS:
        result = response["arms"][arm]
        arm_summary[arm] = {
            name: {key: float(np.median(np.asarray(result[key])[mask])) for key in metrics}
            for name, mask in counts.items()
        }
        arm_summary[arm]["runtime_median_eval_seconds"] = float(np.median(
            [row["eval_wall_seconds_including_transfer"] for row in result["physical"]]
        ))
    for row in rows:
        image = row["image"]
        if image not in scenes:
            continue
        per_scene = scenes[image]
        for arm, data in per_scene["arms"].items():
            rms = data["stable_rms"]["objects"]
            half = (rms["hi"] - rms["lo"]) / 2
            if half > P["scene"]["precision_rms_half_width"]:
                failures.append({"image": image, "arm": arm, "metric": "object_total_rms_half_width", "half_width": half})
            for roi in ROIS:
                for event, ci in data["signed"].get(roi, {}).items():
                    half = (ci["hi"] - ci["lo"]) / 2
                    if half > P["scene"]["precision_energy_half_width"]:
                        failures.append({"image": image, "arm": arm, "metric": "signed_energy_half_width",
                                         "roi": roi, "event": event, "half_width": half})
    for name, candidate, control in (("C-A", "C", "A"), ("B-A", "B", "A"), ("D-A", "D", "A"),
                                     ("D-B", "D", "B"), ("D-C", "D", "C")):
        a, b = response["arms"][control], response["arms"][candidate]
        gain = {key: np.asarray(a[key]) - np.asarray(b[key]) for key in metrics}
        profile = (
            np.asarray(a["probes"]["eshape"]) - np.asarray(b["probes"]["eshape"])
        ).reshape(len(rows), len(angles), -1).mean(axis=2)
        value = {cohort: {
            **{key: float(np.median(arr[mask])) for key, arr in gain.items()},
            "shape_gain_by_angle": dict(zip(angles, np.median(profile[mask], axis=0).tolist())),
        } for cohort, mask in counts.items()}
        if len(scenes) == len(rows):
            difference = np.asarray([
                scenes[row["image"]]["arms"][control]["stable_rms"]["objects"]["point"]
                - scenes[row["image"]]["arms"][candidate]["stable_rms"]["objects"]["point"] for row in rows
            ])
            for cohort, mask in counts.items():
                value[cohort]["scene_total_rms_gain"] = float(np.median(difference[mask]))
        contrasts[name] = value
    coefficient = np.array([[0.5, 0.5, -0.5, -0.5], [0.5, -0.5, 0.5, -0.5], [-1.0, 1.0, 1.0, -1.0]])
    angular = coefficient @ np.array([response["arms"][arm]["band_shape"] for arm in ARMS[:4]])
    rendered = None
    if len(scenes) == len(rows):
        rendered = coefficient @ np.array([
            [scenes[row["image"]]["arms"][arm]["stable_rms"]["objects"]["point"] for row in rows]
            for arm in ARMS[:4]
        ])
    factorial = {}
    for i, factor in enumerate(("data", "objective", "interaction")):
        factorial[factor] = {
            cohort: {"band_shape_gain": float(np.median(angular[i, mask]))}
            for cohort, mask in counts.items()
        }
        if rendered is not None:
            for cohort, mask in counts.items():
                factorial[factor][cohort]["scene_total_rms_gain"] = float(np.median(rendered[i, mask]))
    return {"response_and_runtime": arm_summary, "descriptive_contrasts_positive_is_better": contrasts,
            "descriptive_factorial_effects_positive_is_better": factorial,
            "scene_precision_failures": failures, "completed_scenes": len(scenes)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    ref = read(args.raw / "reference_precision.json")
    reference_plot(ref, args.output)
    if (args.raw / "response.json").exists():
        response = read(args.raw / "response.json")
        rows = read(args.raw / "materials.json")
        if all(key in response["arms"] for key in ARMS):
            response_plot(response, rows, args.output)
            path = args.raw / "scenes.json"
            scenes = read(path) if path.exists() else {}
            if len(scenes) == len(rows):
                comparison_plot(response, scenes, rows, args.output)
            summary = summarize(response, scenes, rows)
            (args.output / "review_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
            for image in scenes:
                maps(args.raw, image, args.output)
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(args.output.iterdir())
              if p.is_file() and p.name != "review_manifest.json"}
    manifest = {"script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "output_sha256": hashes}
    (args.output / "review_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
