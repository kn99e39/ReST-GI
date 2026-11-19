"""Create reviewer figures from the completed portions of a WL104D run."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib.colors import SymLogNorm
import matplotlib.pyplot as plt
import numpy as np

LUMA = np.array([0.2126, 0.7152, 0.0722])
ARMS = ("A", "B", "C", "D")
COLORS = ("#455c78", "#956ca0", "#2d8067", "#cc7743")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def coverage_plot(raw, out):
    fig, ax = plt.subplots(figsize=(8.3, 3.8), layout="constrained")
    colors = ["#53677e", "#da7840"]
    edges = np.arange(91)
    ax.axvspan(55, 65, color="#f5d3bd", alpha=0.55)
    for label, color in zip(("original", "covered"), colors):
        with np.load(raw / f"train_{label}.npz") as z:
            mu = z["wi"][:, :, 0, 2]
        hist, _ = np.histogram(np.degrees(np.arccos(np.clip(mu, -1, 1))), edges)
        ax.stairs(hist, edges, color=color, linewidth=1.55, label=label.capitalize())
    ax.set(xlim=(0, 90), xlabel="Incident polar angle (degrees)", ylabel="Fit groups per degree",
           title="Equal incident group budget with added support at 55–65°")
    ax.legend(frameon=False)
    fig.savefig(out / "angular_coverage.png", dpi=155)
    plt.close(fig)


def response_plot(report, rows, out):
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.08), sharex=True, sharey=True, layout="constrained")
    theta = report["angles_deg"]
    for ax, cohort in zip(axes, ("development", "held_out")):
        mask = [row["cohort"] == cohort for row in rows]
        ax.axvspan(55, 65, color="#eac6ab", alpha=0.4)
        for arm, color in zip(ARMS, COLORS):
            data = np.array(report["arms"][arm]["probes"]["eshape"])[mask].reshape(sum(mask), len(theta), -1).mean(axis=2)
            ax.plot(theta, np.median(data, axis=0), ".-", color=color, label=arm)
        historic = np.array(report["arms"]["ER_DF"]["probes"]["eshape"])[mask].reshape(sum(mask), len(theta), -1).mean(axis=2)
        ax.plot(theta, np.median(historic, axis=0), linestyle="--", color="#2a3138", label="WL83 ER_DF")
        ax.set_title("Development" if cohort == "development" else "Held out")
        ax.set_xlabel("Incident polar angle (degrees)")
    axes[0].set_ylabel("Median angular shape error")
    axes[1].legend(frameon=False)
    fig.savefig(out / "independent_response.png", dpi=155)
    plt.close(fig)


def precision_plot(report, protocol, rows, out):
    fig, ax = plt.subplots(figsize=(9.6, 4.15), layout="constrained")
    x = np.arange(len(rows))
    ax.axvspan(-0.5, 2.5, color="#e8eff5", alpha=0.76)
    specs = (
        ("band_shape", "max_band_shape_noise", "55–65° shape", "#a45432"),
        ("grazing_shape", "max_grazing_shape_noise", "Grazing shape", "#5c758e"),
        ("albedo", "max_albedo_noise", "Albedo", "#47927d"),
    )
    for i, (metric, maximum, label, color) in enumerate(specs):
        values = [row[metric] / protocol["response"][maximum] for row in report["precision"]]
        ax.bar(x + 0.22 * (i - 1), values, width=0.21, label=label, color=color)
    ax.axhline(1.0, color="#414950", linestyle="--", linewidth=1.1, label="Frozen limit")
    ax.set(xlim=(-0.5, len(rows) - 0.5), ylabel="Measured half-split noise / frozen limit",
           title="Independent layered reference precision")
    ax.set_xticks(x, [row["tag"].removeprefix("test83_") for row in rows], rotation=22, ha="right")
    ax.margins(y=0.2)
    ax.legend(frameon=False, ncol=4)
    fig.savefig(out / "reference_precision.png", dpi=155)
    plt.close(fig)


def comparison_plot(response, scenes, rows, out):
    fig, axes = plt.subplots(2, 1, figsize=(11, 6.25), layout="constrained", sharex=True)
    x = np.arange(len(rows))
    for ax in axes:
        ax.axvspan(-0.6, 2.5, color="#e8eff5", alpha=0.76)
        ax.set_xlim(-0.6, len(rows) - 0.4)
    for i, (arm, color) in enumerate(zip(ARMS, COLORS)):
        offset = 0.18 * (i - 1.5)
        axes[0].bar(x + offset, response["arms"][arm]["band_shape"], width=0.17, label=arm, color=color)
        bins = [scenes[row["image"]]["arms"][arm]["stable_rms"]["objects"] for row in rows]
        values = np.array([b["point"] for b in bins])
        ci = np.abs(values[None, :] - np.array([[b[key] for b in bins] for key in ("lo", "hi")]))
        axes[1].bar(x + offset, values, width=0.17, yerr=ci, error_kw={"linewidth": 0.8}, color=color)
    axes[0].set_ylabel("55–65° mean shape error")
    axes[1].set_ylabel("Object TOTAL relative RMS")
    axes[0].legend(frameon=False, ncol=4)
    axes[1].set_xticks(x, [row["tag"].removeprefix("test83_") for row in rows], rotation=25, ha="right")
    fig.savefig(out / "factorial_quality.png", dpi=155)
    plt.close(fig)


def maps(raw, image, out):
    with np.load(raw / "scenes" / image / "means.npz") as data:
        deltas = {arm: [((data[f"{arm}_{event}"] - data[f"reference_{event}"]) @ LUMA)
                        for event in ("e1", "e2", "h")] for arm in ARMS}
    for event in deltas.values():
        event.append(sum(event))
    fig, axes = plt.subplots(4, 4, figsize=(8.3, 8.05), layout="constrained")
    for i, name in enumerate(("E1", "E2", "HIGH", "TOTAL")):
        vmax = max(1e-9, max(np.max(np.abs(deltas[arm][i])) for arm in ARMS))
        norm = SymLogNorm(vmax / 100, vmin=-vmax, vmax=vmax)
        for j, arm in enumerate(ARMS):
            ax = axes[i, j]
            art = ax.imshow(deltas[arm][i], norm=norm, cmap="RdBu_r")
            ax.set(xticks=[], yticks=[])
            if i == 0:
                ax.set_title(arm)
            if j == 0:
                ax.set_ylabel(name)
        fig.colorbar(art, ax=axes[i], fraction=0.035, shrink=0.85)
    fig.suptitle(f"{image} · signed luminance error against layered reference")
    fig.savefig(out / f"event_errors_{image}.png", dpi=135)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    raw, out = args.raw, args.output
    out.mkdir(parents=True, exist_ok=True)
    rows = read(raw / "materials.json")
    if (raw / "train_covered.npz").exists():
        coverage_plot(raw, out)
    if (raw / "response.json").exists():
        response = read(raw / "response.json")
        provenance = read(raw / "provenance.json")
        precision_plot(response, provenance["protocol"], rows, out)
        if all(key in response["arms"] for key in ARMS):
            response_plot(response, rows, out)
            if (raw / "scenes.json").exists():
                scenes = read(raw / "scenes.json")
                if len(scenes) == len(rows):
                    comparison_plot(response, scenes, rows, out)
    for row in rows:
        image = row["image"]
        if (raw / "scenes" / image / "means.npz").exists():
            maps(raw, image, out)


if __name__ == "__main__":
    main()
