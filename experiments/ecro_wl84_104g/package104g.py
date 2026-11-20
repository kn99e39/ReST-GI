"""Assemble WL104G reviewer plots, measurements and artifact hashes."""

import argparse
import hashlib
import json
from importlib.metadata import version
from pathlib import Path
from shutil import copyfile

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
ARMS = ("R0", "R1", "R2")
COLORS = ("#253247", "#287665", "#a83d23")
EVENTS = ("e1", "e2", "h", "total")
LUMA = np.array([.2126, .7152, .0722])


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def plot_angular(response, output):
    polar = np.asarray(response["theta_deg"])
    theta = np.unique(polar)
    cohorts = np.array([row["cohort"] for row in response["materials"]])
    fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharex=True, sharey="row", layout="constrained")
    for col, cohort in enumerate(("development", "held_out")):
        mask = cohorts == cohort
        for ax, metric in zip(axes[:, col], ("el1", "eshape")):
            for arm, c in zip(ARMS, COLORS):
                probe = np.asarray(response["arms"][arm]["angular"][metric])[mask]
                series = np.stack([probe[:, polar == t].mean(1) for t in theta], axis=1)
                ax.plot(theta, np.median(series, axis=0), marker="o", lw=1.6, ms=3.5, color=c, label=arm)
                ax.fill_between(theta, series.min(0), series.max(0), color=c, alpha=.10, linewidth=0)
            ax.set_ylabel(metric)
            ax.grid(alpha=.2)
        axes[0, col].set_title(cohort.replace("_", " ").capitalize())
        axes[1, col].set_xlabel("Incident polar angle (°)")
    axes[0, 0].legend(ncols=3, frameon=False, fontsize=9)
    fig.suptitle("Independent layered-response errors across angles")
    fig.supxlabel("Lines show material medians. Shading shows the full material range.", fontsize=9)
    fig.savefig(output, dpi=160, facecolor="white")
    plt.close(fig)


def plot_scene(scene, output):
    with np.load(scene / "means.npz") as z:
        fields = dict(z)
    with np.load(scene / "region_masks.npz") as z:
        mask = z["objects"] | z["floor_near"]
    y, x = np.where(mask)
    roi = np.s_[max(0, y.min()-6):y.max()+7, max(0, x.min()-6):x.max()+7]
    ref = fields["reference_total"]
    mean_ref = np.mean((ref @ LUMA)[mask])
    display = np.quantile((ref @ LUMA)[mask], .98)
    fig, axes = plt.subplots(5, 4, figsize=(10, 9), layout="constrained")
    for ax, arm in zip(axes[0], ("reference", *ARMS)):
        image = fields[f"{arm}_total"][roi]
        ax.imshow(np.clip(image / max(display, 1e-20), 0., 1.) ** (1/2.2))
        ax.set_title("Layered reference" if arm == "reference" else arm)
    axes[0, 0].set_ylabel("Native TOTAL")
    for i, event in enumerate(EVENTS, start=1):
        ref_event = fields[f"reference_{event}"]
        errors = np.stack([(fields[f"{a}_{event}"] - ref_event) @ LUMA / mean_ref * 100 for a in ARMS])
        limit = max(float(np.quantile(np.abs(errors[:, mask]), .99)), 1e-10)
        for ax, ar in zip(axes[i, 1:], errors):
            im = ax.imshow(np.where(mask, ar, np.nan)[roi], cmap="RdBu_r", vmin=-limit, vmax=limit)
        axes[i, 0].text(.95, .55, event.upper() if event != "h" else "HIGH", ha="right", va="center",
                        transform=axes[i, 0].transAxes, fontsize=11)
        fig.colorbar(im, ax=axes[i, 1:], label="%", shrink=.70, extend="both")
    for row in axes:
        for ax in row:
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
    fig.suptitle(scene.name)
    fig.supxlabel("Signed event residuals as a percentage of mean layered TOTAL luminance. Each row has its own scale.",
                  fontsize=9)
    fig.savefig(output, dpi=160, facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    root, out = args.run, args.output
    out.mkdir(parents=True, exist_ok=False)
    for sub, name in (("operator", "operator.json"), ("fit", "fit.json"), ("fit", "fit_states.npz"),
                      ("response", "response.json"), ("response", "compiled_states.npz"), ("scenes", "decision.json")):
        path = root / sub / name
        if path.exists():
            copyfile(path, out / name)
    if (out / "compiled_states.npz").exists():
        plot_angular(read(out / "response.json"), out / "angular_fidelity.png")
    for record in sorted((root / "scenes/references").glob("*/readiness.json")):
        copyfile(record, out / f"{record.parent.name}_reference_readiness.json")
    for metrics in sorted((root / "scenes").glob("*/metrics.json")):
        scene = metrics.parent
        copyfile(metrics, out / f"{scene.name}_metrics.json")
        plot_scene(scene, out / f"{scene.name}_native_errors.png")
    f = read(out / "fit.json") if (out / "fit.json").exists() else {}
    historical = read(REPO / "evaluation/104F/manifest.json")
    raw = {p.relative_to(root).as_posix(): sha(p) for p in sorted(root.rglob("*")) if p.is_file()}
    manifest = {"repo_id": 1379024934, "branch": "ecro_lr3_adapter",
                "runtime": {name: version(name) for name in ("mitsuba", "drjit", "numpy", "scipy", "matplotlib")},
                "local_code": {p.relative_to(REPO).as_posix(): sha(p)
                               for p in sorted(HERE.iterdir()) if p.suffix in (".py", ".json")},
                "frozen_WL104F_input_sha256": historical["input_sha256"],
                "frozen_WL104G_fit_input_sha256": f.get("data_sha256", {}),
                "raw_sha256": raw,
                "review_sha256": {p.name: sha(p) for p in sorted(out.iterdir()) if p.is_file()}}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
