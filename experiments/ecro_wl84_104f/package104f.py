"""Package the frozen Part A replay and plot its paired E2 residuals."""

import argparse
import hashlib
import json
import platform
import sys
from importlib.metadata import version
from pathlib import Path
from shutil import copyfile

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
CASES = ("test83_iid_1_L2", "test83_hard_1_L1")
ARMS = ("AA", "AC", "CA", "CC")
LUMA = np.array([.2126, .7152, .0722])


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def field(folder, key):
    samples = []
    for path in sorted(folder.glob("r[0-9][0-9][0-9].npz")):
        with np.load(path) as z:
            samples.append(z[key])
    if len(samples) != 12:
        raise ValueError(f"Expected 12 paired replicates in {folder}")
    return np.mean(samples, axis=0, dtype=np.float64)


def plot(replay, parent, target):
    fig, axes = plt.subplots(2, 4, figsize=(10, 5), layout="constrained")
    for i, case in enumerate(CASES):
        frozen = parent / "scenes" / case
        with np.load(frozen / "region_masks.npz") as z:
            obj = z["objects"]
        ys, xs = np.where(obj)
        roi = np.s_[max(0, ys.min()-4):ys.max()+5, max(0, xs.min()-4):xs.max()+5]
        ref = field(frozen, "reference_e2")
        total = ref + field(frozen, "reference_e1") + field(frozen, "reference_h")
        scale = (total @ LUMA)[obj].mean()
        images = [(field(replay / case, arm) - ref) @ LUMA / scale * 100 for arm in ARMS]
        limit = float(np.quantile(np.abs(np.stack(images)[:, obj]), .99))
        for ax, arm, values in zip(axes[i], ARMS, images):
            im = ax.imshow(np.where(obj, values, np.nan)[roi], cmap="RdBu_r", vmin=-limit, vmax=limit)
            if i == 0:
                ax.set_title(f"$f_{arm[0]}$ with $p_{arm[1]}$", fontsize=11)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
        axes[i, 0].set_ylabel(case.replace("test83_", ""))
        fig.colorbar(im, ax=axes[i, :], label="%", shrink=.85, extend="both")
    fig.suptitle("First-hit response and proposal cross against layered E2")
    fig.supxlabel("Signed E2 error as a percentage of mean reference TOTAL luminance. Each row has its own scale.",
                  fontsize=9)
    fig.savefig(target, dpi=160, facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("replay", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--resource-profile", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    parent = REPO / "results/ecro_wl84_104e"
    for name in ("operator_audit.json", "decision.json", "status.json"):
        copyfile(args.replay / name, args.output / name)
    if args.resource_profile:
        copyfile(args.resource_profile, args.output / "resources.json")
    for case in CASES:
        for label in ("metrics", "focused"):
            copyfile(args.replay / case / f"{label}.json", args.output / f"{case}_{label}.json")
    plot(args.replay, parent / "parent", args.output / "cross_E2_residuals.png")
    historical = REPO / "evaluation/104E"
    amendment = REPO / "evaluation/104D/amendment1"
    intermediate = parent / "primary"
    assets = REPO / "results/ecro_wl84_104"
    inputs = [HERE / "protocol.json", HERE / "cross104f.py", HERE / "package104f.py",
              historical / "provenance.json", historical / "descriptive_event_residuals.json",
              amendment / "compiled_states.npz", amendment / "response.json", amendment / "provenance.json",
              amendment / "materials.json", intermediate / "provenance.json",
              assets / "inputs63/rest_compiled83/results/data/materials83.npz",
              assets / "scene/uv_sphere_128x256.ply", assets / "scene/quad.ply"]
    for case in CASES:
        inputs.append(historical / f"{case}_focused_tests.json")
        for folder in (parent / "parent/scenes" / case, intermediate / case):
            inputs.extend(sorted(folder.glob("r[0-9][0-9][0-9].npz")))
        inputs.extend(parent / "parent/scenes" / case / name for name in ("metrics.json", "region_masks.npz"))
    manifest = {
        "environment": {"platform": platform.platform(), "python": sys.version.split()[0],
                        "numpy": np.__version__, "mitsuba": version("mitsuba"), "drjit": version("drjit"),
                        "matplotlib": matplotlib.__version__, "variant": "cuda_ad_rgb"},
        "weight_identity": "WL104E full native A/C and layered focused tests",
        "input_sha256": {p.relative_to(REPO).as_posix(): sha(p) for p in inputs},
        "raw_replay_sha256": {p.relative_to(args.replay).as_posix(): sha(p)
                              for p in sorted(args.replay.rglob("*")) if p.is_file()},
        "packaged_sha256": {p.name: sha(p) for p in sorted(args.output.iterdir()) if p.is_file()},
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
