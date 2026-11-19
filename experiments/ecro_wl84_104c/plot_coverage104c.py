"""Plot the incident-angle counts in the frozen fit inputs used by the preflight."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("preflight", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    report = json.loads(args.preflight.read_text(encoding="utf-8"))
    protocol = json.loads((Path(__file__).parent / "protocol/wl104c.json").read_text(encoding="utf-8"))
    coverage = protocol["coverage"]
    edges = np.arange(91)
    fig, axes = plt.subplots(3, 1, figsize=(8.4, 5.3), sharex=True, layout="constrained")
    for ax, name in zip(axes, coverage["sets"], strict=True):
        path = args.data / f"fit82_{name}.npz"
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1 << 20), b""):
                digest.update(chunk)
        if digest.hexdigest() != report["input_sha256"][path.name]:
            raise ValueError(f"Plot input does not match preflight: {path}")
        with np.load(path) as data:
            groups = data["wi"][::int(data["n_wo"]), 2].astype(np.float64)
        theta = np.degrees(np.arccos(np.clip(groups, -1, 1)))
        counts, _ = np.histogram(theta, edges)
        ax.axvspan(*coverage["required_incident_band_deg"], color="#f3b27e", alpha=0.55)
        ax.axvline(60, color="#914111", linewidth=1.2, linestyle="--")
        ax.stairs(counts, edges, fill=True, color="#245e84")
        ax.set_xlim(0, 90)
        ax.text(
            0.025, 0.84, f"{name} · {report['coverage'][name]['materials']} materials",
            transform=ax.transAxes, va="top", color="#17344a",
        )
        ax.spines[["top", "right"]].set_visible(False)
    axes[-1].set_xlabel("Incident polar angle (degrees)")
    fig.supylabel("Fit groups per degree")
    fig.suptitle("WL83 fit samples exclude the 60° lobe probe", fontweight="bold")
    fig.savefig(args.output, dpi=160)


if __name__ == "__main__":
    main()
