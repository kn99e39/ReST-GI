"""Present completed WL104E evidence, including frozen reference-half noise maps."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib.colors import SymLogNorm
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ecro_wl84_104b"))
from stats104b import jk_ci, ms_loo

LUMA = np.array([0.2126, 0.7152, 0.0722])
EVENTS = ("e1", "e2", "h")
COLUMNS = ("A", "C", "AC", "CA", "LC", "CL", "LL", "Layered split noise")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def luminance(data, arm, event):
    values = data[f"{arm}_{event}"] if event != "total" else sum(data[f"{arm}_{k}"] for k in EVENTS)
    return values @ LUMA


def maps(case, results, parent, out):
    fig, axes = plt.subplots(2, len(COLUMNS), figsize=(16, 4.65), layout="constrained")
    with np.load(results / "means.npz") as means:
        for cell, event in enumerate(("e2", "total")):
            ref = luminance(means, "reference", event)
            differences = [luminance(means, arm, event) - ref for arm in COLUMNS[:-1]]
            ref_reps = []
            for path in sorted(parent.glob("r[0-9][0-9][0-9].npz")):
                with np.load(path) as source:
                    ref_reps.append(luminance(source, "reference", event))
            replicas = np.stack(ref_reps)
            half = len(replicas) // 2
            differences.append(0.5 * (replicas[:half].mean(0) - replicas[half:].mean(0)))
            peak = max(1e-6, max(np.max(np.abs(img)) for img in differences))
            norm = SymLogNorm(peak / 200, vmin=-peak, vmax=peak)
            for j, (label, img) in enumerate(zip(COLUMNS, differences, strict=True)):
                ax = axes[cell, j]
                artist = ax.imshow(img, cmap="RdBu_r", norm=norm)
                ax.set(xticks=[], yticks=[])
                if cell == 0:
                    ax.set_title(label, fontsize=10)
                if j == 0:
                    ax.set_ylabel(event.upper())
            ticks = peak * np.array([-1, -0.1, -0.02, 0, 0.02, 0.1, 1])
            fmt = FuncFormatter(lambda value, _: f"{value:.1e}" if value else "0")
            fig.colorbar(artist, ax=axes[cell], fraction=0.018, label="Signed luminance", ticks=ticks, format=fmt)
    fig.suptitle(f"{case} · counterfactual error relative to layered BSDF")
    fig.savefig(out / f"maps_{case}.png", dpi=155)
    plt.close(fig)


def plot_attribution(cases, out):
    keys = ("S1", "N2", "interaction")
    colors = ("#246f80", "#bd8a3c", "#925567")
    fig, axes = plt.subplots(1, len(cases), figsize=(6.7 * len(cases), 3.7),
                             layout="constrained", squeeze=False)
    for ax, (case, evidence) in zip(axes[0], cases.items(), strict=True):
        shift = np.arange(3)
        for j, (key, color) in enumerate(zip(keys, colors, strict=True)):
            estimates = [evidence["factorization"][roi]["shares"].get(key)
                         for roi in ("objects", "cube", "sphere")]
            for i, estimate in enumerate(estimates):
                if estimate is None:
                    continue
                y = i + (j - 1) * 0.22
                ax.errorbar(estimate["point"], y,
                            xerr=[[estimate["point"] - estimate["lo"]], [estimate["hi"] - estimate["point"]]],
                            fmt="o", capsize=3, color=color, label=key if i == 0 else None)
        ax.axvline(0, color="#444444", lw=0.8)
        ax.axvline(0.5, color="#888888", ls="--", lw=1.1)
        ax.set(yticks=shift, yticklabels=["Objects", "Cube", "Sphere"], title=case,
               xlabel="Share of C−A E2 spatial change (95% interval)")
        ax.legend(frameon=False)
    fig.savefig(out / "attribution.png", dpi=155)
    plt.close(fig)


def diagnostic_overview(root):
    inputs = [read(path) for path in sorted(root.glob("diagnostic[0-9][0-9][0-9].json"))]
    final = {}
    for sampler in ("A", "C"):
        final[sampler] = {}
        for roi in ("objects", "cube", "sphere"):
            items = [r[sampler]["summary"][roi] for r in inputs]
            groups = {}
            for field in ("first_theta", "first_mirror", "second_theta", "second_mirror"):
                groups[field] = {key: np.mean([obj[field][key] for obj in items], axis=0).tolist()
                                 for key in items[0][field]}
            names = sorted(set().union(*(obj["routes"] for obj in items)))
            groups["routes"] = {
                name: {key: float(np.mean([obj["routes"].get(name, {}).get(key, 0.0) for obj in items]))
                       for key in ("c_energy", "signed_n2_change", "active_pairs")}
                for name in names
            }
            final[sampler][roi] = groups
    return {"angle_edges_deg": [0, 50, 55, 65, 70, 80, 90.001], "mirror_edges_deg": [0, 3, 12, 180.001],
            "replicates": len(inputs), "sampled_paths": final}


def residual_overview(result, parent):
    """Descriptive event projection on already frozen/saved object buffers."""
    with np.load(result / "masks.npz") as archive:
        mask = archive["objects"]
    values = {arm: {k: [] for k in EVENTS} for arm in ("A", "C", "LC", "LL", "reference")}
    for path in sorted(result.glob("r[0-9][0-9][0-9].npz")):
        with np.load(path) as new, np.load(parent / path.name) as old:
            for arm in ("A", "C", "reference"):
                for key in EVENTS:
                    values[arm][key].append(old[f"{arm}_{key}"][mask].astype(np.float64).ravel())
            for arm in ("LC", "LL"):
                for key, buf in (("e1", old["C_e1"]), ("e2", new[arm]), ("h", new["tail_l"])):
                    values[arm][key].append(buf[mask].astype(np.float64).ravel())
    values = {arm: {key: np.stack(buffers) for key, buffers in x.items()} for arm, x in values.items()}
    norm = (sum(values["reference"].values()).mean(0)**2).mean()
    evidence = {}
    for arm, ref in (("A", "reference"), ("C", "reference"), ("LC", "reference"), ("LL", "reference"), ("C", "A")):
        events = {e: values[arm][e] - values[ref][e] for e in EVENTS}
        total = sum(events.values())
        ms, loo = ms_loo(total, total)
        contributions = {}
        for e, v in events.items():
            self_ms, _ = ms_loo(v, v)
            cross, cross_loo = ms_loo(v, total)
            shares = jk_ci(cross / ms, cross_loo / loo) if ms > 0 and np.all(loo > 0) else None
            contributions[e] = {"rms": float(np.sqrt(max(0, self_ms) / norm)), "spatial_projection_share": shares}
        evidence[f"{arm}_minus_{ref}"] = contributions
    return evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw", type=Path)
    parser.add_argument("parent", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    cases = {}
    diagnostics = {}
    residuals = {}
    for folder in sorted(args.raw.iterdir()):
        if (folder / "metrics.json").exists():
            case = folder.name
            cases[case] = read(folder / "metrics.json")
            diagnostics[case] = diagnostic_overview(folder)
            maps(case, folder, args.parent / "scenes" / case, args.out)
            residuals[case] = residual_overview(folder, args.parent / "scenes" / case)
    if cases:
        plot_attribution(cases, args.out)
        (args.out / "direction_diagnostics.json").write_text(json.dumps(diagnostics, indent=2) + "\n", encoding="utf-8")
        (args.out / "descriptive_event_residuals.json").write_text(json.dumps({
            "status": "Post hoc projection of existing event residuals. Original counterfactual and recovery gates are unchanged.",
            "cases": residuals,
        }, indent=2) + "\n", encoding="utf-8")
    for name in ("status.json", "provenance.json", "cases.json", "decision.json"):
        source = args.raw / name
        if source.exists():
            (args.out / name).write_bytes(source.read_bytes())
    for source in sorted(args.raw.glob("*/*.json")):
        if source.name in ("baseline.json", "focused_tests.json", "identity000.json"):
            (args.out / f"{source.parent.name}_{source.name}").write_bytes(source.read_bytes())
    (args.out / "review_manifest.json").write_text(json.dumps({
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "output_sha256": {x.name: hashlib.sha256(x.read_bytes()).hexdigest() for x in sorted(args.out.iterdir())},
    }, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
