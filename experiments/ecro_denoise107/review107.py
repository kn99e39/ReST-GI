"""Export WL107 reviewer plots and hash manifest from finished raw evidence."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common107 import LAMBDAS, LUMA, SPPS, VIEWS, read, sha, stencil, take_stencil, write
from oracle107 import solve_numpy, synthetic_risk
from synthetic107 import COLS, ROWS, SIZE, SYNTH_SEED, draws, guides, truth

PALETTE = {"constant": "#366b92", "smooth": "#368767", "rim": "#b74a2d"}
DISPLAY_CASES = ((0.125, "bright", "rim", 0.25, False), (0.3, "dark", "rim", 3, True),
                 (0.5, "bright", "constant", None, False))


def plot_risk(data, output):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.8), layout="constrained", sharex=True, sharey=True)
    fig.suptitle("Synthetic grazing-rim bias and variance at fixed 15 × 15 support", fontsize=15)
    for spp, ax in zip(SPPS, axes):
        for field, color in PALETTE.items():
            subset = [c["spp"][str(spp)]["oracle"] for name, c in data["cases"].items() if f"_{field}_" in name]
            for i, row in enumerate(subset):
                a = [100 * row[str(lam)]["band_profile_rms_bias"] for lam in LAMBDAS]
                b = [row[str(lam)]["variance_ratio_to_raw"] for lam in LAMBDAS]
                ax.plot(a, b, "-o", ms=2.3, lw=0.65, color=color,
                        label=field.capitalize() if i == 0 else None, alpha=0.59)
        ax.axvline(1, color="#51565d", ls="--", lw=.9)
        ax.axhline(.5, color="#51565d", ls="--", lw=.9)
        ax.set(xscale="log", yscale="log", title=f"{spp} spp", xlabel="Rim profile RMS bias (%)")
        ax.grid(True, which="major", alpha=.24)
    axes[0].set_ylabel("Variance divided by raw variance")
    axes[-1].legend(loc="lower left", fontsize=9)
    fig.savefig(output / "synthetic_tradeoffs.png", dpi=160)
    plt.close(fig)


def plot_examples(output):
    rng = np.random.default_rng(SYNTH_SEED + 1)
    fig, axes = plt.subplots(len(DISPLAY_CASES), 5, figsize=(13.8, 6.5), layout="constrained")
    fig.suptitle("Synthetic profiles at 4 spp. Oracle weights selected with exact expected radiance", fontsize=15)
    idx = (ROWS[:, None] * SIZE + COLS).ravel()
    taps = stencil(idx, SIZE, SIZE)
    for j, case in enumerate(DISPLAY_CASES):
        mu = truth(case)
        patch = take_stencil(mu, taps).transpose(2, 0, 3, 1)
        initial = np.zeros(patch.shape[:-1])
        initial[:, :, 112] = 1
        for lam in LAMBDAS[:2]:
            f, L = synthetic_risk(patch, 4, lam)
            initial, _, _, _ = solve_numpy(f, L, initial)
        clean = (np.einsum("nkpc,nkp->nc", patch, initial) @ LUMA).reshape(len(ROWS), -1)
        ref = ((mu.sum(axis=0)[:, ROWS[:, None], COLS[None, :]]).transpose(1, 2, 0) @ LUMA)
        # One fixed display-only realization; the curves summarize all 256 evidence draws.
        raw = draws(case, 4, rng, 1)[0]
        samp_patch = take_stencil(raw, taps).transpose(2, 0, 3, 1)
        noisy = (samp_patch[:, :, 112].sum(axis=1) @ LUMA).reshape(len(ROWS), -1)
        recon = (np.einsum("nkpc,nkp->nc", samp_patch, initial) @ LUMA).reshape(len(ROWS), -1)
        cov = guides(case[0])[0][ROWS[:, None], COLS[None, :]]
        for col, (img, lo, hi, cmap) in enumerate(((cov, 0, 1, "Greys"), (ref, 0, 3.5, "magma"),
                   (noisy, 0, 3.5, "magma"), (recon, 0, 3.5, "magma"), (clean - ref, -.12, .12, "RdBu_r"))):
            preview = axes[j, col].imshow(img, vmin=lo, vmax=hi, cmap=cmap, interpolation="nearest", aspect="equal")
            if j == len(DISPLAY_CASES) - 1:
                fig.colorbar(preview, ax=axes[:, col], orientation="horizontal", pad=.05, fraction=.045, shrink=.85)
            axes[j, col].set_xticks([])
            axes[j, col].set_yticks([])
        offset, contrast, field, width, wavy = case
        label = f"{contrast.capitalize()} foreground\n{field.capitalize()}" + (f" {width:g} px" if width else "")
        axes[j, 0].set_ylabel(f"{label}\nOffset {offset:g} px", fontsize=8)
    for ax, title in zip(axes[0], ("8 × 8 coverage", "Expected radiance", "One raw draw", "Oracle on same draw",
                                  "Oracle expected error")):
        ax.set_title(title, fontsize=10)
    fig.savefig(output / "synthetic_error_maps.png", dpi=160)
    plt.close(fig)


def plot_reference(data, output):
    fig, ax = plt.subplots(figsize=(7.5, 4.3), layout="constrained")
    for key, label, color in (("primary", "Primary", "#303d5f"), ("silhouette", "Silhouette", "#b74a2d"),
                              ("material_boundary", "Material boundary", "#368767")):
        values = [100 * data["reference_noise_over_R3_rmse"][str(s)][key] for s in SPPS]
        ax.plot(SPPS, values, "-o", label=label, color=color, lw=1.8)
    ax.axhline(10, color="#777777", ls="--", label="Frozen limit")
    ax.set(xlabel="Reconstruction input (spp)", ylabel="Reference RMS noise / R3 RMSE (%)",
           title="Archived WL105 two-half target precision", xticks=SPPS, ylim=(0, 42))
    ax.legend(loc="upper left")
    ax.grid(alpha=.24)
    fig.savefig(output / "archived_reference_precision.png", dpi=160)
    plt.close(fig)


def summary(data):
    out = {}
    for s in SPPS:
        row = {}
        for field in PALETTE:
            cases = {name: v["spp"][str(s)]["oracle"] for name, v in data["cases"].items()
                     if f"_{field}_" in name}
            qualified = {name: d for name, d in cases.items() if any(c["passes"] for c in d.values())}
            row[field] = {"cases": len(cases), "witness_cases": len(qualified),
                          "no_witness_cases": [n for n in cases if n not in qualified],
                          "median_best_feasible_variance_ratio": float(np.median([
                              min(x["variance_ratio_to_raw"] for x in d.values() if x["passes"])
                              for d in qualified.values()])) if qualified else None}
        out[str(s)] = row
    return out


def main(synthetic, reference, output):
    data, pilot = read(synthetic), read(reference)
    assert data["complete"] and len(data["cases"]) == 64
    assert pilot["archive_pilot"]
    output.mkdir(exist_ok=False, parents=True)
    plot_risk(data, output)
    plot_examples(output)
    plot_reference(pilot, output)
    write(output / "summary107.json", {"synthetic_cases": summary(data),
          "counterexamples": data["counterexamples"], "historical_control": data["WL106_historical_only"],
          "max_synthetic_reference_error": max(v["reference_error_max_abs"] for v in data["cases"].values()),
          "archived_reference": {k: v for k, v in pilot.items() if k != "views"}})
    # Portable reference and complete curves, including all solver gaps.
    for src, dst in ((synthetic, "synthetic107.json"), (reference, "archive_precision107.json")):
        (output / dst).write_bytes(src.read_bytes())
    write(output / "manifest107.json", {
        "protocol_sha256": data["protocol_sha256"], "synthetic_code_commit": data["code_commit"],
        "sources": {"synthetic": sha(synthetic), "archived_precision": sha(reference)},
        "evidence_scope": "Analytic synthetic experiment and a precision audit of the archived two-half real targets.",
        "files": {p.name: sha(p) for p in sorted(output.iterdir())}
    })


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for field in ("synthetic", "reference", "output"):
        p.add_argument(field, type=Path)
    a = p.parse_args()
    main(a.synthetic, a.reference, a.output)
