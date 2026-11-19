"""Export WL107 reviewer plots and hash manifest from finished raw evidence."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common107 import BINS, LAMBDAS, LUMA, SPPS, VIEWS, read, sha, stencil, take_stencil, write
from oracle107 import solve_numpy, synthetic_risk
from synthetic107 import COLS, ROWS, SIZE, SYNTH_SEED, case_name, draws, guides, truth

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



def plot_signed_profiles(data, output):
    fig, axes = plt.subplots(2, 3, figsize=(12, 5.8), layout="constrained", sharex="col", sharey="row")
    fig.suptitle("Expected synthetic rim profiles and signed residuals at 4 spp", fontsize=15)
    for case, grid in zip(DISPLAY_CASES, axes.T):
        record = data["cases"][case_name(case)]
        target = np.asarray(record["target_luma_by_column"])
        x = COLS - 32
        grid[0].plot(x, target, color="#202227", lw=2, label="Target")
        for key, color in (("0.125", "#368767"), ("1", "#b74a2d")):
            y = record["spp"]["4"]["oracle"][key]["profile_luma_expectation"]
            grid[0].plot(x, y, "-o", lw=1, ms=2.5, color=color, label=f"Oracle λ = {key}")
            grid[1].plot(x, y - target, "-o", lw=1, ms=3, color=color)
        for ax in grid:
            ax.axvspan(-.5, .5, color="#a4c3e1", alpha=.27)
            ax.grid(alpha=.22)
        offset, contrast, field, width, _ = case
        name = f"{contrast.capitalize()} foreground, {field}"
        grid[0].set_title(name + (f" {width:g} px" if width else ""), fontsize=10)
        grid[1].set_xlabel("Pixel column relative to mixed coverage")
        grid[1].axhline(0, color="#777777", lw=.75)
    axes[0, 0].set_ylabel("Linear HDR luminance")
    axes[1, 0].set_ylabel("Signed luminance residual")
    axes[0, -1].legend(fontsize=8)
    fig.savefig(output / "synthetic_rim_profiles.png", dpi=160)
    plt.close(fig)


def plot_real(real, output):
    result = read(real / "real107.json")
    assert result["complete"]
    for spp in SPPS:
        fig, axes = plt.subplots(len(VIEWS), 5, figsize=(12, 12), layout="constrained")
        fig.suptitle(f"Held-out grazing-rim crops, {spp} spp", fontsize=15)
        for j, view in enumerate(VIEWS):
            with np.load(real / f"{view}_maps.npz") as data:
                coords = np.argwhere(data["silhouette"])
                mids = np.unique(coords[:, 0])
                yy = int(mids[len(mids) // 2])
                xx = int(coords[coords[:, 0] == yy, 1].min())
                ysl, xsl = slice(max(0, yy - 14), yy + 15), slice(max(0, xx - 22), xx + 23)
                ref = (LUMA @ data["target"][:, ysl, xsl].reshape(3, -1))
                shp = data["target"][:, ysl, xsl].shape[1:]
                images = {"Target": (ref.reshape(shp), 0, 2.8, "magma")}
                for key, name in (("O0", "Frozen R3"), ("O2", "O2 diagnostic")):
                    field = f"{key}_{spp}_mean"
                    if field not in data:
                        continue
                    im = LUMA @ data[field][:, ysl, xsl].reshape(3, -1)
                    images[name] = (im.reshape(shp), 0, 2.8, "magma")
                    images[name + " signed error"] = ((im - ref).reshape(shp), -.15, .15, "RdBu_r")
            for ax in axes[j]:
                ax.set_xticks([])
                ax.set_yticks([])
            axes[j, 0].set_ylabel(view)
            for col, (label, (im, lo, hi, cm)) in enumerate(images.items()):
                axes[j, col].imshow(im, vmin=lo, vmax=hi, cmap=cm)
                if j == 0:
                    axes[j, col].set_title(label, fontsize=8)
            for ax in axes[j, len(images):]:
                ax.set_visible(False)
        fig.savefig(output / f"heldout_crops_spp{spp}.png", dpi=145)
        plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), layout="constrained", sharey=True)
    fig.suptitle("Held-out coverage-conditioned residuals on the fixed O1 panel", fontsize=15)
    xs = .5 * (np.array(BINS[1:]) + np.array(BINS[:-1]))
    for spp, ax in zip(SPPS, axes):
        o0, o1 = [], []
        for view in VIEWS:
            rec = result["views"][view]["O1"][str(spp)]
            o0.append(rec["O0"]["coverage_profile"])
            o1.append(rec["rows"]["1"]["reference_mean"]["coverage_profile"])
        for values, name, color in ((o0, "Frozen R3", "#343c49"), (o1, "O1 λ = 1", "#b74a2d")):
            matrix = np.array(values, dtype=float)
            keep = np.isfinite(matrix).any(axis=0)
            ax.plot(xs[keep], 100 * np.nanmean(matrix[:, keep], axis=0), "o-", label=name, color=color)
        ax.axhline(0, color="#777777", lw=.75)
        ax.set(title=f"{spp} spp", xlabel="Sphere coverage")
        ax.grid(alpha=.22)
    axes[0].set_ylabel("Signed luminance residual (%)")
    axes[-1].legend()
    fig.savefig(output / "heldout_coverage_profiles.png", dpi=155)
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


def main(synthetic, reference, output, audit=None, real=None):
    data, pilot = read(synthetic), read(reference)
    assert data["complete"] and len(data["cases"]) == 64
    assert pilot["archive_pilot"]
    output.mkdir(exist_ok=False, parents=True)
    plot_risk(data, output)
    plot_examples(output)
    plot_signed_profiles(data, output)
    plot_reference(pilot, output)
    if real is not None:
        plot_real(real, output)
    if audit is not None:
        (output / "archive_audit107.json").write_bytes(audit.read_bytes())
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
        "evidence_scope": ("Analytic synthetic experiment and a precision audit of the archived two-half real targets."
                           if real is None else "Synthetic, archived precision, and fresh held-out experiments."),
        "archive_audit_sha256": sha(audit) if audit is not None else None,
        "real_sha256": sha(real / "real107.json") if real is not None else None,
        "files": {p.name: sha(p) for p in sorted(output.iterdir())}
    })


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for field in ("synthetic", "reference", "output"):
        p.add_argument(field, type=Path)
    p.add_argument("--audit", type=Path)
    p.add_argument("--real", type=Path)
    a = p.parse_args()
    main(a.synthetic, a.reference, a.output, a.audit, a.real)
