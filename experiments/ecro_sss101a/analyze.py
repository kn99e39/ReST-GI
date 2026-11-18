"""Attribute fixed-scene WL101A error in linear HDR and generate reviewer evidence."""
import argparse
import json
from pathlib import Path
import platform

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import EXECUTION, P, PROTOCOL, RES, components, image, masks, sample_seed, sha, write_json


LABELS = ("full", "low", "high", "e1", "e2")
DRAW_COUNT = 4000
BLOCK = 16


def average(value):
    """Pool valid pixels and RGB, retaining all final-axis components."""
    return np.mean(value, axis=(-3, -2), dtype=np.float64)


def mapping(values):
    return dict(zip(LABELS, np.asarray(values).tolist()))


def ci(draws):
    return np.quantile(draws, [0.025, 0.975], axis=0).T.tolist()


def block_bootstrap(fields, mask, rng):
    fields = np.where(mask[..., None], fields, 0.0)
    sums = fields.reshape(RES // BLOCK, BLOCK, RES // BLOCK, BLOCK, -1).sum(axis=(1, 3))
    count = mask.reshape(RES // BLOCK, BLOCK, RES // BLOCK, BLOCK).sum(axis=(1, 3))
    sums, count = sums[count > 0], count[count > 0]
    index = rng.integers(len(count), size=(DRAW_COUNT, len(count)))
    return (sums[index].sum(axis=1) / count[index].sum(axis=1)[:, None], len(count))


def stable_stats(replays, refs, historical, mask, rng):
    diff_a, diff_b = (replays - refs).astype(np.float64)
    rmean = np.mean(refs, axis=0, dtype=np.float64)
    paired_mean = (diff_a + diff_b) * 0.5
    paired_noise = ((diff_a - diff_b) * 0.5) ** 2
    ref_noise = ((refs[0].astype(np.float64) - refs[1]) * 0.5) ** 2
    stable = diff_a * diff_b
    fields = np.concatenate((stable, ref_noise), axis=-1).mean(axis=2)
    draws, tiles = block_bootstrap(fields, mask, rng)
    stable_point = average(stable[mask])
    stable_ci = np.asarray(ci(draws[:, :len(LABELS)]))
    rpower = average(rmean[mask] ** 2)
    hdiff = historical[mask] - rmean[mask]
    summary = {
        "pixels": int(mask.sum()), "nonempty_bootstrap_tiles": tiles,
        "reference_rms_hdr": mapping(np.sqrt(rpower)),
        "reference_mean_noise_mse": mapping(average(ref_noise[mask])),
        "reference_mean_noise_relative_rmse": mapping(np.sqrt(average(ref_noise[mask]) / rpower)),
        "paired_mean_raw_rmse_hdr": mapping(np.sqrt(average(paired_mean[mask] ** 2))),
        "paired_mean_raw_relative_rmse": mapping(np.sqrt(average(paired_mean[mask] ** 2) / rpower)),
        "paired_mean_raw_relative_l1": mapping(average(np.abs(paired_mean[mask])) / average(np.abs(rmean[mask]))),
        "paired_mean_signed_bias_rgb_hdr": mapping(np.mean(paired_mean[mask], axis=0).T),
        "paired_mean_noise_relative_rmse": mapping(np.sqrt(average(paired_noise[mask]) / rpower)),
        "stable_cross_mse": mapping(stable_point),
        "stable_cross_mse_ci95": mapping(stable_ci),
        "stable_cross_relative_rmse_if_positive": dict(zip(
            LABELS, [float(np.sqrt(v / p)) if v >= 0 else None for v, p in zip(stable_point, rpower)]
        )),
        "stable_relative_rmse_ci95_if_positive": dict(zip(
            LABELS, [[float(np.sqrt(x / p)) if x >= 0 else None for x in bounds]
                     for bounds, p in zip(stable_ci, rpower)]
        )),
        "historical_unpaired_4096_relative_rmse": mapping(np.sqrt(average(hdiff ** 2) / rpower)),
        "historical_unpaired_4096_relative_l1": mapping(average(np.abs(hdiff)) / average(np.abs(rmean[mask]))),
    }
    return summary, stable_point, draws[:, :len(LABELS)], draws[:, len(LABELS):]


def variance_draws(values, variance, rng):
    per_pair = np.array([average((a.astype(np.float64) - b) ** 2) * 0.5
                         for a, b in zip(values[::2], values[1::2])])
    index = rng.integers(len(per_pair), size=(DRAW_COUNT, len(per_pair)))
    draws = per_pair[index].mean(axis=1) - per_pair.mean(axis=0) + variance
    return draws, per_pair


def ratio_ci(draws):
    assert np.isfinite(draws).all()
    return np.quantile(draws, [0.025, 0.975]).tolist()


def sweep_stats(frames, refs, mask, stable_point, stable_draws, noise_draws, rng):
    values = frames[:, mask]
    r0, r1 = refs[:, mask].astype(np.float64)
    rmean = (r0 + r1) * 0.5
    n = len(values)
    mean = values.mean(axis=0, dtype=np.float64)
    variance = average(np.var(values, axis=0, ddof=1, dtype=np.float64))
    draws, pairs = variance_draws(values, variance, rng)
    mean_delta = mean - rmean
    observed_noise = average(((r0 - r1) * 0.5) ** 2)
    ref_power = average(rmean ** 2)
    ref_l1 = average(np.abs(rmean))
    squared_errors = np.array([average((frame - rmean) ** 2) for frame in values])
    rel_l1s = np.array([average(np.abs(frame - rmean)) / ref_l1 for frame in values])
    stable_at_spp = average((mean - r0) * (mean - r1)) - variance / n
    twice_cov = variance[0] - variance[1] - variance[2]
    denominator = variance[0] + stable_point[0] + observed_noise[0]
    d_draws = draws[:, 0] + stable_draws[:, 0] + noise_draws[:, 0]
    l_draws = draws[:, 1] / d_draws
    b_draws = stable_draws[:, 0] / d_draws
    cov_draws = draws[:, 0] - draws[:, 1] - draws[:, 2]
    assert np.all(d_draws > 0)
    record = {
        "interseed_variance_hdr2": mapping(variance),
        "interseed_variance_ci95": mapping(ci(draws)),
        "pair_variance_hdr2": mapping(pairs.T),
        "interseed_sd_relative_to_reference_rms": mapping(np.sqrt(variance / ref_power)),
        "twice_low_high_covariance_hdr2": float(twice_cov),
        "twice_low_high_covariance_ci95": ratio_ci(cov_draws),
        "reference_mean_noise_mse": mapping(observed_noise),
        "stable_cross_mse_at_spp": mapping(stable_at_spp),
        "mean_single_render_rmse_hdr": mapping(np.sqrt(squared_errors).mean(axis=0)),
        "mean_single_render_relative_rmse": mapping(np.sqrt(squared_errors / ref_power).mean(axis=0)),
        "mean_single_render_relative_l1": mapping(rel_l1s.mean(axis=0)),
        "single_render_rms_over_seeds_relative_to_reference": mapping(np.sqrt(squared_errors.mean(axis=0) / ref_power)),
        "ensemble_mean_rmse_hdr": mapping(np.sqrt(average(mean_delta ** 2))),
        "ensemble_mean_relative_rmse": mapping(np.sqrt(average(mean_delta ** 2) / ref_power)),
        "ensemble_mean_relative_l1": mapping(average(np.abs(mean_delta)) / ref_l1),
        "ensemble_mean_signed_bias_rgb_hdr": mapping(mean_delta.mean(axis=0).T),
        "per_seed_rmse_hdr": mapping(np.sqrt(squared_errors).T),
        "per_seed_relative_l1": mapping(rel_l1s.T),
        "full_mse_shares": {
            "low_variance": float(variance[1] / denominator),
            "low_variance_ci95": ratio_ci(l_draws),
            "high_variance": float(variance[2] / denominator),
            "twice_low_high_covariance": float(twice_cov / denominator),
            "stable_bias": float(stable_point[0] / denominator),
            "stable_bias_ci95": ratio_ci(b_draws),
            "reference_noise": float(observed_noise[0] / denominator),
            "all_seed_variance": float(variance[0] / denominator),
            "low_variance_excluding_reference_noise": float(variance[1] / (denominator - observed_noise[0])),
            "low_variance_with_half_covariance": float((variance[1] + 0.5 * twice_cov) / denominator),
            "low_variance_over_observed_single_render_mse": float(variance[1] / squared_errors[:, 0].mean()),
        },
    }
    return record, draws


def classify(regions, trends):
    primary = [regions[roi]["spp"]["32"]["full_mse_shares"] for roi in ("sphere", "floor")]
    low = [row["low_variance_ci95"][0] > 0.5 for row in primary]
    bias = [row["stable_bias_ci95"][0] > 0.5 for row in primary]
    falling = all(trends[roi]["low"]["slope_ci95"][1] < 0 for roi in ("sphere", "floor"))
    if all(low) and falling:
        return "SAMPLING-DOMINATED"
    if all(bias):
        return "BIAS-DOMINATED"
    if ((any(low) and any(bias)) or
            (falling and not any(low) and not any(bias)
             and all(row["stable_bias_ci95"][0] > 0 for row in primary))):
        return "MIXED"
    return "INCONCLUSIVE"


def slopes(spps, rows, draws):
    x = np.log(spps)
    weights = (x - x.mean()) / np.sum((x - x.mean()) ** 2)
    raw = np.array([[rows[str(spp)]["interseed_variance_hdr2"][tag] for tag in LABELS] for spp in spps])
    boot = np.stack(draws)
    result = {}
    for i, tag in enumerate(LABELS):
        variance = raw[:, i]
        samples = boot[:, :, i]
        positive = np.all(samples > 0, axis=0)
        limits = ratio_ci(weights @ np.log(samples[:, positive]))
        result[tag] = {
            "log_variance_log_spp_slope": float(weights @ np.log(variance)),
            "slope_ci95": limits, "compatible_with_inverse_spp": limits[0] <= -1 <= limits[1],
            "strictly_decreasing": bool(np.all(np.diff(variance) < 0)),
            "positive_bootstrap_replicates": int(positive.sum()),
            "spp_times_variance_hdr2": (np.asarray(spps) * variance).tolist(),
        }
    return result


def tone(hdr):
    y = P["display"]["exposure"] * np.maximum(hdr, 0)
    return np.clip(y / (1 + y), 0, 1) ** (1 / 2.2)


def cropped(mask):
    rows, cols = np.nonzero(mask)
    return np.s_[max(0, rows.min() - 12):rows.max() + 13, max(0, cols.min() - 12):cols.max() + 13]


def figures(out, regions, previews, ref, conv, noise32, ref_noise, roi_masks):
    def sheet(fig, axes, tiles, title, crop=np.s_[:, :]):
        maps = {}
        cap = P["display"]["absolute_error_max"]
        signed_cap = P["display"]["signed_error_max"]
        for ax, (label, data, kind) in zip(axes.flat, tiles):
            if kind == "HDR":
                ax.imshow(tone(data[crop]), interpolation="none")
            elif kind == "absolute":
                maps[kind] = ax.imshow(np.abs(data[crop]).mean(axis=-1), vmin=0, vmax=cap,
                                       cmap="magma", interpolation="none")
            else:
                maps[kind] = ax.imshow(data[crop].mean(axis=-1), vmin=-signed_cap, vmax=signed_cap,
                                       cmap="RdBu_r", interpolation="none")
            ax.set_title(label, fontsize=9)
            ax.set_axis_off()
        for kind, mappable in maps.items():
            fig.colorbar(mappable, ax=list(axes.flat), shrink=0.65,
                         label=f"{kind.capitalize()} RGB error (linear HDR, display clipped)")
        fig.suptitle(title, fontsize=12)

    keys = list(previews)
    progression = [("Physical reference · 2 × 4096 spp", ref[..., 0], "HDR")]
    progression += [(f"ECRO · {spp} spp", previews[spp][..., 0], "HDR") for spp in keys]
    progression.append(("ECRO · 2 × 4096 spp", conv[..., 0], "HDR"))
    for name, crop in (("progression", np.s_[:, :]), ("sphere_crops", cropped(roi_masks["sphere"]))):
        fig, axs = plt.subplots(2, 4, figsize=(13, 7), layout="constrained")
        sheet(fig, axs, progression, "WL101A SSS cfg0 · independent seeds · shared exposure", crop)
        fig.savefig(out / f"{name}.png", dpi=145)
        plt.close(fig)

    tiles = [
        ("32 spp · full", previews[32][..., 0], "HDR"),
        ("32 spp · physical LOW", previews[32][..., 1], "HDR"),
        ("32 spp · learned HIGH readout", previews[32][..., 2], "HDR"),
        ("32 spp · LOW deviation from 15 independent renders", noise32, "absolute"),
        ("Converged · full signed residual", conv[..., 0] - ref[..., 0], "signed"),
        ("Converged · LOW signed residual", conv[..., 1] - ref[..., 1], "signed"),
        ("Converged · HIGH signed residual", conv[..., 2] - ref[..., 2], "signed"),
        ("Reference mean noise proxy · half difference", ref_noise, "absolute"),
    ]
    for name, crop in (("attribution", np.s_[:, :]), ("contact_shadow_crops", cropped(roi_masks["contact_shadow"]))):
        fig, axs = plt.subplots(2, 4, figsize=(15, 7), layout="constrained")
        sheet(fig, axs, tiles, "WL101A · matched 4096-spp residuals · fixed HDR scales", crop)
        fig.savefig(out / f"{name}.png", dpi=155)
        plt.close(fig)

    fig, axs = plt.subplots(1, 3, figsize=(12, 4), layout="constrained")
    for ax, tag in zip(axs, ("sphere", "floor", "contact_shadow")):
        for comp, color in (("full", "#303038"), ("low", "#007693"), ("high", "#bb5200")):
            var = [regions[tag]["spp"][str(s)]["interseed_variance_hdr2"][comp] for s in keys]
            lo, hi = np.array([regions[tag]["spp"][str(s)]["interseed_variance_ci95"][comp] for s in keys]).T
            ax.loglog(keys, var, "o-", color=color, label=comp.upper())
            ax.fill_between(keys, lo, hi, color=color, alpha=0.17)
        first = regions[tag]["spp"][str(keys[0])]["interseed_variance_hdr2"]["low"]
        ax.loglog(keys, first / np.asarray(keys), "k--", alpha=0.55, label="1/SPP guide")
        ax.set_title(tag.replace("_", " ").title())
        ax.set_xlabel("Samples per pixel")
        ax.set_xticks(keys, labels=keys)
        ax.set_ylabel("Inter-seed variance (HDR²)")
        ax.grid(alpha=0.2)
    axs[0].legend()
    fig.savefig(out / "variance_convergence.png", dpi=150)
    plt.close(fig)

    fig, axs = plt.subplots(2, 3, figsize=(10, 7), layout="constrained")
    background = tone(ref[..., 0]) * 0.65
    for ax, (name, mask) in zip(axs.flat, roi_masks.items()):
        view = background.copy()
        view[mask] = 0.55 * view[mask] + 0.45 * np.array([0.3, 0.95, 0.7])
        ax.imshow(view, interpolation="none")
        ax.set_title(f"{name.replace('_', ' ').title()} · {mask.sum():,} px")
        ax.set_axis_off()
    fig.savefig(out / "regions.png", dpi=135)
    plt.close(fig)


def main(raw, out, ref_dir, analysis_commit):
    manifest = json.loads((raw / "run_manifest.json").read_text())
    assert manifest["protocol_sha256"] == sha(PROTOCOL)
    assert manifest["execution_amendment_sha256"] == sha(EXECUTION)
    assert manifest["gates"]["table_bit_identical"]
    for name, expected in P["references"]["sha256"].items():
        assert sha(ref_dir / name) == expected
    roi_masks = masks(ref_dir / "MAP_0.npz")
    refs = np.stack([components(image(ref_dir / f"REF_0_{h}.npz", "h")) for h in (0, 1)])

    def verified(row):
        path = raw / row["path"]
        assert sha(path) == row["sha256"], path
        return components(image(path))

    replays = np.stack([verified(row) for row in manifest["matched"]])
    historic = components(image(ref_dir / "PRED_B_0.npz"))
    rng = np.random.default_rng(10169999)
    regions, stable, curves = {}, {}, {}
    for tag, mask in roi_masks.items():
        summary, *internal = stable_stats(replays, refs, historic, mask, rng)
        regions[tag] = {"converged": summary, "spp": {}}
        stable[tag] = internal
        curves[tag] = []

    timing, previews = {}, {}
    spp_values = P["sweep"]["spp"]
    for spp in spp_values:
        rows = [row for row in manifest["frames"] if row["spp"] == spp]
        expected_seeds = [sample_seed(spp, rep) for rep in range(P["sweep"]["replicates"])]
        assert [row["seed"] for row in rows] == expected_seeds
        frames = np.stack([verified(row) for row in rows])
        assert len(frames) == P["sweep"]["replicates"]
        previews[spp] = frames[0].copy()
        if spp == 32:
            noise32 = frames[0, ..., 1] - frames[1:, ..., 1].mean(axis=0, dtype=np.float64)
        ms = np.array([row["seconds"] * 1000 for row in rows if not row["foreign_pids_after"]])
        timing[str(spp)] = {
            "valid_frames": len(ms), "occupied_frames_excluded": len(rows) - len(ms),
            "median_ms": float(np.median(ms)) if len(ms) else None,
            "p25_ms": float(np.percentile(ms, 25)) if len(ms) else None,
            "p75_ms": float(np.percentile(ms, 75)) if len(ms) else None,
        }
        for tag, mask in roi_masks.items():
            stats, draws = sweep_stats(frames, refs, mask, *stable[tag], rng)
            regions[tag]["spp"][str(spp)] = stats
            curves[tag].append(draws)

    trends = {tag: slopes(spp_values, regions[tag]["spp"], curve) for tag, curve in curves.items()}
    verdict = classify(regions, trends)
    result = {
        "verdict": verdict, "render_commit": manifest["gates"]["code_commit"],
        "analysis_commit": analysis_commit, "protocol_sha256": sha(PROTOCOL),
        "render_manifest_sha256": sha(raw / "run_manifest.json"), "render_gates": manifest["gates"],
        "render_environment": manifest["environment"],
        "execution_amendment_sha256": sha(EXECUTION),
        "analysis_environment": {"python": platform.python_version(), "numpy": np.__version__,
                                 "matplotlib": matplotlib.__version__},
        "timing": timing, "convergence": trends, "rois": regions,
    }
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "metrics.json", result)
    ref, conv = refs.mean(axis=0, dtype=np.float64), replays.mean(axis=0, dtype=np.float64)
    ref_noise = (refs[0, ..., 0].astype(np.float64) - refs[1, ..., 0]) * 0.5
    figures(out, regions, previews, ref, conv, noise32, ref_noise, roi_masks)
    np.savez_compressed(raw / "analysis_maps.npz", **roi_masks, reference=ref, converged=conv,
                        stochastic_low_32=noise32, reference_noise_full=ref_noise)
    print(verdict)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw", type=Path)
    parser.add_argument("reviewer", type=Path)
    parser.add_argument("reference", type=Path)
    parser.add_argument("commit")
    args = parser.parse_args()
    main(args.raw, args.reviewer, args.reference, args.commit)
