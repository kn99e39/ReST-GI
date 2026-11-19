"""Analytic rim scene family and controlled finite-spp information probes."""
import argparse
from itertools import product
import math
from pathlib import Path

import numpy as np

from common107 import LAMBDAS, LUMA, P, SPPS, read, sha, stencil, take_stencil, write
from oracle107 import empirical_parts, solve_numpy, synthetic_risk

SIZE = 64
ROWS = np.array([8, 16, 24, 32, 40, 48, 55])
COLS = np.arange(25, 40)
SYNTH_SEED = P["seeds"]["synthetic"]
N_DRAWS = 256


def cases():
    for offset, contrast in product((0.125, 0.3, 0.5, 0.875), ("bright", "dark")):
        for field, width, wavy in (("constant", None, False), ("smooth", None, True),
                                   *(("rim", w, y) for w, y in product((0.25, 1, 3), (False, True)))):
            yield offset, contrast, field, width, wavy


def case_name(case):
    offset, contrast, field, width, wavy = case
    return f"{offset:g}_{contrast}_{field}_{width}_{'wavy' if wavy else 'flat'}"


def colors(contrast):
    a = np.array([3.0, 2.4, 1.8])
    b = np.array([0.8, 0.9, 1.0])
    if contrast == "dark":
        a, b = b, a
    # Component definitions are optical-event groups, with distinct color ratios.
    fg_low, bg_low = np.array([0.68, 0.75, 0.61]), np.array([0.57, 0.61, 0.73])
    return np.stack((a * fg_low, a * (1 - fg_low))), np.stack((b * bg_low, b * (1 - bg_low)))


def truth(case):
    offset, contrast, field, width, wavy = case
    edge = SIZE / 2 + offset
    a, b = (np.maximum(np.arange(SIZE) + i - edge, 0) for i in (0, 1))
    c = b - a
    if field == "constant":
        integral = c
    elif field == "smooth":
        integral = 0.82 * c + 0.004 * (b * b - a * a) / 2
    else:
        integral = c - 0.82 * width * (np.exp(-a / width) - np.exp(-b / width))
    omega = 2 * np.pi / 24
    row = np.arange(SIZE)
    amp = 1 + 0.17 * (np.cos(omega * row) - np.cos(omega * (row + 1))) / omega if wavy else np.ones(SIZE)
    fg, bg = colors(contrast)
    return fg[:, :, None, None] * amp[None, None, :, None] * integral + bg[:, :, None, None] * (1 - c)


def midpoint_reference(case, n=1024):
    """Independent tensor-product midpoint check without the analytic antiderivative."""
    offset, contrast, field, width, wavy = case
    sub = (np.arange(n) + 0.5) / n
    inside_x = np.arange(SIZE)[:, None] + sub - (SIZE / 2 + offset)
    fg = inside_x >= 0
    if field == "constant":
        profile = np.ones_like(inside_x)
    elif field == "smooth":
        profile = 0.82 + 0.004 * inside_x
    else:
        profile = 1 - 0.82 * np.exp(-np.maximum(inside_x, 0) / width)
    row = np.arange(SIZE)[:, None] + sub
    modulation = (1 + 0.17 * np.sin(2 * np.pi * row / 24)).mean(axis=1) if wavy else np.ones(SIZE)
    a, b = colors(contrast)
    return (a[:, :, None, None] * modulation[None, None, :, None] * (fg * profile).mean(axis=1)
            + b[:, :, None, None] * (~fg).mean(axis=1))


def guides(offset):
    """Same kind of first moments as WL105, with a normal turning near the rim."""
    x = np.arange(SIZE)[None, :] + ((np.arange(8) + 0.5) / 8)[:, None]
    d = x - (SIZE / 2 + offset)
    on = d >= 0
    cos = np.sqrt(np.maximum(d, 0) / (3 + np.maximum(d, 0)))
    n_x = np.where(on, np.sqrt(1 - cos ** 2), 0)
    n_z = np.where(on, cos, 1)
    coverage = np.broadcast_to(on.mean(axis=0), (SIZE, SIZE))
    normal = np.broadcast_to(np.stack((n_x.mean(0), np.zeros(SIZE), n_z.mean(0)))[:, None, :], (3, SIZE, SIZE))
    depth = np.broadcast_to(np.where(on, 1.6 + 0.002 * np.maximum(d, 0), 2.2).mean(0), (SIZE, SIZE))
    return coverage, normal, depth


def draws(case, spp, rng, n):
    mean = truth(case)
    shape = (n, 1, 1, SIZE, SIZE)
    pixel = rng.integers(0, 2, size=shape, dtype=np.int8) * 2 - 1
    own = rng.integers(0, 2, size=(n, 2, 1, SIZE, SIZE), dtype=np.int8) * 2 - 1
    frame = rng.integers(0, 2, size=(n, 1, 1, 1, 1), dtype=np.int8) * 2 - 1
    amplitude = (0.42 * pixel + np.array([0.63, 0.51])[None, :, None, None, None] * own + 0.10 * frame)
    return (mean[None] * (1 + amplitude / np.sqrt(spp))).astype(np.float32)


def counterexamples():
    """Two finite-sample counterexamples with exact probabilities and limits."""
    # The center is 1 +/- 0.05, all other tap luminances exceed both outcomes.
    # Conditional on a high draw, every normalized nonnegative operator is >= 1.05.
    # With a balanced high/low draw and mean <=1.01, its variance is >=0.04^2.
    variance_floor = (1.05 - 1.01) ** 2
    raw_variance = 0.05 ** 2
    extremum = {"target": 1.0, "center_draws": [0.95, 1.05], "neighbor_min": 1.95,
                "variance_floor_at_one_percent_bias": variance_floor,
                "raw_variance": raw_variance, "ratio_floor": variance_floor / raw_variance,
                "maximum_feasible_reduction": 1 - variance_floor / raw_variance}
    # Geometry is equal in all guide moments. One quarter of samples hit the
    # high-cosine foreground in B and one quarter of those reveal the radiometric atom.
    p, base, amplitude = 1 / 16, 1.0, 0.8
    targets = np.array([base, base + p * amplitude])
    out = {}
    for spp in SPPS:
        miss = (1 - p) ** spp
        post_b = miss / (1 + miss)
        existing = base + (targets[1] - base) * post_b
        worlds = []
        for i, target in enumerate(targets):
            atom = 1 if i == 0 else miss
            expected = atom * existing + (1 - atom) * targets[1]
            variance = atom * (1 - atom) * (existing - targets[1]) ** 2
            worlds.append({"signed_bias": float((expected - target) / target),
                           "variance": float(variance),
                           "mse": float(atom * (existing - target) ** 2)})
        out[str(spp)] = {"prob_common_given_B": miss, "observation_overlap": miss, "existing_at_common_atom": existing,
                         "existing_worlds": worlds, "existing Bayes MSE": float(np.mean([x["mse"] for x in worlds])),
                         "privileged Bayes MSE": 0.0, "raw Bayes MSE": float(0.5 * amplitude ** 2 * p * (1 - p) / spp)}
    return {"adaptive_extremum": extremum, "equal_guide_pair": {"coverage": [0.5, 0.5],
            "foreground_mean_cosine": [0.5, 0.5], "foreground_mean_squared_cosine": [0.25, 0.34],
            "targets": targets.tolist(), "view": out,
            "scope": "Positive-probability identical full input at finite spp; distributions differ unconditionally."}}


def recon_metrics(y, clean, target, cov, nrows):
    """Report the signed sides and stochastic risk for the frozen test panel."""
    target_l = target @ LUMA
    est_l = clean @ LUMA
    st = y.sum(axis=1) @ LUMA
    parts = empirical_parts(y[:, 0] @ LUMA, y[:, 1] @ LUMA)
    band = (cov > 0) & (cov < 1)
    normalizer = target_l[band].mean()
    signed = (est_l - target_l) / normalizer
    report = {
        "signed_band_bias": float(signed[band].mean()),
        "band_profile_rms_bias": float(np.sqrt((signed[band] ** 2).mean())),
        "band_bias_sq_rgb": float(((clean - target)[band] ** 2).mean()),
        "band_variance_rgb": float(y.sum(axis=1)[:, band].var(axis=0, ddof=1).mean()),
        "band_relative_spatial_rmse": float(np.sqrt(((y.sum(axis=1)[:, band] - target[band]) ** 2).mean()
                                                    / (target[band] ** 2).mean())),
        "band_luma_variance_components": {name: float(v[band].mean()) for name, v in
                                          zip(("LOW", "HIGH", "2Cov", "TOTAL"), parts)},
        "band_luma_variance": float(st[:, band].var(axis=0, ddof=1).mean()),
        "profile_luma_expectation": est_l.reshape(nrows, -1).mean(axis=0).tolist(),
        "spatial_relative_rmse": float(np.sqrt(((y.sum(axis=1) - target) ** 2).mean() / (target ** 2).mean())),
        "side_energy": {name: float((est_l - target_l)[m].sum() / target_l[m].sum())
                        for name, m in (("background", cov == 0), ("mixed", band), ("foreground", cov == 1))},
    }
    return report


def run_case(case, rng):
    expected = truth(case)
    ref_error = float(np.max(np.abs(expected - midpoint_reference(case))))
    assert ref_error <= 0.002, (case, ref_error)
    idx = (ROWS[:, None] * SIZE + COLS).ravel()
    taps = stencil(idx, SIZE, SIZE)
    mean = take_stencil(expected, taps).transpose(2, 0, 3, 1)
    target = mean[:, :, 112, :].sum(axis=1)
    box_cover = np.clip(COLS + 1 - (SIZE / 2 + case[0]), 0, 1)
    cov = np.tile(box_cover, len(ROWS))
    gcov, norm, depth = guides(case[0])
    roi = (cov > 0) & (cov < 1)
    result = {"reference_error_max_abs": ref_error, "true_coverage_by_column": box_cover.tolist(),
              "guide_coverage_by_column": gcov[ROWS[0], COLS].tolist(),
              "normal_z_by_column": norm[2, ROWS[0], COLS].tolist(),
              "depth_by_column": depth[ROWS[0], COLS].tolist(),
              "target_luma_by_column": (target @ LUMA).reshape(len(ROWS), -1).mean(0).tolist(),
              "spp": {}}
    identity = np.zeros(mean.shape[:-1])
    identity[:, :, 112] = 1
    for spp in SPPS:
        noise = draws(case, spp, rng, N_DRAWS)
        sampled = take_stencil(noise, taps).transpose(0, 3, 1, 4, 2)
        raw = recon_metrics(sampled[:, :, :, 112, :].transpose(0, 2, 1, 3), target, target, cov, len(ROWS))
        spp_result = {"raw": raw, "oracle": {}}
        initial = identity
        for lam in LAMBDAS:
            f, L = synthetic_risk(mean, spp, lam)
            w, value, lower, gap = solve_numpy(f, L, initial)
            initial = w
            y = np.einsum("rnkpc,nkp->rknc", sampled, w)
            clean = np.einsum("nkpc,nkp->nc", mean, w)
            metrics = recon_metrics(y, clean, target, cov, len(ROWS))
            metrics["variance_ratio_to_raw"] = metrics["band_luma_variance"] / raw["band_luma_variance"]
            metrics["passes"] = bool(abs(metrics["signed_band_bias"]) <= 0.01
                                     and metrics["band_profile_rms_bias"] <= 0.01
                                     and metrics["variance_ratio_to_raw"] <= 0.50)
            metrics.update({"fit_RGB_risk": float(value[roi].mean()),
                            "fit_RGB_risk_lower_bound": float(lower[roi].mean()),
                            "relative_mean_gap": float(gap[roi].mean() / value[roi].mean())})
            spp_result["oracle"][str(lam)] = metrics
        result["spp"][str(spp)] = spp_result
    return result


def main(output, history, commit):
    output.mkdir(exist_ok=False, parents=True)
    historic = read(history)
    result = {"code_commit": commit, "protocol_sha256": sha(Path(__file__).parent / "protocol/denoise107.json"),
              "synthetic_seed": SYNTH_SEED, "draws": N_DRAWS,
              "WL106_historical_only": {"sha256": sha(history), "gate": historic["gate"]},
              "counterexamples": counterexamples(), "cases": {}}
    rng = np.random.default_rng(SYNTH_SEED)
    for case in cases():
        result["cases"][case_name(case)] = run_case(case, rng)
        print("Completed", case_name(case), flush=True)
        write(output / "synthetic107.json", result)
    assert len(result["cases"]) == 64
    result["complete"] = True
    write(output / "synthetic107.json", result)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("output", type=Path)
    p.add_argument("history", type=Path)
    p.add_argument("commit")
    a = p.parse_args()
    main(a.output, a.history, a.commit)
