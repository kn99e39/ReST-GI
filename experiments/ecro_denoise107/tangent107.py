"""Post hoc analytical feasibility check on the entire frozen synthetic family.

This control exploits the known orientation of the synthetic edge. It checks
whether a grid miss or loose solver gap is enough to exclude feasible weights.
"""
import argparse
from pathlib import Path

import numpy as np

from common107 import LUMA, read, sha, write
from synthetic107 import COLS, ROWS, cases, case_name, truth


def match_center(taps):
    """Unit-sum tilted weights preserving the center moment on each 15-tap column."""
    delta = taps - taps[..., 7:8]
    delta /= np.maximum(np.max(np.abs(delta), axis=-1, keepdims=True), 1e-14)
    interior = (delta.min(axis=-1) < -1e-12) & (delta.max(axis=-1) > 1e-12)
    def tilted(scaled):
        v = delta * scaled[..., None]
        e = np.exp(v - v.max(axis=-1, keepdims=True))
        return e / e.sum(axis=-1, keepdims=True)
    lo, hi = np.full(interior.shape, -1e7), np.full(interior.shape, 1e7)
    for _ in range(70):
        mid = (lo + hi) / 2
        negative = (tilted(mid) * delta).sum(axis=-1) < 0
        lo, hi = np.where(negative, mid, lo), np.where(negative, hi, mid)
    exact = np.abs(delta) <= 1e-12
    w = np.where(interior[..., None], tilted((lo + hi) / 2), exact / exact.sum(axis=-1, keepdims=True))
    assert (w >= 0).all() and np.allclose(w.sum(-1), 1, rtol=0, atol=1e-12)
    return w


def luma_variance_factor(luma, weights):
    """Variance times spp, including the LOW/HIGH cross-covariance."""
    low, high = luma[..., 0] * weights, luma[..., 1] * weights
    total = low + high
    local = ((0.42 * total) ** 2 + (0.63 * low) ** 2 + (0.51 * high) ** 2).sum(-1)
    return local + (0.10 * total.sum(-1)) ** 2


def check(case):
    image = truth(case)
    y = ROWS[:, None, None] + np.arange(-7, 8)[None, None, :]
    x = COLS[None, :, None]
    rgb = image[:, :, y, x].transpose(2, 3, 4, 0, 1)
    luma = rgb @ LUMA
    w = match_center(luma.sum(axis=-1))
    ref_rgb = rgb[..., 7, :, :]
    reconstructed = (w[..., None, None] * rgb).sum(axis=-3)
    difference = float(np.abs(reconstructed - ref_rgb).max())
    assert difference <= 1e-10
    raw = np.zeros_like(w)
    raw[..., 7] = 1
    band = int(np.flatnonzero(COLS == 32)[0])
    v = luma_variance_factor(luma, w)[:, band]
    r = luma_variance_factor(luma, raw)[:, band]
    return {"max_expected_RGB_component_error": difference,
            "band_variance_ratio_to_raw_all_spp": float(v.mean() / r.mean()),
            "band_max_pointwise_ratio_to_raw": float((v / r).max()),
            "weights_nonnegative_unit_sum": True}


def main(synthetic, output):
    evidence = read(synthetic)
    assert evidence["complete"]
    results = {case_name(c): check(c) for c in cases()}
    assert set(results) == set(evidence["cases"])
    write(output, {"label": "Post hoc check of every case; frozen risk-grid output retained",
                   "synthetic_evidence_sha256": sha(synthetic), "cases": results})


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("synthetic", type=Path)
    p.add_argument("output", type=Path)
    a = p.parse_args()
    main(a.synthetic, a.output)
