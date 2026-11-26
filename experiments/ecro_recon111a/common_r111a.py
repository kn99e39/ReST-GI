"""WL111A shared definitions. Reuses the frozen WL111 code (experiments/ecro_recon111) unchanged through sys.path; this
batch's outputs go to OUT (RECON111A_OUT, default ~/rest_sss101a/results/ecro_recon111a). RECON111_ROOT keeps pointing at
the WL111 root, whose inputs/ tree holds the frozen renderer inputs.

Seeds: WL111A families are 30,000,000 and above (every WL111 family is below 27,000,000), structured
base + i0 * 100000 + i1 * 1000 + i2, launched through WL111's splitmix64 launch_seed.
"""
import math
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
sys.path.insert(0, str(EXP / "ecro_recon111"))
import common_r111 as C  # noqa: E402

OUT = Path(os.environ.get("RECON111A_OUT", str(Path.home() / "rest_sss101a" / "results" / "ecro_recon111a")))
WL111_EVID = C.ROOT / "evidence"
DEV_VIEWS = tuple(f"Q{i:02d}" for i in range(8))
DEV_SPP = (1, 2, 4, 8)

FAMILIES = {"devref": 30_000_000, "devset": 31_000_000, "smoke": 39_000_000}


def seed(family, i0, i1, i2):
    assert 0 <= i0 < 10 and 0 <= i1 < 100 and 0 <= i2 < 1000, (family, i0, i1, i2)
    return FAMILIES[family] + i0 * 100_000 + i1 * 1_000 + i2


def wl111_weights(ri):
    return WL111_EVID / "models" / f"cand_r{ri}.pt"


class Moments:
    """Per-pixel running sums over independent realizations of one output (C,H,W) on the GPU (float64)."""

    def __init__(self, shape, torch, device="cuda"):
        self.s = torch.zeros(shape, dtype=torch.float64, device=device)
        self.q = torch.zeros(shape, dtype=torch.float64, device=device)
        self.n = 0

    def add(self, y):
        y = y.double()
        self.s += y
        self.q += y * y
        self.n += 1

    def mean_var(self):
        m = self.s / self.n
        v = (self.q / self.n - m * m) * self.n / (self.n - 1)
        return m.cpu().numpy(), np.maximum(v.cpu().numpy(), 0.0)


def decompose(mean, var, n, ref, var_ref, mask):
    """Bias / variance decomposition over a ROI (all channels). mean, var: per-pixel sample mean and unbiased variance of
    n independent reconstructions; ref: reference mean; var_ref: variance of the reference mean.
    -> per-frame debiased MSE = bias2 + variance, with bias2 = mean((mean - ref)^2) - mean(var) / n - mean(var_ref)."""
    d2 = float(((mean - ref)[:, mask] ** 2).mean())
    v = float(var[:, mask].mean())
    vr = float(var_ref[:, mask].mean())
    r2 = float((ref[:, mask] ** 2).mean())
    bias2 = d2 - v / n - vr
    return {"bias2": bias2, "var": v, "mse": bias2 + v, "ref2": r2, "var_ref": vr,
            "rel_rmse": math.sqrt(max(bias2 + v, 0.0) / r2), "rel_bias_rms": math.sqrt(max(bias2, 0.0) / r2),
            "rel_std": math.sqrt(v / r2), "bias_share": bias2 / (bias2 + v) if bias2 + v > 0 else None, "n_pix": int(mask.sum())}
