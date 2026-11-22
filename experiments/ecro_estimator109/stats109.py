"""WL109 replicate statistics (torch, float64). Arrays are [R, 3, H, W] (independent replicates of per-pixel RGB means).

Pixels are never treated as independent repetitions: every uncertainty is a bootstrap over replicate indices (whole frames,
preserving spatial correlation), reported as a normal interval with the bootstrap SE (WL103's coverage-tested form).
"""
import math

import numpy as np
import torch

Z95 = 1.959963984540054
LUMA = torch.tensor((0.2126, 0.7152, 0.0722), dtype=torch.float64)


def _m(mask, device):
    return torch.as_tensor(mask, device=device, dtype=torch.bool)


def pix_var(X, idx=None):
    if idx is not None:
        X = X[idx]
    return X.var(0, unbiased=True)


def pix_cov(X, Y, idx=None):
    if idx is not None:
        X, Y = X[idx], Y[idx]
    R = X.shape[0]
    return ((X - X.mean(0)) * (Y - Y.mean(0))).sum(0) / (R - 1)


def pooled(t, mask):
    """mean over channels and masked pixels of a [3,H,W] per-pixel statistic."""
    m = _m(mask, t.device)
    return float(t[:, m].mean())


class Boot:
    """Replicate-index bootstrap with a fixed generator; the same draws serve every statistic of one call site."""

    def __init__(self, R, draws, seed, device):
        g = torch.Generator(device="cpu").manual_seed(int(seed))
        self.idx = torch.randint(0, R, (draws, R), generator=g).to(device)

    def se(self, fn):
        vals = np.array([fn(i) for i in self.idx], dtype=np.float64)
        return float(np.nanstd(vals, ddof=1)), vals


def _div(a, b):
    return a / b if b != 0 else float("nan")


def ratio_stat(num_fn, den_fn, boot):
    point = num_fn(None) / den_fn(None)
    se, _ = boot.se(lambda i: _div(num_fn(i), den_fn(i)))
    return {"point": float(point), "se": se, "ci95": [point - Z95 * se, point + Z95 * se]}


def region_luma_means(X, mask):
    """[R] replicate means of luminance over the mask."""
    m = _m(mask, X.device)
    y = torch.tensordot(LUMA.to(X.device), X, dims=([0], [1]))       # [R, H, W]
    return y[:, m].mean(1)


T975 = {3: 3.182, 5: 2.571, 7: 2.365, 15: 2.131}


def spatial_bias_power(D, mask, scale, groups):
    """Q = mean over G independent replicate groups of Q_g = mean_pc(Dbar_g^2 - s_g^2 / n_g) / scale (each unbiased for the
    squared paired bias power). SE from the spread of the independent Q_g (a frame bootstrap would share resampling weights
    across all pixels and misstate the pixel-averaged noise). upper95 uses Student t with G - 1 df."""
    m = _m(mask, D.device)
    X = D[:, :, m]
    R = X.shape[0]
    n = R // groups
    qs = []
    for g in range(groups):
        Y = X[g * n:(g + 1) * n]
        qs.append(float(((Y.mean(0) ** 2) - Y.var(0, unbiased=True) / n).mean() / scale))
    qs = np.asarray(qs)
    point = float(qs.mean())
    se = float(qs.std(ddof=1) / math.sqrt(groups))
    return {"point": point, "se": se, "upper95": point + T975[groups - 1] * se, "groups": groups}


def t_test(vals):
    vals = torch.as_tensor(vals, dtype=torch.float64)
    R = vals.numel()
    mean = float(vals.mean())
    sd = float(vals.std(unbiased=True))
    se = sd / math.sqrt(R)
    return {"mean": mean, "se": se, "t": (mean / se) if se > 0 else (0.0 if mean == 0 else math.inf)}


def numpy_pooled_var(X, mask):
    """CPU float64 reference of pooled(pix_var(X), mask) for the CPU/GPU agreement check."""
    X = np.asarray(X, np.float64)
    v = X.var(0, ddof=1)
    return float(v[:, mask].mean())
