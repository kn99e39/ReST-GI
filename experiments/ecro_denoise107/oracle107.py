"""Two-simplex risk minimization and lower bounds for adaptive kernel outputs."""
import numpy as np


def project_simplex(v, xp=np):
    """Project the final axis onto the probability simplex."""
    s = xp.sort(v, axis=-1)[..., ::-1]
    k = xp.arange(1, v.shape[-1] + 1, dtype=v.dtype)
    offsets = (xp.cumsum(s, axis=-1) - 1) / k
    active = xp.sum(s > offsets, axis=-1, keepdims=True)
    threshold = xp.take_along_axis(offsets, active - 1, axis=-1)
    return xp.maximum(v - threshold, 0)


def solve_numpy(objective, lipschitz, initial, steps=320):
    """Vectorized FISTA, with a Frank–Wolfe lower certificate for each pixel."""
    w = initial.copy()
    z = w.copy()
    t = 1.0
    for _ in range(steps):
        _, grad = objective(z)
        updated = project_simplex(z - grad / lipschitz[:, None, None])
        t_next = (1 + np.sqrt(1 + 4 * t * t)) / 2
        z = updated + ((t - 1) / t_next) * (updated - w)
        w, t = updated, t_next
    value, grad = objective(w)
    gap = (np.sum(grad * w, axis=-1) - np.min(grad, axis=-1)).sum(axis=-1)
    assert np.isfinite(w).all() and np.min(w) >= 0
    assert np.max(np.abs(w.sum(axis=-1) - 1)) < 1e-10
    assert np.min(gap) >= -1e-10
    return w, value, np.maximum(0, value - gap), gap


def synthetic_risk(mean, spp, strength):
    """Exact RGB covariance for local independent draws plus a shared frame draw.

    mean[pixel, component, tap, rgb]. LOW/HIGH share the same local random sign,
    and all taps and both components share the frame sign. Each color sees the
    same sign drawn for its component at that tap.
    """
    centered = mean - mean[:, :, 112:113, :]
    independent = np.array([0.63, 0.51]) ** 2
    common = 0.42 ** 2
    frame = 0.10 ** 2
    scale = strength / spp
    norm = (mean * mean).mean(axis=-1)
    cross = (mean[:, 0] * mean[:, 1]).mean(axis=-1)

    def evaluate(w):
        b = np.einsum("nktc,nkt->nc", centered, w)
        out = np.einsum("nktc,nkt->nc", mean, w)
        bias = np.mean(b * b, axis=-1)
        diagonal = (norm * w * w * (independent + common)[None, :, None]).sum(axis=(1, 2))
        covar = 2 * common * (cross * w[:, 0] * w[:, 1]).sum(axis=-1)
        global_draw = frame * np.mean(out * out, axis=-1)
        variance = (diagonal + covar + global_draw) / spp
        bias_grad = np.einsum("nktc,nc->nkt", centered, b) / 3
        frame_grad = frame * np.einsum("nktc,nc->nkt", centered, out) / 3
        local_grad = norm * w * (independent + common)[None, :, None]
        local_grad += common * cross[:, None] * w[:, ::-1]
        return bias + strength * variance, 2 * (bias_grad + scale * (frame_grad + local_grad))

    # On the affine simplex, the common-frame columns can be centered. A trace
    # bound for the low-rank part and a row-sum bound for local noise give L.
    local_bound = (norm * (independent + common)[None, :, None]
                   + common * np.abs(cross)[:, None]).max(axis=(1, 2))
    low_rank_bound = (centered * centered).sum(axis=(1, 2, 3)) / 3
    bound = 2 * ((1 + scale * frame) * low_rank_bound + scale * local_bound)
    return evaluate, bound


def adaptive_intervals(luma_taps):
    """Every R3 output luminance lies in this interval, even with adaptive weights."""
    return luma_taps.min(axis=-1).sum(axis=-1), luma_taps.max(axis=-1).sum(axis=-1)


def adaptive_risk_bound(lo, hi, target):
    """Squared distance to the interval is a label-aware lower bound on MSE."""
    return np.square(np.clip(target, lo, hi) - target)


def empirical_parts(low, high):
    """Per-location replicate variance and the signed cross component."""
    low = np.asarray(low, dtype=np.float64)
    high = np.asarray(high, dtype=np.float64)
    l = low - low.mean(axis=0)
    h = high - high.mean(axis=0)
    n = len(l) - 1
    vl = (l * l).sum(axis=0) / n
    vh = (h * h).sum(axis=0) / n
    twice_cov = 2 * (l * h).sum(axis=0) / n
    vt = ((l + h) ** 2).sum(axis=0) / n
    assert np.allclose(vl + vh + twice_cov, vt, rtol=1e-6, atol=1e-10)
    return vl, vh, twice_cov, vt
