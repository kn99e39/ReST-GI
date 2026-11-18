"""WL101H estimators (torch, float64, CPU or CUDA). Validated by tests_h.py on synthetic data with known moments.

Conventions: X has shape [R, M] (R independent replicates, M = ROI pixels x 3 RGB channels). ROI aggregate = mean over M of the
per-element moment (equal per-pixel / per-channel weight, as WL101A). Bootstrap weights W [B, R] are multinomial counts
summing to R; unit weights reproduce the point estimate exactly.
"""
import math

import numpy as np
import torch


def gram(vecs, W=None, chunk=64):
    """ROI-aggregated ddof=1 covariance matrix between components.

    vecs: list of K tensors [R, M]. Returns [K, K] (W None) or [B, K, K].
    Element (a, b) = mean_m [ sum_r w_r a_rm b_rm - (sum_r w_r a_rm)(sum_r w_r b_rm) / R ] / (R - 1).
    """
    R, M = vecs[0].shape
    K = len(vecs)
    if W is None:
        W = torch.ones(1, R, dtype=torch.float64, device=vecs[0].device)
        squeeze = True
    else:
        squeeze = False
    B = W.shape[0]
    out = torch.zeros(B, K, K, dtype=torch.float64, device=vecs[0].device)
    sq = torch.zeros(R, K, K, dtype=torch.float64, device=vecs[0].device)       # per-replicate sum_m a_rm b_rm
    for a in range(K):
        for b in range(a, K):
            v = (vecs[a] * vecs[b]).sum(1)
            sq[:, a, b] = v; sq[:, b, a] = v
    t1 = torch.einsum('br,rkl->bkl', W, sq)
    for s in range(0, B, chunk):
        Wc = W[s:s + chunk]
        means = [Wc @ v for v in vecs]                                            # [b, M] weighted sums
        for a in range(K):
            for b in range(a, K):
                v = (means[a] * means[b]).sum(1) / R
                out[s:s + chunk, a, b] = v; out[s:s + chunk, b, a] = v
    out = (t1 - out) / ((R - 1) * M)
    return out[0] if squeeze else out


def pair_moment(da, db):
    """Per-replicate ROI sums of 0.5 * da * db (pick-freeze difference estimator). da, db: [R, M]. Returns [R]."""
    return 0.5 * (da * db).sum(1)


def pf_decompose(A, Bv, C, W=None):
    """Pick-freeze stage decomposition. A, Bv, C: lists of K tensors [R, M] for triples
    A = (J, P, S), B = (J, P, S'), C = (J, P', S'') sharing the listed stream seeds within a replicate.
    Returns K x K (or [Bw, K, K]) matrices: tot (ddof=1 across replicates of A), J, P, S with tot = J + P + S.
    S = E[Cov(.|J,P)] = mean 0.5 (A-B)(A-B)^T ; P + S = E[Cov(.|J)] = mean 0.5 (A-C)(A-C)^T.
    """
    K = len(A)
    R, M = A[0].shape
    dS = [a - b for a, b in zip(A, Bv)]
    dPS = [a - c for a, c in zip(A, C)]
    ps_rep = torch.zeros(R, K, K, dtype=torch.float64, device=A[0].device)
    s_rep = torch.zeros_like(ps_rep)
    for a in range(K):
        for b in range(a, K):
            v = pair_moment(dPS[a], dPS[b]) / M; ps_rep[:, a, b] = v; ps_rep[:, b, a] = v
            v = pair_moment(dS[a], dS[b]) / M; s_rep[:, a, b] = v; s_rep[:, b, a] = v
    if W is None:
        tot = gram(A)
        ps, s = ps_rep.mean(0), s_rep.mean(0)
    else:
        tot = gram(A, W)
        ps, s = weighted_mean(ps_rep, W), weighted_mean(s_rep, W)
    return {'tot': tot, 'J': tot - ps, 'P': ps - s, 'S': s}


def weighted_mean(per_rep, W):
    """per_rep [R, ...] -> [B, ...] weighted mean with weights W [B, R] (counts summing to R)."""
    R = per_rep.shape[0]
    return torch.einsum('br,r...->b...', W, per_rep) / R


def boot_weights(R, B, seed, device):
    g = np.random.default_rng(seed)
    idx = g.integers(0, R, size=(B, R))
    W = np.zeros((B, R))
    np.add.at(W, (np.repeat(np.arange(B), R), idx.ravel()), 1.0)
    return torch.as_tensor(W, dtype=torch.float64, device=device)


def ci(draws, point, level=0.95):
    """Normal interval point +- z * bootstrap SE (scalar draws [B]).

    Percentile / basic intervals of replicate-resampled ddof=1 variances under-cover badly when M >> R (the duplicate-induced
    downward bias of each draw exceeds the spread); the bootstrap SE itself is reliable (tests_h.py, coverage check).
    """
    d = np.asarray(draws, np.float64)
    se = float(d.std(ddof=1))
    z = z_two_sided(1 - level)
    point = float(point)
    return {'point': point, 'lo': point - z * se, 'hi': point + z * se, 'se': se}


def stable_bias_sq(g1, g2, r1, r2):
    """Unbiased squared-bias cross product (WL101C convention, both pairings averaged). Inputs [K, M] component stacks.

    Returns [K, K]: mean_m 0.5 * [(g1 - r1)_a (g2 - r2)_b + (g1 - r2)_a (g2 - r1)_b], symmetrised in (a, b).
    """
    d1, d2 = g1 - r1, g2 - r2
    e1, e2 = g1 - r2, g2 - r1
    M = g1.shape[1]
    m = 0.5 * (d1 @ d2.T + e1 @ e2.T) / M
    return 0.5 * (m + m.T)


def ref_mean_noise(r1, r2):
    """Variance of the mean of two independent halves, per element: (r1 - r2)^2 / 4, ROI mean. [K, M] -> [K]."""
    return ((r1 - r2) ** 2).mean(1) / 4.0


def log_slope(spp, vals):
    x = np.log(np.asarray(spp, np.float64))
    y = np.log(np.asarray(vals, np.float64))
    A = np.vstack([x, np.ones_like(x)]).T
    return float(np.linalg.lstsq(A, y, rcond=None)[0][0])


def z_two_sided(alpha):
    """Normal quantile for a two-sided level alpha (no scipy dependency)."""
    lo, hi = 0.0, 10.0
    target = 1 - alpha / 2
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if 0.5 * (1 + math.erf(mid / math.sqrt(2))) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)
