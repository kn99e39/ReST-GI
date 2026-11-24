"""WL109B shared paths, protocol, seeds and the WL109B spatial-bias inference (numpy; no GPU).

WL109B is a suffix batch of WL109 / WL109A. The estimator (frozen C1, M = 1), renderer, checkpoint, table, ROIs, the Q
estimand, its normalization, the limits and B1 are imported unchanged from experiments/ecro_estimator109 (through the
WL109A helpers, which are themselves unchanged imports). Only the confidence procedure for Q, the calibration, the
independent captures, statistical reporting and dynamic-closure instrumentation are new.

Q estimand (WL109 stats109.spatial_bias_power, unchanged):
    Q = mean_{x in ROI, c in RGB} mu_D(x, c)^2 / scale,   D = H_C1 - H_R0 (paired; LOW cancels bitwise),
    scale = mean_{ROI, RGB} Var(TOTAL_R0)  (per-pixel sample variance over the paired R0 frames).
"""
import json
import math
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
W109A = EXP / "ecro_estimator109a"
sys.path.insert(0, str(W109A))

import common109a as CA                      # noqa: E402  (unchanged WL109A helpers; they import the frozen WL109 code)
from common109a import (C9, P9, W109, EVAL_VIEWS, GRID, NOMINAL, DECISION_ROIS, BIAS_SPP, QLIM, TAU, LUMA,  # noqa: E402,F401
                        sha, read, write, array_sha, launch_seed, t975, all_masks, raw_pixels, b1_state, q_from_group_sums,
                        q_summary)

import os  # noqa: E402
if os.environ.get("WL109B_ADMISSION") == "lab70":           # Amendment 2: LabServer70 admission (environment pins only)
    import renderer108 as _R108                                # noqa: E402
    import admission70 as _A70                                 # noqa: E402
    _R108.frozen_runtime = _A70.frozen_runtime

PROTOCOL = HERE / "protocol109b.json"
P = json.loads(PROTOCOL.read_text())
INF = P["inference"]


# ------------------------------------------------------------------------------------------------ seeds (WL109B families)
def _off(smoke):
    return P["seeds"]["smoke_offset"] if smoke else 0


def seed_null(vi, si, k, smoke=False):
    assert vi < 10 and si < 10 and k < 10000
    return P["seeds"]["null"] + 100000 * vi + 10000 * si + k + _off(smoke)


def seed_paired(vi, si, r, smoke=False):
    assert vi < 10 and si < 10 and r < 10000
    return P["seeds"]["paired"] + 100000 * vi + 10000 * si + r + _off(smoke)


def seed_replication(vi, si, r, smoke=False):
    assert vi < 10 and si < 10 and r < 10000
    return P["seeds"]["replication"] + 100000 * vi + 10000 * si + r + _off(smoke)


def seed_grid(vi, gi, r, smoke=False):
    assert vi < 10 and gi < 100 and r < 1000
    return P["seeds"]["grid"] + 100000 * vi + 1000 * gi + r + _off(smoke)


def seed_timing(vi, slot, i, smoke=False):
    assert vi < 10 and slot < 100 and i < 100
    return P["seeds"]["timing"] + 10000 * vi + 100 * slot + i + _off(smoke)


def seed_dynamic(slot, r, smoke=False):
    assert slot < 90 and r < 10000
    return P["seeds"]["dynamic"] + 10000 * slot + r + _off(smoke)


def seed_dynamic_reference(block, smoke=False):
    assert block < 1000
    return P["seeds"]["dynamic_reference"] + block + _off(smoke)


def seed_mech(slot, r, smoke=False):
    assert slot < 100 and r < 10000
    return P["seeds"]["mechanism"] + 10000 * slot + r + _off(smoke)


# ------------------------------------------------------------------------------------------------ WL109B inference (XJL)
def t_quant(p, df):
    from scipy.stats import t
    return float(t.ppf(p, df))


def xj_core(dbar, scale):
    """Cross-product U-statistic over distinct groups and its delete-one-group jackknife variance.
    dbar [G, X]: group means of D (X = ROI pixels x RGB). Returns (U, se2_jack).
        U = sum_{g != h} <dbar_g, dbar_h> / (G (G - 1) X scale)      (unbiased for Q when the G groups are independent)."""
    dbar = np.asarray(dbar, np.float64)
    G, X = dbar.shape
    S = dbar.sum(0); sq = (dbar * dbar).sum(0)
    U = float((S * S - sq).sum() / (G * (G - 1) * X * scale))
    Sm = S[None] - dbar; sqm = sq[None] - dbar * dbar
    Um = (Sm * Sm - sqm).sum(1) / ((G - 1) * (G - 2) * X * scale)
    ps = G * U - (G - 1) * Um
    return U, float(ps.var(ddof=1) / G)


def lam_max(sub_means, sub):
    """Largest eigenvalue of the per-pair covariance of D over the ROI elements, estimated from k subgroup means of `sub`
    pairs each ([k, X]) as sub x top eigenvalue of their sample covariance. Upward-biased when X >> k (conservative)."""
    m = np.asarray(sub_means, np.float64)
    k = m.shape[0]
    m = m - m.mean(0, keepdims=True)
    if m.shape[1] > k:
        g = m @ m.T
        top = float(np.linalg.eigvalsh(g)[-1])
    else:
        top = float(np.linalg.eigvalsh(m.T @ m)[-1])
    return top / (k - 1) * sub


def xjl_bounds(dbar, scale, lam, R, alpha_up=None, alpha_lo=None):
    """WL109B primary procedure (frozen): inverted bounds with the jackknife variance plus the worst-case linear-term variance.

    With V(q) = se2_jack + b q,  b = 4 lam / (R X scale)  (Var of the linear term 2 <mu, ebar> / (X scale) is at most
    4 lam |mu|^2 / (R X^2 scale^2) = b q for any bias field mu with mean square q scale),
        upper = sup{q >= 0 : U >= q - t_up sqrt(V(q))},   lower = inf{q >= 0 : U <= q + t_lo sqrt(V(q))}  (0 if none),
    t from Student t with G - 1 df at one-sided levels alpha_up / alpha_lo."""
    G, X = np.asarray(dbar).shape
    U, se2 = xj_core(dbar, scale)
    a_up = INF["alpha_upper_one_sided"] if alpha_up is None else alpha_up
    a_lo = INF["alpha_lower_one_sided_family"] if alpha_lo is None else alpha_lo
    b = 4.0 * lam / (R * X * scale)
    a = t_quant(1 - a_up, G - 1) ** 2
    disc = a * a * b * b / 4 + a * se2 + a * b * U
    upper = max(0.0, U + a * b / 2 + math.sqrt(disc)) if disc > 0 else 0.0
    al = t_quant(1 - a_lo, G - 1) ** 2
    disc = al * al * b * b / 4 + al * se2 + al * b * U
    lower = (U + al * b / 2 - math.sqrt(disc)) if disc > 0 else 0.0
    lower = max(lower, 0.0) if lower == lower else 0.0
    return {"point": U, "se_jack": math.sqrt(max(se2, 0.0)), "b": b, "lam": lam, "upper": upper, "lower_family": lower,
            "groups": G, "alpha_upper": a_up, "alpha_lower": a_lo}


def q_state_b(summ_by_spp, limit):
    """EQUIVALENT if every upper <= limit (intersection-union: no multiplicity adjustment needed for the equivalence claim);
    BIASED if any family-adjusted lower > limit; otherwise UNRESOLVED."""
    if any(s["lower_family"] > limit for s in summ_by_spp.values()):
        return "BIASED"
    if all(s["upper"] <= limit for s in summ_by_spp.values()):
        return "EQUIVALENT"
    return "UNRESOLVED"


def groups_from_subgroups(SG1, sub, G):
    """[nsub, ...] 16-pair subgroup sums -> [G, ...] group means (contiguous subgroups, n = nsub * sub / G pairs per group)."""
    nsub = SG1.shape[0]
    assert nsub % G == 0, (nsub, G)
    k = nsub // G
    return SG1.reshape((G, k) + SG1.shape[1:]).sum(1) / (k * sub)


def bias_pattern(kind, P_pix, aux_vals, rs):
    """Unit-power injection patterns (WL109A definitions with WL109B protocol parameters)."""
    if kind == "uniform":
        b = np.ones(P_pix)
    elif kind == "noise_proportional":
        b = np.asarray(aux_vals["noise_sd"], np.float64).copy()
    elif kind == "localized":
        xy = aux_vals["xy"]
        c = xy[rs.integers(0, P_pix)]
        d = np.hypot(*(xy - c).T)
        k = max(1, int(round(P["calibration"]["localized_fraction"] * P_pix)))
        b = np.zeros(P_pix); b[np.argsort(d)[:k]] = 1.0
    elif kind == "signed_smooth":
        xy = aux_vals["xy"]
        f = rs.normal(size=2) * 2 * math.pi / P["calibration"]["smooth_wavelength_px"]
        ph = rs.uniform(0, 2 * math.pi)
        b = np.sign(np.sin(xy @ f + ph))
        b[b == 0] = 1
    else:
        raise KeyError(kind)
    b = b / math.sqrt(max((b * b).mean(), 1e-30))
    return np.repeat(b[None, :], 3, 0)


def xy_of(mask_idx, W=512):
    return np.stack(np.divmod(mask_idx, W), 1).astype(np.float64)


def clopper_pearson(k, n, conf=0.95):
    from scipy.stats import beta
    a = (1 - conf) / 2
    lo = 0.0 if k == 0 else float(beta.ppf(a, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - a, k + 1, n - k))
    return lo, hi


def coverage_class(k, n, tol):
    """PASS if the 95% Clopper-Pearson upper bound of the non-coverage rate <= tol; FAIL if its lower bound > tol;
    otherwise BORDERLINE."""
    lo, hi = clopper_pearson(k, n)
    return ("PASS" if hi <= tol else ("FAIL" if lo > tol else "BORDERLINE")), lo, hi
