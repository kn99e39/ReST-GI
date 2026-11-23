"""WL109A shared paths, protocol, seeds, masks and the original Q statistic in sufficient-statistic form (numpy; no GPU).

WL109A is a suffix batch of WL109: the estimator, renderer, checkpoint, table, ROIs and the Q / B1 definitions and limits are
imported unchanged from experiments/ecro_estimator109. Only statistical qualification, independent captures, focused
diagnostics and (conditionally) the dynamic stage are new.
"""
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
W109 = EXP / "ecro_estimator109"
sys.path.insert(0, str(W109))
for d in ("ecro_denoise105", "ecro_denoise107", "ecro_denoise108"):
    sys.path.insert(0, str(EXP / d))

import common109 as C9                      # noqa: E402  (frozen WL109 helpers: ROIs, bands, launch_seed, reading/writing)

PROTOCOL = HERE / "protocol109a.json"
P = json.loads(PROTOCOL.read_text())
P9 = C9.P                                   # frozen WL109 protocol (thresholds read from here, never redefined)
EVAL_VIEWS = C9.EVAL_VIEWS
GRID = C9.GRID
NOMINAL = C9.NOMINAL
DECISION_ROIS = C9.DECISION_ROIS
BIAS_SPP = tuple(P9["analysis"]["bias_spp"])
QLIM = P9["analysis"]["spatial_bias_power_max"]
TAU = P9["analysis"]["bias_tolerance_relative"]
LUMA = C9.LUMA
sha, read, write, array_sha, launch_seed = C9.sha, C9.read, C9.write, C9.array_sha, C9.launch_seed

# Student-t 97.5% quantiles for G - 1 degrees of freedom: the frozen stats109.T975 table (7 df = 2.365, as used by WL109)
# where it is defined, scipy otherwise (group-count sensitivity only)
import stats109 as _ST  # noqa: E402


def t975(df):
    if df in _ST.T975:
        return _ST.T975[df]
    from scipy.stats import t
    return float(t.ppf(0.975, df))


# ------------------------------------------------------------------------------------------------ seeds (WL109A families)
def _off(smoke):
    return P["seeds"]["smoke_offset"] if smoke else 0


def seed_null(vi, si, k, smoke=False):
    s = P["seeds"]
    assert vi < 10 and si < 10 and k < 10000
    return s["null"] + 100000 * vi + 10000 * si + k + _off(smoke)


def seed_paired(vi, si, r, smoke=False):
    s = P["seeds"]
    assert vi < 10 and si < 10 and r < 10000
    return s["paired"] + 100000 * vi + 10000 * si + r + _off(smoke)


def seed_grid(vi, gi, r, smoke=False):
    s = P["seeds"]
    assert vi < 10 and gi < 100 and r < 1000
    return s["grid"] + 100000 * vi + 1000 * gi + r + _off(smoke)


def seed_timing(vi, slot, i, smoke=False):
    s = P["seeds"]
    assert vi < 10 and slot < 100 and i < 100
    return s["timing"] + 10000 * vi + 100 * slot + i + _off(smoke)


def seed_dynamic(slot, r, smoke=False):
    s = P["seeds"]
    assert slot < 90 and r < 10000
    return s["dynamic"] + 10000 * slot + r + _off(smoke)


def seed_dynamic_reference(block, smoke=False):
    assert block < 1000
    return P["seeds"]["dynamic_reference"] + block + _off(smoke)


def seed_mech(slot, r, smoke=False):
    s = P["seeds"]
    assert slot < 100 and r < 10000
    return s["mechanism"] + 10000 * slot + r + _off(smoke)


# ------------------------------------------------------------------------------------------------ masks
def all_masks(aux):
    """Frozen WL109 decision ROIs + contact (descriptive) + sphere-silhouette distance bands (WL109 definition) +
    material-boundary distance bands and sphere-coverage bins (WL109A descriptive diagnostics)."""
    from scipy.ndimage import distance_transform_edt
    m = C9.rois(aux)
    out = {k: m[k] for k in list(DECISION_ROIS) + ["contact"]}
    for k, v in C9.distance_bands(aux, P9["analysis"]["band_width"]).items():
        out[f"band{k}"] = v
    valid = (aux["valid_frac"] > 0) & ~m["contact"]
    mb = m["material_boundary"]
    d = distance_transform_edt(~mb)
    for k in range(0, P["diagnostics"]["mb_band_width"] + 1):
        out[f"mbband{k}"] = (np.rint(d) == k) & valid & (aux["cover"][6] == 0) if k else mb
    sph = aux["cover"][6]
    edges = P["diagnostics"]["coverage_bins"]
    for lo, hi in zip(edges, edges[1:]):
        out[f"cov{lo:.2f}-{hi:.2f}"] = m["silhouette"] & (sph >= lo) & (sph < hi)
    return {k: v for k, v in out.items() if v.sum() > 0}


def raw_pixels(masks):
    """Pixels kept per frame in raw form (small, high-variance decision ROIs)."""
    u = np.zeros_like(masks["primary"])
    for k in P["raw_rois"]:
        if k in masks:
            u |= masks[k]
    return np.flatnonzero(u)


# ------------------------------------------------------------------------------------------------ the original Q statistic
def q_from_group_sums(S1, S2, n, scale):
    """Original WL109 Q (stats109.spatial_bias_power) from per-group sums over n replicates of D and D^2.
    S1, S2: [G, ...pixels-and-channels...]. Returns per-group Q_g = mean(Dbar_g^2 - s_g^2 / n) / scale."""
    S1 = np.asarray(S1, np.float64)
    S2 = np.asarray(S2, np.float64)
    G = S1.shape[0]
    dbar = S1 / n
    s2 = (S2 - S1 * S1 / n) / (n - 1)
    return ((dbar * dbar - s2 / n).reshape(G, -1).mean(1)) / scale


def q_summary(qg):
    """point, SE (spread of the independent group values), Student-t two-sided 95% bounds with G - 1 df (WL109 form)."""
    qg = np.asarray(qg, np.float64)
    G = qg.size
    point = float(qg.mean())
    se = float(qg.std(ddof=1) / math.sqrt(G))
    t = t975(G - 1)
    return {"point": point, "se": se, "upper95": point + t * se, "lower95": point - t * se, "groups": G}


def q_state(summ_by_spp, limit):
    """WL109A three-way Q state over the bias spp: EQUIVALENT (every upper95 <= limit), BIASED (any lower95 > limit),
    otherwise UNRESOLVED."""
    ups = [s["upper95"] for s in summ_by_spp.values()]
    los = [s["lower95"] for s in summ_by_spp.values()]
    if any(lo > limit for lo in los):
        return "BIASED"
    if all(u <= limit for u in ups):
        return "EQUIVALENT"
    return "UNRESOLVED"


def b1_state(d_by_spp, ref_luma):
    """Original WL109 B1: inverse-variance pooled paired luma difference over the bias spp, relative to the reference."""
    ests, ws = [], []
    for d in d_by_spp.values():
        d = np.asarray(d, np.float64)
        se = d.std(ddof=1) / math.sqrt(len(d))
        ests.append(d.mean()); ws.append(1 / se ** 2 if se > 0 else 1e30)
    ests, ws = np.asarray(ests), np.asarray(ws)
    m = float((ests * ws).sum() / ws.sum()); se = float(1 / math.sqrt(ws.sum()))
    z = 1.959963984540054
    rel, rse = m / ref_luma, se / abs(ref_luma)
    lo, hi = rel - z * rse, rel + z * rse
    state = "EQUIVALENT" if (lo >= -TAU and hi <= TAU) else ("BIASED" if (lo > TAU or hi < -TAU) else "UNRESOLVED")
    return {"relative": rel, "se": rse, "ci95": [lo, hi], "state": state, "resolved_nonzero": bool(lo > 0 or hi < 0)}


def bias_pattern(kind, P_pix, aux_vals, rs):
    """Unit-power injection patterns b over P_pix ROI pixels (3 equal channels), mean(b^2) = 1 before scaling.
    aux_vals: dict with 'noise_sd' [P] and 'xy' [P, 2] pixel coordinates."""
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
