"""WL104A estimators over independent replicates (validated by tests104a.py).

Energy: ratio of means sum_r num_r / sum_r den_r with a delete-one jackknife SE and a t_{R-1} 95 % interval.
Spatial: unbiased cross-replicate mean square of a difference field, E[D]^2 averaged over the ROI elements, from distinct replicate pairs
  S = mean_m [ (sum_r A_rm)(sum_r B_rm) - sum_r A_rm B_rm ] / (R (R - 1)),
  with A = B for paired-noise fields and A, B independent copies (e.g. two independent field tables) otherwise; jackknife interval.
"""
import numpy as np
from scipy import stats as sst


def t975(df):
    return float(sst.t.ppf(0.975, df))


def ratio_jk(num, den):
    num = np.asarray(num, np.float64); den = np.asarray(den, np.float64); R = len(num)
    pt = num.sum() / den.sum()
    loo = (num.sum() - num) / (den.sum() - den)
    se = float(np.sqrt((R - 1) / R * ((loo - loo.mean()) ** 2).sum()))
    h = t975(R - 1) * se
    return {'point': float(pt), 'se': se, 'lo': float(pt - h), 'hi': float(pt + h), 'R': R}


def stable_ms_jk(A, B=None):
    """A, B: [R, M] float64. Returns point S, jackknife SE and 95 % interval (t_{R-1})."""
    A = np.asarray(A, np.float64)
    B = A if B is None else np.asarray(B, np.float64)
    R, M = A.shape
    SA, SB = A.sum(0), B.sum(0)
    Pp = (A * B).sum(0)
    pt = float(((SA * SB - Pp) / (R * (R - 1))).mean())
    # leave-one-out: ((SA - A_i)(SB - B_i) - (P - A_i B_i)) / ((R - 1)(R - 2)), averaged over M
    m_SASB, m_P = float((SA * SB).mean()), float(Pp.mean())
    m_SA_Bi = (B @ SA) / M; m_Ai_SB = (A @ SB) / M; m_AiBi = (A * B).mean(1)
    loo = (m_SASB - m_SA_Bi - m_Ai_SB + m_AiBi - m_P + m_AiBi) / ((R - 1) * (R - 2))
    se = float(np.sqrt((R - 1) / R * ((loo - loo.mean()) ** 2).sum()))
    h = t975(R - 1) * se
    return {'point': pt, 'se': se, 'lo': pt - h, 'hi': pt + h, 'R': R}


def rel_rms(ms, norm_ms):
    """Interval of sqrt(max(S, 0)) / sqrt(norm_ms) from the interval of S (monotone transform)."""
    f = lambda v: float(np.sqrt(max(v, 0.0) / norm_ms))
    return {'point': f(ms['point']), 'lo': f(ms['lo']), 'hi': f(ms['hi']), 'raw_ms': ms}


def abs_bounds(ci):
    """Bounds of |x| from an interval of x: (lower, upper)."""
    lo, hi = ci['lo'], ci['hi']
    lower = 0.0 if lo <= 0 <= hi else min(abs(lo), abs(hi))
    return lower, max(abs(lo), abs(hi))
