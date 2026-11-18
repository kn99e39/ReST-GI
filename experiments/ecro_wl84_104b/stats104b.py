"""WL104B estimators: cross-replicate U-statistic with delete-one values (for ratios of mean squares) and the jackknife interval."""
import numpy as np
from scipy import stats as sst


def ms_loo(A, B):
    """Point and delete-one values of mean_m [(sum_r A)(sum_r B) - sum_r A B] / (R (R - 1)); A, B [R, M]."""
    A = np.asarray(A, np.float64); B = np.asarray(B, np.float64)
    R, M = A.shape
    SA, SB = A.sum(0), B.sum(0); Pp = (A * B).sum(0)
    pt = float(((SA * SB - Pp) / (R * (R - 1))).mean())
    m_SASB, m_P = float((SA * SB).mean()), float(Pp.mean())
    m_SA_Bi = (B @ SA) / M; m_Ai_SB = (A @ SB) / M; m_AiBi = (A * B).mean(1)
    loo = (m_SASB - m_SA_Bi - m_Ai_SB + m_AiBi - m_P + m_AiBi) / ((R - 1) * (R - 2))
    return pt, loo


def jk_ci(pt, loo):
    R = len(loo)
    se = float(np.sqrt((R - 1) / R * ((loo - loo.mean()) ** 2).sum()))
    h = float(sst.t.ppf(0.975, R - 1)) * se
    return {'point': float(pt), 'se': se, 'lo': float(pt - h), 'hi': float(pt + h)}
