"""WL104A comparison core (pilot and final analysis share it)."""
import json
from pathlib import Path

import numpy as np

import stats104a as S

LUMA = np.array([0.2126, 0.7152, 0.0722])
P = json.loads((Path(__file__).resolve().parent / 'protocol' / 'wl104a.json').read_text(encoding='utf-8'))
TAU_E, TAU_S = P['thresholds']['tau_E'], P['thresholds']['tau_S']
PRIMARY = P['rois']['primary']
ROIS = PRIMARY + P['rois']['secondary']


def load(d, keys):
    files = sorted(Path(d).glob('r*.npz'))
    out = {k: [] for k in keys}
    for f in files:
        z = np.load(f)
        for k in keys:
            out[k].append(z[k].astype(np.float64))
    return {k: np.stack(v) for k, v in out.items()}        # [R, H, W, 3]


def compare(X, Y, Ytot, Yhigh, m, XB=None):
    """X, Y: [R, H, W, 3] HIGH fields (X may be a table-A readout with XB its independent table-B readout); Ytot / Yhigh: reference
    TOTAL / HIGH [R, H, W, 3]. Returns energy bias (relative to TOTAL and to HIGH), stable relative RMS, descriptive rel L1."""
    R = X.shape[0]
    lum = lambda a: (a[:, m] @ LUMA).sum(1)                   # [R]
    Xm = X if XB is None else 0.5 * (X + XB)
    num = lum(Xm) - lum(Y)
    eT, eH = S.ratio_jk(num, lum(Ytot)), S.ratio_jk(num, lum(Yhigh))
    if XB is not None:                                        # fixed-table noise: half the A / B energy difference as an extra SE term
        dA = (lum(X) - lum(Y)).sum() / lum(Ytot).sum(); dB = (lum(XB) - lum(Y)).sum() / lum(Ytot).sum()
        extra = abs(dA - dB) / 2
        for e in (eT,):
            se = float(np.hypot(e['se'], extra)); h = S.t975(R - 1) * se
            e.update(se=se, lo=e['point'] - h, hi=e['point'] + h, table_noise_se=extra)
    A = (X - Y)[:, m].reshape(R, -1)
    Bf = None if XB is None else (XB - Y)[:, m].reshape(R, -1)
    ms = S.stable_ms_jk(A, Bf)
    norm = float((Ytot.mean(0)[m] ** 2).mean())
    rms = S.rel_rms(ms, norm)
    Dm = (Xm - Y).mean(0)[m]; h1 = (Xm - Y)[:R // 2].mean(0)[m]; h2 = (Xm - Y)[R // 2:].mean(0)[m]
    l1 = {'rel_l1_vs_ref_high': float(np.abs(Dm).sum() / Yhigh.mean(0)[m].sum()),
          'rel_l1_vs_ref_total': float(np.abs(Dm).sum() / Ytot.mean(0)[m].sum()),
          'noise_floor_vs_ref_high': float(0.5 * np.abs(h1 - h2).sum() / Yhigh.mean(0)[m].sum())}
    return {'energy_bias_vs_total': eT, 'energy_bias_vs_high': eH, 'rel_rms_vs_total': rms, 'descriptive': l1}


def significance(c):
    lo, hi = S.abs_bounds(c['energy_bias_vs_total'])
    r = c['rel_rms_vs_total']
    return {'energy_sig': lo > TAU_E, 'energy_below': hi < TAU_E, 'rms_sig': r['lo'] > TAU_S, 'rms_below': r['hi'] < TAU_S,
            'energy_halfwidth': (c['energy_bias_vs_total']['hi'] - c['energy_bias_vs_total']['lo']) / 2,
            'rms_halfwidth': (r['hi'] - r['lo']) / 2}
