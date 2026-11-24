"""WL112 frozen metric definitions (numpy only)."""
import math
import numpy as np

LUMW = np.array([0.2126, 0.7152, 0.0722])
T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306}


def lum(x):
    return x @ LUMW


def region_mean_lum(img, mask):
    return float(lum(img[mask]).mean()) if mask.any() else float('nan')


def endpoint_stats(res_by_rep, mask_near, mask_flank):
    """res_by_rep: list over table replicas of [res_stream0, res_stream1] (RGB residual images, readout - physical).
    Returns near / contrast estimates, SE_stream, SE_table, 95% half-widths (frozen formula)."""
    R = len(res_by_rep)
    near = np.array([[region_mean_lum(r, mask_near) for r in pair] for pair in res_by_rep])        # [R, 2]
    flank = np.array([[region_mean_lum(r, mask_flank) for r in pair] for pair in res_by_rep])
    out = {}
    for name, val in (('near', near), ('contrast', near - flank)):
        est = float(val.mean())
        se_tab = float(val.mean(1).std(ddof=1) / math.sqrt(R)) if R > 1 else 0.0
        out[name] = {'estimate': est, 'se_table': se_tab}
    # stream SE from paired pixel differences (systematic part cancels between the two streams)
    sn = []; sf = []
    for pair in res_by_rep:
        dn = lum(pair[0][mask_near]) - lum(pair[1][mask_near]); df = lum(pair[0][mask_flank]) - lum(pair[1][mask_flank])
        sn.append(dn.var(ddof=1) / (4 * dn.size)); sf.append(df.var(ddof=1) / (4 * df.size))
    se_near = math.sqrt(np.mean(sn) / R); se_flank = math.sqrt(np.mean(sf) / R)
    out['near']['se_stream'] = se_near
    out['contrast']['se_stream'] = math.sqrt(se_near ** 2 + se_flank ** 2)
    t = T975.get(R - 1, 1.96) if R > 1 else 0.0
    for name in ('near', 'contrast'):
        o = out[name]
        o['half_width'] = math.sqrt((1.96 * o['se_stream']) ** 2 + (t * o['se_table']) ** 2)
        o['lo'] = o['estimate'] - o['half_width']; o['hi'] = o['estimate'] + o['half_width']
    return out


def rmse_debiased(res_by_rep, h_pair, mask):
    """Noise-debiased RMSE: cross product of residuals of the two independent streams with distinct table replicas
    (pairs (a, a+1 mod R)); relative to the debiased reference RMS sqrt(mean h0 h1). Returns (relative, absolute, signed bias RGB)."""
    R = len(res_by_rep)
    if not mask.any():
        return float('nan'), float('nan'), [float('nan')] * 3
    pairs = [(a, (a + 1) % R) for a in range(R)] if R > 1 else [(0, 0)]
    ms = np.mean([np.mean(res_by_rep[a][0][mask] * res_by_rep[b][1][mask]) for a, b in pairs])
    ref = np.mean(h_pair[0][mask] * h_pair[1][mask])
    absr = math.copysign(math.sqrt(abs(ms)), ms)
    rel = absr / math.sqrt(max(ref, 1e-30))
    bias = np.mean([0.5 * (res_by_rep[a][0][mask] + res_by_rep[a][1][mask]).mean(0) for a in range(R)], 0)
    return float(rel), float(absr), bias.tolist()


def band_profile(res_by_rep, bands):
    out = []
    for m in bands:
        out.append(float(np.mean([region_mean_lum(0.5 * (p[0] + p[1]), m) for p in res_by_rep])) if m.any() else float('nan'))
    return out


def coverage(Wlum, native):
    """Native (non-fallback) share of the luminance query weight. Wlum [ncs] over sub-cells; native [N,4,NB] or [N,NB]."""
    tot = Wlum.sum()
    if tot <= 0:
        return float('nan')
    nat = native
    if nat.ndim == 2:                                  # coarse native broadcast over sub-cells
        nat = np.repeat(nat[:, None, :], 4, axis=1)
    return float((Wlum * nat.ravel()).sum() / tot)
