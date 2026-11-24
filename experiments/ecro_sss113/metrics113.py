"""WL113 frozen metrics: WL112 historical quantities (metrics112) and the approved 3 % SSS gate quantities (SSS_NEAR_CONTACT_GATE.md
§3.1, conditionally approved 2026-11-24) for a complete method frame against a reference (full depth or depth-matched)."""
import math, sys
from pathlib import Path
import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'ecro_sss112'))
import metrics112 as M2  # noqa: E402
import common113 as C  # noqa: E402

LUMW = M2.LUMW
lum = M2.lum
endpoint_stats, rmse_debiased, band_profile, pixel_kappa, region_mean_lum = M2.endpoint_stats, M2.rmse_debiased, M2.band_profile, M2.pixel_kappa, M2.region_mean_lum
TAU = 0.03
RINGS = [0, 0.10, 0.14, 0.18, 0.22, 0.26, 0.30, 0.35, 0.40, 0.50]


def coverage16(Wlum, native16):
    tot = Wlum.sum()
    return float((Wlum * native16.ravel()).sum() / tot) if tot > 0 else float('nan')


def gate_masks(gb, geom, light_pos):
    reg, bands = C.masks(gb, geom, light_pos)
    pos = gb['p']; ctr = np.asarray(geom['S']['c'])
    r = np.linalg.norm(pos[..., :2] - ctr[:2], axis=-1)
    floor, sphere, valid = reg['floor'], reg['sphere'], gb['ok']
    sil = (gb['coverage'] > 0) & (gb['coverage'] < 1)
    disk = floor & (r < 0.25)
    g = {'near': reg['near'], 'flank': reg['flank'], 'disk': disk, 'silhouette': sil,
         'off_target': valid & ~disk & ~sil & ~sphere,
         'sphere_interior': reg['sphere_interior'], 'contact_shadow_out': reg['contact_shadow'] & ~disk,
         'wall': valid & np.isin(gb['sid'], [1, 2, 3, 4]),
         'win_floor': floor & (r < 0.50), 'win_sphere_low': sphere & (pos[..., 2] < ctr[2])}
    g['rings'] = [floor & (r >= a) & (r < b) for a, b in zip(RINGS, RINGS[1:])]
    return g, reg, bands


def rel_bias(Ms, Rs, mask, kappa):
    """Regional relative bias b = (mean Y_M - mean Y_R) / mean Y_R over two paired streams; 95 % half-width from the paired
    two-stream pixel difference (x kappa) of the residual and of the reference."""
    if not mask.any():
        return {'b': float('nan'), 'hw': float('nan')}
    yM = [lum(m[mask]) for m in Ms]; yR = [lum(r[mask]) for r in Rs]
    mR = 0.5 * (yR[0].mean() + yR[1].mean())
    d = [yM[j] - yR[j] for j in range(2)]
    b = 0.5 * (d[0].mean() + d[1].mean()) / mR
    se_res = math.sqrt(kappa * (d[0] - d[1]).var(ddof=1) / (4 * d[0].size)) / mR if d[0].size > 1 else 0.
    se_ref = math.sqrt(kappa * (yR[0] - yR[1]).var(ddof=1) / (4 * yR[0].size)) / mR * abs(b) if yR[0].size > 1 else 0.
    hw = 1.96 * math.sqrt(se_res ** 2 + se_ref ** 2)
    return {'b': float(b), 'hw': float(hw), 'lo': float(b - hw), 'hi': float(b + hw)}


def contrast(Ms, Rs, near, flank, kappa):
    mean = lambda imgs, m: [lum(x[m]).mean() for x in imgs]
    mn, mf, rn, rf = mean(Ms, near), mean(Ms, flank), mean(Rs, near), mean(Rs, flank)
    rnear = 0.5 * (rn[0] + rn[1])
    c = (0.5 * (mn[0] + mn[1] - mf[0] - mf[1]) - 0.5 * (rn[0] + rn[1] - rf[0] - rf[1])) / rnear
    dn = [lum(Ms[j][near]) - lum(Rs[j][near]) for j in range(2)]; df = [lum(Ms[j][flank]) - lum(Rs[j][flank]) for j in range(2)]
    se = math.sqrt(kappa * ((dn[0] - dn[1]).var(ddof=1) / (4 * dn[0].size) + (df[0] - df[1]).var(ddof=1) / (4 * df[0].size))) / rnear
    hw = 1.96 * se
    return {'c': float(c), 'hw': float(hw), 'lo': float(c - hw), 'hi': float(c + hw)}


def rel_rmse(Ms, Rs, mask):
    """Relative debiased HDR RMSE of RGB over the ROI (cross product of the two independent streams)."""
    if not mask.any():
        return float('nan')
    ms = np.mean((Ms[0][mask] - Rs[0][mask]) * (Ms[1][mask] - Rs[1][mask]))
    ref = np.mean(Rs[0][mask] * Rs[1][mask])
    return float(math.copysign(math.sqrt(abs(ms)), ms) / math.sqrt(max(ref, 1e-30)))


def windows(Ms, Rs, centres, kappa, size=5):
    """R1L: 5 x 5 windows centred on every pixel of `centres`; detection test with Bonferroni over the windows."""
    h = size // 2
    yM = [lum(m) for m in Ms]; yR = [lum(r) for r in Rs]
    d0, d1 = yM[0] - yR[0], yM[1] - yR[1]
    from numpy.lib.stride_tricks import sliding_window_view as swv
    pad = lambda a: np.pad(a, h, mode='edge')
    md = swv(pad(0.5 * (d0 + d1)), (size, size)).mean((-1, -2))
    mr = swv(pad(0.5 * (yR[0] + yR[1])), (size, size)).mean((-1, -2))
    vd = swv(pad(d0 - d1), (size, size)).var((-1, -2), ddof=1)
    e = md / np.maximum(mr, 1e-9)
    se = np.sqrt(kappa * vd / (4 * size * size)) / np.maximum(mr, 1e-9)
    ys, xs = np.nonzero(centres)
    nw = len(ys)
    if nw == 0:
        return {'windows': 0, 'failing': 0}
    z = float(stats.norm.ppf(1 - 0.05 / (2 * nw)))
    lower = np.abs(e[ys, xs]) - z * se[ys, xs]
    fail = lower > TAU
    worst = int(np.argmax(np.abs(e[ys, xs])))
    return {'windows': nw, 'z_bonferroni': z, 'failing': int(fail.sum()), 'max_abs_e': float(np.abs(e[ys, xs]).max()),
            'worst_pixel': [int(ys[worst]), int(xs[worst])], 'worst_e': float(e[ys[worst], xs[worst]])}


def gate3(Ms, Rs, g, kappa, ctrl=None):
    """Approved 3 % quantities of one view. Ms / Rs: two paired streams of the complete method frame / reference.
    ctrl: optional (Ms_ctrl) for the RMSE non-regression parts (1.10 x)."""
    o = {'R1_near': rel_bias(Ms, Rs, g['near'], kappa), 'R2_contrast': contrast(Ms, Rs, g['near'], g['flank'], kappa) if g['near'].any() and g['flank'].any() else None,
         'R3_disk_bias': rel_bias(Ms, Rs, g['disk'], kappa), 'R4_rings': [rel_bias(Ms, Rs, m, kappa) for m in g['rings']],
         'R5a_sil_bias': rel_bias(Ms, Rs, g['silhouette'], kappa), 'R5a_sil_rmse': rel_rmse(Ms, Rs, g['silhouette']),
         'R5b_off_bias': rel_bias(Ms, Rs, g['off_target'], kappa), 'R5b_off_rmse': rel_rmse(Ms, Rs, g['off_target']),
         'R5b_regions': {k: rel_bias(Ms, Rs, g[k], kappa) for k in ('sphere_interior', 'contact_shadow_out', 'wall')},
         'R1L': windows(Ms, Rs, g['win_floor'] | g['win_sphere_low'] | g['silhouette'], kappa)}
    inside = lambda x: x is not None and not math.isnan(x.get('b', x.get('c', float('nan')))) and max(abs(x['lo']), abs(x['hi'])) <= TAU
    rings = [x for x in o['R4_rings'] if not math.isnan(x['b'])]
    signs = [np.sign(x['b']) if abs(x['b']) > TAU / 2 else 0 for x in rings]
    run = max((len(list(gp)) for s, gp in __import__('itertools').groupby(signs) if s != 0), default=0)
    o['pass'] = {'R1': inside(o['R1_near']), 'R1L': o['R1L']['failing'] == 0, 'R2': inside(o['R2_contrast']), 'R3_bias': inside(o['R3_disk_bias']),
                 'R4': bool(all(abs(x['b']) <= TAU for x in rings) and run < 3), 'R5a_bias': inside(o['R5a_sil_bias']),
                 'R5b_bias': bool(inside(o['R5b_off_bias']) and all(inside(v) or math.isnan(v['b']) for v in o['R5b_regions'].values()))}
    if ctrl is not None:
        o['ctrl_sil_rmse'] = rel_rmse(ctrl, Rs, g['silhouette']); o['ctrl_off_rmse'] = rel_rmse(ctrl, Rs, g['off_target'])
        o['pass']['R5a_rmse'] = bool(o['R5a_sil_rmse'] <= 1.10 * o['ctrl_sil_rmse'] + 1e-9)
        o['pass']['R5b_rmse'] = bool(o['R5b_off_rmse'] <= 1.10 * o['ctrl_off_rmse'] + 1e-9)
    return o
