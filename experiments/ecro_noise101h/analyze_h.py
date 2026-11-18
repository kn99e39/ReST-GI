"""WL101H offline analysis: G5 (variance identity), G7 (split equivalence), moments, stage decomposition, bias, budgets,
verdicts, reviewer figures. Reads only the raw outputs of run_h.py and the two archived physical halves.
  <venv-cu128 python> analyze_h.py <raw_root> <reviewer_dir>
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy import stats as sst

import common_h as CH
import stats_h as S

P = CH.P
RAW = Path(sys.argv[1]).resolve()
OUT = Path(sys.argv[2]).resolve()
OUT.mkdir(parents=True, exist_ok=True)
AN = RAW / 'analysis'
AN.mkdir(exist_ok=True)
dev = torch.device('cuda')
RES = 512
NB = 2000
SEED = 10189999
SPPS = [1, 4, 16]
ROIS = list(P['rois']['expected_counts'])
PRIMARY = P['rois']['primary']
ZF = S.z_two_sided(0.05 / 36)
masks = dict(np.load(RAW / 'masks.npz'))
IDX = {k: np.flatnonzero(masks[k].ravel()) for k in ROIS}
t_start = time.time()


def ref_half(h):
    with np.load(CH.REND100 / f'REF_0_{h}.npz') as z:
        e1, e2, hh = (z[k].astype(np.float64).reshape(-1, 3).T for k in ('e1', 'e2', 'h'))   # [3, P]
    return {'L': e1 + e2, 'H': hh, 'T': e1 + e2 + hh, 'E1': e1, 'E2': e2}


R1, R2 = ref_half(0), ref_half(1)
RM = {k: 0.5 * (R1[k] + R2[k]) for k in R1}
DEN = {k: float((RM['T'][:, IDX[k]] ** 2).mean()) for k in ROIS}          # mean squared reference full radiance per ROI


def roi_vec(arr, k, comp, spp):
    """arr [R, 9, P] float32 sums -> torch [R, 3 n] float64 per-pixel means of component comp (0 e1, 1 e2, 2 hl)."""
    x = torch.as_tensor(np.ascontiguousarray(arr[:, 3 * comp:3 * comp + 3][:, :, IDX[k]]), device=dev).double() / spp
    return x.reshape(x.shape[0], -1)


def derived(G, den):
    VE1, VE2, VH, VT = G[..., 0, 0], G[..., 1, 1], G[..., 2, 2], G[..., 3, 3]
    CE1E2, CE1H, CE2H = G[..., 0, 1], G[..., 0, 2], G[..., 1, 2]
    VL = VE1 + VE2 + 2 * CE1E2
    CLH = CE1H + CE2H
    return {'V_E1': VE1, 'V_E2': VE2, 'V_L': VL, 'V_H': VH, 'V_T': VT, 'C_E1E2': CE1E2, 'C_E1H': CE1H, 'C_E2H': CE2H, 'C_LH': CLH,
            'two_C_LH': 2 * CLH, 'identity_residual': VT - (VL + VH + 2 * CLH),
            's_L': VL / VT, 's_H': VH / VT, 's_C': 2 * CLH / VT, 'rho_LH': CLH / torch.sqrt(VL * VH),
            'rel_V_L': VL / den, 'rel_V_H': VH / den, 'rel_V_T': VT / den, 'rel_two_C_LH': 2 * CLH / den}


def pack(dp, dw):
    return {k: S.ci(dw[k].cpu().numpy(), float(dp[k])) for k in dp}


def mag_lo(c):
    return c['lo'] if c['lo'] > 0 else (-c['hi'] if c['hi'] < 0 else 0.0)


def W_for(arm, spp, R):
    return S.boot_weights(R, NB, [SEED, arm, spp], dev)


# -------------------------------------------------------------------------------------------------------- J arm
J, JREP, JMAP = {}, {}, {}
for spp in SPPS:
    ch = np.load(RAW / 'joint' / f'spp{spp}_ch.npy')
    co = np.load(RAW / 'joint' / f'spp{spp}_comp.npy')
    R_ = ch.shape[0]
    W = W_for(1, spp, R_)
    J[spp], JREP[spp] = {}, {}
    for k in ROIS:
        v = [roi_vec(ch, k, c, spp) for c in range(3)]
        t = torch.as_tensor(np.ascontiguousarray(co[:, :, IDX[k]]), device=dev).double().reshape(R_, -1)
        vecs = v + [t]
        dp = derived(S.gram(vecs), DEN[k]); dw = derived(S.gram(vecs, W), DEN[k])
        means = {n: x.mean(1).cpu().numpy() for n, x in (('E1', v[0]), ('E2', v[1]), ('L', v[0] + v[1]), ('H', v[2]), ('T', t))}
        J[spp][k] = {'moments': pack(dp, dw), 'mean': {n: float(m.mean()) for n, m in means.items()},
                     'ref_mean': {n: float(RM[n][:, IDX[k]].mean()) for n in ('E1', 'E2', 'L', 'H', 'T')}}
        JREP[spp][k] = means
        del v, t, vecs
    # per-pixel maps (RGB-mean over channels) on the full frame
    x = torch.as_tensor(ch, device=dev).double().view(R_, 3, 3, -1) / spp
    Lx, Hx = x[:, 0] + x[:, 1], x[:, 2]
    Tx = torch.as_tensor(co, device=dev).double()
    var = lambda a: a.var(0, unbiased=True).mean(0)
    cov = lambda a, b: ((a - a.mean(0)) * (b - b.mean(0))).sum(0).div(R_ - 1).mean(0)
    JMAP[spp] = {'V_L': var(Lx), 'V_H': var(Hx), 'V_T': var(Tx), 'C_LH': cov(Lx, Hx), 'rep0_L': Lx[0], 'rep0_H': Hx[0], 'rep0_T': Tx[0]}
    JMAP[spp] = {a: b.cpu().numpy() for a, b in JMAP[spp].items()}
    del ch, co, x, Lx, Hx, Tx
    torch.cuda.empty_cache()
    print('J', spp, time.time() - t_start, flush=True)

# G5 variance identity (V_T from the contract-composed image)
g5 = max(abs(J[s][k]['moments']['identity_residual']['point']) / J[s][k]['moments']['V_T']['point'] for s in SPPS for k in ROIS)
G5 = {'max_abs_identity_residual_over_VT': g5, 'pass': g5 <= 1e-5}

# -------------------------------------------------------------------------------------------------------- PF arm
PF, PFREP, PFMAP = {}, {}, {}
for spp in SPPS:
    arr = np.load(RAW / 'pf' / f'spp{spp}.npy')                  # [R, 3, 9, P]
    R_ = arr.shape[0]
    W = W_for(2, spp, R_)
    PF[spp], PFREP[spp] = {}, {}
    for k in ROIS:
        trip = []
        for j in range(3):
            v = [roi_vec(arr[:, j], k, c, spp) for c in range(3)]
            trip.append(v + [v[0] + v[1] + v[2]])
        Dp = S.pf_decompose(*trip); Dw = S.pf_decompose(*trip, W=W)
        res = {st: pack(derived(Dp[st], DEN[k]), derived(Dw[st], DEN[k])) for st in ('tot', 'J', 'P', 'S')}
        shares = {}
        for comp, key in (('T', 'V_T'), ('L', 'V_L'), ('H', 'V_H'), ('C', 'C_LH')):
            for st in ('J', 'P', 'S'):
                pp = float(derived(Dp[st], DEN[k])[key] / derived(Dp['tot'], DEN[k])[key])
                ww = (derived(Dw[st], DEN[k])[key] / derived(Dw['tot'], DEN[k])[key]).cpu().numpy()
                shares[f'{comp}_{st}'] = S.ci(ww, pp)
        res['stage_share'] = shares
        A = trip[0]
        PFREP[spp][k] = {'T': A[3].mean(1).cpu().numpy(), 'L': (A[0] + A[1]).mean(1).cpu().numpy(), 'H': A[2].mean(1).cpu().numpy()}
        PF[spp][k] = res
        del trip, A
    x = torch.as_tensor(arr, device=dev).double().view(R_, 3, 3, 3, -1) / spp      # r, member, comp, rgb, P
    T_ = x[:, :, 0] + x[:, :, 1] + x[:, :, 2]
    vs = (0.5 * (T_[:, 0] - T_[:, 1]) ** 2).mean(0).mean(0)
    vps = (0.5 * (T_[:, 0] - T_[:, 2]) ** 2).mean(0).mean(0)
    vt = T_[:, 0].var(0, unbiased=True).mean(0)
    PFMAP[spp] = {'T_J': (vt - vps).cpu().numpy(), 'T_P': (vps - vs).cpu().numpy(), 'T_S': vs.cpu().numpy(), 'T_tot': vt.cpu().numpy()}
    del arr, x, T_
    torch.cuda.empty_cache()
    print('PF', spp, time.time() - t_start, flush=True)

# G7: split (member A) vs normal arm, primary ROIs x spp x (T, L, H)
g7 = []
for spp in SPPS:
    for k in PRIMARY:
        for comp, key in (('T', 'V_T'), ('L', 'V_L'), ('H', 'V_H')):
            a, b = PF[spp][k]['tot'][key], J[spp][k]['moments'][key]
            zv = abs(np.log(a['point'] / b['point'])) / np.hypot(a['se'] / a['point'], b['se'] / b['point'])
            xa, xb = PFREP[spp][k][comp], JREP[spp][k][comp]
            zm = abs(xa.mean() - xb.mean()) / np.sqrt(xa.var(ddof=1) / len(xa) + xb.var(ddof=1) / len(xb))
            g7.append({'spp': spp, 'roi': k, 'comp': comp, 'var_ratio': a['point'] / b['point'], 'z_var': float(zv), 'z_mean': float(zm),
                       'pass': bool(zv <= ZF and zm <= ZF)})
G7 = {'z_critical': ZF, 'tests': g7, 'pass': all(t['pass'] for t in g7)}

# -------------------------------------------------------------------------------------------------------- HQ bias
seeds = P['arms']['HQ_normal']['seeds']
MANI = json.loads((RAW / 'manifest.json').read_text(encoding='utf-8'))
HQ_SPP, Q_SPP = MANI['config']['hq_spp'], MANI['config']['q_spp']
hq = np.stack([np.load(RAW / 'hq' / f'hq_{s}.npy') for s in seeds]) / HQ_SPP      # [8, 9, P]
HQC = {'L': hq[:, 0:3] + hq[:, 3:6], 'H': hq[:, 6:9]}
HQC['T'] = HQC['L'] + HQC['H']
G1m = {c: HQC[c][:4].mean(0) for c in 'LHT'}
G2m = {c: HQC[c][4:].mean(0) for c in 'LHT'}
HQM = {c: HQC[c].mean(0) for c in 'LHT'}
yy, xx = np.divmod(np.arange(RES * RES), RES)
BLOCK = (yy // 16) * (RES // 16) + (xx // 16)
LUMA = CH.LUMA
BIAS = {}
for ri, k in enumerate(ROIS):
    ix = IDX[k]
    st = lambda d: np.stack([d[c][:, ix].ravel() for c in 'LHT'])          # [3, 3n] channel-major then pixel
    g1, g2, r1, r2 = st(G1m), st(G2m), st(R1), st(R2)
    bs = S.stable_bias_sq(*(torch.as_tensor(a) for a in (g1, g2, r1, r2))).numpy()
    d1, d2, e1, e2 = g1 - r1, g2 - r2, g1 - r2, g2 - r1
    blk = np.tile(BLOCK[ix], 3)
    ub, inv = np.unique(blk, return_inverse=True)
    nb = len(ub)
    cnt = np.bincount(inv, minlength=nb).astype(np.float64)
    pairs = {}
    for a in range(3):
        for b in range(a, 3):
            prod = 0.25 * (d1[a] * d2[b] + e1[a] * e2[b] + d1[b] * d2[a] + e1[b] * e2[a])
            pairs[(a, b)] = np.bincount(inv, weights=prod, minlength=nb)
    g = np.random.default_rng([SEED, 9, ri])
    draws = g.integers(0, nb, size=(NB, nb))
    cw = np.zeros((NB, nb)); np.add.at(cw, (np.repeat(np.arange(NB), nb), draws.ravel()), 1.0)
    denom = cw @ cnt
    bdraw = {ab: (cw @ s) / denom for ab, s in pairs.items()}
    names = 'LHT'
    out = {}
    for (a, b), dd in bdraw.items():
        out[f'b2_{names[a]}{names[b]}'] = S.ci(dd, bs[a, b])
    # relative stable RMS against each component's own reference RMS (WL101C convention), and against full-ref RMS
    for c, a in zip('LHT', range(3)):
        own = float((RM[c][:, ix] ** 2).mean())
        out[f'rel_b2_own_{c}'] = bs[a, a] / own
        out[f'stable_rel_rms_own_{c}'] = float(np.sign(bs[a, a]) * np.sqrt(abs(bs[a, a]) / own))
    # signed ROI-mean bias, Welch (8 HQ seeds vs 2 halves), RGB-mean and luminance
    signed = {}
    for c in 'LHT':
        for name, wts in (('rgb', np.ones(3) / 3), ('luma', LUMA)):
            hs = np.array([wts @ HQC[c][s][:, ix].mean(1) for s in range(8)])
            rs = np.array([wts @ R[c][:, ix].mean(1) for R in (R1, R2)])
            diff = hs.mean() - rs.mean()
            va, vb = hs.var(ddof=1) / 8, rs.var(ddof=1) / 2
            se = np.sqrt(va + vb)
            dfw = (va + vb) ** 2 / (va ** 2 / 7 + vb ** 2 / 1)
            q = sst.t.ppf(0.975, dfw)
            signed[f'{c}_{name}'] = {'diff': float(diff), 'lo': float(diff - q * se), 'hi': float(diff + q * se), 'df': float(dfw),
                                     'ref_mean': float(rs.mean()), 'rel': float(diff / rs.mean())}
    out['signed_bias'] = signed
    out['ref_mean_noise'] = {c: float(((R1[c][:, ix] - R2[c][:, ix]) ** 2).mean() / 4) for c in 'LHT'}
    out['hq_mean_noise'] = {c: float(HQC[c][:, :, ix].var(0, ddof=1).mean() / 8) for c in 'LHT'}
    out['hq_single_var_T'] = float(HQC['T'][:, :, ix].var(0, ddof=1).mean())
    out['blocks'] = int(nb)
    BIAS[k] = out
BMAP = {c: (HQM[c] - RM[c]).mean(0) for c in 'LHT'}                          # RGB-mean signed bias maps
BMAP['ref_noise'] = np.sqrt(((R1['T'] - R2['T']) ** 2).mean(0) / 4)

# -------------------------------------------------------------------------------------------------------- budgets and verdicts
BUD, VERD = {}, {}
for spp in SPPS:
    BUD[spp], VERD[spp] = {}, {}
    Wv = None
    for k in ROIS:
        m = J[spp][k]['moments']
        vt = m['V_T']; b2 = BIAS[k]['b2_TT']
        rng = np.random.default_rng([SEED, 11, spp, ROIS.index(k)])
        vt_d = rng.normal(vt['point'], vt['se'], 20000); b2_d = rng.normal(b2['point'], b2['se'], 20000)
        sh_p = b2['point'] / (vt['point'] + b2['point'])
        sh = S.ci(b2_d / (vt_d + b2_d), sh_p)
        mse = vt['point'] + b2['point']
        BUD[spp][k] = {'MSE_T': mse, 'rel_MSE_T': mse / DEN[k], 'rel_rmse_T': float(np.sqrt(max(mse, 0) / DEN[k])),
                       'share_V_L': m['V_L']['point'] / mse, 'share_V_H': m['V_H']['point'] / mse, 'share_two_C_LH': m['two_C_LH']['point'] / mse,
                       'share_b2_T': sh, 'share_b2_L': BIAS[k]['b2_LL']['point'] / mse,
                       'share_b2_H': BIAS[k]['b2_HH']['point'] / mse, 'share_two_b_LH': 2 * BIAS[k]['b2_LH']['point'] / mse}
        sL, sH, sC = m['s_L'], m['s_H'], m['s_C']
        if sh['lo'] > 0.5:
            cls = 'BIAS-DOMINANT'
        elif sL['lo'] > 0.5:
            cls = 'LOW-DOMINANT'
        elif sH['lo'] > 0.5:
            cls = 'HIGH-DOMINANT'
        elif sC['lo'] > 0.5 or sC['hi'] < -0.5:
            cls = 'COVARIANCE-DOMINATED'
        elif sum(x > 0.2 for x in (sL['lo'], sH['lo'], mag_lo(sC))) >= 2:
            cls = 'MIXED'
        else:
            cls = 'INCONCLUSIVE'
        flags = []
        if b2['lo'] > 0:
            flags.append('bias resolved')
        if sC['lo'] > 0 or sC['hi'] < 0:
            flags.append('covariance resolved')
        stage = None
        if G7['pass']:
            ss = PF[spp][k]['stage_share']
            stv = {st: ss[f'T_{st}'] for st in ('J', 'P', 'S')}
            dom = [st for st, c in stv.items() if c['lo'] > 0.5]
            name = {'J': 'JITTER', 'P': 'PREFIX', 'S': 'SUFFIX'}
            if dom:
                stage = name[dom[0]] + '-DOMINANT'
            elif sum(c['lo'] > 0.2 for c in stv.values()) >= 2:
                stage = 'MIXED-STAGE'
            else:
                stage = 'INCONCLUSIVE'
        VERD[spp][k] = {'class': cls, 'flags': flags, 'stage_class': stage if G7['pass'] else 'INVALID (G7 failed)'}
    prim = {VERD[spp][k]['class'] for k in PRIMARY}
    VERD[spp]['overall'] = 'REGION-DEPENDENT' if len(prim) > 1 else prim.pop()

# spp scaling (log-log slope per component, draws independent per spp)
SCAL = {}
for k in ROIS:
    SCAL[k] = {}
    for key in ('V_L', 'V_H', 'V_T', 'two_C_LH'):
        pts = [J[s][k]['moments'][key]['point'] for s in SPPS]
        nv = [p * s for p, s in zip(pts, SPPS)]
        if all(p > 0 for p in pts):
            rng = np.random.default_rng([SEED, 12, ROIS.index(k)])
            dr_ = np.stack([np.abs(rng.normal(J[s][k]['moments'][key]['point'], J[s][k]['moments'][key]['se'], 4000)) for s in SPPS], 1)
            sl = [S.log_slope(SPPS, d) for d in dr_]
            SCAL[k][key] = {'slope': S.ci(sl, S.log_slope(SPPS, pts)), 'n_times_V': nv}
        else:
            SCAL[k][key] = {'slope': None, 'n_times_V': nv, 'note': 'sign change across spp; slope undefined'}

# -------------------------------------------------------------------------------------------------------- Q diagnostic
man = json.loads((RAW / 'manifest.json').read_text(encoding='utf-8'))
base = np.load(RAW / 'quad' / 'base.npy')[6:9] / Q_SPP
Q = {'tables': man['stages']['quad'], 'roi': {}}
for k in ROIS:
    ix = IDX[k]
    refH = RM['H'][:, ix]
    q = {}
    for kind, key in (('source', 'source_offset_redraws'), ('state', 'state_offset_redraws')):
        rows = []
        for sd in P['arms']['Q_quadrature'][key]['rng_seeds']:
            d = (np.load(RAW / 'quad' / f'{kind}_{sd}_hl.npy') / Q_SPP - base)[:, ix]
            rows.append({'seed': sd, 'rel_mean_luma': float((LUMA @ d).mean() / (LUMA @ refH).mean()),
                         'rel_rms_own_H': float(np.sqrt((d ** 2).mean() / (refH ** 2).mean()))})
        b2h = BIAS[k]['b2_HH']['point']
        q[kind] = {'draws': rows, 'rms_over_draws_rel_rms': float(np.sqrt(np.mean([r['rel_rms_own_H'] ** 2 for r in rows]))),
                   'stable_bias_rel_rms_own_H': BIAS[k]['stable_rel_rms_own_H']}
    Q['roi'][k] = q

# -------------------------------------------------------------------------------------------------------- metrics
def clean(o):
    if isinstance(o, dict):
        return {str(a): clean(b) for a, b in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, float) and not np.isfinite(o):
        return None
    if isinstance(o, torch.Tensor):
        return o.item()
    if isinstance(o, np.ndarray):
        return clean(o.tolist())
    return o


metrics = {'id': P['id'], 'commit_run': man.get('commit'), 'gates_run': man['gates'], 'G5': G5, 'G7': G7,
           'roi_counts': {k: int(masks[k].sum()) for k in ROIS}, 'roi_mean_sq_ref_T': DEN,
           'joint': J, 'stage': PF, 'bias': BIAS, 'budget': BUD, 'verdict': VERD, 'spp_scaling': SCAL, 'quadrature': Q,
           'timing': man['stages']['timing'], 'versions': man['versions'], 'bootstrap': {'draws': NB, 'seed': SEED, 'interval': 'normal, bootstrap SE'}}
CH.write_json(AN / 'metrics.json', clean(metrics))
CH.write_json(OUT / 'metrics.json', clean(metrics))
np.savez_compressed(AN / 'maps.npz', **{f'J{s}_{a}': b for s in SPPS for a, b in JMAP[s].items()},
                    **{f'PF{s}_{a}': b for s in SPPS for a, b in PFMAP[s].items()}, **{f'bias_{a}': b for a, b in BMAP.items()})
print('metrics written', time.time() - t_start, flush=True)

# -------------------------------------------------------------------------------------------------------- figures
import matplotlib  # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

plt.rcParams.update({'font.size': 9, 'axes.edgecolor': '#52514e', 'axes.labelcolor': '#0b0b0b', 'xtick.color': '#52514e',
                     'ytick.color': '#52514e', 'axes.grid': False, 'figure.facecolor': '#fcfcfb', 'axes.facecolor': '#fcfcfb'})
CAT = {'L': '#2a78d6', 'H': '#eb6834', 'C': '#1baf7a', 'B': '#eda100', 'T': '#52514e'}
SEQ = LinearSegmentedColormap.from_list('seq', ['#fcfcfb', '#cde2fb', '#86b6ef', '#3987e5', '#1c5cab', '#0d366b'])
DIV = LinearSegmentedColormap.from_list('div', ['#1c5cab', '#86b6ef', '#f0efec', '#f3a07f', '#c4471a'])
EX = P['display']['exposure']
tone = lambda x: (np.clip(EX * x, 0, None) / (1 + np.clip(EX * x, 0, None))) ** (1 / 2.2)
img = lambda v: v.reshape(3, RES, RES).transpose(1, 2, 0)
mp = lambda v: v.reshape(RES, RES)
SIL = (slice(272, 368), slice(112, 208))          # sphere left limb (rows, cols)
x0, y0, x1, y1 = 168, 377, 344, 457               # WL101C contact crop
CON = (slice(y0, y1), slice(x0, x1))

# family scales (worklog-global maxima, sqrt display): stochastic deviation family and systematic residual family
sroot = lambda v: np.sign(v) * np.sqrt(np.abs(v))
dev_maps = [np.sqrt(np.clip(JMAP[s][a], 0, None)) for s in SPPS for a in ('V_L', 'V_H', 'V_T')] + \
           [np.abs(sroot(2 * JMAP[s]['C_LH'])) for s in SPPS] + [np.abs(sroot(PFMAP[s][a])) for s in SPPS for a in ('T_J', 'T_P', 'T_S', 'T_tot')]
M_DEV = float(max(m.max() for m in dev_maps))
res_maps = [np.abs(BMAP[c]) for c in 'LHT'] + [BMAP['ref_noise']]
M_RES = float(max(m.max() for m in res_maps))
SCALES = {'stochastic_family_max_linear_hdr': M_DEV, 'residual_family_max_linear_hdr': M_RES,
          'mapping': 'display = sign(x) * sqrt(|x| / M); covariance and stage-variance maps shown as signed root sign(v) sqrt(|v|) in radiance units',
          'colorbar_ticks': 'real linear-HDR units'}


def show_pos(ax, v, M, title):
    im = ax.imshow(np.sqrt(np.clip(v, 0, None) / M), cmap=SEQ, vmin=0, vmax=1)
    ax.set_title(title, fontsize=8); ax.set_xticks([]); ax.set_yticks([])
    return im


def show_sig(ax, v, M, title):
    im = ax.imshow(np.sign(v) * np.sqrt(np.abs(v) / M), cmap=DIV, vmin=-1, vmax=1)
    ax.set_title(title, fontsize=8); ax.set_xticks([]); ax.set_yticks([])
    return im


def cbar(fig, im, axes, M, signed, label):
    fr = np.array([0, 0.01, 0.05, 0.25, 1.0])
    ticks = np.concatenate([-np.sqrt(fr[::-1][:-1]), np.sqrt(fr)]) if signed else np.sqrt(fr)
    labs = [f'{(-1 if t < 0 else 1) * t * t * M:.3g}' for t in ticks]
    cb = fig.colorbar(im, ax=axes, shrink=0.8, ticks=ticks); cb.ax.set_yticklabels(labs); cb.set_label(label, fontsize=8)


# F1 raw components
rows = [(f'{s} spp (seed rep 0)', JMAP[s]['rep0_L'], JMAP[s]['rep0_H'], JMAP[s]['rep0_T']) for s in SPPS]
rows.append((f'ECRO 8 x {HQ_SPP} spp mean', HQM['L'], HQM['H'], HQM['T']))
rows.append(('physical ref 2 x 4096 spp', RM['L'], RM['H'], RM['T']))
fig, ax = plt.subplots(5, 3, figsize=(10.5, 17))
for i, (lab, L_, H_, T_) in enumerate(rows):
    for j, (v, n) in enumerate(((L_, 'LOW = E1 + E2'), (H_, 'HIGH = learned HL' if i < 4 else 'HIGH = physical E3-E6'), (T_, 'TOTAL'))):
        ax[i, j].imshow(tone(img(v))); ax[i, j].set_xticks([]); ax[i, j].set_yticks([])
        ax[i, j].set_title(f'{lab} - {n}', fontsize=8)
fig.suptitle(f'WL101H raw linear-HDR components, fixed WL101A exposure {EX:.4f} (display only, no reconstruction)', fontsize=9)
fig.tight_layout(); fig.savefig(OUT / 'raw_components.png', dpi=80); plt.close(fig)

# F2 per-component variance maps
fig, ax = plt.subplots(3, 4, figsize=(16, 11), constrained_layout=True)
for i, s in enumerate(SPPS):
    im = show_pos(ax[i, 0], mp(JMAP[s]['V_L']) ** 0.5, M_DEV, f'{s} spp: std LOW')
    show_pos(ax[i, 1], mp(JMAP[s]['V_H']) ** 0.5, M_DEV, f'{s} spp: std HIGH')
    im2 = show_sig(ax[i, 2], sroot(mp(2 * JMAP[s]['C_LH'])), M_DEV, f'{s} spp: signed root 2Cov(LOW,HIGH)')
    show_pos(ax[i, 3], mp(np.clip(JMAP[s]['V_T'], 0, None)) ** 0.5, M_DEV, f'{s} spp: std TOTAL')
cbar(fig, im, ax[:, [0, 1, 3]].ravel().tolist(), M_DEV, False, 'per-pixel std, linear HDR (RGB-mean variance)')
cbar(fig, im2, ax[:, 2].ravel().tolist(), M_DEV, True, 'sign(2C) sqrt|2C|, linear HDR')
fig.suptitle('Inter-seed variance maps, 64 independent seeds per spp (normal arm). One stochastic-family scale for the whole batch.', fontsize=9)
fig.savefig(OUT / 'variance_maps.png', dpi=75); plt.close(fig)

# F3 stage maps
fig, ax = plt.subplots(3, 4, figsize=(16, 11), constrained_layout=True)
for i, s in enumerate(SPPS):
    for j, (a, n) in enumerate((('T_J', 'jitter stage J'), ('T_P', 'first-event stage P'), ('T_S', 'k=2 walk stage S'), ('T_tot', 'total'))):
        im = show_sig(ax[i, j], sroot(mp(PFMAP[s][a])), M_DEV, f'{s} spp TOTAL variance: {n}')
cbar(fig, im, ax.ravel().tolist(), M_DEV, True, 'sign(v) sqrt|v|, linear HDR')
fig.suptitle('Pick-freeze stage decomposition of TOTAL variance (split diagnostic arm; conditional on G7). Per-pixel J and P are differences of estimates and can be negative.', fontsize=9)
fig.savefig(OUT / 'stage_maps.png', dpi=75); plt.close(fig)

# F4 bias / residual maps
fig, ax = plt.subplots(1, 4, figsize=(18, 4.8), constrained_layout=True)
for j, (a, n) in enumerate((('L', 'LOW bias (ECRO - physical E1+E2)'), ('H', 'HIGH bias (learned - physical E3-E6)'), ('T', 'TOTAL bias'))):
    im = show_sig(ax[j], mp(BMAP[a]), M_RES, n)
im2 = show_pos(ax[3], mp(BMAP['ref_noise']), M_RES, 'physical reference-mean noise (std of 2-half mean)')
cbar(fig, im, ax[:3].tolist(), M_RES, True, 'RGB-mean signed residual, linear HDR')
fig.suptitle('Residual maps: 8 x 4096-spp ECRO mean minus 2 x 4096-spp physical mean (residual family scale). Contains finite-sample noise of both means.', fontsize=9)
fig.savefig(OUT / 'bias_maps.png', dpi=75); plt.close(fig)

# F5 crops
for crop, name in ((SIL, 'silhouette'), (CON, 'contact')):
    fig, ax = plt.subplots(2, 5, figsize=(18, 5.6), constrained_layout=True)
    for i, s in enumerate((1, 16)):
        ax[i, 0].imshow(tone(img(JMAP[s]['rep0_T']))[crop]); ax[i, 0].set_title(f'{s} spp TOTAL (rep 0)', fontsize=8)
        im = show_pos(ax[i, 1], (mp(JMAP[s]['V_L']) ** 0.5)[crop], M_DEV, f'{s} spp std LOW')
        show_pos(ax[i, 2], (mp(JMAP[s]['V_H']) ** 0.5)[crop], M_DEV, f'{s} spp std HIGH')
        im2 = show_sig(ax[i, 3], sroot(mp(2 * JMAP[s]['C_LH']))[crop], M_DEV, f'{s} spp signed root 2Cov')
        im3 = show_sig(ax[i, 4], mp(BMAP['T'])[crop] if i == 0 else mp(BMAP['H'])[crop], M_RES, 'TOTAL bias (spp-independent)' if i == 0 else 'HIGH bias (spp-independent)')
        for a_ in ax[i]:
            a_.set_xticks([]); a_.set_yticks([])
    cbar(fig, im, ax[:, 1:3].ravel().tolist(), M_DEV, False, 'std, linear HDR')
    cbar(fig, im3, ax[:, 4].ravel().tolist(), M_RES, True, 'signed residual, linear HDR')
    fig.suptitle(f'{name} crop: noise components (stochastic family scale) and bias (residual family scale)', fontsize=9)
    fig.savefig(OUT / f'crop_{name}.png', dpi=80); plt.close(fig)

# F6 spp scaling, small multiples per primary ROI (one y axis each, relative variance)
fig, ax = plt.subplots(1, 4, figsize=(16, 4.2), sharey=False)
for a_, k in zip(ax, PRIMARY):
    for key, col, lab in (('V_L', CAT['L'], 'Var LOW'), ('V_H', CAT['H'], 'Var HIGH'), ('two_C_LH', CAT['C'], '|2 Cov| (hollow = negative)'), ('V_T', CAT['T'], 'Var TOTAL')):
        ys = np.array([J[s][k]['moments'][key]['point'] for s in SPPS]) / DEN[k]
        es = np.array([J[s][k]['moments'][key]['se'] for s in SPPS]) / DEN[k]
        a_.errorbar(SPPS, np.abs(ys), yerr=1.96 * es, color=col, lw=2, ms=0, capsize=3, label=lab)
        for s_, y_ in zip(SPPS, ys):
            a_.plot(s_, abs(y_), 'o', ms=8, mfc=col if y_ > 0 else '#fcfcfb', mec=col, mew=2)
    b2 = BIAS[k]['b2_TT']['point'] / DEN[k]
    if b2 > 0:
        a_.axhline(b2, color=CAT['B'], lw=2, ls='--', label='stable bias^2 TOTAL')
    a_.set_xscale('log', base=2); a_.set_yscale('log'); a_.set_xticks(SPPS); a_.set_xticklabels(SPPS)
    a_.set_title(f'{k} ({int(masks[k].sum())} px)', fontsize=9); a_.set_xlabel('spp')
    a_.grid(alpha=0.25, which='major', color='#d9d8d4')
ax[0].set_ylabel('variance / mean(ref TOTAL^2)')
ax[0].legend(fontsize=7, frameon=False, loc='lower left')
fig.suptitle('Variance vs spp (95% intervals) and spp-independent stable squared bias', fontsize=9)
fig.tight_layout(); fig.savefig(OUT / 'spp_scaling.png', dpi=90); plt.close(fig)

# F7 MSE budget per spp (signed shares of MSE_T)
fig, ax = plt.subplots(1, 3, figsize=(16, 4.4), sharey=True)
terms = (('share_V_L', 'Var LOW', CAT['L']), ('share_V_H', 'Var HIGH', CAT['H']), ('share_two_C_LH', '2 Cov(LOW,HIGH)', CAT['C']), ('share_b2_T', 'stable bias^2', CAT['B']))
for a_, s in zip(ax, SPPS):
    xs = np.arange(len(PRIMARY))
    for t_i, (key, lab, col) in enumerate(terms):
        vals = [BUD[s][k][key]['point'] if isinstance(BUD[s][k][key], dict) else BUD[s][k][key] for k in PRIMARY]
        a_.bar(xs + (t_i - 1.5) * 0.19, vals, 0.17, color=col, label=lab, edgecolor='#fcfcfb', linewidth=2)
    a_.axhline(0, color='#52514e', lw=0.8)
    a_.set_xticks(xs); a_.set_xticklabels(PRIMARY, fontsize=8); a_.set_title(f'{s} spp', fontsize=9)
    a_.grid(axis='y', alpha=0.25, color='#d9d8d4')
ax[0].set_ylabel('signed share of TOTAL MSE'); ax[0].legend(fontsize=7, frameon=False)
fig.suptitle('TOTAL MSE = Var LOW + Var HIGH + 2 Cov + stable bias^2 (signed shares; sum = 1)', fontsize=9)
fig.tight_layout(); fig.savefig(OUT / 'mse_budget.png', dpi=90); plt.close(fig)

# F8 stage shares (4 spp)
fig, ax = plt.subplots(1, 3, figsize=(16, 4.4), sharey=True)
for a_, comp, nm in zip(ax, ('T', 'L', 'H'), ('TOTAL', 'LOW', 'HIGH')):
    xs = np.arange(len(PRIMARY))
    for t_i, (st, lab, col) in enumerate((('J', 'jitter J', CAT['L']), ('P', 'first-event P', CAT['H']), ('S', 'k=2 walk S', CAT['C']))):
        c = [PF[4][k]['stage_share'][f'{comp}_{st}'] for k in PRIMARY]
        a_.bar(xs + (t_i - 1) * 0.26, [x['point'] for x in c], 0.24, yerr=[1.96 * x['se'] for x in c], color=col, label=lab,
               edgecolor='#fcfcfb', linewidth=2, capsize=3)
    a_.axhline(0, color='#52514e', lw=0.8); a_.set_xticks(xs); a_.set_xticklabels(PRIMARY, fontsize=8)
    a_.set_title(f'stage share of Var {nm}, 4 spp', fontsize=9); a_.grid(axis='y', alpha=0.25, color='#d9d8d4')
ax[0].legend(fontsize=7, frameon=False); ax[0].set_ylabel('share (95% interval)')
fig.suptitle('Pick-freeze stage shares (split diagnostic arm, valid only if G7 passed: ' + str(G7['pass']) + ')', fontsize=9)
fig.tight_layout(); fig.savefig(OUT / 'stage_shares.png', dpi=90); plt.close(fig)

# F9 ROI masks
ov = tone(img(RM['T'])).copy()
for k, col in (('sss_interior', (0.16, 0.47, 0.84)), ('near', (0.92, 0.41, 0.2)), ('control', (0.11, 0.69, 0.48)), ('silhouette', (0.9, 0.1, 0.9))):
    mm = masks[k]; ov[mm] = 0.45 * ov[mm] + 0.55 * np.array(col)
fig, ax = plt.subplots(figsize=(7, 7))
ax.imshow(ov); ax.set_xticks([]); ax.set_yticks([])
for crop, nm in ((SIL, 'silhouette crop'), (CON, 'contact crop')):
    ax.add_patch(plt.Rectangle((crop[1].start, crop[0].start), crop[1].stop - crop[1].start, crop[0].stop - crop[0].start, fill=False, ec='w', lw=1.2))
    ax.text(crop[1].start, crop[0].start - 4, nm, color='w', fontsize=8)
ax.set_title('Primary ROIs: sss_interior (blue), silhouette (magenta), near (orange), control (aqua)', fontsize=8)
fig.tight_layout(); fig.savefig(OUT / 'roi_masks.png', dpi=90); plt.close(fig)

# manifest + gallery
files = sorted(p.name for p in OUT.glob('*.png'))
CH.write_json(OUT / 'evaluation_manifest.json', {'id': P['id'], 'display_scales': SCALES, 'files': {f: CH.sha(OUT / f) for f in files + ['metrics.json']},
                                                 'raw_manifest_sha256': CH.sha(RAW / 'manifest.json'), 'source_raw': str(RAW)})
html = ['<!doctype html><meta charset="utf-8"><title>WL103 noise attribution</title><body style="font-family:sans-serif;background:#fcfcfb">',
        '<h1>WL103 — ECRO SSS LOW/HIGH noise attribution (raw, no reconstruction; frozen experiment id WL101H)</h1>',
        f'<p>Display scales: stochastic family max {M_DEV:.4g}, residual family max {M_RES:.4g} (linear HDR), sqrt mapping. Numbers: metrics.json.</p>']
for f in files:
    html.append(f'<h3>{f}</h3><img src="{f}" style="max-width:100%">')
(OUT / 'index.html').write_text('\n'.join(html), encoding='utf-8')
print('ANALYSIS COMPLETE', time.time() - t_start, flush=True)
