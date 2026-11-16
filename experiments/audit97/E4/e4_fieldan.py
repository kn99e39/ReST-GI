"""WL97 audit, track E4: field-level analysis (numpy only, own weights and metrics).

  (1) event accounting: CKF_none families vs the audit's tagged tracer (new seeds), alternative-hypothesis discrimination
  (2) composition of e3_o by last-sender class and by an earlier panel interaction (per order, per receiver region)
  (3) explicit tail vs reference: per family / last-sender class / order / receiver region, noise floors, debiased L2, cross-independent
  (4) prefix-noise propagation into the explicit tail
  (5) render tables for e4_render.py

Usage: python e4_fieldan.py <CKF_none.npz> <tails97.npz> <out.json> <tables_out.npz> <TF_1.npz> ...
"""
import json, math, sys
import numpy as np

CKF, T97, OUT, TOUT = sys.argv[1:5]
TFS = sys.argv[5:]
N, NB, NZ, NPHI = 1270, 64, 4, 16
DOM = 2 * math.pi / NB
WT = ((np.arange(NB) // NPHI + 0.5) / NZ) * DOM              # mean cos of the ring x bin solid angle (sums to pi)
assert abs(WT.sum() - math.pi) < 1e-12
W3 = WT[None, :, None]
CLS = ('floor', 'panel', 'sphere', 'lambert')
OCL = [0, 2, 3]

tf = [np.load(f) for f in TFS]
PMAT = tf[0]['PMAT']; PSID = tf[0]['PSID']
REG = {'boxA': np.arange(1130, 1184), 'panel_face': np.arange(1130, 1139), 'S1': np.arange(1088, 1130), 'floor': np.arange(0, 256),
       'sphere': np.arange(1238, 1270), 'lambert': np.nonzero(PMAT >= 3)[0], 'nonS1': np.nonzero(PMAT != 1)[0], 'all': np.arange(N)}
assert (PMAT[REG['S1']] == 1).all() and (PSID[REG['boxA']] == 7).all() and (PMAT[REG['floor']] == 0).all() and (PSID[REG['sphere']] == 6).all()

z = np.load(CKF)
C = {k: np.stack([z[f'{k}_a'], z[f'{k}_b']]).astype(np.float64) for k in ('e2_p', 'e2_o', 'e3_p', 'e3_o')}       # [2, N, NB, 3]
# audit halves: K[k] = [H, N, NB, 8, 3], tag = cls1 * 2 + ep
KK = {k: np.stack([t[f'k{k}'] for t in tf]).astype(np.float64) for k in range(2, 7)}
H = len(tf)
cls_of = np.arange(8) // 2; ep_of = np.arange(8) % 2


def sel(k, cls=None, ep=None):
    m = np.ones(8, bool)
    if cls is not None:
        m &= np.isin(cls_of, cls)
    if ep is not None:
        m &= ep_of == ep
    return KK[k][:, :, :, m].sum(3)                           # [H, N, NB, 3]


A = {'e2_p': sel(2, [1]), 'e2_o': sel(2, OCL), 'e3_p': sum(sel(k, [1]) for k in (3, 4, 5)), 'e3_o': sum(sel(k, OCL) for k in (3, 4, 5))}
ex = lambda T, idx: (T[..., idx, :, :] * W3).sum((-1, -2))                         # per-patch exitance
flux = lambda T, idx: float((T[..., idx, :, :] * W3).sum()) if T.ndim == 3 else (T[..., idx, :, :] * W3).sum((-1, -2, -3))
tw = lambda T, idx: float((np.abs(T[idx]) * W3).sum())


def l1noise(halves, idx):
    """L1 noise of the MEAN of independent halves (table level): two-group split estimate 0.5 sum|mean(g1) - mean(g2)| (W-weighted)."""
    h = len(halves)
    return 0.5 * tw(halves[:h // 2].mean(0) - halves[h // 2:].mean(0), idx)


R = {'halves_audit': H, 'weights': 'WT = (ring + 0.5) / 4 * 2 pi / 64 (4 rings x 16 sectors)'}

# ------------------------------------------------------------------------------------------------ (1) event accounting
ea = {}
for fam in ('e2_p', 'e2_o', 'e3_p', 'e3_o'):
    for rn in ('boxA', 'S1', 'floor', 'all'):
        idx = REG[rn]
        fc = flux(C[fam], idx); fa = flux(A[fam], idx)                 # per-half fluxes
        se = math.sqrt(fc.var(ddof=1) / 2 + fa.var(ddof=1) / H)
        mc, ma = C[fam].mean(0), A[fam].mean(0)
        den = max(tw(0.5 * (mc + ma), idx), 1e-30)
        nc, na = l1noise(C[fam], idx), l1noise(A[fam], idx)
        ea[f'{fam}|{rn}'] = {'flux_ckf': float(fc.mean()), 'flux_audit': float(fa.mean()), 'ratio': float(fa.mean() / max(fc.mean(), 1e-30)),
                             'z_flux': float((fa.mean() - fc.mean()) / max(se, 1e-30)), 'table_rel_l1_audit_vs_ckf': tw(ma - mc, idx) / den,
                             'l1noise_ckf_mean': nc / den, 'l1noise_audit_mean': na / den, 'null_expect_quadrature': math.sqrt(nc ** 2 + na ** 2) / den}
R['event_accounting'] = ea
# alternative hypotheses for the stored families (all receivers; flux and table rel L1 against the CKF mean)
alt = {}
ce3 = (C['e3_p'] + C['e3_o']).mean(0)
idx = REG['all']
for nm, ks in (('k3..5 (claimed)', (3, 4, 5)), ('k3..4', (3, 4)), ('k3..6', (3, 4, 5, 6)), ('k3 only', (3,))):
    m = sum(sel(k) for k in ks).mean(0)
    alt[f'e3 = {nm}'] = {'flux_ratio_vs_ckf': flux(m, idx) / flux(ce3, idx), 'table_rel_l1': tw(m - ce3, idx) / tw(ce3, idx)}
ce3p = C['e3_p'].mean(0)
for nm, T in (('panel at s1 (claimed)', A['e3_p'].mean(0)),
              ('panel at any vertex >= 3 (earlier), s1 any', sum(sel(k, None, 1) for k in (3, 4, 5)).mean(0)),
              ('panel anywhere (s1 or earlier)', (A['e3_p'] + sum(sel(k, OCL, 1) for k in (3, 4, 5))).mean(0))):
    alt[f'e3_p = {nm}'] = {'flux_ratio_vs_ckf': flux(T, idx) / flux(ce3p, idx), 'table_rel_l1': tw(T - ce3p, idx) / tw(ce3p, idx)}
alt['noise_ckf_e3_all'] = l1noise(C['e3_p'] + C['e3_o'], idx) / tw(ce3, idx)
alt['noise_ckf_e3p_all'] = l1noise(C['e3_p'], idx) / tw(ce3p, idx)
alt['k6_over_e3_flux_all'] = flux(sel(6).mean(0), idx) / flux(ce3, idx)
alt['order_flux_all'] = {f'k{k}': flux(sel(k).mean(0), idx) for k in range(2, 7)}
R['alternatives'] = alt

# ------------------------------------------------------------------------------------------------ (2) composition of e3_o
comp = {}
for rn, idx in REG.items():
    tot_o = flux(A['e3_o'].mean(0), idx)
    tot_e3 = tot_o + flux(A['e3_p'].mean(0), idx)
    d = {'e3_o_flux': tot_o, 'e3_o_share_of_e3': tot_o / max(tot_e3, 1e-30)}
    for ci in OCL:
        d[f'last_{CLS[ci]}'] = sum(flux(sel(k, [ci]).mean(0), idx) for k in (3, 4, 5)) / max(tot_o, 1e-30)
    d['earlier_panel_any'] = sum(flux(sel(k, OCL, 1).mean(0), idx) for k in (3, 4, 5)) / max(tot_o, 1e-30)
    d['earlier_panel_k3 (light->panel->s1->r; exact in the prefix)'] = flux(sel(3, OCL, 1).mean(0), idx) / max(tot_o, 1e-30)
    d['earlier_panel_k4_5 (binned panel sender enters the explicit recurrence)'] = sum(flux(sel(k, OCL, 1).mean(0), idx) for k in (4, 5)) / max(tot_o, 1e-30)
    hv = np.array([sum(flux(sel(k, OCL, 1)[h], idx) for k in (3, 4, 5)) / max(flux(A['e3_o'][h], idx), 1e-30) for h in range(H)])
    d['earlier_panel_any_se'] = float(hv.std(ddof=1) / math.sqrt(H))
    d['glossy_last (floor + sphere)'] = d['last_floor'] + d['last_sphere']
    d['order_share'] = {f'k{k}': flux(sel(k, OCL).mean(0), idx) / max(tot_o, 1e-30) for k in (3, 4, 5)}
    for ci in OCL:
        d[f'earlier_panel_within_last_{CLS[ci]}'] = sum(flux(sel(k, [ci], 1).mean(0), idx) for k in (3, 4, 5)) / max(sum(flux(sel(k, [ci]).mean(0), idx) for k in (3, 4, 5)), 1e-30)
    d['e3_p_earlier_panel_share'] = sum(flux(sel(k, [1], 1).mean(0), idx) for k in (3, 4, 5)) / max(flux(A['e3_p'].mean(0), idx), 1e-30)
    comp[rn] = d
R['e3_o_composition'] = comp

# ------------------------------------------------------------------------------------------------ (3) explicit vs reference
t97 = np.load(T97)
dep = {k.split('|')[1]: t97[k].astype(np.float64) for k in t97.files if k.startswith('dep|')}     # [3 orders, 4 classes, N, NB, 3]
pool = {k.split('|')[1]: t97[k].astype(np.float64) for k in t97.files if k.startswith('pool|')}
o_of = lambda T: T[:, OCL].sum((0, 1))
p_of = lambda T: T[:, 1].sum(0)
# references: halves [h, N, NB, 3]
REFH = {'ckf': {'e3_o': C['e3_o'], 'e3_p': C['e3_p']}, 'audit': {'e3_o': A['e3_o'], 'e3_p': A['e3_p']}}
REFH['all6'] = {f: np.concatenate([REFH['ckf'][f], REFH['audit'][f]]) for f in ('e3_o', 'e3_p')}


def fmet(Xt, halves, idx):
    """own field metrics of table Xt against the mean of independent reference halves."""
    h = len(halves); ref = halves.mean(0)
    eX, eR = ex(Xt, idx), ex(ref, idx)
    den_t, den_e = max(tw(ref, idx), 1e-30), max(eR.sum(), 1e-30)
    eh = ex(halves, idx)                                                          # [h, n]
    o = {'flux_ratio': float(eX.sum() / den_e), 'exit_rel_l1': float(np.abs(eX - eR).sum() / den_e),
         'exit_n0': float(0.5 * np.abs(eh[:h // 2].mean(0) - eh[h // 2:].mean(0)).sum() / den_e),
         'table_rel_l1': tw(Xt - ref, idx) / den_t, 'table_n0': l1noise(halves, idx) / den_t}
    fl = eh.sum(1)
    o['flux_ratio_se'] = float(fl.std(ddof=1) / math.sqrt(h) / den_e)
    # debiased relative L2 (table, W-weighted): ||X - mu||^2 estimated as ||X - ref||^2 - sum var(ref)
    d2 = float((((Xt - ref)[idx]) ** 2 * W3).sum())
    v = float((halves[:, idx].var(0, ddof=1) / h * W3).sum())
    n2 = max(float((ref[idx] ** 2 * W3).sum()), 1e-300)
    o['table_rel_l2_raw'] = math.sqrt(d2 / n2); o['table_rel_l2_noise'] = math.sqrt(v / n2)
    o['table_rel_l2_debiased'] = math.copysign(math.sqrt(abs(d2 - v) / n2), d2 - v)
    # debiased relative L2 of the per-patch exitance
    d2e = float(((eX - eR) ** 2).sum()); ve = float((eh.var(0, ddof=1) / h).sum()); n2e = max(float((eR ** 2).sum()), 1e-300)
    o['exit_rel_l2_raw'] = math.sqrt(d2e / n2e); o['exit_rel_l2_noise'] = math.sqrt(ve / n2e)
    o['exit_rel_l2_debiased'] = math.copysign(math.sqrt(abs(d2e - ve) / n2e), d2e - ve)
    if len(idx) > 2:
        o['corr_exit'] = float(np.corrcoef(eX, eR)[0, 1])
    return o


fm = {}
arms = {'XD(prefix ckf)': dep['ckf'], 'XP(prefix ckf)': pool['ckf'], 'XD(prefix all6)': dep['all'], 'XP(prefix all6)': pool['all'],
        'XD(prefix audit)': dep['new']}
for an, T in arms.items():
    for fam, X_ in (('e3_o', o_of(T)), ('e3_p', p_of(T))):
        for rfn in ('ckf', 'audit', 'all6'):
            for rn, idx in REG.items():
                if fam == 'e3_p' and rn == 'S1':
                    continue
                fm[f'{an}|{fam}|ref {rfn}|{rn}'] = fmet(X_, REFH[rfn][fam], idx)
R['field'] = fm
# per last-sender class and per order (reference: audit halves, which carry the tags), deployed + pooled, prefix all6
po = {}
for an, T in (('XD(prefix all6)', dep['all']), ('XP(prefix all6)', pool['all']), ('XD(prefix ckf)', dep['ckf'])):
    for ci in range(4):
        halves = sum(sel(k, [ci]) for k in (3, 4, 5))
        for rn in ('boxA', 'S1', 'floor', 'sphere', 'lambert', 'all'):
            if ci == 1 and rn == 'S1':
                continue
            po[f'{an}|last {CLS[ci]}|{rn}'] = fmet(T[:, ci].sum(0), halves, REG[rn])
    for j, k in enumerate((3, 4, 5)):
        halves = sel(k, OCL)
        for rn in ('boxA', 'S1', 'floor', 'all'):
            po[f'{an}|order k{k} (e3_o part)|{rn}'] = fmet(T[j][OCL].sum(0), halves, REG[rn])
R['field_by_class_and_order'] = po
# ep-conditional check: explicit cannot tag ep; reference with / without an earlier panel (sizes only) are in e3_o_composition.

# ------------------------------------------------------------------------------------------------ (4) prefix-noise propagation
pn = {}
names = [k for k in dep if k.startswith('new') and k != 'new']
for rn in ('boxA', 'S1', 'floor', 'all'):
    idx = REG[rn]
    den = tw(o_of(dep['all']), idx)
    d = {'ckf_a_vs_ckf_b (two 4096-spp halves)': 0.5 * tw(o_of(dep['ckf_a']) - o_of(dep['ckf_b']), idx) / den,
         'explicit(prefix ckf) vs explicit(prefix audit) table rel L1': tw(o_of(dep['ckf']) - o_of(dep['new']), idx) / den,
         'flux ratio explicit(ckf) / explicit(audit)': flux(o_of(dep['ckf']), idx) / flux(o_of(dep['new']), idx)}
    if len(names) >= 4:
        g1 = 0.5 * (o_of(dep[names[0]]) + o_of(dep[names[1]])); g2 = 0.5 * (o_of(dep[names[2]]) + o_of(dep[names[3]]))
        d['audit pairs (2 x 2 halves): L1 noise of the 4-half mean'] = 0.5 * tw(g1 - g2, idx) / den
    d['reference tail e3_o: L1 noise of the ckf 2-half mean'] = l1noise(C['e3_o'], idx) / tw(C['e3_o'].mean(0), idx)
    d['reference prefix e2: L1 noise of the ckf 2-half mean'] = l1noise(C['e2_p'] + C['e2_o'], idx) / tw((C['e2_p'] + C['e2_o']).mean(0), idx)
    pn[rn] = d
R['prefix_noise_propagation (explicit e3_o, deployed)'] = pn
# cross-independent: explicit from one independent prefix set vs the reference tail of the other set
ci_ = {}
for rn in ('boxA', 'S1', 'floor', 'all'):
    idx = REG[rn]
    ci_[rn] = {'explicit(prefix audit) vs tail ckf': fmet(o_of(dep['new']), C['e3_o'], idx), 'explicit(prefix ckf) vs tail audit': fmet(o_of(dep['ckf']), A['e3_o'], idx),
               'explicit(prefix ckf_a) vs tail ckf_b: table rel L1': tw(o_of(dep['ckf_a']) - C['e3_o'][1], idx) / tw(C['e3_o'][1], idx),
               'tail ckf_a vs tail ckf_b: table rel L1': tw(C['e3_o'][0] - C['e3_o'][1], idx) / tw(C['e3_o'][1], idx)}
R['cross_independent'] = ci_

# ------------------------------------------------------------------------------------------------ magnitudes
mg = {}
for rn, idx in REG.items():
    mg[rn] = {'e2_ckf': flux((C['e2_p'] + C['e2_o']).mean(0), idx), 'e3_p_ckf': flux(C['e3_p'].mean(0), idx), 'e3_o_ckf': flux(C['e3_o'].mean(0), idx),
              'e2_audit': flux((A['e2_p'] + A['e2_o']).mean(0), idx), 'e3_p_audit': flux(A['e3_p'].mean(0), idx), 'e3_o_audit': flux(A['e3_o'].mean(0), idx),
              'k6_audit': flux(sel(6).mean(0), idx)}
R['magnitude'] = mg
json.dump(R, open(OUT, 'w'), indent=1)

# ------------------------------------------------------------------------------------------------ (5) render tables
E2c = (C['e2_p'] + C['e2_o']).mean(0); e3pc = C['e3_p'].mean(0); e3oc = C['e3_o'].mean(0)
E2n = (A['e2_p'] + A['e2_o']).mean(0); e3pn = A['e3_p'].mean(0); e3on = A['e3_o'].mean(0)
xo, xp = o_of(dep['ckf']), p_of(dep['ckf'])
D = xo - e3oc
msk = lambda idx: np.isin(np.arange(N), idx)[:, None, None]
tab = {'F': E2c + e3pc + e3oc, 'Fa': E2c + C['e3_p'][0] + C['e3_o'][0], 'Fb': E2c + C['e3_p'][1] + C['e3_o'][1], 'XOD': E2c + e3pc + xo,
       'dOab': C['e3_o'][0] - C['e3_o'][1], 'XD': E2c + xp + xo, 'XOP': E2c + e3pc + o_of(pool['ckf']), 'Z': E2c,
       'Pc': E2c + e3pc, 'Fn': E2n + e3pn + e3on, 'Pn': E2n + e3pn, 'XODn': E2n + e3pn + o_of(dep['new']),
       'D_S1': D * msk(REG['S1']), 'D_floor': D * msk(REG['floor']), 'D_sphere': D * msk(REG['sphere']),
       'dOn': 0.5 * (A['e3_o'][0] + A['e3_o'][1]) - 0.5 * (A['e3_o'][2] + A['e3_o'][3]) if H >= 4 else e3on * 0,
       'k6': sel(6).mean(0)}
np.savez_compressed(TOUT, **{k: v.astype(np.float32) for k, v in tab.items()})
print('E4FIELDAN DONE', OUT, flush=True)
