"""WL97 audit, track E4: extra checks (numpy only).
  (a) per-half fluxes of the reference families (is the e3_p flux z-score of fieldan.json real or an SE artefact of 2 halves?)
  (b) new-seed renders really differ from the stored ones (camera noise sanity)
  (c) validity of the table readout as an image-level proxy, split by event count, against INDEPENDENT event-split camera references
      (REFA_k3 = exactly 3 camera events, REFA_k6 = 3..6 events; same random streams, so k6 - k3 = 4..6 events exactly)

Usage: python e4_extra.py <ORIG renders> <masks94.npz> <AUDIT renders> <REFA dir> <CKF_none.npz> <out.json> <TF_1.npz> ...
"""
import json, math, os, sys
import numpy as np

ORIG, MASKS, MINE, REFA, CKF, OUT = sys.argv[1:7]
TFS = sys.argv[7:]
N, NB, NZ, NPHI = 1270, 64, 4, 16
WT = ((np.arange(NB) // NPHI + 0.5) / NZ) * (2 * math.pi / NB)
R = {}

# ---------------------------------------------------------------- (a)
z = np.load(CKF)
tf = [np.load(f) for f in TFS]
PMAT = tf[0]['PMAT']
fl = lambda T, idx: float((T[idx].astype(np.float64) * WT[None, :, None]).sum())
reg = {'all': np.arange(N), 'lambert': np.nonzero(PMAT >= 3)[0], 'boxA': np.arange(1130, 1184), 'floor': np.arange(0, 256)}
pa = {}
for rn, idx in reg.items():
    d = {}
    for fam, cls in (('e3_p', [1]), ('e3_o', [0, 2, 3]), ('e2_p', [1]), ('e2_o', [0, 2, 3])):
        ks = (2,) if fam.startswith('e2') else (3, 4, 5)
        tags = [c * 2 + e for c in cls for e in (0, 1)]
        ck = [fl(z[f'{fam}_{h}'], idx) for h in 'ab']
        au = [sum(fl(t[f'k{k}'][:, :, tags].sum(2), idx) for k in ks) for t in tf]
        allv = np.array(ck + au)
        d[fam] = {'ckf_halves': ck, 'audit_halves': au, 'pooled_cv_per_half': float(allv.std(ddof=1) / allv.mean()),
                  'ckf_mean_minus_audit_mean_in_pooled_sd_units': float((np.mean(ck) - np.mean(au)) / (allv.std(ddof=1) * math.sqrt(1 / 2 + 1 / len(au))))}
    pa[rn] = d
R['per_half_flux'] = pa

# ---------------------------------------------------------------- images
lo = lambda key, n=4: np.stack([np.load(os.path.join(ORIG, f'CK_none_{key}_X_CK2_T1_r{r}.npz'))['hl'].astype(np.float64) for r in range(n)])
lm = lambda key, n=4: np.stack([np.load(os.path.join(MINE, f'A_{key}_s{97600 + r}.npz'))['hl'].astype(np.float64) for r in range(n)])
ref = {k: np.stack([np.load(os.path.join(ORIG, f'REF_ER_CK_none_CK2_T1_r{r}.npz'))[k].astype(np.float64) for r in range(4)]) for k in ('e1', 'e2', 'h')}
tot = (ref['e1'] + ref['e2'] + ref['h']).mean(0)
mk = np.load(MASKS); sid = mk['sid']
MS = {'ROI_T': mk['ROI_T'], 'DIFF': mk['DIFF'], 'WHOLE': np.ones_like(mk['ROI_T']), 'PANEL': mk['panel'], 'PANEL_NOT_ROI': mk['panel'] & ~mk['ROI_T'],
      'BOXA_PIX': sid == 7, 'FLOOR_PIX': sid == 0, 'WALL_PIX': (sid >= 1) & (sid <= 4)}
Fo, Fm = lo('F'), lm('F')
R['seed_sanity'] = {'F_new_seed0_vs_stored_r0_rel_l1': float(np.abs(Fm[0] - Fo[0]).sum() / Fo[0].sum()),
                    'F_stored_r0_vs_r1_rel_l1': float(np.abs(Fo[0] - Fo[1]).sum() / Fo[0].sum())}
seeds = [97700 + r for r in range(4)]
have = all(os.path.exists(os.path.join(REFA, f'REFA_k{k}_s{s}.npz')) for k in (3, 6) for s in seeds)
if have:
    ld = lambda k, s, key: np.load(os.path.join(REFA, f'REFA_k{k}_s{s}.npz'))[key].astype(np.float64)
    h3 = np.stack([ld(3, s, 'h') for s in seeds])
    h6 = np.stack([ld(6, s, 'h') for s in seeds])
    R['refa_streams_identical (e1 + e2 of k3 vs k6) max abs'] = float(max(np.abs(ld(3, s, 'e1') + ld(3, s, 'e2') - ld(6, s, 'e1') - ld(6, s, 'e2')).max() for s in seeds))
    Zo, Zm = lo('Z'), lm('Z')
    Xo, Xm = lo('XOD'), lm('XOD')
    F8 = np.concatenate([Fo, Fm]); Z8 = np.concatenate([Zo, Zm]); X8 = np.concatenate([Xo, Xm])

    def mv(reps):
        k = reps.shape[0]
        return reps.mean(0), reps.var(0, ddof=1) / k, 0.5 * (reps[:k // 2].mean(0) - reps[k // 2:].mean(0))

    def sub(a, b):
        return a[0] - b[0], a[1] + b[1], a[2] - b[2]

    def blockL1(img, m, b):
        x = img * m[..., None]
        H, W = m.shape
        return float(np.abs(x.reshape(H // b, b, W // b, b, 3).sum((1, 3))).sum())

    def cmp(x, y, m):
        """x, y = (mean, variance of the mean, half-split noise realization); own statistics of x - y inside mask m (relative to the mask total)."""
        den = tot[m].sum(); t2 = float((tot[m] ** 2).sum())
        d = (x[0] - y[0]); v = (x[1] + y[1]); nz = x[2] - y[2]
        dm, vm = d[m], v[m]
        o = {'L1': float(np.abs(dm).sum() / den), 'L1_null_expectation': float(np.sqrt(vm * 2 / math.pi).sum() / den),
             'L1_noise_realization': float(np.abs(nz[m]).sum() / den), 'signed': float(dm.sum() / den),
             'y_share_of_tot': float(y[0][m].sum() / den), 'x_over_y': float(x[0][m].sum() / max(y[0][m].sum(), 1e-30))}
        d2 = float((dm ** 2).sum()); vs = float(vm.sum())
        o['rel_l2_raw_vs_tot'] = math.sqrt(d2 / t2); o['rel_l2_noise_vs_tot'] = math.sqrt(vs / t2)
        o['rel_l2_debiased_vs_tot'] = math.copysign(math.sqrt(abs(d2 - vs) / t2), d2 - vs)
        y2 = float((y[0][m] ** 2).sum())
        o['rel_l2_debiased_vs_y'] = math.copysign(math.sqrt(abs(d2 - vs) / max(y2, 1e-300)), d2 - vs)
        for b in (8, 16, 64):
            o[f'L1_block{b}'] = blockL1(d, m, b) / den; o[f'L1_block{b}_noise'] = blockL1(nz, m, b) / den
        return o

    def signed_se(xr, yr, m):
        den = tot[m].sum()
        sx = np.array([a[m].sum() for a in xr]); sy = np.array([a[m].sum() for a in yr])
        return float(math.sqrt(sx.var(ddof=1) / len(sx) + sy.var(ddof=1) / len(sy)) / den)

    C = {}
    for mn, m in MS.items():
        c = {}
        c['hstored_vs_h6new (reference consistency)'] = cmp(mv(ref['h']), mv(h6), m)
        c['F_vs_hstored (3..6 events)'] = cmp(mv(F8), mv(ref['h']), m)
        c['F_vs_h6new (3..6 events)'] = cmp(mv(F8), mv(h6), m)
        c['Z_vs_h3new (table e2 read = 3 camera events)'] = cmp(mv(Z8), mv(h3), m)
        c['tail: (F - Z) vs (h6 - h3) new, paired (4..6 camera events)'] = cmp(mv(F8 - Z8), mv(h6 - h3), m)
        c['tail: (F - Z) vs (hstored - h3new)'] = cmp(mv(F8 - Z8), sub(mv(ref['h']), mv(h3)), m)
        c['tail explicit: (XOD - Z) vs (h6 - h3) new'] = cmp(mv(X8 - Z8), mv(h6 - h3), m)
        c['tail explicit: (XOD - Z) vs (hstored - h3new)'] = cmp(mv(X8 - Z8), sub(mv(ref['h']), mv(h3)), m)
        c['signed_se F_vs_hstored'] = signed_se(F8, ref['h'], m)
        c['signed_se tail paired'] = signed_se(F8 - Z8, h6 - h3, m)
        c['signed_se Z_vs_h3new'] = signed_se(Z8, h3, m)
        C[mn] = c
    R['table_readout_proxy'] = C
json.dump(R, open(OUT, 'w'), indent=1)
print('E4EXTRA DONE', OUT, 'refa', have, flush=True)
