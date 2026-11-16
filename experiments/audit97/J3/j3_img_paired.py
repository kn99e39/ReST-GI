"""J3: paired image-level statistics for WL95A CK2 ROI_T renders (read-only inputs).
dE_H(arm) = E_H(arm) - E_H(floor); noise from (a) per-replicate pairing (same camera seed) x reference halves,
(b) 16x16-pixel block bootstrap of the per-pixel paired difference (conservative), (c) unpaired cross-seed control."""
import glob, json, os, sys, hashlib
import numpy as np
R = os.path.expanduser('~/rest_x95a/results'); OUT = sys.argv[1]
ref = [np.load(f) for f in sorted(glob.glob(f'{R}/renders/REF_ER_CK_none_CK2_T1_r*.npz'))]
H = np.stack([z['h'].astype(np.float64) for z in ref]); TOT = np.stack([(z['e1'] + z['e2'] + z['h']).astype(np.float64) for z in ref]).mean(0)
m = np.load(f'{R}/pre/masks94.npz')['ROI_T'].astype(bool)
den = TOT[m].sum(); h = H.mean(0); hA, hB = H[:2].mean(0), H[2:].mean(0)
def X(k): return np.stack([np.load(f)['hl'].astype(np.float64) for f in sorted(glob.glob(f'{R}/renders/CK_none_{k}_X_CK2_T1_r*.npz'))])
pix = lambda x, hh: np.abs(x - hh).sum(-1)  # per-pixel abs error summed over channels
iy, ix = np.nonzero(m); blk = (iy // 16) * 64 + ix // 16; ub, inv = np.unique(blk, return_inverse=True)
rng = np.random.default_rng(97031); B = 2000
bs_idx = rng.integers(0, len(ub), size=(B, len(ub)))
F = X('floor'); cF = pix(F.mean(0), h)[m]
arms = ['pool_X1', 'pool_X1c', 'pool_q256', 'pool_M64_spool', 'pool_A4', 'dep_A4', 'pool_A1', 'pool_A3', 'dep_A1', 'dep_A3',
        'dep_A0_s8601', 'dep_A0_s8602', 'G_s8601', 'G_s8602']
o = {'den': float(den), 'E_H_floor': float(cF.sum() / den), 'arms': {}}
cache = {}
for k in arms:
    A = X(k); cache[k] = A
    d = pix(A.mean(0), h)[m] - cF
    dE = float(d.sum() / den)
    paired = [float((pix(A[i], hh)[m] - pix(F[i], hh)[m]).sum() / den) for i in range(2) for hh in (hA, hB)]
    unpaired = [float((pix(A[i], hh)[m] - pix(F[1 - i], hh)[m]).sum() / den) for i in range(2) for hh in (hA, hB)]
    bsum = np.bincount(inv, weights=d, minlength=len(ub))
    bs = bsum[bs_idx].sum(1) / den
    o['arms'][k] = dict(E_H=float(pix(A.mean(0), h)[m].sum() / den), dE_H_vs_floor=dE, paired_reps=paired,
                        paired_sd=float(np.std(paired, ddof=1)), unpaired_reps=unpaired, unpaired_sd=float(np.std(unpaired, ddof=1)),
                        block_boot_se=float(bs.std(ddof=1)))
def pair(k1, k2):
    A1, A2 = cache[k1], cache[k2]
    d = pix(A1.mean(0), h)[m] - pix(A2.mean(0), h)[m]
    reps = [float((pix(A1[i], hh)[m] - pix(A2[i], hh)[m]).sum() / den) for i in range(2) for hh in (hA, hB)]
    bs = np.bincount(inv, weights=d, minlength=len(ub))[bs_idx].sum(1) / den
    return dict(dE_H=float(d.sum() / den), reps=reps, block_boot_se=float(bs.std(ddof=1)))
o['pairs'] = {f'{a}-{b}': pair(a, b) for a, b in (('pool_A4', 'pool_X1'), ('pool_M64_spool', 'pool_X1'), ('pool_q256', 'pool_X1c'),
                                                  ('pool_X1c', 'pool_X1'), ('dep_A4', 'pool_X1'), ('pool_A4', 'dep_A4'), ('pool_A1', 'pool_A4'))}
o['protocol_tol'] = 0.0167
json.dump(o, open(OUT, 'w'), indent=1)
for k, v in o['arms'].items(): print(k, round(v['dE_H_vs_floor'], 5), 'pairedSD', '%.1e' % v['paired_sd'], 'unpairedSD', '%.1e' % v['unpaired_sd'], 'bootSE', '%.1e' % v['block_boot_se'])
for k, v in o['pairs'].items(): print(k, round(v['dE_H'], 5), [round(x, 5) for x in v['reps']], 'bootSE', '%.1e' % v['block_boot_se'])
