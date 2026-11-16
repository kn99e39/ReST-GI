"""WL97 audit, Track A, check A1 (numpy only, local).
(1) bin layout / weights: true layout from scene86 (NZ x NPHI), exact per-bin cos-weighted solid angle, an94.W as coded;
(2) M64 (S1) stored in results/ecro_attr95a/ops95a.npz:A4 : directional albedo under W_T, reciprocity symmetry, unconstrained rank bounds;
(3) WL95A delivery (results/ecro_attr95a/deliv.npz): flux ratios of every arm on receiver groups other than box A.
Run:  python experiments/audit97/A/a1_bins_m64_deliv.py
"""
import hashlib, json, math, os, sys
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'experiments', 'ecro86'))
import scene86 as S  # numpy only

OUT = os.path.join(ROOT, 'results', 'audit97', 'A'); os.makedirs(OUT, exist_ok=True)
sha = lambda fn: hashlib.sha256(open(fn, 'rb').read()).hexdigest()[:16]
res = {}

# ---------------------------------------------------------------- (1) bins and weights
NB, NZ, NPHI = S.NB, S.NZ, S.NPHI
b = np.arange(NB)
iz, ip = b // NPHI, b % NPHI
# exact integral of cos over the bin: int_{z0}^{z1} z dz * dphi
z0, z1 = iz / NZ, (iz + 1) / NZ
W_exact = 0.5 * (z1 ** 2 - z0 ** 2) * (2 * math.pi / NPHI)
WT = (iz + 0.5) / NZ * S.DOMEGA                              # centre-cos x dOmega (a95 / w96a / diag92 / c94 form)
W94 = ((b // 8) + 0.5) / 8 * (2 * math.pi / 64)              # an94.W as coded (NPHI = 8, NZ = 8)
ratio = W94 / WT
res['bins'] = {'NZ': NZ, 'NPHI': NPHI, 'NB': NB, 'DOMEGA': S.DOMEGA,
               'WT_equals_exact_bin_integral_max_abs': float(np.abs(WT - W_exact).max()), 'sum_WT': float(WT.sum()), 'sum_W94': float(W94.sum()),
               'W94_over_WT_by_ring_first_half_azimuth': [float(ratio[r * NPHI]) for r in range(NZ)],
               'W94_over_WT_by_ring_second_half_azimuth': [float(ratio[r * NPHI + 8]) for r in range(NZ)],
               'ring_sums_W94_over_WT': [float(W94[r * NPHI:(r + 1) * NPHI].sum() / WT[r * NPHI:(r + 1) * NPHI].sum()) for r in range(NZ)],
               'azimuth_convention': 'iphi = floor((atan2(y, x) + pi) * NPHI / 2pi): sectors 0..7 <-> local y < 0 (-bitangent side), 8..15 <-> y >= 0'}
# Lambert operator rho / pi: directional albedo under each weight
rho = 0.7
res['lambert'] = {'albedo_WT': float((rho / math.pi * WT).sum()), 'albedo_W94': float((rho / math.pi * W94).sum())}
# anisotropic test fields: radiance only in one azimuth half of one ring
for nm, sel in (('ring0_yneg', (iz == 0) & (ip < 8)), ('ring0_ypos', (iz == 0) & (ip >= 8)), ('ring3_yneg', (iz == 3) & (ip < 8)), ('ring3_ypos', (iz == 3) & (ip >= 8))):
    res['bins'][f'exitance_bias_W94_{nm}'] = float(W94[sel].sum() / WT[sel].sum())

# ---------------------------------------------------------------- (2) M64 of S1
fn = os.path.join(ROOT, 'results', 'ecro_attr95a', 'ops95a.npz')
ops = np.load(fn)
M = ops['A4'].astype(np.float64)                             # [bin_in, bin_out, 3]
alb = (M * WT[None, :, None]).sum(1)                         # [bin_in, 3]
alb94 = (M * W94[None, :, None]).sum(1)
sym = np.abs(M - M.transpose(1, 0, 2)).sum() / M.sum()
zz = np.sqrt(np.outer((iz + 0.5) / NZ, (iz + 0.5) / NZ))    # sqrt of the flux weights zb_in zb_out (Frobenius weighting)
sv = {}
for c in range(3):
    U_, s_, Vt = np.linalg.svd(M[..., c] * zz)
    e = np.sqrt(np.append(np.cumsum((s_ ** 2)[::-1])[::-1], 0.0))   # e[k] = Frobenius error of the best rank-k approximation
    sv[c] = {'rel_fro_err_rank8': float(e[8] / e[0]), 'rel_fro_err_rank16': float(e[16] / e[0]), 'rel_fro_err_rank32': float(e[32] / e[0]),
             'rank_for_10pct': int(np.argmax(e / e[0] <= 0.10)), 'rank_for_5pct': int(np.argmax(e / e[0] <= 0.05))}
# ring-0 rows only (grazing incidence, the CK2 regime): best rank-k approximation of the whole matrix, error measured on those rows
ring0 = iz == 0
r0 = {}
for c in range(3):
    A = M[..., c] * zz
    U_, s_, Vt = np.linalg.svd(A)
    for k in (16, 32):
        Ak = (U_[:, :k] * s_[:k]) @ Vt[:k]
        r0[f'c{c}_rank{k}_ring0_rows_rel_fro'] = float(np.linalg.norm((A - Ak)[ring0]) / np.linalg.norm(A[ring0]))
        pk = A[ring0].argmax(1)
        r0[f'c{c}_rank{k}_ring0_peak_retention'] = float(Ak[ring0][np.arange(ring0.sum()), pk].sum() / A[ring0][np.arange(ring0.sum()), pk].sum())
res['M64_S1'] = {'file': os.path.relpath(fn, ROOT), 'sha256_16': sha(fn), 'albedo_max_WT': float(alb.max()), 'albedo_mean_WT': float(alb.mean()),
                 'albedo_max_W94': float(alb94.max()), 'albedo_by_ring_mean_WT': [float(alb[iz == r].mean()) for r in range(NZ)],
                 'reciprocity_asymmetry_rel_l1': float(sym), 'min_entry': float(M.min()),
                 'svd_flux_weighted': sv, 'ring0_rows': r0,
                 'row_peak_share_ring0': float(np.mean([M[i, :, 1].max() / M[i, :, 1].sum() for i in np.nonzero(ring0)[0]])),
                 'A1_albedo_max_WT': float((ops['A1'] * WT[None, :, None]).sum(1).max()), 'A2_albedo_max_WT': float((ops['A2'] * WT[None, :, None]).sum(1).max())}

# ---------------------------------------------------------------- (3) delivery flux ratios by receiver group
ck = json.load(open(os.path.join(ROOT, 'experiments', 'ecro_local93d', 'addendum_checker', 'scene_ck2.json')))
G = S.Geom(ck['geom'])
fn = os.path.join(ROOT, 'results', 'ecro_attr95a', 'deliv.npz')
z = np.load(fn)
ref = 0.5 * (z['ref_a'] + z['ref_b'])
N = G.N
grp = {'boxA': np.arange(1130, 1184), 'boxA_panel_face': np.arange(1130, 1139), 'lambert_walls_ceiling': np.nonzero(G.PMAT == 3)[0],
       'boxB': np.nonzero(G.PSID == S.SID_BOXB)[0], 'floor_ER': np.nonzero(G.PMAT == 0)[0], 'sphere_ER': np.nonzero(G.PMAT == 2)[0],
       'all_lambert': np.nonzero(G.PMAT >= 3)[0], 'all_non_panel': np.nonzero(G.PMAT != 1)[0]}
dl = {'ref_share_of_total': {g: float(ref[i].sum() / ref.sum()) for g, i in grp.items()},
      'ref_half_noise': {g: float(0.5 * np.abs(z['ref_a'][i] - z['ref_b'][i]).sum() / max(ref[i].sum(), 1e-30)) for g, i in grp.items()}}
for arm in ('A4', 'M64_spool', 'q256', 'X1c', 'X1', 'A0_s8601', 'A0_s8602', 'A1', 'A3'):
    d = z[f'draws|{arm}']; dep = z[f'dep|{arm}']; pm = d.mean(0)
    o = {}
    for g, i in grp.items():
        den = max(ref[i].sum(), 1e-30)
        per = d[:, i].sum(1) / den
        o[g] = {'pooled_ratio': float(pm[i].sum() / den), 'pooled_ratio_se': float(per.std(ddof=1) / math.sqrt(len(per))),
                'deployed_ratio': float(dep[i].sum() / den), 'pooled_abs_rel_l1': float(np.abs(pm[i] - ref[i]).sum() / den),
                'per_draw_abs_rel_l1_median': float(np.median(np.abs(d[:, i] - ref[i]).sum(1) / den))}
    dl[arm] = o
res['delivery'] = {'file': os.path.relpath(fn, ROOT), 'sha256_16': sha(fn), 'draws': int(z['draws|A4'].shape[0]), **dl}
json.dump(res, open(os.path.join(OUT, 'a1_bins_m64_deliv.json'), 'w'), indent=1)
print(json.dumps(res['bins'], indent=1)); print(json.dumps(res['lambert'])); print(json.dumps(res['M64_S1'], indent=1))
print('ref share', dl['ref_share_of_total']); print('ref half noise', dl['ref_half_noise'])
for arm in ('A4', 'M64_spool', 'q256', 'X1c', 'X1', 'A0_s8601', 'A0_s8602'):
    print(arm, {g: (round(v['pooled_ratio'], 3), round(v['pooled_ratio_se'], 3), round(v['deployed_ratio'], 3), round(v['pooled_abs_rel_l1'], 3), round(v['per_draw_abs_rel_l1_median'], 3)) for g, v in dl[arm].items()})
