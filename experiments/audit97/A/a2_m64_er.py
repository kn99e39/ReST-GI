"""WL97 audit, Track A, check A2 (numpy only, local): the M64 teacher against the ER closure itself.

(1) convention: eval(wi, wo) carries cos(wo) (Lambert state id 3 -> rho/pi cos(wo)); reciprocity of f.
(2) independent deterministic recomputation of M64[S1] = mean over the 8 x 8 incident cell centres of a bin of
    (1 / DOMEGA) int_{bo} f(v, w) dv  (quad87.enc_matrix) and comparison with the stored operator (ops95a.npz:A4 = mat94.mq_ext()[1]).
(3) what M64 is NOT: (a) response to a single incident direction; (b) a flux-conserving projection. For a single direction w the
    discrete system emits  sum_bo ZB[bo] int_bo f dv  per unit irradiance, while the physical exitance is int f cos dv.
Run:  python experiments/audit97/A/a2_m64_er.py
"""
import hashlib, json, math, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
sys.path.insert(0, HERE)
import er_np as E

OUT = os.path.join(ROOT, 'results', 'audit97', 'A')
mats = np.load(os.path.join(ROOT, 'experiments', 'ecro86', 'data', 'mats86.npz'))
ST = mats['states'].astype(np.float64)
res = {'mats86_sha256_16': hashlib.sha256(open(os.path.join(ROOT, 'experiments', 'ecro86', 'data', 'mats86.npz'), 'rb').read()).hexdigest()[:16]}
rng = np.random.default_rng(97001)

# ---------------------------------------------------------------- (1) conventions
v = rng.normal(size=(4096, 3)); v[:, 2] = np.abs(v[:, 2]) + 0.02; v /= np.linalg.norm(v, axis=1, keepdims=True)
w = rng.normal(size=(4096, 3)); w[:, 2] = np.abs(w[:, 2]) + 0.02; w /= np.linalg.norm(w, axis=1, keepdims=True)
lam = E.eval_cos(ST[3], w, v)
res['convention'] = {'lambert_eval_equals_rho_over_pi_cos_wo_max_rel': float(np.abs(lam[:, 1] / (0.7 / math.pi * v[:, 2]) - 1).max()),
                     'reciprocity_fr_max_rel': {str(m): float((np.abs(E.fr(ST[m], w, v) - E.fr(ST[m], v, w)) / np.maximum(E.fr(ST[m], w, v), 1e-12)).max()) for m in (0, 1, 2)}}

# ---------------------------------------------------------------- (2) M64 recomputation (S1 = material 1)
NOUT = 24                                     # outgoing strata per bin side
SUB = 8                                       # incident cell centres per bin side (quad87.SUB)
og = E.bin_grid(NOUT)                         # [NB, NOUT^2, 3]
ig = E.bin_grid(SUB)                          # [NB, 64, 3]   (cell centres, as quad87.cell_dirs)
t0 = time.time()
M = np.zeros((E.NB, E.NB, 3)); Mc = np.zeros_like(M)      # solid-angle mean; cos-weighted (flux-conserving) mean
alb_true = np.zeros((E.NB, SUB * SUB, 3)); alb_tab = np.zeros_like(alb_true)
for bi in range(E.NB):
    wi = ig[bi][:, None, None, :]             # [64, 1, 1, 3]
    f = E.fr(ST[1], wi, og[None])             # [64, NB, NOUT^2, 3]
    fb = f.mean(2)                            # (1/DOMEGA) int_bo f dv            [64, NB, 3]
    fc = (f * og[None, ..., 2:3]).mean(2)     # (1/DOMEGA) int_bo f cos dv        [64, NB, 3]
    M[bi] = fb.mean(0)
    Mc[bi] = fc.mean(0) / E.ZB[:, None]
    alb_true[bi] = fc.sum(1) * E.DOMEGA       # int f cos dv                       [64, 3]
    alb_tab[bi] = (fb * E.WT[None, :, None]).sum(1)   # sum_bo ZB DOMEGA fbar     [64, 3]
res['timing_s_m64'] = time.time() - t0
fn = os.path.join(ROOT, 'results', 'ecro_attr95a', 'ops95a.npz')
A4 = np.load(fn)['A4'].astype(np.float64)
ZZ = np.outer(E.ZB, E.ZB)[..., None]
res['M64_vs_stored'] = {'stored': os.path.relpath(fn, ROOT), 'stored_sha256_16': hashlib.sha256(open(fn, 'rb').read()).hexdigest()[:16],
                        'rel_l1': float(np.abs(M - A4).sum() / A4.sum()), 'flux_weighted_rel_l1': float((np.abs(M - A4) * ZZ).sum() / (A4 * ZZ).sum()),
                        'mass_ratio_mine_over_stored': float((M * ZZ).sum() / (A4 * ZZ).sum()),
                        'max_abs_over_max': float(np.abs(M - A4).max() / A4.max()),
                        'peak_entry_ratio_mean': float(np.mean([M[i, :, 1].max() / A4[i, :, 1].max() for i in range(E.NB)])),
                        'albedo_max_WT_mine': float((M * E.WT[None, :, None]).sum(1).max()), 'albedo_max_WT_stored': float((A4 * E.WT[None, :, None]).sum(1).max()),
                        'n_out_strata': NOUT * NOUT, 'note': 'mine: deterministic midpoint quadrature, numpy port; stored: MIS Monte Carlo (quad87.fbar), Dr.Jit closure'}
# ---------------------------------------------------------------- (3) single directions and flux conservation
iz = np.arange(E.NB) // E.NPHI
r = alb_tab / np.maximum(alb_true, 1e-30)      # discrete emitted flux / physical exitance, per incident cell centre
cosi = ig[..., 2]                              # [NB, 64]
res['flux_conservation_S1'] = {
    'definition': 'ratio = sum_bo ZB[bo] int_bo f(v,w) dv / int f(v,w) cos(v) dv for a single incident direction w (green channel)',
    'physical_albedo_max': float(alb_true.max()), 'physical_albedo_mean': float(alb_true.mean()),
    'table_albedo_single_direction_max': float(alb_tab.max()),
    'ratio_by_incident_cos': {f'{lo:.3f}-{hi:.3f}': float(r[..., 1][(cosi >= lo) & (cosi < hi)].mean())
                              for lo, hi in ((0, 0.03125), (0.03125, 0.0625), (0.0625, 0.125), (0.125, 0.1875), (0.1875, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0))},
    'ratio_min': float(r.min()), 'ratio_max': float(r.max()),
    'cos_weighted_bin_mean_operator_albedo_max': float((Mc * E.WT[None, :, None]).sum(1).max()),
    'cos_weighted_vs_solid_angle_mean_operator_rel_l1': float((np.abs(Mc - M) * ZZ).sum() / (M * ZZ).sum()),
    'ring0_row_flux_ratio_solid_angle_mean_over_true': float(((M * E.WT[None, :, None]).sum(1)[iz == 0]).sum() / ((Mc * E.WT[None, :, None]).sum(1)[iz == 0]).sum())}
# same for the other ER materials (floor B = 0, sphere S2 = 2): albedo_max of the single-direction discrete emission
for m in (0, 2):
    at = np.zeros((E.NB, SUB * SUB, 3)); ab = np.zeros_like(at)
    og12 = E.bin_grid(12)
    for bi in range(E.NB):
        f = E.fr(ST[m], ig[bi][:, None, None, :], og12[None])
        at[bi] = (f * og12[None, ..., 2:3]).mean(2).sum(1) * E.DOMEGA
        ab[bi] = (f.mean(2) * E.WT[None, :, None]).sum(1)
    res[f'flux_conservation_mat{m}'] = {'physical_albedo_max': float(at.max()), 'table_albedo_single_direction_max': float(ab.max()),
                                        'ratio_min': float((ab / np.maximum(at, 1e-30)).min()), 'ratio_max': float((ab / np.maximum(at, 1e-30)).max())}
np.savez_compressed(os.path.join(OUT, 'a2_m64_mine.npz'), M=M, Mc=Mc, alb_true=alb_true, alb_tab=alb_tab)
json.dump(res, open(os.path.join(OUT, 'a2_m64_er.json'), 'w'), indent=1)
print(json.dumps(res, indent=1))
