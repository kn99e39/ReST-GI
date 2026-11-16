"""WL97 audit, Track A, check A4 (numpy only, local): the grazing rows of the M64 teacher with a quadrature that resolves the lobe.

a2_m64_er.py used a midpoint rule (24 x 24 strata per outgoing bin) whose nodes coincide with the mirror direction of every incident
cell centre; at grazing incidence the S1 lobe is ~1e-3 rad wide in azimuth, so that rule is NOT reliable for the ring0 -> ring0 block
(all other blocks agree with the stored operator to ~0.2 %). Here: log-azimuth grid centred on the mirror azimuth (as a3), NZQ x NPQ.

For the incident cell centres of selected incident bins (quad87.cell_dirs layout, SUB = 8):
  fb[cell, bo] = (1 / DOMEGA) int_bo f(v, w) dv            (quad87.fbar definition)
  fc[cell]     = int f(v, w) cos(v) dv                      (physical directional albedo)
  row[bo]      = mean_cell fb                               (quad87.enc_matrix = M64 row)
  alb_WT       = sum_bo row[bo] ZB[bo] DOMEGA               (a95 'albedo' of the row)
Compared with the stored operator results/ecro_attr95a/ops95a.npz:A4.
Run:  python experiments/audit97/A/a4_m64_ring0.py [NZQ NPQ]
"""
import json, math, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
sys.path.insert(0, HERE)
import er_np as E

OUT = os.path.join(ROOT, 'results', 'audit97', 'A')
NZQ = int(sys.argv[1]) if len(sys.argv) > 1 else 256
NPQ = int(sys.argv[2]) if len(sys.argv) > 2 else 1536
BINS = [int(x) for x in sys.argv[3].split(',')] if len(sys.argv) > 3 else [0, 5, 11, 16 + 3, 48 + 7]
ST = np.load(os.path.join(ROOT, 'experiments', 'ecro86', 'data', 'mats86.npz'))['states'].astype(np.float64)
A4 = np.load(os.path.join(ROOT, 'results', 'ecro_attr95a', 'ops95a.npz'))['A4'].astype(np.float64)
S_LOG = 3e-4; A_LOG = math.asinh(math.pi / S_LOG)
# z grid: uniform + refinement near 0 (z = t^2 on the first quarter is not needed: f is bounded in z; uniform midpoint)
zq = (np.arange(NZQ) + 0.5) / NZQ
uq = (np.arange(NPQ) + 0.5) / NPQ
dphi = S_LOG * np.sinh(A_LOG * (2 * uq - 1)); wphi = 2 * A_LOG * S_LOG * np.cosh(A_LOG * (2 * uq - 1)) / NPQ
Zg, Dg = np.meshgrid(zq, dphi, indexing='ij'); Wg = np.meshgrid(np.full(NZQ, 1.0 / NZQ), wphi, indexing='ij'); Wg = (Wg[0] * Wg[1]).ravel()
Rg = np.sqrt(1 - Zg * Zg).ravel(); Zr = Zg.ravel(); Dr_ = Dg.ravel()
SUB = 8


def hemi_bins(st, w):
    phi = math.atan2(-w[1], -w[0]) + Dr_
    v = np.stack([Rg * np.cos(phi), Rg * np.sin(phi), Zr], 1)
    f = E.fr(st, w[None], v) * Wg[:, None]
    bn = E.bin_of(v)
    a = np.stack([np.bincount(bn, weights=f[:, c], minlength=E.NB) for c in range(3)], 1)
    b = (f * Zr[:, None]).sum(0)
    return a, b


def cells(bn):
    sz, sp = np.meshgrid(np.arange(SUB), np.arange(SUB), indexing='ij')
    z = (bn // E.NPHI + (sz.ravel() + 0.5) / SUB) / E.NZ
    phi = (bn % E.NPHI + (sp.ravel() + 0.5) / SUB) * (2 * math.pi / E.NPHI) - math.pi
    r = np.sqrt(1 - z * z)
    return np.stack([r * np.cos(phi), r * np.sin(phi), z], 1)


res = {'NZQ': NZQ, 'NPQ': NPQ, 'material': 'S1 (mats86 state 1)', 'rows': {}}
iz_out = np.arange(E.NB) // E.NPHI
t0 = time.time()
for bn in BINS:
    cd = cells(bn)
    fb = np.zeros((SUB * SUB, E.NB, 3)); fc = np.zeros((SUB * SUB, 3))
    for k in range(SUB * SUB):
        a, b = hemi_bins(ST[1], cd[k])
        fb[k] = a / E.DOMEGA; fc[k] = b
    row = fb.mean(0)                                           # M64 row (solid-angle mean over incident cell centres)
    st = A4[bn]
    alb_cell_tab = (fb * E.WT[None, :, None]).sum(1)           # discrete emitted flux per unit irradiance, per incident cell
    o = {'incident_ring': int(bn // E.NPHI),
         'row_sum_ratio_mine_over_stored_rgb': (row.sum(0) / st.sum(0)).tolist(),
         'block_ratio_mine_over_stored_green_by_out_ring': [float(row[iz_out == r, 1].sum() / st[iz_out == r, 1].sum()) for r in range(4)],
         'row_rel_l1_green': float(np.abs(row[:, 1] - st[:, 1]).sum() / st[:, 1].sum()),
         'albedo_WT_mine_rgb': (row * E.WT[:, None]).sum(0).tolist(), 'albedo_WT_stored_rgb': (st * E.WT[:, None]).sum(0).tolist(),
         'physical_albedo_cell_mean_rgb': fc.mean(0).tolist(), 'physical_albedo_cell_max_rgb': fc.max(0).tolist(),
         'physical_albedo_cos_weighted_mean_rgb': ((fc * cd[:, 2:3]).sum(0) / cd[:, 2].sum()).tolist(),
         'discrete_flux_over_physical_by_incident_z_green': {f'{z:.4f}': float(alb_cell_tab[np.isclose(cd[:, 2], z), 1].mean() / fc[np.isclose(cd[:, 2], z), 1].mean())
                                                             for z in np.unique(cd[:, 2])},
         'discrete_albedo_single_cell_max_green': float(alb_cell_tab[:, 1].max())}
    res['rows'][str(bn)] = o
    print(bn, json.dumps(o), round(time.time() - t0, 1), flush=True)
json.dump(res, open(os.path.join(OUT, f'a4_m64_ring0_z{NZQ}_p{NPQ}.json'), 'w'), indent=1)
