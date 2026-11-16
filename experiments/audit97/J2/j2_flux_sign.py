"""WL97 audit, Track J2 decisive check (numpy only, local). Imports Track A's numpy ER port (er_np) read-only; own quadrature.

(1) Flux convention, independent of Track A's log-azimuth grid: for single incident directions at the CK2 grazing cosines, dense
    uniform (z, phi) midpoint grid over the whole hemisphere -> per ring  ZB * int_ring f dOmega  (what a solid-angle-mean table
    emits when the flight multiplies by the exact cosine)  vs  int_ring f cos dOmega (physical).
(2) Sign / receiver dependence: far-field exitant intensity of the whole CK2 S1 panel, I(w) = sum_x A_x E_l(x) f(w_l(x), w) cos(w),
    exact vs patch x 64-bin tables (solid-angle mean 'sa', cos-weighted flux-conserving mean 'cos') on a fine direction grid.
    Reports the intensity-weighted share of directions where each table over / under delivers, and table/exact ratio quantiles.
Run: python experiments/audit97/J2/j2_flux_sign.py
"""
import json, math, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'experiments', 'audit97', 'A')); sys.path.insert(0, os.path.join(ROOT, 'experiments', 'ecro86'))
import er_np as E
import scene86 as S
OUT = os.path.join(ROOT, 'results', 'audit97', 'J2')
ck = json.load(open(os.path.join(ROOT, 'experiments', 'ecro_local93d', 'addendum_checker', 'scene_ck2.json')))
G = S.Geom(ck['geom']); ST = np.load(os.path.join(ROOT, 'experiments', 'ecro86', 'data', 'mats86.npz'))
S1 = ST['states'][1].astype(np.float64)
LPOS = np.array(ck['lights'][0]['pos'], np.float64); LRGB = np.array(ck['lights'][0]['rgb'], np.float64)
PANEL = np.arange(1088, 1130); T_, B_, N_ = G.PT[PANEL[0]], G.PB[PANEL[0]], G.PN[PANEL[0]]
res = {}
t0 = time.time()

# ---------------- (1) flux convention, single incident directions
NZ, NPH = 1024, 8192
z = (np.arange(NZ) + 0.5) / NZ
ph = (np.arange(NPH) + 0.5) / NPH * 2 * math.pi - math.pi
dw = (1.0 / NZ) * (2 * math.pi / NPH)


def ring_integrals(ci):
    wi = np.array([math.sqrt(1 - ci * ci), 0.0, ci])
    a = np.zeros((4, 3)); b = np.zeros((4, 3))
    for k0 in range(0, NZ, 32):
        Z, P = np.meshgrid(z[k0:k0 + 32], ph, indexing='ij'); R = np.sqrt(1 - Z * Z)
        v = np.stack([R * np.cos(P), R * np.sin(P), Z], -1).reshape(-1, 3)
        f = E.fr(S1, wi[None], v) * dw
        rg = np.clip((v[:, 2] * 4).astype(int), 0, 3)
        for r in range(4):
            m = rg == r
            a[r] += f[m].sum(0); b[r] += (f[m] * v[m, 2:3]).sum(0)
    return a, b


conv = {}
zb = (np.arange(4) + 0.5) / 4
for ci in (0.025, 0.047, 0.10, 0.154, 0.40, 0.80):
    a, b = ring_integrals(ci)
    emit_sa = (zb[:, None] * a).sum(0); emit_true = b.sum(0)
    conv[str(ci)] = {'albedo_true_rgb': emit_true.tolist(), 'albedo_sa_table_rgb': emit_sa.tolist(),
                     'ratio_sa_over_true_rgb': (emit_sa / emit_true).tolist(),
                     'ring_share_true': (b.sum(1) / b.sum()).tolist(),
                     'ring_ratio_sa_over_true': ((zb * a.sum(1)) / np.maximum(b.sum(1), 1e-30)).tolist()}
    print('flux', ci, [round(x, 4) for x in conv[str(ci)]['ratio_sa_over_true_rgb']],
          [round(x, 3) for x in conv[str(ci)]['ring_share_true']], round(time.time() - t0), flush=True)
res['flux_convention_single_direction'] = conv

# ---------------- (2) far-field panel intensity: exact vs binned tables
NSP = 4


def patch_points(p, n):
    a_ = (np.arange(n) + 0.5) / n; A_, B2 = np.meshgrid(a_, a_, indexing='ij')
    return G.PO[p][None] + A_.ravel()[:, None] * G.PE1[p][None] + B2.ravel()[:, None] * G.PE2[p][None]


xs = np.concatenate([patch_points(p, NSP) for p in PANEL]); pid = np.repeat(np.arange(len(PANEL)), NSP * NSP)
d = LPOS[None] - xs; r2 = (d * d).sum(1); w = d / np.sqrt(r2)[:, None]
wl = np.stack([w @ T_, w @ B_, w @ N_], 1); El = LRGB[0] * np.maximum(wl[:, 2], 0) / r2
area = G.PAREA[PANEL][pid] / (NSP * NSP)
nb = 48; dirs = E.bin_grid(nb)                                   # [64, nb*nb, 3]; uniform in solid angle within each bin
qa = np.zeros((len(PANEL), 64, 3)); qc = np.zeros_like(qa)
for i in range(len(xs)):
    f = E.fr(S1, wl[i][None, None], dirs) * El[i]
    qa[pid[i]] += f.mean(1) * area[i]
    qc[pid[i]] += (f * dirs[..., 2:3]).mean(1) / E.ZB[:, None] * area[i]   # int f cos / int cos  (mean cos in bin = ZB exactly)
print('tables', round(time.time() - t0), flush=True)
FZ, FP = 256, 1024
zf = (np.arange(FZ) + 0.5) / FZ; pf = (np.arange(FP) + 0.5) / FP * 2 * math.pi - math.pi
Zf, Pf = np.meshgrid(zf, pf, indexing='ij'); Rf = np.sqrt(1 - Zf * Zf)
vf = np.stack([Rf * np.cos(Pf), Rf * np.sin(Pf), Zf], -1).reshape(-1, 3); bf = E.bin_of(vf)
Iex = np.zeros((len(vf), 3))
for i in range(len(xs)):
    Iex += E.fr(S1, wl[i][None], vf) * (El[i] * area[i])
Iex *= vf[:, 2:3]
Isa = qa.sum(0)[bf] * vf[:, 2:3]; Ico = qc.sum(0)[bf] * vf[:, 2:3]
ex = Iex.sum(1); out = {}
wgt = ex / ex.sum()
for nm, It in (('sa', Isa), ('cos', Ico)):
    t = It.sum(1); ratio = t / np.maximum(ex, 1e-30)
    o = np.argsort(ratio); cw = np.cumsum(wgt[o])
    out[nm] = {'total_ratio': float(t.sum() / ex.sum()),
               'exact_weighted_share_dirs_table_over': float(wgt[ratio > 1].sum()),
               'exact_weighted_share_dirs_table_under': float(wgt[ratio < 1].sum()),
               'exact_weighted_ratio_quantiles_10_50_90': [float(ratio[o][np.searchsorted(cw, q)]) for q in (0.1, 0.5, 0.9)],
               'ratio_at_peak_exact_direction': float(ratio[np.argmax(ex)]),
               'abs_rel_l1_intensity': float(np.abs(t - ex).sum() / ex.sum())}
    print('farfield', nm, out[nm], flush=True)
out['peak_exact_direction_local_xyz'] = vf[np.argmax(ex)].tolist()
BOXA = np.arange(1130, 1184); ca = (G.PO[BOXA] + 0.5 * G.PE1[BOXA] + 0.5 * G.PE2[BOXA]).mean(0)
cp = (G.PO[PANEL] + 0.5 * G.PE1[PANEL] + 0.5 * G.PE2[PANEL]).mean(0); dd = ca - cp
out['boxA_dist_m'] = float(np.linalg.norm(dd)); dd /= np.linalg.norm(dd)
out['boxA_dir_from_panel_centre_local_xyz'] = [float(dd @ T_), float(dd @ B_), float(dd @ N_)]
out['panel_sqrt_area_m'] = float(np.sqrt(G.PAREA[PANEL].sum()))
i_bA = E.bin_of(np.array(out['boxA_dir_from_panel_centre_local_xyz'])[None])[0]
out['boxA_centre_bin'] = int(i_bA)
res['farfield_panel_intensity'] = out
res['settings'] = {'NZ': NZ, 'NPH': NPH, 'NSP': NSP, 'nb': nb, 'FZ': FZ, 'FP': FP, 'seconds': time.time() - t0}
json.dump(res, open(os.path.join(OUT, 'j2_flux_sign.json'), 'w'), indent=1)
print('done', round(time.time() - t0), 's')
