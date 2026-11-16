"""WL97 audit, Track A, check A3 (numpy only, local): independent deterministic recomputation of the CK2 'none' S1-mediated exactly-2-event
delivery to box A (WL95A quantity: light -> S1 panel -> box A patch -> exitance), with no path tracer, no Dr.Jit, no state rays.

  E(y) = int_panel L(x -> y) cos_y cos_x / r^2 dA_x        (box A is convex and nothing lies between it and the panel; the light sees
                                                            the whole panel: it sits 5 cm in front of the panel plane, box A is behind it)
  exact     L(x -> y) = E_l(x) f(w_l(x), w_xy)                                (continuous direction and hit point = WL95A arm X1)
  binned    L(x -> y) = qbar[patch(x), bin(w_xy)]                             (patch x 64-bin sender table in the limit of infinite state rays)
     qbar_sa   = patch mean of E_l (1/DOMEGA) int_bin f dv                    (solid-angle bin mean = quad87.qtable / pt86.field convention)
     qbar_cos  = patch mean of E_l int_bin f cos dv / int_bin cos dv          (flux-conserving bin mean; audit counterfactual, not a repair)
     q_M64     = sum_bi s[patch, bi] M64[bi, bin]  with the stored operator   (ops95a.npz:A4) and s from the same light
Reference: results/ecro_attr95a/deliv.npz ref_a / ref_b (W_T exitance of the path-traced CKF_none e2_p halves).
Run:  python experiments/audit97/A/a3_ck2_delivery.py [NS NP NR]
"""
import hashlib, json, math, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(ROOT, 'experiments', 'ecro86'))
import er_np as E
import scene86 as S

OUT = os.path.join(ROOT, 'results', 'audit97', 'A')
NS = int(sys.argv[1]) if len(sys.argv) > 1 else 6        # sender-table points per panel patch side
NP = int(sys.argv[2]) if len(sys.argv) > 2 else 48       # panel quadrature points per patch side (delivery)
NR = int(sys.argv[3]) if len(sys.argv) > 3 else 3        # receiver points per box-A patch side
TAG = f'ns{NS}_np{NP}_nr{NR}'
ck = json.load(open(os.path.join(ROOT, 'experiments', 'ecro_local93d', 'addendum_checker', 'scene_ck2.json')))
G = S.Geom(ck['geom'])
ST = np.load(os.path.join(ROOT, 'experiments', 'ecro86', 'data', 'mats86.npz'))
S1ST = ST['states'][1].astype(np.float64)
RHO_A = float(ST['albedo'][4])
LPOS = np.array(ck['lights'][0]['pos'], np.float64); LRGB = np.array(ck['lights'][0]['rgb'], np.float64)
PANEL = np.arange(1088, 1130); BOXA = np.arange(1130, 1184)
assert (G.PMAT[PANEL] == 1).all() and (G.PMAT[BOXA] == 4).all() and LRGB.min() == LRGB.max()
T, B, N = G.PT[PANEL[0]], G.PB[PANEL[0]], G.PN[PANEL[0]]


def patch_points(p, n):
    a = (np.arange(n) + 0.5) / n
    A_, B_ = np.meshgrid(a, a, indexing='ij')
    return G.PO[p][None] + A_.ravel()[:, None] * G.PE1[p][None] + B_.ravel()[:, None] * G.PE2[p][None]


def light_at(x):
    d = LPOS[None] - x
    r2 = (d * d).sum(1); w = d / np.sqrt(r2)[:, None]
    wl = np.stack([w @ T, w @ B, w @ N], 1)
    return wl, LRGB[0] * np.maximum(wl[:, 2], 0) / r2          # local light direction, irradiance (per channel; white light)


# ---------------------------------------------------------------- sender tables
NZQ, NPQ = 256, 1536
S_LOG = 3e-4; A_LOG = math.asinh(math.pi / S_LOG)
zq = (np.arange(NZQ) + 0.5) / NZQ
uq = (np.arange(NPQ) + 0.5) / NPQ
dphi = S_LOG * np.sinh(A_LOG * (2 * uq - 1)); wphi = 2 * A_LOG * S_LOG * np.cosh(A_LOG * (2 * uq - 1)) / NPQ
Zg, Dg = np.meshgrid(zq, dphi, indexing='ij'); Wg = np.meshgrid(np.full(NZQ, 1.0 / NZQ), wphi, indexing='ij'); Wg = (Wg[0] * Wg[1]).ravel()
Rg = np.sqrt(1 - Zg * Zg).ravel(); Zr = Zg.ravel(); Dr_ = Dg.ravel()


def hemi_bins(w):
    """per-bin int f dv and int f cos dv of f(w, .) (log-azimuth grid centred on the mirror azimuth) -> [NB, 3] x 2."""
    phi = math.atan2(-w[1], -w[0]) + Dr_
    v = np.stack([Rg * np.cos(phi), Rg * np.sin(phi), Zr], 1)
    f = E.fr(S1ST, w[None], v) * Wg[:, None]
    bn = E.bin_of(v)
    a = np.stack([np.bincount(bn, weights=f[:, c], minlength=E.NB) for c in range(3)], 1)
    b = np.stack([np.bincount(bn, weights=f[:, c] * Zr, minlength=E.NB) for c in range(3)], 1)
    return a, b


t0 = time.time()
q_sa = np.zeros((len(PANEL), E.NB, 3)); q_cos = np.zeros_like(q_sa); flux_true = np.zeros((len(PANEL), 3)); irr = np.zeros(len(PANEL))
cos_i = []
for j, p in enumerate(PANEL):
    wl, El = light_at(patch_points(p, NS))
    for k in range(len(wl)):
        a, b = hemi_bins(wl[k])
        q_sa[j] += El[k] * a / E.DOMEGA; q_cos[j] += El[k] * b / E.WT[:, None]; flux_true[j] += El[k] * b.sum(0)
    q_sa[j] /= len(wl); q_cos[j] /= len(wl); flux_true[j] /= len(wl); irr[j] = El.mean(); cos_i.append(wl[:, 2])
cos_i = np.concatenate(cos_i)
t_q = time.time() - t0
A4 = np.load(os.path.join(ROOT, 'results', 'ecro_attr95a', 'ops95a.npz'))['A4'].astype(np.float64)
s_enc = np.zeros((len(PANEL), E.NB))
for j, p in enumerate(PANEL):
    wl, El = light_at(patch_points(p, 32))
    s_enc[j] = np.bincount(E.bin_of(wl), weights=El, minlength=E.NB) / len(wl)
q_m64 = np.einsum('nm,mdc->ndc', s_enc, A4)
flux = lambda q: (q * E.WT[None, :, None]).sum((1, 2))
sender = {'irradiance_weighted_cos_i_mean': float((cos_i * 1).mean()), 'cos_i_min': float(cos_i.min()), 'cos_i_max': float(cos_i.max()),
          'incident_ring_share_of_irradiance': [float(s_enc[:, (np.arange(64) // 16) == r].sum() / s_enc.sum()) for r in range(4)],
          'emitted_flux_ratio_table_over_true': {'qbar_sa': float(flux(q_sa).sum() / flux_true.sum()), 'qbar_cos': float(flux(q_cos).sum() / flux_true.sum()),
                                                 'q_M64_stored': float(flux(q_m64).sum() / flux_true.sum())},
          'physical_albedo_of_panel_for_this_light_rgb': (flux_true.sum(0) / irr.sum()).tolist(),
          'q_M64_vs_qbar_sa_field_rel_l1_WT': float((np.abs(q_m64 - q_sa) * E.WT[None, :, None]).sum() / (q_sa * E.WT[None, :, None]).sum()),
          'seconds': t_q}
print('sender', json.dumps(sender), flush=True)

# ---------------------------------------------------------------- delivery to box A
t0 = time.time()
xp = np.concatenate([patch_points(p, NP) for p in PANEL]); pidx = np.repeat(np.arange(len(PANEL)), NP * NP)
dA = np.repeat(G.PAREA[PANEL] / (NP * NP), NP * NP)
wl_p, El_p = light_at(xp)
arms = ('exact', 'qbar_sa', 'qbar_cos', 'q_M64')
Eirr = {a: np.zeros((len(BOXA), 3)) for a in arms}
ring_share = np.zeros((len(BOXA), 4))                     # share of the exact irradiance arriving from sender rings 0..3
tabs = {'qbar_sa': q_sa, 'qbar_cos': q_cos, 'q_M64': q_m64}
for i, pr in enumerate(BOXA):
    ny = G.PN[pr]
    for y in patch_points(pr, NR):
        d = y[None] - xp
        r2 = (d * d).sum(1); r = np.sqrt(r2); w = d / r[:, None]
        cx = w @ N; cy = -(w @ ny)
        vis = (cx > 1e-6) & (cy > 1e-6)
        if not vis.any():
            continue
        g = np.where(vis, cx * cy / r2 * dA, 0.0)
        wo = np.stack([w @ T, w @ B, np.maximum(cx, 1e-9)], 1)
        Lx = E.fr(S1ST, wl_p[vis], wo[vis]) * El_p[vis][:, None]
        Eirr['exact'][i] += (Lx * g[vis][:, None]).sum(0)
        bn = E.bin_of(wo)
        rs = np.bincount(bn[vis] // 16, weights=Lx.sum(1) * g[vis], minlength=4)
        ring_share[i] += rs
        for a, q in tabs.items():
            Eirr[a][i] += (q[pidx[vis], bn[vis]] * g[vis][:, None]).sum(0)
for a in arms:
    Eirr[a] /= NR * NR
ring_share /= np.maximum(ring_share.sum(), 1e-30)
t_d = time.time() - t0
ex = {a: RHO_A * Eirr[a].sum(1) for a in arms}             # exitance per box-A patch, channels summed (WL95A quantity)
z = np.load(os.path.join(ROOT, 'results', 'ecro_attr95a', 'deliv.npz'))
ra, rb = z['ref_a'][BOXA], z['ref_b'][BOXA]; ref = 0.5 * (ra + rb)
pf = np.arange(9)
met = {}
for a in arms:
    met[a] = {'ratio_boxA_vs_pathtraced_ref': float(ex[a].sum() / ref.sum()), 'abs_rel_l1_vs_ref': float(np.abs(ex[a] - ref).sum() / ref.sum()),
              'corr_vs_ref': float(np.corrcoef(ex[a], ref)[0, 1]), 'ratio_panel_face_vs_ref': float(ex[a][pf].sum() / ref[pf].sum()),
              'ratio_vs_my_exact': float(ex[a].sum() / ex['exact'].sum()), 'abs_rel_l1_vs_my_exact': float(np.abs(ex[a] - ex['exact']).sum() / ex['exact'].sum())}
res = {'tag': TAG, 'NS': NS, 'NP': NP, 'NR': NR, 'sender': sender, 'delivery': met,
       'ref_half_noise_boxA': float(0.5 * np.abs(ra - rb).sum() / ref.sum()), 'ref_total_boxA': float(ref.sum()), 'my_exact_total_boxA': float(ex['exact'].sum()),
       'exact_irradiance_share_by_sender_ring': ring_share.sum(0).tolist(),
       'wl95a_pooled_for_comparison': {'X1': [1.0224, 0.1106], 'X1c': [1.0371, 0.2677], 'q256': [1.4221, 0.4331], 'M64_spool': [1.4949, 0.4996], 'A4_M64_s16': [1.5441, 0.5464],
                                       'format': '[ratio, abs_rel_l1] on box A, 64 pooled relation draws (results/ecro_attr95a/an.json)'},
       'seconds_delivery': t_d,
       'inputs': {'deliv_sha256_16': hashlib.sha256(open(os.path.join(ROOT, 'results', 'ecro_attr95a', 'deliv.npz'), 'rb').read()).hexdigest()[:16]}}
np.savez_compressed(os.path.join(OUT, f'a3_ck2_delivery_{TAG}.npz'), q_sa=q_sa, q_cos=q_cos, q_m64=q_m64, flux_true=flux_true, ref=ref,
                    **{f'ex_{a}': ex[a] for a in arms})
json.dump(res, open(os.path.join(OUT, f'a3_ck2_delivery_{TAG}.json'), 'w'), indent=1)
print(json.dumps(res, indent=1))
