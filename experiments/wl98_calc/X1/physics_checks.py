"""WL98 critic X1 scratch #2 (design input only, not evidence). numpy only.
(1) Panel S1 (scene86 fixture, 6 x 7 patches of 0.2 m) under CK2 / L1 / L2 point lights: principal angular extents of the
    mirror-direction set across each patch versus the S1 lobe half-width at that incidence (cone-equivalent of the D2
    half-max solid angle, interpolated in log cos_i). Counts how many separable (anchor x direction) lobes a per-element
    lobe state needs to keep the reflected point-light image at lobe resolution; an image-source (sheared) lobe needs 1.
(2) Flux-density lobe normalised on the clipped hemisphere and read as radiance L = Phi p / cos(theta): divergence of
    int L dOmega and of the second moment near the horizon (finite flux, unbounded radiance).
(3) Receiver-anchored gather: per-draw column sum (delivered / emitted) for a sharp sender, Poisson model anchored on
    WL96's 0.46 expected glint rays per receiver patch per draw: P(per-draw delivered flux > emitted flux).
(4) LER-Delta splitting: iteration factor of b = (I+H)[e + (M-B) b] for a scalar environment gain t and blocked share beta,
    correct condition rho((I+H)B) < 1 versus the condition written in D3 (rho(H B) < 1).
"""
import json, math, os
import numpy as np
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..')
fx = json.load(open(os.path.join(R, 'experiments/ecro86/protocol/ecro86.json')))['fixture']
ck = json.load(open(os.path.join(R, 'experiments/ecro_local93d/addendum_checker/scene_ck2.json')))
d2 = json.load(open(os.path.join(R, 'results/wl98_calc/D2/angular_capacity.json')))
cosi_tab = np.array([0.025, 0.05, 0.1, 0.25, 0.4, 0.8]); hm_tab = np.array([4.2e-4, 4.7e-4, 1.21e-3, 3.34e-3, 1.073e-2, 2.833e-2])  # D2 log
def lobe_half_deg(ci):
    hm = np.exp(np.interp(np.log(np.clip(ci, 0.025, 0.8)), np.log(cosi_tab), np.log(hm_tab)))
    return np.degrees(np.sqrt(hm / np.pi))
pc = np.array(fx['panel']['center']); hy, hz = fx['panel']['half']
lights = {'CK2': np.array(ck['lights'][0]['pos'])}
for k, v in fx['light_sets'].items():
    if k in ('L1', 'L2'):
        lights[k] = np.array(v[0]['pos'])
rng = np.random.default_rng(98111)
out = {}
for name, L in lights.items():
    img = L.copy(); img[0] = 2 * pc[0] - L[0]
    rows = []
    for iy in range(6):
        for iz in range(7):
            y0 = pc[1] - hy + 2 * hy * iy / 6; z0 = pc[2] - hz + 2 * hz * iz / 7
            u = rng.random((8192, 2))
            P = np.stack([np.full(8192, pc[0]), y0 + u[:, 0] * 2 * hy / 6, z0 + u[:, 1] * 2 * hz / 7], 1)
            d = L - P; r2 = (d ** 2).sum(1); d /= np.sqrt(r2)[:, None]
            ci = d[:, 0]
            if np.median(ci) <= 0:
                continue
            E = np.clip(ci, 0, None) / r2
            o = P - img; o /= np.linalg.norm(o, axis=1)[:, None]
            m = (E[:, None] * o).sum(0); m /= np.linalg.norm(m)
            a1 = np.cross(m, [1.0, 0, 0]); a1 /= np.linalg.norm(a1); a2 = np.cross(m, a1)
            X = np.degrees(np.stack([o @ a1, o @ a2], 1))           # gnomonic-ish tangent coordinates (deg)
            w = E / E.sum(); mu = (w[:, None] * X).sum(0); C = ((X - mu).T * w) @ (X - mu)
            ev = np.sqrt(np.clip(np.linalg.eigvalsh(C), 0, None))[::-1]   # principal SD (deg)
            ext = 2 * math.sqrt(3) * ev                                     # full extent of a uniform spread with that SD
            h = float(lobe_half_deg(np.median(ci)))
            nlob = max(1.0, ext[0] / (2 * h)) * max(1.0, ext[1] / (2 * h))
            rows.append(dict(patch=(iy, iz), cos_i_med=float(np.median(ci)), E=float(E.mean()), ext_major_deg=float(ext[0]),
                             ext_minor_deg=float(ext[1]), lobe_half_deg=h, sep_lobes_needed=float(nlob),
                             out_elev_deg=float(np.degrees(np.arcsin(np.clip(m[0], -1, 1))))))
    Es = np.array([r['E'] for r in rows]); Es = Es / Es.sum()
    for r, e in zip(rows, Es):
        r['E_share'] = float(e)
    srt = sorted(rows, key=lambda r: -r['E_share'])
    wmed = lambda key: float(np.average([r[key] for r in rows], weights=Es))
    out[name] = dict(light=L.tolist(), n_lit_patches=len(rows), top4=srt[:4],
                     Eweighted_mean=dict(cos_i=wmed('cos_i_med'), ext_major_deg=wmed('ext_major_deg'), ext_minor_deg=wmed('ext_minor_deg'),
                                         lobe_half_deg=wmed('lobe_half_deg'), sep_lobes_needed=wmed('sep_lobes_needed')),
                     share_E_needing_gt8_lobes=float(sum(e for r, e in zip(rows, Es) if r['sep_lobes_needed'] > 8)))
# (2) flux-density lobe at the horizon
sig = math.radians(1.0)
res2 = {}
for e0_deg in (1.0, 3.0, 6.0):
    e0 = math.radians(e0_deg)
    for eps_deg in (0.1, 0.01, 0.001):
        e = np.linspace(math.radians(eps_deg), math.radians(30), 400001)
        q = np.exp(-0.5 * ((e - e0) / sig) ** 2)                    # flux density per solid angle (elevation profile; azimuth factor dropped)
        de = e[1] - e[0]
        Z = (q * np.cos(e)).sum() * de                             # solid-angle normalisation (d omega = cos(elev) d elev d phi)
        Lr = q / np.sin(e) / Z                                     # radiance up to the azimuth factor
        res2[f'e0={e0_deg},eps={eps_deg}'] = dict(int_L_domega=float((Lr * np.cos(e)).sum() * de),
                                                     int_L2_domega=float((Lr ** 2 * np.cos(e)).sum() * de),
                                                     L_ratio_lobe_edges=float((1 / math.sin(max(e0 - sig, 1e-4))) / (1 / math.sin(e0 + sig))))
# (3) per-draw passivity of a receiver-anchored gather
from math import exp, factorial
def pois_sf(lam, k):   # P(X >= k)
    return 1.0 - sum(exp(-lam) * lam ** j / factorial(j) for j in range(int(k)))
res3 = {}
for nrec in (1, 3, 6):
    lam = 0.46 * nrec
    for f in (0.3, 0.5, 0.8):
        kmin = math.floor(lam / f) + 1        # delivered = hits * (f * Phi / lam) > Phi  <=> hits > lam / f
        res3[f'n_glint_receivers={nrec},expected_fraction={f}'] = dict(lam=lam, P_draw_exceeds_emitted=pois_sf(lam, kmin), P_zero_hits=exp(-lam))
# (4) LER-Delta splitting
res4 = {}
for t in (0.4, 0.5, 0.6, 0.7, 0.8):
    res4[f't={t}'] = dict(beta_max_correct=(1 - t) / t, beta_max_D3_written=(1 - t) / t ** 2,
                          factor_at_beta_0p5=0.5 * t / (1 - t))
json.dump(dict(spread=out, horizon=res2, gather_passivity=res3, ler_delta=res4),
          open(os.path.join(R, 'results/wl98_calc/X1/physics_checks.json'), 'w'), indent=1)
for k, v in out.items():
    print(k, v['n_lit_patches'], {a: round(b, 2) for a, b in v['Eweighted_mean'].items()}, 'E share >8 lobes', round(v['share_E_needing_gt8_lobes'], 3))
for k, v in res2.items():
    print(k, {a: round(b, 3) for a, b in v.items()})
for k, v in res3.items():
    print(k, {a: round(b, 3) for a, b in v.items()})
print(res4)
