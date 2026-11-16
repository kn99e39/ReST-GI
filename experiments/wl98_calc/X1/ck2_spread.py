"""WL98 critic X1 scratch (design input only, not evidence).
(1) CK2 geometry: across each 0.2 m S1 panel patch, how far does the mirror (specular) direction of the single CK2
    point light move? Compared with the S1 grazing lobe width, this is how many direction-separable lobes a
    per-element outgoing lobe state (FLS) would need, versus a sheared / image-source parameterisation.
(2) Sphere S2 patches: normal span inside one 8x4 patch (patch-frame vs pointwise-frame measure mismatch).
(3) LER-Delta splitting iteration: scalar convergence bound beta < (1-t)/t.
Inputs: experiments/ecro86/protocol/ecro86.json fixture (panel), experiments/ecro_local93d/addendum_checker/scene_ck2.json (light).
"""
import json, math, os
import numpy as np
R = os.path.join(os.path.dirname(__file__), '..', '..', '..')
fx = json.load(open(os.path.join(R, 'experiments/ecro86/protocol/ecro86.json')))['fixture']
ck = json.load(open(os.path.join(R, 'experiments/ecro_local93d/addendum_checker/scene_ck2.json')))
pc = np.array(fx['panel']['center']); hy, hz = fx['panel']['half']
L = np.array(ck['lights'][0]['pos'])
img = L.copy(); img[0] = 2 * pc[0] - L[0]           # image source across plane x = pc.x (normal +x)
NY, NZ = 6, 7                                         # scene86 panel split (6 x 7)
rng = np.random.default_rng(98101)
rows = []
for iy in range(NY):
    for iz in range(NZ):
        y0 = pc[1] - hy + 2 * hy * iy / NY; z0 = pc[2] - hz + 2 * hz * iz / NZ
        u = rng.random((4096, 2))
        P = np.stack([np.full(4096, pc[0]), y0 + u[:, 0] * 2 * hy / NY, z0 + u[:, 1] * 2 * hz / NZ], 1)
        d_in = L - P; r2 = (d_in ** 2).sum(1); d_in /= np.sqrt(r2)[:, None]
        cos_i = d_in[:, 0]
        E = cos_i / r2                                  # irradiance weight (unoccluded)
        refl = P - img; refl /= np.linalg.norm(refl, axis=1)[:, None]   # specular outgoing direction
        m = (E[:, None] * refl).sum(0); m /= np.linalg.norm(m)
        ang = np.degrees(np.arccos(np.clip(refl @ m, -1, 1)))
        rows.append(dict(patch=(iy, iz), cos_i_med=float(np.median(cos_i)), cos_i_min=float(cos_i.min()), cos_i_max=float(cos_i.max()),
                         spec_spread_deg_p95=float(np.percentile(ang, 95)), spec_spread_deg_max=float(ang.max()),
                         outgoing_elev_deg_med=float(np.degrees(np.arcsin(np.clip(np.median(refl[:, 0]), -1, 1)))),
                         E_share=float(E.mean())))
Es = np.array([r['E_share'] for r in rows]); Es /= Es.sum()
for r, e in zip(rows, Es):
    r['E_share'] = float(e)
top = sorted(rows, key=lambda r: -r['E_share'])[:8]
# S1 grazing half-max solid angle 4.2e-4 sr at cos_i 0.025-0.05 (D2 scratch); equivalent cone half-angle
hm = 4.2e-4; half_deg = math.degrees(math.sqrt(hm / math.pi))
sph = []
for iz in range(4):
    z0, z1 = iz / 4 * 2 - 1, (iz + 1) / 4 * 2 - 1
    th0, th1 = math.degrees(math.acos(z1)), math.degrees(math.acos(z0))
    sph.append(dict(band=iz, polar_span_deg=th1 - th0, azimuth_span_deg=45.0))
ler = {f't={t}': (1 - t) / t for t in (0.5, 0.6, 0.7, 0.8, 0.9)}
out = dict(top_patches_by_irradiance=top, s1_grazing_halfmax_sr=hm, s1_equiv_cone_half_angle_deg=half_deg,
           ratio_spread_p95_to_lobe_halfangle=[r['spec_spread_deg_p95'] / half_deg for r in top],
           sphere_patch_normal_span=sph, ler_delta_scalar_beta_max=ler)
os.makedirs(os.path.join(R, 'results/wl98_calc/X1'), exist_ok=True)
json.dump(out, open(os.path.join(R, 'results/wl98_calc/X1/ck2_spread.json'), 'w'), indent=1)
for r in top:
    print(r)
print('lobe half-angle deg', half_deg)
print('ratios', [round(x, 1) for x in out['ratio_spread_p95_to_lobe_halfangle']])
print(sph); print(ler)
