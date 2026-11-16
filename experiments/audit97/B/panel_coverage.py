"""WL97 audit track B: geometry-only coverage of CK2-like sharp-panel transport in the ECRO training populations.

  python panel_coverage.py <population_regen.json> <out_json>

Independent numpy re-implementation (no renderer, no stored field): for every training lighting, the direct irradiance of the S1 panel
(42 patches x the 16 deployed source points, rng 86001 as in pt86.Ctx; shadowing by boxes A / B and the sphere), its split into the 4
incident cos(theta) rings of the 64-bin interface, and an ideal-mirror proxy of light -> panel -> box delivery (mirror ray of each lit
panel point, first hit against box A / box B / sphere). The same quantities for CK2. The mirror proxy ignores the S1 lobe width
(alpha 0.026-0.075) and Fresnel: it is a coverage proxy, not radiometry.
"""
import json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'experiments', 'ecro86'))
import scene86 as S  # noqa: E402

FX = S.FX
pop = json.load(open(sys.argv[1]))
CK2 = json.load(open(os.path.join(ROOT, 'experiments', 'ecro_local93d', 'addendum_checker', 'scene_ck2.json')))
ex92a = {e[0] for e in json.load(open(os.path.join(ROOT, 'results', 'ecro_own92a', 'results', 'exclusion.json')))['excluded']['train']}
sel94 = {r['file'] for r in json.load(open(os.path.join(ROOT, 'results', 'ecro_mat94', 'select94.json')))['splits']['train']['files']}

# deployed source-point offsets (pt86.Ctx.__init__: rs = default_rng(86001); us = rs.random((4,1,K)); u_src = rs.random((2,1,PSRC)))
rs = np.random.default_rng(S.REP['table_seed'])
_ = rs.random((4, 1, S.K), np.float32)
USRC = rs.random((2, 1, S.PSRC), np.float32)[:, 0, :].astype(np.float64)      # [2, 16]
SC = np.array(FX['sphere']['center'], np.float64)
SR = float(FX['sphere']['radius'])
PN = np.array([1.0, 0.0, 0.0])


def panel_points(G):
    pid = np.where(G.PMAT == 1)[0]
    assert len(pid) == 42 and pid[0] == 1088
    p = G.PO[pid][:, None, :] + USRC[0][None, :, None] * G.PE1[pid][:, None, :] + USRC[1][None, :, None] * G.PE2[pid][:, None, :]
    return p.reshape(-1, 3)                                                       # [672, 3]


def ray_obb(o, d, obb):
    c, ax, h = obb
    lo = (o - c) @ ax.T
    ld = d @ ax.T
    ld = np.where(np.abs(ld) < 1e-12, 1e-12, ld)
    t0 = (-h - lo) / ld
    t1 = (h - lo) / ld
    tmin = np.minimum(t0, t1).max(1)
    tmax = np.maximum(t0, t1).min(1)
    hit = (tmax >= np.maximum(tmin, 1e-6))
    return np.where(hit, np.maximum(tmin, 1e-6), np.inf)


def ray_sphere(o, d):
    oc = o - SC
    b = (oc * d).sum(1)
    c = (oc * oc).sum(1) - SR * SR
    disc = b * b - c
    t = -b - np.sqrt(np.maximum(disc, 0))
    return np.where((disc > 0) & (t > 1e-6), t, np.inf)


def analyse(geom, light_pos, intensity=30.0):
    G = S.Geom(geom)
    p = panel_points(G)
    obbs = [G.obb(b, pad=0.0) for b in 'AB']
    L = np.array(light_pos, np.float64)
    dl = L - p
    d2 = (dl * dl).sum(1)
    dist = np.sqrt(d2)
    wl = dl / dist[:, None]
    cs = wl @ PN
    on = cs > 1e-6
    tocc = np.minimum(np.minimum(ray_obb(p, wl, obbs[0]), ray_obb(p, wl, obbs[1])), ray_sphere(p, wl))
    vis = on & (tocc >= dist)
    e = np.where(vis, intensity * cs / d2, 0.0)                                   # per point; patch mean = sum / 16
    ring = np.clip(np.floor(cs * S.NZ), 0, S.NZ - 1).astype(int)
    ering = np.array([e[ring == k].sum() for k in range(4)]) / S.PSRC             # sum over patches of patch-mean irradiance
    # ideal-mirror proxy
    r = 2 * cs[:, None] * PN[None, :] - wl
    tA, tB, tS = ray_obb(p, r, obbs[0]), ray_obb(p, r, obbs[1]), ray_sphere(p, r)
    first = np.argmin(np.stack([tA, tB, tS], 1), 1)
    tmin = np.minimum(np.minimum(tA, tB), tS)
    hitA = np.isfinite(tmin) & (first == 0)
    hitB = np.isfinite(tmin) & (first == 1)
    tot = e.sum() / S.PSRC
    hb = (hitA | hitB) & vis
    return {'E_panel': float(tot), 'E_ring': ering.tolist(), 'ring0_share': float(ering[0] / tot) if tot > 0 else None,
            'cos_min_lit': float(cs[vis].min()) if vis.any() else None, 'cos_max_lit': float(cs[vis].max()) if vis.any() else None,
            'lit_points': int(vis.sum()),
            'E_mirror_to_boxA': float(e[hitA].sum() / S.PSRC), 'E_mirror_to_boxB': float(e[hitB].sum() / S.PSRC),
            'E_ring0_mirror_to_box': float(e[(hitA | hitB) & (ring == 0)].sum() / S.PSRC),
            'mirror_dist_box_median': float(np.median(tmin[hb])) if hb.any() else None}


def main():
    ck = analyse(CK2['geom'], CK2['lights'][0]['pos'])
    res = {'CK2': ck, 'populations': {}}
    print('CK2', json.dumps(ck))
    rows = []
    for it in pop['train']['items']:
        for li, l in enumerate(it['lights']):
            a = analyse(it['geom'], l[0]['pos'])
            a['file'] = it['file']
            a['light'] = li
            a['light_pos'] = l[0]['pos']
            rows.append(a)
    ckbox = ck['E_mirror_to_boxA'] + ck['E_mirror_to_boxB']
    ck0 = ck['E_ring0_mirror_to_box']
    for name, keep in (('P86_all1024', lambda f: True), ('P92A_928', lambda f: f not in ex92a), ('P94_96', lambda f: f in sel94)):
        rr = [r for r in rows if keep(r['file'])]
        E = np.array([r['E_panel'] for r in rr])
        ER = np.array([r['E_ring'] for r in rr])
        mb = np.array([r['E_mirror_to_boxA'] + r['E_mirror_to_boxB'] for r in rr])
        mb0 = np.array([r['E_ring0_mirror_to_box'] for r in rr])
        r0 = np.array([r['ring0_share'] if r['ring0_share'] is not None else 0.0 for r in rr])
        cmax = np.array([r['cos_max_lit'] if r['cos_max_lit'] is not None else 9.0 for r in rr])
        lit = E > 0
        graz = lit & (r0 >= 0.9)                                                 # >= 90 % of the panel irradiance at cos < 0.25
        allr0 = lit & (cmax < 0.25)                                              # like CK2: every lit point in ring 0
        top = sorted([(float(m), r['file'], r['light'], [round(v, 3) for v in r['light_pos']], round(float(c), 3))
                      for r, m, c in zip(rr, mb0, cmax)], reverse=True)[:5]
        o = {'n_lightings': len(rr), 'n_files': len({r['file'] for r in rr}),
             'ring_share_of_panel_irradiance': (ER.sum(0) / ER.sum()).tolist(),
             'lightings_panel_unlit': int((~lit).sum()),
             'lightings_ring0_share_ge_0.9': int(graz.sum()), 'lightings_all_lit_points_ring0': int(allr0.sum()),
             'lightings_cos_max_le_CK2_cos_max': int((lit & (cmax <= ck['cos_max_lit'])).sum()),
             'panel_irradiance_share_of_ring0_dominated_lightings': float(E[graz].sum() / E.sum()),
             'mirror_to_box_share_of_panel_irradiance': float(mb.sum() / E.sum()),
             'ring0_mirror_to_box_share_of_panel_irradiance': float(mb0.sum() / E.sum()),
             'lightings_mirror_to_box_ge_CK2': int((mb >= ckbox).sum()), 'lightings_mirror_to_box_ge_half_CK2': int((mb >= 0.5 * ckbox).sum()),
             'lightings_ring0_mirror_to_box_ge_CK2': int((mb0 >= ck0).sum()),
             'lightings_ring0_mirror_to_box_ge_half_CK2': int((mb0 >= 0.5 * ck0).sum()),
             'lightings_ring0_mirror_to_box_ge_tenth_CK2': int((mb0 >= 0.1 * ck0).sum()),
             'lightings_grazing_and_any_mirror_to_box': int((graz & (mb > 0)).sum()),
             'files_with_a_ring0_mirror_to_box_ge_tenth_CK2': len({r['file'] for r, m in zip(rr, mb0) if m >= 0.1 * ck0}),
             'top5_ring0_mirror_to_box': top}
        res['populations'][name] = o
        print(name, json.dumps(o))
    res['rows'] = rows
    json.dump(res, open(sys.argv[2], 'w'))


if __name__ == '__main__':
    main()
