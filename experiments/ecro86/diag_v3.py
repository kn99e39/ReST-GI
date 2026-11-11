import sys, json, numpy as np, os
sys.path.insert(0, sys.argv[1])
import pt86 as PT
mi, dr, S = PT.mi, PT.dr, PT.S
FX = PT.FX; CAM = FX['camera']; RES = 256
out = {}
for e in ('E1', 'E2'):
    ex = FX[e]; l = PT.Lights(FX['light_sets'][ex['light_set']])
    fresh1 = PT.Ctx(ex['T1'])
    c = PT.Ctx(ex['T0']); c.set_geom(ex['T1'])
    r = {}
    for name in ('boxA', 'boxB'):
        va = np.asarray(fresh1.params[f'{name}.vertex_positions'], np.float64); vb = np.asarray(c.params[f'{name}.vertex_positions'], np.float64)
        na = np.asarray(fresh1.params[f'{name}.vertex_normals'], np.float64); nb = np.asarray(c.params[f'{name}.vertex_normals'], np.float64)
        r[name] = {'vertex_max_abs_diff_m': float(np.abs(va - vb).max()), 'normal_max_abs_diff': float(np.abs(na - nb).max()) if na.size else None, 'n_vertices': int(va.size // 3)}
    for spp in (256, 1024):
        f1, _ = fresh1.render(l, 'ER', CAM, RES, spp, 31, 'cam', plain=True); i1, _ = c.render(l, 'ER', CAM, RES, spp, 31, 'cam', plain=True)
        a = PT.pack(f1, RES, spp)['tot'].astype(np.float64); b = PT.pack(i1, RES, spp)['tot'].astype(np.float64)
        d = np.abs(a - b).max(-1); mm = a.mean()
        bad = d > 2e-4 * mm
        g = fresh1.gbuffer(CAM, RES)
        # distance of differing pixels to a moved-box silhouette: neighbourhood contains a box pixel and a non-box pixel
        sid = g['sid']; isb = np.isin(sid, [7, 8])
        edge = np.zeros_like(isb)
        for dy in (-2, -1, 0, 1, 2):
            for dx in (-2, -1, 0, 1, 2):
                edge |= np.roll(np.roll(isb, dy, 0), dx, 1) != isb
        r[f'spp{spp}'] = {'pixels_differing': int(bad.sum()), 'fraction': float(bad.mean()), 'max_rel_mean': float(d.max() / mm),
                          'frame_rel_diff': float(abs(a.sum() - b.sum()) / a.sum()), 'rel_l1': float(np.abs(a - b).sum() / a.sum()),
                          'differing_on_box_silhouette_2px': int((bad & edge).sum()), 'differing_elsewhere': int((bad & ~edge).sum()),
                          'sample_quantum_rel_mean': float(1.0 / spp)}
    st_a = fresh1.state_rays(); st_b = c.state_rays()
    r['state_identical'] = bool(np.array_equal(st_a[0], st_b[0]) and np.array_equal(st_a[1], st_b[1]))
    out[e] = r
print(json.dumps(out, indent=1))
json.dump(out, open(sys.argv[2], 'w'), indent=1)
os._exit(0)
