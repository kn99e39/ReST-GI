"""WL104 gates. Exit code 0 = pass, 2 = gate failure (the chain stops; thresholds are never relaxed).

  python gates104.py pre <out_json>             G0 inputs (sha256 = LabServer63 originals), learned-ER state hashes = WL84 audit,
                                                G1 camera (adapter rays = WL81 formula), G4 layout (N < 32768)
  python gates104.py refs <refs_root> <out_json>  G2 tessellation (mesh LAY4 vs WL84 analytic ref84), G3 physical path (ECRO tracer
                                                  D = 6 vs Mitsuba max_depth 7, learned-ER), per image
"""
import json
import os
import sys

import numpy as np

import common104 as C


def rel_l1(a, b, m):
    return float(np.abs(a - b)[m].sum() / max(np.abs(b)[m].sum(), 1e-30))


def pre(out):
    import rt104 as RT
    res = {'inputs': {}, 'states': {}}
    for k, v in C.P['inputs_sha256'].items():
        res['inputs'][k] = C.sha(C.IN63 / k) == v
    for tag, v in C.STATE_SHA.items():
        res['states'][tag] = RT.register(tag)[2] == v
    ctx = RT.Ctx(RT.scene('test83_iid_1', 'er'))
    ray, _, _ = ctx.gen_rays(1, None)
    r81 = RT.R81.camera_rays(0, False)
    res['camera_max_abs_dir_diff'] = float(max(np.abs(np.asarray(ray.d[k]) - np.asarray(r81.d[k])).max() for k in range(3)))
    res['N'] = int(ctx.N)
    res['sphere_ply_sha256_ok'] = C.sha(C.SPHERE_PLY) == C.P['scene']['sphere_ply_sha256']
    res['quad_ply_sha256_ok'] = C.sha(C.QUAD_PLY) == C.P['scene']['quad_ply_sha256']
    ok = (all(res['inputs'].values()) and all(res['states'].values()) and res['camera_max_abs_dir_diff'] <= 1e-6 and res['N'] < 32768
          and res['sphere_ply_sha256_ok'] and res['quad_ply_sha256_ok'])
    res['pass'] = bool(ok)
    C.write_json(out, res)
    print('GATES_PRE', 'PASS' if ok else 'FAIL', json.dumps(res)[:500], flush=True)
    os._exit(0 if ok else 2)


def refs(root, out):
    import rt104 as RT
    res = {}
    allok = True
    for image in C.IMAGES:
        tag, light, _, _ = C.image_spec(image)
        RT.register(tag)
        obj, _ = RT.R81.masks(RT.scene(tag, 'lay', light=light, with_sensor=False))
        d = os.path.join(root, image)
        h = lambda arm, i: np.load(os.path.join(d, f'{arm}_h{i}.npz'))['mean'].astype(np.float64)
        # G2: mesh layered depth 4 (2 x 16384) vs WL84 analytic depth-4 reference (2 x 32768); noise-only expectation from WL84 halves
        mesh4 = 0.5 * (h('LAY4', 0) + h('LAY4', 1))
        r84 = np.load(C.W84REF / f'{image}_ref84.npy').astype(np.float64)
        rA = np.load(C.W84REF / f'{image}_ref84A.npy').astype(np.float64); rB = np.load(C.W84REF / f'{image}_ref84B.npy').astype(np.float64)
        # sd(mesh4 - r84) / sd(rA - rB) = sqrt(1/32768 + 1/65536) / sqrt(2/32768) = 0.866 for unit per-sample variance
        exp2 = 0.866 * rel_l1(rA, rB, obj)
        g2 = {'obj_rel_l1': rel_l1(mesh4, r84, obj), 'noise_expectation': exp2, 'obj_mean_ratio': float(mesh4[obj].mean() / r84[obj].mean())}
        g2['pass'] = bool(g2['obj_rel_l1'] <= 1.5 * exp2 and abs(g2['obj_mean_ratio'] - 1) <= 0.005)
        # G3: ECRO tracer (pt102 cam, D = 6, 2 x 4096) vs Mitsuba path max_depth 7 (2 x 32768), learned-ER, same scene / camera
        ptc = [np.load(os.path.join(d, f'PTC_h{i}.npz')) for i in (0, 1)]
        pt = [z['e1'].astype(np.float64) + z['e2'] + z['h'] for z in ptc]
        er7 = [h('ER7', i) for i in (0, 1)]
        ptm, erm = 0.5 * (pt[0] + pt[1]), 0.5 * (er7[0] + er7[1])
        # noise-only expectation of rel L1(ptm - erm): sd(ptm - erm)^2 = sd(pt0 - pt1)^2 / 4 + sd(er0 - er1)^2 / 4 (independent halves)
        exp3 = 0.5 * float(np.hypot(rel_l1(pt[0], pt[1], obj), rel_l1(er7[0], er7[1], obj)))
        full = np.ones_like(obj)
        g3 = {'obj_rel_l1': rel_l1(ptm, erm, obj), 'noise_expectation': exp3, 'obj_mean_ratio': float(ptm[obj].mean() / erm[obj].mean()),
              'full_mean_ratio': float(ptm.mean() / erm.mean()), 'full_rel_l1': rel_l1(ptm, erm, full)}
        g3['pass'] = bool(g3['obj_rel_l1'] <= 1.5 * exp3 and abs(g3['obj_mean_ratio'] - 1) <= 0.01 and abs(g3['full_mean_ratio'] - 1) <= 0.01)
        res[image] = {'G2_tessellation': g2, 'G3_physical_path': g3}
        allok &= g2['pass'] and g3['pass']
    res['pass'] = bool(allok)
    C.write_json(out, res)
    print('GATES_REFS', 'PASS' if allok else 'FAIL', json.dumps(res)[:800], flush=True)
    os._exit(0 if allok else 2)


if __name__ == '__main__':
    {'pre': lambda: pre(sys.argv[2]), 'refs': lambda: refs(sys.argv[2], sys.argv[3])}[sys.argv[1]]()
