"""WL97 audit, track J4: table reads only, on the SAME camera prefixes as j4_attr.py (same seeds / launch size -> identical lanes).
Reads the CKF_none reference tables (T2c, T3c) and the INDEPENDENT audit reference tables of track E4 (TF_9740x.npz, 4 halves;
T2n, T3n = mean of 4; T2n01 / T2n23, T3n01 / T3n23 = means of two halves) through the unchanged table lookup convention
hl = throughput(x1) * B[patch(x2), bin(x2 -> x1)].  Also a consistency check of the patch layout: distance between the camera hit x2 and a
uniform random point of patch(x2) (patch_point) relative to the patch diagonal, per shape (max and mean over lanes).

Usage: python j4_tread.py <CODE/experiments> <CKF_none.npz> <outdir> <spp> <seed0> <nrep> <b> <TF_1.npz> ...
"""
import json, os, sys, time
import numpy as np

CODE, CKF, OUTD, SPP, SEED0, NREP, BB = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6]), int(sys.argv[7])
TFS = sys.argv[8:]
sys.path.insert(0, os.path.join(CODE, 'ecro86'))
import pt86 as PT  # noqa: E402
mi, dr, F, U, NB, CD = PT.mi, PT.dr, PT.F, PT.U, PT.NB, PT.CD

SC = json.load(open(os.path.join(CODE, 'ecro_local93d', 'addendum_checker', 'scene_ck2.json')))
ctx = PT.Ctx(SC['geom']); lt = PT.Lights(SC['lights']); cam = SC['camera']; res = cam['res']; n = res * res
z = np.load(CKF)
e2 = lambda s: z['e2_p_' + s].astype(np.float64) + z['e2_o_' + s].astype(np.float64)
e3 = lambda s: z['e3_p_' + s].astype(np.float64) + z['e3_o_' + s].astype(np.float64)
tf = [np.load(f) for f in TFS]
a2 = [t['k2'].astype(np.float64).sum(2) for t in tf]
a3 = [sum(t[f'k{k}'].astype(np.float64) for k in (3, 4, 5)).sum(2) for t in tf]
TABS = {'T2c': 0.5 * (e2('a') + e2('b')), 'T3c': 0.5 * (e3('a') + e3('b')), 'T2n': np.mean(a2, 0), 'T3n': np.mean(a3, 0),
        'T2n01': 0.5 * (a2[0] + a2[1]), 'T2n23': 0.5 * (a2[2] + a2[3]), 'T3n01': 0.5 * (a3[0] + a3[1]), 'T3n23': 0.5 * (a3[2] + a3[3])}
assert all(v.shape == (ctx.N, NB, 3) for v in TABS.values())
TABS = {k: PT.fl(v.astype(np.float32)) for k, v in TABS.items()}
LAY = {}


def launch(seed, li, bb, layout):
    N = n * bb
    sd = dr.opaque(mi.UInt64, seed * 1000003 + li)
    rng = mi.PCG32(size=N, initstate=sd, initseq=mi.UInt64(0))
    ray, pix = PT.gen_rays(cam, res, bb, rng)
    si = ctx.scene.ray_intersect(ray, mi.Bool(True))
    active = si.is_valid()
    mid = dr.gather(U, ctx.SMAT, ctx.shape_id(si))
    wi = si.to_local(-ray.d)
    active &= wi.z > 1e-6
    wi = dr.select(active, wi, mi.Vector3f(0, 0, 1))
    cl = CD.DrClosure83('ER', ctx.mats.states, mid)
    u1 = rng.next_float32(); u2 = mi.Point2f(rng.next_float32(), rng.next_float32())
    wo, pdf, w, valid = cl.sample(wi, u1, u2, active)
    thr = mi.Color3f(1.0) * dr.select(valid, w, 0.0)
    active &= valid
    ray2 = si.spawn_ray(si.to_world(wo))
    si2 = ctx.scene.ray_intersect(ray2, active)
    active &= si2.is_valid()
    sid2 = ctx.shape_id(si2)
    active &= si2.to_local(-ray2.d).z > 1e-6
    pid, bn = ctx.patch_bin(si2, sid2, -ray2.d)
    pid = dr.select(active, pid, U(0)); bn = dr.select(active, bn, U(0))
    thr = dr.select(active, thr, 0.0)
    i3 = (pid * NB + bn) * 3
    d = {}
    for name, B in TABS.items():
        for c, tc in enumerate((thr.x, thr.y, thr.z)):
            d[f'{name}_{c}'] = dr.select(active, tc * dr.gather(F, B, i3 + c, active), 0.0)
    if layout:
        rq = mi.PCG32(size=N, initstate=sd, initseq=mi.UInt64(7))
        pp, nnp, _ = ctx.patch_point(pid, rq.next_float32(), rq.next_float32())
        g = lambda k: dr.gather(F, ctx.PTB, pid * 21 + k)
        diag = dr.norm(mi.Vector3f(g(3), g(4), g(5)) + mi.Vector3f(g(6), g(7), g(8)))
        dist = dr.select(active, dr.norm(pp - si2.p), 0.0)
        sph = g(18) > 0.5
        rel = dr.select(active & ~sph, dist / dr.maximum(diag, 1e-9), 0.0)
        dsp = dr.select(active & sph, dist, 0.0)
        ndot = dr.select(active & ~sph, dr.abs(dr.dot(nnp, mi.Vector3f(si2.n))), 1.0)
        dr.eval(rel, dsp, ndot, sid2, active)
        reln, dspn, nd, s2, ac = np.asarray(rel), np.asarray(dsp), np.asarray(ndot), np.asarray(sid2), np.asarray(active)
        for s in range(9):
            mm = ac & (s2 == s)
            if mm.any():
                LAY[PT.SHAPE_NAMES[s]] = {'lanes': int(mm.sum()), 'max_dist_over_patch_diagonal': float(reln[mm].max()), 'mean': float(reln[mm].mean()),
                                         'min_abs_normal_dot': float(nd[mm].min()), 'max_dist_sphere_m': float(dspn[mm].max())}
    return PT.Ctx.accum(d, pix, n)


for r in range(NREP):
    seed = SEED0 + r
    fn = os.path.join(OUTD, f'J4T_s{seed}.npz')
    if os.path.exists(fn):
        continue
    t0 = time.time(); tot = None; done = 0; li = 0
    while done < SPP:
        bb = min(BB, SPP - done)
        a = launch(seed, li, bb, r == 0 and li == 0)
        tot = a if tot is None else {k: tot[k] + v for k, v in a.items()}
        done += bb; li += 1
    o = {nm: np.stack([(tot[f'{nm}_{c}'] / SPP).reshape(res, res) for c in range(3)], -1).astype(np.float32) for nm in TABS}
    np.savez_compressed(fn, spp=SPP, seed=seed, **o)
    print('J4TREAD', seed, SPP, round(time.time() - t0, 1), flush=True)
json.dump(LAY, open(os.path.join(OUTD, 'layout_check.json'), 'w'), indent=1)
print('LAYOUT', json.dumps(LAY))
print('J4TREAD DONE', flush=True)
os._exit(0)
