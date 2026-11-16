"""WL97 audit, track J4: attribution of the table-readout error at camera vertex 2 (decisive check for E4-F6).

The 'ecro' camera mode reads hl = throughput(x1) * B[patch(x2), bin(x2 -> x1)] (pt86.py:228-233).  This script traces the SAME camera
prefix (x1, x2) and replaces the table lookup by fresh probe estimates (unchanged pt86.Ctx.trace, mode 'probe') started
  A: at the exact hit point x2 in the exact direction x2 -> x1          (= path-traced continuation; must agree with REF h)
  B: at a uniform random point of patch(x2), exact world direction       (spatial patch averaging only)
  C: at the exact point x2, uniform random direction inside bin(x2->x1)  (angular bin averaging only)
  D: at a uniform random point of patch(x2), random direction in the bin (= the EXPECTATION of the table as the field tracer defines
     it, pt86.py:327-351, with no table sampling noise)
Each probe is run with kmax 2 (table 2-event content = camera 3 events) and kmax 5 (2..5 = camera 3..6) on identical random streams.
On the same camera paths the stored reference tables are read as well (T2 = e2 mean of halves, T3 = e3 mean, halves a / b).
All arms share the camera prefix and the probe random streams (common random numbers).  Nothing is trained; no existing file is touched.

Usage: python j4_attr.py <CODE/experiments> <CKF_none.npz> <outdir> <spp> <seed0> <nrep> [b]
Output: <outdir>/J4_s<seed>.npz  (per-pixel means [res, res, 3] float32 per key; 'fail_<arm>' = throughput weight of probe samples that
        failed the probe-hit test, [res, res])
"""
import json, os, sys, time
import numpy as np

CODE, CKF, OUTD, SPP, SEED0, NREP = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6])
BB = int(sys.argv[7]) if len(sys.argv) > 7 else 16
sys.path.insert(0, os.path.join(CODE, 'ecro86'))
import pt86 as PT  # noqa: E402
mi, dr, F, U, NB, CD = PT.mi, PT.dr, PT.F, PT.U, PT.NB, PT.CD

SC = json.load(open(os.path.join(CODE, 'ecro_local93d', 'addendum_checker', 'scene_ck2.json')))
os.makedirs(OUTD, exist_ok=True)
ctx = PT.Ctx(SC['geom'])
lt = PT.Lights(SC['lights'])
cam = SC['camera']; res = cam['res']; n = res * res
z = np.load(CKF)
e2 = lambda s: z['e2_p_' + s].astype(np.float64) + z['e2_o_' + s].astype(np.float64)
e3 = lambda s: z['e3_p_' + s].astype(np.float64) + z['e3_o_' + s].astype(np.float64)
TABS = {'T2': 0.5 * (e2('a') + e2('b')), 'T3': 0.5 * (e3('a') + e3('b')), 'T2a': e2('a'), 'T2b': e2('b'), 'T3a': e3('a'), 'T3b': e3('b'),
        'T3o': 0.5 * (z['e3_o_a'].astype(np.float64) + z['e3_o_b']), 'T3p': 0.5 * (z['e3_p_a'].astype(np.float64) + z['e3_p_b'])}
TABS = {k: PT.fl(v.astype(np.float32)) for k, v in TABS.items()}


def launch(seed, li, bb):
    N = n * bb
    sd = dr.opaque(mi.UInt64, seed * 1000003 + li)
    rng = mi.PCG32(size=N, initstate=sd, initseq=mi.UInt64(0))
    ray, pix = PT.gen_rays(cam, res, bb, rng)
    # camera vertex 1 (identical to pt86.Ctx.trace, k = 1, ER)
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
    # camera vertex 2
    si2 = ctx.scene.ray_intersect(ray2, active)
    active &= si2.is_valid()
    sid2 = ctx.shape_id(si2)
    wi2 = si2.to_local(-ray2.d)
    active &= wi2.z > 1e-6
    pid, bn = ctx.patch_bin(si2, sid2, -ray2.d)
    pid = dr.select(active, pid, U(0)); bn = dr.select(active, bn, U(0))
    thr = dr.select(active, thr, 0.0)
    x2 = mi.Point3f(si2.p); w2 = mi.Vector3f(-ray2.d); n2 = mi.Vector3f(si2.n)
    rq = mi.PCG32(size=N, initstate=sd, initseq=mi.UInt64(7))
    uu = [rq.next_float32() for _ in range(6)]
    dr.eval(pix, active, pid, bn, thr, x2, w2, n2, uu); dr.sync_thread()

    acc = {}

    def put(d):
        a = PT.Ctx.accum(d, pix, n)
        acc.update(a)

    # stored-table reads on the same camera paths
    i3 = (pid * NB + bn) * 3
    d = {}
    for name, B in TABS.items():
        for c, tc in enumerate((thr.x, thr.y, thr.z)):
            d[f'{name}_{c}'] = dr.select(active, tc * dr.gather(F, B, i3 + c, active), 0.0)
    put(d)

    pp, nnp, (t, bv, nf) = ctx.patch_point(pid, uu[0], uu[1])
    wl1 = PT.Ctx.dir_in_bin(bn, uu[2], uu[3]); wl2 = PT.Ctx.dir_in_bin(bn, uu[4], uu[5])
    wb1 = t * wl1.x + bv * wl1.y + nf * wl1.z; wb2 = t * wl2.x + bv * wl2.y + nf * wl2.z

    def pick(nrm):                                              # first in-bin direction that passes the field tracer's ok test
        return dr.select(dr.dot(wb1, nrm) > 1e-3, wb1, wb2)

    arms = {'A': (x2, w2, n2), 'B': (pp, w2, nnp), 'C': (x2, pick(n2), n2), 'D': (pp, pick(nnp), nnp)}
    for an, (p, wv, nrm) in arms.items():
        ok = active & (dr.dot(wv, nrm) > 1e-3)
        d = {}
        for km in (2, 5):
            rc = mi.PCG32(size=N, initstate=sd, initseq=mi.UInt64(11))
            out, okp = ctx.trace(mi.Ray3f(p + wv * 2e-3, -wv), N, rc, sd, 'ER', km, lt, 'probe', probe_y=p, probe_ok=ok)
            for c, tc in enumerate((thr.x, thr.y, thr.z)):
                d[f'{an}{km}_{c}'] = tc * out[f'f{c}']
            if km == 2:
                d[f'fail_{an}'] = dr.select(active & ~okp, PT.mean3(thr), 0.0)
                d[f'wsum_{an}'] = dr.select(active, PT.mean3(thr), 0.0)
        put(d)
    return acc


for r in range(NREP):
    seed = SEED0 + r
    fn = os.path.join(OUTD, f'J4_s{seed}.npz')
    if os.path.exists(fn):
        continue
    t0 = time.time()
    tot = None; done = 0; li = 0
    while done < SPP:
        bb = min(BB, SPP - done)
        a = launch(seed, li, bb)
        tot = a if tot is None else {k: tot[k] + v for k, v in a.items()}
        done += bb; li += 1
    o = {}
    names = sorted({k.rsplit('_', 1)[0] for k in tot if not k.startswith(('fail_', 'wsum_'))})
    for nm in names:
        o[nm] = np.stack([(tot[f'{nm}_{c}'] / SPP).reshape(res, res) for c in range(3)], -1).astype(np.float32)
    for k in tot:
        if k.startswith(('fail_', 'wsum_')):
            o[k] = (tot[k] / SPP).reshape(res, res).astype(np.float32)
    np.savez_compressed(fn, spp=SPP, seed=seed, **o)
    print('J4ATTR', seed, SPP, round(time.time() - t0, 1), flush=True)
print('J4ATTR DONE', flush=True)
os._exit(0)
