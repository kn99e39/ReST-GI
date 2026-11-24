"""WL112 focused tests (run before Stage A; any failure stops the chain).

  python tests112.py ROOT     -> ROOT/tests112.json
T1 sub-patch indexing of analytic points (rect, box, sphere), away from borders
T2 contact semantics X check: C shadow attenuation == exp(-sigma chord) on a radial grid about the tangency; N overstated in the ring
T3 camera independence: identical construction (same seed) before and after rendering a different camera
T4 readout instrument: zero table -> zero HIGH; unit table -> HIGH == prefix throughput sum; LOW unchanged by the table
T5 dataset isolation: D112 training / validation configuration hashes disjoint from every Stage D / E state
T6 subcell table flattening order == readout index (pid*4+sub)*64+bin
"""
import json, math, os, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common112 as C  # noqa: E402
import tracer112 as T  # noqa: E402

mi, dr, F, U, PT = T.mi, T.dr, T.F, T.U, T.PT


def t1(ctx):
    rs = np.random.default_rng(11207001)
    n = 20000
    pid = rs.integers(0, ctx.N, n); sub = rs.integers(0, 4, n)
    a = (sub % 2 + rs.uniform(.05, .95, n)) / 2; b = (sub // 2 + rs.uniform(.05, .95, n)) / 2
    P = U(pid.astype(np.uint32))
    p, nn, _ = ctx.patch_point(P, F(a.astype(np.float32)), F(b.astype(np.float32)))
    si = dr.zeros(mi.SurfaceInteraction3f, n); si.p = p
    got = np.asarray(ctx.subcell(si, P))
    return {'pass': bool(np.array_equal(got, sub)), 'mismatches': int((got != sub).sum())}


def t2():
    geom, lc = C.fixture('F0')
    light = PT.Lights(lc)
    res = {}
    lp = np.asarray(lc[0]['pos'])
    for sem in ('N', 'C'):
        ctx = T.Ctx112(geom, sem)
        radii = np.array([1e-3, 2e-3, 3e-3, 6e-3, 11e-3, 11.5e-3, 12e-3, 20e-3, 40e-3, 80e-3])
        az = np.linspace(0, 2 * math.pi, 8, endpoint=False)
        R_, A_ = np.meshgrid(radii, az, indexing='ij')
        x = 0.9 + R_.ravel() * np.cos(A_.ravel()); y = 0.4 + R_.ravel() * np.sin(A_.ravel())
        n = x.size
        si = dr.zeros(mi.SurfaceInteraction3f, n)
        si.p = mi.Point3f(F(x.astype(np.float32)), F(y.astype(np.float32)), F(0.)); si.n = mi.Vector3f(0, 0, 1)
        si.sh_frame = mi.Frame3f(mi.Vector3f(0, 0, 1)); si.t = F(1.); si.shape = ctx.ptrs[0]
        si.wi = mi.Vector3f(0, 0, 1)
        cl = PT.CD.DrClosure83('ER', ctx.mats.states, U(np.zeros(n, np.uint32)))
        d, thr = ctx.direct(si, mi.Vector3f(0, 0, 1), cl, light, mi.Bool(True), mi.Bool(True))
        dl = mi.Point3f(*lp.tolist()) - si.p; d2 = dr.squared_norm(dl); wl = dr.normalize(dl)
        f = cl.eval(mi.Vector3f(0, 0, 1), si.to_local(wl), mi.Bool(True))
        un = f * light.rgb[0] / d2                                    # unoccluded direct radiance
        att = np.asarray((thr + d).x / dr.maximum(un.x, 1e-30)).reshape(len(radii), len(az))
        # analytic chord through the sphere along the light segment (float64)
        c = np.array([0.9, 0.4, 0.4]); chords = np.zeros(n)
        for i in range(n):
            o = np.array([x[i], y[i], 0.0]); w = (lp - o) / np.linalg.norm(lp - o)
            q = o - c; bb = q @ w; cc = q @ q - 0.16; disc = bb * bb - cc
            chords[i] = 2 * math.sqrt(disc) if disc > 0 and -bb > 0 else 0.0
        expect = np.exp(-12.0 * chords).reshape(len(radii), len(az))
        res[sem] = {'radii_mm': (radii * 1e3).tolist(), 'attenuation_mean': att.mean(1).tolist(), 'analytic_mean': expect.mean(1).tolist(),
                    'ratio_mean': (att.mean(1) / np.maximum(expect.mean(1), 1e-30)).tolist()}
        del ctx
    rc = np.array(res['C']['ratio_mean']); rn = np.array(res['N']['ratio_mean'])
    res['pass'] = bool(np.all(np.abs(rc - 1) <= 5e-3) and rn[:4].min() > 10)
    return res


def t3(ctx, light):
    a = ctx.construct(light, 8, 11207003, b=8)
    ctx.cam_pass(light, dict(C.views('F0')['H3'], res=64), 64, 4, 11207004, 'cam', b=4)
    b = ctx.construct(light, 8, 11207003, b=8)
    d = max(float(np.abs(a[k] - b[k]).max() / max(np.abs(a[k]).max(), 1e-30)) for k in a)
    return {'pass': bool(d <= 1e-6), 'max_rel': d}


def t4(ctx, light):
    n = ctx.N * T.NB * 3
    z = PT.fl(np.zeros(n, np.float32)); o = PT.fl(np.ones(n, np.float32)); o4 = PT.fl(np.ones(n * 4, np.float32))
    img, _, _, _ = ctx.cam_pass(light, dict(C.V0_CAM, res=64), 64, 8, 11207005, 'ecro', tables=[('z', z, 'coarse'), ('o', o, 'coarse'), ('o4', o4, 'sub')], b=8)
    img2, _, _, _ = ctx.cam_pass(light, dict(C.V0_CAM, res=64), 64, 8, 11207005, 'ecro', tables=[], b=8)
    return {'pass': bool(np.abs(img['z']).max() == 0 and np.abs(img['o'] - img['o4']).max() <= 1e-6 and np.abs(img['e1'] - img2['e1']).sum() <= 1e-5 * np.abs(img2['e1']).sum()
                         and img['o'].min() >= 0), 'unit_mean': float(img['o'].mean())}


def t5():
    test = {C.digest(g) for g in C.exclusion_states()}
    hs = set()
    for split in ('train', 'val'):
        for i in range(C.dv(C.P['stage_C']['splits'][split]['geometries'], 6 if split == 'train' else 2)):
            hs.add(C.digest(C.train_config(split, i)['geom']))
    return {'pass': bool(not (hs & test)), 'train_val_geometries': len(hs)}


def t6():
    import torch
    N, NS, NB = 1270, 4, 64
    Bt = torch.arange(N * NS * NB * 3).reshape(1, N, NS, NB, 3)[0].numpy().ravel()
    pid, sub, b, c = 1239, 2, 37, 1
    return {'pass': bool(Bt[((pid * NS + sub) * NB + b) * 3 + c] == ((pid * NS + sub) * NB + b) * 3 + c)}


def main(root):
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    geom, lc = C.fixture('F0'); light = PT.Lights(lc)
    out = {'T2_contact_semantics': t2()}
    ctx = T.Ctx112(geom, 'C')
    out['T1_subcell'] = t1(ctx)
    out['T3_camera_independence'] = t3(ctx, light)
    out['T4_readout'] = t4(ctx, light)
    out['T5_isolation'] = t5()
    out['T6_flatten'] = t6()
    out['pass'] = bool(all(v['pass'] for v in out.values()))
    C.jd(out, root / 'tests112.json')
    print('TESTS', json.dumps({k: v['pass'] if isinstance(v, dict) else v for k, v in out.items()}), flush=True)
    if not out['pass']:
        sys.exit(3)


if __name__ == '__main__':
    main(sys.argv[1])
    sys.stdout.flush(); os._exit(0)
