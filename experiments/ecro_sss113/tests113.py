"""WL113 focused tests (before Stage A; any failure stops the chain).

T1 4x4 addressing of analytic points; T2 nested aggregation (coarse / 2x2 / 4x4 poolings and dense expansions); T3 compact
storage == dense expansion; T4 VHS support: contact formation / removal / revisit determinism on a lift sequence; T5 deep pass
events 1-6 == depth-6 pass; T6 learned output indexing ([R,16,64,3] -> dense rows) and source linearity of ECRO-L4;
T7 LOW unchanged by tables; zero table -> zero HIGH; T8 training / evaluation isolation (configuration hashes disjoint).
"""
import json, os, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common113 as C  # noqa: E402
import tracer113 as T  # noqa: E402

mi, dr, F, U, PT = T.mi, T.dr, T.F, T.U, T.PT


def main(root):
    root = Path(root); root.mkdir(parents=True, exist_ok=True); out = {}
    geom, lc = C.fixture('F0'); light = PT.Lights(lc)
    ctx = T.Ctx113(geom, 'C')
    rs = np.random.default_rng(11307001); n = 20000
    pid = rs.integers(0, ctx.N, n); sub = rs.integers(0, 16, n)
    a = (sub % 4 + rs.uniform(.05, .95, n)) / 4; b = (sub // 4 + rs.uniform(.05, .95, n)) / 4
    P = U(pid.astype(np.uint32)); p, _, _ = ctx.patch_point(P, F(a.astype(np.float32)), F(b.astype(np.float32)))
    si = dr.zeros(mi.SurfaceInteraction3f, n); si.p = p
    got = np.asarray(ctx.subcell(si, P)); out['T1'] = {'pass': bool(np.array_equal(got, sub)), 'mismatch': int((got != sub).sum())}
    acc = ctx.construct4(light, 8, 11307002, b=8)
    Tc, T2, T4, nat = T.tables4(acc)
    w4 = acc['U_w'].reshape(1270, 16, 64)
    rc = (T4 * w4[..., None]).sum(1) / np.maximum(w4.sum(1), 1e-30)[..., None]
    w2 = w4.reshape(1270, 2, 2, 2, 2, 64).sum((2, 4)).reshape(1270, 4, 64)
    rc2 = (T2 * w2[..., None]).sum(1) / np.maximum(w2.sum(1), 1e-30)[..., None]
    d16 = T.to_dense16(T2, 'sub2')
    out['T2'] = {'pass': bool(np.abs(rc - Tc)[nat['c']].max() <= 1e-5 * np.abs(Tc).max() and np.abs(rc2 - Tc)[nat['c']].max() <= 1e-5 * np.abs(Tc).max()
                              and np.array_equal(d16.reshape(1270, 4, 4, 64, 3)[:, ::2, ::2], T2.reshape(1270, 2, 2, 64, 3)))}
    desc = ctx.vhs(); sup = T.support_from_desc(desc, ctx.G)
    dense = T.to_dense16(T4, 'compact', sup, Tc)
    out['T3'] = {'pass': bool(np.array_equal(dense[sup], T4[sup]) and np.array_equal(dense[~sup], np.repeat(Tc[~sup][:, None], 16, 1))),
                 'support_fraction': float(sup.mean())}
    seq = C.sequences()['SEQ_A_lift']['states']
    sups = []
    for g in seq:
        c2 = T.Ctx113(g, 'C'); sups.append(T.support_from_desc(c2.vhs(), c2.G)); del c2
    floor_near = lambda s: int(s[:256].sum())
    out['T4'] = {'floor_support_per_state': [floor_near(s) for s in sups], 'sphere_support_per_state': [int(s[1238:].sum()) for s in sups],
                 'revisit_identical': bool(np.array_equal(sups[0], sups[5]) and np.array_equal(sups[3], sups[6]))}
    out['T4']['pass'] = bool(out['T4']['revisit_identical'] and all(x == 32 for x in out['T4']['sphere_support_per_state']))
    cam = dict(C.V0_CAM, res=64)
    a6, _, _, _ = ctx.cam_pass(light, cam, 64, 8, 11307003, 'cam', b=8); f6, _ = ctx.full_pass(light, cam, 64, 8, 11307003, b=8)
    out['T5'] = {'pass': bool(sum(np.abs(a6[k] - f6[k]).sum() for k in ('e1', 'e2', 'h')) <= 1e-6 * sum(np.abs(a6[k]).sum() for k in ('e1', 'e2', 'h')))}
    import torch
    sys.path.insert(0, str(C.REPO / 'experiments/ecro86'))
    import model86 as M
    import model113 as M3
    hit, bn, _ = ctx.state_rays(); gs = M.GeoState(ctx.G, hit, bn, torch.device('cuda'))
    gs.sup = torch.as_tensor(np.flatnonzero(sup), device='cuda'); gs.desc = torch.as_tensor(desc, device='cuda')
    torch.manual_seed(1); m = M3.ECROL4().cuda(); m.mat_states = torch.as_tensor(T.C100.materials('sss'), device='cuda')
    torch.manual_seed(1); m0 = M.ECRO().cuda()
    shared_equal = all(torch.equal(p, m.state_dict()[k]) for k, p in m0.state_dict().items() if k in m.state_dict() and k != 'mat_states')
    s1 = torch.as_tensor(ctx.source(light), device='cuda').unsqueeze(0)
    with torch.no_grad():
        f = m.factors(gs); B, Bs = m.outputs(f, gs, s1); B2, Bs2 = m.outputs(f, gs, 2 * s1)
    out['T6'] = {'pass': bool(Bs.shape[1] == sup.sum() and Bs.shape[2] == 16 and float((Bs2 - 2 * Bs).abs().max()) <= 1e-5 * float(Bs.abs().max()) and shared_equal),
                 'shared_init_identical': bool(shared_equal)}
    z = PT.fl(np.zeros(1270 * 16 * 64 * 3, np.float32))
    i1, _, _, _ = ctx.cam_pass(light, cam, 64, 8, 11307004, 'ecro', tables=[('z', z, 'sub')], b=8)
    i2, _, _, _ = ctx.cam_pass(light, cam, 64, 8, 11307004, 'ecro', tables=[], b=8)
    out['T7'] = {'pass': bool(np.abs(i1['z']).max() == 0 and np.abs(i1['e1'] - i2['e1']).sum() <= 1e-5 * np.abs(i2['e1']).sum())}
    test = {C.digest(g) for g in C.exclusion_states()}
    tr = {C.digest(C.train_config(sp, i)['geom']) for sp, n_ in (('train', C.dv(192, 6)), ('val', C.dv(16, 2))) for i in range(n_)}
    out['T8'] = {'pass': bool(not (tr & test))}
    out['pass'] = bool(all(v['pass'] for v in out.values()))
    C.jd(out, root / 'tests113.json')
    print('TESTS', json.dumps({k: v['pass'] for k, v in out.items() if isinstance(v, dict)}), out['T4'], flush=True)
    if not out['pass']:
        sys.exit(3)


if __name__ == '__main__':
    main(sys.argv[1])
    sys.stdout.flush(); os._exit(0)
