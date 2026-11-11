"""WL86 region masks from geometry and visibility only (no radiance, no candidate output, no reference error).

  python masks86.py <root> <E1|E2> [--ls=name] [--res=N]
Per pixel (centre ray), with T0 and T1 scenes of the edit and the lights of the workload:
  stationary  same surface point in T0 and T1, not on a moved box
  A           stationary, direct light visibility differs for at least one light
  Bset        stationary, direct visibility unchanged
  u           exposure to change: fraction of RAYS cosine-distributed directions whose first hit differs between T0 and T1 or lies on a
              surface point whose direct visibility differs
  sc          (panel / sphere receivers) fraction of directions in the mirror cone (half-angle = protocol) with a changed hit as above
  fc          receiver in the footprint of a light -> panel -> receiver mirror path that is unoccluded in exactly one configuration
  C           Bset and (fc or (panel / sphere receiver and sc >= sc_min))        localized sharp / moderate interreflection
  D           Bset, not C, broad receiver, u >= u_hi                              broad multi-bounce change
  E           Bset, not C, u <= u_lo                                              (nearly) unaffected
"""
import json, math, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pt86 as PT  # noqa: E402
mi, dr, S = PT.mi, PT.dr, PT.S
F, U = PT.F, PT.U
FX = PT.FX
MK = S.P['regions']


def probe(c, lt, o, d, active):
    """One scene per kernel (OptiX allows a single shader binding table per kernel): evaluated hit record + direct visibility at the hit."""
    v, sid, p, n, f = c.hit_info(o, d, active)
    lv = [x & f for x in c.light_vis(p, n, lt)]
    dr.eval(v, sid, p, lv); dr.sync_thread()
    return v, sid, p, lv


def changed(c0, c1, lt, o, d, active):
    dr.eval(o, d, active)
    v0, s0, p0, l0 = probe(c0, lt, o, d, active)
    v1, s1, p1, l1 = probe(c1, lt, o, d, active)
    diff = (v0 ^ v1) | (v0 & v1 & (dr.neq(s0, s1) | (dr.norm(p0 - p1) > 1e-3)))
    dv = mi.Bool(False)
    for a, b in zip(l0, l1):
        dv |= a ^ b
    return active & (diff | (v0 & v1 & dv))


def ev(x):
    dr.eval(x); dr.sync_thread()
    return x


def compute(ename, ls=None, res=None, cam=None):
    ex = FX[ename]
    cam = cam or FX['camera']; res = res or cam['res']
    c0, c1 = PT.Ctx(ex['T0']), PT.Ctx(ex['T1'])
    lights = FX['light_sets'][ls or ex['light_set']]
    lt = PT.Lights(lights)
    g0, g1 = c0.gbuffer(cam, res), c1.gbuffer(cam, res)
    moved = [S.SID_BOXA if b == 'A' else S.SID_BOXB for b in ex['moved']]
    n = res * res
    stat = g0['ok'] & g1['ok'] & (g0['sid'] == g1['sid']) & (np.abs(g0['p'] - g1['p']).max(-1) < 1e-4) & ~np.isin(g0['sid'], moved)
    P3 = lambda a: mi.Point3f(*[PT.fl(a[..., k]) for k in range(3)])
    V3 = lambda a: mi.Vector3f(*[PT.fl(a[..., k]) for k in range(3)])
    p, nn, dcam = P3(g0['p']), V3(g0['n']), V3(g0['d'])
    nn = dr.select(dr.dot(nn, dcam) < 0, nn, -nn)
    dr.eval(p, nn, dcam)
    act = mi.Bool(np.ascontiguousarray(stat.ravel()))
    # ---- A: direct visibility change
    l0 = ev(c0.light_vis(p, nn, lt)); l1 = ev(c1.light_vis(p, nn, lt))
    ach = mi.Bool(False)
    for a, b in zip(l0, l1):
        ach |= a ^ b
    A = np.asarray(ach).reshape(res, res) & stat
    # ---- exposure u
    R = MK['exposure_rays']; chunk = 64
    cnt = np.zeros(n)
    idx = dr.arange(U, n * chunk) % n
    gp = lambda v: dr.gather(type(v), v, idx)
    pe, ne, ae = gp(p), gp(nn), gp(act)
    for it in range(R // chunk):
        rng = mi.PCG32(size=n * chunk, initstate=dr.opaque(mi.UInt64, 86900 + it), initseq=mi.UInt64(0))
        w = mi.Frame3f(ne).to_world(mi.warp.square_to_cosine_hemisphere(mi.Point2f(rng.next_float32(), rng.next_float32())))
        ch = dr.select(changed(c0, c1, lt, pe + ne * 1e-4, w, ae), F(1.0), F(0.0))
        dr.eval(ch); dr.sync_thread()
        buf = dr.zeros(F, n); dr.scatter_reduce(dr.ReduceOp.Add, buf, ch, idx); cnt += np.asarray(buf, np.float64)
    u = (cnt / R).reshape(res, res)
    # ---- sc: mirror cone on panel / sphere receivers
    sid = g0['sid']
    cone = np.where(sid == 5, MK['mirror_cone_rad']['panel'], MK['mirror_cone_rad']['sphere'])
    recv_sm = (sid == 5) | (sid == S.SID_SPHERE)
    ns = MK['mirror_rays']
    idx2 = dr.arange(U, n * ns) % n
    g2 = lambda v: dr.gather(type(v), v, idx2)
    p2, n2, d2 = g2(p), g2(nn), g2(dcam)
    a2 = g2(mi.Bool(np.ascontiguousarray((stat & recv_sm).ravel())))
    cc = dr.cos(g2(PT.fl(cone)))
    rdir = d2 - 2.0 * dr.dot(d2, n2) * n2
    rng = mi.PCG32(size=n * ns, initstate=dr.opaque(mi.UInt64, 86950), initseq=mi.UInt64(0))
    u1, u2 = rng.next_float32(), rng.next_float32()
    ct = 1.0 - u1 * (1.0 - cc); st = dr.sqrt(dr.maximum(1.0 - ct * ct, 0.0)); ph = u2 * (2 * math.pi)
    w2 = mi.Frame3f(rdir).to_world(mi.Vector3f(st * dr.cos(ph), st * dr.sin(ph), ct))
    a2 &= dr.dot(w2, n2) > 1e-3
    ch = dr.select(changed(c0, c1, lt, p2 + n2 * 1e-4, w2, a2), F(1.0), F(0.0))
    dr.eval(ch); dr.sync_thread()
    buf = dr.zeros(F, n); dr.scatter_reduce(dr.ReduceOp.Add, buf, ch, idx2)
    sc = (np.asarray(buf, np.float64) / ns).reshape(res, res)
    # ---- fc: light -> panel (mirror) -> receiver path enabled / blocked
    pc = FX['panel']['center']; ph_ = FX['panel']['half']
    fc = mi.Bool(False)
    for l in lights:
        Lp = mi.Point3f(*[float(v) for v in l['pos']])
        Lm = mi.Point3f(2 * pc[0] - l['pos'][0], l['pos'][1], l['pos'][2])
        den = p.x - Lm.x
        t = (pc[0] - Lm.x) * dr.rcp(dr.select(dr.abs(den) < 1e-9, 1e-9, den))
        q = Lm + t * (p - Lm)
        ok = act & (p.x > pc[0] + 1e-3) & (t > 0) & (t < 1) & (dr.abs(q.y - pc[1]) <= ph_[0]) & (dr.abs(q.z - pc[2]) <= ph_[1]) & (dr.dot(nn, q - p) > 1e-6)
        if 'aim' in l:
            ax = np.array(l['aim'], np.float64) - np.array(l['pos']); ax /= np.linalg.norm(ax)
            ok &= dr.dot(dr.normalize(q - Lp), mi.Vector3f(*[float(v) for v in ax])) > math.cos(math.radians(l['cone_deg']))
        dr.eval(q, ok)
        v0 = ev(c0.seg_clear(q, Lp, ok) & c0.seg_clear(p, q, ok))
        v1 = ev(c1.seg_clear(q, Lp, ok) & c1.seg_clear(p, q, ok))
        fc |= v0 ^ v1
    fc = np.asarray(fc).reshape(res, res) & stat
    Bset = stat & ~A
    C = Bset & (fc | (recv_sm & (sc >= MK['sc_min'])))
    broad = np.isin(sid, [0, 1, 2, 3, 4, S.SID_BOXA, S.SID_BOXB])
    Dm = Bset & ~C & broad & (u >= MK['u_hi'])
    E = Bset & ~C & (u <= MK['u_lo'])
    m = {'frame': g0['ok'] | g1['ok'], 'stationary_all': stat, 'A': A, 'Bset': Bset, 'C': C, 'D': Dm, 'E': E, 'B_other': Bset & ~C & ~Dm & ~E,
         'mover': np.isin(g0['sid'], moved) | np.isin(g1['sid'], moved), 'C_footprint': Bset & fc, 'C_mirror': Bset & recv_sm & (sc >= MK['sc_min']) & ~fc}
    for name, ids in [('floor', [0]), ('ceiling', [1]), ('walls', [2, 3, 4]), ('panel', [5]), ('sphere', [6]), ('boxes_static', [S.SID_BOXA, S.SID_BOXB])]:
        m['obj_' + name] = stat & np.isin(sid, ids)
    return m, {'u': u.astype(np.float32), 'sc': sc.astype(np.float32), 'sid': sid.astype(np.int16), 'pid': g0['pid'].astype(np.int32)}


def main():
    root, ename = sys.argv[1], sys.argv[2]
    opt = {a.split('=')[0]: a.split('=')[1] for a in sys.argv[3:] if '=' in a}
    m, aux = compute(ename, opt.get('--ls'), int(opt['--res']) if '--res' in opt else None)
    out = os.path.join(root, 'results', 'masks'); os.makedirs(out, exist_ok=True)
    tag = ename + ('_' + opt['--ls'] if '--ls' in opt else '')
    np.savez_compressed(os.path.join(out, f'{tag}.npz'), **m, **aux)
    cnt = {k: int(v.sum()) for k, v in m.items()}
    json.dump(cnt, open(os.path.join(out, f'{tag}.json'), 'w'), indent=1)
    print(tag, json.dumps(cnt), flush=True)


if __name__ == '__main__':
    main()
    os._exit(0)
