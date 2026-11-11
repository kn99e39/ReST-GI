"""WL88 geometry / visibility-only region masks for the glossy-mover family (no radiance, no candidate output, no reference error).

Mirror footprints (geometry only): photons leave each light (uniform over the sphere / inside the spot cone), are reflected
SPECULARLY (about the surface normal) at S1 surfaces (panel, glossy mover G) and land on the first non-S1 surface after >= 1 such
reflection. The sequence of S1 surfaces gives the code: P (panel only), G (G only), PG (panel then G), GP (G then panel), M3 (>= 3
specular reflections). Landing points are binned into 2 cm voxels and dilated by `fp_dilate` voxels. The S1 lobe (alpha ~ 0.05) blurs
the real footprint slightly; the dilation covers it. No BSDF value or radiance is used.

Per pixel (centre ray), with the T0 / T1 scenes of a workload and its lights:
  stationary   same surface point in T0 and T1, not on the mover (as WL86)
  A            stationary, direct light visibility differs (as WL86)
  C_ss         stationary broad receiver x1 inside a two-mirror footprint (PG or GP) in exactly one of T0 / T1:
               its higher-order radiance contains light -> S1 -> S1 -> x1 (a 3-event camera path that reads the table at the S1 vertex
               next to x1: sharp-to-sharp transport in the table)
  R_mirror     stationary panel pixels: fraction of the mirror cone (WL86 panel cone) whose hit x2 lies on G (in T0 or T1) or on a
               stationary point whose G-involving footprint status (G, PG, GP, M3) differs between T0 and T1, >= sc_min:
               a sharp reader whose output is the table entry of a surface lit by the moving sharp sender
  D, E         as WL86 (broad exposure u from first-hit / direct-visibility change), excluding C_ss and R_mirror
"""
import json, math, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fx88  # noqa: E402
PT = fx88.PT
mi, dr, S = PT.mi, PT.dr, PT.S
F, U = PT.F, PT.U
FX = PT.FX
sys.path.insert(0, fx88.fx87.E86)
import masks86 as M86  # noqa: E402
MK = S.P['regions']
MR = fx88.P88['regions']
VOX = MR['fp_voxel_m']
LO = np.array([-2.0, -2.0, 0.0]); DIMS = np.ceil((np.array([2.0, 2.0, 3.0]) - LO) / VOX).astype(int) + 1
CODES = {'P': 0, 'G': 1, 'PG': 2, 'GP': 3, 'M3': 4}


def vox_id(p):
    ijk = np.clip(np.floor((p - LO) / VOX).astype(np.int64), 0, DIMS - 1)
    return (ijk[..., 0] * DIMS[1] + ijk[..., 1]) * DIMS[2] + ijk[..., 2]


def footprint(ctx, lights, nph, seed, chunk=4_000_000):
    """Boolean voxel maps [5, nvox] (codes P, G, PG, GP, M3), dilated."""
    nvox = int(np.prod(DIMS))
    cnt = np.zeros((5, nvox), np.int32)
    lt = PT.Lights(lights)
    done, it = 0, 0
    while done < nph:
        n = min(chunk, nph - done)
        for l in range(lt.n):
            rng = mi.PCG32(size=n, initstate=dr.opaque(mi.UInt64, seed * 1000003 + 97 * it + l), initseq=mi.UInt64(0))
            d = mi.warp.square_to_uniform_sphere(mi.Point2f(rng.next_float32(), rng.next_float32()))
            active = dr.dot(d, lt.axis[l]) > lt.cc[l]
            ray = mi.Ray3f(lt.pos[l], d)
            code = U(0); nsh = U(0)
            land = mi.Bool(False); lp = mi.Point3f(0.0)
            for b in range(5):
                si = ctx.scene.ray_intersect(ray, active)
                active &= si.is_valid()
                active &= dr.dot(-ray.d, si.n) > 1e-6
                mid = dr.gather(U, ctx.SMAT, ctx.shape_id(si))
                sharp = dr.eq(mid, 1) | dr.eq(mid, 4)
                stop = active & ~sharp
                newl = stop & (nsh > 0)
                lp = dr.select(newl, si.p, lp)
                land |= newl
                active &= sharp
                code = dr.select(active, code * 3 + dr.select(dr.eq(mid, 1), U(1), U(2)), code)
                nsh = dr.select(active, nsh + 1, nsh)
                dref = ray.d - 2.0 * dr.dot(ray.d, si.n) * si.n
                ray = si.spawn_ray(dref)
            dr.eval(land, lp, code, nsh); dr.sync_thread()
            ok = np.asarray(land)
            P = np.stack([np.asarray(lp[k]) for k in range(3)], 1)[ok].astype(np.float64)
            c = np.asarray(code)[ok]; ns = np.asarray(nsh)[ok]
            cls = np.where(ns >= 3, 4, np.where(c == 1, 0, np.where(c == 2, 1, np.where(c == 5, 2, np.where(c == 7, 3, 4)))))
            v = vox_id(P)
            for k in range(5):
                cnt[k] += np.bincount(v[cls == k], minlength=nvox).astype(np.int32)
        done += n; it += 1
    fp = cnt > 0
    r = MR['fp_dilate']
    if r > 0:
        g = fp.reshape(5, *DIMS)
        out = g.copy()
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                for dz in range(-r, r + 1):
                    out |= np.roll(np.roll(np.roll(g, dx, 1), dy, 2), dz, 3)
        fp = out.reshape(5, -1)
    return fp, cnt


def compute(wname, ls=None, res=None, cam=None, nph=None):
    fx88.setup(wname + '_T0')
    ex = FX[wname]
    cam = cam or FX['camera']; res = res or cam['res']
    c0, c1 = PT.Ctx(ex['T0']), PT.Ctx(ex['T1'])
    lights = FX['light_sets'][ls or ex['light_set']]
    nph = nph or MR['fp_photons']
    fp0, _ = footprint(c0, lights, nph, MR['fp_seed'])
    fp1, _ = footprint(c1, lights, nph, MR['fp_seed'])
    m, aux = M86.compute(wname, ls, res, cam)                     # WL86 stationary / A / u / sc and its C (panel family), D, E
    g0 = c0.gbuffer(cam, res)
    stat = m['stationary_all']
    sid = g0['sid']
    x1 = vox_id(g0['p'].reshape(-1, 3)).reshape(res, res)
    broad = np.isin(sid, [0, 1, 2, 3, 4, S.SID_BOXB])
    two = lambda fp: (fp[CODES['PG']] | fp[CODES['GP']])[x1]
    Css = stat & broad & (two(fp0) ^ two(fp1))
    # ---- R_mirror: panel readers
    n = res * res
    ns = MK['mirror_rays']
    P3 = lambda a: mi.Point3f(*[PT.fl(a[..., k]) for k in range(3)])
    V3 = lambda a: mi.Vector3f(*[PT.fl(a[..., k]) for k in range(3)])
    p, nn, dcam = P3(g0['p']), V3(g0['n']), V3(g0['d'])
    nn = dr.select(dr.dot(nn, dcam) < 0, nn, -nn)
    idx2 = dr.arange(U, n * ns) % n
    g2 = lambda v: dr.gather(type(v), v, idx2)
    p2, n2, d2 = g2(p), g2(nn), g2(dcam)
    a2 = g2(mi.Bool(np.ascontiguousarray((stat & (sid == 5)).ravel())))
    cc = math.cos(MK['mirror_cone_rad']['panel'])
    rdir = d2 - 2.0 * dr.dot(d2, n2) * n2
    rng = mi.PCG32(size=n * ns, initstate=dr.opaque(mi.UInt64, 88950), initseq=mi.UInt64(0))
    u1, u2 = rng.next_float32(), rng.next_float32()
    ct = 1.0 - u1 * (1.0 - cc); st_ = dr.sqrt(dr.maximum(1.0 - ct * ct, 0.0)); ph = u2 * (2 * math.pi)
    w2 = mi.Frame3f(rdir).to_world(mi.Vector3f(st_ * dr.cos(ph), st_ * dr.sin(ph), ct))
    a2 &= dr.dot(w2, n2) > 1e-3
    o2 = p2 + n2 * 1e-4
    dr.eval(o2, w2, a2)
    gi = [CODES['G'], CODES['PG'], CODES['GP'], CODES['M3']]
    hits = []
    for c in (c0, c1):
        v, sd, ph_, _, _ = c.hit_info(o2, w2, a2)
        dr.eval(v, sd, ph_); dr.sync_thread()
        hits.append((np.asarray(v), np.asarray(sd), np.stack([np.asarray(ph_[k]) for k in range(3)], 1).astype(np.float64)))
    (v0, s0, q0), (v1, s1, q1) = hits
    onG = (v0 & (s0 == S.SID_BOXA)) | (v1 & (s1 == S.SID_BOXA))
    vq = vox_id(q0)
    st0 = fp0[gi].any(0)[vq]; st1 = fp1[gi].any(0)[vq]
    chg = onG | (v0 & ~onG & (st0 ^ st1))
    act = np.asarray(a2)
    frac = np.bincount(np.asarray(idx2)[act], weights=chg[act].astype(np.float64), minlength=n) / ns
    frac = frac.reshape(res, res)
    Rm = stat & (sid == 5) & (frac >= MR['sc_min'])
    Bset = stat & ~m['A']
    Css &= Bset; Rm &= Bset
    D = Bset & ~Css & ~Rm & broad & (aux['u'] >= MK['u_hi'])
    E = Bset & ~Css & ~Rm & (aux['u'] <= MK['u_lo'])
    out = {'frame': m['frame'], 'stationary_all': stat, 'A': m['A'], 'Bset': Bset, 'C_ss': Css, 'R_mirror': Rm, 'D': D, 'E': E,
           'C86': m['C'], 'mover': m['mover']}
    for name, ids in [('floor', [0]), ('ceiling', [1]), ('walls', [2, 3, 4]), ('panel', [5]), ('sphere', [6]), ('boxB', [S.SID_BOXB])]:
        out['obj_' + name] = stat & np.isin(sid, ids)
    g1 = c1.gbuffer(cam, res)
    out['obj_G_T1'] = g1['ok'] & (g1['sid'] == S.SID_BOXA)
    for k, ci in CODES.items():
        out[f'fp_{k}_T0'] = fp0[ci][x1] & g0['ok']
        out[f'fp_{k}_T1'] = fp1[ci][vox_id(g1['p'].reshape(-1, 3)).reshape(res, res)] & g1['ok']
    return out, {'u': aux['u'], 'sc_frac_R': frac.astype(np.float32), 'sid': sid.astype(np.int16), 'pid': g0['pid'].astype(np.int32)}


def main():
    root, wname = sys.argv[1], sys.argv[2]
    opt = {a.split('=')[0]: a.split('=')[1] for a in sys.argv[3:] if '=' in a}
    m, aux = compute(wname, opt.get('--ls'), int(opt['--res']) if '--res' in opt else None, nph=int(opt['--nph']) if '--nph' in opt else None)
    out = os.path.join(root, 'results', 'masks'); os.makedirs(out, exist_ok=True)
    tag = wname + ('_' + opt['--ls'] if '--ls' in opt else '')
    np.savez_compressed(os.path.join(out, f'{tag}.npz'), **m, **aux)
    cnt = {k: int(v.sum()) for k, v in m.items()}
    json.dump(cnt, open(os.path.join(out, f'{tag}.json'), 'w'), indent=1)
    print(tag, json.dumps(cnt), flush=True)


if __name__ == '__main__':
    main()
    os._exit(0)
