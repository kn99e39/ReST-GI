"""WL114A (port of gen102 to pt114a / Mitsuba 3.6.4, deep targets). Original docstring:
WL102 ECRO dataset on the fixed LR3 geometry (WL100 split sizes, physical targets, point lights in free space).

  python gen102.py <scene.xml> <out_root> [split ...]       splits: train (128 x 4 lights, 64 spp), val (16 x 4, 2 x 128 spp),
                                                            test (1 x 2 held-out control lights, 2 x 256 spp)
Writes <out_root>/geometry.npz (patch tables, state rays, encoder rows, lights of every split), <out_root>/<split>/g###.npz
(s, f2, fh [+ f2b, fhb]) and <out_root>/gen_<split>.json (timing). Nothing here reads a reference image or the evaluation light.

Light rule (protocol wl102.json arms.ECRO.training_data): white point lights (ECRO fixture intensity), uniform in the scene's
bounding box, kept if no surface is within 0.15 m along 26 directions and the point is not inside a closed mesh (first hit along +y
is not a back face of a closed object); rejected within 0.3 m of the evaluation light (the scene emitter's centroid) or of a
held-out control light. Control lights: the first 2 accepted points of seed 102900 (drawn before any other light).
"""
import json, math, os, sys, time
import numpy as np
import pt114a as P

S, mi, dr = P.S, P.mi, P.dr
NB = P.NB
SPLITS = {'train': {'items': 128, 'lights': 4, 'spp': 64, 'halves': 1, 'seed': 102100},
          'val': {'items': 16, 'lights': 4, 'spp': 128, 'halves': 2, 'seed': 102200},
          'test': {'items': 1, 'lights': 2, 'spp': 256, 'halves': 2, 'seed': 102300}}
CLEAR, EXCL = 0.15, 0.3
INTENSITY = S.FX['light_intensity']


KSTAR = int(os.environ.get('WL114A_KSTAR', '96'))
SPP_SCALE = os.environ.get('WL114A_DEV') == '1'


def deep_targets(ctx, L, spp, seed):
    """Deep probe (camera events up to K*): D6 targets f2 / fh and full-depth fhF = fh + deep buckets <= K*, same samples."""
    if SPP_SCALE:
        spp = max(2, spp // 16)
    fd, cnt = ctx.field_deep(L, spp, seed, KSTAR - 1, b=16)
    fhF = fd['fh'] + sum((fd[n] for n in P.deep_through(KSTAR) if n in fd), np.zeros_like(fd['fh']))
    return fd['f2'], fd['fh'], fhF.astype(np.float32)


def dirs26():
    d = np.array([[x, y, z] for x in (-1, 0, 1) for y in (-1, 0, 1) for z in (-1, 0, 1) if (x, y, z) != (0, 0, 0)], np.float64)
    return d / np.linalg.norm(d, axis=1, keepdims=True)


def free_space(ctx, pts):
    """Bool per point: clearance >= CLEAR along 26 directions, and not inside a closed object."""
    D = dirs26()
    n = len(pts)
    o = np.repeat(pts, len(D), 0); d = np.tile(D, (n, 1))
    ray = mi.Ray3f(mi.Point3f(*[mi.Float(o[:, k].astype(np.float32)) for k in range(3)]),
                   mi.Vector3f(*[mi.Float(d[:, k].astype(np.float32)) for k in range(3)]))
    si = ctx.scene.ray_intersect(ray)
    t = np.where(np.asarray(si.is_valid()), np.asarray(si.t), np.inf).reshape(n, len(D))
    clear = t.min(1) >= CLEAR
    up = mi.Ray3f(mi.Point3f(*[mi.Float(pts[:, k].astype(np.float32)) for k in range(3)]), mi.Vector3f(0, 1, 0))
    su = ctx.scene.ray_intersect(up)
    sid = np.asarray(ctx.shape_id(su))
    back = np.asarray(su.is_valid()) & (np.asarray(dr.dot(su.n, mi.Vector3f(0, 1, 0))) > 0)
    closed = np.array([o_['closed'] for o_ in ctx.G.objects])
    inside = back & closed[np.clip(sid, 0, len(closed) - 1)]
    return clear & ~inside


def sample_lights(ctx, rng, count, avoid):
    lo, hi = np.array(ctx.scene.bbox().min), np.array(ctx.scene.bbox().max)
    out = []
    while len(out) < count:
        c = rng.uniform(lo, hi, (4096, 3))
        ok = free_space(ctx, c)
        for p in c[ok]:
            if all(np.linalg.norm(p - a) >= EXCL for a in avoid + out):
                out.append(p)
                if len(out) == count:
                    break
    return [np.asarray(p).tolist() for p in out]


def emitter_centroid(ctx):
    for (s, sid, V, F, two, emit) in ctx.meshes:
        if emit:
            return V.mean(0)
    raise RuntimeError('no emitter mesh')


def encoder_rows(ctx, pts_per_patch=32, seed=4):
    """69-float encoder feature rows per patch (features only; the physical dispatcher evaluates the Mitsuba BSDF)."""
    N = ctx.N
    lane = dr.arange(mi.UInt32, N * pts_per_patch); pid = lane // pts_per_patch
    rng = mi.PCG32(size=N * pts_per_patch, initstate=mi.UInt64(seed), initseq=mi.UInt64(0))
    p, n, _ = ctx.patch_point(pid, rng.next_float32(), rng.next_float32())
    si, sid, pp, ok = ctx.surface(mi.Ray3f(p + n * 1e-3, -n), mi.Bool(True))
    alb = si.bsdf().eval_diffuse_reflectance(si, ok)
    ch = [np.asarray(dr.select(ok, c, 0.0), np.float64) for c in (alb.x, alb.y, alb.z)]
    okn = np.asarray(ok, np.float64)
    cnt = np.bincount(np.asarray(pid), weights=okn, minlength=N)
    diff = np.stack([np.bincount(np.asarray(pid), weights=c, minlength=N) for c in ch], 1) / np.maximum(cnt, 1)[:, None]
    rows = np.zeros((N, 69), np.float32)
    rows[:, 24:38] = .3; rows[:, 38:45] = .5; rows[:, 53:] = 1
    obj_spec = []
    for (s, sid, V, Fc, two, emit) in ctx.meshes:
        b = s.bsdf(); name = str(b); p_ = mi.traverse(b)
        val = lambda key: np.atleast_1d(np.asarray(p_[key], np.float64)).reshape(-1) if key in p_.keys() else None
        alpha = next((val(k) for k in p_.keys() if k.endswith('alpha') or k.endswith('alpha.value')), None)
        w = np.zeros(3); a = 0.3
        if 'RoughPlastic' in name:
            eta = float(np.ravel(next(val(k) for k in p_.keys() if k.endswith('eta')))[0])   # numpy 2: 1-element array -> scalar
            w[:] = ((eta - 1) / (eta + 1)) ** 2; a = float(alpha[0])
        elif 'RoughConductor' in name:
            eta = next(val(k) for k in p_.keys() if k.endswith('eta.value')); kk = next(val(k) for k in p_.keys() if k.endswith('k.value'))
            sr = next((val(k) for k in p_.keys() if k.endswith('specular_reflectance.value')), np.ones(3))
            w = np.broadcast_to(sr, (3,)) * (((eta - 1) ** 2 + kk ** 2) / ((eta + 1) ** 2 + kk ** 2)); a = float(alpha[0])
        obj_spec.append((w, a, name.split('[')[0]))
    for i in range(N):
        w, a, _ = obj_spec[ctx.G.POBJ[i]]
        for c in range(3):
            rows[i, c * 8] = diff[i, c]; rows[i, c * 8 + 1] = w[c]
        rows[i, 24] = rows[i, 31] = a; rows[i, 38] = float(np.mean(w))
        tot = diff[i].mean() + w.mean()
        rows[i, 45] = diff[i].mean() / tot if tot > 0 else 1.0; rows[i, 46] = 1 - rows[i, 45]
    return rows, [{'object': ctx.G.objects[j]['id'], 'bsdf': o[2], 'spec_weight': o[0].tolist(), 'alpha': o[1]} for j, o in enumerate(obj_spec)]


def main():
    xml, root = sys.argv[1], sys.argv[2]
    splits = sys.argv[3:] or ['train', 'val', 'test']
    if SPP_SCALE:
        for c_ in SPLITS.values():
            c_['items'] = min(c_['items'], 3)
    os.makedirs(root, exist_ok=True)
    ctx = P.Ctx(P.load_scene(xml))
    gfile = os.path.join(root, 'geometry.npz')
    if not os.path.exists(gfile):
        t0 = time.perf_counter()
        hit, dbin, _ = ctx.state_rays()
        t_state = time.perf_counter() - t0
        rows, spec = encoder_rows(ctx)
        ev = emitter_centroid(ctx)
        ctl = sample_lights(ctx, np.random.default_rng(102900), 2, [ev])
        lights = {}
        for sp, c in SPLITS.items():
            if sp == 'test':
                lights[sp] = [ctl]
            else:
                rng = np.random.default_rng(c['seed'])
                L = sample_lights(ctx, rng, c['items'] * c['lights'], [ev] + ctl)
                lights[sp] = [L[i * c['lights']:(i + 1) * c['lights']] for i in range(c['items'])]
        g = ctx.G
        np.savez_compressed(gfile, pc=g.PC, pn=g.PN, pt=g.PT, parea=g.PAREA, pmat=g.PMAT, pobj=g.POBJ, pside=g.PSIDE, hit=hit, dbin=dbin,
                            rows=rows, lights=json.dumps(lights), evaluation_light_centroid=ev, state_seconds=t_state)
        json.dump({'N': g.N, 'objects_spec': spec, 'state_rays_s': t_state, 'control_lights': ctl, 'evaluation_light_centroid': ev.tolist(),
                   'light_intensity': INTENSITY, 'layout': g.summary()}, open(os.path.join(root, 'geometry.json'), 'w'), indent=1)
    z = np.load(gfile)
    lights = json.loads(str(z['lights']))
    for sp in splits:
        c = SPLITS[sp]
        od = os.path.join(root, sp); os.makedirs(od, exist_ok=True)
        t0 = time.perf_counter(); times = []
        for i, ls in enumerate(lights[sp]):
            fn = os.path.join(od, f'g{i:03d}.npz')
            if os.path.exists(fn):
                continue
            ti = time.perf_counter()
            s, f2, fh, fhF, f2b, fhb, fhFb = [], [], [], [], [], [], []
            for j, pos in enumerate(ls):
                L = P.Lights([{'pos': pos, 'rgb': [INTENSITY] * 3}])
                s.append(ctx.source(L))
                seed = c['seed'] * 1000 + i * 10 + j
                a, b, cF = deep_targets(ctx, L, c['spp'], seed)
                f2.append(a); fh.append(b); fhF.append(cF)
                if c['halves'] == 2:
                    a, b, cF = deep_targets(ctx, L, c['spp'], seed + 500000)
                    f2b.append(a); fhb.append(b); fhFb.append(cF)
            d = {'lights': json.dumps(ls), 's': np.stack(s), 'f2': np.stack(f2), 'fh': np.stack(fh), 'fhF': np.stack(fhF), 'K_star': KSTAR}
            if f2b:
                d.update(f2b=np.stack(f2b), fhb=np.stack(fhb), fhFb=np.stack(fhFb))
            assert all(np.isfinite(v).all() for v in d.values() if isinstance(v, np.ndarray)), 'STOP_NONFINITE_TARGET'
            np.savez(fn + '.tmp.npz', **d); os.replace(fn + '.tmp.npz', fn)
            times.append(time.perf_counter() - ti)
            print('GEN', sp, i, round(times[-1], 2), flush=True)
        json.dump({'split': sp, **c, 'seconds': time.perf_counter() - t0, 'item_seconds': times}, open(os.path.join(root, f'gen_{sp}.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
