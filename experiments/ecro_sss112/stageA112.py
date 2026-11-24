"""WL112 Stage A: tracer / pinning identity checks and the camera-independent constructions (contract v3) on F0.

  python stageA112.py ROOT        -> ROOT/stageA/{identity.json, C_r*.npz, N_r*.npz, legacy_r*.npz, frame_*.npz, synthetic.npz}
Every output is written atomically; an existing output is skipped (resumable).
"""
import json, os, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common112 as C  # noqa: E402
import tracer112 as T  # noqa: E402

mi, dr, F, U, PT = T.mi, T.dr, T.F, T.U, T.PT
PA = C.P['stage_A']


def save(acc, path, meta):
    tmp = path.with_name(path.stem + '.tmp.npz')
    np.savez_compressed(tmp, **{k: np.asarray(v, np.float32 if k not in ('bad_origin',) else np.float64) for k, v in acc.items()},
                        meta=json.dumps(meta))
    os.replace(tmp, path)


def identity(geom, light):
    """A1 (tracer copy == pt100 under N) and A2 (analytic pinned == native first hit with the native frame)."""
    out = {}
    cam = dict(C.V0_CAM, res=128)
    ref = T.pt100.Ctx({k: v for k, v in geom.items() if k != 'S'}, 'sss')
    ctx = T.Ctx112(geom, 'N')
    for name in ('RT', 'PTB', 'BX', 'PMATU'):
        out[f'A1_table_{name}_equal'] = bool(np.array_equal(np.asarray(getattr(ref, name)), np.asarray(getattr(ctx, name))))
    res, spp, seed = 128, 4, 11206000 + C.SO
    n = res * res
    B = PT.fl(np.random.default_rng(11206000).random(ref.N * T.NB * 3).astype(np.float32))
    mx = 0.
    for mode, kmax in (('cam', 6), ('ecro', 2)):
        sd = dr.opaque(mi.UInt64, seed * 1000003)
        rng = mi.PCG32(size=n * spp, initstate=sd, initseq=mi.UInt64(0)); ray, pix = PT.gen_rays(cam, res, spp, rng)
        o1, _ = ref.trace(ray, n * spp, rng, sd, 'ER', kmax, light, mode, B=B); dr.eval(o1); dr.sync_thread()
        rng = mi.PCG32(size=n * spp, initstate=sd, initseq=mi.UInt64(0)); ray, pix = PT.gen_rays(cam, res, spp, rng)
        o2, cap = ctx.trace112(ray, n * spp, rng, sd, kmax, light, mode); dr.eval(o2, cap); dr.sync_thread()
        for k in o1:
            if k.startswith('hl'):
                c = int(k[2]); act = cap['act']; th = (cap['thr'].x, cap['thr'].y, cap['thr'].z)[c]
                v2 = dr.select(act, th * dr.gather(F, B, (cap['pid'] * T.NB + cap['bin']) * 3 + c, act), 0.)
            else:
                v2 = o2[k]
            mx = max(mx, float(dr.max(dr.abs(o1[k] - v2))[0]))
    out['A1_lane_maxabs'] = mx
    del ref
    # A2
    ctxc = T.Ctx112(geom, 'C')
    Np = 8192
    rs = np.random.default_rng(11206001)
    pid = U(rs.integers(0, ctxc.N, Np).astype(np.uint32)); bn = U(rs.integers(0, 64, Np).astype(np.uint32))
    a = F(rs.uniform(.02, .98, Np).astype(np.float32)); b = F(rs.uniform(.02, .98, Np).astype(np.float32))
    t, bv, nf = ctxc.patch_frame(pid)
    loc = ctxc.dir_in_bin(bn, F(rs.uniform(.02, .98, Np).astype(np.float32)), F(rs.uniform(.02, .98, Np).astype(np.float32)))
    w = t * loc.x + bv * loc.y + nf * loc.z
    si, sid, nn = ctxc.first_si(pid, a, b, w)
    ray = mi.Ray3f(si.p + w * 2e-3, -w)
    nat = ctxc.scene.ray_intersect(ray)
    same = (nat.is_valid() & dr.eq(ctxc.shape_id(nat), sid) & (dr.norm(nat.p - si.p) < 4e-6) & (nat.to_local(w).z > 1e-6)
            & ~ctxc.buried(si.p, sid, nn))
    si.sh_frame = mi.Frame3f(dr.select(same, nat.sh_frame.s, si.sh_frame.s), dr.select(same, nat.sh_frame.t, si.sh_frame.t),
                             dr.select(same, nat.sh_frame.n, si.sh_frame.n))
    si.wi = si.to_local(w)
    rr = lambda: mi.PCG32(size=Np, initstate=mi.UInt64(11206002), initseq=mi.UInt64(0))
    o_pin, _ = ctxc.trace112(ray, Np, rr(), mi.UInt64(1), 5, light, 'probe', first=si, first_ok=same); dr.eval(o_pin)
    o_nat, _ = ctxc.trace112(ray, Np, rr(), mi.UInt64(1), 5, light, 'probe', first=nat, first_ok=same); dr.eval(o_nat)
    err = en = 0.
    for k in ('f20', 'f21', 'f22', 'fh0', 'fh1', 'fh2'):
        err += float(dr.sum(dr.abs(dr.select(same, o_pin[k] - o_nat[k], 0.)))[0])
        en += float(dr.sum(dr.abs(dr.select(same, o_nat[k], 0.)))[0])
    out['A2_paired'] = int(np.asarray(same).sum()); out['A2_rel_l1'] = err / max(en, 1e-30)
    out['A2_normal_maxdiff'] = float(dr.max(dr.select(same, dr.norm(nat.n - nn), 0.))[0])
    return out


def main(root):
    root = Path(root); out = root / 'stageA'; out.mkdir(parents=True, exist_ok=True)
    geom, light_cfg = C.fixture('F0')
    light = PT.Lights(light_cfg)
    status = out / 'stageA.status'
    if not (out / 'identity.json').exists():
        t0 = time.perf_counter(); ident = identity(geom, light); ident['seconds'] = time.perf_counter() - t0
        C.jd(ident, out / 'identity.json'); print('IDENTITY', json.dumps(ident), flush=True)
    n = C.dv(PA['proposals_per_subcell'], 32); b = PA['launch_subcell_batch']; so = C.SO; sm = C.dv(256, 16)
    jobs = [('C', r, 11201001 + r + so, n, {}) for r in range(C.RC)]
    jobs += [('N', r, 11202001 + r + so, n, {}) for r in range(C.RN)]
    jobs += [('legacy', r, 11203001 + r + so, sm, {'legacy': True}) for r in range(2)]
    jobs += [('frame_normal', 0, 11204001 + so, sm, {}), ('frame_patch', 0, 11204001 + so, sm, {'frame': 'patch'})]
    ctxs = {}
    timing = {}
    for kind, r, seed, nn, kw in jobs:
        path = out / (f'{kind}_r{r}.npz' if kind in ('C', 'N', 'legacy') else f'{kind}.npz')
        if path.exists():
            continue
        sem = 'N' if kind in ('N', 'legacy') else 'C'
        if sem not in ctxs:
            ctxs[sem] = T.Ctx112(geom, sem)
        t0 = time.perf_counter()
        acc = ctxs[sem].construct(light, nn, seed, b=b if not kw.get('legacy') else 16, **kw)
        dt = time.perf_counter() - t0
        timing[path.name] = dt
        save(acc, path, {'kind': kind, 'replica': r, 'seed': seed, 'proposals_per_subcell': nn, 'semantics': sem, 'seconds': dt, **kw})
        print('CONSTRUCT', path.name, round(dt, 2), 's', {k: float(np.sum(v)) for k, v in acc.items()
                                                          if k in ('U_w', 'G_w', 'buried', 'singular', 'mismatch', 'bad_origin', 'misroute', 'backcount')}, flush=True)
    # A4 synthetic: analytic radiance, receivers forced (exact quadrature comparison in the analysis)
    path = out / 'synthetic.npz'
    if not path.exists():
        def L(p, w, nn):
            v = 1. + 0.5 * p.x + 0.3 * p.y + 0.4 * dr.dot(w, nn) + 0.2 * w.x
            return mi.Color3f(v, v, v)
        t0 = time.perf_counter()
        acc = ctxs['C'].construct(light, C.dv(256, 64), 11205001 + C.SO, b=32, synthetic=L)
        save(acc, path, {'kind': 'synthetic', 'seed': 11205001 + C.SO, 'proposals_per_subcell': C.dv(256, 64), 'seconds': time.perf_counter() - t0})
        print('SYNTHETIC done', flush=True)
    C.jd({'timing': timing, 'done': True}, out / 'stageA_done.json')
    status.write_text('DONE')


if __name__ == '__main__':
    main(sys.argv[1])
    sys.stdout.flush(); os._exit(0)
