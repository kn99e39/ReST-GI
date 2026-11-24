"""WL112 development smoke (throwaway seeds 112990xx; never evidence). Checks the tracer copy against pt100 and probes speed."""
import json, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import tracer112 as T  # noqa: E402

mi, dr, F, U, PT = T.mi, T.dr, T.F, T.U, T.PT
cfg = json.loads((T.EXP.parent / 'results/ecro_material100/configs.json').read_text())['sss']['test'][0]
geom = cfg['geom']; light = PT.Lights(cfg['lights'][0])
cam = dict(json.loads((T.EXP / 'ecro_material100/protocol.json').read_text())['datasets']['sss']['camera'], res=128)
out = {}

# 1. tables and camera trace identical to pt100 under native semantics
ref = T.pt100.Ctx(geom, 'sss')
ctx = T.Ctx112(geom, 'N')
for name in ('RT', 'PTB', 'BX', 'PMATU'):
    a, b = np.asarray(getattr(ref, name)), np.asarray(getattr(ctx, name))
    assert np.array_equal(a, b), name
res = 128; spp = 4; seed = 11299001
n = res * res
sd = dr.opaque(mi.UInt64, seed * 1000003)
def rays():
    rng = mi.PCG32(size=n * spp, initstate=sd, initseq=mi.UInt64(0))
    ray, pix = PT.gen_rays(cam, res, spp, rng)
    return rng, ray, pix
for mode, kmax in (('cam', 6), ('ecro', 2)):
    B = PT.fl(np.random.default_rng(1).random(ref.N * T.NB * 3).astype(np.float32))
    rng, ray, pix = rays()
    o1, _ = ref.trace(ray, n * spp, rng, sd, 'ER', kmax, light, mode, B=B)
    dr.eval(o1); dr.sync_thread()
    rng, ray, pix = rays()
    o2, cap = ctx.trace112(ray, n * spp, rng, sd, kmax, light, mode)
    dr.eval(o2, cap); dr.sync_thread()
    for k in o1:
        if k.startswith('hl'):
            c = int(k[2]); act = cap['act']; th = (cap['thr'].x, cap['thr'].y, cap['thr'].z)[c]
            v2 = dr.select(act, th * dr.gather(F, B, (cap['pid'] * T.NB + cap['bin']) * 3 + c, act), 0.)
        else:
            v2 = o2[k]
        d = float(dr.max(dr.abs(o1[k] - v2))[0])
        out[f'{mode}_{k}_maxabs'] = d
        assert d == 0.0, (mode, k, d)
print('pt100 identity OK', flush=True)

# 2. construction speed and labels (C semantics)
ctxc = T.Ctx112(geom, 'C')
for nsub, bb in ((8, 8),):
    t0 = time.perf_counter()
    acc = ctxc.construct(light, nsub, 11299010, b=bb)
    dt = time.perf_counter() - t0
    tot = {k: float(v.sum()) for k, v in acc.items() if not k.endswith(('f20', 'f21', 'f22', 'fh0', 'fh1', 'fh2', 'sq0', 'sq1', 'sq2'))}
    out[f'construct_{nsub}'] = {'seconds': dt, 'samples': nsub * ctxc.N * 4 * 64, 'totals': tot}
    print('construct', nsub, dt, json.dumps(tot), flush=True)
Tu, wu, _, _ = T.tables_from(acc, 'U', 'coarse')
Tg, wg, _, _ = T.tables_from(acc, 'G', 'coarse')
out['U_mean'] = float(Tu.mean()); out['G_mean'] = float(Tg.mean())
print('tables', Tu.mean(), Tg.mean(), (wu > 0).mean(), (wg > 0).mean(), flush=True)

# 3. analytic pinned vs native first hit suffix (same RNG)
N = 4096
rs = np.random.default_rng(7)
pid = U(rs.integers(0, ctxc.N, N).astype(np.uint32)); bn = U(rs.integers(0, 64, N).astype(np.uint32))
a = F(rs.uniform(.05, .95, N).astype(np.float32)); b = F(rs.uniform(.05, .95, N).astype(np.float32))
t, bv, nf = ctxc.patch_frame(pid)
loc = ctxc.dir_in_bin(bn, F(rs.uniform(.05, .95, N).astype(np.float32)), F(rs.uniform(.05, .95, N).astype(np.float32)))
w = t * loc.x + bv * loc.y + nf * loc.z
si, sid, nn = ctxc.first_si(pid, a, b, w)
ray = mi.Ray3f(si.p + w * 2e-3, -w)
nat = ctxc.scene.ray_intersect(ray)
same = nat.is_valid() & dr.eq(ctxc.shape_id(nat), sid) & (dr.norm(nat.p - si.p) < 4e-6) & (nat.to_local(w).z > 1e-6) & ~ctxc.buried(si.p, sid, nn)
si.sh_frame = mi.Frame3f(dr.select(same, nat.sh_frame.s, si.sh_frame.s), dr.select(same, nat.sh_frame.t, si.sh_frame.t), dr.select(same, nat.sh_frame.n, si.sh_frame.n))
si.wi = si.to_local(w)
out['frame_n_maxdiff'] = float(dr.max(dr.select(same, dr.norm(nat.sh_frame.n - nn), 0.))[0])
def rng2():
    return mi.PCG32(size=N, initstate=mi.UInt64(11299020), initseq=mi.UInt64(0))
o_pin, c_pin = ctxc.trace112(ray, N, rng2(), mi.UInt64(1), 5, light, 'probe', first=si, first_ok=same)
o_nat, c_nat = ctxc.trace112(ray, N, rng2(), mi.UInt64(1), 5, light, 'probe', first=nat, first_ok=same)
err = 0.; en = 0.; mx = 0.
for k in ('f20', 'f21', 'f22', 'fh0', 'fh1', 'fh2'):
    d = dr.abs(dr.select(same, o_pin[k] - o_nat[k], 0.))
    err += float(dr.sum(d)[0]); en += float(dr.sum(dr.abs(dr.select(same, o_nat[k], 0.)))[0]); mx = max(mx, float(dr.max(d)[0]))
out['pinned_vs_native'] = {'paired': int(np.asarray(same).sum()), 'rel_l1': err / en, 'max_abs': mx}
print('pinned vs native', out['pinned_vs_native'], flush=True)
DEV = T.EXP.parent / 'results/ecro_sss112/dev'; DEV.mkdir(parents=True, exist_ok=True)
(DEV / 'dev_smoke112.json').write_text(json.dumps(out, indent=1))
import os
os._exit(0)
