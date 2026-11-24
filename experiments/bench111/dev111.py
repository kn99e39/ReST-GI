"""WL111 DEVELOPMENT smoke (not evidence): adapter validation against the WL100 finite-depth reference, Russian-roulette
unbiasedness at fixed depth, rough cost of full-depth references, OptiX denoiser call, ECRO / PT frame-time scale.

python dev111.py OUT CASE      (CASE = sss | near)

No comparison between methods against a reference is computed here; only adapter-vs-own-reference checks and timings.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

OUT = Path(sys.argv[1])
CASE = sys.argv[2]
OUT.mkdir(parents=True, exist_ok=True)
import pt111  # noqa: E402
import common111 as C  # noqa: E402

R, mi, dr, torch = pt111.R, pt111.mi, pt111.dr, pt111.torch
LOG = open(OUT / f'dev_{CASE}.log', 'a', encoding='utf-8')


def log(*a):
    s = time.strftime('%H:%M:%S ') + ' '.join(str(x) for x in a)
    print(s, flush=True)
    LOG.write(s + '\n')
    LOG.flush()


def build(case):
    cfg = json.loads((R.ROOT100 / 'configs.json').read_text())[case]['test'][0]
    model = R.load_model(case)
    rt = pt111.PT111(cfg['geom'], case, model, cfg['lights'][0])
    s = rt.source()
    B, _ = rt.respond(s)
    return rt, B, cfg


def sync():
    dr.sync_thread()
    torch.cuda.synchronize()


rec = {'case': CASE, 'versions': C.versions(mi, dr, torch), 'snapshot': C.gpu_snapshot()}
t0 = time.perf_counter()
rt, B, cfg = build(CASE)
Bd = R.PT.fl(B.detach().cpu().numpy())
rec['build_s'] = time.perf_counter() - t0
cam = dict(cfg['camera'], res=512)
RES = 512
ref = [np.load(R.ROOT100 / CASE / 'renders' / f'REF_0_{h}.npz') for h in (0, 1)]
refm = {k: 0.5 * (ref[0][k] + ref[1][k]) for k in ('e1', 'e2', 'h')}
refv = {k: 0.25 * (ref[0][k] - ref[1][k]) ** 2 for k in ('e1', 'e2', 'h')}
log('built', rec['build_s'])


def comp(acc, spp):
    a = acc / spp
    g = lambda i: np.stack([a[i + c].reshape(RES, RES) for c in range(3)], -1)
    return {'e1': g(0), 'e2': g(3), 'h': g(6), 'alb': g(9), 'nrm': g(12)}


# 1. adapter at the reference's own depth (kmax 6, no RR) vs the WL100 E1..E6 reference: means and noise-normalised RMSE
chk = {}
for name, kmax, rr in (('k6_norr', 6, 0), ('k6_rr4', 6, 4)):
    t = time.perf_counter()
    acc = pt111.pt_hq(rt, cam, RES, 1024, C.seed_of('dev111', CASE, name), kmax=kmax, rr=rr)
    sync()
    wall = time.perf_counter() - t
    c = comp(acc, 1024)
    row = {'wall_s_1024spp': wall}
    for k in ('e1', 'e2', 'h'):
        m_new, m_ref = c[k].reshape(-1, 3).mean(0), refm[k].reshape(-1, 3).mean(0)
        row[k] = {'mean_new': m_new.tolist(), 'mean_ref': m_ref.tolist(), 'rel_mean_diff': ((m_new - m_ref) / np.maximum(m_ref, 1e-12)).tolist(),
                  # half-to-half reference variance scaled to 1024 spp of the new image + the reference mean's own variance
                  'rmse_over_noise': float(np.sqrt(np.mean((c[k] - refm[k]) ** 2) /
                                                   np.mean(refv[k] * (8192 / 1024) + refv[k])))}
    chk[name] = row
    log(name, json.dumps(row))
rec['adapter_check'] = chk

# 2. full depth: energy beyond 6 events, cost per spp
for kmax in (256,):
    t = time.perf_counter()
    acc = pt111.pt_hq(rt, cam, RES, 256, C.seed_of('dev111', CASE, 'full', kmax), kmax=kmax, rr=4)
    sync()
    wall = time.perf_counter() - t
    c = comp(acc, 256)
    tot = (c['e1'] + c['e2'] + c['h']).reshape(-1, 3).mean(0)
    ref6 = (refm['e1'] + refm['e2'] + refm['h']).reshape(-1, 3).mean(0)
    rec[f'full_k{kmax}'] = {'wall_s_256spp': wall, 'mean_total': tot.tolist(), 'mean_total_D6_ref': ref6.tolist(),
                            'beyond_D6_share': ((tot - ref6) / tot).tolist()}
    log('full', kmax, json.dumps(rec[f'full_k{kmax}']))
    full = c

# 3. frame-time scale (not evidence): ECRO frame and PT full-depth frame, 1 and 4 spp, 20 warm frames each
tim = {}
for spp in (1, 4):
    for arm in ('ecro', 'pt'):
        f = (lambda s: rt.frame(Bd, cam, RES, spp, s)) if arm == 'ecro' else (lambda s: rt.pt_frame(cam, RES, spp, s))
        for w in range(3):
            f(C.seed_of('warm', arm, spp, w))
        sync()
        ts = []
        for i in range(20):
            sync()
            t = time.perf_counter()
            f(C.seed_of('time', arm, spp, i))
            sync()
            ts.append(1e3 * (time.perf_counter() - t))
        tim[f'{arm}_{spp}'] = {'median_ms': float(np.median(ts)), 'p95_ms': float(np.percentile(ts, 95))}
        log('time', arm, spp, tim[f'{arm}_{spp}'])
rec['timing_dev'] = tim

# 4. OptiX denoiser call on a 4-spp PT frame (guides from the same samples)
out = rt.pt_frame(cam, RES, 4, C.seed_of('dn', CASE))
a = pt111.to_np(out) / 4.0
img = np.stack([(a[0 + c] + a[3 + c] + a[6 + c]).reshape(RES, RES) for c in range(3)], -1)
alb = np.stack([a[9 + c].reshape(RES, RES) for c in range(3)], -1)
nrm = np.stack([a[12 + c].reshape(RES, RES) for c in range(3)], -1)
o, fwd, right, up, s = R.PT.cam_basis(cam)
to_world = mi.ScalarTransform4f().look_at(origin=o.tolist(), target=(o + fwd).tolist(), up=up.tolist())
den = mi.OptixDenoiser(input_size=[RES, RES], albedo=True, normals=True, temporal=False)
t = time.perf_counter()
y = den(mi.TensorXf(img.astype(np.float32)), albedo=mi.TensorXf(np.clip(alb, 0, 1).astype(np.float32)),
        normals=mi.TensorXf(nrm.astype(np.float32)), to_sensor=to_world.inverse())
dr.eval(y)
sync()
rec['denoiser_first_call_ms'] = 1e3 * (time.perf_counter() - t)
ts = []
for i in range(20):
    sync()
    t = time.perf_counter()
    y = den(mi.TensorXf(img.astype(np.float32)), albedo=mi.TensorXf(np.clip(alb, 0, 1).astype(np.float32)),
            normals=mi.TensorXf(nrm.astype(np.float32)), to_sensor=to_world.inverse())
    dr.eval(y)
    sync()
    ts.append(1e3 * (time.perf_counter() - t))
rec['denoiser_ms_incl_host_upload'] = float(np.median(ts))
yd = np.asarray(y)
rec['denoised_finite'] = bool(np.isfinite(yd).all())
rec['denoised_mean_over_noisy_mean'] = float(yd.mean() / img.mean())
log('denoiser', rec['denoiser_first_call_ms'], rec['denoiser_ms_incl_host_upload'], rec['denoised_mean_over_noisy_mean'])
np.savez_compressed(OUT / f'dev_{CASE}_images.npz', noisy4=img, den4=yd, alb=alb, nrm=nrm,
                    full256=full['e1'] + full['e2'] + full['h'])
rec['snapshot_end'] = C.gpu_snapshot()
C.write_json(OUT / f'dev_{CASE}.json', rec)
log('DEV DONE')
