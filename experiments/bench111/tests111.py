"""WL111 focused tests (run on Main before the evidence chain). python tests111.py OUT_JSON

CPU: metric arithmetic (debiasing, relative error, LDR), seed determinism / disjointness.
GPU: denoiser device path == host path; S0 V0 masks reproduce the WL101C / WL101A counts; ECRO frame composition;
     edit correspondence (selective relation == full rebuild at T1; T0 -> T1 -> T0 restores state and table bit-exactly);
     light change leaves the relation state unchanged.
"""
import json
import sys
from pathlib import Path

import numpy as np

import common111 as C

OUT = Path(sys.argv[1])
res = {}


def check(name, ok, detail=''):
    res[name] = {'pass': bool(ok), 'detail': detail}
    print(('PASS ' if ok else 'FAIL ') + name, detail, flush=True)


# ---- CPU: metrics
rng = np.random.default_rng(0)
ref = rng.uniform(0.5, 1.5, (64, 64, 3))
x = ref + 0.1
m = C.metrics(x, ref)
check('metric_rel_rmse', abs(m['rel_rmse'] - 0.1 / ref.mean()) < 1e-12, m['rel_rmse'])
check('metric_bias', abs(m['rel_bias'] - 0.1 / ref.mean()) < 1e-12, m['rel_bias'])
nv = np.full_like(ref, 0.004)
m2 = C.metrics(x, ref, ref_noise_var=nv)
check('metric_debias', abs(m2['rmse_db'] - np.sqrt(0.01 - 0.004)) < 1e-12, m2['rmse_db'])
check('metric_debias_floor', C.metrics(ref, ref, ref_noise_var=nv)['rmse_db'] == 0.0)
check('metric_ldr', abs(C.metrics(ref, ref)['ldr_rmse']) < 1e-15)
mask = np.zeros((64, 64), bool)
mask[:8] = True
check('metric_mask_count', C.metrics(x, ref, mask)['n_px'] == 8 * 64)
# ---- CPU: seeds
s1 = C.seed_of('q111', 'S0', 'V0', 'PT', 4, 0)
check('seed_deterministic', s1 == C.seed_of('q111', 'S0', 'V0', 'PT', 4, 0))
seeds = set()
for kind in ('q111', 'ref111', 't111', 'stale111'):
    for a in ('ECRO', 'PT', 'FULL', 'D6'):
        for spp in (1, 2, 4, 8, 16, 32, 64, 128, 256, 1024, 32768):
            for rep in range(16):
                seeds.add(C.seed_of(kind, 'S0', 'V0', a, spp, rep))
check('seed_distinct', len(seeds) == 4 * 4 * 11 * 16, len(seeds))
check('seed_range', max(seeds) < (1 << 40))

# ---- GPU
import run111 as RN  # noqa: E402
RN.ROOT = OUT.parent
RN.SMOKE = False
R, mi, dr, torch, pt111 = RN.R, RN.mi, RN.dr, RN.torch, RN.pt111
W = RN.World()
rt, Bd, case = W.get('S0')
cam = RN.camera('V0', 'sss')
o = rt.pt_frame(cam, RN.RES, 4, 12345)
den = RN.Denoiser(cam)
yd = np.asarray(den.device(o, 4))
a = pt111.to_np(o) / 4.0
img = np.stack([(a[0 + c] + a[3 + c] + a[6 + c]).reshape(512, 512) for c in range(3)], -1).astype(np.float32)
alb = np.clip(np.stack([a[9 + c].reshape(512, 512) for c in range(3)], -1), 0, 1).astype(np.float32)
nrm = np.stack([a[12 + c].reshape(512, 512) for c in range(3)], -1).astype(np.float32)
yh = np.asarray(den.den(mi.TensorXf(img), albedo=mi.TensorXf(alb), normals=mi.TensorXf(nrm), to_sensor=den.to_sensor))
check('denoiser_device_equals_host', float(np.abs(yd - yh).max()) <= 1e-4 * max(1.0, float(np.abs(yh).max())),
      float(np.abs(yd - yh).max()))
check('denoiser_energy', abs(yd.mean() / img.mean() - 1) < 0.05, float(yd.mean() / img.mean()))
# masks vs frozen WL101A / WL101C counts at V0
mk = RN.masks_for(rt, 'sss', cam, 'V0')
cnt = {k: int(mk[k].sum()) for k in ('near', 'flank', 'sphere')}
check('mask_near_wl101c', cnt['near'] == 1318, cnt)
check('mask_flank_wl101c', cnt['flank'] == 6042, cnt)
check('mask_sphere_wl101a', cnt['sphere'] == 49660, cnt)
# ECRO frame: deterministic for a seed; LOW + HL composition finite
f1 = pt111.to_np(rt.frame(Bd, cam, 512, 4, 777))
f2 = pt111.to_np(rt.frame(Bd, cam, 512, 4, 777))
check('ecro_frame_deterministic', np.array_equal(f1, f2))
check('ecro_frame_finite', bool(np.isfinite(f1).all()))
# edit correspondence
h0, d0 = rt.hit.clone(), rt.dbin.clone()
B0 = np.asarray(Bd).copy()
_, gB, lB = RN.state_def('S1')
_, gA, lA = RN.state_def('S0')
moved, old = rt.set_geom(gB)
nd = rt.update_state(old, moved)
hs, ds = rt.hit.clone(), rt.dbin.clone()
rt.rebuild_state()
check('selective_equals_full_T1', bool(torch.equal(hs, rt.hit) and torch.equal(ds, rt.dbin)), {'dirty_rays': nd, 'moved': moved})
moved, old = rt.set_geom(gA)
rt.update_state(old, moved)
rt.set_light(lA)
B, _ = rt.respond(rt.source())
check('roundtrip_state_T0', bool(torch.equal(h0, rt.hit) and torch.equal(d0, rt.dbin)))
check('roundtrip_table_T0', bool(np.array_equal(B.view(-1).cpu().numpy(), B0)), float(np.abs(B.view(-1).cpu().numpy() - B0).max()))
_, gL, lL = RN.state_def('SL')
moved, old = rt.set_geom(gL)
rt.set_light(lL)
check('light_change_no_geometry_move', moved == [] and bool(torch.equal(h0, rt.hit)))
rt.set_light(lA)
ok, info = RN.exclusive_ok()
res['exclusive_probe'] = {'ok': ok, 'info': info}
res['all_pass'] = all(v['pass'] for k, v in res.items() if isinstance(v, dict) and 'pass' in v)
res['versions'] = C.versions(mi, dr, torch)
C.write_json(OUT, res)
print('ALL_PASS', res['all_pass'], flush=True)
