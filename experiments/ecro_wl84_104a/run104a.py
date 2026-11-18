"""WL104A evidence run (local RTX 5080): A0 baseline gates -> field tables -> pilot (half-widths only) -> R rule -> replicates with the
A1 LOW-pairing gate checked per replicate. Resumable (existing files are kept). Stops (exit 2) on a gate failure.
  <venv-cu128 python> run104a.py <raw_root>
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

import h104a as H
import core104a as K
import stats104a as S

C = H.C
ROOT = Path(sys.argv[1]).resolve(); ROOT.mkdir(parents=True, exist_ok=True)
SMOKE = os.environ.get('WL104A_SMOKE') == '1'
SPP = 256 if SMOKE else 4096
FIELD_SPP = 1024 if SMOKE else 16384
PILOT_R = 4 if SMOKE else 8
RCHOICES = [8] if SMOKE else [32, 64, 128, 256]
P = K.P
ST = ROOT / 'status.json'
status = json.loads(ST.read_text(encoding='utf-8')) if ST.exists() else {}
status.update(commit=subprocess.check_output(['git', '-C', str(C.REPO), 'rev-parse', 'HEAD'], text=True).strip(), smoke=SMOKE,
              dirty=subprocess.check_output(['git', '-C', str(C.REPO), 'status', '--porcelain', '--', 'experiments/ecro_wl84_104a'], text=True).strip() != '')
status.pop('stopped', None)


def save():
    C.write_json(ST, status)


def stop(why, detail):
    status['stopped'] = why; status['stop_detail'] = detail; save()
    print('STOP', why, json.dumps(detail)[:800], flush=True)
    os._exit(2)


log = lambda *a: print(time.strftime('%H:%M:%S'), *a, flush=True)
if status['dirty'] and not SMOKE:
    stop('dirty_experiment_dir', {})
cfg = P['rois']['parameters']

# ---------------------------------------------------------------------------------------------------------------- A0
a0 = {}
ctxs, tables, masks = {}, {}, {}
for i, image in enumerate(C.IMAGES):
    tag = image.rsplit('_L', 1)[0]
    a0[image] = {'checkpoint_sha256_ok': C.sha(H.E104 / 'models' / tag / 'model.pt') == P['baseline']['checkpoints_sha256'][tag]}
    ctx = H.ctx_for(tag, 'er'); ctxs[image] = ctx
    B, _, _, _ = H.learned_table(ctx, image); tables[image] = B
    a0[image]['table_bitwise'] = bool(np.array_equal(B, np.load(H.E104 / 'eval' / image / 'table.npz')['B']))
    r4 = H.render(ctx, image, 4, 104500 + 1000 * i, 'ecro', B=B, b=4)
    w4 = np.load(H.E104 / 'eval' / image / 'ecro_spp4.npy')[0]
    a0[image]['ecro4_replay_max_abs'] = float(max(np.abs(r4[k].astype(np.float32) - w4[j]).max() for j, k in enumerate(('e1', 'e2', 'hl'))))
    masks[image] = H.masks(ctx, cfg)
    a0[image]['roi_counts'] = {k: int(v.sum()) for k, v in masks[image].items()}
ok = all(v['checkpoint_sha256_ok'] and v['table_bitwise'] and v['ecro4_replay_max_abs'] <= 1e-6 and v['roi_counts'] == P['rois']['development_counts']
         for v in a0.values())
status['A0'] = {'pass': ok, 'detail': a0}; save()
np.savez_compressed(ROOT / 'masks.npz', **{f'{im}|{k}': v for im in C.IMAGES for k, v in masks[im].items()})
if not ok:
    stop('A0_baseline', a0)
log('A0 pass')

# ---------------------------------------------------------------------------------------------------------------- field tables
(ROOT / 'field').mkdir(exist_ok=True)
fields = {}
for i, image in enumerate(C.IMAGES):
    for h in (0, 1):
        fn = ROOT / 'field' / f'{image}_{"AB"[h]}.npz'
        if not fn.exists():
            t0 = time.perf_counter()
            F, cnt = H.field_table(ctxs[image], image, FIELD_SPP, 1080000 + 10 * i + h)
            np.savez(fn, F=F, cnt=cnt, seed=1080000 + 10 * i + h, spp=FIELD_SPP)
            log('field', image, 'AB'[h], round(time.perf_counter() - t0, 1))
        fields[(image, h)] = np.load(fn)['F']


def a1(image, er, ec):
    lo_c = er['e1'] + er['e2']; lo_e = ec['e1'] + ec['e2']
    rel = abs(lo_e.sum() - lo_c.sum()) / lo_c.sum(); mx = float(np.abs(lo_e - lo_c).max())
    if not (rel <= 1e-6 and mx <= 1e-4):
        stop('A1_low_pairing', {'image': image, 'rel_sum': rel, 'max_abs': mx})
    return rel, mx


# ---------------------------------------------------------------------------------------------------------------- pilot
pil = {}
for i, image in enumerate(C.IMAGES):
    d = ROOT / 'pilot' / image; d.mkdir(parents=True, exist_ok=True)
    for r in range(PILOT_R):
        fn = d / f'r{r:03d}.npz'
        if fn.exists():
            continue
        sd = 1070000 + 10000 * i + r
        ec = H.render(ctxs[image], image, SPP, sd, 'ecro', B=tables[image]); er = H.render(ctxs[image], image, SPP, sd, 'cam')
        a1(image, er, ec)
        np.savez(fn, ecro_hl=ec['hl'].astype(np.float32), er_e1=er['e1'].astype(np.float32), er_e2=er['e2'].astype(np.float32), er_h=er['h'].astype(np.float32))
    z = K.load(d, ['ecro_hl', 'er_e1', 'er_e2', 'er_h'])
    tot = z['er_e1'] + z['er_e2'] + z['er_h']
    pil[image] = {}
    for roi in K.PRIMARY:
        sig = K.significance(K.compare(z['ecro_hl'], z['er_h'], tot, z['er_h'], masks[image][roi]))
        pil[image][roi] = {'energy_halfwidth': sig['energy_halfwidth'], 'rms_halfwidth': sig['rms_halfwidth']}    # half-widths only
need = lambda R: all(v['energy_halfwidth'] * np.sqrt(PILOT_R / R) <= K.TAU_E / 4 and v['rms_halfwidth'] * np.sqrt(PILOT_R / R) <= K.TAU_S / 4
                     for im in pil.values() for v in im.values())
R = next((r for r in RCHOICES if need(r)), RCHOICES[-1])
status['pilot'] = {'R_pilot': PILOT_R, 'halfwidths': pil, 'R_selected': R, 'rule': P['sampling']['R_rule']}; save()
log('pilot -> R =', R)

# ---------------------------------------------------------------------------------------------------------------- replicates
lows = []
for i, image in enumerate(C.IMAGES):
    tag = image.rsplit('_L', 1)[0]
    ctxl = H.ctx_for(tag, 'lay')
    d = ROOT / 'rep' / image; d.mkdir(parents=True, exist_ok=True)
    H.RT.register(tag)
    t0 = time.perf_counter()
    for r in range(R):
        fn = d / f'r{r:03d}.npz'
        if fn.exists():
            continue
        sd = 1050000 + 10000 * i + r
        ec = H.render(ctxs[image], image, SPP, sd, 'ecro', B=tables[image])
        oa = H.render(ctxs[image], image, SPP, sd, 'ecro', B=fields[(image, 0)])
        ob = H.render(ctxs[image], image, SPP, sd, 'ecro', B=fields[(image, 1)])
        er = H.render(ctxs[image], image, SPP, sd, 'cam')
        la = H.render(ctxl, image, SPP, 1060000 + 10000 * i + r, 'cam', pass_seed=True)
        lows.append(a1(image, er, ec))
        f32 = lambda a: a.astype(np.float32)
        np.savez(fn.with_suffix('.tmp.npz'), ecro_e1=f32(ec['e1']), ecro_e2=f32(ec['e2']), ecro_hl=f32(ec['hl']), orcA_hl=f32(oa['hl']), orcB_hl=f32(ob['hl']),
                 er_e1=f32(er['e1']), er_e2=f32(er['e2']), er_h=f32(er['h']), lay_e1=f32(la['e1']), lay_e2=f32(la['e2']), lay_h=f32(la['h']))
        os.replace(fn.with_suffix('.tmp.npz'), fn)
        if r % 16 == 0:
            log('rep', image, r, round(time.perf_counter() - t0, 1))
status['A1'] = {'pass': True, 'max_rel_sum': max([x[0] for x in lows], default=None), 'max_abs': max([x[1] for x in lows], default=None)}
status['complete'] = True; status['finished'] = time.strftime('%Y-%m-%d %H:%M:%S'); save()
log('RUN COMPLETE')
os._exit(0)
