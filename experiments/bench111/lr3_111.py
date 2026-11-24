"""WL111 W4 Living Room 3 (static V0, 640 x 360, GI output semantics; WL102 / TransGI contract).

Runs in the TransGI reproduction venv with Mitsuba 3.9.1 / Dr.Jit 1.5.0 first on PYTHONPATH (lr3.cmd), TRANSGI_DATA set to
the isolated copy C:\\Projects\\bench111\\lr3\\tgdata (scene living-room-3, Ref-65 / Ref-7 of LabServer69, sha-checked).

python lr3_111.py OUT pt        PT_LR3 / PTDN_LR3 quality ladder + timing (Mitsuba stock 'path', GI = full(65) - full(2), CRN)
python lr3_111.py OUT score     score TransGI evaluator images (and the historical WL102 ECRO images if present) with WL111 metrics
"""
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
P = json.loads((HERE / 'protocol' / 'bench111.json').read_text(encoding='utf-8'))
L = json.loads((HERE / 'protocol' / 'bench111_lr3.json').read_text(encoding='utf-8'))
import common111 as C  # noqa: E402

OUT = Path(sys.argv[1])
MODE = sys.argv[2]
OUT.mkdir(parents=True, exist_ok=True)
DATA = Path(os.environ['TRANSGI_DATA'])
REF = DATA / 'runs' / 'living_room_3' / 'reference'


def log(*a):
    s = time.strftime('%Y-%m-%d %H:%M:%S ') + ' '.join(str(x) for x in a)
    print(s, flush=True)
    with open(OUT / 'lr3.log', 'a', encoding='utf-8') as f:
        f.write(s + '\n')


def load_ref():
    z = np.load(REF / 'static_V0.npz')
    z7 = np.load(REF / 'static_V0_d7.npz')
    gi = z['gi'].astype(np.float64)
    var = 0.25 * (z['gi_A'].astype(np.float64) - z['gi_B']) ** 2
    return {'gi': gi, 'var': var, 'gi7': z7['gi'].astype(np.float64), 'var7': 0.25 * (z7['gi_A'].astype(np.float64) - z7['gi_B']) ** 2,
            'halves': (z['gi_A'].astype(np.float64), z['gi_B'].astype(np.float64)), 'di': z['di'].astype(np.float64)}


def score(img, ref):
    m = C.metrics(img, ref['gi'], None, ref['var'])
    m7 = C.metrics(img, ref['gi7'], None, ref['var7'])
    return {'vs_ref65': m, 'vs_ref7': {k: m7[k] for k in ('rel_rmse_db', 'ldr_rmse', 'rel_bias')}}


def exclusive_ok():
    other = subprocess.run(['tasklist', '/fi', 'imagename eq python.exe', '/fo', 'csv', '/nh'], capture_output=True, text=True).stdout
    pids = [int(r.split('","')[1]) for r in other.strip().splitlines() if r.startswith('"python')]
    pids = [p for p in pids if p not in {os.getpid(), os.getppid()}]
    util = []
    for _ in range(6):
        q = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'], capture_output=True, text=True).stdout
        util.append(float(q.strip().splitlines()[0]))
        time.sleep(0.5)
    return (not pids) and float(np.median(util)) <= P['timing']['idle_util_max_pct'], {'other_python_pids': pids, 'idle_util': util}


def summ(v):
    v = np.asarray(v, np.float64)
    return {'n': int(v.size), 'median': float(np.median(v)), 'mean': float(v.mean()), 'p95': float(np.percentile(v, 95)),
            'p99': float(np.percentile(v, 99))}


def mode_pt():
    import mitsuba as mi
    mi.set_variant('cuda_ad_rgb')
    import drjit as dr
    import torch
    sys.path.insert(0, os.environ['TRANSGI_REPO'])
    from transgi import config
    from transgi.session import Session
    cfg = config.load(os.path.join(os.environ['TRANSGI_REPO'], 'configs', 'living_room_3.yaml'))
    res = L['res']
    sess = Session(cfg, res=res)
    scene, sensor = sess.sd.mi_scene, sess.sensor
    man = {'versions': C.versions(mi, dr, torch), 'scene_manifest': {k: sess.sd.manifest[k] for k in ('modified_sha256', 'removed_dielectric_shapes')},
           'ref_sha256': C.sha(REF / 'static_V0.npz'), 'ref7_sha256': C.sha(REF / 'static_V0_d7.npz'), 'snapshot': C.gpu_snapshot()}
    import hashlib
    xml = (DATA / 'scenes' / 'living-room-3' / 'scene_transgi.xml').read_bytes().replace(b'\r\n', b'\n')
    man['scene_xml_lf_sha256'] = hashlib.sha256(xml).hexdigest()
    assert man['scene_xml_lf_sha256'] == L['scene_modified_sha256'], man   # Windows writes CRLF; content must equal LabServer69's
    assert man['ref_sha256'] == L['ref65_sha256'], man['ref_sha256']
    ref = load_ref()
    i65 = mi.load_dict({'type': 'path', 'max_depth': 65})
    i2 = mi.load_dict({'type': 'path', 'max_depth': 2})
    iaov = mi.load_dict({'type': 'aov', 'aovs': 'albedo:albedo,nn:sh_normal'})
    W, H = res
    den = mi.OptixDenoiser(input_size=[W, H], albedo=True, normals=True, temporal=False)
    to_sensor = sensor.world_transform().inverse()

    def sync():
        dr.sync_thread()
        torch.cuda.synchronize()

    def gi_of(spp, seed):
        full = mi.render(scene, sensor=sensor, integrator=i65, spp=spp, seed=seed)
        di = mi.render(scene, sensor=sensor, integrator=i2, spp=spp, seed=seed)
        return full, di

    def denoise(gi_t, spp, seed):
        a = mi.render(scene, sensor=sensor, integrator=iaov, spp=L['ptdn_guide_spp'], seed=seed + 7)
        alb = a[:, :, 0:3]
        nrm = a[:, :, 3:6]
        y = den(gi_t, albedo=alb, normals=nrm, to_sensor=to_sensor)
        dr.eval(y)
        return y

    rows = []
    imgs = {}
    for spp in L['pt_spp']:
        reps = next(n for lim, n in L['reps_by_spp'] if spp <= lim)
        for rep in range(reps):
            seed = C.seed_of('lr3q111', spp, rep) % (1 << 31)
            full, di = gi_of(spp, seed)
            gi_t = full - di
            dr.eval(gi_t)
            gi = np.asarray(gi_t, np.float64)
            row = {'arm': 'PT_LR3', 'spp': spp, 'rep': rep, **score(gi, ref)}
            rows.append(row)
            if rep == 0:
                imgs[f'PT_{spp}'] = gi.astype(np.float32)
                row['ssim_ldr_vs_ref65'] = C.ssim_ldr(gi, ref['gi'])
            if spp in L['ptdn_spp']:
                y = np.asarray(denoise(mi.TensorXf(gi_t), spp, seed), np.float64)
                r2 = {'arm': 'PTDN_LR3', 'spp': spp, 'rep': rep, **score(y, ref)}
                if rep == 0:
                    imgs[f'PTDN_{spp}'] = y.astype(np.float32)
                    r2['ssim_ldr_vs_ref65'] = C.ssim_ldr(y, ref['gi'])
                rows.append(r2)
        log('quality', spp, reps)
    ref_self = {'half_vs_half_gi': C.metrics(ref['halves'][0], ref['halves'][1]), 'ref7_vs_ref65': score(ref['gi7'], ref)}
    np.savez_compressed(OUT / 'lr3_pt_images.npz', ref65_gi=ref['gi'].astype(np.float32), ref7_gi=ref['gi7'].astype(np.float32), **imgs)
    C.write_json(OUT / 'lr3_pt_quality.json', {'rows': rows, 'reference_self': ref_self, 'manifest': man})
    # ---- timing: exclusive windows, warm-up, randomized order, full-depth render per frame (+ guide pass + denoise for PTDN)
    t = P['timing']
    ops = [('PT_LR3', s) for s in L['timing_pt_spp']] + [('PTDN_LR3', s) for s in L['timing_ptdn_spp']]

    def call(arm, spp, i):
        full = mi.render(scene, sensor=sensor, integrator=i65, spp=spp, seed=100000 + i)
        if arm == 'PTDN_LR3':
            denoise(full, spp, 100000 + i)
        else:
            dr.eval(full)

    for attempt in range(t['max_attempts']):
        while True:
            ok, pre = exclusive_ok()
            if ok:
                break
            log('timing wait', pre)
            time.sleep(30)
        rec = {'first_call_ms': {}, 'frame_ms': {f'{a}_{s}': [] for a, s in ops}}
        for a, s in ops:
            sync(); t0 = time.perf_counter(); call(a, s, -1); sync()
            rec['first_call_ms'][f'{a}_{s}'] = 1e3 * (time.perf_counter() - t0)
            for w in range(t['warmup']):
                call(a, s, -2 - w)
        rng = random.Random(C.seed_of('lr3order111'))
        for rnd in range(t['rounds']):
            order = list(ops)
            rng.shuffle(order)
            for a, s in order:
                for i in range(t['frames_per_round'] if s <= 64 else max(1, t['frames_per_round'] // 4)):
                    sync(); t0 = time.perf_counter(); call(a, s, rnd * 1000 + i); sync()
                    rec['frame_ms'][f'{a}_{s}'].append(1e3 * (time.perf_counter() - t0))
        ok2, post = exclusive_ok()
        if ok2:
            break
        log('timing contaminated, repeat', post)
    rec['summary'] = {k: summ(v) for k, v in rec['frame_ms'].items()}
    rec['exclusivity'] = {'pre': pre, 'post': post, 'attempt': attempt, 'contaminated': not ok2}
    rec['snapshot_end'] = C.gpu_snapshot()
    C.write_json(OUT / 'lr3_pt_timing.json', rec)
    log('timing', {k: round(v['median'], 2) for k, v in rec['summary'].items()})


def mode_score():
    """Score images written by the TransGI evaluator (tg_eval/<tag>/static_V0.npz) with the same metric code."""
    ref = load_ref()
    out = {}
    for tag_dir in sorted((DATA / 'runs' / 'living_room_3' / 'eval').glob('wl111_*')):
        f = tag_dir / 'static_V0.npz'
        if not f.exists():
            continue
        z = np.load(f)
        key = 'gi' if 'gi' in z.files else [k for k in z.files if 'gi' in k][0]
        img = z[key].astype(np.float64)
        out[tag_dir.name] = {'file_sha256': C.sha(f), 'key': key, 'keys': z.files, **score(img, ref),
                             'ssim_ldr_vs_ref65': C.ssim_ldr(img, ref['gi'])}
        log('scored', tag_dir.name, out[tag_dir.name]['vs_ref65']['ldr_rmse'])
    C.write_json(OUT / 'lr3_transgi_scored.json', out)


if __name__ == '__main__':
    {'pt': mode_pt, 'score': mode_score}[MODE]()
