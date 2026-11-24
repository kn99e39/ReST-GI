"""WL111 evidence harness for the ECRO fixture workloads (W1 SSS, W2 near-mirror, W3 SSS rigid box edit, W5 SSS light change).

python run111.py ROOT STAGE [--smoke]
  STAGE: gates | refs | quality | timing | dynamic | all

Arms (protocol/bench111.json):
  ECRO   frozen canonical ReST-GI frame (WL100 checkpoint, WL101 runtime, LOW E1+E2 physical + learned HIGH table, D = 6)
  PT     full-depth physical path tracer with identical scene semantics (pt111; kmax 256, Russian roulette from event 4)
  PTDN   PT + OptiX AI denoiser (HDR, albedo + normal guides from the same samples, no temporal history)
References: REF_FULL (pt111 full depth, 2 independent halves) primary; REF_D6 (kmax 6, no RR) depth-matched diagnostic.
Every quality number is computed here by the frozen metric code at capture time; images of replicate 0 are kept.
"""
import argparse
import json
import math
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PROTOCOL = HERE / 'protocol' / 'bench111.json'
P = json.loads(PROTOCOL.read_text(encoding='utf-8'))
import common111 as C  # noqa: E402
import pt111  # noqa: E402

R, mi, dr, torch = pt111.R, pt111.mi, pt111.dr, pt111.torch
RES = 512
SMOKE = False


def log(*a):
    s = time.strftime('%Y-%m-%d %H:%M:%S ') + ' '.join(str(x) for x in a)
    print(s, flush=True)
    with open(ROOT / 'run.log', 'a', encoding='utf-8') as f:
        f.write(s + '\n')


def sync():
    dr.sync_thread()
    torch.cuda.synchronize()


# ------------------------------------------------------------------------------------------------ states and cameras
def rows(case):
    return json.loads((R.ROOT100 / 'configs.json').read_text())[case]['test']


def camera(view, case='sss'):
    """V0 = the WL100 dataset camera of the case (configs.json test row 0); held-out views are defined for 'sss' only."""
    v0 = dict(P['cameras']['V0'][case])
    if view == 'V0':
        return dict(v0, res=RES)
    assert case == 'sss', (view, case)
    h = P['cameras']['held_out'][view]
    T = np.array(P['cameras']['held_out_target'])
    az, el, d = math.radians(h['azimuth_deg']), math.radians(h['elevation_deg']), h['horizontal_m']
    o = T + np.array([d * math.sin(az), -d * math.cos(az), d * math.tan(el)])
    return {'origin': o.tolist(), 'target': T.tolist(), 'fov_deg': v0['fov_deg'], 'res': RES}


def state_def(sid):
    s = P['states'][sid]
    rr = rows(s['case'])
    geom = rr[s['geom_index']]['geom']
    light = rr[s['light_from_index']]['lights'][s['light_index']]
    return s['case'], geom, light


class World:
    """ONE case runtime at a time: rt101 keeps the closure table in a module-level dict (_CL) that belongs to the most
    recently constructed runtime, so a second live runtime would make the first read the wrong material table (found in
    the code-path smoke). Switching case drops the old runtime and builds a fresh one. set_state() applies the frozen
    WL101 selective ECRO lifecycle for geometry / light changes within a case."""

    def __init__(self):
        self.case = None
        self.rt_ = None
        self.cur = {}
        self.Bd = {}

    @property
    def rt(self):
        return {self.case: self.rt_} if self.rt_ is not None else {}

    def get(self, sid):
        import gc
        case, geom, light = state_def(sid)
        if case != self.case:
            self.rt_ = None
            self.Bd = {}
            gc.collect()
            dr.flush_malloc_cache()
            self.rt_ = pt111.PT111(geom, case, R.load_model(case), light)
            self.case = case
            self.cur = {case: None}
        rt = self.rt_
        if self.cur.get(case) != sid:
            moved, old = rt.set_geom(geom)
            rt.update_state(old, moved)
            rt.set_light(light)
            s = rt.source()
            B, _ = rt.respond(s)
            self.Bd[case] = R.t2d(B.view(-1))
            self.cur[case] = sid
        return rt, self.Bd[case], case


# ------------------------------------------------------------------------------------------------ images
def comp15(arr, spp):
    a = np.asarray(arr, np.float64) / spp
    g = lambda i: np.stack([a[i + c].reshape(RES, RES) for c in range(3)], -1)
    return {'e1': g(0), 'e2': g(3), 'h': g(6), 'alb': g(9), 'nrm': g(12)}


def ecro_image(out, spp):
    a = pt111.to_np(out).astype(np.float64) / spp
    g = lambda i: np.stack([a[i + c].reshape(RES, RES) for c in range(3)], -1)
    return {'e1': g(0), 'e2': g(3), 'hl': g(6)}


class Denoiser:
    def __init__(self, cam):
        o, fwd, right, up, s = R.PT.cam_basis(cam)
        self.to_sensor = mi.ScalarTransform4f().look_at(origin=o.tolist(), target=(o + fwd).tolist(), up=up.tolist()).inverse()
        self.den = mi.OptixDenoiser(input_size=[RES, RES], albedo=True, normals=True, temporal=False)

    def device(self, out, spp):
        """Denoise directly from the PT device arrays (no host round trip). -> mi.TensorXf [RES, RES, 3]."""
        t = torch.stack([o.torch() for o in out]).view(15, RES, RES) * (1.0 / spp)
        img = (t[0:3] + t[3:6] + t[6:9]).permute(1, 2, 0).contiguous()
        alb = t[9:12].clamp(0.0, 1.0).permute(1, 2, 0).contiguous()
        nrm = t[12:15].permute(1, 2, 0).contiguous()
        y = self.den(mi.TensorXf(img), albedo=mi.TensorXf(alb), normals=mi.TensorXf(nrm), to_sensor=self.to_sensor)
        dr.eval(y)
        return y


# ------------------------------------------------------------------------------------------------ masks
def masks_for(rt, case, cam, view):
    g = rt.gbuffer(dict(cam), RES)
    valid, sid, pos = np.asarray(g['ok']).reshape(RES, RES), np.asarray(g['sid']).reshape(RES, RES), np.asarray(g['p']).reshape(RES, RES, 3)
    m = {'full': np.ones((RES, RES), bool), 'valid': valid}
    # 8 x 8 subpixel first-hit ids (geometry only) for coverage / silhouette / pure-material masks
    sub = 8
    o, fwd, right, up, s = R.PT.cam_basis(cam)
    F, U, I32 = mi.Float, mi.UInt32, mi.Int32
    n = RES * RES * sub * sub
    lane = dr.arange(U, n)
    pix = lane // (sub * sub)
    k = lane % (sub * sub)
    ii, jj = pix // RES, pix % RES
    x = (F(jj) + (F(k % sub) + 0.5) / sub) * (2.0 / RES) - 1.0
    y = 1.0 - (F(ii) + (F(k // sub) + 0.5) / sub) * (2.0 / RES)
    d = dr.normalize(mi.Vector3f(*[float(fwd[q]) + s * x * float(right[q]) + s * y * float(up[q]) for q in range(3)]))
    si = rt.scene.ray_intersect(mi.Ray3f(mi.Point3f(*[F(float(o[q])) for q in range(3)]), d))
    ids = dr.select(si.is_valid(), I32(rt.shape_id(si)), I32(-1))
    dr.eval(ids)
    ids = np.asarray(ids).reshape(RES, RES, sub * sub)
    if case == 'sss':
        frac = (ids == 6).mean(-1)
        sphere = valid & (sid == 6)
        pad = np.pad(sphere, 4, constant_values=False)
        interior = np.logical_and.reduce([pad[yy:yy + RES, xx:xx + RES] for yy in range(9) for xx in range(9)])
        m['sphere'] = sphere
        m['sphere_interior'] = interior & (frac >= 1.0)
        m['silhouette'] = (frac > 0.0) & (frac < 1.0)
        floor = valid & (sid == 0)
        sup = np.array(P['sss_gate']['support_point'])
        r = np.linalg.norm(pos[..., :2] - sup[:2], axis=-1)
        front = floor & (pos[..., 1] <= sup[1])
        g = P['sss_gate']
        if view == 'V0':
            m['near'] = front & (r < g['near_r_m'])
            m['flank'] = front & (r >= g['flank_r_m'][0]) & (r < g['flank_r_m'][1])
        else:
            m['near'] = floor & (r < g['near_r_m'])
            m['flank'] = floor & (r >= g['flank_r_m'][0]) & (r < g['flank_r_m'][1])
        edges = g['radial_edges_m']
        for a, b in zip(edges[:-1], edges[1:]):
            m[f'ring_{a:.2f}_{b:.2f}'] = (front if view == 'V0' else floor) & (r >= a) & (r < b)
        m['contact_disk'] = floor & (r < g['contact_excluded_r_m'])
        m['off_target'] = valid & ~m['contact_disk'] & ~m['silhouette'] & ~sphere
    else:
        frac = (ids == 5).mean(-1)
        m['panel'] = valid & (sid == 5)
        m['panel_interior'] = m['panel'] & (frac >= 1.0)
        m['non_panel'] = valid & ~m['panel']
    m['mixed_boundary'] = ~(ids == ids[..., :1]).all(-1)
    return m


# ------------------------------------------------------------------------------------------------ evaluation of one frame
def evaluate(img, ref, masks, case):
    """img, ref['mean'] [H, W, 3]. Primary: rel_rmse_db on 'full'. ROIs per protocol; SSS: signed near / flank / excess."""
    out = {}
    for name in P['metrics']['rois'][case]:
        if name in masks and masks[name].sum() > 0:
            out[name] = C.metrics(img, ref['mean'], masks[name], ref['var'])
    if case == 'sss':
        L = C.luminance(img)
        Lr = C.luminance(ref['mean'])
        for nm in [k for k in masks if k.startswith('ring_')] + ['near', 'flank']:
            mk = masks[nm]
            if mk.sum() > 0:
                out.setdefault('signed', {})[nm] = {'x': float(L[mk].mean()), 'ref': float(Lr[mk].mean()),
                                                    'diff': float(L[mk].mean() - Lr[mk].mean()), 'n': int(mk.sum())}
        sg = out.get('signed', {})
        if 'near' in sg and 'flank' in sg:
            out['near_minus_flank_excess'] = (sg['near']['x'] - sg['flank']['x']) - (sg['near']['ref'] - sg['flank']['ref'])
            out['near_signed'] = sg['near']['diff']
            out['near_rel_signed'] = sg['near']['diff'] / sg['near']['ref']
    return out


def load_ref(sid, view, kind):
    z = [np.load(ROOT / 'refs' / f'{sid}_{view}_{kind}_h{h}.npz') for h in (0, 1)]
    tot = [zz['e1'] + zz['e2'] + zz['h'] for zz in z]
    return {'mean': 0.5 * (tot[0] + tot[1]), 'var': 0.25 * (tot[0] - tot[1]) ** 2,
            'halves': tot, 'comp': {k: 0.5 * (z[0][k] + z[1][k]) for k in ('e1', 'e2', 'h')}}


# ------------------------------------------------------------------------------------------------ stages
def stage_refs(world):
    rp = P['references']
    for sid in P['run_states']:
        case = P['states'][sid]['case']
        for view in P['states'][sid]['views']:
            rt, Bd, case = world.get(sid)
            cam = camera(view, case)
            mfile = ROOT / 'refs' / f'{sid}_{view}_masks.npz'
            if not mfile.exists():
                m = masks_for(rt, case, cam, view)
                np.savez_compressed(mfile, **m)
                C.write_json(mfile.with_suffix('.json'), {k: int(v.sum()) for k, v in m.items()})
            for kind, spp, kmax, rr in (('FULL', rp['full_spp_per_half'], rp['full_kmax'], rp['full_rr']),
                                        ('D6', rp['d6_spp_per_half'], 6, 0)):
                if SMOKE:
                    spp = 256
                for h in (0, 1):
                    f = ROOT / 'refs' / f'{sid}_{view}_{kind}_h{h}.npz'
                    if f.exists():
                        continue
                    seed = C.seed_of('ref111', sid, view, kind, h)
                    t = time.perf_counter()
                    acc = pt111.pt_hq(rt, cam, RES, spp, seed, kmax=kmax, rr=rr)
                    sync()
                    wall = time.perf_counter() - t
                    c = comp15(acc, spp)
                    tmp = f.with_name(f.stem + '.tmp.npz')
                    np.savez_compressed(tmp, **{k: c[k].astype(np.float32) for k in ('e1', 'e2', 'h')})
                    os.replace(tmp, f)
                    C.write_json(f.with_suffix('.json'), {'state': sid, 'view': view, 'kind': kind, 'half': h, 'seed': seed,
                                                          'spp': spp, 'kmax': kmax, 'rr': rr, 'wall_s': wall, 'camera': cam,
                                                          'sha256': C.sha(f), 'snapshot': C.gpu_snapshot()})
                    log('REF', sid, view, kind, h, f'{wall:.1f}s')


def stage_quality(world):
    q = P['quality']
    for sid in P['run_states']:
        for view in P['states'][sid]['views']:
            out = ROOT / 'quality' / f'{sid}_{view}.json'
            if out.exists():
                continue
            rt, Bd, case = world.get(sid)
            cam = camera(view, case)
            ref = load_ref(sid, view, 'FULL')
            ref6 = load_ref(sid, view, 'D6')
            with np.load(ROOT / 'refs' / f'{sid}_{view}_masks.npz') as z:
                masks = {k: z[k] for k in z.files}
            den = Denoiser(cam)
            rec = {'state': sid, 'view': view, 'camera': cam, 'rows': [], 'reference_self': {}}
            # reference self-consistency: half vs half, and D6 vs FULL (truncation, diagnostic)
            rec['reference_self']['half_vs_half'] = C.metrics(ref['halves'][0], ref['halves'][1], masks['full'])
            rec['reference_self']['D6_vs_FULL'] = evaluate(ref6['mean'], ref, masks, case)
            imgs = {}

            def add(arm, spp, rep, img, extra=None):
                row = {'arm': arm, 'spp': spp, 'rep': rep, 'vs_full': evaluate(img, ref, masks, case),
                       'vs_d6': {'full': C.metrics(img, ref6['mean'], masks['full'], ref6['var'])}}
                if rep == 0:
                    row['ssim_ldr_vs_full'] = C.ssim_ldr(img, ref['mean'])
                    imgs[f'{arm}_{spp}'] = img.astype(np.float32)
                if extra:
                    row.update(extra)
                rec['rows'].append(row)

            for spp in (q['ecro_spp'] if not SMOKE else [1, 4]):
                for rep in range(reps_for(spp)):
                    seed = C.seed_of('q111', sid, view, 'ECRO', spp, rep)
                    o = rt.frame(Bd, cam, RES, spp, seed)
                    e = ecro_image(o, spp)
                    add('ECRO', spp, rep, e['e1'] + e['e2'] + e['hl'],
                        {'low_vs_full_low': C.metrics(e['e1'] + e['e2'], ref['comp']['e1'] + ref['comp']['e2'], masks['full'])} if rep == 0 else None)
            # converged ECRO (bias of the canonical estimator, diagnostic)
            spp_c = q['ecro_converged_spp'] if not SMOKE else 64
            acc = R.render_hq(rt, Bd, cam, RES, spp_c, C.seed_of('q111', sid, view, 'ECROconv'))
            a = acc / spp_c
            gg = lambda i: np.stack([a[i + c].reshape(RES, RES) for c in range(3)], -1)
            ec = gg(0) + gg(3) + gg(6)
            add('ECRO_CONV', spp_c, 0, ec, {'hl_vs_full_higher': C.metrics(gg(6), ref['comp']['h'] + 0.0, masks['full'])})
            for spp in (q['pt_spp'] if not SMOKE else [1, 4]):
                for rep in range(reps_for(spp)):
                    seed = C.seed_of('q111', sid, view, 'PT', spp, rep)
                    o = rt.pt_frame(cam, RES, spp, seed, kmax=q['pt_kmax'], rr=q['pt_rr'])
                    c = comp15(pt111.to_np(o), spp)
                    add('PT', spp, rep, c['e1'] + c['e2'] + c['h'])
                    if spp in q['ptdn_spp']:
                        y = np.asarray(den.device(o, spp), np.float64)
                        add('PTDN', spp, rep, y)
            np.savez_compressed(ROOT / 'quality' / f'{sid}_{view}_images.npz', ref=ref['mean'].astype(np.float32),
                                ref_d6=ref6['mean'].astype(np.float32), **imgs)
            rec['snapshot'] = C.gpu_snapshot()
            C.write_json(out, rec)
            log('QUALITY', sid, view, len(rec['rows']), 'rows')


def reps_for(spp):
    q = P['quality']
    for lim, n in q['reps_by_spp']:
        if spp <= lim:
            return n if not SMOKE else 2
    return 1


def exclusive_ok():
    """Windows WDDM lists every desktop process as a GPU client, so exclusivity is: no other python / CUDA worker process
    and an idle GPU (utilization median <= threshold over 3 s) right before and after the block."""
    t = P['timing']
    other = subprocess.run(['tasklist', '/fi', 'imagename eq python.exe', '/fo', 'csv', '/nh'], capture_output=True, text=True).stdout
    pids = [int(r.split('","')[1]) for r in other.strip().splitlines() if r.startswith('"python')]
    mine = {os.getpid(), os.getppid()}          # venv Scripts\\python.exe is a launcher whose child is the interpreter
    pids = [p for p in pids if p not in mine]
    util = []
    for _ in range(6):
        q = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'], capture_output=True, text=True).stdout
        util.append(float(q.strip().splitlines()[0]))
        time.sleep(0.5)
    ok = (not pids) and (float(np.median(util)) <= t['idle_util_max_pct'])
    return ok, {'other_python_pids': pids, 'idle_util_samples': util}


def timed_block(fn, label):
    t = P['timing']
    for attempt in range(t['max_attempts']):
        while True:
            ok, pre = exclusive_ok()
            if ok:
                break
            log('timing wait (not exclusive)', label, pre)
            time.sleep(30)
        res = fn()
        ok2, post = exclusive_ok()
        if ok2:
            return res, {'pre': pre, 'post': post, 'attempt': attempt, 'contaminated': False}
        log('timing block contaminated, repeat', label, post)
    return res, {'pre': pre, 'post': post, 'attempt': attempt, 'contaminated': True}


def stage_gates(world):
    """G_ADAPT (protocol references.validity_gate): pt111 at kmax 6 vs the archived WL100 E1..E6 reference (pt100.trace)."""
    out = ROOT / 'gates' / 'g_adapt.json'
    if out.exists():
        assert C.read_json(out)['pass'], 'G_ADAPT failed earlier: fixture workloads stopped'
        return
    rec = {'states': {}}
    ok = True
    for sid in ('S0', 'N0'):
        rt, Bd, case = world.get(sid)
        cam = camera('V0', case)
        ref = [np.load(R.ROOT100 / case / 'renders' / f'REF_0_{h}.npz') for h in (0, 1)]
        refm = {k: 0.5 * (ref[0][k] + ref[1][k]) for k in ('e1', 'e2', 'h')}
        refv = {k: 0.25 * (ref[0][k] - ref[1][k]) ** 2 for k in ('e1', 'e2', 'h')}
        n_ref = int(json.loads((R.ROOT100 / case / 'renders' / 'REF_0_0.json').read_text())['spp']) * 2
        rows_ = {}
        imgs = {}
        for name, kmax, rr in (('k6_norr', 6, 0), ('k6_rr4', 6, 4)):
            acc = pt111.pt_hq(rt, cam, RES, 1024, C.seed_of('gate111', sid, name), kmax=kmax, rr=rr)
            c = comp15(acc, 1024)
            row = {}
            for k in ('e1', 'e2', 'h'):
                imgs[(name, k)] = c[k]
                diff = (c[k] - refm[k]).reshape(-1, 3)
                mn, mr = c[k].reshape(-1, 3).mean(0), refm[k].reshape(-1, 3).mean(0)
                rel = (mn - mr) / np.maximum(mr, 1e-12)
                z = (mn - mr) / np.maximum(diff.std(0, ddof=1) / np.sqrt(diff.shape[0]), 1e-30)
                ron = float(np.sqrt(np.mean((c[k] - refm[k]) ** 2) / np.mean(refv[k] * (n_ref / 1024) + refv[k])))
                mean_ok = bool(np.all((np.abs(rel) <= 3e-3) | (np.abs(z) <= 4.0)))
                row[k] = {'rel_mean_diff': rel.tolist(), 'z': z.tolist(), 'mean_ok': mean_ok, 'rmse_over_noise': ron}
                if name == 'k6_norr':
                    ok &= mean_ok and (0.75 <= ron <= 1.25)
            rows_[name] = row
        for k in ('e1', 'e2', 'h'):
            a, b = imgs[('k6_rr4', k)], imgs[('k6_norr', k)]
            diff = (a - b).reshape(-1, 3)
            d = diff.mean(0) / np.maximum(b.reshape(-1, 3).mean(0), 1e-12)
            z = diff.mean(0) / np.maximum(diff.std(0, ddof=1) / np.sqrt(diff.shape[0]), 1e-30)
            mean_ok = bool(np.all((np.abs(d) <= 3e-3) | (np.abs(z) <= 4.0)))
            rows_.setdefault('rr_vs_norr', {})[k] = {'rel_mean_diff': d.tolist(), 'z': z.tolist(), 'mean_ok': mean_ok}
            ok &= mean_ok
        rec['states'][sid] = rows_
    rec['pass'] = bool(ok)
    C.write_json(out, rec)
    log('G_ADAPT', 'PASS' if ok else 'FAIL')
    assert ok, 'G_ADAPT failed: fixture workloads stopped (protocol stops)'


def stage_timing(world):
    t = P['timing']
    for sid in t['states']:
        for view in t['views']:
            out = ROOT / 'timing' / f'{sid}_{view}.json'
            if out.exists():
                continue
            rt, Bd, case = world.get(sid)
            cam = camera(view, case)
            den = Denoiser(cam)
            ops = [('ECRO', s) for s in t['ecro_spp']] + [('PT', s) for s in t['pt_spp']] + [('PTDN', s) for s in t['ptdn_spp']]
            if SMOKE:
                ops = [('ECRO', 1), ('PT', 1), ('PTDN', 1)]

            def call(arm, spp, i):
                seed = C.seed_of('t111', sid, view, arm, spp, i)
                if arm == 'ECRO':
                    rt.frame(Bd, cam, RES, spp, seed)
                elif arm == 'PT':
                    rt.pt_frame(cam, RES, spp, seed, kmax=P['quality']['pt_kmax'], rr=P['quality']['pt_rr'])
                else:
                    den.device(rt.pt_frame(cam, RES, spp, seed, kmax=P['quality']['pt_kmax'], rr=P['quality']['pt_rr']), spp)

            def block():
                rec = {'first_call_ms': {}, 'frame_ms': {f'{a}_{s}': [] for a, s in ops}, 'denoise_only_ms': {}}
                for a, s in ops:
                    sync(); t0 = time.perf_counter(); call(a, s, -1); sync()
                    rec['first_call_ms'][f'{a}_{s}'] = 1e3 * (time.perf_counter() - t0)
                    for w in range(t['warmup']):
                        call(a, s, -2 - w)
                sync()
                n_rounds = t['rounds'] if not SMOKE else 3
                rng = random.Random(C.seed_of('order111', sid, view))
                for rnd in range(n_rounds):
                    order = list(ops)
                    rng.shuffle(order)
                    for a, s in order:
                        reps = t['frames_per_round'] if a == 'ECRO' or s <= 64 else max(1, t['frames_per_round'] // 4)
                        for i in range(reps):
                            sync(); t0 = time.perf_counter(); call(a, s, rnd * 1000 + i); sync()
                            rec['frame_ms'][f'{a}_{s}'].append(1e3 * (time.perf_counter() - t0))
                # denoiser alone (input already on the device), for cost accounting
                for s in t['ptdn_spp'][:1]:
                    o = rt.pt_frame(cam, RES, s, 777, kmax=P['quality']['pt_kmax'], rr=P['quality']['pt_rr'])
                    sync()
                    ds = []
                    for i in range(50 if not SMOKE else 3):
                        sync(); t0 = time.perf_counter(); den.device(o, s); sync()
                        ds.append(1e3 * (time.perf_counter() - t0))
                    rec['denoise_only_ms'] = ds
                return rec
            rec, excl = timed_block(block, f'{sid}_{view}')
            rec['summary'] = {k: summ(v) for k, v in rec['frame_ms'].items()}
            rec['summary']['denoise_only'] = summ(rec['denoise_only_ms'])
            rec['exclusivity'] = excl
            rec['snapshot_end'] = C.gpu_snapshot()
            C.write_json(out, rec)
            log('TIMING', sid, view, {k: round(v['median'], 3) for k, v in rec['summary'].items()})


def summ(v):
    v = np.asarray(v, np.float64)
    if v.size == 0:
        return {}
    return {'n': int(v.size), 'median': float(np.median(v)), 'mean': float(v.mean()), 'p95': float(np.percentile(v, 95)),
            'p99': float(np.percentile(v, 99)), 'min': float(v.min()), 'max': float(v.max())}


def stage_dynamic(world):
    """W3 edit lifecycle (T0 <-> T1 box geometry) and W5 light change, edit -> first displayed frame, ECRO selective / full
    rebuild, PT, PTDN; plus stale-ECRO quality control at T1 (table of T0 with T1 geometry)."""
    d = P['dynamic']
    out = ROOT / 'dynamic' / 'dynamic.json'
    if out.exists():
        return
    rec = {}
    for name, spec in d['transitions'].items():
        caseA, gA, lA = state_def(spec['from'])
        caseB, gB, lB = state_def(spec['to'])
        rt, Bd, case = world.get(spec['from'])
        cam = camera('V0', case)
        den = Denoiser(cam)
        res_ = {}

        def ecro_edit(target, full):
            g, l = (gB, lB) if target else (gA, lA)
            moved, old = rt.set_geom(g)
            if full:
                rt.rebuild_state()
            else:
                rt.update_state(old, moved)
            rt.set_light(l)
            s = rt.source()
            B, _ = rt.respond(s)
            Bn = R.t2d(B.view(-1))
            rt.frame(Bn, cam, RES, d['spp'], 4242)
            return Bn

        def pt_edit(target, dn):
            g, l = (gB, lB) if target else (gA, lA)
            rt.set_geom(g)
            rt.set_light(l)
            o = rt.pt_frame(cam, RES, d['spp'], 4243, kmax=P['quality']['pt_kmax'], rr=P['quality']['pt_rr'])
            if dn:
                den.device(o, d['spp'])

        arms = {'ECRO_sel': lambda tg: ecro_edit(tg, False), 'ECRO_full': lambda tg: ecro_edit(tg, True),
                'PT': lambda tg: pt_edit(tg, False), 'PTDN': lambda tg: pt_edit(tg, True)}

        def block():
            r = {k: [] for k in arms}
            for k, f in arms.items():
                for w in range(d['warmup_pairs'] if not SMOKE else 1):
                    f(True); f(False)
            sync()
            rng = random.Random(C.seed_of('dyn111', name))
            for rnd in range(d['rounds'] if not SMOKE else 2):
                order = list(arms)
                rng.shuffle(order)
                for k in order:
                    for tg in (True, False):
                        sync(); t0 = time.perf_counter(); arms[k](tg); sync()
                        r[k].append({'to_target': tg, 'ms': 1e3 * (time.perf_counter() - t0)})
            return r
        r, excl = timed_block(block, name)
        res_['edit_to_frame_ms'] = {k: summ([x['ms'] for x in v if x['to_target']]) for k, v in r.items()}
        res_['edit_to_frame_ms_reverse'] = {k: summ([x['ms'] for x in v if not x['to_target']]) for k, v in r.items()}
        res_['raw'] = r
        res_['exclusivity'] = excl
        # stale-ECRO quality control at the target state (table of the source state, target geometry / light)
        world.cur[case] = None
        rt2, BdA, _ = world.get(spec['from'])
        BA = BdA
        moved, old = rt2.set_geom(gB)
        rt2.update_state(old, moved)
        rt2.set_light(lB)
        world.cur[case] = None
        ref = load_ref(spec['to'], 'V0', 'FULL')
        with np.load(ROOT / 'refs' / f"{spec['to']}_V0_masks.npz") as z:
            masks = {k: z[k] for k in z.files}
        rows_ = []
        for spp in d['stale_spp']:
            for rep in range(reps_for(spp)):
                o = rt2.frame(BA, cam, RES, spp, C.seed_of('stale111', name, spp, rep))
                e = ecro_image(o, spp)
                rows_.append({'arm': 'ECRO_STALE', 'spp': spp, 'rep': rep, 'vs_full': evaluate(e['e1'] + e['e2'] + e['hl'], ref, masks, case)})
        res_['stale_quality'] = rows_
        rec[name] = res_
        log('DYNAMIC', name, {k: round(v['median'], 3) for k, v in res_['edit_to_frame_ms'].items()})
    rec['snapshot_end'] = C.gpu_snapshot()
    C.write_json(out, rec)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('root')
    ap.add_argument('stage')
    ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args()
    SMOKE = a.smoke
    ROOT = Path(a.root)
    for sub in ('gates', 'refs', 'quality', 'timing', 'dynamic'):
        (ROOT / sub).mkdir(parents=True, exist_ok=True)
    man = ROOT / 'manifest.json'
    if not man.exists():
        C.write_json(man, {'protocol_sha256': C.sha(PROTOCOL), 'code_sha256': {p.name: C.sha(p) for p in sorted(HERE.glob('*.py'))},
                           'versions': C.versions(mi, dr, torch), 'smoke': SMOKE, 'snapshot': C.gpu_snapshot(),
                           'restgi_repo': str(pt111.REPO), 'frozen_sources_sha256': {
                               n: C.sha(pt111.REPO / 'experiments' / n) for n in P['frozen_sources']},
                           'checkpoints_sha256': {c: C.sha(R.ROOT100 / c / 'model.pt') for c in ('sss', 'near')}})
    else:
        m = C.read_json(man)
        assert m['protocol_sha256'] == C.sha(PROTOCOL), 'protocol changed after the run started'
    world = World()
    stages = ['gates', 'refs', 'quality', 'timing', 'dynamic'] if a.stage == 'all' else [a.stage]
    for st in stages:
        log('STAGE', st, 'start')
        {'gates': stage_gates, 'refs': stage_refs, 'quality': stage_quality, 'timing': stage_timing,
         'dynamic': stage_dynamic}[st](world)
        log('STAGE', st, 'done')
    log('RUN111 DONE', a.stage)
