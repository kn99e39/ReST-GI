"""WL99A: diagnosis of the WL99 Q3 failures and Stage 0 qualification under protocol rbdto99a_v2.

  python qual99a.py <wl99_stage0_dir> <out_dir>
Phases (cached, resumable):
  1 D1  reproduce WL99 Q3 spatial values from the stored 384 references.
  2 refs at 768 (8192 spp per half) for T0 / T1 of every WL99 candidate; Q3 v1 and v2 at 384 and 768; footprints.
  3 independent candidates: v1 Stage 0 references at 384 (T0, T1, stale [, co]) with the v1 extension rule, + 768 halves.
  4 1536 T0 (4096 spp per half) for every candidate whose v2 C changes by > 0.20 between 384 and 768.
  5 matched-spp co-moved references for Q4 where the T0 reference was extended (A1, A10).
  6 v2 gates, classification A/B/C/D for WL99 Q3 failures, required set, verdict.
"""
import hashlib, json, os, sys, time
import numpy as np
import pt99, wl99, q3v2
import scene99 as S
import qual99 as Q1

S0, OUT = sys.argv[1], sys.argv[2]
DEV = '--dev' in sys.argv                     # development smoke only: tiny spp, never evidence
SPP = (lambda s: max(16, s // 256)) if DEV else (lambda s: s)
os.makedirs(os.path.join(OUT, 'refs'), exist_ok=True)
P = json.load(open(os.path.join(wl99.HERE, 'protocol', 'rbdto99a.json')))
IND = json.load(open(os.path.join(wl99.HERE, 'protocol', 'workloads99a_indep.json')))['workloads']
QV1 = json.load(open(os.path.join(S0, 'qual99.json')))['workloads']
ctx = None


def status(msg):
    with open(os.path.join(OUT, 'qual99a.status'), 'a') as f:
        f.write(time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()) + ' ' + msg + '\n')


def set_ctx(cfg):
    global ctx
    if ctx is None:
        ctx = pt99.Ctx(cfg)
    else:
        ctx.set_cfg(cfg)


def ref(name, cfg, half, spp, res, store):
    """Reference half at resolution res; WL99 384 references are read from the WL99 store (never re-rendered)."""
    path = os.path.join(store, 'refs', f'{name}_{half}_{spp}.npz') if res == 384 else os.path.join(OUT, 'refs', f'{name}_{half}_{spp}_r{res}.npz')
    if os.path.exists(path):
        return dict(np.load(path))
    set_ctx(cfg)
    seed = (990001 if half == 'A' else 990501) + int(hashlib.sha256(name.encode()).hexdigest()[:6], 16) % 400 + (0 if res == 384 else 7000 + res)
    b = max(1, int(24_000_000 // (res * res)))                      # ~24 M lanes per launch (38 M exhausted GPU memory at 384 px, 05:23 stall)
    acc, Tm = ctx.render(cfg['camera'], res, spp, seed=seed, b=b)
    r = pt99.pack(acc, spp)
    out = {k: (v.reshape(res, res, 3) if v.ndim == 2 else v.reshape(res, res)).astype(np.float32) for k, v in r.items()}
    np.savez(path, **out)
    status(f'REF {name} {half} spp={spp} res={res} wall={Tm["wall_s"]:.1f}s gpu_ms={Tm["gpu_ms"]:.0f}')
    return out


JOBS = {}


def ref_path(name, half, spp, res, store):
    return os.path.join(store, 'refs', f'{name}_{half}_{spp}.npz') if res == 384 else os.path.join(OUT, 'refs', f'{name}_{half}_{spp}_r{res}.npz')


def pair(name, cfg, spp, res, store):
    A = ref(name, cfg, 'A', spp, res, store); B = ref(name, cfg, 'B', spp, res, store)
    return A, B, {k: (A[k].astype(np.float64) + B[k]) / 2 for k in A}


def masks_res(R, res):
    """WL99 masks at any resolution (wl99.masks uses only the images)."""
    return wl99.masks(R)


def footprint(cfg, res, roi, nmax=4000):
    set_ctx(cfg)
    pix = np.nonzero(roi.ravel())[0]
    if len(pix) == 0:
        return float('nan')
    pix = pix[np.linspace(0, len(pix) - 1, min(nmax, len(pix))).astype(int)]
    y, wcam, isb = ctx.pixel_hits(cfg['camera'], res, pix)
    o = np.array(cfg['camera']['origin'], float)
    d = np.linalg.norm(y - o[None], axis=1)
    # surface normal of the hit plate: find via the body plates (world)
    Mp = S.pose_matrix(cfg['body']['pose'])
    plates = S.body_plates(cfg['body']['kind'], cfg['body']['prm'])
    nrm = np.array([Mp[:3, :3] @ p['n'] for p in plates]); ctr = np.array([Mp[:3, :3] @ p['c'] + Mp[:3, 3] for p in plates])
    dist = np.abs(((y[:, None, :] - ctr[None]) * nrm[None]).sum(-1))
    n = nrm[np.argmin(dist, 1)]
    cos = np.clip(np.abs((wcam * n).sum(1)), 0.05, 1.0)
    pitch = 2 * np.tan(np.radians(cfg['camera']['fov_deg']) / 2) / res
    return float(np.median(d * pitch / np.sqrt(cos)))


def v1_q3(R, sharp):
    return Q1.q3_spatial(R, sharp)


def analyse(name, cfg, A, B, R, res):
    """v2 (cheap) now; the WL99 v1 Q3 statistics (expensive window quantiles) are computed after all renders by v1_pass()."""
    body, roi, sharp = masks_res(R, res)
    f = footprint(cfg, res, roi)
    return {'v2': q3v2.measure(A, B, roi, R['mb'], f, res), 'footprint_mm': 1e3 * f}


def _v1_job(args):
    pa, pb = args
    A = dict(np.load(pa)); B = dict(np.load(pb))
    R = {k: (A[k].astype(np.float64) + B[k]) / 2 for k in A}
    sharp = wl99.masks(R)[2]
    return {'v1': v1_q3(R, sharp), 'v1_halfA': v1_q3(A, sharp), 'v1_halfB': v1_q3(B, sharp)}


def v1_pass(R_, jobs):
    """WL99 v1 Q3 on every analysed (candidate, state, resolution) in parallel CPU processes."""
    from multiprocessing import Pool
    keys = list(jobs)
    with Pool(min(24, len(keys))) as pool:
        res = pool.map(_v1_job, [jobs[k] for k in keys])
    for (wid, key), r in zip(keys, res):
        R_['candidates'][wid][key].update(r)


def cand_list():
    wl = [dict(w, set='development') for w in wl99.WL]
    out = wl + [dict(w, set='independent') for w in IND]
    return [out[0], out[3], out[15]] if DEV else out


def main():
    status('START')
    res_path = os.path.join(OUT, 'qual99a.json')
    R_ = json.load(open(res_path)) if os.path.exists(res_path) else {'protocol': P['protocol'], 'candidates': {}}
    save = lambda: json.dump(R_, open(res_path, 'w'), indent=1, default=float)
    for w in cand_list():
        wid = w['id']; c = R_['candidates'].setdefault(wid, {'set': w['set'], 'type': w['type'], 'role': w['role']})
        cf = wl99.configs(w)
        if w['set'] == 'development':
            spp = QV1[wid]['spp_per_half']; store = S0
        else:
            spp = SPP(16384); store = OUT
            # v1 Stage 0 references at 384 for independent candidates (with the v1 extension rule for T0 / T1)
            refs = {n: pair(f'{wid}_{n}', cf[n], spp, 384, store) for n in cf}
            ev = Q1.evaluate(w, cf, refs, spp)
            while not DEV and w['type'] != 'C' and not ev['gates']['Q_ref'] and spp < 65536:
                spp *= 2
                status(f'EXTEND {wid} T0/T1 -> {spp}')
                for n in ('T0', 'T1'):
                    refs[n] = pair(f'{wid}_{n}', cf[n], spp, 384, store)
                ev = Q1.evaluate(w, cf, refs, spp)
            c['v1_eval'] = ev
        c['spp_384'] = spp
        if w['type'] == 'C':
            c['v2_admitted'] = bool(QV1[wid]['admitted']) if wid in QV1 else None
            save(); continue
        for st in ('T0', 'T1'):
            a384 = pair(f'{wid}_{st}', cf[st], spp, 384, store)
            a768 = pair(f'{wid}_{st}', cf[st], SPP(8192), 768, OUT)
            c[f'{st}_384'] = analyse(wid, cf[st], *a384, 384)
            c[f'{st}_768'] = analyse(wid, cf[st], *a768, 768)
            JOBS[(wid, f'{st}_384')] = (ref_path(f'{wid}_{st}', 'A', spp, 384, store), ref_path(f'{wid}_{st}', 'B', spp, 384, store))
            JOBS[(wid, f'{st}_768')] = (ref_path(f'{wid}_{st}', 'A', SPP(8192), 768, OUT), ref_path(f'{wid}_{st}', 'B', SPP(8192), 768, OUT))
            C3, C7 = c[f'{st}_384']['v2']['C'], c[f'{st}_768']['v2']['C']
            c[f'{st}_conv_384_768'] = abs(C3 - C7) / max(C7, 1e-12)
            if c[f'{st}_conv_384_768'] > 0.20 and st == 'T0':
                a1536 = pair(f'{wid}_{st}', cf[st], SPP(4096), 1536, OUT)
                c[f'{st}_1536'] = analyse(wid, cf[st], *a1536, 1536)
                JOBS[(wid, f'{st}_1536')] = (ref_path(f'{wid}_{st}', 'A', SPP(4096), 1536, OUT), ref_path(f'{wid}_{st}', 'B', SPP(4096), 1536, OUT))
                c[f'{st}_conv_768_1536'] = abs(C7 - c[f'{st}_1536']['v2']['C']) / max(c[f'{st}_1536']['v2']['C'], 1e-12)
            save()
        status(f'Q3 {wid} done')
    # matched-spp co-moved references (Q4) for development candidates whose T0 was extended
    for w in wl99.WL:
        if not DEV and w.get('co_check') and QV1[w['id']]['spp_per_half'] > 16384:
            spp = QV1[w['id']]['spp_per_half']; cf = wl99.configs(w)
            Aco, Bco, Rco = pair(f"{w['id']}_co", cf['co'], spp, 384, OUT)
            A0, B0, R0 = pair(f"{w['id']}_T0", cf['T0'], spp, 384, S0)
            m0 = wl99.masks(R0)
            d = wl99.rel_l2(Rco['Sd'], R0['Sd'], m0[1])
            lim = max(3 * wl99.rel_noise(A0['Sd'], B0['Sd'], m0[1]), 0.02)
            R_['candidates'][w['id']]['Q4_matched'] = {'spp': spp, 'rel_diff': d, 'limit': lim, 'pass': bool(d <= lim)}
            save()
    status('V1_PASS start')
    v1_pass(R_, JOBS)
    finalize(R_)
    save(); status('END ' + R_['verdict'])


def finalize(R_):
    cmin = P['Q3_v2']['c_min']
    for wid, c in R_['candidates'].items():
        if c['type'] == 'C':
            continue
        v1 = QV1.get(wid) or c.get('v1_eval')
        g = dict(v1['gates'])
        if 'Q4_matched' in c:
            g['Q4'] = bool(c['Q4_matched']['pass'] and v1['cert']['T0'] and v1['cert']['T1'])
        ang = v1['Q3_angular']['pass']
        out = {}
        for st in ('T0', 'T1'):
            r384, r768 = c[f'{st}_384']['v2'], c[f'{st}_768']['v2']
            conv = c[f'{st}_conv_384_768']
            if r384['scale_px'] >= 4 and conv <= 0.20:
                er, C, se = 384, r384['C'], r384['se']
            elif r768['scale_px'] >= 4 and c.get(f'{st}_conv_768_1536', 1e9) <= 0.20:
                er, C, se = 768, r768['C'], r768['se']
            else:
                er, C, se = None, r768['C'], r768['se']
            out[st] = {'eval_res': er, 'C': C, 'se': se, 'structure': bool(C >= cmin), 'resolved': er is not None, 'reliable': bool(se <= 0.25 * C)}
        q3v2_pass = all(o['structure'] and o['resolved'] and o['reliable'] for o in out.values()) and ang
        c['Q3v2'] = out; c['Q3v2_pass'] = bool(q3v2_pass)
        c['eval_res'] = max([o['eval_res'] or 0 for o in out.values()]) or None
        g2 = {k: v for k, v in g.items() if k != 'Q3'}; g2['Q3v2'] = bool(q3v2_pass)
        c['gates_v2'] = g2; c['v2_admitted'] = bool(all(g2.values()))
        c['v1_admitted'] = bool(v1['admitted'])
        if not v1['gates']['Q3']:
            cls = []
            if not any(o['structure'] for o in out.values()) or not all(o['structure'] for o in out.values()):
                cls.append('A_insufficient_structure' if all(o['reliable'] for o in out.values()) else 'A_or_C_unreliable')
            if not all(o['resolved'] for o in out.values()):
                cls.append('B_under_resolved')
            w1 = [c[f'{st}_384']['v1']['edge_width_px'] for st in ('T0', 'T1')]
            w2 = [c[f'{st}_768']['v1']['edge_width_px'] for st in ('T0', 'T1')]
            if all(o['structure'] and o['resolved'] and o['reliable'] for o in out.values()):
                cls.append('D_definition_false_rejection')
            c['v1_q3_failure_class'] = cls or ['unclassified']
            c['v1_edge_width_384_vs_768'] = [w1, w2]
    adm = lambda pred: any(c.get('v2_admitted') and pred(wid, c) for wid, c in R_['candidates'].items())
    WI = {w['id']: w for w in IND}
    req = {'dev_A': adm(lambda k, c: c['set'] == 'development' and c['type'] == 'A'),
           'dev_B': adm(lambda k, c: c['set'] == 'development' and c['type'] == 'B'),
           'C_null': bool(R_['candidates'].get('C_box', {}).get('v2_admitted')),
           'indep_A_nongroove': adm(lambda k, c: c['set'] == 'independent' and c['type'] == 'A' and WI[k]['body']['kind'] != 'groove'),
           'indep_second': sum(1 for k, c in R_['candidates'].items() if c['set'] == 'independent' and c.get('v2_admitted')) >= 2,
           'Q6_intrusion': any(QV1[k].get('Q6', {}).get('pass') for k in QV1)}
    R_['required_set_v2'] = req
    R_['verdict'] = 'QUALIFIED_V2' if all(req.values()) else 'STOPPED_WORKLOAD_UNQUALIFIED_V2'


if __name__ == '__main__':
    main()
