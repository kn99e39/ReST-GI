"""WL99 Stage 0: references (independent halves), qualification metrics and admission (protocol rbdto99_v1).

  python qual99.py <out_dir> [--dev]      (--dev: tiny spp, development only, never evidence)
"""
import hashlib, json, os, sys, time
import numpy as np
import pt99, wl99
import scene99 as S

OUT = sys.argv[1]; DEV = '--dev' in sys.argv
os.makedirs(os.path.join(OUT, 'refs'), exist_ok=True)
P0 = wl99.P['stage0']
SPP = 64 if DEV else P0['reference']['spp_per_half']
RES = wl99.RES
LIM_REF = 0.025
ctx = None


def status(msg):
    with open(os.path.join(OUT, 'qual.status'), 'a') as f:
        f.write(time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()) + ' ' + msg + '\n')


def set_ctx(cfg):
    global ctx
    if ctx is None:
        ctx = pt99.Ctx(cfg)
    else:
        ctx.set_cfg(cfg)


def render_half(name, cfg, half, spp):
    path = os.path.join(OUT, 'refs', f'{name}_{half}_{spp}.npz')
    if os.path.exists(path):
        return dict(np.load(path))
    set_ctx(cfg)
    seed = (990001 if half == 'A' else 990501) + int(hashlib.sha256(name.encode()).hexdigest()[:6], 16) % 400
    acc, Tm = ctx.render(cfg['camera'], RES, spp, seed=seed)
    r = pt99.pack(acc, spp)
    out = {k: (v.reshape(RES, RES, 3) if v.ndim == 2 else v.reshape(RES, RES)).astype(np.float32) for k, v in r.items()}
    np.savez_compressed(path, **out)
    status(f'REF {name} {half} spp={spp} wall={Tm["wall_s"]:.1f}s')
    return out


def ref_pair(name, cfg, spp):
    A = render_half(name, cfg, 'A', spp); B = render_half(name, cfg, 'B', spp)
    R = {k: (A[k].astype(np.float64) + B[k]) / 2 for k in A}
    return A, B, R


def qref(A, B, roi):
    return {'tot': wl99.rel_noise(A['tot'], B['tot'], roi), 'Sd': wl99.rel_noise(A['Sd'], B['Sd'], roi)}


def q3_spatial(R, sharp):
    sd = wl99.lum(R['Sd'])
    if sharp.sum() < 10:
        return {'n_sharp': int(sharp.sum()), 'edge_width_px': None, 'contrast': None, 'pass': False}
    rng, gmax, mean = wl99.window_stats(sd, 3)
    w = rng[sharp] / np.maximum(gmax[sharp], 1e-30)
    c = rng[sharp] / np.maximum(mean[sharp], 1e-30)
    ew, ct = float(np.median(w)), float(np.median(c))
    return {'n_sharp': int(sharp.sum()), 'edge_width_px': ew, 'contrast': ct, 'pass': bool(ew >= 2.0 and ct >= 0.30)}


def q3_angular(cfg, R, sharp, spp):
    set_ctx(cfg)
    sd = wl99.lum(R['Sd'])
    gy, gx = np.gradient(sd)
    g = np.where(sharp, np.hypot(gx, gy), -1.0)
    pix = np.argsort(g.ravel())[::-1][:3]
    y, wcam, isb = ctx.pixel_hits(cfg['camera'], RES, pix)
    steps = np.arange(-16, 17) * 0.25
    mid = len(steps) // 2
    pitch_deg = cfg['camera']['fov_deg'] / RES
    res = []
    for p in range(len(pix)):
        if not isb[p]:
            continue
        w0 = wcam[p]
        a1 = np.cross(w0, [0, 0, 1.0]); a1 /= np.linalg.norm(a1); a2 = np.cross(w0, a1)
        dirs = np.array([w0 * np.cos(np.radians(s)) + ax * np.sin(np.radians(s)) for ax in (a1, a2) for s in steps])
        acc = ctx.probe(np.repeat(y[p:p + 1], len(dirs), 0), dirs, spp, seed=995001 + p)
        rr = pt99.pack(acc, spp)
        prof = rr['Sd'].mean(1).reshape(2, len(steps))
        c0 = prof[:, mid].mean()
        th = None
        for i in range(1, mid + 1):
            if c0 > 0 and max(abs(prof[pl, mid + sg * i] - c0) / c0 for pl in range(2) for sg in (-1, 1)) >= 0.25:
                th = float(steps[mid + i]); break
        res.append({'pixel': int(pix[p]), 'theta25_deg': th, 'ok_frac': float(rr['ok'].mean()), 'profile': prof.tolist()})
    ths = [r['theta25_deg'] for r in res if r['theta25_deg'] is not None]
    tmin = min(ths) if ths else 4.25                                   # > scanned range: smooth over +-4 deg
    return {'points': res, 'theta25_min_deg': tmin, 'pixel_pitch_deg': pitch_deg,
            'pass': bool(len(res) > 0 and tmin >= 2 * pitch_deg)}


def evaluate(w, cf, refs, spp):
    res = {'type': w['type'], 'role': w['role'], 'cert': {k: S.certificate(c) for k, c in cf.items()}, 'spp_per_half': spp}
    R0, R1 = refs['T0'][2], refs['T1'][2]
    m0 = wl99.masks(R0); m1 = wl99.masks(R1)
    qr = {'T0': qref(refs['T0'][0], refs['T0'][1], m0[1]), 'T1': qref(refs['T1'][0], refs['T1'][1], m1[1])}
    res['Q_ref'] = qr
    res['roi_px'] = {'T0': [int(m.sum()) for m in m0], 'T1': [int(m.sum()) for m in m1]}
    tags = {}
    for t, (R, m) in (('T0', (R0, m0)), ('T1', (R1, m1))):
        body = m[0]
        tot = wl99.lum(R['tot'])[body].sum()
        tags[t] = {k: float(wl99.lum(R[k])[body].sum() / max(tot, 1e-300)) for k in ('B1', 'S2', 'S3', 'SX', 'X', 'Sd', 'self')}
        tags[t]['S3_of_Sd'] = float(wl99.lum(R['S3'])[body].sum() / max(wl99.lum(R['Sd'])[body].sum(), 1e-300))
    res['shares_body'] = tags
    if w['type'] == 'C':
        null = {t: float(sum(np.abs(refs[t][2][k]).sum() for k in ('S2', 'S3', 'SX', 'ES')) / max(np.abs(refs[t][2]['tot']).sum(), 1e-300))
                for t in ('T0', 'T1')}
        res['Q1_null'] = null
        res['gates'] = {'Q1_null': all(v <= 1e-6 for v in null.values())}
        res['note'] = 'null workload: ROI_self is empty by construction, Q_ref is not applicable; admission by Q1-null'
        res['admitted'] = res['gates']['Q1_null']
        return res
    qref_pass = bool(max(v for d in qr.values() for v in d.values()) <= LIM_REF)
    res['Q1'] = {t: tags[t]['Sd'] for t in tags}
    res['Q3_spatial'] = {'T0': q3_spatial(R0, m0[2]), 'T1': q3_spatial(R1, m1[2])}
    res['Q3_angular'] = q3_angular(cf['T0'], R0, m0[2], 64 if DEV else 4096)
    res['Q5'] = wl99.rel_l2(refs['stale'][2]['Sd'], R1['Sd'], m1[1])
    q4 = bool(res['cert']['T0'] and res['cert']['T1'])
    if 'co' in refs:
        d = wl99.rel_l2(refs['co'][2]['Sd'], R0['Sd'], m0[1])
        lim = max(3 * qr['T0']['Sd'], 0.02)
        res['Q4'] = {'rel_diff': d, 'limit': lim, 'pass': bool(d <= lim)}
        q4 = q4 and res['Q4']['pass']
    if 'T1x' in refs:
        d = wl99.rel_l2(refs['T1x'][2]['Sd'], R1['Sd'], m1[1])
        res['Q6'] = {'cert_T1x': res['cert']['T1x'], 'rel_change': d, 'pass': bool((not res['cert']['T1x']) and d >= 0.10)}
    res['gates'] = {'Q_ref': qref_pass, 'Q1': all(v >= 0.20 for v in res['Q1'].values()),
                    'Q3': bool(res['Q3_spatial']['T0']['pass'] and res['Q3_spatial']['T1']['pass'] and res['Q3_angular']['pass']),
                    'Q4': q4, 'Q5': bool(res['Q5'] >= 0.30)}
    res['admitted'] = bool(all(res['gates'].values()))
    return res


def main():
    status(f'START dev={DEV} spp={SPP}')
    summary = {'protocol': wl99.P['protocol'], 'spp_per_half': SPP, 'dev': DEV, 'workloads': {}}
    for w in wl99.WL:
        cf = wl99.configs(w)
        refs = {n: ref_pair(f"{w['id']}_{n}", cf[n], SPP) for n in cf}
        res = evaluate(w, cf, refs, SPP)
        spp = SPP
        while not DEV and w['type'] != 'C' and not res['gates']['Q_ref'] and spp < 65536:
            spp *= 2                                                   # predeclared extension of T0 / T1 only
            status(f"EXTEND {w['id']} T0/T1 -> {spp}")
            for n in ('T0', 'T1'):
                refs[n] = ref_pair(f"{w['id']}_{n}", cf[n], spp)
            res = evaluate(w, cf, refs, spp)
        summary['workloads'][w['id']] = res
        json.dump(summary, open(os.path.join(OUT, 'qual99.json'), 'w'), indent=1)
        status(f"DONE {w['id']} admitted={res['admitted']} gates={res['gates']}")
    adm = {k: v['admitted'] for k, v in summary['workloads'].items()}
    WLd = {w['id']: w for w in wl99.WL}
    got = lambda pred: any(adm[k] and pred(WLd[k]) for k in adm)
    req = {
        'A_dev_geometry': got(lambda w: w['type'] == 'A' and w['body']['kind'] == 'groove' and 'held-out' not in w['role']),
        'B': got(lambda w: w['type'] == 'B'),
        'C_null': got(lambda w: w['type'] == 'C'),
        'A_heldout_geometry': got(lambda w: w['type'] == 'A' and 'held-out geometry' in w['role']),
        'A_heldout_illumination': got(lambda w: w['type'] == 'A' and 'held-out illumination' in w['role']),
        'Q6_intrusion': any(v.get('Q6', {}).get('pass', False) for v in summary['workloads'].values()),
    }
    summary['required_set'] = req
    summary['stage0_verdict'] = 'QUALIFIED' if all(req.values()) else 'NOT_QUALIFIED_WITHIN_FIRST_CANDIDATES'
    json.dump(summary, open(os.path.join(OUT, 'qual99.json'), 'w'), indent=1)
    status('END ' + summary['stage0_verdict'])


if __name__ == '__main__':
    main()
