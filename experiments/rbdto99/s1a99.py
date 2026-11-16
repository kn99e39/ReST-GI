"""WL99 Stage 1a: zero-cost-oracle utility ceiling U0 (learning-free).

  python s1a99.py ladder <stage0_dir> <out_dir>     errors of E1 (mode cam) and E_oracle (mode oracle_self + reference self) vs reference
  python s1a99.py timing <out_dir>                  GPU kernel ms per spp of both modes (run with the GPU otherwise idle)
  python s1a99.py u0 <stage0_dir> <out_dir>         combine -> U0(tau) per admitted workload / state

Error metric: relative RMSE of the frame total on ROI_self, reference-noise corrected (err^2 - sigma_ref^2).
E_oracle's reference noise is the noise of the non-self part (tot - self) only, because its self part IS the reference.
Variance model err^2 = v / spp (unbiased estimators), v fitted per arm as the mean of spp * (err^2 - sigma^2) over the ladder.
"""
import hashlib, json, os, sys, time
import numpy as np
import pt99, wl99

LADDER = [16, 64, 256]
SEEDS = [991101, 991203, 991305, 991407]
TAUS = [0.05, 0.10]
RES = wl99.RES


def load_ref(s0, wid, state, spp):
    A = dict(np.load(os.path.join(s0, 'refs', f'{wid}_{state}_A_{spp}.npz')))
    B = dict(np.load(os.path.join(s0, 'refs', f'{wid}_{state}_B_{spp}.npz')))
    return A, B, {k: (A[k].astype(np.float64) + B[k]) / 2 for k in A}


def admitted(s0):
    q = json.load(open(os.path.join(s0, 'qual99.json')))
    return {k: v for k, v in q['workloads'].items() if v['admitted'] and v['type'] != 'C'}


def ladder(s0, out):
    os.makedirs(out, exist_ok=True)
    WLd = {w['id']: w for w in wl99.WL}
    ctx = None
    res = {}
    for wid, q in admitted(s0).items():
        cf = wl99.configs(WLd[wid])
        for state in ('T0', 'T1'):
            A, B, R = load_ref(s0, wid, state, q['spp_per_half'])
            roi = wl99.masks(R)[1]
            ns = lambda X: X['tot'] - X['self']
            sig_tot = wl99.rel_noise(A['tot'], B['tot'], roi)
            sig_ns = float(np.sqrt(((ns(A) - ns(B))[roi] ** 2).sum()) / (2 * np.sqrt((R['tot'][roi] ** 2).sum())))
            if ctx is None:
                ctx = pt99.Ctx(cf[state])
            else:
                ctx.set_cfg(cf[state])
            rows = []
            for spp in LADDER:
                for sd in SEEDS:
                    seed = sd + spp
                    for mode in ('cam', 'oracle_self'):
                        acc, Tm = ctx.render(cf[state]['camera'], RES, spp, seed, mode=mode)
                        r = pt99.pack(acc, spp)
                        tot = r['tot'].reshape(RES, RES, 3)
                        if mode == 'oracle_self':
                            tot = tot + R['self']
                        e2 = float(((tot - R['tot'])[roi] ** 2).sum() / (R['tot'][roi] ** 2).sum())
                        sig = sig_tot if mode == 'cam' else sig_ns
                        rows.append({'mode': mode, 'spp': spp, 'seed': seed, 'err2': e2, 'sig2': sig ** 2, 'v': spp * (e2 - sig ** 2)})
            res[f'{wid}/{state}'] = {'sigma_ref_tot': sig_tot, 'sigma_ref_nonself': sig_ns, 'roi_px': int(roi.sum()), 'rows': rows}
            json.dump(res, open(os.path.join(out, 'ladder.json'), 'w'), indent=1)
    return res


def timing(out):
    """GPU kernel ms per spp, both modes, every admitted workload state; 3 repeats of 256 spp after one warm-up render."""
    s0 = sys.argv[3] if len(sys.argv) > 3 else None
    WLd = {w['id']: w for w in wl99.WL}
    ctx = None
    res = {}
    for wid in admitted(s0):
        cf = wl99.configs(WLd[wid])
        for state in ('T0', 'T1'):
            if ctx is None:
                ctx = pt99.Ctx(cf[state])
            else:
                ctx.set_cfg(cf[state])
            row = {}
            for mode in ('cam', 'oracle_self'):
                ctx.render(cf[state]['camera'], RES, 16, 7, mode=mode)               # warm-up (compilation excluded)
                g, wl = [], []
                for rep in range(3):
                    _, Tm = ctx.render(cf[state]['camera'], RES, 256, 100 + rep, mode=mode)
                    g.append(Tm['gpu_ms'] / 256); wl.append(Tm['wall_s'] / 256)
                row[mode] = {'gpu_ms_per_spp': float(np.median(g)), 'wall_s_per_spp': float(np.median(wl)), 'reps_gpu': g}
            res[f'{wid}/{state}'] = row
            json.dump(res, open(os.path.join(out, 'timing.json'), 'w'), indent=1)
    return res


def u0(s0, out):
    L = json.load(open(os.path.join(out, 'ladder.json')))
    Tt = json.load(open(os.path.join(out, 'timing.json')))
    res = {}
    for key, d in L.items():
        v = {}
        for mode in ('cam', 'oracle_self'):
            vs = [r['v'] for r in d['rows'] if r['mode'] == mode]
            v[mode] = {'v_mean': float(np.mean(vs)), 'v_se': float(np.std(vs, ddof=1) / np.sqrt(len(vs)))}
        t = Tt[key]
        row = {'v': v, 't_ms_per_spp': {m: t[m]['gpu_ms_per_spp'] for m in t}}
        for tau in TAUS:
            sE = max(v['cam']['v_mean'], 0) / tau ** 2
            sO = max(max(v['oracle_self']['v_mean'], 0) / tau ** 2, 1.0)
            TE = sE * t['cam']['gpu_ms_per_spp']; TO = sO * t['oracle_self']['gpu_ms_per_spp']
            row[f'tau{tau}'] = {'spp_E1': sE, 'spp_oracle': sO, 'T_E1_ms': TE, 'T_oracle_ms': TO, 'U0': TE / TO if TO > 0 else None}
        res[key] = row
    json.dump(res, open(os.path.join(out, 'u0.json'), 'w'), indent=1)
    print(json.dumps({k: {t: round(r[f'tau{t}']['U0'], 3) for t in TAUS} for k, r in res.items()}, indent=1))
    return res


if __name__ == '__main__':
    cmd = sys.argv[1]
    if cmd == 'ladder':
        ladder(sys.argv[2], sys.argv[3])
    elif cmd == 'timing':
        timing(sys.argv[2])
    elif cmd == 'u0':
        u0(sys.argv[2], sys.argv[3])
