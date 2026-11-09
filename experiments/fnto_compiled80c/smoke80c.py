"""WL80C SCOPE-CORRECTED minimal smoke (user correction 2026-11-09).

Gate A  representability: direct per-material R=8 fit (oracle-P, upper-bound diagnostic,
        NOT a learned rule) on the first 8 reference-passive materials of the existing
        Stage-0a survey (seed 80300; no selection by error).
Contract: existing contract80c checks on the fitted closures + adversarial closures.
Gate B  GPU: Mitsuba principled (C++) vs compiled closure, same Dr.Jit CUDA path.
Gate C  (learned compiler) NOT RUN: would need dataset generation + training.

  python smoke80c.py <survey_npz> <out_dir>
"""
import json, os, sys, time
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(__file__))
import ref80c as R
mi = R.mi_init()
import drjit as dr
import closure80c as C
from data80c import P, train_style, eval_style
import train80c as TR
import eval80c as EV
import contract80c as CO
import bench80c as BE

S = {"n_materials": 8, "fit_wi": 64, "fit_wo": 16, "oracle_steps": 3000, "seed": 80900,
     "bench_materials": 4, "sizes_io": [65536, 4194304, 16777216], "sizes_fused": [1048576, 4194304],
     "reps": 10, "warmup": 2, "n_adversarial": 50}
dev_t = torch.device('cuda')


def main():
    surv, out = sys.argv[1:3]
    os.makedirs(out, exist_ok=True)
    t0 = time.time(); log = {'settings': S}
    z = np.load(surv)
    passive = z['alb'].max((1, 2)) <= 1.0
    d = z['desc'][passive][:S['n_materials']].astype(np.float32)
    idx_survey = np.nonzero(passive)[0][:S['n_materials']]
    log['materials'] = {'survey_index': idx_survey.tolist(), 'desc': d.tolist()}
    np.savez(os.path.join(out, 'materials.npz'), test_iid=d, test_block=d)
    rng = np.random.default_rng(S['seed'])
    A = P['angular']
    bsdf, n = R.textured_principled(d)
    fit = train_style(bsdf, n, np.arange(len(d)), S['fit_wi'], S['fit_wo'], rng, A['heldout_incident_band_deg'])
    ev = eval_style(bsdf, n, np.arange(len(d)), rng, A['eval_theta_deg'], A['eval_n_phi'], A['eval_wo'])
    # ---- Gate A: direct fit (generic init only; no compiler exists)
    ft = {k: torch.from_numpy(fit[k]).to(dev_t) for k in ['m', 'wi', 'wo', 'f']}
    O = P['oracle']
    z0 = C.init_bias().to(dev_t)[None].repeat(len(d), 1)
    zP, lP = TR.fit_free(z0, ft, len(d), S['oracle_steps'], O['lr'], O['lr_end'], 'passive')
    pP = C.decode(zP, 'passive')
    lg = zP.clone(); v = lg[:, 3 * C.N_GGX:].view(len(d), 3, C.R_LOBES + 1)
    v[..., :C.R_LOBES] = torch.log(torch.expm1(pP['lam'].clamp_min(1e-6)))
    zU, lU = TR.fit_free(lg, ft, len(d), S['oracle_steps'], O['lr'], O['lr_end'], 'nonneg')
    tA = time.time()
    res = {}
    for arm, p in [('oracleP', pP), ('oracleU_diag', C.decode(zU, 'nonneg'))]:
        el1, ealb, _ = EV.lane_metrics(p, ev)
        res[arm] = EV.summarise(el1, ealb)
        res[arm]['per_material_E_L1_mean'] = el1.mean((1, 2)).tolist()
        res[arm]['per_material_by_theta_median'] = np.median(el1, 2).tolist()
    res['highlight'] = EV.highlight_eval({'oracleP': pP}, 'smoke', d, ev, rng)
    g = P['gates']['representation']; o = res['oracleP']; h = res['highlight']['oracleP']
    import math
    res['gate_A'] = {'R1': o['E_L1']['median'] <= g['E_L1_median'] and o['E_L1']['p90'] <= g['E_L1_p90'],
                     'R2': o['E_alb']['median'] <= g['E_alb_median'] and o['E_alb']['p95'] <= g['E_alb_p95'],
                     'R3': (h['peak_err_deg_median'] <= g['peak_err_deg_median']) and (h['abs_logw_median'] <= math.log(g['width_ratio_max'])),
                     'clear_fail': o['E_L1']['median'] > g['clear_fail_E_L1_median'] or o['E_alb']['median'] > g['clear_fail_E_alb_median'],
                     'fit_loss_P_median': float(lP.median()), 'fit_loss_U_median': float(lU.median()),
                     'fit_seconds': tA - t0}
    res['gate_A']['PASS'] = res['gate_A']['R1'] and res['gate_A']['R2'] and res['gate_A']['R3']
    log['gate_A'] = res
    statesP = C.compiled_state(pP).detach().cpu().numpy()
    np.savez(os.path.join(out, 'fit.npz'), zP=zP.cpu().numpy(), zU=zU.cpu().numpy(), states=statesP)
    json.dump(log, open(os.path.join(out, 'smoke.json'), 'w'), indent=1)
    print('GATE A', json.dumps(res['gate_A']), json.dumps(o['E_L1']), json.dumps(o['E_alb']), flush=True)
    if res['gate_A']['clear_fail']:
        log['stop'] = 'representation clear fail'
        json.dump(log, open(os.path.join(out, 'smoke.json'), 'w'), indent=1)
        return
    # ---- physical contract (existing checks) on fitted + adversarial closures
    def states_for(_fit_dir, _mats):
        gg = torch.Generator().manual_seed(P['seeds']['adversarial'])
        na = S['n_adversarial']
        za = torch.randn(na, C.N_RAW, generator=gg) * P['contract']['adversarial_sigma']
        za[: na // 4, :3 * C.N_GGX] = torch.sign(za[: na // 4, :3 * C.N_GGX]) * 12.0
        return {'oracleP_smoke': statesP, 'adversarial': C.compiled_state(C.decode(za)).numpy()}
    CO.states_for = states_for
    orig_su = CO.sampling_utility
    CO.sampling_utility = lambda tabs, mats, rng_: orig_su({f"s{P['seeds']['train'][0]}_test_iid": tabs['oracleP_smoke']}, mats, rng_)
    CO.run(None, out, out)
    st3 = json.load(open(os.path.join(out, 'stage3.json')))
    log['contract'] = st3['verdict']
    json.dump(log, open(os.path.join(out, 'smoke.json'), 'w'), indent=1)
    # ---- Gate B: GPU cost
    dr.set_flag(dr.JitFlag.KernelHistory, True)
    env = {'apps_before': BE.gpu_apps(), 'gpu': torch.cuda.get_device_name(0)}
    rows = []
    brng = np.random.default_rng(P['seeds']['bench'])
    for k in range(S['bench_materials']):
        arms = {'REF': BE.RefArm(R.uniform_principled(d[k])), 'CMP': BE.CmpArm(statesP[k])}
        if k == 0:
            arms['SIMPLE'] = BE.RefArm(mi.load_dict({'type': 'roughconductor', 'distribution': 'ggx', 'alpha_u': 0.2, 'alpha_v': 0.05}))
        for mode, sizes in [('io', S['sizes_io']), ('fused16', S['sizes_fused'])]:
            for nn in sizes:
                inputs = None
                if mode == 'io':
                    wi = BE.V3(BE.rand_dirs(brng, nn)); wo = BE.V3(BE.rand_dirs(brng, nn))
                    u1 = mi.Float(brng.random(nn).astype(np.float32))
                    u2 = mi.Point2f(mi.Float(brng.random(nn).astype(np.float32)), mi.Float(brng.random(nn).astype(np.float32)))
                    dr.eval(wi, wo, u1, u2); dr.sync_thread()
                    inputs = (wi, wo, u1, u2)
                for op in ['eval_pdf', 'eval', 'pdf', 'sample']:
                    for an, arm in arms.items():
                        r = BE.time_op(arm, op, mode, nn, inputs, S['reps'], S['warmup'])
                        r.update({'material': k, 'arm': an, 'mode': mode, 'n': nn, 'op': op,
                                  'queries': nn * (16 if mode == 'fused16' else 1)})
                        rows.append(r)
                print('bench', k, mode, nn, flush=True)
    env['apps_after'] = BE.gpu_apps()
    agg = {}
    for mode, sizes in [('io', S['sizes_io']), ('fused16', S['sizes_fused'])]:
        for op in ['eval_pdf', 'eval', 'pdf', 'sample']:
            for nn in sizes:
                sp, rn, cn = [], [], []
                for k in range(S['bench_materials']):
                    gq = {r['arm']: r for r in rows if r['material'] == k and r['mode'] == mode and r['op'] == op and r['n'] == nn}
                    sp.append(gq['REF']['gpu_ms_median'] / max(gq['CMP']['gpu_ms_median'], 1e-9))
                    q = gq['REF']['queries']
                    rn.append(gq['REF']['gpu_ms_median'] * 1e6 / q); cn.append(gq['CMP']['gpu_ms_median'] * 1e6 / q)
                agg[f'{mode}|{op}|{nn}'] = {'speedup_REF_over_CMP_median': float(np.median(sp)), 'speedup_min': float(min(sp)),
                                            'speedup_max': float(max(sp)), 'ref_ns_per_query': float(np.median(rn)),
                                            'cmp_ns_per_query': float(np.median(cn))}
    log['gate_B'] = {'env': env, 'agg': agg, 'contention': env['apps_before'] != '' or env['apps_after'] != ''}
    json.dump(rows, open(os.path.join(out, 'bench_rows.json'), 'w'))
    log['seconds'] = time.time() - t0
    json.dump(log, open(os.path.join(out, 'smoke.json'), 'w'), indent=1)
    print('GATE B', json.dumps(agg, indent=0), flush=True)


if __name__ == '__main__':
    main()
    os._exit(0)
