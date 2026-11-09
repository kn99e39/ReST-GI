"""WL80D: corrected sampler regression, energy-parameterisation comparison, contract, GPU cost.

  python run80d.py <wl80c_smoke_results_dir> <out_dir> <stage>
  stage in {tests, main}   (main = fit + compile + eval + passivity + contract + bench + decision)
"""
import json, math, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
C80 = os.path.join(HERE, '..', 'fnto_compiled80c')
sys.path.insert(0, HERE); sys.path.insert(0, C80)
import ref80c as R
mi = R.mi_init()
import drjit as dr
import torch
import closure80c as C
import closure80d_dr as CD
import energy80d as E
from data80c import P as P80C, train_style, eval_style
import eval80c as EV
import contract80c as CO
import bench80c as BE
import train80c as TR

CO.CD = CD   # corrected runtime closure for the reused WL80C contract / bench code
BE.CD = CD
P = json.load(open(os.path.join(HERE, 'protocol', 'closure80d.json')))
SD = P['seeds']
dev_t = torch.device('cuda')


def V3(a):
    a = np.ascontiguousarray(a, np.float32)
    return mi.Vector3f(mi.Float(a[:, 0]), mi.Float(a[:, 1]), mi.Float(a[:, 2]))


def P2(a):
    return mi.Point2f(mi.Float(np.ascontiguousarray(a[:, 0], np.float32)), mi.Float(np.ascontiguousarray(a[:, 1], np.float32)))


def np3(v):
    return np.stack([np.asarray(v[k]) for k in range(3)], 1).astype(np.float64)


def regen_sets(d):
    rng = np.random.default_rng(SD['wl80c_sets'])
    A = P80C['angular']
    bsdf, n = R.textured_principled(d)
    fit = train_style(bsdf, n, np.arange(len(d)), 64, 16, rng, A['heldout_incident_band_deg'])
    ev = eval_style(bsdf, n, np.arange(len(d)), rng, A['eval_theta_deg'], A['eval_n_phi'], A['eval_wo'])
    return fit, ev, rng


def adversarial_raw(seed, na=50):
    g = torch.Generator().manual_seed(seed)
    z = torch.randn(na, C.N_RAW, generator=g) * P80C['contract']['adversarial_sigma']
    z[: na // 4, :3 * C.N_GGX] = torch.sign(z[: na // 4, :3 * C.N_GGX]) * 12.0
    return z


def np_pdf(tab, wi, wo):
    st = tab.astype(np.float64)
    au, av, pr = st[:, 24:31], st[:, 31:38], st[:, 45:53]
    ci, co = wi[:, 2], wo[:, 2]
    h = wi + wo; h /= np.linalg.norm(h, axis=1, keepdims=True)
    pdf = pr[:, 0] * co / math.pi
    for k in range(7):
        den = h[:, 0] ** 2 / au[:, k] ** 2 + h[:, 1] ** 2 / av[:, k] ** 2 + h[:, 2] ** 2
        D = 1 / (math.pi * au[:, k] * av[:, k] * den ** 2)
        Li = 0.5 * (np.sqrt(1 + (au[:, k] ** 2 * wi[:, 0] ** 2 + av[:, k] ** 2 * wi[:, 1] ** 2) / ci ** 2) - 1)
        pdf += pr[:, k + 1] * D / (1 + Li) / (4 * ci)
    return pdf


# ---------------------------------------------------------------- tests
def tests(c80, out):
    res = {}
    rng = np.random.default_rng(SD['tests'])
    z = torch.randn(64, C.N_RAW, generator=torch.Generator().manual_seed(SD['tests'])) * 3.0
    tab = C.compiled_state(C.decode(z)).numpy()
    # T1 lane selection (per-lane and uniform)
    N = 1 << 20
    idx = rng.integers(0, 64, N); u1 = rng.random(N).astype(np.float32)
    pr = tab[:, 45:53].astype(np.float32)
    acc = pr[idx, 0].copy(); sel_np = np.zeros(N, np.int64)
    for k in range(1, 8):
        sel_np = np.where(u1 >= acc, k, sel_np); acc = (acc + pr[idx, k]).astype(np.float32)
    cl = CD.DrClosure(tab, mi.UInt32(idx.astype(np.uint32)))
    sel, au, av = cl._select_lobe(mi.Float(u1))
    sel = np.asarray(sel).astype(np.int64); au = np.asarray(au); av = np.asarray(av)
    au_np = np.where(sel_np == 0, 1.0, tab[idx, 24 + np.maximum(sel_np, 1) - 1])
    av_np = np.where(sel_np == 0, 1.0, tab[idx, 31 + np.maximum(sel_np, 1) - 1])
    mm = sel != sel_np
    a_ok = np.abs(au - au_np) <= 1e-6 * np.abs(au_np) + 1e-12
    b_ok = np.abs(av - av_np) <= 1e-6 * np.abs(av_np) + 1e-12
    cu = CD.DrClosure(tab[3]); su, auu, _ = cu._select_lobe(mi.Float(u1))
    su = np.asarray(su).astype(np.int64); auu = np.asarray(auu)
    acc = np.full(N, pr[3, 0], np.float32); su_np = np.zeros(N, np.int64)
    for k in range(1, 8):
        su_np = np.where(u1 >= acc, k, su_np); acc = (acc + pr[3, k]).astype(np.float32)
    au_u = np.where(su_np == 0, 1.0, tab[3, 24 + np.maximum(su_np, 1) - 1])
    res['T1'] = {'perlane_index_mismatch': int(mm.sum()), 'perlane_alpha_mismatch': int((~(a_ok & b_ok)).sum()),
                 'uniform_index_mismatch': int((su != su_np).sum()),
                 'uniform_alpha_mismatch': int((np.abs(auu - au_u) > 1e-6 * np.abs(au_u) + 1e-12).sum()),
                 'lanes': N, 'selected_lobe_histogram': np.bincount(sel, minlength=8).tolist()}
    res['T1']['PASS'] = all(res['T1'][k] == 0 for k in ['perlane_index_mismatch', 'perlane_alpha_mismatch', 'uniform_index_mismatch', 'uniform_alpha_mismatch'])
    # T2 eval / pdf consistency
    fitw = np.load(os.path.join(c80, 'fit.npz'))
    tab2 = np.concatenate([tab, fitw['states']]).astype(np.float32)
    raw = torch.cat([z, torch.from_numpy(fitw['zP']).float()])
    M = len(tab2)
    idx = rng.integers(0, M, N)
    wi = CO.unit(rng, N, True); wo = CO.unit(rng, N, True)
    cl = CD.DrClosure(tab2, mi.UInt32(idx.astype(np.uint32)))
    f_dr, p_dr = cl.eval_pdf(V3(wi), V3(wo))
    f_dr = np3(f_dr); p_dr = np.asarray(p_dr, np.float64)
    p = C.decode(raw.to(dev_t))
    with torch.no_grad():
        f_t = C.eval_closure(C.gather(p, torch.from_numpy(idx).to(dev_t)), torch.from_numpy(wi.astype(np.float32)).to(dev_t),
                             torch.from_numpy(wo.astype(np.float32)).to(dev_t)).double().cpu().numpy()
    p_np = np_pdf(tab2[idx], wi, wo)
    ok_f = np.abs(f_t) > 1e-6; ok_p = p_np > 1e-6
    rf = (np.abs(f_dr - f_t) / np.maximum(np.abs(f_t), 1e-30))[ok_f]
    rp = (np.abs(p_dr - p_np) / p_np)[ok_p]
    res['T2'] = {'eval_median_rel': float(np.median(rf)), 'eval_p999_rel': float(np.quantile(rf, .999)), 'eval_max_rel': float(rf.max()),
                 'pdf_median_rel': float(np.median(rp)), 'pdf_p999_rel': float(np.quantile(rp, .999)), 'pdf_max_rel': float(rp.max())}
    res['T2']['PASS'] = res['T2']['eval_median_rel'] <= 1e-6 and res['T2']['eval_p999_rel'] <= 1e-3 and \
        res['T2']['pdf_median_rel'] <= 1e-6 and res['T2']['pdf_p999_rel'] <= 1e-3
    # T3 sampler vs quadrature
    def st(lams, probs, alphas, f0=1.0):
        s = np.zeros(53, np.float32); s[24:38] = 0.3; s[38:45] = f0
        for r, l in lams.items():
            s[[r, 8 + r, 16 + r]] = l
        for r, q in probs.items():
            s[45 + r] = q
        for r, (a, b) in alphas.items():
            s[24 + r - 1] = a; s[31 + r - 1] = b
        return s
    cases = {'lambert': st({0: 1}, {0: 1}, {}), 'ggx_0.3': st({1: 1}, {1: 1}, {1: (.3, .3)}),
             'ggx_0.3x0.1': st({1: 1}, {1: 1}, {1: (.3, .1)}), 'ggx_0.05': st({1: 1}, {1: 1}, {1: (.05, .05)}),
             'lamb+ggx': st({0: .5, 1: .5}, {0: .5, 1: .5}, {1: (.3, .3)}),
             'ggx0.05+ggx0.3': st({2: .5, 5: .5}, {2: .5, 5: .5}, {2: (.05, .05), 5: (.3, .3)}),
             'uniform_probs_one_lobe': st({7: 1}, {r: 1 / 8 for r in range(8)}, {7: (.3, .3)})}
    t3 = {}
    allok = True
    for name, s in cases.items():
        for th in [10, 45, 75]:
            c1 = CD.DrClosure(s); Ns = 1 << 20
            wi1 = R.sph(np.radians(th), 0.7); wiN = np.repeat(wi1[None], Ns, 0)
            wo_, p_, w_, v_ = c1.sample(V3(wiN), mi.Float(rng.random(Ns).astype(np.float32)), P2(rng.random((Ns, 2))))
            w0 = np.asarray(w_[0], np.float64); v_ = np.asarray(v_).astype(np.float64)
            Mq = 1 << 22; wq = R.uniform_dirs(rng.random((Mq, 2)))
            fq, pq = c1.eval_pdf(V3(np.repeat(wi1[None], Mq, 0)), V3(wq))
            gq = np.asarray(fq[0], np.float64) * 2 * np.pi; hq = np.asarray(pq, np.float64) * 2 * np.pi
            se1 = math.sqrt(w0.var() / Ns + gq.var() / Mq); se2 = math.sqrt(v_.var() / Ns + hq.var() / Mq)
            ok = abs(w0.mean() - gq.mean()) <= 4 * se1 + 2e-3 and abs(v_.mean() - hq.mean()) <= 4 * se2 + 2e-3
            allok &= ok
            t3[f'{name}|{th}'] = {'albedo_sampler': float(w0.mean()), 'albedo_quad': float(gq.mean()),
                                  'valid_frac': float(v_.mean()), 'int_pdf': float(hq.mean()), 'ok': bool(ok)}
    res['T3'] = {'cases': t3, 'PASS': bool(allok)}
    # T4 WL80C compatibility
    d = np.load(os.path.join(c80, 'materials.npz'))['test_iid']
    fit, ev, _ = regen_sets(d)
    pP = C.decode(torch.from_numpy(fitw['zP']).to(dev_t), 'passive')
    el1, ealb, _ = EV.lane_metrics(pP, ev)
    s = EV.summarise(el1, ealb)
    ref = json.load(open(os.path.join(c80, 'smoke.json')))['gate_A']['oracleP']
    pairs = [(s['E_L1'][k], ref['E_L1'][k]) for k in ['median', 'p90', 'max']] + [(s['E_alb'][k], ref['E_alb'][k]) for k in ['median', 'p95', 'max']]
    rel = max(abs(a - b) / abs(b) for a, b in pairs)
    res['T4'] = {'max_rel_vs_wl80c': rel, 'reproduced': {'E_L1': s['E_L1'], 'E_alb': s['E_alb']}, 'PASS': rel <= 1e-5}
    res['PASS'] = all(res[k]['PASS'] for k in ['T1', 'T2', 'T3', 'T4'])
    json.dump(res, open(os.path.join(out, 'tests.json'), 'w'), indent=1)
    print(json.dumps({k: res[k]['PASS'] for k in ['T1', 'T2', 'T3', 'T4']}), json.dumps(res['T1']), json.dumps(res['T2']), json.dumps(res['T4']['max_rel_vs_wl80c']), flush=True)
    return res['PASS']


# ---------------------------------------------------------------- candidate
def fit_grids():
    B = P['part_B_energy']
    wf = E.wi_grid([0, 20, 40, 55, 65, 72, 78, 82, 85, 87, 88.5, 89.5], [0, 30, 60, 90], dev_t)
    uf = E.strat_u(16, 32, SD['fit_M_u'], dev_t)
    th = np.concatenate([np.linspace(0, 80, 33), np.linspace(80.5, 89.9, 31)])
    wc = E.wi_grid(th, np.linspace(0, 90, 16), dev_t)
    uc = E.strat_u(128, 256, SD['compile_M_u'], dev_t)
    return wf, uf, wc, uc


def init_candidate(n):
    z = C.init_bias().to(dev_t)[None].repeat(n, 1)
    lg = z[:, 3 * C.N_GGX:].view(n, 3, C.R_LOBES + 1)
    lg[...] = math.log(math.expm1(1 / 9))
    return z


def fit_candidate(fit, n_mat, wf, uf):
    O = P80C['oracle']
    z = torch.nn.Parameter(init_candidate(n_mat))
    opt = torch.optim.Adam([z], lr=O['lr'])
    idx = TR.per_material_index(fit['m'], n_mat)
    lanes = idx.reshape(-1)
    mid = torch.arange(n_mat, device=dev_t).repeat_interleave(idx.shape[1])
    wi, wo, f = fit['wi'][lanes], fit['wo'][lanes], fit['f'][lanes]
    steps = O['steps']
    log = []
    for step in range(steps):
        for pg in opt.param_groups:
            pg['lr'] = TR.lr_at(step, steps, O['lr'], O['lr_end'], 50)
        p = E.decode_candidate(z)
        Mf = E.fit_M(p, wf, uf)
        q = E.scaled(p, Mf)
        pred = C.eval_closure(C.gather(q, mid), wi, wo)
        per = ((torch.log1p(pred) - torch.log1p(f)) ** 2).mean(1).view(n_mat, -1).mean(1)
        opt.zero_grad(set_to_none=True)
        per.sum().backward()
        opt.step()
        if step % 500 == 0 or step == steps - 1:
            log.append({'step': step, 'loss_median': float(per.median()), 'M_fit_max': float(Mf.max())})
            print(json.dumps(log[-1]), flush=True)
    return z.detach(), per.detach(), log


def compile_candidate(z, wc, uc):
    p = E.decode_candidate(z)
    B = z.shape[0]
    chunk = max(1, int(2.5e7 // (B * 7 * uc.shape[0])))
    torch.cuda.synchronize(); t = time.perf_counter()
    M, idx, amax, se = E.compile_M(p, wc, uc, wi_chunk=chunk)
    torch.cuda.synchronize(); dt = time.perf_counter() - t
    q = E.scaled(p, M)
    return q, M, idx, amax, se, dt


def passivity_check(tab, n_wi, ns, rng):
    """Corrected Dr.Jit sampler albedo at random wi; returns violations, max estimate."""
    M = len(tab)
    viol, amax, cases = 0, 0.0, 0
    per_chunk = max(1, (1 << 24) // ns)
    G_all = M * n_wi
    for g0 in range(0, G_all, per_chunk):
        g = np.arange(g0, min(G_all, g0 + per_chunk))
        idx = np.repeat(g // n_wi, ns)
        cmin = math.cos(math.radians(89.95))
        ct = rng.uniform(cmin, 1.0, len(g)); ph = rng.uniform(0, 2 * np.pi, len(g))
        wi = np.repeat(R.sph(np.arccos(ct), ph), ns, 0)
        cl = CD.DrClosure(tab, mi.UInt32(idx.astype(np.uint32)))
        _, _, w, _ = cl.sample(V3(wi), mi.Float(rng.random(len(idx)).astype(np.float32)), P2(rng.random((len(idx), 2))))
        w = np3(w).reshape(len(g), ns, 3)
        a = w.mean(1); se = w.std(1) / math.sqrt(ns)
        v = (a - 1 > np.maximum(4 * se, 1e-3))
        viol += int(v.any(1).sum()); amax = max(amax, float(a.max())); cases += len(g)
    return {'cases': cases, 'violations': viol, 'max_albedo_estimate': amax, 'samples_per_case': ns}


def main_stage(c80, out):
    t0 = time.time()
    timeline = {}
    d = np.load(os.path.join(c80, 'materials.npz'))['test_iid']
    fitw = np.load(os.path.join(c80, 'fit.npz'))
    np.savez(os.path.join(out, 'materials.npz'), test_iid=d, test_block=d)
    fit, ev, rng = regen_sets(d)
    ft = {k: torch.from_numpy(fit[k]).to(dev_t) for k in ['m', 'wi', 'wo', 'f']}
    wf, uf, wc, uc = fit_grids()
    n = len(d)
    # ---- fit + compile candidate
    t = time.time()
    zC, lossC, flog = fit_candidate(ft, n, wf, uf)
    timeline['fit_s'] = time.time() - t
    qC, MC, Midx, amaxC, seC, t_comp = compile_candidate(zC, wc, uc)
    with torch.no_grad():
        Mfit = E.fit_M(E.decode_candidate(zC), wf, uf)
    pP = C.decode(torch.from_numpy(fitw['zP']).to(dev_t), 'passive')
    tabP = C.compiled_state(pP).cpu().numpy()
    tabC = C.compiled_state(qC).detach().cpu().numpy()
    zadv = adversarial_raw(SD['adversarial_candidate']).to(dev_t)
    qA, MA, _, _, _, t_compA = compile_candidate(zadv, wc, uc)
    tabA = C.compiled_state(qA).detach().cpu().numpy()
    tabAP = C.compiled_state(C.decode(adversarial_raw(P80C['seeds']['adversarial']))).numpy()
    np.savez(os.path.join(out, 'candidate.npz'), zC=zC.cpu().numpy(), M=MC.cpu().numpy(), M_fit=Mfit.cpu().numpy(),
             a_max=amaxC.cpu().numpy(), se_at_max=seC.cpu().numpy(), M_argmax=Midx.cpu().numpy(), states=tabC,
             adv_states=tabA, adv_M=MA.cpu().numpy(), fit_loss=lossC.cpu().numpy())
    res = {'fit_log': flog, 'compile': {'seconds_8_materials': t_comp, 'seconds_50_adversarial': t_compA,
                                         'M': MC.cpu().numpy().tolist(), 'M_fit_estimate': Mfit.cpu().numpy().tolist(),
                                         'sum_lambda_scaled': qC['lam'].sum(2).detach().cpu().numpy().tolist(),
                                         'sum_lambda_control': pP['lam'].sum(2).cpu().numpy().tolist(),
                                         'fit_loss_candidate_median': float(lossC.median()),
                                         'fit_loss_control_median_wl80c': json.load(open(os.path.join(c80, 'smoke.json')))['gate_A']['gate_A']['fit_loss_P_median']}}
    json.dump(res, open(os.path.join(out, 'main_partial.json'), 'w'), indent=1)
    # ---- evaluation
    t = time.time()
    S = {}
    for arm, p in [('control', pP), ('candidate', qC)]:
        el1, ealb, _ = EV.lane_metrics(p, ev)
        S[arm] = EV.summarise(el1, ealb)
        S[arm]['per_material_E_L1_mean'] = el1.mean((1, 2)).tolist()
        S[arm]['per_material_E_alb_max'] = ealb.max((1, 2)).tolist()
        np.savez_compressed(os.path.join(out, f'lane_{arm}.npz'), el1=el1, ealb=ealb)
        print(arm, json.dumps(S[arm]['E_L1']), json.dumps(S[arm]['E_alb']), flush=True)
    H = EV.highlight_eval({'control': pP, 'candidate': qC}, 'wl80d', d, ev, rng)
    g = P80C['gates']['representation']
    gates = {}
    for arm in ['control', 'candidate']:
        o, h = S[arm], H[arm]
        x = {'R1': o['E_L1']['median'] <= g['E_L1_median'] and o['E_L1']['p90'] <= g['E_L1_p90'],
             'R2': o['E_alb']['median'] <= g['E_alb_median'] and o['E_alb']['p95'] <= g['E_alb_p95'],
             'R3': h['peak_err_deg_median'] <= g['peak_err_deg_median'] and h['abs_logw_median'] <= math.log(g['width_ratio_max'])}
        x['PASS'] = x['R1'] and x['R2'] and x['R3']
        gates[arm] = x
    gates['materially_addresses'] = gates['candidate']['R2'] and S['candidate']['E_L1']['median'] <= S['control']['E_L1']['median']
    res.update({'eval': S, 'highlight': H, 'representation_gates': gates})
    timeline['eval_s'] = time.time() - t
    json.dump(res, open(os.path.join(out, 'main_partial.json'), 'w'), indent=1)
    print('REPR', json.dumps(gates), flush=True)
    # ---- independent passivity check
    t = time.time()
    prng = np.random.default_rng(SD['passivity_check'])
    res['passivity_independent'] = {'candidate': passivity_check(tabC, 1024, 65536, prng),
                                    'adversarial_candidate': passivity_check(tabA, 1024, 65536, prng),
                                    'control': passivity_check(tabP, 1024, 65536, prng)}
    timeline['passivity_s'] = time.time() - t
    print('PASSIVITY', json.dumps(res['passivity_independent']), flush=True)
    # ---- contract (WL80C checks, corrected sampler)
    t = time.time()
    CO.states_for = lambda _f, _m: {'control': tabP, 'adversarial_control': tabAP, 'candidate': tabC, 'adversarial_candidate': tabA}
    orig_su = CO.sampling_utility
    s0 = P80C['seeds']['train'][0]

    def su(tabs, mats, rng_):
        a = orig_su({f's{s0}_test_iid': tabs['control']}, mats, rng_)
        b = orig_su({f's{s0}_test_iid': tabs['candidate']}, mats, rng_)
        return {'control': a, 'candidate': b, 'PASS': a['PASS'] and b['PASS']}
    CO.sampling_utility = su
    CO.run(None, out, out)
    st3 = json.load(open(os.path.join(out, 'stage3.json')))
    checks = ['C1', 'C2', 'C3', 'C4', 'C5a', 'C6', 'C7']
    per = {k: {c: st3[k][c]['PASS'] for c in checks} for k in ['control', 'adversarial_control', 'candidate', 'adversarial_candidate']}
    res['contract'] = {'per_set': per, 'C5b_all_sets': st3['C5b'], 'sampling_utility': st3['sampling_utility']}
    timeline['contract_s'] = time.time() - t
    json.dump(res, open(os.path.join(out, 'main_partial.json'), 'w'), indent=1)
    print('CONTRACT', json.dumps(per), json.dumps(st3['C5b']), flush=True)
    # ---- GPU benchmark
    t = time.time()
    res['bench'] = bench(d, tabP, tabC, zC)
    timeline['bench_s'] = time.time() - t
    # ---- decision
    samp = all(per['control'].values()) and all(per['adversarial_control'].values()) and st3['C5b']['PASS']
    tests = json.load(open(os.path.join(out, 'tests.json')))
    samp = samp and tests['PASS']
    pi = res['passivity_independent']
    phys = all(per['candidate'].values()) and all(per['adversarial_candidate'].values()) and st3['C5b']['PASS'] and \
        pi['candidate']['violations'] == 0 and pi['adversarial_candidate']['violations'] == 0
    b = res['bench']['rules']
    dec = {'SAMPLER_CORRECTNESS': samp, 'REPRESENTATION_FEASIBILITY_candidate': gates['candidate']['PASS'],
           'REPRESENTATION_control': gates['control']['PASS'], 'materially_addresses': gates['materially_addresses'],
           'PHYSICAL_CONTRACT_candidate_empirical': phys, 'GPU_advantage': b['advantage'], 'GPU_parity_feasibility': b['parity'],
           'expensive_reference': 'UNRESOLVED (no readily available reference)', 'LEARNED_RULE': 'UNTESTED'}
    dec['justifies_learned_rule_training'] = samp and gates['candidate']['PASS'] and phys and b['parity']
    res['decision'] = dec
    res['timeline'] = timeline
    res['seconds'] = time.time() - t0
    json.dump(res, open(os.path.join(out, 'main.json'), 'w'), indent=1)
    print('DECISION', json.dumps(dec), flush=True)


def bench(d, tabP, tabC, zC):
    dr.set_flag(dr.JitFlag.KernelHistory, True)
    G = P['part_C_gpu']
    env = {'apps_before': BE.gpu_apps(), 'gpu': torch.cuda.get_device_name(0)}
    rng = np.random.default_rng(SD['bench'])
    rows = []
    sizes = {'io': [65536, 4194304, 16777216], 'fused16': [1048576, 4194304]}
    ops = ['eval_pdf', 'eval', 'pdf', 'sample']
    for k in range(4):
        arms = {'REF': BE.RefArm(R.uniform_principled(d[k])), 'CTRL': BE.CmpArm(tabP[k]), 'CAND': BE.CmpArm(tabC[k])}
        if k == 0:
            arms['SIMPLE'] = BE.RefArm(mi.load_dict({'type': 'roughconductor', 'distribution': 'ggx', 'alpha_u': 0.2, 'alpha_v': 0.05}))
        for mode in ['io', 'fused16']:
            for n in sizes[mode]:
                inputs = None
                if mode == 'io':
                    wi = BE.V3(BE.rand_dirs(rng, n)); wo = BE.V3(BE.rand_dirs(rng, n))
                    u1 = mi.Float(rng.random(n).astype(np.float32))
                    u2 = mi.Point2f(mi.Float(rng.random(n).astype(np.float32)), mi.Float(rng.random(n).astype(np.float32)))
                    dr.eval(wi, wo, u1, u2); dr.sync_thread()
                    inputs = (wi, wo, u1, u2)
                for op in ops:
                    for an, arm in arms.items():
                        r = BE.time_op(arm, op, mode, n, inputs, 10, 2)
                        r.update({'material': k, 'arm': an, 'mode': mode, 'n': n, 'op': op,
                                  'queries': n * (16 if mode == 'fused16' else 1)})
                        rows.append(r)
                print('bench', k, mode, n, flush=True)
    env['apps_after'] = BE.gpu_apps()
    agg = {}
    for arm in ['CTRL', 'CAND']:
        for mode in sizes:
            for n in sizes[mode]:
                for op in ops:
                    sp = []
                    for k in range(4):
                        gq = {r['arm']: r for r in rows if r['material'] == k and r['mode'] == mode and r['op'] == op and r['n'] == n}
                        sp.append(gq['REF']['gpu_ms_median'] / gq[arm]['gpu_ms_median'])
                    q = n * (16 if mode == 'fused16' else 1)
                    rr = [r for r in rows if r['mode'] == mode and r['op'] == op and r['n'] == n]
                    agg[f'{arm}|{mode}|{op}|{n}'] = {
                        'speedup_REF_over_arm_median': float(np.median(sp)), 'min': float(min(sp)), 'max': float(max(sp)),
                        'ref_ns_per_query': float(np.median([r['gpu_ms_median'] for r in rr if r['arm'] == 'REF']) * 1e6 / q),
                        'arm_ns_per_query': float(np.median([r['gpu_ms_median'] for r in rr if r['arm'] == arm]) * 1e6 / q),
                        'arm_kernels': int(np.median([r['kernels'] for r in rr if r['arm'] == arm]))}
    adv = all(agg[f'{a}|{m}|{op}|{max(sizes[m])}']['speedup_REF_over_arm_median'] >= 1.10
              for a in ['CAND'] for m in sizes for op in ['eval_pdf', 'sample'])
    par = all(1.0 / agg[f'CAND|fused16|{op}|{max(sizes["fused16"])}']['speedup_REF_over_arm_median'] <= 1.25 for op in ['eval_pdf', 'sample'])
    contention = env['apps_before'] != '' or env['apps_after'] != ''
    # compile / memory costs
    cc = {}
    wf, uf, wc, uc = fit_grids()
    p = E.decode_candidate(zC)
    for name, sl in [('single', slice(0, 1)), ('batch8', slice(0, 8))]:
        ts = []
        for i in range(6):
            q = {k: v[sl] for k, v in p.items()}
            torch.cuda.synchronize(); t = time.perf_counter()
            E.compile_M(q, wc, uc, wi_chunk=max(1, int(2.5e7 // (q['au'].shape[0] * 7 * uc.shape[0]))))
            torch.cuda.synchronize(); ts.append(time.perf_counter() - t)
        cc[f'M_compile_{name}_s_median'] = float(np.median(ts[1:]))
    net = C.Compiler(128, 3).to(dev_t).eval()
    for name, nb in [('single', 1), ('batch1024', 1024)]:
        x = torch.zeros(nb, 9, device=dev_t)
        ts = []
        for i in range(60):
            torch.cuda.synchronize(); t = time.perf_counter()
            with torch.no_grad():
                st = C.compiled_state(C.decode(net(x)))
            torch.cuda.synchronize(); ts.append(time.perf_counter() - t)
        cc[f'mlp_forward_{name}_ms_median'] = float(np.median(ts[10:]) * 1e3)
    ts = []
    for i in range(30):
        t = time.perf_counter(); cl = CD.DrClosure(tabC[0]); dr.eval(cl.au[0]); dr.sync_thread(); ts.append(time.perf_counter() - t)
    cc['runtime_bind_ms_median'] = float(np.median(ts[5:]) * 1e3)
    mem = {'compiled_state_floats': 53, 'compiled_state_bytes': 212, 'independent_floats': 45,
           'runtime_table_floats': 88, 'runtime_table_bytes': 352, 'candidate_extra_state': 'none at runtime (M folded into lambda); M (3 floats) kept for provenance',
           'reference_principled_param_floats': 13, 'compiler_params': sum(x.numel() for x in net.parameters())}
    json.dump(rows, open(os.path.join(sys.argv[2], 'bench_rows.json'), 'w'))
    return {'env': env, 'contention': contention, 'agg': agg, 'compile_cost': cc, 'memory': mem,
            'rules': {'advantage': adv and not contention, 'parity': par and not contention}}


if __name__ == '__main__':
    c80, out, stage = sys.argv[1:4]
    os.makedirs(out, exist_ok=True)
    if stage == 'tests':
        ok = tests(c80, out)
        sys.stdout.flush()
        os._exit(0 if ok else 3)
    main_stage(c80, out)
    sys.stdout.flush()
    os._exit(0)
