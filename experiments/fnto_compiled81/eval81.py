"""WL81 evaluation: directional / reflectance / highlight metrics, family strata and the frozen gates.

  python eval81.py stageC <data> <fitC> <out>                       # gate set, direct fits of control / candidate / unbounded
  python eval81.py stageD <data> <fitD> <out> <arm> [<arm> ...]     # test_iid / test_block / train_eval, compiler seeds + oracles

Reused from WL80C (imported, with closure80c.eval_closure replaced by the horizon-safe form via closure81.install()):
eval80c.lane_metrics, summarise, q, grid_dirs, highlight_stats.
"""
import json, math, os, sys, time
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import closure81 as K
C = K.install()
sys.path.insert(0, os.path.join(HERE, '..', 'fnto_compiled80c'))
import eval80c as EV  # noqa: E402
import energy81 as E  # noqa: E402
from data81 import normalise, P  # noqa: E402
import fit81 as F  # noqa: E402

dev_t = torch.device('cuda')
G = P['gates']
SD = P['seeds']
DESC = P['materials']['descriptor']


def compile_runtime(arm, z):
    """Raw z (torch) -> runtime params, compiled state [M,53] (numpy) and compile diagnostics (candidate)."""
    if arm == 'unbounded':
        p = C.decode(z, 'nonneg')
        return p, C.compiled_state(p).cpu().numpy(), None
    grids = E.compile_grids(dev_t) if arm == 'candidate' else None
    u = K.strat_u(128, 128, SD['compile_M'], dev_t) if arm == 'candidate' else None
    torch.cuda.synchronize(); t0 = time.time()
    p, md = E.compiled_params(arm, z, grids, u)
    torch.cuda.synchronize()
    st = C.compiled_state(p).cpu().numpy()
    diag = None
    if md is not None:
        M, dg = md
        diag = {'compile_seconds': time.time() - t0, 'n_materials': int(z.shape[0]),
                'M_median': float(M.median()), 'M_max': float(M.max()), 'frac_scaled': float((M > 1).float().mean()),
                'refine_margin_max': float(dg['refine'].max()), 'local_margin_max': float(dg['local'].max()),
                'argmax_at_mu0_frac': float((dg['argmax_mu_index'] == 0).float().mean()),
                'sum_lambda_runtime_max': float(p['lam'].sum(2).max())}
    return p, st, diag


def family_strata(el1, d):
    mm = el1.mean((1, 2))
    res = {}
    cols = {'alpha_i': d[:, 0], 'tau': d[:, 2], 'albedo_mean': d[:, 3:6].mean(1), 'r_mean': d[:, 6:9].mean(1), 'alpha_b': d[:, 9], 'eta': d[:, 1]}
    for nme, v in cols.items():
        qs = np.quantile(v, [1 / 3, 2 / 3])
        b = np.digitize(v, qs)
        res[nme] = [EV.q(mm[b == k], .5) for k in range(3)]
    return res


def highlight(arms_p, hl):
    """Closure-side highlight statistics on the stored reference grids (reference statistics precomputed in data81)."""
    A = P['angular']['highlight']
    n = A['grid']
    Gn = len(hl['mat'])
    chunk = max(1, (1 << 22) // (n * n))
    acc = {a: {'peak_err': [], 'logw': []} for a in arms_p}
    for c0 in range(0, Gn, chunk):
        cs = np.arange(c0, min(Gn, c0 + chunk))
        wi = hl['wi'][cs]
        wo, sx, sy = EV.grid_dirs(wi, hl['smax'][cs], n)
        ok = wo[..., 2] > 1e-4
        wo_c = wo.copy(); wo_c[..., 2] = np.maximum(wo_c[..., 2], 1e-4)
        wo_c /= np.linalg.norm(wo_c, axis=-1, keepdims=True)
        mm = torch.from_numpy(np.repeat(hl['mat'][cs], n * n)).to(dev_t)
        wiL = torch.from_numpy(np.repeat(wi, n * n, 0).astype(np.float32)).to(dev_t)
        woL = torch.from_numpy(wo_c.reshape(-1, 3).astype(np.float32)).to(dev_t)
        for a, p in arms_p.items():
            with torch.no_grad():
                pr = K.eval_closure(C.gather(p, mm), wiL, woL)
            vc = np.where(ok, pr.mean(1).cpu().numpy().reshape(len(cs), -1), -1.0)
            sc = EV.highlight_stats(vc, wo, sx, sy, n)
            dots = np.clip((hl['full_peak_wo'][cs] * sc['peak_wo']).sum(1), -1, 1)
            acc[a]['peak_err'].append(np.degrees(np.arccos(dots)))
            lw = 0.5 * np.log(np.maximum(sc['area'], 1) / np.maximum(hl['full_area'][cs], 1))
            lw = np.where(sc['touch'] | hl['full_touch'][cs], np.nan, lw)
            acc[a]['logw'].append(lw)
    out = {}
    pron = hl['pron']
    for a in arms_p:
        pk = np.concatenate(acc[a]['peak_err']); lw = np.concatenate(acc[a]['logw'])
        sel = pron & np.isfinite(lw)
        out[a] = {'n_cases': int(Gn), 'n_pronounced': int(pron.sum()), 'n_pronounced_resolved': int(sel.sum()),
                  'peak_err_deg_median': EV.q(pk[pron], .5), 'peak_err_deg_p90': EV.q(pk[pron], .9),
                  'abs_logw_median': EV.q(np.abs(lw[sel]), .5), 'abs_logw_p90': EV.q(np.abs(lw[sel]), .9),
                  'logw_median_signed': EV.q(lw[sel], .5)}
    return out


def representation_gate(S, H):
    g = G['representation']
    r = {'R1': S['E_L1']['median'] <= g['E_L1_median'] and S['E_L1']['p90'] <= g['E_L1_p90'],
         'R2': S['E_alb']['median'] <= g['E_alb_median'] and S['E_alb']['p95'] <= g['E_alb_p95'],
         'R3': H['peak_err_deg_median'] <= g['peak_err_deg_median'] and H['abs_logw_median'] <= math.log(g['width_ratio_max'])}
    r['clear_fail'] = S['E_L1']['median'] > g['clear_fail_E_L1_median'] or S['E_alb']['median'] > g['clear_fail_E_alb_median']
    r['PASS'] = bool(r['R1'] and r['R2'] and r['R3'])
    return r


def eval_arm(p, ev, d):
    el1, ealb, alb = EV.lane_metrics(p, ev)
    return EV.summarise(el1, ealb), family_strata(el1, d), el1, ealb


def stageC(data, fitC, out):
    os.makedirs(out, exist_ok=True)
    t0 = time.time()
    mats = np.load(os.path.join(data, 'materials.npz'))
    d = mats['gate']
    ev = dict(np.load(os.path.join(data, 'eval_gate.npz')))
    hl = dict(np.load(os.path.join(data, 'hl_gate.npz')))
    S, H, ST, diag, states = {}, {}, {}, {}, {}
    ps = {}
    for arm in ['control', 'candidate', 'unbounded']:
        z = torch.from_numpy(np.load(os.path.join(fitC, f'direct_{arm}_gate.npz'))['z']).to(dev_t)
        p, st, dg = compile_runtime(arm, z)
        ps[arm] = p; states[arm] = st; diag[arm] = dg
        S[arm], ST[arm], el1, ealb = eval_arm(p, ev, d)
        np.savez_compressed(os.path.join(out, f'lane_{arm}_gate.npz'), el1=el1, ealb=ealb)
        print(arm, json.dumps(S[arm]['E_L1']), json.dumps(S[arm]['E_alb']), flush=True)
    H = highlight(ps, hl)
    V = {arm: representation_gate(S[arm], H[arm]) for arm in ['control', 'candidate']}
    V['unbounded_diagnostic'] = representation_gate(S['unbounded'], H['unbounded'])
    for arm in ['control', 'candidate', 'unbounded']:
        np.save(os.path.join(out, f'states_{arm}_gate.npy'), states[arm])
    res = {'summary': S, 'highlight': H, 'strata': ST, 'compile': diag, 'representation_gate': V,
           'runtime_state_floats': {'control': 53, 'candidate': 53, 'note': 'identical runtime state; the candidate stores lambda / max(1, M)'},
           'noise_floor_gate_set': json.load(open(os.path.join(data, 'noise_gate.json'))),
           'highlight_noise_gate_set': json.load(open(os.path.join(data, 'hl_noise_gate.json'))),
           'seconds': time.time() - t0}
    json.dump(res, open(os.path.join(out, 'stageC_eval.json'), 'w'), indent=1)
    print(json.dumps(V, indent=1))


def stageD(data, fitD, out, arms):
    os.makedirs(out, exist_ok=True)
    t0 = time.time()
    mats = np.load(os.path.join(data, 'materials.npz'))
    seeds = FIT_SEEDS = P['fitting']['compiler']['seeds']
    S, H, ST, diag, states = {}, {}, {}, {}, {}
    for setname in ['test_iid', 'test_block', 'train_eval']:
        ev = dict(np.load(os.path.join(data, f'eval_{setname}.npz')))
        d = mats['train'][:len(ev['phis'])] if setname == 'train_eval' else mats[setname]
        x = torch.from_numpy(normalise(d)).to(dev_t)
        ps = {}
        for arm in arms:
            srcs = [f's{s}' for s in seeds]
            if setname != 'train_eval':
                srcs += ['oracleP'] + (['oracleU'] if arm == 'control' else [])
            for src in srcs:
                if src.startswith('s'):
                    net, _ = F.load_compiler(os.path.join(fitD, f'compiler_{arm}_{src}.pt'))
                    with torch.no_grad():
                        z = net(x)
                    a_eff = arm
                else:
                    o = np.load(os.path.join(fitD, f'oracle_{arm}_{setname}.npz'))
                    z = torch.from_numpy(o['zP' if src == 'oracleP' else 'zU']).to(dev_t)
                    a_eff = arm if src == 'oracleP' else 'unbounded'
                p, st, dg = compile_runtime(a_eff, z)
                key = f'{arm}_{src}'
                ps[key] = p
                states[f'{key}_{setname}'] = st
                if dg is not None:
                    diag[f'{key}_{setname}'] = dg
                S.setdefault(key, {})[setname], ST.setdefault(key, {})[setname], el1, ealb = eval_arm(p, ev, d)
                np.savez_compressed(os.path.join(out, f'lane_{key}_{setname}.npz'), el1=el1, ealb=ealb)
                print(setname, key, json.dumps(S[key][setname]['E_L1']), json.dumps(S[key][setname]['E_alb']), flush=True)
        if setname == 'test_iid':
            H = highlight(ps, dict(np.load(os.path.join(data, 'hl_test_iid.npz'))))
    for k, v in states.items():
        np.save(os.path.join(out, f'states_{k}.npy'), v)
    # material-count scaling (descriptive): primary arm = first listed arm, seed 8101, 750 / 1500 training materials
    scal = {}
    ev = dict(np.load(os.path.join(data, 'eval_test_iid.npz')))
    x = torch.from_numpy(normalise(mats['test_iid'])).to(dev_t)
    for n_tr in [750, 1500]:
        pth = os.path.join(fitD, f'compiler_{arms[0]}_s{seeds[0]}_n{n_tr}.pt')
        if os.path.exists(pth):
            net, _ = F.load_compiler(pth)
            with torch.no_grad():
                z = net(x)
            p, _, _ = compile_runtime(arms[0], z)
            scal[n_tr] = eval_arm(p, ev, mats['test_iid'])[0]
    scal[3000] = S[f'{arms[0]}_s{seeds[0]}']['test_iid']
    g = G['learned_rule']
    V = {}
    for arm in arms:
        o = S[f'{arm}_oracleP']['test_iid']
        lr = {}
        for s in seeds:
            key = f'{arm}_s{s}'
            ti, tb, tr = S[key]['test_iid'], S[key]['test_block'], S[key]['train_eval']
            hh = H[key]
            x = {'L1': ti['E_L1']['median'] <= g['iid_E_L1_median'] and
                       ti['E_L1']['median'] <= max(g['iid_vs_oracle_ratio'] * o['E_L1']['median'], o['E_L1']['median'] + g['iid_vs_oracle_abs']),
                 'L2': tb['E_L1']['median'] <= g['block_E_L1_median'] and tb['E_L1']['p90'] <= g['block_E_L1_p90'],
                 'L3': ti['E_L1']['median'] <= g['gen_gap_ratio'] * tr['E_L1']['median'],
                 'L4': ti['E_alb']['median'] <= g['iid_E_alb_median'] and tb['E_alb']['median'] <= g['block_E_alb_median'],
                 'L5': hh['peak_err_deg_median'] <= g['peak_err_deg_median'] and hh['abs_logw_median'] <= math.log(g['width_ratio_max']),
                 'L6': ti['band_median'] <= g['band_vs_neighbour_ratio'] * ti['band_neighbour_median'] and
                       ti['grazing_median'] <= g['grazing_vs_all_ratio'] * ti['E_L1']['median']}
            x['PASS'] = bool(all(x.values()))
            lr[key] = x
        lr['PASS'] = bool(all(lr[f'{arm}_s{s}']['PASS'] for s in seeds))
        V[arm] = lr
    res = {'summary': S, 'highlight': H, 'strata': ST, 'compile': diag, 'learned_rule_gate': V, 'material_count_scaling': scal,
           'noise_floor': {s: json.load(open(os.path.join(data, f'noise_{s}.json'))) for s in ['test_iid', 'test_block', 'train_eval']},
           'highlight_noise_test_iid': json.load(open(os.path.join(data, 'hl_noise_test_iid.json'))),
           'sizes': {'compiler_params': sum(p.numel() for p in F.Compiler('control', 128, 3).parameters()),
                     'compiled_state_floats': 53, 'compiled_state_independent_floats': 45, 'descriptor_floats': 10},
           'seconds': time.time() - t0}
    json.dump(res, open(os.path.join(out, 'stageD_eval.json'), 'w'), indent=1)
    print(json.dumps(V, indent=1))


if __name__ == '__main__':
    if sys.argv[1] == 'stageC':
        stageC(sys.argv[2], sys.argv[3], sys.argv[4])
    else:
        stageD(sys.argv[2], sys.argv[3], sys.argv[4], [a for a in sys.argv[5:] if not a.startswith('--')])
    os._exit(0)
