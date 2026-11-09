"""WL82 evaluation: energy / angular-shape decomposition, oracle albedo correction, highlight geometry, strata.

  python eval82.py <data82> <w81data> <w81C> <fitdir> <set> <out> [--dev]
Evaluates every fit file fit_<set>_*.npz in <fitdir> (and, for the gate set, the WL81 frozen Stage C direct fits W81_control /
W81_candidate / W81_unbounded from <w81C>/fit) on the set's evaluation lanes (WL81 eval_<set>.npz or eval82_real.npz).

Per (material, wi) group g, with importance weights w = use / q and a_c = sum_k w f_c on the evaluation samples:
  E_L1    = sum_k w |f_cl - f_ref|_1 / sum_c a_ref,c                                    (WL81 / WL80C definition)
  E_alb   = sum_c |a_cl,c - a_ref,c| / sum_c a_ref,c                                    (WL81 / WL80C definition)
  E_shape = sum_c a_ref,c TV_c / sum_c a_ref,c,  TV_c = sum_k w |f_cl,c / a_cl,c - f_ref,c / a_ref,c|   (in [0, 2]; a_cl,c = 0 -> the
            closure's normalised term is 0, i.e. TV_c = 1, the relative L1 of predicting nothing; counted)
  Oracle (diagnostic, uses reference albedo at evaluation time): f_or = f_cl A_ref^ind / A_cl^ind with A^ind from the INDEPENDENT
            albedo set (alb82_<set>.npz: fresh wo samples and fresh walks; A_cl^ind on the same independent samples).
  Identity check: the same-sample oracle (ratio a_ref / a_cl from the evaluation samples) has E_L1 == E_shape and E_alb == 0 exactly.
"""
import json, math, os, sys, time
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import closure82 as CL  # noqa: E402
sys.path.insert(0, os.path.join(HERE, '..', 'fnto_compiled81'))
sys.path.insert(0, os.path.join(HERE, '..', 'fnto_compiled80c'))
import eval80c as EV  # noqa: E402
import eval81 as EV81  # noqa: E402
import data81 as DA  # noqa: E402

P82 = json.load(open(os.path.join(HERE, 'protocol', 'compiled82.json')))
dev_t = torch.device('cuda')
TH = np.array(DA.A['eval_theta_deg'])
DEV = '--dev' in sys.argv


def descs(w81, name):
    d = np.array([m['descriptor'] for m in P82['real_materials']['list']], np.float64) if name == 'real' else \
        np.load(os.path.join(w81, 'materials.npz'))[name]
    return d[:16] if (DEV and name != 'real') else d


def arm_params(name, fitdir, setname, w81C):
    if name.startswith('W81_'):
        a = name[4:]
        z = torch.from_numpy(np.load(os.path.join(w81C, 'fit', f'direct_{a}_gate.npz'))['z']).to(dev_t)
        return 'C0', EV81.compile_runtime(a, z)[0]
    arm, mode = name.split('_')[:2]
    z = torch.from_numpy(np.load(os.path.join(fitdir, f'fit_{setname}_{name}.npz'))['z']).to(dev_t)
    with torch.no_grad():
        return arm, CL.decode(arm, mode, z)


def closure_lanes(arm, p, m, wi, wo, chunk=1 << 21):
    out = []
    with torch.no_grad():
        for s in range(0, len(m), chunk):
            e = slice(s, min(len(m), s + chunk))
            mm = torch.as_tensor(m[e], device=dev_t).long()
            out.append(CL.eval_closure(arm, CL.gather(p, mm), torch.as_tensor(wi[e], device=dev_t), torch.as_tensor(wo[e], device=dev_t)))
    return torch.cat(out)


def group_metrics(pred, f, w, r_ind):
    """pred, f [G,n,3] (double), w [G,n,1], r_ind [G,3] -> dict of numpy [G] arrays."""
    a_cl = (pred * w).sum(1); a_ref = (f * w).sum(1)
    den = a_ref.sum(1).clamp_min(1e-300)
    el1 = ((pred - f).abs() * w).sum((1, 2)) / den
    ealb = (a_cl - a_ref).abs().sum(1) / den
    zero = a_cl <= 1e-300
    pc = torch.where(zero[:, None], torch.zeros_like(pred), pred / a_cl.clamp_min(1e-300)[:, None])
    tv = (w * (pc - f / a_ref.clamp_min(1e-300)[:, None]).abs()).sum(1)
    eshape = (a_ref * tv).sum(1) / den
    r_same = torch.where(zero, torch.zeros_like(a_cl), a_ref / a_cl.clamp_min(1e-300))
    el1_same = ((pred * r_same[:, None] - f).abs() * w).sum((1, 2)) / den
    ealb_same = (a_cl * r_same - a_ref).abs().sum(1) / den
    el1_or = ((pred * r_ind[:, None] - f).abs() * w).sum((1, 2)) / den
    ealb_or = (a_cl * r_ind - a_ref).abs().sum(1) / den
    o = {'el1': el1, 'ealb': ealb, 'eshape': eshape, 'el1_or': el1_or, 'ealb_or': ealb_or, 'el1_same': el1_same, 'ealb_same': ealb_same,
         'zero_channels': zero.sum(1).double()}
    return {k: v.cpu().numpy() for k, v in o.items()}, a_cl.cpu().numpy(), a_ref.cpu().numpy()


def highlight_arm(arm, p, hl, ratio_case):
    """WL81 highlight statistics for the closure and for the oracle-scaled closure (channel ratio per case)."""
    A = DA.A['highlight']
    n = A['grid']
    Gn = len(hl['mat'])
    chunk = max(1, (1 << 22) // (n * n))
    acc = {v: {'peak_err': [], 'logw': []} for v in ['base', 'oracle']}
    for c0 in range(0, Gn, chunk):
        cs = np.arange(c0, min(Gn, c0 + chunk))
        wi = hl['wi'][cs]
        wo, sx, sy = EV.grid_dirs(wi, hl['smax'][cs], n)
        ok = wo[..., 2] > 1e-4
        wo_c = wo.copy(); wo_c[..., 2] = np.maximum(wo_c[..., 2], 1e-4)
        wo_c /= np.linalg.norm(wo_c, axis=-1, keepdims=True)
        pr = closure_lanes(arm, p, np.repeat(hl['mat'][cs], n * n), np.repeat(wi, n * n, 0).astype(np.float32),
                           wo_c.reshape(-1, 3).astype(np.float32)).double().cpu().numpy().reshape(len(cs), n * n, 3)
        for v, arr in [('base', pr), ('oracle', pr * ratio_case[cs][:, None, :])]:
            vc = np.where(ok, arr.mean(2), -1.0)
            sc = EV.highlight_stats(vc, wo, sx, sy, n)
            dots = np.clip((hl['full_peak_wo'][cs] * sc['peak_wo']).sum(1), -1, 1)
            acc[v]['peak_err'].append(np.degrees(np.arccos(dots)))
            lw = 0.5 * np.log(np.maximum(sc['area'], 1) / np.maximum(hl['full_area'][cs], 1))
            acc[v]['logw'].append(np.where(sc['touch'] | hl['full_touch'][cs], np.nan, lw))
    out = {}
    pron = hl['pron']
    for v in acc:
        pk = np.concatenate(acc[v]['peak_err']); lw = np.concatenate(acc[v]['logw'])
        sel = pron & np.isfinite(lw)
        out[v] = {'n_cases': int(Gn), 'n_pronounced': int(pron.sum()), 'n_pronounced_resolved': int(sel.sum()),
                  'peak_err_deg_median': EV.q(pk[pron], .5), 'peak_err_deg_p90': EV.q(pk[pron], .9),
                  'abs_logw_median': EV.q(np.abs(lw[sel]), .5), 'abs_logw_p90': EV.q(np.abs(lw[sel]), .9),
                  'logw_median_signed': EV.q(lw[sel], .5)}
    return out


def shape_summary(es):
    gi = [i for i, t in enumerate(TH) if t >= DA.A['grazing_min_deg']]
    return {'median': EV.q(es.ravel(), .5), 'p90': EV.q(es.ravel(), .9), 'p99': EV.q(es.ravel(), .99), 'max': float(es.max()),
            'by_theta_median': {int(t): EV.q(es[:, i].ravel(), .5) for i, t in enumerate(TH)}, 'grazing_median': EV.q(es[:, gi].ravel(), .5)}


def cluster_mask(d):
    """Failure cluster (predeclared): lowest tercile of tau, lowest tercile of alpha_b, highest tercile of mean conductor reflectance
    (terciles of the set itself)."""
    q = lambda v, a: np.quantile(v, a)
    return (d[:, 2] <= q(d[:, 2], 1 / 3)) & (d[:, 9] <= q(d[:, 9], 1 / 3)) & (d[:, 6:9].mean(1) >= q(d[:, 6:9].mean(1), 2 / 3))


def main():
    data, w81, w81C, fitdir, setname, out = sys.argv[1:7]
    os.makedirs(out, exist_ok=True)
    t0 = time.time()
    d = descs(w81, setname)
    ev = dict(np.load(os.path.join(data, 'eval82_real.npz') if setname == 'real' else os.path.join(w81, f'eval_{setname}.npz')))
    nm, nth, nph, nwo = [int(v) for v in ev['shape']]
    nm = min(nm, len(d))
    G = nm * nth * nph
    L = G * nwo
    alb = dict(np.load(os.path.join(data, f'alb82_{setname}.npz')))
    Na = int(alb['Na'])
    hlp = os.path.join(data, 'hl82_real.npz') if setname == 'real' else os.path.join(w81, f'hl_{setname}.npz')
    hl = dict(np.load(hlp)) if os.path.exists(hlp) and setname != 'test_block' else None
    if hl is not None and DEV:
        k = hl['mat'] < nm
        hl = {kk: (v[k] if (hasattr(v, 'shape') and v.shape[:1] == k.shape) else v) for kk, v in hl.items()}
    f = torch.from_numpy(ev['f'][:L]).to(dev_t).double().view(G, nwo, 3)
    w = (torch.from_numpy(ev['use'][:L].astype(np.float64)).to(dev_t) / torch.from_numpy(ev['q'][:L]).to(dev_t).double()).view(G, nwo, 1)
    # reference noise floors (fA vs fB halves): E_L1 / E_alb (WL81 definition) and E_shape
    fA = torch.from_numpy(ev['fA'][:L]).to(dev_t).double().view(G, nwo, 3); fB = torch.from_numpy(ev['fB'][:L]).to(dev_t).double().view(G, nwo, 3)
    nz, _, _ = group_metrics(fA, fB, w, torch.ones(G, 3, dtype=torch.float64, device=dev_t))
    noise = {'E_L1_median': EV.q(nz['el1'] / 2, .5), 'E_L1_p90': EV.q(nz['el1'] / 2, .9), 'E_alb_median': EV.q(nz['ealb'] / 2, .5),
             'E_alb_p95': EV.q(nz['ealb'] / 2, .95), 'E_shape_median': EV.q(nz['eshape'] / 2, .5), 'E_shape_p90': EV.q(nz['eshape'] / 2, .9),
             'note': 'half-vs-half difference / 2 (walk noise; the wo samples are shared)'}
    del fA, fB
    # independent albedo samples (lanes: group-major, Na per group)
    wiG = ev['wi'][: L: nwo]
    mG = ev['m'][: L: nwo]
    alb_wi = np.repeat(wiG, Na, 0); alb_m = np.repeat(mG, Na)
    walb = torch.from_numpy(alb['use'][: G * Na].astype(np.float64) / alb['q'][: G * Na]).to(dev_t).view(G, Na, 1)
    A_ref_ind = torch.from_numpy(alb['A'][:G]).to(dev_t)
    names = sorted({fn[len(f'fit_{setname}_'):-4] for fn in os.listdir(fitdir) if fn.startswith(f'fit_{setname}_') and fn.endswith('.npz')})
    if setname == 'gate' and os.path.exists(os.path.join(w81C, 'fit')):
        names = [f'W81_{a}' for a in ['control', 'candidate', 'unbounded']] + names
    # case -> group index for the highlight grids (highlight wi = eval theta index, first eval phi)
    if hl is not None:
        th_idx = np.array([int(np.argmin(np.abs(np.radians(TH) - t))) for t in hl['theta']])
        case_g = hl['mat'] * nth * nph + th_idx * nph
    res = {'set': setname, 'n_materials': nm, 'groups': G, 'noise_floor': noise,
           'albedo_set': json.load(open(os.path.join(data, f'alb82_{setname}.json'))), 'arms': {}}
    lanes = {}
    for name in names:
        ta = time.time()
        arm, p = arm_params(name, fitdir, setname, w81C)
        if DEV and not name.startswith('W81_'):
            p = {k: v[:nm] for k, v in p.items()}
        pred = closure_lanes(arm, p, ev['m'][:L], ev['wi'][:L], ev['wo'][:L]).double().view(G, nwo, 3)
        pa = closure_lanes(arm, p, alb_m, alb_wi, alb['wo'][: G * Na]).double().view(G, Na, 3)
        A_cl_ind = (pa * walb).mean(1)
        del pa
        r_ind = torch.where(A_cl_ind > 1e-300, A_ref_ind / A_cl_ind.clamp_min(1e-300), torch.zeros_like(A_cl_ind))
        gm, a_cl, a_ref = group_metrics(pred, f, w, r_ind)
        del pred
        sh = lambda v: v.reshape(nm, nth, nph)
        S = EV.summarise(sh(gm['el1']), sh(gm['ealb']))
        So = EV.summarise(sh(gm['el1_or']), sh(gm['ealb_or']))
        r = {'arm': arm, 'summary': S, 'shape': shape_summary(sh(gm['eshape'])), 'oracle': So,
             'identity': {'max_abs_el1_same_minus_eshape': float(np.abs(gm['el1_same'] - gm['eshape']).max()),
                          'max_ealb_same': float(gm['ealb_same'].max())},
             'zero_albedo_channels': int(gm['zero_channels'].sum()),
             'strata_E_L1': EV81.family_strata(sh(gm['el1']), d), 'strata_E_shape': EV81.family_strata(sh(gm['eshape']), d),
             'strata_E_alb': EV81.family_strata(sh(gm['ealb']), d)}
        cm = cluster_mask(d)
        mmean = lambda v: sh(v).mean((1, 2))
        r['failure_cluster'] = {'n': int(cm.sum()), 'E_L1_median': EV.q(mmean(gm['el1'])[cm], .5), 'E_shape_median': EV.q(mmean(gm['eshape'])[cm], .5),
                                'E_alb_median': EV.q(mmean(gm['ealb'])[cm], .5), 'oracle_E_L1_median': EV.q(mmean(gm['el1_or'])[cm], .5)}
        if setname == 'gate':
            r['named'] = {str(i): {'E_L1_mean': float(mmean(gm['el1'])[i]), 'E_shape_mean': float(mmean(gm['eshape'])[i]),
                                   'E_alb_mean': float(mmean(gm['ealb'])[i]), 'oracle_E_L1_mean': float(mmean(gm['el1_or'])[i]),
                                   'E_L1_by_theta': sh(gm['el1'])[i].mean(1).tolist()} for i in P82['evaluation']['named_gate_materials'] if i < nm}
        if setname == 'real':
            r['per_material'] = {P82['real_materials']['list'][i]['name']: {'E_L1_mean': float(mmean(gm['el1'])[i]), 'E_shape_mean': float(mmean(gm['eshape'])[i]),
                                 'E_alb_mean': float(mmean(gm['ealb'])[i]), 'oracle_E_L1_mean': float(mmean(gm['el1_or'])[i])} for i in range(nm)}
        if hl is not None:
            ratio_case = r_ind.cpu().numpy()[case_g]
            H = highlight_arm(arm, p, hl, ratio_case)
            r['highlight'] = H['base']; r['highlight_oracle'] = H['oracle']
            r['gate'] = EV81.representation_gate(S, H['base'])
            r['gate_oracle'] = EV81.representation_gate(So, H['oracle'])
        else:
            g = P82['evaluation']['wl81_thresholds']
            r['gate_no_highlight'] = {'R1': S['E_L1']['median'] <= g['E_L1_median'] and S['E_L1']['p90'] <= g['E_L1_p90'],
                                      'R2': S['E_alb']['median'] <= g['E_alb_median'] and S['E_alb']['p95'] <= g['E_alb_p95']}
            r['gate_oracle_no_highlight'] = {'R1': So['E_L1']['median'] <= g['E_L1_median'] and So['E_L1']['p90'] <= g['E_L1_p90']}
        r['seconds'] = time.time() - ta
        res['arms'][name] = r
        lanes[name] = {k: sh(gm[k]).astype(np.float32) for k in ['el1', 'ealb', 'eshape', 'el1_or', 'ealb_or']}
        lanes[name]['A_cl_eval'] = a_cl.astype(np.float32); lanes[name]['A_cl_ind'] = A_cl_ind.cpu().numpy().astype(np.float32)
        print(setname, name, 'E_L1', round(S['E_L1']['median'], 4), 'E_alb', round(S['E_alb']['median'], 4), 'E_shape',
              round(r['shape']['median'], 4), 'oracle E_L1', round(So['E_L1']['median'], 4), round(time.time() - ta, 1), 's', flush=True)
    np.savez_compressed(os.path.join(out, f'lanes82_{setname}.npz'), A_ref_ind=A_ref_ind.cpu().numpy(),
                        **{f'{n}__{k}': v for n, dd in lanes.items() for k, v in dd.items()})
    res['seconds'] = time.time() - t0
    json.dump(res, open(os.path.join(out, f'eval82_{setname}.json'), 'w'), indent=1, default=float)


if __name__ == '__main__':
    main()
    os._exit(0)
