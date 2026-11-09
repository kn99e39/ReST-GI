"""WL81 POST HOC attribution of the Stage C representation failure (run after the frozen Stage C verdict; does not change it).

  python diag81_posthoc.py <data> <fitC> <out>

On the gate set, with the same basis / rank / energy arms:
  P1 optimisation: frozen loss (log1p MSE), 4x longer (12000 steps) from the frozen Stage C solution, and a second random init (12000 steps)
  P2 objective:    metric-matched objective = per-material energy-weighted relative L1 on the fit directions (importance weights 1/q of the
                   frozen wo mixture) + relative albedo error, 6000 steps from the frozen Stage C solution
  Arms: unbounded (basis capacity without energy bound), candidate, control. Evaluated with the frozen Stage C metrics (E_L1, E_alb, highlight).
If P2 reaches the representation thresholds, the basis can represent the targets and the frozen training objective limits the fits; if not,
the limit is basis capacity (with the unbounded arm separating the energy constraint).
"""
import json, math, os, sys, time
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import closure81 as K
C = K.install()
sys.path.insert(0, os.path.join(HERE, '..', 'fnto_compiled80c'))
import train80c as TR  # noqa: E402
import energy81 as E  # noqa: E402
import fit81 as F  # noqa: E402
import eval81 as EV81  # noqa: E402
import data81 as DA  # noqa: E402

dev_t = torch.device('cuda')


def fit_q(desc, fit):
    """Exact density of the frozen wo mixture for the fit pairs (data81.mixture)."""
    m = fit['m'].cpu().numpy(); wi = fit['wi'].cpu().numpy().astype(np.float64); wo = fit['wo'].cpu().numpy().astype(np.float64)
    d = desc[m]
    a1 = d[:, 0]; a2 = np.minimum(1.0, d[:, 1] * d[:, 9])
    W = DA.W_MIX
    q = W[0] * np.maximum(wo[:, 2], 0) / np.pi + W[1] / (2 * np.pi) + W[2] * DA.vndf_pdf(wi, wo, a1) + W[3] * DA.vndf_pdf(wi, wo, a2)
    return torch.from_numpy(q.astype(np.float32)).to(dev_t)


def run_fit(arm, z0, fit, n_mat, steps, objective, q=None, lr0=0.03, lr_end=3e-4):
    wf, uf = F.fit_grids()
    z = torch.nn.Parameter(z0.clone())
    opt = torch.optim.Adam([z], lr=lr0)
    idx = TR.per_material_index(fit['m'], n_mat)
    lanes = idx.reshape(-1)
    mid = torch.arange(n_mat, device=dev_t).repeat_interleave(idx.shape[1])
    wi, wo, f = fit['wi'][lanes], fit['wo'][lanes], fit['f'][lanes]
    w = (1.0 / q[lanes])[:, None] if q is not None else None
    for step in range(steps):
        for pg in opt.param_groups:
            pg['lr'] = TR.lr_at(step, steps, lr0, lr_end, 50)
        p = F.arm_params(arm, z, wf, uf)
        pred = K.eval_closure(C.gather(p, mid), wi, wo)
        if objective == 'log1p':
            per = ((torch.log1p(pred) - torch.log1p(f)) ** 2).mean(1).view(n_mat, -1).mean(1)
        else:
            num = ((pred - f).abs() * w).sum(1).view(n_mat, -1).sum(1)
            den = (f * w).sum(1).view(n_mat, -1).sum(1).clamp_min(1e-12)
            ac = (pred * w).view(n_mat, -1, 3).sum(1); ar = (f * w).view(n_mat, -1, 3).sum(1)
            per = num / den + (ac - ar).abs().sum(1) / ar.sum(1).clamp_min(1e-12)
        opt.zero_grad(set_to_none=True)
        per.sum().backward()
        opt.step()
    return z.detach()


def main():
    data, fitC, out = sys.argv[1:4]
    os.makedirs(out, exist_ok=True)
    mats = np.load(os.path.join(data, 'materials.npz'))
    d = mats['gate']; n = len(d)
    fit = F.load_pairs(os.path.join(data, 'fit_gate.npz'))
    q = fit_q(d, fit)
    ev = dict(np.load(os.path.join(data, 'eval_gate.npz')))
    hl = dict(np.load(os.path.join(data, 'hl_gate.npz')))
    torch.manual_seed(81900)
    res = {'label': 'POST HOC diagnostic after the frozen Stage C verdict (representation clear fail for every arm); does not change any verdict'}
    ps = {}
    arms = sys.argv[4:] or ['unbounded', 'candidate', 'control']
    for arm in arms:
        z0 = torch.from_numpy(np.load(os.path.join(fitC, f'direct_{arm}_gate.npz'))['z']).to(dev_t)
        variants = {}
        t0 = time.time()
        variants['P1_long'] = run_fit(arm, z0, fit, n, 12000, 'log1p')
        zr = E.init_raw(arm, n, dev_t) + 0.5 * torch.randn(n, C.N_RAW, device=dev_t)
        variants['P1_reinit_long'] = run_fit(arm, zr, fit, n, 12000, 'log1p')
        variants['P2_metric'] = run_fit(arm, z0, fit, n, 6000, 'metric', q, lr0=0.01, lr_end=1e-4)
        for k, z in variants.items():
            p, st, dg = EV81.compile_runtime(arm, z)
            S, ST, el1, ealb = EV81.eval_arm(p, ev, d)
            ps[f'{arm}_{k}'] = p
            res[f'{arm}_{k}'] = {'E_L1': S['E_L1'], 'E_alb': S['E_alb'], 'grazing_median': S['grazing_median'], 'strata': ST, 'compile': dg}
            print(arm, k, json.dumps(S['E_L1']), json.dumps(S['E_alb']), flush=True)
        res[f'{arm}_seconds'] = time.time() - t0
    H = EV81.highlight(ps, hl)
    for k, h in H.items():
        res[k]['highlight'] = h
        res[k]['representation_rule_applied'] = EV81.representation_gate({'E_L1': res[k]['E_L1'], 'E_alb': res[k]['E_alb']}, h)
    json.dump(res, open(os.path.join(out, f'diag_posthoc_{"_".join(arms)}.json'), 'w'), indent=1, default=float)
    print(json.dumps({k: v['representation_rule_applied'] for k, v in res.items() if isinstance(v, dict) and 'representation_rule_applied' in v}, indent=1))


if __name__ == '__main__':
    main()
    os._exit(0)
