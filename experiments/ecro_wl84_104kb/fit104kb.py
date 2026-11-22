"""WL104K-B fitting arms for the UNCHANGED ER-v1-SDM (P) representation.

  python fit104kb.py <out_dir> --method adam|lbfgs --data original|coverage --objective original|e2
                     [--starts 8|4] [--synthetic] [--budget 12000]

method adam  : O0, the WL104K method (WL82 Adam, lr 0.03 -> 3e-4 cosine, warm-up 50, float32, full batch, gates clamped to [0, 1]
               after every step), --budget steps.
method lbfgs : O1, batched projected L-BFGS (protocol optimization_O1), --budget objective evaluations per unit.
objective e2 : el1 + ealb + e_core + e_shoulder (region-normalized relative L1 of the < 3 deg cap and the 3-12 deg annulus) per probe.
--synthetic  : protocol synthetic control targets (truth + per-lane noise of the chosen physical set); starts 1-4.
Outputs: states.npz (P_y [R * n, 58], P_g [R * n, 8, 3, 5], start-major as WL104K) and fit.json.
"""
import argparse
import platform
import time
from pathlib import Path

import numpy as np
import torch

import common104kb as KB

CM, IO, GF, FT = KB.CM, KB.IO, KB.GF, KB.FT
DEV = KB.DEV
OPT = FT.OPT
LAMBDA_CORE = 1.0
LAMBDA_SHOULDER = 1.0
CORE_DEG = 3.0
SHOULDER_DEG = 12.0
LOG = 100
NPAR = 178


class Data:
    """Full-batch lane data for R starts x n materials (one independent unit per start and material)."""

    def __init__(self, ds, n, R, objective, target=None):
        self.n, self.R, self.objective = n, R, objective
        self.P, self.L = ds["wi"].shape[1:3]
        g = lambda a: torch.as_tensor(np.ascontiguousarray(a), device=DEV, dtype=torch.float32)
        self.wi = g(ds["wi"]).reshape(-1, 3).repeat(R, 1)
        self.wo = g(ds["wo"]).reshape(-1, 3).repeat(R, 1)
        self.ref = g(ds["mean"] if target is None else target).reshape(-1, 3).repeat(R, 1)
        self.w = g(ds["w"]).reshape(-1).repeat(R)
        rad = KB.mirror_radius(ds["wi"], ds["wo"])
        self.core = g(rad < CORE_DEG).reshape(-1).repeat(R)
        self.shoulder = g((rad >= CORE_DEG) & (rad < SHOULDER_DEG)).reshape(-1).repeat(R)

    def terms(self, x):
        """x [U, 178] (U = R * n, start-major) -> per-unit objective [U] and per-unit term means."""
        U = x.shape[0]
        y, gt = x[:, :58], x[:, 58:].reshape(U, 8, 3, 5)
        p = GF.decode(y)
        rep = self.P * self.L
        pp = {k: v[:, None].expand(U, rep, *v.shape[1:]).reshape(-1, *v.shape[1:]) for k, v in p.items()}
        gg = gt[:, None].expand(U, rep, 8, 3, 5).reshape(-1, 8, 3, 5)
        pred = CM.eval_sdm(pp, gg, self.wi, self.wo)
        t = FT.F.group_terms(pred, self.ref, self.w, self.L)
        out = {"el1": t["e1"].reshape(U, self.P).mean(-1), "ealb": t["e2"].reshape(U, self.P).mean(-1)}
        obj = (t["e1"] + t["e2"]).reshape(U, self.P).mean(-1)
        if self.objective == "e2":
            G = U * self.P
            Pv, Fr = pred.view(G, self.L, 3), self.ref.view(G, self.L, 3)
            for name, mask, lam in (("e_core", self.core, LAMBDA_CORE), ("e_shoulder", self.shoulder, LAMBDA_SHOULDER)):
                W = (self.w * mask).view(G, self.L, 1)
                e = ((Pv - Fr).abs() * W).sum((1, 2)) / (Fr * W).sum((1, 2)).clamp_min(1e-12)
                out[name] = e.reshape(U, self.P).mean(-1)
                obj = obj + lam * out[name]
        out["objective"] = obj
        return out

    def f(self, x):
        return self.terms(x)["objective"]


def clamp_gates(x):
    return torch.cat([x[:, :58], x[:, 58:].clamp(0.0, 1.0)], 1)


# ------------------------------------------------------------------ O0: original Adam method (same operations as fit104k.train)
def adam(data, x0, steps):
    y = torch.nn.Parameter(x0[:, :58].clone())
    g = torch.nn.Parameter(x0[:, 58:].clone())
    opt = torch.optim.Adam([y, g], lr=OPT["lr"])
    hist, evals = {}, 0
    for step in range(steps + 1):
        sc = data.f(torch.cat([y, g], 1))
        evals += 1
        if step % LOG == 0 or step == steps:
            v = sc.detach().cpu().numpy()
            if not np.isfinite(v).all():
                raise IO.Stop(f"NONFINITE_OBJECTIVE adam {step}")
            hist[step] = v
        if step < steps:
            opt.param_groups[0]["lr"] = FT.TR.lr_at(step, steps, OPT["lr"], OPT["lr_end"], OPT["warmup"])
            opt.zero_grad(set_to_none=True)
            sc.sum().backward()
            opt.step()
            with torch.no_grad():
                g.clamp_(0.0, 1.0)
    x = torch.cat([y.detach(), g.detach()], 1)
    U = x.shape[0]
    return x, hist, {"evaluations_per_unit": [steps] * U, "gradient_evaluations_per_unit": [steps] * U,
                     "terminated_early": [False] * U}


# ------------------------------------------------------------------ O1: batched projected L-BFGS (protocol optimization_O1)
def fgrad(data, x):
    x = x.detach().requires_grad_(True)
    f = data.f(x)
    (gr,) = torch.autograd.grad(f.sum(), x)
    return f.detach(), gr.detach()


def free_mask(x, gr):
    """1 for free coordinates; gate knots at a bound with the gradient pointing outward are fixed."""
    m = torch.ones_like(x)
    gx, gg = x[:, 58:], gr[:, 58:]
    fixed = ((gx <= 0.0) & (gg > 0)) | ((gx >= 1.0) & (gg < 0))
    m[:, 58:] = (~fixed).to(x.dtype)
    return m


def lbfgs(data, x0, budget, mem=20, c1=1e-4, max_halvings=12, first_step=0.01):
    U, D = x0.shape
    dev, dt = x0.device, x0.dtype
    x = clamp_gates(x0.clone())
    S = torch.zeros(U, mem, D, device=dev, dtype=dt)
    Y = torch.zeros(U, mem, D, device=dev, dtype=dt)
    V = torch.zeros(U, mem, dtype=torch.bool, device=dev)
    evals = torch.zeros(U, dtype=torch.long, device=dev)
    gevals = torch.zeros(U, dtype=torch.long, device=dev)
    fails = torch.zeros(U, dtype=torch.long, device=dev)
    frozen = torch.zeros(U, dtype=torch.bool, device=dev)
    resets = torch.zeros(U, dtype=torch.long, device=dev)
    f, gr = fgrad(data, x)
    evals += 1; gevals += 1
    hist_e, hist_f = [[0] for _ in range(U)], [[float(v)] for v in f.cpu()]
    need_scale = torch.ones(U, dtype=torch.bool, device=dev)
    it = 0
    while True:
        active = (~frozen) & (evals < budget - 1)          # keep one evaluation for the gradient at the accepted point
        if not bool(active.any()):
            break
        it += 1
        fm = free_mask(x, gr)
        q = gr * fm
        Sm, Ym = S * fm[:, None], Y * fm[:, None]
        sy = (Sm * Ym).sum(-1)
        ok = V & (sy > 1e-10 * Sm.norm(dim=-1) * Ym.norm(dim=-1)) & (sy > 0)
        rho = torch.where(ok, 1.0 / sy.clamp_min(1e-30), torch.zeros_like(sy))
        alph = torch.zeros(U, mem, device=dev, dtype=dt)
        for i in range(mem - 1, -1, -1):                     # newest pair at index mem - 1
            a = rho[:, i] * (Sm[:, i] * q).sum(-1)
            alph[:, i] = a
            q = q - a[:, None] * Ym[:, i]
        # initial scaling from the newest valid pair
        idx = torch.where(ok, torch.arange(mem, device=dev)[None].expand(U, mem), torch.full((U, mem), -1, device=dev)).max(-1).values
        has = idx >= 0
        ii = idx.clamp_min(0)
        sy_n = sy.gather(1, ii[:, None])[:, 0]
        yy_n = (Ym.gather(1, ii[:, None, None].expand(U, 1, D))[:, 0] ** 2).sum(-1)
        gam = torch.where(has, sy_n / yy_n.clamp_min(1e-30), torch.ones_like(sy_n))
        r = gam[:, None] * q
        for i in range(mem):
            b = rho[:, i] * (Ym[:, i] * r).sum(-1)
            r = r + (alph[:, i] - b)[:, None] * Sm[:, i]
        d = -r * fm
        gd = (gr * d).sum(-1)
        bad = (gd >= 0) | ~torch.isfinite(gd) | ~has
        reset = bad & has
        resets += reset.long()
        V[reset] = False
        d = torch.where(bad[:, None], -gr * fm, d)
        need_scale = need_scale | bad
        step = torch.ones(U, device=dev, dtype=dt)
        dmax = d.abs().max(-1).values.clamp_min(1e-30)
        step = torch.where(need_scale, torch.clamp(first_step / dmax, max=1.0), step)
        # projected Armijo backtracking, batched over units
        searching = active.clone()
        accepted = torch.zeros(U, dtype=torch.bool, device=dev)
        x_new, f_new = x.clone(), f.clone()
        with torch.no_grad():
            for _ in range(max_halvings + 1):
                if not bool(searching.any()):
                    break
                xt = clamp_gates(x + step[:, None] * d)
                ft = data.f(xt)
                evals += searching.long()
                okk = searching & torch.isfinite(ft) & (ft <= f + c1 * (gr * (xt - x)).sum(-1))
                x_new = torch.where(okk[:, None], xt, x_new)
                f_new = torch.where(okk, ft, f_new)
                accepted |= okk
                searching &= ~okk
                searching &= evals < budget - 1
                step = torch.where(searching, 0.5 * step, step)
        failed = active & ~accepted
        fails = torch.where(failed, fails + 1, torch.zeros_like(fails))
        V[failed] = False
        need_scale = failed.clone()
        frozen |= fails >= 3
        if bool(accepted.any()):
            f2, g2 = fgrad(data, x_new)
            gevals += accepted.long()
            evals += accepted.long()
            s = x_new - x
            yv = g2 - gr
            store = accepted & ((s * yv).sum(-1) > 1e-10 * s.norm(dim=-1) * yv.norm(dim=-1))
            S = torch.where(store[:, None, None], torch.cat([S[:, 1:], s[:, None]], 1), S)
            Y = torch.where(store[:, None, None], torch.cat([Y[:, 1:], yv[:, None]], 1), Y)
            V = torch.where(store[:, None], torch.cat([V[:, 1:], torch.ones(U, 1, dtype=torch.bool, device=dev)], 1), V)
            x = torch.where(accepted[:, None], x_new, x)
            f = torch.where(accepted, f2, f)
            gr = torch.where(accepted[:, None], g2, gr)
        if not torch.isfinite(f).all():
            raise IO.Stop("NONFINITE_OBJECTIVE lbfgs")
        fc, ec = f.cpu().numpy(), evals.cpu().numpy()
        for u in range(U):
            if ec[u] - hist_e[u][-1] >= LOG or (frozen[u] or ec[u] >= budget):
                if hist_e[u][-1] != ec[u]:
                    hist_e[u].append(int(ec[u])); hist_f[u].append(float(fc[u]))
    info = {"evaluations_per_unit": evals.cpu().tolist(), "gradient_evaluations_per_unit": gevals.cpu().tolist(),
            "terminated_early": (frozen & (evals < budget)).cpu().tolist(), "memory_resets_per_unit": resets.cpu().tolist(),
            "iterations": it, "history_evaluations": hist_e, "history_objective": hist_f}
    return x, None, info


def stationarity(hist_steps, hist_vals, window=2000):
    """Relative change of the unit's own objective over the last `window` steps / evaluations."""
    s = np.asarray(hist_steps, float); v = np.asarray(hist_vals, float)
    if s[-1] < window:
        return None
    k = np.searchsorted(s, s[-1] - window, side="right") - 1
    return float(abs(v[-1] - v[k]) / abs(v[-1]))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("out", type=Path)
    ap.add_argument("--method", choices=("adam", "lbfgs"), required=True)
    ap.add_argument("--data", choices=("original", "coverage"), required=True)
    ap.add_argument("--objective", choices=("original", "e2"), required=True)
    ap.add_argument("--starts", type=int, choices=(4, 8), default=8)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--budget", type=int, default=12000)
    ap.add_argument("--data-dir", type=Path, default=KB.DATA_DIR, help="run-root data directory (train_coverage.npz)")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    t_all = time.monotonic()
    rows, raw, _, _, _, seen = CM.inputs()
    n = len(rows)
    ds = KB.training_set(args.data, n, args.data_dir)
    keys = KB.START_KEYS[:args.starts]
    R = len(keys)
    target, truth = None, None
    if args.synthetic:
        truth = KB.synthetic_truth(raw["R0"])
        tgt = CM.predict("P", truth, ds["wi"], ds["wo"])
        sigma = 0.5 * np.abs(ds["a"] - ds["b"])
        target = tgt + sigma * np.random.default_rng(KB.P["seeds"]["synthetic_noise"]).standard_normal(tgt.shape)
    data = Data(ds, n, R, args.objective, target)
    y0 = KB.starts_y(n, keys)
    x0 = torch.as_tensor(np.concatenate([y0, np.ones((R * n, 120))], 1), device=DEV, dtype=torch.float32)
    assert x0.shape[1] == NPAR
    with torch.no_grad():
        init_terms = {k: v.cpu().numpy().tolist() for k, v in data.terms(x0).items()}
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    t0 = time.monotonic()
    if args.method == "adam":
        x, hist, info = adam(data, x0, args.budget)
        steps = sorted(hist)
        H = np.stack([hist[s] for s in steps])
        info["history_steps"] = steps
        info["history_objective"] = H.T.tolist()
        stat = [stationarity(steps, H[:, u]) for u in range(R * n)]
    else:
        x, _, info = lbfgs(data, x0, args.budget)
        stat = [stationarity(info["history_evaluations"][u], info["history_objective"][u]) for u in range(R * n)]
    torch.cuda.synchronize()
    secs = time.monotonic() - t0
    with torch.no_grad():
        fin = {k: v.cpu().numpy().tolist() for k, v in data.terms(x).items()}
        xr = x.clone().requires_grad_(True)
    f = data.f(xr)
    (gr,) = torch.autograd.grad(f.sum(), xr)
    fm = free_mask(x, gr.detach())
    pg = (gr.detach() * fm).norm(dim=-1).cpu().numpy()
    xn = x.detach().cpu().numpy().astype(np.float64)
    gates = xn[:, 58:]
    out = {"P_y": xn[:, :58], "P_g": gates.reshape(-1, 8, 3, 5)}
    if truth is not None:
        out.update({"truth_y": truth["y"], "truth_g": truth["g"]})
    np.savez(args.out / "states.npz", **out)
    budget_ok = max(info["evaluations_per_unit"]) <= args.budget
    log = {"arm": {"method": args.method, "data": args.data, "objective": args.objective, "synthetic": args.synthetic},
           "start_keys": keys, "materials": [r["tag"] for r in rows], "budget": args.budget, "budget_respected": bool(budget_ok),
           "parameters_per_material": NPAR, "seconds": secs, "seconds_total": time.monotonic() - t_all,
           "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()), "initial_terms": init_terms, "final_terms": fin,
           "relative_change_last_2000": stat,
           "stationary_fraction_0.5pct": float(np.mean([s is not None and s <= 0.005 for s in stat])),
           "terminal_projected_gradient_norm": pg.tolist(),
           "gate_fraction_at_bounds": float(((gates <= 1e-7) | (gates >= 1 - 1e-7)).mean()),
           "lambda_core": LAMBDA_CORE if args.objective == "e2" else None, "lambda_shoulder": LAMBDA_SHOULDER if args.objective == "e2" else None,
           "core_deg": CORE_DEG, "shoulder_deg": SHOULDER_DEG,
           "optimizer": ({**OPT, "log_interval": LOG, "dtype": "float32", "gate_projection": "clamp [0,1] after every step"}
                         if args.method == "adam" else {"memory": 20, "c1": 1e-4, "max_halvings": 12, "first_step_max_abs": 0.01,
                                                         "dtype": "float32", "freeze_after_failed_line_searches": 3,
                                                         "reset_after_failed_line_search": "memory cleared and first-step scaling re-applied (disclosed)"}),
           **{k: v for k, v in info.items() if k not in ("history_objective",)},
           "history_objective": info["history_objective"],
           "data_dir": str(args.data_dir.resolve()) if args.data == "coverage" else None,
           "data_sha256": (KB.sha(args.data_dir / "train_coverage.npz") if args.data == "coverage" else
                           seen.get("results/ecro_wl84_104j/primary/data/fit_set.npz")),
           "commit": KB.commit(), "protocol_sha256": KB.sha(KB.PROTOCOL), "inputs_sha256": seen, "host": platform.platform(),
           "torch": torch.__version__, "gpu": torch.cuda.get_device_name(), "states_sha256": KB.sha(args.out / "states.npz")}
    KB.write(args.out / "fit.json", log)
    print("fit done", args.method, args.data, args.objective, "synthetic" if args.synthetic else "physical",
          round(secs, 1), "s median final", float(np.median(fin["objective"])), flush=True)


if __name__ == "__main__":
    main()
