"""WL104K-C continuation arms for the UNCHANGED ER-v1-SDM (178 parameters per material), warm-started from a fixed-shape state.

  python fit104kc.py <out_dir> --arm CR|O0C|E2W --warm <states.npz> --starts 8|4 --data original|coverage [--synthetic]
                     [--steps 6000] [--warm-r4 <states.npz>] [--inspect]

CR  : constrained refinement (protocol stage_B): minimize the specular-core error F(x) = mean_p e_core,p(x) / mean_p e_core,p(x0)
      subject to, for EVERY probe p, el1_p, ealb_p, e_shoulder_p, e_tail_p <= b_{k,p} = max((1 + tau_k) g_{k,p}(x0), g_{k,p}(x0) + delta_k)
      (x0 = the warm-start state of the same unit). Powell-Hestenes-Rockafellar augmented Lagrangian with per-unit penalty rho and
      per-(unit, constraint) multipliers; inner solver = the O0 Adam operations (gates projected to [0, 1] after every step) with a
      cosine schedule over the whole refinement; multiplier / penalty update after every inner block. Returned state: the outer
      checkpoint (including x0 itself) with the lowest F among FEASIBLE checkpoints (max_k,p c_{k,p} <= eps_feas); x0 is always
      feasible, so CR never returns a state that violates its own training constraints (fallback = x0).
O0C : compute control: the same continuation schedule, same inner Adam operations, ORIGINAL objective (el1 + ealb); returned state =
      the checkpoint (including x0) with the lowest original training objective.
E2W : unconstrained objective control: same continuation with the WL104K-B E2 weighted sum (el1 + ealb + e_core + e_shoulder);
      returned state = the checkpoint (including x0) with the lowest E2 training objective.
Every arm runs at the fixed shape of its warm-start file (72 units = 9 materials x 8 starts physical, 36 synthetic; start-major).
--warm-r4: replace the units of starts 1-4 by the states of a 4-start file (batch-size sensitivity arm; shape unchanged).
--synthetic: protocol synthetic targets, generated with the identical code / seed as fit104kb (truth + per-lane noise of that set).
Outputs: states.npz (P_y, P_g [+ truth]) and fit.json (fit104kb-compatible fields + constraint / multiplier / checkpoint records).
"""
import argparse
import platform
import time
from pathlib import Path

import numpy as np
import torch

import common104kc as C

KB, CM, FT, GF = C.KB, C.CM, C.FT, C.GF
import fit104kb as FK  # noqa: E402

DEV = C.DEV
NPAR = 178
CORE_DEG, SHOULDER_DEG = FK.CORE_DEG, FK.SHOULDER_DEG
CON = ("el1", "ealb", "e_shoulder", "e_tail")


def cfg():
    """Stage B constants from protocol.json (frozen) or the design defaults (design runs only, recorded)."""
    s = C.P.get("stage_B", {}).get("constants")
    if s:
        return s
    return {"steps": 6000, "inner": 500, "lr0": 3e-4, "lr_end": 3e-5, "warmup": 50,
            "tau": {"el1": 0.02, "ealb": 0.05, "e_shoulder": 0.05, "e_tail": 0.05},
            "delta": {"el1": 0.0, "ealb": 2e-4, "e_shoulder": 0.0, "e_tail": 0.0},
            "rho0": 10.0, "rho_factor": 5.0, "rho_max": 1e4, "violation_progress": 0.25, "eps_feas": 1e-3}


class Terms(FK.Data):
    """fit104kb.Data plus per-probe region terms; identical lane tensors and response operations."""

    def __init__(self, ds, n, R, target=None):
        super().__init__(ds, n, R, "original", target)
        rad = KB.mirror_radius(ds["wi"], ds["wo"])
        g = lambda a: torch.as_tensor(np.ascontiguousarray(a), device=DEV, dtype=torch.float32)
        self.core = g(rad < CORE_DEG).reshape(-1).repeat(R)
        self.shoulder = g((rad >= CORE_DEG) & (rad < SHOULDER_DEG)).reshape(-1).repeat(R)
        self.tail = g(rad >= SHOULDER_DEG).reshape(-1).repeat(R)

    def probe_terms(self, x):
        """x [U, 178] -> dict of per-probe terms [U, P] (float32): el1, ealb, e_core, e_shoulder, e_tail."""
        U = x.shape[0]
        y, gt = x[:, :58], x[:, 58:].reshape(U, 8, 3, 5)
        p = GF.decode(y)
        rep = self.P * self.L
        pp = {k: v[:, None].expand(U, rep, *v.shape[1:]).reshape(-1, *v.shape[1:]) for k, v in p.items()}
        gg = gt[:, None].expand(U, rep, 8, 3, 5).reshape(-1, 8, 3, 5)
        pred = CM.eval_sdm(pp, gg, self.wi, self.wo)
        t = FT.F.group_terms(pred, self.ref, self.w, self.L)
        out = {"el1": t["e1"].reshape(U, self.P), "ealb": t["e2"].reshape(U, self.P)}
        G = U * self.P
        Pv, Fr = pred.view(G, self.L, 3), self.ref.view(G, self.L, 3)
        for name, mask in (("e_core", self.core), ("e_shoulder", self.shoulder), ("e_tail", self.tail)):
            W = (self.w * mask).view(G, self.L, 1)
            out[name] = (((Pv - Fr).abs() * W).sum((1, 2)) / (Fr * W).sum((1, 2)).clamp_min(1e-12)).reshape(U, self.P)
        return out


def original_obj(t):
    return (t["el1"] + t["ealb"]).mean(-1)


def e2_obj(t):
    return (t["el1"] + t["ealb"] + FK.LAMBDA_CORE * t["e_core"] + FK.LAMBDA_SHOULDER * t["e_shoulder"]).mean(-1)


class AL:
    """Constraint bookkeeping for CR (per unit u, constraint k, probe p)."""

    def __init__(self, t0, c):
        self.c = c
        self.F0 = t0["e_core"].mean(-1).detach()                                   # [U]
        self.b = torch.stack([torch.maximum((1 + c["tau"][k]) * t0[k], t0[k] + c["delta"][k]).detach() for k in CON], 1)  # [U, K, P]
        self.lam = torch.zeros_like(self.b)
        self.rho = torch.full_like(self.F0, float(c["rho0"]))
        self.prev_viol = None

    def F(self, t):
        return t["e_core"].mean(-1) / self.F0

    def cviol(self, t):
        g = torch.stack([t[k] for k in CON], 1)
        return g / self.b - 1.0                                                     # [U, K, P], <= 0 feasible

    def lagrangian(self, t):
        cv = self.cviol(t)
        rho = self.rho[:, None, None]
        pen = (torch.clamp(self.lam + rho * cv, min=0.0) ** 2 - self.lam ** 2).sum((1, 2)) / (2 * self.rho)
        return self.F(t) + pen

    def update(self, t):
        """Multiplier and penalty update after an inner block; returns max violation per unit."""
        cv = self.cviol(t).detach()
        self.lam = torch.clamp(self.lam + self.rho[:, None, None] * cv, min=0.0)
        mv = cv.clamp_min(0).amax((1, 2))
        if self.prev_viol is not None:
            slow = mv > self.c["violation_progress"] * self.prev_viol
            slow &= mv > self.c["eps_feas"]
            self.rho = torch.where(slow, torch.clamp(self.rho * self.c["rho_factor"], max=self.c["rho_max"]), self.rho)
        self.prev_viol = mv
        return mv


def run(data, x0, arm, c, log):
    steps, inner = int(c["steps"]), int(c["inner"])
    y = torch.nn.Parameter(x0[:, :58].clone())
    g = torch.nn.Parameter(x0[:, 58:].clone())
    opt = torch.optim.Adam([y, g], lr=c["lr0"])
    with torch.no_grad():
        t0 = data.probe_terms(x0)
    al = AL(t0, c) if arm == "CR" else None
    crit = {"CR": None, "O0C": original_obj, "E2W": e2_obj}[arm]
    U = x0.shape[0]
    # checkpoint 0 = x0
    if arm == "CR":
        best_val = torch.ones(U, device=DEV)                                        # F(x0) = 1, feasible by construction
    else:
        best_val = crit(t0).detach().clone()
    best_x = x0.clone()
    best_ck = torch.zeros(U, dtype=torch.long, device=DEV)
    hist = []
    for step in range(steps):
        lr = FT.TR.lr_at(step, steps, c["lr0"], c["lr_end"], c["warmup"])
        opt.param_groups[0]["lr"] = lr
        x = torch.cat([y, g], 1)
        t = data.probe_terms(x)
        loss = al.lagrangian(t) if arm == "CR" else crit(t)
        if step == 0 and not torch.isfinite(loss).all():
            raise C.Stop(f"NONFINITE_OBJECTIVE {arm} step 0")
        opt.zero_grad(set_to_none=True)
        loss.sum().backward()
        opt.step()
        with torch.no_grad():
            g.clamp_(0.0, 1.0)
        if (step + 1) % inner == 0:
            ck = (step + 1) // inner
            with torch.no_grad():
                xc = torch.cat([y, g], 1)
                tc = data.probe_terms(xc)
                if not all(torch.isfinite(v).all() for v in tc.values()):
                    raise C.Stop(f"NONFINITE_OBJECTIVE {arm} checkpoint {ck}")
                rec = {"step": step + 1, "lr": lr}
                if arm == "CR":
                    Fv = al.F(tc)
                    mv = al.cviol(tc).clamp_min(0).amax((1, 2))
                    feas = mv <= c["eps_feas"]
                    better = feas & (Fv < best_val)
                    al.update(tc)
                    rec.update({"F": Fv.cpu().tolist(), "max_violation": mv.cpu().tolist(), "feasible": feas.cpu().tolist(),
                                "rho": al.rho.cpu().tolist(), "lambda_max": al.lam.amax((1, 2)).cpu().tolist(),
                                "active_constraints": (al.lam > 0).sum((1, 2)).cpu().tolist()})
                else:
                    v = crit(tc)
                    better = v < best_val
                    rec.update({"criterion": v.cpu().tolist()})
                    Fv = v
                best_val = torch.where(better, Fv, best_val)
                best_x = torch.where(better[:, None], xc, best_x)
                best_ck = torch.where(better, torch.full_like(best_ck, ck), best_ck)
                rec["training_terms_mean"] = {k: v.mean(-1).cpu().tolist() for k, v in tc.items()}
                hist.append(rec)
            log(f"{arm} ck {ck} step {step + 1} median {'F' if arm == 'CR' else 'crit'} {float(Fv.median()):.5f}"
                + (f" feasible {float(feas.float().mean()):.3f} max_viol_med {float(mv.median()):.2e}" if arm == "CR" else ""))
    # final KKT-type record for CR at the returned state (projected gradient of the Lagrangian with the final multipliers)
    kkt = None
    if arm == "CR":
        xr = best_x.clone().requires_grad_(True)
        tr = data.probe_terms(xr)
        Lr = al.lagrangian(tr)
        (gr,) = torch.autograd.grad(Lr.sum(), xr)
        fm = FK.free_mask(best_x, gr)
        kkt = {"projected_lagrangian_gradient_norm": (gr * fm).norm(dim=-1).detach().cpu().tolist()}
    return best_x.detach(), best_ck.cpu().tolist(), hist, kkt, (al.b.cpu().numpy() if al is not None else None)


def synthetic_targets(ds, raw):
    """fit104kb.main synthetic target construction (identical code and seed)."""
    truth = KB.synthetic_truth(raw["R0"])
    tgt = CM.predict("P", truth, ds["wi"], ds["wo"])
    sigma = 0.5 * np.abs(ds["a"] - ds["b"])
    target = tgt + sigma * np.random.default_rng(KB.P["seeds"]["synthetic_noise"]).standard_normal(tgt.shape)
    return truth, target


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("out", type=Path)
    ap.add_argument("--arm", choices=("CR", "O0C", "E2W"), required=True)
    ap.add_argument("--warm", type=Path, required=True)
    ap.add_argument("--warm-r4", type=Path, default=None)
    ap.add_argument("--starts", type=int, choices=(4, 8), required=True)
    ap.add_argument("--data", choices=("original", "coverage"), required=True)
    ap.add_argument("--data-dir", type=Path, default=C.KB_ROOT / "data")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--steps", type=int, default=None, help="override (design / smoke only; recorded)")
    ap.add_argument("--inner", type=int, default=None, help="override (smoke only; recorded)")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    c = dict(cfg())
    if args.steps is not None:
        c["steps"] = args.steps
    if args.inner is not None:
        c["inner"] = args.inner
    t_all = time.monotonic()
    rows, raw, _, _, _, seen = CM.inputs()
    n = len(rows)
    ds = KB.training_set(args.data, n, args.data_dir)
    keys = KB.START_KEYS[:args.starts]
    R = len(keys)
    truth, target = (synthetic_targets(ds, raw) if args.synthetic else (None, None))
    data = Terms(ds, n, R, target)
    warm = C.load_states(args.warm, R, n)
    warm_sha = {"warm": C.sha(args.warm)}
    if args.warm_r4 is not None:
        r4 = C.load_states(args.warm_r4, 4, n)
        warm = r4 + warm[4:]
        warm_sha["warm_r4"] = C.sha(args.warm_r4)
    x0 = torch.as_tensor(C.flat_x(warm), device=DEV, dtype=torch.float32)
    assert x0.shape == (R * n, NPAR)
    lines = []

    def log(s):
        lines.append(s)
        print(s, flush=True)
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    t0 = time.monotonic()
    x, ck, hist, kkt, bounds = run(data, x0, args.arm, c, log)
    torch.cuda.synchronize()
    secs = time.monotonic() - t0
    with torch.no_grad():
        fin = {k: v.cpu().numpy().tolist() for k, v in data.probe_terms(x).items()}
        ini = {k: v.cpu().numpy().tolist() for k, v in data.probe_terms(x0).items()}
    xn = x.cpu().numpy().astype(np.float64)
    out = {"P_y": xn[:, :58], "P_g": xn[:, 58:].reshape(-1, 8, 3, 5)}
    if truth is not None:
        out.update({"truth_y": truth["y"], "truth_g": truth["g"]})
    np.savez(args.out / "states.npz", **out)
    if bounds is not None:
        np.savez(args.out / "bounds.npz", bounds=bounds, constraints=np.array(CON))
    steps = int(c["steps"])
    fj = {"arm": {"method": f"continuation_{args.arm}", "data": args.data, "objective": {"CR": "constrained_core",
                                                                                       "O0C": "original", "E2W": "e2"}[args.arm],
                  "synthetic": args.synthetic},
          "start_keys": keys, "materials": [r["tag"] for r in rows], "budget": steps, "budget_respected": True,
          "evaluations_per_unit": [steps] * (R * n), "parameters_per_material": NPAR, "constants": c,
          "warm_start_sha256": warm_sha, "warm_r4": args.warm_r4 is not None, "seconds": secs,
          "seconds_total": time.monotonic() - t_all, "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
          "returned_checkpoint": ck, "checkpoint_history": hist, "kkt": kkt, "initial_probe_terms": ini, "final_probe_terms": fin,
          "data_dir": str(args.data_dir.resolve()) if args.data == "coverage" else None,
          "data_sha256": (C.sha(args.data_dir / "train_coverage.npz") if args.data == "coverage"
                          else seen.get("results/ecro_wl84_104j/primary/data/fit_set.npz")),
          "commit": C.commit(), "protocol_sha256": C.sha(C.PROTOCOL) if C.PROTOCOL.exists() else None, "inputs_sha256": seen,
          "host": platform.platform(), "torch": torch.__version__, "gpu": torch.cuda.get_device_name(),
          "gpu_cap": "owner 50 % cap: gpu_duty.ps1 suspension + allocator fraction 0.5 (wall seconds include suspension)",
          "log": lines, "states_sha256": C.sha(args.out / "states.npz")}
    C.write(args.out / "fit.json", fj)
    print("fit done", args.arm, round(secs, 1), "s", flush=True)


if __name__ == "__main__":
    main()
