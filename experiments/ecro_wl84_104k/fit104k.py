"""WL104K Stage D: matched JOINT fitting of A (shared ER-v1-GF), T (shared gate) and P (per-component gates).

Usage:
  python fit104k.py primary   <out_dir> [--steps N]     A, T, P from the 4 matched physical starts (gates at identity)
  python fit104k.py extended  <out_dir> [--steps N]     generic start, 2x steps, A, T, P
  python fit104k.py synthetic <out_dir> --control S0|ST|SP [--steps N]
"""
import argparse
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch

import common104k as CM

GF, CL, IO = CM.GF, CM.CL, CM.IO
DEV = CM.DEV
sys.path.insert(0, str(CM.HERE.parent / "fnto_compiled82"))
sys.path.insert(0, str(CM.HERE.parent / "fnto_compiled80c"))
import fit82 as F  # noqa: E402
import train80c as TR  # noqa: E402

OPT = F.OPT
LOG = 100
TRAINABLE = {"A": ("y",), "T": ("y", "g"), "P": ("y", "g")}


class FitData:
    def __init__(self, fit, n, R, target=None):
        self.n, self.R = n, R
        self.P, self.L = fit["wi"].shape[1:3]
        g = lambda a: torch.as_tensor(np.ascontiguousarray(a), device=DEV, dtype=torch.float32)
        self.wi = g(fit["wi"]).reshape(-1, 3).repeat(R, 1)
        self.wo = g(fit["wo"]).reshape(-1, 3).repeat(R, 1)
        self.ref = g(fit["mean"] if target is None else target).reshape(-1, 3).repeat(R, 1)
        self.w = g(fit["w"]).reshape(-1).repeat(R)

    def score(self, arm, st):
        p, gt = CM.decode(arm, st, torch.float32)
        rep = self.P * self.L
        pp = {k: v[:, None].expand(v.shape[0], rep, *v.shape[1:]).reshape(-1, *v.shape[1:]) for k, v in p.items()}
        gg = None if gt is None else gt[:, None].expand(gt.shape[0], rep, *gt.shape[1:]).reshape(-1, *gt.shape[1:])
        pred = CM.lane_eval(arm, pp, gg, self.wi, self.wo)
        m = F.group_terms(pred, self.ref, self.w, self.L)["metric"]
        return m.reshape(self.R, self.n, self.P).mean(-1)


def tensors(st):
    return {k: torch.as_tensor(np.asarray(v), device=DEV, dtype=torch.float32) for k, v in st.items()}


def objective(data, arm, st):
    with torch.no_grad():
        return data.score(arm, tensors(st)).cpu().numpy()


def train(data, arm, init, steps):
    st = tensors(init)
    params = {k: torch.nn.Parameter(st[k].clone()) for k in TRAINABLE[arm]}
    opt = torch.optim.Adam(list(params.values()), lr=OPT["lr"])
    hist = {}
    for step in range(steps + 1):
        sc = data.score(arm, {**st, **params})
        if step % LOG == 0 or step == steps:
            v = sc.detach().cpu().numpy()
            if not np.isfinite(v).all():
                raise IO.Stop(f"NONFINITE_OBJECTIVE {arm} {step}")
            hist[step] = v
        if step < steps:
            opt.param_groups[0]["lr"] = TR.lr_at(step, steps, OPT["lr"], OPT["lr_end"], OPT["warmup"])
            opt.zero_grad(set_to_none=True)
            sc.sum().backward()
            opt.step()
            if "g" in params:
                with torch.no_grad():
                    params["g"].clamp_(0.0, 1.0)                                # projection onto the feasible box
    term = {k: (params[k].detach().cpu().numpy().astype(np.float64) if k in params else np.asarray(v, np.float64))
            for k, v in init.items()}
    return term, hist


def run_arm(data, arm, init, steps, log):
    init_obj = objective(data, arm, init)
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    t0 = time.monotonic()
    term, hist = train(data, arm, init, steps)
    torch.cuda.synchronize()
    secs = time.monotonic() - t0
    st = np.array(sorted(hist))
    H = np.stack([hist[s] for s in st])
    k = 2000 // LOG
    rep = {"history_steps": st.tolist(), "history": H.tolist(), "initial_objective": init_obj.tolist(),
           "final_objective": H[-1].tolist(), "seconds": secs, "steps": steps, "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
           "last_2000_relative_change": (np.abs(H[-1] - H[-1 - k]) / np.abs(H[-1])).tolist() if len(H) > k else None}
    if arm != "A":
        g = term["g"]
        rep["gate_stats"] = {"min": float(g.min()), "fraction_at_1": float((g >= 1 - 1e-7).mean()),
                             "fraction_at_0": float((g <= 1e-7).mean()), "max_abs_move_from_identity": float(np.abs(1 - g).max())}
    log[arm] = rep
    print(arm, "seconds", round(secs, 1), "median final", float(np.median(H[-1])), flush=True)
    return term


def starts_y(n, keys):
    so = CM.J.CI.training_starts(n)
    return np.concatenate([GF.from_original(so[k]) for k in keys])


SYN = {"ST": {"shared": [0.45, 0.6, 0.8, 0.95, 1.0]},
       "SP": {"O0": [0.4, 0.55, 0.8, 1, 1], "O1": [1, 1, 0.85, 0.6, 0.45], "O2": [1, 1, 1, 1, 1], "L": [1, 0.8, 0.6, 0.5, 0.5],
              "I0": [0.5, 0.7, 0.9, 1, 1], "I1": [1, 0.9, 0.7, 0.6, 0.5], "I2": [0.6, 1, 0.6, 1, 0.6], "I3": [1, 1, 1, 1, 1]}}


def synthetic_truth(control, y_star):
    n = len(y_star)
    if control == "S0":
        return "A", {"y": y_star}
    if control == "ST":
        g = np.broadcast_to(np.array(SYN["ST"]["shared"], float), (n, 1, 3, 5)).copy()
        return "T", {"y": y_star, "g": g}
    g = np.stack([np.broadcast_to(np.array(SYN["SP"][c], float), (3, 5)) for c in CM.COMP])     # [8, 3, 5]
    return "P", {"y": y_star, "g": np.broadcast_to(g, (n,) + g.shape).copy()}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=("primary", "extended", "synthetic"))
    ap.add_argument("out", type=Path)
    ap.add_argument("--control")
    ap.add_argument("--steps", type=int, default=12000)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    rows, raw, _, _, _, seen = CM.inputs()
    n = len(rows)
    fit = CM.load_set("fit_set", n)
    log = {"mode": args.mode, "control": args.control}
    out = {}
    if args.mode in ("primary", "extended"):
        keys = CM.STARTS if args.mode == "primary" else ["generic"]
        steps = args.steps if args.mode == "primary" else 2 * args.steps
        R = len(keys)
        data = FitData(fit, n, R)
        y0 = starts_y(n, keys)
        for arm in CM.ARMS:
            init = {"y": y0} if arm == "A" else {"y": y0, "g": CM.identity_gates(R * n, arm)}
            if arm != "A":
                log[f"{arm}_identity_init_vs_A_init_max_relative"] = float(np.max(np.abs(objective(data, arm, init) - objective(data, "A", {"y": y0}))
                                                                                  / np.abs(objective(data, "A", {"y": y0}))))
            term = run_arm(data, arm, init, steps, log)
            for k, v in term.items():
                out[f"{arm}_{k}"] = v
        log["start_keys"] = keys
    else:
        y_star = GF.from_original(raw["R0"])
        tarm, truth = synthetic_truth(args.control, y_star)
        tgt = CM.predict(tarm, truth, fit["wi"], fit["wo"])
        sigma = 0.5 * np.abs(fit["a"] - fit["b"])
        noisy = tgt + sigma * np.random.default_rng(CM.P["seeds"]["synthetic_noise"]).standard_normal(tgt.shape)
        data = FitData(fit, n, 1, target=noisy)
        y0 = y_star + 0.1 * np.random.default_rng(CM.P["seeds"]["synthetic_start"]).standard_normal(y_star.shape)
        for arm in CM.ARMS:
            init = {"y": y0} if arm == "A" else {"y": y0, "g": CM.identity_gates(n, arm)}
            term = run_arm(data, arm, init, args.steps, log)
            for k, v in term.items():
                out[f"{arm}_{k}"] = v
        for k, v in truth.items():
            out[f"truth_{k}"] = v
        log["truth_arm"] = tarm
        log["truth_objective_on_noisy_target"] = objective(data, tarm, truth).tolist()
        log["start_keys"] = ["synthetic"]
    np.savez(args.out / "states.npz", **out)
    log.update({"materials": [r["tag"] for r in rows], "optimizer": {**OPT, "log_interval": LOG, "dtype": "float32",
                                                                     "gate_projection": "[0, 1] after every step"},
                "commit": CM.commit(), "protocol_sha256": CM.sha(CM.HERE / "protocol.json"), "inputs_sha256": seen,
                "host": platform.platform(), "torch": torch.__version__, "gpu": torch.cuda.get_device_name(),
                "states_sha256": CM.sha(args.out / "states.npz"), "trainable_per_material": {"A": 58, "T": 73, "P": 178}})
    CM.write(args.out / "fit.json", log)
    print("fit done", args.mode, args.control, flush=True)


if __name__ == "__main__":
    main()
