"""WL104J Stages 1-3: matched fitting of A (shared ER-v1-GF), B (amplitude-local), C (full-local-lobe) and C_noF (secondary).

Usage:
  python fit104j.py primary   <data_dir> <out_dir> [--steps N]      A (4 starts) -> B -> C (+ fallback) -> C_noF
  python fit104j.py extended  <data_dir> <out_dir> [--steps N]      generic lineage, 2x steps: A -> B -> C
  python fit104j.py synthetic <data_dir> <out_dir> --control S0a|S0b|S1|S2 [--steps N]
"""
import argparse
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch

import common104j as CM

GF, CL, IO = CM.GF, CM.CL, CM.IO
DEV = CM.DEV
sys.path.insert(0, str(CM.HERE.parent / "fnto_compiled82"))
sys.path.insert(0, str(CM.HERE.parent / "fnto_compiled80c"))
import fit82 as F  # noqa: E402
import train80c as TR  # noqa: E402

OPT = F.OPT
LOG = 100
TRAINABLE = {"A": ("y",), "B": ("amp",), "C": ("amp", "shape_raw"), "C_noF": ("amp", "shape_raw")}
F0_COLS = np.arange(2, 21, 3)


class FitData:
    """Fit lanes for R replicas x n materials x P probes x L lanes (rep-major)."""

    def __init__(self, d, n, R, target=None):
        self.n, self.R = n, R
        self.L = d["wi"].shape[1]
        self.P = d["wi"].shape[0] // n
        f = 0.5 * (d["a"] + d["b"]) if target is None else target
        g = lambda a: torch.as_tensor(np.ascontiguousarray(a), device=DEV, dtype=torch.float32)
        self.wi = g(d["wi"]).reshape(-1, 3).repeat(R, 1)
        self.wo = g(d["wo"]).reshape(-1, 3).repeat(R, 1)
        self.ref = g(f).reshape(-1, 3).repeat(R, 1)
        self.w = g(d["use"] / d["q"]).reshape(-1).repeat(R)

    def score(self, arm, st):
        p = CM.lane_params(arm, st, torch.float32)
        pp = CM.to_lanes(CM.per_probe(p, arm, self.R * self.n, CM.PROBE_GROUP), self.L)
        pred = CL.eval_closure("ER", pp, self.wi, self.wo)
        g = F.group_terms(pred, self.ref, self.w, self.L)["metric"]
        return g.reshape(self.R, self.n, self.P).mean(-1)


def tensors(st):
    return {k: torch.as_tensor(np.asarray(v), device=DEV, dtype=torch.float32) for k, v in st.items()}


def objective(data, arm, st):
    with torch.no_grad():
        return data.score(arm, tensors(st)).cpu().numpy()


def train(data, arm, init, steps):
    """init: dict of numpy arrays with leading dim R * n. Returns terminal (numpy) and history {step: [R, n]}."""
    st = tensors(init)
    params = {k: torch.nn.Parameter(st[k].clone()) for k in TRAINABLE[arm]}
    if arm == "C_noF":
        mask = torch.ones_like(params["shape_raw"])
        mask[..., F0_COLS] = 0
        params["shape_raw"].register_hook(lambda g: g * mask)
    opt = torch.optim.Adam(list(params.values()), lr=OPT["lr"])
    hist = {}
    for step in range(steps + 1):
        cur = {**st, **params}
        sc = data.score(arm, cur)
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
    term = {k: (params[k].detach().cpu().numpy().astype(np.float64) if k in params else np.asarray(v, np.float64))
            for k, v in init.items()}
    return term, hist


def history_report(hist, init_obj):
    steps = np.array(sorted(hist))
    H = np.stack([hist[s] for s in steps])                     # [T, R, n]
    final = H[-1]
    k = 2000 // LOG
    last = np.abs(H[-1] - H[-1 - k]) / np.abs(final) if len(H) > k else None
    return {"history_steps": steps.tolist(), "history": H.tolist(), "final_objective": final.tolist(),
            "initial_objective": init_obj.tolist(),
            "last_2000_relative_change": None if last is None else last.tolist()}


def run_arm(data, arm, init, steps, log):
    init_obj = objective(data, arm, init)
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    t0 = time.monotonic()
    term, hist = train(data, arm, init, steps)
    torch.cuda.synchronize()
    secs = time.monotonic() - t0
    rep = history_report(hist, init_obj)
    rep.update({"seconds": secs, "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()), "steps": steps})
    log[arm] = rep
    print(arm, "seconds", round(secs, 1), "median final", float(np.median(rep["final_objective"])), flush=True)
    return term, np.asarray(rep["final_objective"])


def local_from_A(yA, n_groups=CM.G):
    return CM.expand_groups(CM.section(yA), n_groups)


def chain(data, yA0, steps, log, with_cnof=True, with_fallback=True):
    """A from yA0 [R*n, 58] -> B -> C (fallback) [-> C_noF]. Returns states dict."""
    states = {}
    tA, objA = run_arm(data, "A", {"y": yA0}, steps, log)
    states["A"] = tA
    initB = local_from_A(tA["y"])
    log["B_init_vs_A_final_max_relative"] = float(np.max(np.abs(objective(data, "B", initB) - objA) / np.abs(objA)))
    tB, objB = run_arm(data, "B", initB, steps, log)
    states["B"] = tB
    initC = {**tB, "shape_raw": initB["shape_raw"]}
    log["C_init_vs_B_final_max_relative"] = float(np.max(np.abs(objective(data, "C", initC) - objB) / np.abs(objB)))
    tC, objC = run_arm(data, "C", initC, steps, log)
    trig = objC > objB * (1 + 1e-6)
    log["C_fallback_triggered"] = trig.tolist()
    if with_fallback and trig.any():
        sub = {}
        tF, objF = run_arm(data, "C", {**initB}, steps, sub)
        log["C_fallback_run"] = sub["C"]
        use = trig & (objF < objC)
        log["C_fallback_used"] = use.tolist()
        R, n = objC.shape
        u = use.reshape(-1)
        for k in tC:
            tC[k] = np.where(u.reshape((-1,) + (1,) * (tC[k].ndim - 1)), tF[k], tC[k])
    states["C"] = tC
    if with_cnof:
        initN = {**tB, "shape_raw": initB["shape_raw"]}
        tN, _ = run_arm(data, "C_noF", initN, steps, log)
        if np.abs(tN["shape_raw"][..., F0_COLS] - initN["shape_raw"][..., F0_COLS]).max() != 0:
            raise IO.Stop("C_noF_F0_MASK_LEAK")
        states["C_noF"] = tN
    return states


def synthetic_truth(control, y_star):
    """Known truth (float64): shared y* (S0) or per-group local arrays (S1, S2)."""
    loc = local_from_A(y_star)
    mu = np.cos(np.radians(CM.ANGLES))
    u = 2 * (mu - mu.min()) / (mu.max() - mu.min()) - 1               # [G]
    if control in ("S0a", "S0b"):
        return {"arm": "A", "state": {"y": y_star}}
    if control == "S1":
        d = np.zeros((CM.G, 3, 8))
        d[:, :, 1:4] = 0.5 * u[:, None, None]
        d[:, :, 0] = -0.5 * u[:, None]
        d[:, :, 4:8] = -0.5 * u[:, None, None]
        loc["amp"] = loc["amp"] + d[None]
        return {"arm": "B", "state": loc}
    if control == "S2":
        d = np.zeros((CM.G, 21))
        for k in range(3):
            d[:, 3 * k] = 0.5 * u
            d[:, 3 * k + 1] = 0.5 * u
        loc["shape_raw"] = loc["shape_raw"] + d[None]
        return {"arm": "C", "state": loc}
    raise ValueError(control)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=("primary", "extended", "synthetic"))
    ap.add_argument("data", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--control")
    ap.add_argument("--steps", type=int, default=12000)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    rows, raw, _, desc, _, _, seen = CM.inputs()
    dj = IO.read(args.data / "data.json")
    for f, h in dj["outputs_sha256"].items():
        if CM.sha(args.data / f) != h:
            raise IO.Stop(f"DATA_HASH {f}")
    if not dj["qualified"]["all"]:
        raise IO.Stop("DATA_UNQUALIFIED")
    with np.load(args.data / "fit_set.npz") as z:
        d = {k: z[k] for k in z.files}
    n = len(rows)
    log = {"mode": args.mode, "control": args.control, "steps": args.steps}
    starts_o = CM.CI.training_starts(n)
    keys = list(starts_o)
    out = {}
    if args.mode == "primary":
        R = len(keys)
        yA0 = np.concatenate([GF.from_original(starts_o[k]) for k in keys])
        data = FitData(d, n, R)
        states = chain(data, yA0, args.steps, log)
        log["start_keys"] = keys
    elif args.mode == "extended":
        R = 1
        data = FitData(d, n, R)
        states = chain(data, GF.from_original(starts_o["generic"]), 2 * args.steps, log, with_cnof=False, with_fallback=False)
        log["start_keys"] = ["generic"]
    else:
        R = 1
        y_star = GF.from_original(raw["R0"])
        truth = synthetic_truth(args.control, y_star)
        tgt = CM.predict(truth["arm"], truth["state"], d["wi"].reshape(n, CM.NP, -1, 3), d["wo"].reshape(n, CM.NP, -1, 3),
                         CM.PROBE_GROUP)
        sigma = 0.5 * np.abs(d["a"] - d["b"]).reshape(tgt.shape)
        noisy = tgt + sigma * np.random.default_rng(CM.P["seeds"]["synthetic_noise"]).standard_normal(tgt.shape)
        data = FitData(d, n, R, target=noisy.reshape(-1, d["wi"].shape[1], 3))
        if args.control == "S0b":
            yA0 = GF.from_original(starts_o["generic"])
        else:
            yA0 = y_star + 0.1 * np.random.default_rng(CM.P["seeds"]["synthetic_start"]).standard_normal(y_star.shape)
        states = chain(data, yA0, args.steps, log, with_cnof=False, with_fallback=False)
        log["truth_arm"] = truth["arm"]
        log["truth_objective_on_noisy_fit_target"] = objective(data, truth["arm"], truth["state"]).tolist()
        for k, v in truth["state"].items():
            out[f"truth_{k}"] = v
        log["start_keys"] = ["synthetic"]
    for arm, st in states.items():
        for k, v in st.items():
            out[f"{arm}_{k}"] = v
    np.savez(args.out / "states.npz", **out)
    log.update({"materials": [r["tag"] for r in rows], "optimizer": {**OPT, "log_interval": LOG, "dtype": "float32"},
                "commit": CM.commit(), "protocol_sha256": CM.sha(CM.HERE / "protocol.json"), "inputs_sha256": seen,
                "data_sha256": dj["outputs_sha256"], "host": platform.platform(), "torch": torch.__version__,
                "gpu": torch.cuda.get_device_name(), "states_sha256": CM.sha(args.out / "states.npz"),
                "trainable_per_material": {"A": 58, "B": 24 * CM.G, "C": 45 * CM.G, "C_noF": 38 * CM.G}})
    CM.write(args.out / "fit.json", log)
    print("fit done", args.mode, args.control, flush=True)


if __name__ == "__main__":
    main()
