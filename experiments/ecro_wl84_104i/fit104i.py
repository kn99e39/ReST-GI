"""WL104I Stage D: matched original-objective fits, A1 (original 64) vs B1 (ER-v1-GF 58), identical physical starts.

Usage: python fit104i.py <out_dir> [--smoke]
"""
import argparse
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch

import common104i as CM

GF, CL, IO = CM.GF, CM.CL, CM.IO
DEV = CM.DEV
sys.path.insert(0, str(CM.HERE.parent.parent / "experiments/fnto_compiled82"))
sys.path.insert(0, str(CM.HERE.parent.parent / "experiments/fnto_compiled80c"))
import fit82 as F  # noqa: E402
import train80c as TR  # noqa: E402

OPT = F.OPT                      # WL82 direct-fit optimiser: lr 0.03 -> 3e-4 cosine, 50 warm-up
STEPS = 12000
LOG = 100


class FitData:
    def __init__(self, arrays, reps):
        self.n, self.groups, self.lanes, _ = arrays["f"].shape
        g = {k: torch.as_tensor(arrays[k], device=DEV, dtype=torch.float32) for k in ("wi", "wo", "use", "q", "f")}
        self.reps = reps
        self.wi = g["wi"].repeat(reps, 1, 1, 1).reshape(-1, 3)
        self.wo = g["wo"].repeat(reps, 1, 1, 1).reshape(-1, 3)
        self.ref = g["f"].repeat(reps, 1, 1, 1).reshape(-1, 3)
        self.w = (g["use"] / g["q"]).repeat(reps, 1, 1).reshape(-1)

    def score(self, raw, decode):
        p = decode(raw)
        L = self.groups * self.lanes
        gathered = {k: v[:, None].expand(len(v), L, *v.shape[1:]).reshape(-1, *v.shape[1:]) for k, v in p.items()}
        pred = CL.eval_closure("ER", gathered, self.wi, self.wo)
        g = F.group_terms(pred, self.ref, self.w, self.lanes)["metric"]
        return g.reshape(self.reps, self.n, self.groups).mean(-1)


DECODE = {"A1": lambda z: CL.decode("ER", "cert", z), "B1": GF.decode}


def train(data, start, arm, steps):
    t = torch.nn.Parameter(torch.tensor(start.reshape(-1, start.shape[-1]), device=DEV, dtype=torch.float32))
    opt = torch.optim.Adam([t], lr=OPT["lr"])
    hist = {}
    for step in range(steps + 1):
        sc = data.score(t, DECODE[arm])
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
    return t.detach().cpu().numpy().reshape(start.shape), hist


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("out", type=Path)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    rows, raw, _, seen = CM.inputs()
    with np.load(CM.TRAIN) as z:
        arrays = {k: z[k] for k in ("wi", "wo", "use", "q", "f")}
    starts_o = CM.training_starts(len(rows))
    keys = list(starts_o)
    S_o = np.stack([starts_o[k] for k in keys])                          # [4, 9, 64]
    S_g = np.stack([GF.from_original(starts_o[k]) for k in keys])        # [4, 9, 58]  identical physical starts
    data = FitData(arrays, len(keys))
    steps = 20 if args.smoke else STEPS
    out = {"start_keys": np.array(keys), "A1_start": S_o, "B1_start": S_g, "R0": raw["R0"]}
    report = {}
    with torch.no_grad():
        s0o = data.score(torch.as_tensor(S_o.reshape(-1, 64), device=DEV, dtype=torch.float32), DECODE["A1"]).cpu().numpy()
        s0g = data.score(torch.as_tensor(S_g.reshape(-1, 58), device=DEV, dtype=torch.float32), DECODE["B1"]).cpu().numpy()
        one = FitData(arrays, 1)
        r0 = one.score(torch.as_tensor(raw["R0"], device=DEV, dtype=torch.float32), DECODE["A1"]).cpu().numpy()[0]
    report["start_objective_gap_max_rel"] = float(np.abs(s0o - s0g).max() / np.abs(s0o).max())
    report["R0_objective"] = r0.tolist()
    for arm, S in (("A1", S_o), ("B1", S_g)):
        a, _ = train(data, S, arm, 2)
        b, _ = train(data, S, arm, 2)
        replay = float(np.abs(a - b).max())
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.monotonic()
        term, hist = train(data, S, arm, steps)
        torch.cuda.synchronize()
        secs = time.monotonic() - t0
        out[f"{arm}_terminal"] = term
        for i, k in enumerate(keys):
            out[f"{arm}_{k}"] = term[i]
        H = np.stack([hist[s] for s in sorted(hist)])                    # [T, 4, 9]
        final = H[-1]
        rel = (H - final[None]) / np.abs(final)[None]
        steps_logged = np.array(sorted(hist))
        # steps to within 1 % of final: first logged step after which the objective stays within 1 % of its final value
        stw = np.zeros(H.shape[1:], int)
        for i in range(H.shape[1]):
            for m in range(H.shape[2]):
                ok = np.abs(rel[:, i, m]) <= 0.01
                tail_ok = np.flip(np.cumprod(np.flip(ok)))
                stw[i, m] = int(steps_logged[np.argmax(tail_ok)]) if tail_ok.any() else int(steps_logged[-1])
        report[arm] = {"seconds": secs, "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                       "two_step_replay_gap": replay, "final_objective": final.tolist(), "steps_to_within_1pct": stw.tolist(),
                       "history_steps": steps_logged.tolist(), "history": H.tolist(),
                       "last_2000_change": (np.abs(H[-1] - H[-1 - 2000 // LOG]).tolist() if len(H) > 2000 // LOG else None),
                       "n_params": 64 if arm == "A1" else 58}
        print(arm, "seconds", round(secs, 1), "median final", float(np.median(final)), flush=True)
    np.savez(args.out / "fit_states.npz", **out)
    IO.write(args.out / "fit.json", {"start_keys": keys, "materials": [r["tag"] for r in rows], "arms": report,
                                     "optimizer": {**OPT, "steps": steps, "log_interval": LOG, "dtype": "float32"},
                                     "commit": IO.commit(), "protocol_sha256": IO.sha(CM.HERE / "protocol.json"),
                                     "inputs_sha256": seen, "host": platform.platform(), "torch": torch.__version__,
                                     "gpu": torch.cuda.get_device_name(), "fit_states_sha256": IO.sha(args.out / "fit_states.npz")})


if __name__ == "__main__":
    main()
