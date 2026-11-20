"""Fit matched WL104G arms on the archived original per-material data."""

import argparse
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from qualify104g import C, H, HERE, P, frozen_inputs, read, sha_ok, write
from common104d import F, REPO, historical_states

PARENT = REPO / "evaluation/104D"
FROZEN_FIT = REPO / "results/ecro_wl84_104f/inputs/train_original.npz"
ANGULAR = REPO / "results/ecro_wl84_104d/amendment1/amended_reference.npz"
WL83_FITS = REPO / "results/fnto_compiled83/results"
STAGE_A = "05f806f"
CFG = read(HERE.parent / "ecro_wl84_104f/protocol.json")["parts_bc"]["optimizer"]


def check_input(path, digest, inputs):
    sha_ok(path, digest)
    inputs[path.relative_to(REPO).as_posix()] = digest


def archived():
    provenance, rows, descriptors, states = frozen_inputs()
    inputs = {}
    paths = {
        FROZEN_FIT: provenance["parent_output_sha256"]["train_original.npz"],
        PARENT / "fit_states.npz": provenance["parent_output_sha256"]["fit_states.npz"],
        ANGULAR: provenance["output_sha256"]["amended_reference.npz"],
        PARENT / "amendment1/compiled_states.npz": provenance["output_sha256"]["compiled_states.npz"],
        PARENT / "amendment1/materials.json": provenance["output_sha256"]["materials.json"],
    }
    for path, digest in paths.items():
        check_input(path, digest, inputs)
    for name, digest in provenance["parent_source_sha256"].items():
        if name.startswith("fit/fit_"):
            check_input(WL83_FITS / name, digest, inputs)
        elif name == "data/materials83.npz":
            check_input(C.W83 / name, digest, inputs)
    with np.load(PARENT / "fit_states.npz") as z:
        starts = np.stack([z[arm] for arm in ("A", "C")])
    r0 = historical_states(WL83_FITS, rows)["ER_DF"]
    if starts.shape != (2, len(rows), 64) or r0.shape != (len(rows), 64):
        raise ValueError("Frozen fit layout differs")
    for name, raw in (("A", starts[0]), ("C", starts[1]), ("ER_DF", r0)):
        compiled = compile_states(raw)
        if not np.allclose(compiled, states[name], atol=2e-6, rtol=1e-5):
            raise ValueError(f"Compiled frozen state {name} differs from archived state")
    return rows, descriptors, states, starts, r0, inputs


def compile_states(raw):
    with torch.no_grad():
        z = torch.as_tensor(raw, device=H.dev, dtype=torch.float32)
        return H.CD.compiled_state("ER", H.CL.decode("ER", "cert", z))


class FitData:
    def __init__(self, arrays, starts):
        self.n, self.groups, self.lanes, channels = arrays["f"].shape
        if (self.groups, self.lanes, channels) != (64, 64, 3):
            raise ValueError("Original angular layout differs")
        gpu = {k: torch.as_tensor(arrays[k], device=H.dev, dtype=torch.float32)
               for k in ("wi", "wo", "use", "q", "f")}
        self.wi = gpu["wi"].repeat(starts, 1, 1, 1).reshape(-1, 3)
        self.wo = gpu["wo"].repeat(starts, 1, 1, 1).reshape(-1, 3)
        self.ref = gpu["f"].repeat(starts, 1, 1, 1).reshape(-1, 3)
        self.weight = (gpu["use"] / gpu["q"]).repeat(starts, 1, 1).reshape(-1)
        truth = self.ref.reshape(-1, self.lanes, 3)
        w = self.weight.reshape(-1, self.lanes, 1)
        self.squared_den = (w * truth.square()).sum((1, 2))
        if not bool((torch.isfinite(self.squared_den) & (self.squared_den > 0)).all()):
            raise ValueError("F-squared normalization is nonpositive or invalid")

    def score(self, raw, arm):
        params = H.CL.decode("ER", "cert", raw)
        total_lanes = self.groups * self.lanes
        gathered = {k: p[:, None].expand(len(p), total_lanes, *p.shape[1:]).reshape(-1, *p.shape[1:])
                    for k, p in params.items()}
        pred = H.CL.eval_closure("ER", gathered, self.wi, self.wo)
        if arm == "R1":
            group = F.group_terms(pred, self.ref, self.weight, self.lanes)["metric"]
        else:
            error = (pred - self.ref).reshape(-1, self.lanes, 3)
            group = (error.square() * self.weight.reshape(-1, self.lanes, 1)).sum((1, 2)) / self.squared_den
        return group.reshape(-1, self.n, self.groups).mean(dim=-1)


def train(data, initial, arm, steps, emit):
    tensor = torch.nn.Parameter(torch.tensor(initial.reshape(-1, 64), device=H.dev, dtype=torch.float32))
    opt = torch.optim.Adam([tensor], lr=CFG["learning_rate"])
    progress = {}
    for step in range(steps + 1):
        score = data.score(tensor, arm)
        if step % CFG["log_interval"] == 0 or step == steps:
            values = score.detach().cpu().numpy()
            if not np.isfinite(values).all():
                raise ValueError(f"Invalid {arm} score at step {step}")
            progress[step] = values
            if emit:
                print(arm, step, "median objective per start", np.median(values, axis=1), flush=True)
        if step < steps:
            opt.param_groups[0]["lr"] = F.TR.lr_at(step, steps, CFG["learning_rate"], CFG["learning_rate_end"], CFG["warmup"])
            opt.zero_grad(set_to_none=True)
            score.sum().backward()
            opt.step()
    return tensor.detach().cpu().numpy().reshape(initial.shape), progress


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operator", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    operator = read(args.operator)
    if not operator["passed"] or not operator["commit"].startswith(STAGE_A) or operator["protocol_sha256"] != C.sha(HERE / "protocol.json"):
        raise ValueError("Frozen operator qualification is required")
    rows, _, _, starts, base, inputs = archived()
    with np.load(FROZEN_FIT) as z:
        arrays = {k: z[k] for k in ("wi", "wo", "use", "q", "f")}
    fit_data = FitData(arrays, 2)
    baseline_data = FitData(arrays, 1)
    args.output.mkdir(parents=True, exist_ok=False)
    output = {"R0": base, "starts": starts}
    reports = {}
    for arm in ("R1", "R2"):
        with torch.no_grad():
            z = torch.as_tensor(base, device=H.dev, dtype=torch.float32)
            r0_score = baseline_data.score(z, arm).cpu().numpy()[0]
        short1, _ = train(fit_data, starts, arm, 2, False)
        short2, _ = train(fit_data, starts, arm, 2, False)
        short_gap = float(np.max(np.abs(short1 - short2)))
        if short_gap > 1e-6:
            raise ValueError(f"Two-step deterministic replay failed for {arm}")
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.monotonic()
        terminal, logs = train(fit_data, starts, arm, CFG["steps"], True)
        torch.cuda.synchronize()
        steps = CFG["steps"]
        selected = np.argmin(logs[steps], axis=0)
        output[arm + "_terminal"] = terminal
        output[arm] = terminal[selected, np.arange(len(rows))]
        flat = np.abs(logs[steps] - logs[steps - 2000])
        reports[arm] = {"seconds": time.monotonic()-started, "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                        "peak_reserved_bytes": torch.cuda.max_memory_reserved(), "selected_start": selected.tolist(),
                        "two_step_replay_gap": short_gap, "max_last_2000_change": float(flat.max()),
                        "last_2000_change": flat.tolist(), "converged": bool(flat.max() <= CFG["convergence_abs"]),
                        "R0_same_objective": r0_score.tolist(), "progress": {str(k): v.tolist() for k, v in logs.items()}}
    np.savez(args.output / "fit_states.npz", **output)
    results = {"repo": P["repository"], "branch": P["branch"], "parent": P["parent"],
               "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
               "protocol_sha256": C.sha(HERE / "protocol.json"), "operator_sha256": C.sha(args.operator),
               "data_sha256": inputs, "materials": [row["tag"] for row in rows], "host": platform.platform(),
               "python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda,
               "gpu": torch.cuda.get_device_name(), "optimizer": CFG, "arms": reports,
               "fit_states_sha256": C.sha(args.output / "fit_states.npz")}
    write(args.output / "fit.json", results)


if __name__ == "__main__":
    main()
