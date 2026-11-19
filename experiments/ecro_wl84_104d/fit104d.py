"""Matched certified ER direct fitting using the frozen WL82 primitives."""

import time

import numpy as np
import torch

from common104d import ARMS, CL, F, P, S, write


def objective(terms, shape_flag):
    return torch.where(shape_flag, terms["shape"], terms["e1"]) + terms["e2"]


def inputs(original, covered, device, starts=2):
    bundles = {"original": original, "covered": covered}
    ns = len(original["f"])
    groups, lanes = original["f"].shape[1:3]
    arrays = {}
    for key in ("wi", "wo", "f", "q", "use"):
        joined = np.stack([bundles[P["arms"][arm][0]][key] for arm in ARMS])
        arrays[key] = torch.tensor(np.repeat(joined[:, None], starts, axis=1), device=device, dtype=torch.float32)
    return {
        "wi": arrays["wi"].reshape(-1, 3), "wo": arrays["wo"].reshape(-1, 3),
        "f": arrays["f"].reshape(-1, 3), "w": (arrays["use"] / arrays["q"]).reshape(-1),
        "shape_flag": torch.tensor([P["arms"][arm][1] == "shape_energy" for arm in ARMS], device=device)[:, None, None, None],
        "layout": (len(ARMS), starts, ns, groups), "lanes": lanes,
    }


def loss_vector(z, batch):
    nlanes = batch["layout"][-1] * batch["lanes"]
    decoded = CL.decode("ER", "cert", z)
    params = {
        key: p[:, None].expand(len(p), nlanes, *p.shape[1:]).reshape(-1, *p.shape[1:])
        for key, p in decoded.items()
    }
    predict = CL.eval_closure("ER", params, batch["wi"], batch["wo"])
    terms = F.group_terms(predict, batch["f"], batch["w"], batch["lanes"])
    by_group = {key: val.reshape(batch["layout"]) for key, val in terms.items()}
    scores = objective(by_group, batch["shape_flag"]).mean(dim=-1)
    return scores


def optimize(batch, initial, steps, every, emit=False):
    config = P["optimization"]
    z = torch.nn.Parameter(initial.clone())
    optimizer = torch.optim.Adam([z], lr=F.OPT["lr"])
    log = {}
    for step in range(steps + 1):
        if step < steps:
            optimizer.param_groups[0]["lr"] = F.TR.lr_at(
                step, steps, F.OPT["lr"], F.OPT["lr_end"], config["warmup"]
            )
        scores = loss_vector(z, batch)
        if step % every == 0 or step == steps:
            log[step] = scores.detach().cpu().numpy()
            if not np.isfinite(log[step]).all():
                raise ValueError("Non-finite WL104D objective")
            if emit:
                print("fit", step, "median by arm/start", np.median(log[step], axis=-1).tolist(), flush=True)
        if step < steps:
            optimizer.zero_grad(set_to_none=True)
            scores.sum().backward()
            optimizer.step()
    return z.detach(), log


def fit(original, covered, out):
    config = P["optimization"]
    ns = len(original["f"])
    device = torch.device("cuda")
    generic = CL.init_raw("ER", "cert", ns, device)
    generator = torch.Generator(device="cpu").manual_seed(S["init"])
    perturb = torch.randn(generic.shape, generator=generator).to(device) * config["restart_noise"]
    starts = torch.stack((generic, generic + perturb))
    initial = starts[None].repeat(len(ARMS), 1, 1, 1).reshape(-1, generic.shape[-1])
    batch = inputs(original, covered, device)
    steps = config["steps"]
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start_time = time.monotonic()
    z, logs = optimize(batch, initial, steps, config["log_interval"], emit=True)
    torch.cuda.synchronize()
    elapsed = time.monotonic() - start_time
    candidates = z.cpu().numpy().reshape(len(ARMS), 2, ns, -1)
    best = np.argmin(logs[steps], axis=1)
    selected = {arm: candidates[i, best[i], np.arange(ns), :] for i, arm in enumerate(ARMS)}
    flatness = np.abs(logs[steps] - logs[steps - 2000])
    report = {
        "converged": bool(flatness.max() <= config["convergence_abs"]), "steps_per_start": steps,
        "max_final_2k_loss_change_by_arm": dict(zip(ARMS, flatness.max(axis=(1, 2)).tolist())),
        "selected_start": {arm: best[i].tolist() for i, arm in enumerate(ARMS)},
        "arm_order": list(ARMS), "seconds_joint_fit": elapsed,
        "peak_torch_bytes_joint_fit": torch.cuda.max_memory_allocated(),
        "optimiser_progress": {str(step): scores.tolist() for step, scores in logs.items()},
    }
    np.savez(out / "fit_states.npz", all_initial=initial.cpu().numpy(), all_terminal=candidates, **selected)
    write(out / "fit.json", report)
    return selected, report
