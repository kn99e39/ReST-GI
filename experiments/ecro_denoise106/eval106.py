"""WL106 evaluation inference: R0-R4 on every held-out view x spp x replicate (identical noisy inputs), mixing terms of
R3 and R4 against the qualified target, coverage moments, reconstruction cost.

python eval106.py DATA105 MODELS105 MODELS106 TARGETS106 OUT
"""
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import model106 as M6  # noqa: E402
import tilt106 as T  # noqa: E402
sys.path.insert(0, str(HERE.parent / "ecro_denoise105"))
import data105 as D  # noqa: E402
from train105 import D_aux, sha  # noqa: E402
M = M6.M


def qualified_target(d105, d106):
    """Mean of the two WL105 halves and every new half; per-pixel variance of the mean (sample variance / n)."""
    files = [d105 / f"target_half{h}.npz" for h in range(2)] + sorted(d106.glob("target_new_h*.npz"))
    s = s2 = None
    for f in files:
        x = np.load(f)["comp"].astype(np.float64)
        s = x if s is None else s + x
        tot = x[0:3] + x[3:6] + x[6:9]
        s2 = tot ** 2 if s2 is None else s2 + tot ** 2
    n = len(files)
    mean = s / n
    Y = mean[0:3] + mean[3:6] + mean[6:9]
    var_mean = np.maximum(s2 / n - Y ** 2, 0) * n / (n - 1) / n
    return mean, var_mean, n, [f.name for f in files]


def main(data, models105, models106, targets, out):
    dev = "cuda"
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    M.FUSED = True
    M6.FUSED = True
    nets = {}
    for arm in ("R1", "R2", "R3"):
        m = M.Reconstructor(arm).to(dev)
        m.load_state_dict(torch.load(models105 / f"{arm}.pt", map_location=dev))
        nets[arm] = m.eval()
    r4 = M6.R4().to(dev)
    r4.load_state_dict(torch.load(models106 / "R4.pt", map_location=dev))
    nets["R4"] = r4.eval()
    dman = json.loads((data / "run_manifest.json").read_text())
    names = [k.split("/")[1] for k in dman["views"] if k.startswith("eval/")]
    reps = 8
    out.mkdir(parents=True, exist_ok=False)
    stored = data.parent / "eval"
    rec = {"views": names, "reps": reps, "files": {}, "targets": {},
           "weights_sha256": {a: sha(models105 / f"{a}.pt") for a in ("R1", "R2", "R3")} | {"R4": sha(models106 / "R4.pt")},
           "data_manifest_sha256": sha(data / "run_manifest.json"), "code_sha256": {f.name: sha(f) for f in sorted(HERE.glob("*.py"))},
           "input_hash_mismatch": [], "reproduction_max_rel_vs_WL105": 0.0, "fused_fallbacks": 0}
    with torch.no_grad():
        for n in names:
            d = data / "eval" / n
            mean, var_mean, nh, files = qualified_target(d, targets / n)
            np.savez_compressed(out / f"target_{n}.npz", comp=mean.astype(np.float32), var_mean=var_mean.astype(np.float32), n=nh)
            rec["targets"][n] = {"n_halves": nh, "files": files, "sha256": sha(out / f"target_{n}.npz")}
            base, _, _, nz = D.load_view(d, reps, eval_split=True)
            tlow = torch.from_numpy((mean[0:3] + mean[3:6]).astype(np.float32))[None].to(dev)
            thigh = torch.from_numpy(mean[6:9].astype(np.float32))[None].to(dev)
            for spp in D.SPPS:
                aux = torch.from_numpy(D_aux(base, spp))[None].to(dev)
                cov = aux[:, :9]
                for r in range(reps):
                    comp_np = nz[spp][r]
                    fname = f"noisy_spp{spp}_r{r}.npz"
                    if sha(d / fname) != dman["views"][f"eval/{n}"]["files"][fname]:
                        rec["input_hash_mismatch"].append(f"{n}/{fname}")
                    before = hashlib.sha256(comp_np.tobytes()).hexdigest()
                    comp = torch.from_numpy(comp_np)[None].to(dev)
                    low, high, _ = M.features(comp, aux)
                    outs = {"R0": (comp[0, 0:3] + comp[0, 3:6] + comp[0, 6:9]).cpu().numpy()}
                    for arm in ("R1", "R2", "R3"):
                        outs[arm] = nets[arm](comp, aux)[0][0].cpu().numpy()
                    y4, _ = nets["R4"](comp, aux)
                    outs["R4"] = y4[0].cpu().numpy()
                    # mixing terms: each component kernel applied to the qualified target component
                    zl3 = nets["R3"].net_low(torch.cat([torch.log1p(low), aux], 1))
                    zh3 = nets["R3"].net_high(torch.cat([torch.log1p(high), aux], 1))
                    zl4 = nets["R4"].net_low(torch.cat([torch.log1p(low), aux], 1))
                    zh4 = nets["R4"].net_high(torch.cat([torch.log1p(high), aux], 1))
                    outs["mix_R3"] = (M.apply_kernel(tlow, zl3) + M.apply_kernel(thigh, zh3))[0].cpu().numpy()
                    outs["mix_R4"] = (T.constrained_apply_fused(tlow, zl4, cov) + T.constrained_apply_fused(thigh, zh4, cov))[0].cpu().numpy()
                    c10 = T.coverage10(cov)
                    c12 = torch.cat([c10, torch.zeros_like(c10[:, :2])], 1)
                    for tag, z, constrained in (("R3_low", zl3, False), ("R3_high", zh3, False), ("R4_low", zl4, True), ("R4_high", zh4, True)):
                        mom = torch.cat([(T.constrained_apply_fused(c12[:, j:j + 3].contiguous(), z, cov) if constrained else
                                          M.apply_kernel(c12[:, j:j + 3].contiguous(), z)) for j in (0, 3, 6, 9)], 1)[0, :10]
                        outs[f"moment_{tag}"] = (mom - c10[0]).abs().sum(0).cpu().numpy().astype(np.float16)
                    assert hashlib.sha256(comp_np.tobytes()).hexdigest() == before, "input modified"
                    with np.load(stored / f"{n}_spp{spp}_r{r}.npz") as f:
                        for arm in ("R1", "R2", "R3"):
                            rec["reproduction_max_rel_vs_WL105"] = max(rec["reproduction_max_rel_vs_WL105"],
                                                                       float(np.abs(outs[arm] - f[arm]).max() / np.abs(f[arm]).max()))
                    name = f"{n}_spp{spp}_r{r}.npz"
                    np.savez_compressed(out / name, **outs)
                    rec["files"][name] = sha(out / name)
            print("eval", n, flush=True)
        # Cost (uncontended): synchronized 512 x 512 latency, peak extra memory, parameters.
        base, _, _, nz = D.load_view(data / "eval" / names[0], 1, eval_split=True)
        comp = torch.from_numpy(nz[4][0])[None].to(dev)
        aux = torch.from_numpy(D_aux(base, 4))[None].to(dev)
        cost = {}
        for arm, m in nets.items():
            for _ in range(10):
                m(comp, aux)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            base_mem = torch.cuda.memory_allocated()
            times = []
            for _ in range(50):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                m(comp, aux)
                torch.cuda.synchronize()
                times.append(1e3 * (time.perf_counter() - t0))
            cost[arm] = {"latency_ms_median": float(np.median(times)), "latency_ms_p95": float(np.percentile(times, 95)),
                         "peak_extra_mib": (torch.cuda.max_memory_allocated() - base_mem) / 2 ** 20, "params": M.n_params(m)}
        y1, _ = nets["R4"](comp, aux)
        y2, _ = nets["R4"](comp, aux)
        rec["deterministic_inference_R4"] = bool(torch.equal(y1, y2))
        M6.FUSED = False
        y3, _ = nets["R4"](comp, aux)
        rec["R4_fused_vs_reference_max_rel"] = float(((y1 - y3).abs() / y3.abs().clamp_min(1e-3)).max())
        rec["cost"] = cost
        rec["gpu"] = torch.cuda.get_device_name(0)
    (out / "eval_manifest.json").write_text(json.dumps(rec, indent=1))
    print("EVAL COMPLETE", json.dumps(cost), rec["reproduction_max_rel_vs_WL105"], rec["input_hash_mismatch"], flush=True)


if __name__ == "__main__":
    main(*[Path(a).resolve() for a in sys.argv[1:6]])
