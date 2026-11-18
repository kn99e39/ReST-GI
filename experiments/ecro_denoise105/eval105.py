"""WL105 evaluation inference: R0-R3 on every held-out view x spp x replicate, plus reconstruction cost.

python eval105.py DATA MODELS OUT [--smoke]
"""
from pathlib import Path
import hashlib
import json
import sys
import time

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import data105 as D  # noqa: E402
import model105 as M  # noqa: E402
from train105 import D_aux, sha  # noqa: E402

P = json.loads((HERE / "protocol/denoise105.json").read_text())
ARMS = ("R1", "R2", "R3")


def main(data, models, out, smoke):
    device = "cuda"
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    M.FUSED = True
    nets = {}
    for arm in ARMS:
        m = M.Reconstructor(arm).to(device)
        m.load_state_dict(torch.load(models / f"{arm}.pt", map_location=device))
        m.eval()
        nets[arm] = m
    man = json.loads((data / "run_manifest.json").read_text())
    names = [k.split("/")[1] for k in man["views"] if k.startswith("eval/")]
    reps = 1 if smoke else P["eval_reps"]
    out.mkdir(parents=True, exist_ok=False)
    rec = {"views": names, "reps": reps, "files": {}, "weights_sha256": {a: sha(models / f"{a}.pt") for a in ARMS},
           "data_manifest_sha256": sha(data / "run_manifest.json"), "code_sha256": {f.name: sha(f) for f in sorted(HERE.glob("*.py"))}}
    with torch.no_grad():
        for n in names:
            base, target, halves, nz = D.load_view(data / "eval" / n, reps, eval_split=True)
            for spp in D.SPPS:
                aux = torch.from_numpy(D_aux(base, spp))[None].to(device)
                for r in range(reps):
                    comp_np = nz[spp][r]
                    before = hashlib.sha256(comp_np.tobytes()).hexdigest()
                    comp = torch.from_numpy(comp_np)[None].to(device)
                    outs = {"R0": (comp[0, 0:3] + comp[0, 3:6] + comp[0, 6:9]).cpu().numpy()}
                    for arm, m in nets.items():
                        y, parts = m(comp, aux)
                        outs[arm] = y[0].cpu().numpy()
                    assert hashlib.sha256(comp_np.tobytes()).hexdigest() == before, "input modified"
                    name = f"{n}_spp{spp}_r{r}.npz"
                    np.savez_compressed(out / name, **outs)
                    rec["files"][name] = sha(out / name)
            print("eval", n, flush=True)
        # Cost: synchronized 512 x 512 latency, peak memory, parameters (one representative input).
        base, target, halves, nz = D.load_view(data / "eval" / names[0], 1, eval_split=True)
        comp = torch.from_numpy(nz[4][0])[None].to(device)
        aux = torch.from_numpy(D_aux(base, 4))[None].to(device)
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
        # Determinism: identical output for a repeated call.
        y1, _ = nets["R2"](comp, aux)
        y2, _ = nets["R2"](comp, aux)
        rec["deterministic_inference"] = bool(torch.equal(y1, y2))
        rec["cost"] = cost
        rec["gpu"] = torch.cuda.get_device_name(0)
    (out / "eval_manifest.json").write_text(json.dumps(rec, indent=1))
    print("EVAL COMPLETE", json.dumps(cost), flush=True)


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve(), Path(sys.argv[3]).resolve(), "--smoke" in sys.argv)
