"""WL106 R4 training: the WL105 R3 recipe unchanged (data, crops, batch sequence, loss, optimiser, schedule, seed).

python train106.py DATA105 OUT [--smoke]
"""
import json
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import model106 as M6  # noqa: E402
sys.path.insert(0, str(HERE.parent / "ecro_denoise105"))
import data105 as D  # noqa: E402
import train105 as T5  # noqa: E402
M = M6.M


def main(data, out, smoke):
    device = "cuda"
    P = T5.P
    torch.manual_seed(P["seeds"]["training"])
    man = json.loads((data / "run_manifest.json").read_text())
    train_names = sorted(k.split("/")[1] for k in man["views"] if k.startswith("train/"))
    val_names = sorted(k.split("/")[1] for k in man["views"] if k.startswith("val/"))
    reps = 1 if smoke else P["train_reps"]
    ts = D.TrainSet(data / "train", train_names, reps, device)
    val = []
    for n in val_names:
        base, target, _, nz = D.load_view(data / "val" / n, reps)
        val.append((base, target, nz))
    model = M6.R4().to(device)
    iters = 200 if smoke else 20000
    opt = torch.optim.Adam(model.parameters(), lr=2e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters, eta_min=1e-5)
    gen = torch.Generator().manual_seed(P["seeds"]["training"])
    log = {"arm": "R4", "params": M.n_params(model), "iters": iters, "losses": [], "validation": []}
    out.mkdir(parents=True, exist_ok=True)
    stamp = time.perf_counter()
    for it in range(iters):
        comp, tgt, aux = ts.batch(gen, 16, 128)
        y, parts = model(comp, aux)
        loss = M.rel_mse(parts["low"], tgt[:, 0:3] + tgt[:, 3:6]) + M.rel_mse(parts["high"], tgt[:, 6:9])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if it % 100 == 0:
            log["losses"].append([it, float(loss)])
        if (it + 1) % (100 if smoke else 2000) == 0:
            v = T5.validate(model, val, device)
            log["validation"].append({"iter": it + 1, "relative_rmse": v})
            print("R4", it + 1, float(loss), v, round(time.perf_counter() - stamp), flush=True)
    log["seconds"] = time.perf_counter() - stamp
    torch.save(model.state_dict(), out / "R4.pt")
    log["weights_sha256"] = T5.sha(out / "R4.pt")
    log["data_manifest_sha256"] = T5.sha(data / "run_manifest.json")
    log["code_sha256"] = {f.name: T5.sha(f) for f in sorted(HERE.glob("*.py"))}
    (out / "R4_log.json").write_text(json.dumps(log, indent=1))
    print("TRAIN COMPLETE R4", flush=True)


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve(), "--smoke" in sys.argv)
