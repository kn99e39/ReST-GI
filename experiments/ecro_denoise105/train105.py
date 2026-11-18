"""WL105 training: identical data, budget and schedule for R1, R2, R3. Final weights are evaluated (no selection).

python train105.py DATA OUT ARM [--smoke]
"""
from pathlib import Path
import hashlib
import json
import math
import sys
import time

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import data105 as D  # noqa: E402
import model105 as M  # noqa: E402

P = json.loads((HERE / "protocol/denoise105.json").read_text())


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def validate(model, val, device):
    out = {}
    model.eval()
    with torch.no_grad():
        for spp in D.SPPS:
            num = den = 0.0
            for base, target, nz in val:
                comp = torch.from_numpy(nz[spp][0])[None].to(device)
                aux = torch.from_numpy(D_aux(base, spp))[None].to(device)
                y, _ = model(comp, aux)
                t = torch.from_numpy(target[0:3] + target[3:6] + target[6:9])[None].to(device)
                num += float(((y - t) ** 2).mean())
                den += float((t ** 2).mean())
            out[spp] = math.sqrt(num / den)
    model.train()
    return out


def D_aux(base, spp):
    sppf = np.full((1,) + base["depth"].shape, np.log2(spp) / 5.0, np.float32)
    return np.concatenate([base["cover"], base["normal"], base["depth"][None] / 5.0, sppf], 0).astype(np.float32)


def main(data, out, arm, smoke):
    device = "cuda"
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
    model = M.Reconstructor(arm).to(device)
    iters = 200 if smoke else 20000
    opt = torch.optim.Adam(model.parameters(), lr=2e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters, eta_min=1e-5)
    gen = torch.Generator().manual_seed(P["seeds"]["training"])
    log = {"arm": arm, "params": M.n_params(model), "iters": iters, "losses": [], "validation": []}
    out.mkdir(parents=True, exist_ok=True)
    stamp = time.perf_counter()
    for it in range(iters):
        comp, tgt, aux = ts.batch(gen, 16, 128)
        y, parts = model(comp, aux)
        if arm == "R3":
            loss = M.rel_mse(parts["low"], tgt[:, 0:3] + tgt[:, 3:6]) + M.rel_mse(parts["high"], tgt[:, 6:9])
        else:
            loss = M.rel_mse(y, tgt[:, 0:3] + tgt[:, 3:6] + tgt[:, 6:9])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if it % 100 == 0:
            log["losses"].append([it, float(loss)])
        if (it + 1) % (100 if smoke else 2000) == 0:
            v = validate(model, val, device)
            log["validation"].append({"iter": it + 1, "relative_rmse": v})
            print(arm, it + 1, float(loss), v, flush=True)
    log["seconds"] = time.perf_counter() - stamp
    torch.save(model.state_dict(), out / f"{arm}.pt")
    log["weights_sha256"] = sha(out / f"{arm}.pt")
    log["data_manifest_sha256"] = sha(data / "run_manifest.json")
    log["code_sha256"] = {f.name: sha(f) for f in sorted(HERE.glob("*.py"))}
    (out / f"{arm}_log.json").write_text(json.dumps(log, indent=1))
    print("TRAIN COMPLETE", arm, flush=True)


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve(), sys.argv[3], "--smoke" in sys.argv)
