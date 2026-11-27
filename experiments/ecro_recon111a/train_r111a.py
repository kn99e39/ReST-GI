"""WL111A Stage B (H1 x H4 factorial): WL111's CT-AK training (train_r111, unchanged code) with exactly two switches:

  --age mixed | cold   mixed = WL111's age-map synthesis; cold = every pixel at age 1 (ESS = base spp). The cold arm draws
                       the age map with the same generator calls and then discards it, so view / spp / replicate / crop
                       sequences are identical to the mixed arm's (matched data stream).
  --sub 4 | 8          guide files: WL111's sub-4 ladder guides, or WL111A's sub-8 guides of the same views.

Architecture (CompactNet m = 1.0), ladder frames, targets, spp distribution, loss, optimizer, schedule, steps, batch and
seeds (17950001 / 17950002, so initial weights equal WL111's) are WL111's. Final weights only (no selection).

python train_r111a.py ARM AGE SUB RI      -> OUT/models/<ARM>_r<RI>.pt + _log.json
"""
import argparse
import math
import os
import time
from pathlib import Path

import numpy as np
import torch

import common_r111a as A
import common_r111 as C
import recon_r111 as RC
import recon_r111a as RA
import train_r111 as T


class Data(T.Data):
    def __init__(self, data, names, device, guide_dir, cold):
        self.cold = cold
        lad, gd, tg = [], [], []
        for n in names:
            with np.load(data / "ladder" / f"{n}.npz") as z:
                lad.append(torch.from_numpy(z["comps"]))
            with np.load(guide_dir / f"{n}_guides.npz") as z:
                gd.append(torch.from_numpy(np.concatenate([z["cover"], z["normal"], z["depth"][None]], 0).astype(np.float16)))
            split = "train" if n.startswith("T") else "val"
            with np.load(C.D105 / split / n / "target.npz") as z:
                tg.append(torch.from_numpy(z["comp"].astype(np.float32)))
        self.ladder = torch.stack(lad).to(device)
        self.guides = torch.stack(gd).to(device)
        self.mixed = (self.guides[:, :9].float().amax(1) < 1.0)
        self.target = torch.stack(tg).to(device)
        self.V = self.ladder.shape[0]
        self.H, self.W = self.target.shape[-2:]
        self.device = device
        self.gk, self.gr = T.gauss_kernel(16.0, device)

    def age_exponent(self, gen, crop, mixed):
        u = super().age_exponent(gen, crop, mixed)          # same generator consumption in both arms
        return torch.zeros_like(u) if self.cold else u


def main(arm, age, sub, ri, net_kind="ctak"):
    data = A.WL111_EVID / "data"
    guide_dir = data / "ladder" if sub == 4 else A.OUT / "ladder8"
    seed_ = C.P["seeds"]["training_realizations"][ri]
    torch.manual_seed(seed_)
    device = "cuda"
    names = sorted(p.stem for p in (data / "ladder").glob("T*.npz") if not p.stem.endswith("_guides"))
    ds = Data(data, names, device, guide_dir, age == "cold")
    net = RA.NETS[net_kind](1.0).to(device)
    init_sha = C.array_sha(torch.cat([p.detach().flatten().cpu() for p in net.parameters()]).numpy())
    iters = int(os.environ.get("RECON111A_ITERS", 20000))      # smoke only; evidence runs use 20000
    opt = torch.optim.Adam(net.parameters(), lr=2e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters, eta_min=1e-5)
    scaler = torch.cuda.amp.GradScaler()
    gen = torch.Generator().manual_seed(seed_)
    out = A.OUT / "models"
    out.mkdir(parents=True, exist_ok=True)
    logd = {"arm": arm, "net": net_kind, "device": torch.cuda.get_device_name(0), "age": age, "sub": sub, "realization": ri, "seed": seed_, "params": RC.n_params(net),
            "init_sha256": init_sha, "iters": iters, "losses": [], "train_views": names, "guide_dir": str(guide_dir)}
    t0 = time.perf_counter()
    for it in range(iters):
        comps, tgts, feats = ds.batch(gen, 16, 128)
        loss = T.loss_fn(net, comps, tgts, feats)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        sched.step()
        if it % 100 == 0:
            logd["losses"].append([it, float(loss)])
            if it % 2000 == 0:
                print(arm, ri, "iter", it, float(loss), round(time.perf_counter() - t0, 1), flush=True)
    logd["seconds"] = time.perf_counter() - t0
    wpath = out / f"{arm}_r{ri}.pt"
    torch.save(net.state_dict(), wpath)
    logd["weights_sha256"] = C.sha(wpath)
    logd["code_sha256"] = {f.name: C.sha(f) for f in sorted(A.HERE.glob("*.py"))}
    logd["wl111_code_sha256"] = {f.name: C.sha(f) for f in sorted(C.HERE.glob("*.py"))}
    C.write(out / f"{arm}_r{ri}_log.json", logd)
    print("TRAIN_COMPLETE", arm, ri, round(logd["seconds"]), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("arm")
    ap.add_argument("age", choices=("mixed", "cold"))
    ap.add_argument("sub", type=int, choices=(4, 8))
    ap.add_argument("ri", type=int, choices=(0, 1))
    ap.add_argument("--net", default="ctak", choices=tuple(RA.NETS))
    a = ap.parse_args()
    main(a.arm, a.age, a.sub, a.ri, a.net)
