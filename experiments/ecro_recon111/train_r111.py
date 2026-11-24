"""WL111 candidate training (Stage D). Frozen schedule and objective (protocol training); final weights only.

python train_r111.py DATA DESIGN_JSON OUT REALIZATION_INDEX [--smoke]
DATA = the data_r111 output (ladder + guides); targets = WL105 4096-spp frozen-ECRO targets (train T*, val Q*).
"""
import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as Fn

import common_r111 as C
import recon_r111 as RC

SPP_P = {1: 0.25, 2: 0.25, 4: 0.25, 8: 0.1, 16: 0.1, 32: 0.05}
LEVELS = 12


def gauss_kernel(sigma, device):
    r = int(3 * sigma)
    x = torch.arange(-r, r + 1, device=device, dtype=torch.float32)
    k = torch.exp(-0.5 * (x / sigma) ** 2)
    return (k / k.sum()).view(1, 1, 1, -1), r


class Data:
    def __init__(self, data, names, device):
        lad, gd, tg = [], [], []
        for n in names:
            with np.load(data / "ladder" / f"{n}.npz") as z:
                lad.append(torch.from_numpy(z["comps"]))
            with np.load(data / "ladder" / f"{n}_guides.npz") as z:
                gd.append(torch.from_numpy(np.concatenate([z["cover"], z["normal"], z["depth"][None]], 0).astype(np.float16)))
                pure = torch.from_numpy(z["pure"])
            split = "train" if n.startswith("T") else "val"
            with np.load(C.D105 / split / n / "target.npz") as z:
                tg.append(torch.from_numpy(z["comp"].astype(np.float32)))
        self.ladder = torch.stack(lad).to(device)              # (V, 12, 2, 9, H, W) fp16
        self.guides = torch.stack(gd).to(device)               # (V, 13, H, W) fp16
        self.mixed = (self.guides[:, :9].float().amax(1) < 1.0)  # (V, H, W)
        self.target = torch.stack(tg).to(device)               # (V, 9, H, W)
        self.V = self.ladder.shape[0]
        self.H, self.W = self.target.shape[-2:]
        self.device = device
        self.gk, self.gr = gauss_kernel(16.0, device)

    def age_exponent(self, gen, crop, mixed):
        """u(x) in {0..6}: log2 of the age map (frames) for one crop (protocol training.input_synthesis)."""
        dev = self.device
        r = torch.rand((), generator=gen).item()
        if r < 0.35:
            return torch.zeros(crop, crop, device=dev)
        if r < 0.65:
            u = int(torch.randint(1, 7, (), generator=gen))
            return torch.full((crop, crop), float(u), device=dev)
        noise = torch.rand((1, 1, crop, crop), generator=gen).to(dev)
        p = self.gr
        x = Fn.conv2d(Fn.pad(noise, (p, p, 0, 0), mode="reflect"), self.gk)
        x = Fn.conv2d(Fn.pad(x, (0, 0, p, p), mode="reflect"), self.gk.transpose(-1, -2))[0, 0]
        x = (x - x.min()) / (x.max() - x.min()).clamp(min=1e-12)
        u = torch.round(6.0 * x)
        nd = int(torch.randint(0, 4, (), generator=gen))
        yy, xx = torch.meshgrid(torch.arange(crop, device=dev), torch.arange(crop, device=dev), indexing="ij")
        for _ in range(nd):
            cy, cx = torch.randint(0, crop, (2,), generator=gen).tolist()
            rad = float(torch.randint(4, 25, (), generator=gen))
            u = torch.where((yy - cy) ** 2 + (xx - cx) ** 2 <= rad * rad, torch.zeros_like(u), u)
        if torch.rand((), generator=gen).item() < 0.5:
            u = torch.where(mixed, torch.zeros_like(u), u)
        return u

    def batch(self, gen, n, crop):
        comps, tgts, feats = [], [], []
        spps = list(SPP_P)
        probs = torch.tensor([SPP_P[s] for s in spps])
        for _ in range(n):
            v = int(torch.randint(0, self.V, (), generator=gen))
            s = spps[int(torch.multinomial(probs, 1, generator=gen))]
            rep = int(torch.randint(0, 2, (), generator=gen))
            y0 = int(torch.randint(0, self.H - crop + 1, (), generator=gen))
            x0 = int(torch.randint(0, self.W - crop + 1, (), generator=gen))
            ys, xs = slice(y0, y0 + crop), slice(x0, x0 + crop)
            u = self.age_exponent(gen, crop, self.mixed[v, ys, xs])
            level = (u + math.log2(s)).clamp(max=LEVELS - 1).long()                  # (crop, crop)
            stack = self.ladder[v, :, rep, :, ys, xs]                                 # (12, 9, c, c) fp16
            comp = torch.gather(stack, 0, level[None, None].expand(1, 9, crop, crop))[0].float()
            comps.append(comp)
            ess = (2.0 ** level.float())[None]
            g = self.guides[v, :, ys, xs].float()
            low, high = comp[0:3] + comp[3:6], comp[6:9]
            feats.append(RC.features(low[None], high[None], g[None, 0:9], g[None, 9:12], g[None, 12:13], ess[None])[0])
            tgts.append(self.target[v, :, ys, xs])
        return torch.stack(comps), torch.stack(tgts), torch.stack(feats)


def loss_fn(net, comps, tgts, feats):
    with torch.autocast("cuda", dtype=torch.float16):
        logits = net(feats)
    low, high = comps[:, 0:3] + comps[:, 3:6], comps[:, 6:9]
    lo, hi = RC.spatial_ref(low, high, logits.float())
    tl, th = tgts[:, 0:3] + tgts[:, 3:6], tgts[:, 6:9]
    return RC.rel_mse(lo, tl) + RC.rel_mse(hi, th) + RC.rel_mse(lo + hi, tl + th)


def main(data, design, out, ri, smoke):
    dsg = C.read(design)
    T = C.P["training"]
    seed_ = C.P["seeds"]["training_realizations"][ri]
    torch.manual_seed(seed_)
    device = "cuda"
    names = sorted(p.stem for p in (data / "ladder").glob("T*.npz") if not p.stem.endswith("_guides"))
    vnames = sorted(p.stem for p in (data / "ladder").glob("Q*.npz") if not p.stem.endswith("_guides"))
    ds = Data(data, names, device)
    net = RC.CompactNet(dsg["m"]).to(device)
    init_sha = C.array_sha(torch.cat([p.detach().flatten().cpu() for p in net.parameters()]).numpy())
    iters = 200 if smoke else 20000
    opt = torch.optim.Adam(net.parameters(), lr=2e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters, eta_min=1e-5)
    scaler = torch.cuda.amp.GradScaler()
    gen = torch.Generator().manual_seed(seed_)
    out.mkdir(parents=True, exist_ok=True)
    logd = {"realization": ri, "seed": seed_, "m": dsg["m"], "sub": dsg["sub"], "params": RC.n_params(net),
            "init_sha256": init_sha, "iters": iters, "losses": [], "train_views": names, "val_views": vnames}
    t0 = time.perf_counter()
    for it in range(iters):
        comps, tgts, feats = ds.batch(gen, 16, 128)
        loss = loss_fn(net, comps, tgts, feats)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        sched.step()
        if it % 100 == 0:
            logd["losses"].append([it, float(loss)])
            if it % 1000 == 0:
                print("iter", it, float(loss), round(time.perf_counter() - t0, 1), flush=True)
    logd["seconds"] = time.perf_counter() - t0
    wpath = out / f"cand_r{ri}.pt"
    torch.save(net.state_dict(), wpath)
    logd["weights_sha256"] = C.sha(wpath)
    # validation (logged only; selects nothing): cold (ESS = spp) and uniform-age inputs at 1 / 4 spp
    del ds
    torch.cuda.empty_cache()
    vds = Data(data, vnames, device)
    net.eval()
    val = {}
    with torch.no_grad():
        for spp in (1, 4):
            for age in (1, 16):
                num = den = 0.0
                lv = int(math.log2(spp * age))
                for v in range(vds.V):
                    comp = vds.ladder[v, lv, 0].float()
                    g = vds.guides[v].float()
                    low, high = comp[0:3] + comp[3:6], comp[6:9]
                    ess = torch.full((1, 1) + low.shape[-2:], float(2 ** lv), device=device)
                    f = RC.features(low[None], high[None], g[None, 0:9], g[None, 9:12], g[None, 12:13], ess)
                    with torch.autocast("cuda", dtype=torch.float16):
                        lg = net(f)
                    lo, hi = RC.spatial_ref(low[None], high[None], lg.float())
                    t = vds.target[v]
                    tt = t[0:3] + t[3:6] + t[6:9]
                    num += float(((lo + hi)[0] - tt).pow(2).mean())
                    den += float(tt.pow(2).mean())
                val[f"spp{spp}_age{age}"] = math.sqrt(num / den)
    logd["validation_rel_rmse_logged_only"] = val
    logd["data_manifest_sha256"] = C.sha(data / "data_manifest.json") if (data / "data_manifest.json").exists() else None
    logd["code_sha256"] = {f.name: C.sha(f) for f in sorted(C.HERE.glob("*.py"))}
    logd["protocol_sha256"] = C.sha(C.PROTOCOL)
    C.write(out / f"cand_r{ri}_log.json", logd)
    print("TRAIN_COMPLETE", ri, val, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("data", type=Path)
    ap.add_argument("design", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("ri", type=int)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    main(a.data.resolve(), a.design.resolve(), a.out.resolve(), a.ri, a.smoke)
