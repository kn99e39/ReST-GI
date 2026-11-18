"""WL105 data access: component means, targets and aux features as tensors (no renderer import)."""
import json
from pathlib import Path

import numpy as np
import torch

SPPS = (4, 16, 32)


def aux_tensor(aux_npz, spp):
    with np.load(aux_npz) as a:
        cover, normal, depth = a["cover"], a["normal"], a["depth"]
    sppf = np.full((1,) + depth.shape, np.log2(spp) / 5.0, np.float32)
    return np.concatenate([cover, normal, depth[None] / 5.0, sppf], 0).astype(np.float32)   # (14, H, W)


def load_view(d, reps, spps=SPPS, eval_split=False):
    d = Path(d)
    with np.load(d / "aux.npz") as a:
        base = {k: a[k] for k in ("cover", "normal", "depth", "sid", "ok", "p", "valid_frac")}
    if eval_split:
        halves = [np.load(d / f"target_half{h}.npz")["comp"] for h in range(2)]
        target = 0.5 * (halves[0] + halves[1])
    else:
        halves = None
        target = np.load(d / "target.npz")["comp"]
    noisy = {spp: [np.load(d / f"noisy_spp{spp}_r{r}.npz")["comp"] for r in range(reps)] for spp in spps}
    return base, target, halves, noisy


class TrainSet:
    """All training views resident on the GPU: noisy components fp16, targets fp32, aux fp16."""
    def __init__(self, root, names, reps, device):
        self.device = device
        noisy, targets, aux = [], [], []
        for n in names:
            base, target, _, nz = load_view(Path(root) / n, reps)
            targets.append(torch.from_numpy(target))
            noisy.append(torch.stack([torch.stack([torch.from_numpy(x) for x in nz[s]]) for s in SPPS]))   # (S, R, 9, H, W)
            a = np.concatenate([base["cover"], base["normal"], base["depth"][None] / 5.0], 0)
            aux.append(torch.from_numpy(a.astype(np.float32)))
        self.noisy = torch.stack(noisy).to(device, torch.float16)       # (V, S, R, 9, H, W)
        self.target = torch.stack(targets).to(device)                   # (V, 9, H, W)
        self.aux = torch.stack(aux).to(device, torch.float16)            # (V, 13, H, W)
        self.V, self.S, self.R = self.noisy.shape[:3]
        self.H, self.W = self.target.shape[-2:]

    def batch(self, gen, n, crop):
        v = torch.randint(0, self.V, (n,), generator=gen)
        s = torch.randint(0, self.S, (n,), generator=gen)
        r = torch.randint(0, self.R, (n,), generator=gen)
        y0 = torch.randint(0, self.H - crop + 1, (n,), generator=gen)
        x0 = torch.randint(0, self.W - crop + 1, (n,), generator=gen)
        comp, tgt, aux = [], [], []
        for i in range(n):
            ys, xs = slice(int(y0[i]), int(y0[i]) + crop), slice(int(x0[i]), int(x0[i]) + crop)
            comp.append(self.noisy[v[i], s[i], r[i], :, ys, xs].float())
            tgt.append(self.target[v[i], :, ys, xs])
            spp = SPPS[int(s[i])]
            a = self.aux[v[i], :, ys, xs].float()
            aux.append(torch.cat([a, torch.full((1, crop, crop), np.log2(spp) / 5.0, device=self.device)], 0))
        return torch.stack(comp), torch.stack(tgt), torch.stack(aux)
