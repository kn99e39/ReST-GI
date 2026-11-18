"""WL105 reconstructors: shared U-Net backbone with per-pixel 15 x 15 softmax kernels (local-mean preserving).

R1  combined RGB      : features log1p(TOTAL) + aux -> one kernel on linear TOTAL
R2  coupled LOW/HIGH  : features log1p(LOW), log1p(HIGH) + aux -> two kernels, applied to LOW and HIGH, summed
R3  separate (ablation): two independent half-capacity U-Nets (LOW + aux, HIGH + aux), one kernel each, summed
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

K = 15
AUX = 14          # 9 coverage + 3 normal + 1 depth + 1 log2(spp)


def block(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.ReLU(inplace=True),
                         nn.Conv2d(cout, cout, 3, padding=1), nn.ReLU(inplace=True))


class UNet(nn.Module):
    def __init__(self, cin, cout, widths=(32, 64, 96, 128)):
        super().__init__()
        w = list(widths)
        self.enc = nn.ModuleList([block(cin, w[0])] + [block(w[i], w[i + 1]) for i in range(len(w) - 1)])
        self.down = nn.ModuleList([nn.Conv2d(w[i], w[i], 3, stride=2, padding=1) for i in range(len(w) - 1)])
        self.dec = nn.ModuleList([block(w[i + 1] + w[i], w[i]) for i in reversed(range(len(w) - 1))])
        self.head = nn.Conv2d(w[0], cout, 1)

    def forward(self, x):
        skips = []
        for i, e in enumerate(self.enc):
            x = e(x)
            if i < len(self.down):
                skips.append(x)
                x = self.down[i](x)
        for d in self.dec:
            s = skips.pop()
            x = F.interpolate(x, size=s.shape[-2:], mode="bilinear", align_corners=False)
            x = d(torch.cat([x, s], 1))
        return self.head(x)


FUSED = False     # inference only: Triton fused application (kpn_triton105), numerically equivalent


def apply_kernel(img, logits):
    """img (B,C,H,W) linear radiance; logits (B,K*K,H,W) -> softmax-weighted neighbourhood average (B,C,H,W)."""
    if FUSED and not torch.is_grad_enabled():
        from kpn_triton105 import apply_kernel_fused
        return apply_kernel_fused(img, logits, K)
    B, C, H, W = img.shape
    r = K // 2
    w = torch.softmax(logits, 1)
    patches = F.unfold(F.pad(img, (r, r, r, r), mode="reflect"), K).view(B, C, K * K, H, W)
    return (patches * w[:, None]).sum(2)


def features(comp, aux):
    """comp: (B,9,H,W) component means e1 rgb, e2 rgb, hl rgb -> LOW, HIGH, TOTAL and log features."""
    low = comp[:, 0:3] + comp[:, 3:6]
    high = comp[:, 6:9]
    total = low + high
    return low, high, total


class Reconstructor(nn.Module):
    def __init__(self, arm):
        super().__init__()
        self.arm = arm
        if arm == "R1":
            self.net = UNet(3 + AUX, K * K)
        elif arm == "R2":
            self.net = UNet(6 + AUX, 2 * K * K)
        elif arm == "R3":
            s = 1 / math.sqrt(2)
            widths = tuple(int(round(c * s)) for c in (32, 64, 96, 128))
            self.net_low = UNet(3 + AUX, K * K, widths)
            self.net_high = UNet(3 + AUX, K * K, widths)
        else:
            raise ValueError(arm)

    def forward(self, comp, aux):
        """Returns (total_hat, parts) where parts holds per-component outputs when they exist."""
        low, high, total = features(comp, aux)
        if self.arm == "R1":
            k = self.net(torch.cat([torch.log1p(total), aux], 1))
            return apply_kernel(total, k), {}
        if self.arm == "R2":
            k = self.net(torch.cat([torch.log1p(low), torch.log1p(high), aux], 1))
            lo, hi = apply_kernel(low, k[:, :K * K]), apply_kernel(high, k[:, K * K:])
            return lo + hi, {"low": lo, "high": hi}
        lo = apply_kernel(low, self.net_low(torch.cat([torch.log1p(low), aux], 1)))
        hi = apply_kernel(high, self.net_high(torch.cat([torch.log1p(high), aux], 1)))
        return lo + hi, {"low": lo, "high": hi}


def rel_mse(pred, target):
    return ((pred - target) ** 2 / (pred.detach() ** 2 + 0.01)).mean()


def n_params(m):
    return sum(p.numel() for p in m.parameters())
