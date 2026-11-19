"""WL106 R4: R3-derived reconstructor with coverage-moment-constrained kernels (tilt106).

Identical to WL105 R3 in every respect (two independent U-Nets with widths x 1/sqrt(2), inputs log1p(LOW) + aux and
log1p(HIGH) + aux, one 15 x 15 softmax kernel each, applied to linear LOW / HIGH, summed; same parameter names and
count, same initialisation under the same seed) except that each predicted kernel is replaced by its minimum-KL
coverage-moment-constrained version before application. R3 code and checkpoints are imported unchanged.
"""
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ecro_denoise105"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import model105 as M  # noqa: E402
import tilt106 as T  # noqa: E402

FUSED = False      # inference only: fused constrained application (kpn_triton106), equivalent to the reference path


def constrained_apply(img, logits, cov):
    if FUSED and not torch.is_grad_enabled():
        y = T.constrained_apply_fused(img, logits, cov)
        if y is not None:
            return y
    B, C, H, W = img.shape
    r = M.K // 2
    w = torch.softmax(T.constrain_logits(logits, cov), 1)
    patches = torch.nn.functional.unfold(torch.nn.functional.pad(img, (r, r, r, r), mode="reflect"), M.K).view(B, C, M.K * M.K, H, W)
    return (patches * w[:, None]).sum(2)


class R4(nn.Module):
    def __init__(self):
        super().__init__()
        ref = M.Reconstructor("R3")
        self.arm = "R4"
        self.net_low, self.net_high = ref.net_low, ref.net_high

    def forward(self, comp, aux):
        low, high, total = M.features(comp, aux)
        cov = aux[:, :9]
        lo = constrained_apply(low, self.net_low(torch.cat([torch.log1p(low), aux], 1)), cov)
        hi = constrained_apply(high, self.net_high(torch.cat([torch.log1p(high), aux], 1)), cov)
        return lo + hi, {"low": lo, "high": hi}
