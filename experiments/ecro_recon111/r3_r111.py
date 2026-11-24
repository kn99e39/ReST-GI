"""Frozen WL105 R3 (checkpoint d51b6919..., model105.Reconstructor('R3'), fused Triton kernel application kpn_triton105)
with stage-resolved timing. The staged forward runs the same modules and the same operations in the same order as
Reconstructor.forward; its output equals the frozen forward bitwise (Stage A test).

Stages: features (LOW / HIGH sums, log1p, concat with aux) | net_low | apply_low | net_high | apply_high | sum.
Diagnostics (latency only, never used for quality): fp16 autocast of the two U-Nets; unfused unfold application.
"""
import sys

import torch

import common_r111 as C

sys.path.insert(0, str(C.EXP / "ecro_denoise105"))
import model105 as M  # noqa: E402

M.FUSED = True


def load_r3(path=None, device="cuda"):
    path = path or (C.M105 / "R3.pt")
    assert C.sha(path) == C.P["gates"]["G0"]["r3_sha256"], "R3 checkpoint identity"
    m = M.Reconstructor("R3").to(device)
    m.load_state_dict(torch.load(path, map_location=device))
    return m.eval()


def aux14(g, spp):
    """R3 aux from 8 x 8 guides: coverage (9), mean normal (3), mean depth / 5, log2(spp) / 5 -> (1,14,H,W)."""
    H, W = g["depth"].shape
    sppf = torch.full((1, H, W), torch.log2(torch.tensor(float(spp))).item() / 5.0, device=g["depth"].device)
    return torch.cat([g["cover"], g["normal"], g["depth"][None] / 5.0, sppf], 0)[None].float()


@torch.no_grad()
def forward(model, comp, aux):
    """Frozen forward (comp (1,9,H,W) component means, aux (1,14,H,W)) -> TOTAL (1,3,H,W)."""
    y, _ = model(comp, aux)
    return y


class Staged:
    """Same computation as Reconstructor.forward('R3') with CUDA events between stages."""

    def __init__(self, model, fp16=False, fused=True):
        self.m, self.fp16, self.fused = model, fp16, fused
        self.names = ("features", "net_low", "apply_low", "net_high", "apply_high", "sum")

    @torch.no_grad()
    def __call__(self, comp, aux):
        ev = [torch.cuda.Event(enable_timing=True) for _ in range(len(self.names) + 1)]
        m = self.m
        M.FUSED = self.fused
        ev[0].record()
        low, high, _ = M.features(comp, aux)
        xl = torch.cat([torch.log1p(low), aux], 1)
        xh = torch.cat([torch.log1p(high), aux], 1)
        ev[1].record()
        with torch.autocast("cuda", dtype=torch.float16, enabled=self.fp16):
            kl = m.net_low(xl)
        ev[2].record()
        lo = M.apply_kernel(low, kl.float() if self.fp16 else kl)
        ev[3].record()
        with torch.autocast("cuda", dtype=torch.float16, enabled=self.fp16):
            kh = m.net_high(xh)
        ev[4].record()
        hi = M.apply_kernel(high, kh.float() if self.fp16 else kh)
        ev[5].record()
        y = lo + hi
        ev[6].record()
        torch.cuda.synchronize()
        M.FUSED = True
        return y, {n: ev[i].elapsed_time(ev[i + 1]) for i, n in enumerate(self.names)}, {
            "logits_bytes": int(kl.numel() * kl.element_size() + kh.numel() * kh.element_size())}


def n_params(model):
    return sum(p.numel() for p in model.parameters())
