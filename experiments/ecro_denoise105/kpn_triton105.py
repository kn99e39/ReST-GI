"""Fused per-pixel softmax-kernel application for inference (Triton). Numerically equivalent to
model105.apply_kernel (reflect padding, softmax over the K*K taps), without materialising unfold."""
import torch
import triton
import triton.language as tl


@triton.jit
def _kpn(img, logits, out, H, W, HW, K: tl.constexpr, C: tl.constexpr, BLOCK: tl.constexpr):
    b = tl.program_id(1)
    pix = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = pix < HW
    y = pix // W
    x = pix % W
    R: tl.constexpr = K // 2
    lbase = logits + b * (K * K) * HW
    m = tl.full((BLOCK,), -1e30, tl.float32)
    for k in range(K * K):
        l = tl.load(lbase + k * HW + pix, mask=mask, other=-1e30)
        m = tl.maximum(m, l)
    s = tl.zeros((BLOCK,), tl.float32)
    a0 = tl.zeros((BLOCK,), tl.float32)
    a1 = tl.zeros((BLOCK,), tl.float32)
    a2 = tl.zeros((BLOCK,), tl.float32)
    ibase = img + b * C * HW
    for k in range(K * K):
        dy = k // K - R
        dx = k % K - R
        yy = y + dy
        xx = x + dx
        yy = tl.where(yy < 0, -yy, yy)
        yy = tl.where(yy >= H, 2 * H - 2 - yy, yy)
        xx = tl.where(xx < 0, -xx, xx)
        xx = tl.where(xx >= W, 2 * W - 2 - xx, xx)
        q = yy * W + xx
        e = tl.exp(tl.load(lbase + k * HW + pix, mask=mask, other=-1e30) - m)
        s += e
        a0 += e * tl.load(ibase + q, mask=mask, other=0.0)
        a1 += e * tl.load(ibase + HW + q, mask=mask, other=0.0)
        a2 += e * tl.load(ibase + 2 * HW + q, mask=mask, other=0.0)
    obase = out + b * C * HW
    tl.store(obase + pix, a0 / s, mask=mask)
    tl.store(obase + HW + pix, a1 / s, mask=mask)
    tl.store(obase + 2 * HW + pix, a2 / s, mask=mask)


def apply_kernel_fused(img, logits, K=15, BLOCK=256):
    B, C, H, W = img.shape
    assert C == 3 and logits.shape[1] == K * K
    img, logits = img.contiguous().float(), logits.contiguous().float()
    out = torch.empty_like(img)
    HW = H * W
    _kpn[(triton.cdiv(HW, BLOCK), B)](img, logits, out, H, W, HW, K=K, C=C, BLOCK=BLOCK)
    return out
