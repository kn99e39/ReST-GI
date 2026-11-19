"""WL106 Triton kernels (inference / no-grad parts of the coverage-moment-constrained operator, tilt106).

solve2: the tilt106.solve_torch Newton iteration for rows with <= 2 tilt coordinates (same steps, same candidate
        order and first-minimum rule; float32 moments, float64 2 x 2 solve).
apply_constrained_fused: softmax over (logit + lambda . c_q + MASK * forbidden_q) applied to linear radiance with
        reflect padding; equal to tilt106.constrain_logits followed by model105.apply_kernel (tested).
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _F(b, c0, c1, l0, l1, p0, p1, eps):
    a = b + l0 * c0 + l1 * c1
    mx = tl.max(a, 0)
    return mx + tl.log(tl.sum(tl.exp(a - mx), 0)) - l0 * p0 - l1 * p1 + 0.5 * eps * (l0 * l0 + l1 * l1)


@triton.jit
def _solve2(base, cq, cp, lam_out, eps, cap, ITERS: tl.constexpr, KK: tl.constexpr, BLK: tl.constexpr):
    n = tl.program_id(0)
    k = tl.arange(0, BLK)
    km = k < KK
    b = tl.load(base + n * KK + k, mask=km, other=-1e30)
    c0 = tl.load(cq + n * 2 * KK + k, mask=km, other=0.0)
    c1 = tl.load(cq + n * 2 * KK + KK + k, mask=km, other=0.0)
    p0 = tl.load(cp + 2 * n)
    p1 = tl.load(cp + 2 * n + 1)
    l0 = p0 * 0.0
    l1 = p0 * 0.0
    for it in range(ITERS):
        a = b + l0 * c0 + l1 * c1
        mx = tl.max(a, 0)
        e = tl.exp(a - mx)
        w = e / tl.sum(e, 0)
        m0 = tl.sum(w * c0, 0)
        m1 = tl.sum(w * c1, 0)
        h00 = (tl.sum(w * c0 * c0, 0) - m0 * m0).to(tl.float64) + eps
        h01 = (tl.sum(w * c0 * c1, 0) - m0 * m1).to(tl.float64)
        h11 = (tl.sum(w * c1 * c1, 0) - m1 * m1).to(tl.float64) + eps
        g0 = (m0 - p0 + eps * l0).to(tl.float64)
        g1 = (m1 - p1 + eps * l1).to(tl.float64)
        det = h00 * h11 - h01 * h01
        s0 = ((h11 * g0 - h01 * g1) / det).to(tl.float32)
        s1 = ((h00 * g1 - h01 * g0) / det).to(tl.float32)
        sc = tl.minimum(cap / tl.maximum(tl.maximum(tl.abs(s0), tl.abs(s1)), 1e-30), 1.0)
        s0 = s0 * sc
        s1 = s1 * sc
        best = _F(b, c0, c1, l0 - s0, l1 - s1, p0, p1, eps)
        bt = p0 * 0.0 + 1.0
        for j in tl.static_range(1, 4):
            t = 0.5 ** j
            f = _F(b, c0, c1, l0 - t * s0, l1 - t * s1, p0, p1, eps)
            better = f < best
            best = tl.where(better, f, best)
            bt = tl.where(better, t, bt)
        f = _F(b, c0, c1, l0, l1, p0, p1, eps)
        bt = tl.where(f < best, 0.0, bt)
        l0 = l0 - bt * s0
        l1 = l1 - bt * s1
    tl.store(lam_out + 2 * n, l0)
    tl.store(lam_out + 2 * n + 1, l1)


def solve2(base, cq, cp, iters, eps, steps, cap):
    assert tuple(steps) == (1.0, 0.5, 0.25, 0.125, 0.0)
    N, D, KK = cq.shape
    if D < 2:
        cq = torch.cat([cq, torch.zeros(N, 2 - D, KK, device=cq.device)], 1)
        cp = torch.cat([cp, torch.zeros(N, 2 - D, device=cp.device)], 1)
    base, cq, cp = base.contiguous().float(), cq.contiguous().float(), cp.contiguous().float()
    lam = torch.empty(N, 2, device=cq.device)
    if N:
        _solve2[(N,)](base, cq, cp, lam, eps, cap, ITERS=iters, KK=KK, BLK=triton.next_power_of_2(KK))
    return lam[:, :D]


@triton.jit
def _kpn_c(img, logits, sup, lam, ch, cov, out, H, W, HW, MASKV,
           K: tl.constexpr, C: tl.constexpr, NC: tl.constexpr, DF: tl.constexpr, BLOCK: tl.constexpr):
    b = tl.program_id(1)
    pix = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = pix < HW
    y = pix // W
    x = pix % W
    R: tl.constexpr = K // 2
    lbase = logits + b * (K * K) * HW
    sbase = sup + b * HW
    cbase = cov + b * NC * HW
    sp = tl.load(sbase + pix, mask=mask, other=0)
    nsp = ~sp
    l0 = tl.load(lam + (b * DF + 0) * HW + pix, mask=mask, other=0.0)
    l1 = tl.load(lam + (b * DF + 1) * HW + pix, mask=mask, other=0.0)
    l2 = tl.load(lam + (b * DF + 2) * HW + pix, mask=mask, other=0.0)
    l3 = tl.load(lam + (b * DF + 3) * HW + pix, mask=mask, other=0.0)
    k0 = tl.load(ch + (b * DF + 0) * HW + pix, mask=mask, other=0)
    k1 = tl.load(ch + (b * DF + 1) * HW + pix, mask=mask, other=0)
    k2 = tl.load(ch + (b * DF + 2) * HW + pix, mask=mask, other=0)
    k3 = tl.load(ch + (b * DF + 3) * HW + pix, mask=mask, other=0)
    m = tl.full((BLOCK,), -1e30, tl.float32)
    for k in range(K * K):
        yy = y + (k // K - R)
        xx = x + (k % K - R)
        yy = tl.where(yy < 0, -yy, yy)
        yy = tl.where(yy >= H, 2 * H - 2 - yy, yy)
        xx = tl.where(xx < 0, -xx, xx)
        xx = tl.where(xx >= W, 2 * W - 2 - xx, xx)
        q = yy * W + xx
        sh = (l0 * tl.load(cbase + k0 * HW + q, mask=mask & (l0 != 0), other=0.0)
              + l1 * tl.load(cbase + k1 * HW + q, mask=mask & (l1 != 0), other=0.0)
              + l2 * tl.load(cbase + k2 * HW + q, mask=mask & (l2 != 0), other=0.0)
              + l3 * tl.load(cbase + k3 * HW + q, mask=mask & (l3 != 0), other=0.0))
        forb = (tl.load(sbase + q, mask=mask, other=0) & nsp) != 0
        l = tl.load(lbase + k * HW + pix, mask=mask, other=-1e30) + tl.where(forb, MASKV, 0.0) + sh
        m = tl.maximum(m, l)
    s = tl.zeros((BLOCK,), tl.float32)
    a0 = tl.zeros((BLOCK,), tl.float32)
    a1 = tl.zeros((BLOCK,), tl.float32)
    a2 = tl.zeros((BLOCK,), tl.float32)
    ibase = img + b * C * HW
    for k in range(K * K):
        yy = y + (k // K - R)
        xx = x + (k % K - R)
        yy = tl.where(yy < 0, -yy, yy)
        yy = tl.where(yy >= H, 2 * H - 2 - yy, yy)
        xx = tl.where(xx < 0, -xx, xx)
        xx = tl.where(xx >= W, 2 * W - 2 - xx, xx)
        q = yy * W + xx
        sh = (l0 * tl.load(cbase + k0 * HW + q, mask=mask & (l0 != 0), other=0.0)
              + l1 * tl.load(cbase + k1 * HW + q, mask=mask & (l1 != 0), other=0.0)
              + l2 * tl.load(cbase + k2 * HW + q, mask=mask & (l2 != 0), other=0.0)
              + l3 * tl.load(cbase + k3 * HW + q, mask=mask & (l3 != 0), other=0.0))
        forb = (tl.load(sbase + q, mask=mask, other=0) & nsp) != 0
        l = tl.load(lbase + k * HW + pix, mask=mask, other=-1e30) + tl.where(forb, MASKV, 0.0) + sh
        e = tl.exp(l - m)
        s += e
        a0 += e * tl.load(ibase + q, mask=mask, other=0.0)
        a1 += e * tl.load(ibase + HW + q, mask=mask, other=0.0)
        a2 += e * tl.load(ibase + 2 * HW + q, mask=mask, other=0.0)
    obase = out + b * C * HW
    tl.store(obase + pix, a0 / s, mask=mask)
    tl.store(obase + HW + pix, a1 / s, mask=mask)
    tl.store(obase + 2 * HW + pix, a2 / s, mask=mask)


def apply_constrained_fused(img, logits, sup, lam_map, ch_map, cov10, K, MASKV, BLOCK=256):
    B, C, H, W = img.shape
    assert C == 3 and logits.shape[1] == K * K and lam_map.shape[1] == 4 and cov10.shape[1] == 10
    img, logits = img.contiguous().float(), logits.contiguous().float()
    sup_i = sup.round().int().contiguous()
    out = torch.empty_like(img)
    HW = H * W
    _kpn_c[(triton.cdiv(HW, BLOCK), B)](img, logits, sup_i, lam_map.contiguous(), ch_map.contiguous(), cov10.contiguous(),
                                         out, H, W, HW, MASKV, K=K, C=C, NC=10, DF=4, BLOCK=BLOCK)
    return out
