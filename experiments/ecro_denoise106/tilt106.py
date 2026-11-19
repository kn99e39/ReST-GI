"""WL106 coverage-moment-constrained kernels (geometry-consistent spatial mixing).

Coverage vector of a pixel: c = (8 x 8-subpixel coverage of the 9 shapes, empty fraction 1 - sum), i.e. the existing
WL105 guide plus its implied miss fraction; sum_s c_s = 1 and every value is a multiple of 1/64. For the predicted
logits z_q of pixel p (q over the K x K window, reflect-padded exactly as the kernel application) the applied kernel
is the minimum-KL modification of softmax(z) that satisfies the coverage moment

    sum_q w'_q c_q = c_p,   w' >= 0,   sum_q w'_q = 1.

For radiance that is constant per surface inside the window this makes sum_q w'_q E[x_q] = sum_s c_p,s L_s = E[x_p]:
the operator can neither import another surface's radiance nor remove the pixel's own surfaces.

Exact solution structure used here:
1. Support restriction (implied by w' >= 0): a neighbour holding any surface absent from p (c_q,s > 0 = c_p,s) gets
   zero weight; its logit is masked. After this, a pixel covered by a single surface already satisfies the moment
   (every admissible neighbour has the same vector), so only pixels covered by >= 2 surfaces are tilted.
2. Tilt: w'_q proportional to softmax(z)_q exp(lambda . c_q) on admissible neighbours, lambda with one coordinate per
   own surface but the largest one (implied by sum = 1), solving the convex dual
   F(lambda) = logsumexp(log softmax(z) + lambda . c) - lambda . c_p + (eps / 2)|lambda|^2 by ITERS Newton iterations
   with the Newton step capped at CAP (inf-norm) and a monotone step choice among STEPS, then ONE further Newton step taken differentiably (Hessian detached):
   its derivative is the implicit derivative of the converged constraint, so training sees the constrained operator.
The same lambda routine is used in training (reference path, unfold application) and inference (fused Triton
application); kpn_triton106 holds the Triton solver for <= 2 tilt coordinates and the fused masked/tilted application.
No parameters are added; EPS, ITERS and STEPS are fixed operator constants.
"""
import torch
import torch.nn.functional as F

K = 15
EPS = 1e-6
ITERS = 8
STEPS = (1.0, 0.5, 0.25, 0.125, 0.0)
CAP = 16.0          # max |Newton step|_inf in lambda units (logit change per unit coverage); path safeguard only
MASK = -1.0e4
DFUSED = 4          # fused application supports up to 4 tilt coordinates (5 own surfaces)
USE_TRITON = True   # Triton solver for rows with <= 2 tilt coordinates (equivalent to the torch solver; tested)


def coverage10(cov):
    """(B,9,H,W) shape coverage -> (B,10,H,W) with the miss fraction appended (exact for multiples of 1/64)."""
    cov = cov.float()
    return torch.cat([cov, (1.0 - cov.sum(1, keepdim=True)).clamp_min(0.0)], 1)


def support_bits(c):
    """(B,10,H,W) -> (B,1,H,W) float32 bitmask of the surfaces present (exact integers < 1024)."""
    w = 2.0 ** torch.arange(10, device=c.device, dtype=torch.float32)
    return ((c > 0).float() * w[None, :, None, None]).sum(1, keepdim=True)


def active_pixels(c, K=K):
    """(B,H,W) bool: the reflect-padded K x K window holds more than one coverage vector."""
    r = K // 2
    p = F.pad(c, (r, r, r, r), mode="reflect")
    return (F.max_pool2d(p, K, 1) + F.max_pool2d(-p, K, 1)).amax(1) > 0


def gather_window(x, idx, K=K):
    """x (B,C,H,W); idx (N,3) [b, y, x] -> (N,C,K*K) in the unfold ordering (row-major dy, dx), reflect padding."""
    r = K // 2
    pad = F.pad(x, (r, r, r, r), mode="reflect")
    dy = torch.arange(K, device=x.device).repeat_interleave(K)
    dx = torch.arange(K, device=x.device).repeat(K)
    yy = idx[:, 1:2] + dy[None]
    xx = idx[:, 2:3] + dx[None]
    bb = idx[:, 0:1].expand_as(yy)
    return pad[bb, :, yy, xx].permute(0, 2, 1).contiguous()


def chol(H):
    """Batched Cholesky factor of small SPD matrices (N,d,d), float64, explicit loops."""
    d = H.shape[1]
    L = torch.zeros_like(H)
    for j in range(d):
        L[:, j, j] = (H[:, j, j] - (L[:, j, :j] ** 2).sum(1)).clamp_min(1e-300).sqrt()
        for i in range(j + 1, d):
            L[:, i, j] = (H[:, i, j] - (L[:, i, :j] * L[:, j, :j]).sum(1)) / L[:, j, j]
    return L


def chol_solve(L, g):
    """Solve (L L^T) x = g for (N,d) g; differentiable in g (L is treated as a constant)."""
    d = L.shape[1]
    y = []
    for i in range(d):
        acc = g[:, i]
        for j in range(i):
            acc = acc - L[:, i, j] * y[j]
        y.append(acc / L[:, i, i])
    x = [None] * d
    for i in reversed(range(d)):
        acc = y[i]
        for j in range(i + 1, d):
            acc = acc - L[:, j, i] * x[j]
        x[i] = acc / L[:, i, i]
    return torch.stack(x, 1)


def _objective(base, lam, cq, cp, eps):
    return torch.logsumexp(base + torch.einsum("...ns,nsk->...nk", lam, cq), -1) - (lam * cp).sum(-1) + 0.5 * eps * (lam * lam).sum(-1)


def _hess(w, cq, eps):
    m = torch.einsum("nk,nsk->ns", w, cq)
    H = torch.bmm((cq * w[:, None]).double(), cq.transpose(1, 2).double()) - (m[:, :, None] * m[:, None, :]).double()
    return m, H + eps * torch.eye(cq.shape[1], device=cq.device, dtype=torch.float64)[None]


@torch.no_grad()
def solve_torch(base, cq, cp, iters=ITERS, eps=EPS):
    N, S, _ = cq.shape
    lam = torch.zeros(N, S, device=cq.device)
    ts = torch.tensor(STEPS, device=cq.device)[:, None, None]
    ar = torch.arange(N, device=cq.device)
    for _ in range(iters):
        w = torch.softmax(base + torch.einsum("ns,nsk->nk", lam, cq), 1)
        m, H = _hess(w, cq, eps)
        step = chol_solve(chol(H), (m - cp + eps * lam).double()).float()
        step = step * torch.clamp(CAP / step.abs().amax(1, keepdim=True).clamp_min(1e-30), max=1.0)
        cand = lam[None] - ts * step[None]
        lam = cand[_objective(base[None], cand, cq, cp[None], eps).argmin(0), ar]
    return lam


@torch.no_grad()
def solve(base, cq, cp, iters=ITERS, eps=EPS):
    if USE_TRITON and cq.is_cuda and cq.shape[1] <= 2:
        from kpn_triton106 import solve2
        return solve2(base, cq, cp, iters, eps, STEPS, CAP)
    return solve_torch(base, cq, cp, iters, eps)


def tilt_rows(z, c, idx, cp, iters=ITERS, eps=EPS):
    """Rows of masked logits z (N,KK) of pixels idx with own coverage cp (N,10), all with >= 2 own surfaces.
    Returns (lam (N,D), chans (N,D) long, cq (N,D,KK)); lam is differentiable w.r.t. z. D = max own surfaces - 1;
    rows with fewer coordinates are padded with inert zeros (cq = cp = 0, lam = 0)."""
    ns = (cp > 0).sum(1)
    D = int(ns.max()) - 1
    chans = torch.argsort(cp, dim=1, descending=True, stable=True)[:, 1:D + 1]
    cpr = torch.gather(cp, 1, chans)
    on = cpr > 0
    chans = torch.where(on, chans, torch.zeros_like(chans))
    cq = torch.gather(gather_window(c, idx), 1, chans[..., None].expand(-1, -1, K * K)) * on[..., None]
    cpr = cpr * on
    base = torch.log_softmax(z, 1)
    lam0 = solve(base.detach(), cq, cpr, iters, eps)
    with torch.no_grad():
        w0 = torch.softmax(base.detach() + torch.einsum("ns,nsk->nk", lam0, cq), 1)
        L = chol(_hess(w0, cq, eps)[1])
    w = torch.softmax(base + torch.einsum("ns,nsk->nk", lam0, cq), 1)
    g = torch.einsum("nk,nsk->ns", w, cq) - cpr + eps * lam0
    lam = lam0 - chol_solve(L, g.double()).to(g.dtype)
    return lam * on, chans, cq


def constrain_logits(logits, cov, iters=ITERS, eps=EPS, stats=None):
    """Reference / training path. logits (B,K*K,H,W), cov (B,9,H,W) -> constrained logits, differentiable.
    stats (dict, optional) receives active / tilted fractions and the final |moment residual|_inf per active pixel."""
    c = coverage10(cov)
    act = active_pixels(c)
    idx = act.nonzero()
    if idx.numel() == 0:
        return logits
    sup = support_bits(c)
    sq = gather_window(sup, idx)[:, 0].long()
    cp = c[idx[:, 0], :, idx[:, 1], idx[:, 2]]
    sp = sup[idx[:, 0], 0, idx[:, 1], idx[:, 2]].long()
    z = logits[idx[:, 0], :, idx[:, 1], idx[:, 2]].float() + MASK * ((sq & ~sp[:, None]) != 0)
    sel = ((cp > 0).sum(1) >= 2).nonzero()[:, 0]
    if sel.numel():
        lam, chans, cq = tilt_rows(z[sel], c, idx[sel], cp[sel], iters, eps)
        z = z.index_add(0, sel, torch.einsum("ns,nsk->nk", lam, cq))
    if stats is not None:
        with torch.no_grad():
            wf = torch.softmax(z, 1)
            stats["active_fraction"] = float(act.float().mean())
            stats["tilted_fraction"] = float(sel.numel()) / act.numel()
            stats["max_own_surfaces"] = int((cp > 0).sum(1).max())
            stats["residual"] = (torch.einsum("nk,nsk->ns", wf, gather_window(c, idx)) - cp).abs().amax(1)
    out = logits.permute(0, 2, 3, 1).index_put((idx[:, 0], idx[:, 1], idx[:, 2]), z.to(logits.dtype))
    return out.permute(0, 3, 1, 2)


@torch.no_grad()
def constrained_apply_fused(img, logits, cov, iters=ITERS, eps=EPS):
    """Inference path: identical operator, fused masked / tilted softmax application without materialising the
    constrained logits. Falls back to the reference path if a pixel has more than DFUSED tilt coordinates."""
    from kpn_triton106 import apply_constrained_fused
    c = coverage10(cov)
    sup = support_bits(c)
    ns = (c > 0).sum(1)
    idx = (ns >= 2).nonzero()
    B, _, H, W = img.shape
    lam_map = torch.zeros(B, DFUSED, H, W, device=img.device)
    ch_map = torch.zeros(B, DFUSED, H, W, device=img.device, dtype=torch.int32)
    if idx.numel():
        if int(ns.max()) - 1 > DFUSED:
            return None
        sq = gather_window(sup, idx)[:, 0].long()
        sp = sup[idx[:, 0], 0, idx[:, 1], idx[:, 2]].long()
        cp = c[idx[:, 0], :, idx[:, 1], idx[:, 2]]
        z = logits[idx[:, 0], :, idx[:, 1], idx[:, 2]].float() + MASK * ((sq & ~sp[:, None]) != 0)
        lam, chans, _ = tilt_rows(z, c, idx, cp, iters, eps)
        D = lam.shape[1]
        lam_map.permute(0, 2, 3, 1)[idx[:, 0], idx[:, 1], idx[:, 2], :D] = lam
        ch_map.permute(0, 2, 3, 1)[idx[:, 0], idx[:, 1], idx[:, 2], :D] = chans.int()
    return apply_constrained_fused(img, logits, sup, lam_map, ch_map, c, K, MASK)
