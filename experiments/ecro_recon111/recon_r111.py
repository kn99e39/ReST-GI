"""WL111 candidate reconstructor (one design): compact shared-trunk kernel predictor + per-component normalized
nonnegative dilated (a-trous) spatial kernels + causal validated temporal accumulation of the raw LOW / HIGH observations.

Pipeline (ST; S is the same code with history disabled):
  current LOW = E1 + E2 and HIGH = HL means (one frame, spp samples)
    -> temporal accumulation (validated history; global reset on transport-changing edits)
       per component, same weights for LOW and HIGH: running mean with a sample cap, inverse-ESS tracked exactly
    -> compact network on [log1p(LOW_acc), log1p(HIGH_acc), coverage(9), normal(3), depth/5, log2(ESS)/11]
       predicts, for each of 3 passes (dilation 1, 2, 4) and each component, a 5 x 5 softmax kernel
    -> LOW' = K_low3 * K_low2 * K_low1 * LOW_acc, HIGH' likewise (each kernel normalized and nonnegative)
    -> TOTAL = LOW' + HIGH' (HDR; no clipping)

Reference (torch, autograd) and fused (Triton, inference) implementations are provided for the spatial pass and the
temporal step; their equality is Stage E contract 13.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as Fn

KT = 5                    # taps per axis of one pass
DILATIONS = (1, 2, 4)     # effective support 1 + 2 * 2 * (1 + 2 + 4) = 29 px
NCOMP = 2                 # LOW, HIGH
HEAD = len(DILATIONS) * NCOMP * KT * KT      # 150 logits per pixel: [pass][component][tap]
CIN = 20
BASE_WIDTHS = (16, 32, 48, 64)
ESS_SCALE = 11.0          # log2(ESS) / 11 (ESS <= 2048 in the training ladder)


def widths(m):
    return tuple(int(round(c * m)) for c in BASE_WIDTHS)


def conv(cin, cout, k=3, s=1):
    return nn.Conv2d(cin, cout, k, stride=s, padding=k // 2)


class CompactNet(nn.Module):
    """Shared trunk: full-resolution stem, three stride-2 levels (1/2, 1/4, 1/8), decoder back to 1/2, a full-resolution
    1 x 1 fusion with the stem features, and a 1 x 1 head with component-specific kernel logits."""

    def __init__(self, m=1.0):
        super().__init__()
        c0, c1, c2, c3 = widths(m)
        self.stem = conv(CIN, c0)
        self.d1, self.e1 = conv(c0, c1, s=2), conv(c1, c1)
        self.d2, self.e2 = conv(c1, c2, s=2), conv(c2, c2)
        self.d3, self.e3 = conv(c2, c3, s=2), conv(c3, c3)
        self.u2 = conv(c3 + c2, c2)
        self.u1 = conv(c2 + c1, c1)
        self.fuse = conv(c1 + c0, 2 * c0, k=1)
        self.head = conv(2 * c0, HEAD, k=1)
        nn.init.zeros_(self.head.bias)
        nn.init.normal_(self.head.weight, std=1e-3)
        self.m = m

    def forward(self, x):
        r = Fn.relu
        s0 = r(self.stem(x))
        s1 = r(self.e1(r(self.d1(s0))))
        s2 = r(self.e2(r(self.d2(s1))))
        s3 = r(self.e3(r(self.d3(s2))))
        u = Fn.interpolate(s3, size=s2.shape[-2:], mode="bilinear", align_corners=False)
        u = r(self.u2(torch.cat([u, s2], 1)))
        u = Fn.interpolate(u, size=s1.shape[-2:], mode="bilinear", align_corners=False)
        u = r(self.u1(torch.cat([u, s1], 1)))
        u = Fn.interpolate(u, size=s0.shape[-2:], mode="bilinear", align_corners=False)
        f = r(self.fuse(torch.cat([u, s0], 1)))
        if f.shape[0] == 1 and f.is_contiguous(memory_format=torch.channels_last) and not torch.is_grad_enabled():
            # inference (WL111 execution fix 2): the 1 x 1 head as one GEMM on the channels_last features (a view of an
            # (H W, C) matrix), producing NCHW logits directly; a cuDNN head on channels_last features returns
            # channels_last logits whose NCHW conversion cost about 3.3 ms of device time at 512 x 512
            _, Cf, H, W = f.shape
            fm = f.permute(0, 2, 3, 1).reshape(H * W, Cf)
            wt = self.head.weight.view(HEAD, Cf)
            out = torch.addmm(self.head.bias[:, None].to(fm.dtype), wt.to(fm.dtype), fm.t())
            return out.view(1, HEAD, H, W)
        return self.head(f.contiguous())


def n_params(m):
    return sum(p.numel() for p in m.parameters())


def features(low, high, cover, normal, depth, ess):
    """low, high (B,3,H,W) accumulated means; cover (B,9,H,W); normal (B,3,H,W); depth (B,1,H,W); ess (B,1,H,W)."""
    return torch.cat([torch.log1p(low), torch.log1p(high), cover, normal, depth / 5.0,
                      torch.log2(ess.clamp(min=1.0)) / ESS_SCALE], 1)


# ------------------------------------------------------------------------------------------------ spatial passes
def atrous_ref(img, logits, d):
    """img (B,3,H,W) fp32; logits (B,25,H,W); one dilated 5 x 5 softmax-kernel pass with reflect padding."""
    B, Cc, H, W = img.shape
    r = d * (KT // 2)
    w = torch.softmax(logits.float(), 1)
    patches = Fn.unfold(Fn.pad(img, (r, r, r, r), mode="reflect"), KT, dilation=d).view(B, Cc, KT * KT, H, W)
    return (patches * w[:, None]).sum(2)


def split_logits(logits):
    """(B,150,H,W) -> [pass][component] -> (B,25,H,W)."""
    B, _, H, W = logits.shape
    v = logits.view(B, len(DILATIONS), NCOMP, KT * KT, H, W)
    return [[v[:, p, c] for c in range(NCOMP)] for p in range(len(DILATIONS))]


def spatial_ref(low, high, logits):
    L = split_logits(logits)
    comps = [low, high]
    for p, d in enumerate(DILATIONS):
        comps = [atrous_ref(comps[c], L[p][c], d) for c in range(NCOMP)]
    return comps[0], comps[1]


_TRITON = None


def _triton():
    global _TRITON
    if _TRITON is None:
        import triton
        import kern_r111 as K
        _TRITON = {"atrous": K._atrous, "temporal": K._temporal, "triton": triton}
    return _TRITON


def spatial_fused(low, high, logits, block=256):
    """Inference: low/high (1,3,H,W) fp32, logits (1,150,H,W) fp16/fp32 in NCHW or channels_last layout, read in place
    through its strides (WL111 execution fix 2: the former .contiguous() conversion of channels_last fp16 logits cost
    about 3.3 ms of device time at 512 x 512, against 0.28 ms for the three passes)."""
    T = _triton()
    _, _, H, W = low.shape
    HW = H * W
    cur = torch.cat([low, high], 0).contiguous().float()          # (2,3,H,W)
    nxt = torch.empty_like(cur)
    sN, sC, sH, sW = logits.stride()
    assert sH == W * sW, "pixel-linear layout required (NCHW or channels_last)"
    for p, d in enumerate(DILATIONS):
        lp = logits[:, p * NCOMP * 25:(p + 1) * NCOMP * 25]            # view: same strides, offset to pass p
        T["atrous"][(T["triton"].cdiv(HW, block), NCOMP)](cur, lp, nxt, H, W, HW, sC, sW, D=d, BLOCK=block)
        cur, nxt = nxt, cur
    return cur[0:1], cur[1:2]


# ------------------------------------------------------------------------------------------------ temporal history
def temporal_par(prev_cam, cam, spp, cfg, basis):
    """Parameter vector of the fused temporal kernel (layout documented in kern_r111._temporal)."""
    co, cf, cr, cu, ct = basis(prev_cam)
    o = basis(cam)[0]
    return ([float(v) for v in co] + [float(v) for v in cr] + [float(v) for v in cu] + [float(v) for v in cf] + [float(ct)]
            + [float(v) for v in o] + [float(spp), float(cfg["cap_frames"] * spp), float(cfg["normal_min_cos"]),
                                       float(cfg["position_footprint_pixels"]), float(cfg["sss_min_view_cos"])])


class History:
    """Causal per-pixel history of the raw LOW / HIGH observations of ONE camera stream.

    Ownership: one History per (renderer stream, resolution); it never reads the reference or future frames.
    Lifetime: until reset() or the next transport-changing edit; each pixel's samples are capped at smax = cap_frames x spp.
    Validity (camera motion only; transport state unchanged): nearest previous pixel of the current centre-ray point,
      inside the previous frame, same centre shape id, normal agreement, position within a pixel-footprint radius,
      pure sub-pixel coverage now and then (no cross-surface accumulation at mixed pixels), unoccluded from the previous
      eye (WL101B visibility query), and for the view-dependent SSS sphere a view-direction gate.
    Reset: first frame, reset(), resolution change and ANY transport revision change (geometry or light edit): every
      pixel restarts from the current observation (global conservative invalidation; no selective reuse)."""

    def __init__(self, n, device, cfg):
        self.n = n
        self.cfg = cfg
        self.acc = torch.zeros(6, n, device=device)
        self.cnt = torch.zeros(n, device=device)
        self.rinv = torch.zeros(n, device=device)
        self.age = torch.zeros(n, device=device)
        self.nacc, self.ncnt = torch.empty_like(self.acc), torch.empty_like(self.cnt)
        self.nrinv, self.nage = torch.empty_like(self.rinv), torch.empty_like(self.age)
        self.map = torch.empty(n, dtype=torch.int32, device=device)
        self.cen = torch.zeros(7, n, device=device)
        self.pure = torch.zeros(n, dtype=torch.int8, device=device)
        self.cam = None
        self.revision = None
        self.valid = False
        self.dummy = torch.ones(n, dtype=torch.int8, device=device)

    def reset(self):
        self.valid = False

    def step(self, low, high, spp, centre, pure, cam, revision, basis, hidden=None, fused=True):
        """low/high (3,n) current means; centre (7,n); pure (n) bool; cam dict; basis = pt86.cam_basis.
        -> (acc_low (3,n), acc_high (3,n), ess (n), age (n), accept (n bool), moving flag)."""
        cfg = self.cfg
        valid = self.valid and revision == self.revision and self.cam is not None
        moving = valid and cam != self.cam
        if moving and hidden is None:
            raise ValueError("camera reprojection requires the previous-eye visibility mask")
        smax = float(cfg["cap_frames"] * spp)
        if fused:
            T = _triton()
            par = torch.tensor(temporal_par(self.cam if moving else cam, cam, spp, cfg, basis), dtype=torch.float32,
                               device=low.device)
            hid = hidden.to(torch.int8) if moving else self.dummy
            T["temporal"][(T["triton"].cdiv(self.n, 256),)](
                low.contiguous(), high.contiguous(), centre.contiguous(), pure.to(torch.int8).contiguous(), self.cen,
                self.pure, hid, self.acc, self.cnt, self.rinv, self.age,
                self.nacc, self.ncnt, self.nrinv, self.nage, self.map, par, self.n, int(round(math.sqrt(self.n))),
                VALID=bool(valid), MOVING=bool(moving), BZ=256)
        else:
            temporal_ref(self, low, high, spp, centre, pure, cam, basis, hidden, valid, moving, smax)
        self.acc, self.nacc = self.nacc, self.acc
        self.cnt, self.ncnt = self.ncnt, self.cnt
        self.rinv, self.nrinv = self.nrinv, self.rinv
        self.age, self.nage = self.nage, self.age
        self.cen.copy_(centre)
        self.pure.copy_(pure.to(torch.int8))
        self.cam, self.revision, self.valid = dict(cam), revision, True
        accept = self.map >= 0
        return self.acc[0:3], self.acc[3:6], 1.0 / self.rinv, self.age, accept, moving


def temporal_ref(h, low, high, spp, centre, pure, cam, basis, hidden, valid, moving, smax):
    """Torch reference of the fused temporal step (same float32 statements); writes h.nacc / ncnt / nrinv / nage / map."""
    cfg = h.cfg
    n = h.n
    W = int(round(math.sqrt(n)))
    i = torch.arange(n, device=low.device)
    if not valid:
        ok = torch.zeros(n, dtype=torch.bool, device=low.device)
        j = i
    elif not moving:
        ok = torch.ones(n, dtype=torch.bool, device=low.device)
        j = i
    else:
        co, cf, cr, cu, ct = basis(h.cam)
        o = basis(cam)[0]
        f32 = lambda v: torch.tensor(float(v), dtype=torch.float32, device=low.device)
        px, py, pz = centre[0], centre[1], centre[2]
        dx, dy, dz = px - f32(co[0]), py - f32(co[1]), pz - f32(co[2])
        xr = f32(cr[0]) * dx + f32(cr[1]) * dy + f32(cr[2]) * dz
        yu = f32(cu[0]) * dx + f32(cu[1]) * dy + f32(cu[2]) * dz
        zf = f32(cf[0]) * dx + f32(cf[1]) * dy + f32(cf[2]) * dz
        den = zf.clamp(min=1e-6) * f32(ct)
        xx = (xr / den + 1.0) * W / 2 - 0.5
        yy = (1.0 - yu / den) * W / 2 - 0.5
        ix = torch.floor(xx + 0.5).to(torch.int32)
        iy = torch.floor(yy + 0.5).to(torch.int32)
        inside = (ix >= 0) & (ix < W) & (iy >= 0) & (iy < W) & (zf > 0)
        j = (iy.clamp(0, W - 1) * W + ix.clamp(0, W - 1)).long()
        q = h.cen[:, j]
        dist = torch.sqrt((px - q[0]) ** 2 + (py - q[1]) ** 2 + (pz - q[2]) ** 2)
        foot = f32(cfg["position_footprint_pixels"]) * zf * (2.0 * f32(ct) / W)
        ndot = centre[3] * q[3] + centre[4] * q[4] + centre[5] * q[5]
        a = torch.stack([f32(co[k]) - centre[k] for k in range(3)])
        b = torch.stack([f32(o[k]) - centre[k] for k in range(3)])
        vcos = (a * b).sum(0) / (a.norm(dim=0).clamp(min=1e-12) * b.norm(dim=0).clamp(min=1e-12))
        sid = centre[6]
        view_ok = (sid != 6.0) | (vcos >= cfg["sss_min_view_cos"])
        ok = (inside & (sid >= 0) & (sid == q[6]) & (ndot >= cfg["normal_min_cos"]) & (dist <= foot) & view_ok
              & pure.bool() & (h.pure[j] != 0) & ~hidden.bool())
    jj = j.long() if torch.is_tensor(j) else j
    n_prev = torch.where(ok, h.cnt[jj], 0.0)
    r_prev = torch.where(ok, h.rinv[jj], 0.0)
    a_prev = torch.where(ok, h.age[jj], 0.0)
    n_new = torch.where(ok, torch.clamp(n_prev + spp, max=smax), torch.full_like(n_prev, float(spp)))
    w = spp / n_new
    h.nrinv.copy_(torch.where(ok, (1.0 - w) ** 2 * r_prev + w * w / spp, torch.full_like(w, 1.0 / spp)))
    x = torch.cat([low, high], 0)
    hp = h.acc[:, jj]
    h.nacc.copy_(torch.where(ok[None], hp + w[None] * (x - hp), x))
    h.ncnt.copy_(n_new)
    h.nage.copy_(torch.where(ok, a_prev + 1.0, torch.ones_like(a_prev)))
    h.map.copy_(torch.where(ok, jj.to(torch.int32), torch.full_like(jj, -1).to(torch.int32)))


# ------------------------------------------------------------------------------------------------ reconstructor
class Candidate:
    """S / ST runtime around one frozen CompactNet. Inference: fp16 autocast + channels_last for the network, fp32 for
    the spatial passes and the history."""

    def __init__(self, net, cfg, device="cuda", fused=True):
        self.net = net.eval().to(device).to(memory_format=torch.channels_last)
        self.cfg = cfg
        self.fused = fused
        self.dev = device

    @torch.no_grad()
    def spatial(self, low, high, ess, g):
        """low/high (3,H,W); ess (H,W); g guides dict (cover, normal, depth). -> low', high' (3,H,W)."""
        H, W = low.shape[-2:]
        x = features(low[None], high[None], g["cover"][None], g["normal"][None], g["depth"][None, None], ess.view(1, 1, H, W))
        with torch.autocast("cuda", dtype=torch.float16):
            logits = self.net(x.to(memory_format=torch.channels_last))
        if self.fused:
            lo, hi = spatial_fused(low[None].float(), high[None].float(), logits)
        else:
            lo, hi = spatial_ref(low[None].float(), high[None].float(), logits.float())
        return lo[0], hi[0]


class GraphCandidate:
    """CUDA-graph runtime of S / ST around one frozen CompactNet (inference execution only; same kernels as the eager
    path, equality tested). One graph per mode -- S (no history), reset, static (same camera), moving (camera motion) --
    captured once at construction on static buffers; per frame the host decides the mode (history ownership, lifetime
    and validity logic of History), copies the current inputs into the static buffers, writes the parameter vector and
    replays one graph."""

    MODES = ("S", "reset", "static", "moving")

    def __init__(self, cand, n, history):
        self.cand, self.cfg, self.history = cand, cand.cfg, history
        dev = cand.dev
        self.n = n
        self.Wd = int(round(math.sqrt(n)))
        H = W = self.Wd
        z = lambda *sh, dt=torch.float32: torch.zeros(*sh, dtype=dt, device=dev)
        self.lo_in, self.hi_in = z(3, n), z(3, n)
        self.cover, self.normal, self.depth = z(9, H, W), z(3, H, W), z(H, W)
        self.cen, self.pure, self.hid = z(7, n), z(n, dt=torch.int8), z(n, dt=torch.int8)
        self.par = z(24)
        self.acc, self.cnt, self.rinv, self.age = z(6, n), z(n), z(n), z(n)
        self.nacc, self.ncnt, self.nrinv, self.nage = z(6, n), z(n), z(n), z(n)
        self.map = z(n, dt=torch.int32)
        self.pcen, self.ppure = z(7, n), z(n, dt=torch.int8)
        self.out = z(3, H, W)
        self.par.copy_(torch.tensor([0.0] * 16 + [1.0, 32.0, 0.9, 3.0, 0.995] + [0.0] * 3))
        self.graphs = {}
        for mode in (self.MODES if history else ("S",)):
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(2):
                    self._body(mode)
            torch.cuda.current_stream().wait_stream(s)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                self._body(mode)
            self.graphs[mode] = g
        torch.cuda.synchronize()
        self.valid, self.cam, self.revision = False, None, None

    def _body(self, mode):
        H = W = self.Wd
        if mode == "S":
            low, high = self.lo_in, self.hi_in
            ess = self.par[16:17].expand(self.n)
        else:
            T = _triton()
            T["temporal"][(T["triton"].cdiv(self.n, 256),)](
                self.lo_in, self.hi_in, self.cen, self.pure, self.pcen, self.ppure, self.hid, self.acc, self.cnt, self.rinv,
                self.age, self.nacc, self.ncnt, self.nrinv, self.nage, self.map, self.par, self.n, W,
                VALID=mode != "reset", MOVING=mode == "moving", BZ=256)
            self.acc.copy_(self.nacc)
            self.cnt.copy_(self.ncnt)
            self.rinv.copy_(self.nrinv)
            self.age.copy_(self.nage)
            self.pcen.copy_(self.cen)
            self.ppure.copy_(self.pure)
            low, high = self.acc[0:3], self.acc[3:6]
            ess = 1.0 / self.rinv
        L, Hh = low.reshape(1, 3, H, W), high.reshape(1, 3, H, W)
        x = features(L, Hh, self.cover[None], self.normal[None], self.depth[None, None], ess.reshape(1, 1, H, W))
        with torch.autocast("cuda", dtype=torch.float16):
            logits = self.cand.net(x.to(memory_format=torch.channels_last))
        lo, hi = spatial_fused(L, Hh, logits)
        self.out.copy_((lo + hi)[0])

    def reset(self):
        self.valid = False

    def mode_for(self, cam, revision):
        if not self.history:
            return "S"
        valid = self.valid and revision == self.revision and self.cam is not None
        if not valid:
            return "reset"
        return "moving" if cam != self.cam else "static"

    @torch.no_grad()
    def run(self, low, high, spp, g, cam, revision, basis):
        """low / high (3,H,W) current means; g = engine.pack(...) of this frame (cover, normal, depth, and for ST centre,
        pure and hidden w.r.t. the previous camera). -> TOTAL (3,H,W) (a view of the static output buffer)."""
        mode = self.mode_for(cam, revision)
        prev = self.cam if mode == "moving" else cam
        self.par.copy_(torch.tensor(temporal_par(prev, cam, spp, self.cfg, basis) + [0.0] * 3, dtype=torch.float32))
        self.lo_in.copy_(low.reshape(3, self.n))
        self.hi_in.copy_(high.reshape(3, self.n))
        self.cover.copy_(g["cover"])
        self.normal.copy_(g["normal"])
        self.depth.copy_(g["depth"])
        if self.history:
            self.cen.copy_(g["centre"])
            self.pure.copy_(g["pure"].reshape(self.n))
            if mode == "moving":
                self.hid.copy_(g["hidden"].reshape(self.n))
        self.graphs[mode].replay()
        if self.history:
            self.valid, self.cam, self.revision = True, dict(cam), revision
        self.last_mode = mode
        return self.out


def raw_low_high(comp):
    """comp (9,H,W) component means -> LOW (3,H,W) = E1 + E2, HIGH (3,H,W) = HL."""
    return comp[0:3] + comp[3:6], comp[6:9]


def rel_mse(pred, target):
    """WL105 relative MSE (mean-seeking): mean((y - t)^2 / (sg(y)^2 + 0.01))."""
    return ((pred - target) ** 2 / (pred.detach() ** 2 + 0.01)).mean()
