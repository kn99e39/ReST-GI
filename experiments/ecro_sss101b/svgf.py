"""Joint-radiance SVGF adaptation for the unchanged ECRO frame."""

import numpy as np
import torch
import triton
import triton.language as tl


@triton.jit
def _compose(A, B, C, D, E, F, G, H, I, O, N: tl.constexpr, S: tl.constexpr, BZ: tl.constexpr):
    i = tl.program_id(0) * BZ + tl.arange(0, BZ)
    m = i < N
    tl.store(O + i, (tl.load(A+i, m, 0) + tl.load(D+i, m, 0) + tl.load(G+i, m, 0)) / S, m)
    tl.store(O + N+i, (tl.load(B+i, m, 0) + tl.load(E+i, m, 0) + tl.load(H+i, m, 0)) / S, m)
    tl.store(O + 2*N+i, (tl.load(C+i, m, 0) + tl.load(F+i, m, 0) + tl.load(I+i, m, 0)) / S, m)


@triton.jit
def _luma(r, g, b):
    return 0.2126*r + 0.7152*g + 0.0722*b


@triton.jit
def _temporal(R, Old, Map, New, N: tl.constexpr, VALID: tl.constexpr, MOVING: tl.constexpr,
              LIMIT: tl.constexpr, BZ: tl.constexpr):
    i = tl.program_id(0) * BZ + tl.arange(0, BZ)
    m = i < N
    r, g, b = tl.load(R+i, m, 0), tl.load(R+N+i, m, 0), tl.load(R+2*N+i, m, 0)
    y = _luma(r, g, b)
    yy = y*y
    count = tl.full((BZ,), 1.0, tl.float32)
    if VALID:
        j = i
        if MOVING:
            j = tl.load(Map+i, m, -1)
        keep = m & (j >= 0)
        count = tl.minimum(LIMIT, 1 + tl.load(Old + 5*N+j, keep, 0))
        w = 1 - 1/count
        r += w*(tl.load(Old+j, keep, 0)-r)
        g += w*(tl.load(Old+N+j, keep, 0)-g)
        b += w*(tl.load(Old+2*N+j, keep, 0)-b)
        y += w*(tl.load(Old+3*N+j, keep, 0)-y)
        yy += w*(tl.load(Old+4*N+j, keep, 0)-yy)
    tl.store(New+i, r, m)
    tl.store(New+N+i, g, m)
    tl.store(New+2*N+i, b, m)
    tl.store(New+3*N+i, y, m)
    tl.store(New+4*N+i, yy, m)
    tl.store(New+5*N+i, count, m)


@triton.jit
def _guide(G, i, m, N: tl.constexpr):
    return (tl.load(G+i, m, 0), tl.load(G+N+i, m, 0), tl.load(G+2*N+i, m, 0),
            tl.load(G+3*N+i, m, 0), tl.load(G+4*N+i, m, 0), tl.load(G+5*N+i, m, 0),
            tl.load(G+6*N+i, m, -2))


@triton.jit
def _variance(H, G, A, W: tl.constexpr, BZ: tl.constexpr):
    N: tl.constexpr = W*W
    i = tl.program_id(0) * BZ + tl.arange(0, BZ)
    m = i < N
    x, y = i % W, i // W
    _, _, _, nx, ny, nz, sid = _guide(G, i, m, N)
    mom = tl.load(H+3*N+i, m, 0)
    mom2 = tl.load(H+4*N+i, m, 0)
    count = tl.load(H+5*N+i, m, 1)
    s1, s2, wt = mom, mom2, tl.full((BZ,), 1., tl.float32)
    for dy in tl.static_range(-1, 2):
        for dx in tl.static_range(-1, 2):
            if dx != 0 or dy != 0:
                j = i + dy*W + dx
                inside = m & (x+dx >= 0) & (x+dx < W) & (y+dy >= 0) & (y+dy < W)
                _, _, _, qx, qy, qz, qsid = _guide(G, j, inside, N)
                ok = inside & (sid == qsid) & ((nx*qx + ny*qy + nz*qz > 0.9) | (sid < 0))
                s1 += tl.load(H+3*N+j, ok, 0)
                s2 += tl.load(H+4*N+j, ok, 0)
                wt += ok.to(tl.float32)
    temporal = tl.maximum(0.0, mom2 - mom*mom)
    spatial = tl.maximum(0.0, s2/wt - (s1/wt)*(s1/wt))
    v = tl.maximum(temporal, spatial * tl.maximum(0.0, (4-count)/3)) / count
    for c in tl.static_range(0, 3):
        tl.store(A+c*N+i, tl.load(H+c*N+i, m, 0), m)
    tl.store(A+3*N+i, v, m)


@triton.jit
def _atrous(A, G, O, W: tl.constexpr, STEP: tl.constexpr, SCALE: tl.constexpr, BZ: tl.constexpr):
    N: tl.constexpr = W*W
    i = tl.program_id(0) * BZ + tl.arange(0, BZ)
    m = i < N
    x, y = i % W, i // W
    px, py, pz, nx, ny, nz, sid = _guide(G, i, m, N)
    r, g, b = tl.load(A+i, m, 0), tl.load(A+N+i, m, 0), tl.load(A+2*N+i, m, 0)
    cv = tl.load(A+3*N+i, m, 0)
    lum = _luma(r, g, b)
    ar, ag, ab, av, aw = r, g, b, cv, tl.full((BZ,), 1., tl.float32)
    for dy in tl.static_range(-2, 3):
        for dx in tl.static_range(-2, 3):
            if dx != 0 or dy != 0:
                xx, yy = x+dx*STEP, y+dy*STEP
                j = yy*W+xx
                inside = m & (xx >= 0) & (xx < W) & (yy >= 0) & (yy < W)
                qx, qy, qz, qnx, qny, qnz, qsid = _guide(G, j, inside, N)
                ok = inside & (sid == qsid)
                nr, ng, nb = tl.load(A+j, ok, 0), tl.load(A+N+j, ok, 0), tl.load(A+2*N+j, ok, 0)
                nv = tl.load(A+3*N+j, ok, 0)
                normal = tl.maximum(0.0, nx*qnx + ny*qny + nz*qnz)
                normal = normal*normal
                normal = normal*normal
                normal = normal*normal
                normal = normal*normal
                normal = normal*normal
                plane = tl.abs(nx*(qx-px) + ny*(qy-py) + nz*(qz-pz))
                radiance = tl.abs(lum - _luma(nr, ng, nb)) / (SCALE*tl.sqrt(cv+nv) + 0.001)
                wx = 1.0 if dx == 0 else (2.0/3.0 if dx == -1 or dx == 1 else 1.0/6.0)
                wy = 1.0 if dy == 0 else (2.0/3.0 if dy == -1 or dy == 1 else 1.0/6.0)
                w = wx*wy*tl.where(sid < 0, 1.0, normal)*tl.exp(-plane/(0.015+0.005*STEP) - radiance)
                w = tl.where(ok, w, 0.0)
                ar += w*nr
                ag += w*ng
                ab += w*nb
                av += w*w*nv
                aw += w
    tl.store(O+i, ar/aw, m)
    tl.store(O+N+i, ag/aw, m)
    tl.store(O+2*N+i, ab/aw, m)
    tl.store(O+3*N+i, av/(aw*aw), m)


def compose(channels, spp):
    n = channels[0].numel()
    raw = torch.empty((3, n), dtype=torch.float32, device=channels[0].device)
    _compose[(triton.cdiv(n, 256),)](*channels, raw, n, spp, 256)
    return raw


def projection(guide, cam, width, basis):
    origin, forward, right, up, tan_fov = basis(cam)
    position = guide[:3]
    axes = torch.as_tensor(np.stack([right, up, forward]), device=guide.device, dtype=guide.dtype)
    p = position - torch.as_tensor(origin, device=guide.device, dtype=guide.dtype)[:, None]
    x, y, z = axes @ p
    denom = z.clamp_min(1e-6)*tan_fov
    xx = (x/denom+1)*width/2 - 0.5
    yy = (1-y/denom)*width/2 - 0.5
    return xx, yy, z, origin, tan_fov


def reproject(current, old, cam, old_cam, width, basis, settings):
    x, y, z, old_origin, tan_fov = projection(current, old_cam, width, basis)
    ix, iy = x.round().long(), y.round().long()
    inside = (ix >= 0) & (ix < width) & (iy >= 0) & (iy < width) & (z > 0)
    j = iy.clamp(0, width-1)*width + ix.clamp(0, width-1)
    old_samples = old[:, j]
    sid = current[6]
    distance = torch.linalg.vector_norm(current[:3]-old_samples[:3], dim=0)
    normal = (current[3:6]*old_samples[3:6]).sum(dim=0)
    footprint = settings["position_footprint_pixels"] * z * (2*tan_fov/width)
    origin = torch.as_tensor(cam["origin"], device=current.device, dtype=current.dtype)[:, None]
    prev_origin = torch.as_tensor(old_origin, device=current.device, dtype=current.dtype)[:, None]
    prev_dir = torch.nn.functional.normalize(prev_origin-current[:3], dim=0)
    new_dir = torch.nn.functional.normalize(origin-current[:3], dim=0)
    similar_view = (sid != 6) | ((prev_dir*new_dir).sum(dim=0) >= settings["sss_reprojection_min_view_cos"])
    valid = (inside & (sid >= 0) & (sid == old_samples[6]) &
             (normal >= settings["normal_reprojection_min_cos"]) &
             (distance <= footprint) & similar_view)
    return torch.where(valid, j, -1).to(torch.int32)


class SVGF:
    def __init__(self, width, device, basis, settings):
        self.width = width
        self.n = width*width
        self.history = torch.empty((6, self.n), device=device)
        self.scratch = torch.empty_like(self.history)
        self.a = torch.empty((4, self.n), device=device)
        self.b = torch.empty_like(self.a)
        self.placeholder = torch.empty(1, dtype=torch.int32, device=device)
        self.cam = self.guide = self.scene = None
        self.basis = basis
        self.settings = settings

    def apply(self, raw, guide, cam, scene_id, reset=False, blocked=None):
        valid = not reset and self.scene == scene_id and self.guide is not None
        moving = valid and cam != self.cam
        mapping = (reproject(guide, self.guide, cam, self.cam, self.width, self.basis, self.settings)
                   if moving else self.placeholder)
        if moving:
            if blocked is None:
                raise ValueError("Camera reprojection requires a previous-view visibility mask")
            mapping = torch.where(blocked, -1, mapping)
        _temporal[(triton.cdiv(self.n, 256),)](
            raw, self.history, mapping, self.scratch, self.n, valid, moving,
            self.settings["history_frames"], 256)
        self.history, self.scratch = self.scratch, self.history
        _variance[(triton.cdiv(self.n, 128),)](self.history, guide, self.a, self.width, 128)
        src, out = self.a, self.b
        for step in (1, 2, 4):
            _atrous[(triton.cdiv(self.n, 128),)](
                src, guide, out, self.width, step, self.settings["variance_luminance_scale"], 128)
            src, out = out, src
        self.cam, self.guide, self.scene = cam, guide, scene_id
        return src[:3], self.history[5], mapping if moving else None

    @property
    def buffers_bytes(self):
        return sum(x.numel()*x.element_size() for x in
                   (self.history, self.scratch, self.a, self.b, self.placeholder))
