"""WL52 explicit transport on the GPU (Mitsuba 3.5.1 cuda_ad_rgb + torch): the efficient explicit-C baseline C0, the
oracle, the learned model's patch-level supervision, the Lambertian K, the coupled solve and an independent path-traced GI
reference (continuous geometry, no port quantization).

Conventions (identical for every arm): receiver = port; its K cosine-distributed gathers start EPS above its 4 sample points
(K / 4 each, per-port random rotation from `seed`); a hit maps to the port of the hit triangle; a miss or a back-face hit
(buried origin) is NULL and carries no radiance. E_ind(r) = pi / K sum_k L(hit_k). Lambertian K: L_out = rho / pi E_tot.
"""
import math
import time

import numpy as np
import torch

import lc_scene as LS

_MI = None


def mi_init():
    global _MI
    if _MI is None:
        import drjit as dr
        import mitsuba as mi
        mi.set_variant("cuda_ad_rgb"); _MI = (mi, dr)
    return _MI


def to_mi(x, kind="f"):
    mi, _ = mi_init(); x = x.contiguous()
    return mi.Float(x.float()) if kind == "f" else mi.UInt32(x.to(torch.int32))


class Geo:
    """Mitsuba scene (one mesh) + device tables of a Scene."""
    def __init__(self, S, dev):
        mi, dr = mi_init(); t0 = time.time(); self.S, self.dev = S, dev
        me = mi.Mesh("m", len(S.V), len(S.F)); p = mi.traverse(me)
        p["vertex_positions"] = mi.Float(S.V.ravel()); p["faces"] = mi.UInt32(S.F.ravel().astype(np.uint32)); p.update()
        self.scene = mi.load_dict({"type": "scene", "m": me}); self.t_build = time.time() - t0
        T = lambda a, dt=torch.float32: torch.as_tensor(np.asarray(a), device=dev).to(dt)  # noqa: E731
        self.tri2port = T(S.tri2port, torch.long); V = T(S.V); F = T(S.F, torch.long)
        n = torch.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]], dim=1); self.tnrm = n / n.norm(dim=1, keepdim=True).clamp(min=1e-20)
        self.pos, self.nrm, self.area, self.spos, self.snrm = T(S.pos), T(S.nrm), T(S.area), T(S.spos), T(S.snrm)
        self.patch = T(S.patch, torch.long); self.alb = T(S.albedo); self.obj = T(S.obj, torch.long)
        self.pcen, self.pnrm, self.parea, self.pobj = T(S.pcen), T(S.pnrm), T(S.parea), T(S.pobj, torch.long)

    def trace(self, o, d):
        """o, d [n, 3] torch -> (port id or -1 NULL [n], t [n], hit point [n, 3], hit normal [n, 3])."""
        mi, dr = mi_init()
        ray = mi.Ray3f(mi.Point3f(*(to_mi(o[:, k]) for k in range(3))), mi.Vector3f(*(to_mi(d[:, k]) for k in range(3))))
        si = self.scene.ray_intersect(ray)
        valid = si.is_valid().torch().bool(); prim = si.prim_index.torch().long(); t = si.t.torch()
        prim = torch.where(valid, prim, torch.zeros_like(prim)); tn = self.tnrm[prim]
        ok = valid & ((tn * d).sum(1) < 0)                                                     # front face only
        port = torch.where(ok, self.tri2port[prim], torch.full_like(prim, -1))
        return port, torch.where(ok, t, torch.full_like(t, float("inf"))), o + torch.where(ok, t, torch.zeros_like(t))[:, None] * d, tn

    def occluded(self, o, d):
        mi, dr = mi_init()
        ray = mi.Ray3f(mi.Point3f(*(to_mi(o[:, k]) for k in range(3))), mi.Vector3f(*(to_mi(d[:, k]) for k in range(3))))
        return self.scene.ray_test(ray).torch().bool()


def frame(n):
    a = torch.where(n[:, 2:3].abs() < 0.9, n.new_tensor([0.0, 0, 1]).expand_as(n), n.new_tensor([1.0, 0, 0]).expand_as(n))
    t = torch.nn.functional.normalize(torch.cross(a, n, dim=1), dim=1); return t, torch.cross(n, t, dim=1)


def cos_dirs(K, dev):
    """Stratified (Fibonacci) cosine-distributed local directions [K, 3]."""
    i = torch.arange(K, device=dev, dtype=torch.float32) + 0.5; r = torch.sqrt(i / K); ph = i * math.pi * (3 - math.sqrt(5))
    return torch.stack([r * torch.cos(ph), r * torch.sin(ph), torch.sqrt((1 - r * r).clamp(min=0))], 1)


def gather_rays(G, ports, K, seed):
    """Origins / directions [len(ports) * K, 3] (K / 4 per sample point, per-port random rotation)."""
    dev = G.dev; g = torch.Generator(device=dev).manual_seed(int(seed)); n = len(ports); per = K // 4
    loc = cos_dirs(per, dev); rot = torch.rand(n, 4, generator=g, device=dev) * 2 * math.pi
    sn = G.snrm[ports]; t, b = frame(sn.reshape(-1, 3)); t, b = t.view(n, 4, 3), b.view(n, 4, 3); c, s = torch.cos(rot)[..., None], torch.sin(rot)[..., None]
    tt, bb = (c * t + s * b)[:, :, None], (-s * t + c * b)[:, :, None]                                        # [n, 4, 1, 3]
    d = loc[None, None, :, 0:1] * tt + loc[None, None, :, 1:2] * bb + loc[None, None, :, 2:3] * sn[:, :, None]
    o = (G.spos[ports] + LS.EPS * sn)[:, :, None].expand(-1, -1, per, -1)
    return o.reshape(-1, 3), d.reshape(-1, 3)


@torch.no_grad()
def gather(G, K, seed, ports=None, chunk=1 << 22):
    """Explicit C relation table: hit port [n, K] (-1 NULL) and distance."""
    ports = torch.arange(G.S.N, device=G.dev) if ports is None else ports
    H = torch.empty(len(ports), K, dtype=torch.long, device=G.dev); Tt = torch.empty(len(ports), K, device=G.dev); step = max(1, chunk // K)
    for s in range(0, len(ports), step):
        pp = ports[s:s + step]; o, d = gather_rays(G, pp, K, seed + s); h, t, _, _ = G.trace(o, d)
        H[s:s + len(pp)] = h.view(len(pp), K); Tt[s:s + len(pp)] = t.view(len(pp), K)
    return H, Tt


def apply_table(H, L):
    """E_ind [n, 3] = pi / K sum_k L[hit_k] (NULL -> 0)."""
    K = H.shape[1]; Lp = torch.cat([L, torch.zeros(1, L.shape[1], device=L.device, dtype=L.dtype)])
    return Lp[torch.where(H >= 0, H, torch.full_like(H, len(L)))].sum(1) * (math.pi / K)


@torch.no_grad()
def patch_target(G, H):
    """Per-receiver hit distribution over patches + NULL (column M) as counts [n, M + 1] (dense float)."""
    M = G.S.M; idx = torch.where(H >= 0, G.patch[H.clamp(min=0)], torch.full_like(H, M))
    out = torch.zeros(H.shape[0], M + 1, device=G.dev); out.scatter_add_(1, idx, torch.ones_like(idx, dtype=torch.float32)); return out


def patch_mean(G, L):
    """Area-mean radiance per patch [M, 3]."""
    s = torch.zeros(G.S.M, L.shape[1], device=G.dev).index_add_(0, G.patch, L * G.area[:, None])
    return s / torch.zeros(G.S.M, device=G.dev).index_add_(0, G.patch, G.area)[:, None].clamp(min=1e-20)


@torch.no_grad()
def sun_irr(G, w, E=3.0):
    """Port direct irradiance from a sun (direction w towards the sun): E cos * unoccluded fraction of the 4 samples."""
    w = torch.as_tensor(w, device=G.dev, dtype=torch.float32); w = w / w.norm(); cs = (G.snrm @ w).clamp(min=0)            # [N, 4]
    o = (G.spos + LS.EPS * G.snrm).reshape(-1, 3); occ = G.occluded(o, w.expand_as(o)).view(-1, 4)
    return E * (cs * (~occ).float()).mean(1)


@torch.no_grad()
def solve(G, Esun, Capply, tol=1e-5, max_it=200):
    """Lambertian K + linear C: L = rho / pi (E_sun + C L); fixed-point (Jacobi) iteration, no damping."""
    Ed = Esun[:, None] * torch.ones(1, 3, device=G.dev); L = G.alb / math.pi * Ed; hist = []; status = "unconverged"
    for it in range(max_it):
        Ln = G.alb / math.pi * (Ed + Capply(L)); d = float((Ln - L).norm() / Ln.norm().clamp(min=1e-30)); hist.append(d); L = Ln
        if not torch.isfinite(L).all():
            status = "nonfinite"; break
        if it >= 8 and len(hist) > 5 and np.median(np.array(hist[-5:]) / np.maximum(np.array(hist[-6:-1]), 1e-30)) >= 1.0:
            status = "diverged"; break
        if d < tol:
            status = "converged"; break
    Eind = Capply(L)
    return {"L": L, "Eind": Eind, "status": status, "iters": len(hist), "hist": hist}


@torch.no_grad()
def spectral_radius(G, Capply, iters=60, seed=0):
    g = torch.Generator(device=G.dev).manual_seed(seed); L = torch.rand(G.S.N, 3, generator=g, device=G.dev); rates = []
    for _ in range(iters):
        n0 = L.norm(dim=0); L = G.alb / math.pi * Capply(L); n1 = L.norm(dim=0); rates.append((n1 / n0.clamp(min=1e-30)).tolist()); L = L / n1.clamp(min=1e-30)
    return np.median(np.array(rates[-10:]), 0).tolist()


@torch.no_grad()
def path_traced_Eind(G, w, paths, seed, bounces=16, E=3.0, chunk_ports=None):
    """Independent GI reference at the ports: E_ind(r) ~ sum_b prod_{j<=b} rho_j E_sun(y_b) along cosine-sampled paths in the
    continuous geometry (exact hit points, exact sun visibility with a shadow ray per vertex). Ports as receivers only."""
    dev = G.dev; w = torch.as_tensor(w, device=dev, dtype=torch.float32); w = w / w.norm(); N = G.S.N
    chunk_ports = chunk_ports or max(1, (1 << 22) // paths); out = torch.zeros(N, 3, device=dev); gen = torch.Generator(device=dev).manual_seed(seed)
    for s0 in range(0, N, chunk_ports):
        pp = torch.arange(s0, min(N, s0 + chunk_ports), device=dev); n = len(pp)
        smp = torch.randint(0, 4, (n, paths), generator=gen, device=dev)
        ar = torch.arange(n, device=dev)[:, None]; nrm = G.snrm[pp][ar, smp].reshape(-1, 3); o = G.spos[pp][ar, smp].reshape(-1, 3) + LS.EPS * nrm
        thr = torch.ones(n * paths, 3, device=dev); acc = torch.zeros(n * paths, 3, device=dev); alive = torch.ones(n * paths, dtype=torch.bool, device=dev)
        for b in range(bounces):
            u1, u2 = torch.rand(len(o), generator=gen, device=dev), torch.rand(len(o), generator=gen, device=dev)
            r, ph = torch.sqrt(u1), 2 * math.pi * u2; t, bb = frame(nrm)
            d = (r * torch.cos(ph))[:, None] * t + (r * torch.sin(ph))[:, None] * bb + torch.sqrt((1 - u1).clamp(min=0))[:, None] * nrm
            port, tt, p, tn = G.trace(o, d); alive = alive & (port >= 0)
            if not alive.any():
                break
            rho = G.alb[port.clamp(min=0)]; thr = thr * rho; hn = -tn * torch.sign((tn * d).sum(1, keepdim=True))       # hit normal facing the incoming ray
            cs = (hn @ w).clamp(min=0); po = p + LS.EPS * hn; vis = ~G.occluded(po, w.expand_as(po))
            acc = acc + (alive & vis)[:, None] * thr * (E * cs)[:, None]
            o, nrm = po, hn
        out[pp] = acc.view(n, paths, 3).mean(1)
    return out


@torch.no_grad()
def camera_ports(G, eye, target, fov_deg=55.0, res=256):
    """Pinhole camera -> port id per pixel (-1 background), for rendering port fields as images."""
    dev = G.dev; eye = torch.as_tensor(eye, device=dev, dtype=torch.float32); f = torch.nn.functional.normalize(torch.as_tensor(target, device=dev, dtype=torch.float32) - eye, dim=0)
    r = torch.nn.functional.normalize(torch.cross(f, torch.tensor([0.0, 0, 1], device=dev), dim=0), dim=0); u = torch.cross(r, f, dim=0); s = math.tan(math.radians(fov_deg) / 2)
    a = (torch.arange(res, device=dev) + 0.5) / res * 2 - 1; Y, X = torch.meshgrid(-a, a, indexing="ij")
    d = torch.nn.functional.normalize(f + s * X[..., None] * r + s * Y[..., None] * u, dim=-1).reshape(-1, 3)
    port, _, _, _ = G.trace(eye.expand_as(d).contiguous(), d); return port.view(res, res)


CAMS = [((0.9, 0.9, 0.85), (-0.3, -0.3, 0.15)), ((0.95, -0.2, 0.6), (-0.4, -0.1, 0.1)), ((-0.2, 0.95, 0.7), (-0.2, -0.4, 0.1))]
