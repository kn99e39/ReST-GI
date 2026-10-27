"""WL53 geometry, boundary ports, local frames, directional bins and analytic helpers (protocol phys_ref53).

numpy / torch only (no renderer). Conventions (protocol section "conventions"):
  lengths in m; radiance L in W m^-2 sr^-1; flux in W; directional-light irradiance E on a plane perpendicular to the beam.
  A domain boundary is a closed triangle mesh with OUTWARD normals (counter-clockwise faces seen from outside).
  Port = one voxel face of the rest (material) configuration; it keeps its id, its triangles and its rest tangent under any
  rigid or non-rigid map, so ports correspond by material coordinates, never by array position after re-meshing.
  Local frame per triangle: n = outward geometric normal of the CURRENT triangle; t = current image of the port's rest
  tangent (affine map of the triangle), orthonormalised; b = n x t.
  Direction bins: outward hemisphere of the local frame, Shirley-Chiu concentric map of an N x N square grid composed with
  Malley's projection, so every bin has projected solid angle exactly pi / N^2. An INCOMING direction d (travelling into
  the surface) is labelled by the outward direction -d (where the light came from).
"""
import math

import numpy as np
import torch

# ------------------------------------------------------------------ concentric map / bins


def square_to_disk(u, v):
    a = 2 * u - 1; b = 2 * v - 1
    lib = torch if isinstance(u, torch.Tensor) else np
    r1 = a; p1 = (math.pi / 4) * b / lib.where(a == 0, 1, a)
    r2 = b; p2 = (math.pi / 4) * (2 - a / lib.where(b == 0, 1, b))
    r3 = -a; p3 = (math.pi / 4) * (4 + b / lib.where(a == 0, 1, a))
    r4 = -b; p4 = (math.pi / 4) * (6 - a / lib.where(b == 0, 1, b))
    c1 = (a > -b) & (a > b); c2 = (a > -b) & ~(a > b); c3 = ~(a > -b) & (a < b)
    r = lib.where(c1, r1, lib.where(c2, r2, lib.where(c3, r3, r4)))
    p = lib.where(c1, p1, lib.where(c2, p2, lib.where(c3, p3, p4)))
    p = lib.where((a == 0) & (b == 0), 0 * p, p)
    return r * lib.cos(p), r * lib.sin(p)


def disk_to_square(x, y):
    lib = torch if isinstance(x, torch.Tensor) else np
    r = lib.sqrt(x * x + y * y); phi = lib.arctan2(y, x) if lib is np else torch.atan2(y, x)
    phi = lib.where(phi < -math.pi / 4, phi + 2 * math.pi, phi)
    q = math.pi / 4
    a1 = r; b1 = phi * a1 / q
    b2 = r; a2 = -(phi - math.pi / 2) * b2 / q
    a3 = -r; b3 = (phi - math.pi) * a3 / q
    b4 = -r; a4 = -(phi - 3 * math.pi / 2) * b4 / q
    c1 = phi < q; c2 = phi < 3 * q; c3 = phi < 5 * q
    a = lib.where(c1, a1, lib.where(c2, a2, lib.where(c3, a3, a4)))
    b = lib.where(c1, b1, lib.where(c2, b2, lib.where(c3, b3, b4)))
    return (a + 1) / 2, (b + 1) / 2


def local_to_bin(wl, N):
    """wl (..., 3) unit local direction with z >= 0 -> bin index in [0, N*N)."""
    u, v = disk_to_square(wl[..., 0], wl[..., 1])
    if isinstance(wl, torch.Tensor):
        return torch.clamp((u * N).long(), 0, N - 1) + N * torch.clamp((v * N).long(), 0, N - 1)
    return np.clip((u * N).astype(np.int64), 0, N - 1) + N * np.clip((v * N).astype(np.int64), 0, N - 1)


def bin_sample(bins, N, u1, u2):
    """Cosine-weighted direction uniformly (in projected solid angle) inside the given bins; returns local (.., 3)."""
    iu = bins % N; iv = bins // N
    x, y = square_to_disk((iu + u1) / N, (iv + u2) / N)
    lib = torch if isinstance(x, torch.Tensor) else np
    z = lib.sqrt(lib.clip(1 - x * x - y * y, 0, None)) if lib is np else torch.sqrt(torch.clamp(1 - x * x - y * y, min=0))
    return lib.stack([x, y, z], -1)


def bin_parent(N_fine, N_coarse):
    """Map fine bins to the coarse bin containing them (grids nest when N_fine is a multiple of N_coarse)."""
    assert N_fine % N_coarse == 0
    f = N_fine // N_coarse; b = np.arange(N_fine * N_fine)
    return (b % N_fine) // f + N_coarse * ((b // N_fine) // f)


# ------------------------------------------------------------------ voxel-domain boundary meshes

_DIRS = [(0, -1), (0, 1), (1, -1), (1, 1), (2, -1), (2, 1)]  # (axis, sign) of the outward face normal


def voxel_boundary(occ, h, origin=(0.0, 0.0, 0.0), sub=4):
    """Closed boundary of a union of axis-aligned voxels.

    occ bool (nx, ny, nz); h = voxel size per axis; every exposed voxel face is one port, tessellated sub x sub cells x 2
    triangles. Returns dict(V (nv,3) welded, F (nf,3) CCW outward, tri_port (nf,), port_center (np,3), port_normal (np,3),
    port_tangent (np,3), port_area (np,), port_face (np,) axis*2+(sign>0), port_voxel (np,3))."""
    occ = np.asarray(occ, bool); h = np.asarray(h, float); origin = np.asarray(origin, float)
    nx, ny, nz = occ.shape
    pad = np.zeros((nx + 2, ny + 2, nz + 2), bool); pad[1:-1, 1:-1, 1:-1] = occ
    V, F, tp, pc, pn, pt, pa, pf, pv = [], [], [], [], [], [], [], [], []
    for (i, j, k) in zip(*np.nonzero(occ)):
        for ax, sg in _DIRS:
            nb = [i + 1, j + 1, k + 1]; nb[ax] += sg
            if pad[tuple(nb)]:
                continue
            a1 = (ax + 1) % 3; a2 = (ax + 2) % 3
            base = origin + h * np.array([i, j, k], float)
            if sg > 0:
                base[ax] += h[ax]
            e1 = np.zeros(3); e1[a1] = h[a1]; e2 = np.zeros(3); e2[a2] = h[a2]
            nrm = np.zeros(3); nrm[ax] = sg
            if np.dot(np.cross(e1, e2), nrm) < 0:      # make (e1, e2, n) right-handed so faces are CCW from outside
                e1, e2 = e2, e1
            pid = len(pc)
            g = np.linspace(0, 1, sub + 1)
            grid = base[None, None, :] + g[:, None, None] * e1[None, None, :] + g[None, :, None] * e2[None, None, :]
            vi0 = len(V); V.extend(grid.reshape(-1, 3))
            idx = lambda p, q: vi0 + p * (sub + 1) + q  # noqa: E731
            for p in range(sub):
                for q in range(sub):
                    F.append((idx(p, q), idx(p + 1, q), idx(p + 1, q + 1))); F.append((idx(p, q), idx(p + 1, q + 1), idx(p, q + 1)))
                    tp.extend([pid, pid])
            pc.append(base + 0.5 * e1 + 0.5 * e2); pn.append(nrm); pt.append(e1 / np.linalg.norm(e1)); pa.append(h[a1] * h[a2])
            pf.append(2 * ax + (sg > 0)); pv.append((i, j, k))
    V = np.asarray(V); F = np.asarray(F, np.int64)
    key = np.round(V / 1e-9).astype(np.int64)
    _, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    V = V[first]; F = inv.reshape(-1)[F]
    return dict(V=V, F=F, tri_port=np.asarray(tp, np.int64), port_center=np.asarray(pc), port_normal=np.asarray(pn),
                port_tangent=np.asarray(pt), port_area=np.asarray(pa), port_face=np.asarray(pf), port_voxel=np.asarray(pv))


def open_grid(nx, ny, h, origin, sub=1):
    """Single-sided opaque surface (normal +z): an nx x ny grid of h x h ports at z = origin[2]."""
    origin = np.asarray(origin, float)
    g = np.linspace(0, 1, sub + 1)
    V, F, tp, pc = [], [], [], []
    for i in range(nx):
        for j in range(ny):
            base = origin + np.array([i * h, j * h, 0.0])
            grid = base[None, None, :] + g[:, None, None] * np.array([h, 0, 0]) + g[None, :, None] * np.array([0, h, 0])
            vi0 = len(V); V.extend(grid.reshape(-1, 3)); idx = lambda p, q: vi0 + p * (sub + 1) + q  # noqa: E731
            pid = len(pc)
            for p in range(sub):
                for q in range(sub):
                    F.append((idx(p, q), idx(p + 1, q), idx(p + 1, q + 1))); F.append((idx(p, q), idx(p + 1, q + 1), idx(p, q + 1))); tp.extend([pid, pid])
            pc.append(base + np.array([h / 2, h / 2, 0]))
    V = np.asarray(V); F = np.asarray(F, np.int64)
    key = np.round(V / 1e-9).astype(np.int64); _, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    V = V[first]; F = inv.reshape(-1)[F]
    npt = len(pc)
    return dict(V=V, F=F, tri_port=np.asarray(tp, np.int64), port_center=np.asarray(pc), port_normal=np.tile([0.0, 0, 1], (npt, 1)),
                port_tangent=np.tile([1.0, 0, 0], (npt, 1)), port_area=np.full(npt, h * h), port_face=np.full(npt, 5), port_voxel=np.zeros((npt, 3), int))


# ------------------------------------------------------------------ maps (rest -> current)


def rot(axis, deg):
    axis = np.asarray(axis, float); axis = axis / np.linalg.norm(axis); a = math.radians(deg)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + math.sin(a) * K + (1 - math.cos(a)) * K @ K


def map_rigid(X, R, t):
    return X @ np.asarray(R).T + np.asarray(t)


def map_bend(X, theta_deg, L):
    """Cylindrical bend of a slab whose mid-surface is z = 0, x in [-L/2, L/2]: the mid-surface becomes an arc of angle
    theta (length preserved), centre of curvature above (+z), so the +z face becomes concave."""
    th = math.radians(theta_deg)
    if abs(th) < 1e-12:
        return X.copy()
    R = L / th; x, y, z = X[:, 0], X[:, 1], X[:, 2]; phi = x / R
    return np.stack([(R - z) * np.sin(phi), y, R - (R - z) * np.cos(phi)], 1)


def map_dent(X, depth, radius, h, center=(0.0, 0.0)):
    """Top surface (z = +h/2) pushed down by depth * cos^2(pi r / 2 radius) inside r < radius; bottom fixed; linear in z."""
    r = np.hypot(X[:, 0] - center[0], X[:, 1] - center[1])
    b = np.where(r < radius, np.cos(0.5 * math.pi * r / radius) ** 2, 0.0)
    Y = X.copy(); Y[:, 2] = X[:, 2] - (X[:, 2] + h / 2) / h * depth * b
    return Y


# ------------------------------------------------------------------ current-configuration triangle data


def tri_frames(Vrest, Vcur, F, tri_port, port_tangent):
    """Per-triangle outward normal, area and local frame (n, t, b) in the current configuration. t = image of the port's
    rest tangent under the triangle's affine map."""
    P0, P1, P2 = (Vrest[F[:, k]] for k in range(3)); Q0, Q1, Q2 = (Vcur[F[:, k]] for k in range(3))
    e1, e2 = P1 - P0, P2 - P0; f1, f2 = Q1 - Q0, Q2 - Q0
    tr = port_tangent[tri_port]
    G = np.stack([np.einsum("ij,ij->i", e1, e1), np.einsum("ij,ij->i", e1, e2), np.einsum("ij,ij->i", e2, e2)], 1)
    r1 = np.einsum("ij,ij->i", tr, e1); r2 = np.einsum("ij,ij->i", tr, e2)
    det = G[:, 0] * G[:, 2] - G[:, 1] ** 2
    al = (G[:, 2] * r1 - G[:, 1] * r2) / det; be = (G[:, 0] * r2 - G[:, 1] * r1) / det
    tc = al[:, None] * f1 + be[:, None] * f2
    c = np.cross(f1, f2); area = 0.5 * np.linalg.norm(c, axis=1); n = c / (2 * area[:, None])
    t = tc - np.einsum("ij,ij->i", tc, n)[:, None] * n; t /= np.linalg.norm(t, axis=1, keepdims=True)
    b = np.cross(n, t)
    cen = (Q0 + Q1 + Q2) / 3
    return dict(n=n, t=t, b=b, area=area, centroid=cen)


def port_areas(tri_area, tri_port, n_ports):
    return np.bincount(tri_port, weights=tri_area, minlength=n_ports)


def to_local(d, n, t, b):
    lib = torch if isinstance(d, torch.Tensor) else np
    return lib.stack([(d * t).sum(-1), (d * b).sum(-1), (d * n).sum(-1)], -1)


def to_world(wl, n, t, b):
    return wl[..., 0:1] * t + wl[..., 1:2] * b + wl[..., 2:3] * n


def sample_on_tris(V, F, tri_idx, u1, u2):
    """Uniform point on the given triangles (torch)."""
    a = V[F[tri_idx, 0]]; bb = V[F[tri_idx, 1]]; c = V[F[tri_idx, 2]]
    su = torch.sqrt(u1)
    return (1 - su)[:, None] * a + (su * (1 - u2))[:, None] * bb + (su * u2)[:, None] * c


# ------------------------------------------------------------------ analytic: Chandrasekhar H-function (isotropic)


def H_function(omega, mu, n=400, iters=5000, tol=1e-13):
    """Isotropic-scattering H-function, Chandrasekhar's convergent form
    1/H(mu) = sqrt(1 - omega) + (omega/2) int_0^1 mu' H(mu') / (mu + mu') dmu'. Returns H at the given mu (array)."""
    x, w = np.polynomial.legendre.leggauss(n); x = (x + 1) / 2; w = w / 2
    H = np.ones(n); s = math.sqrt(max(0.0, 1 - omega))
    for _ in range(iters):
        Hn = 1.0 / (s + 0.5 * omega * np.sum(w * x * H / (x[:, None] + x[None, :]), axis=1))
        if np.max(np.abs(Hn - H)) < tol:
            H = Hn; break
        H = 0.5 * H + 0.5 * Hn
    mu = np.atleast_1d(np.asarray(mu, float))
    return 1.0 / (s + 0.5 * omega * np.sum(w * x * H / (mu[:, None] + x[None, :]), axis=1))


def semi_infinite_radiance(omega, mu, mu0, E_perp=1.0):
    """Exact reflected radiance of a semi-infinite, index-matched, isotropically scattering medium under a collimated
    beam of perpendicular irradiance E_perp at cos-angle mu0, viewed at cos-angle mu (Chandrasekhar 1960, Ch. IV):
    L = omega / (4 pi) * E_perp * mu0 / (mu + mu0) * H(mu) H(mu0)."""
    H = H_function(omega, [mu, mu0])
    return omega / (4 * math.pi) * E_perp * mu0 / (mu + mu0) * H[0] * H[1]


def plane_albedo(omega, mu0):
    return 1 - H_function(omega, [mu0])[0] * math.sqrt(1 - omega)
