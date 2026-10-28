"""WL54 Lambert transport machinery (protocol fnto_c54): the WL53 production C caster and sun term on arbitrary sub-port
refinements, fine solves, area-weighted projection onto the 1 cm ports, Galerkin aggregation of a fine C onto the coarse
unknowns, and a small Lambert radiosity solver. Everything reuses p53_compose.cast_C / sun_external / SceneSet unchanged.

Definitions (Lambert K, Dout = Din = 1 unless stated; flux units, W):
  C[i, j]      fraction of the flux leaving patch j (uniform radiance over the patch, Lambert emission) whose air ray first
               enters patch i. Row sums + escape = 1 per column (closure).
  phi_out      = phi0 + rho C phi_out                     phi0 = rho * Ep * A       (Ep: unoccluded sun irradiance per patch)
  L            = rho / pi * (Ep + (C phi_out) / A)          radiance of the patch (view independent for Lambert)
  projection   L_coarse[I] = sum_{i in I} A_i L_i / sum_{i in I} A_i          (conservative: coarse flux = sum of fine flux)
  Galerkin     C_g[I, J] = sum_{i in I, j in J} C[i, j] A_j / A_J  (exact coarse operator for a piecewise-constant coarse field)
"""
import math
import time

import numpy as np
import torch

import c54_common as CC
import c54_fixtures as CF
import p53_compose as CP
import p53_fixtures as FX

DEV = "cuda"
RHO = FX.FLOOR_RHO
PROD = dict(M=128, K_ext=256, seed_C=5340, seed_E=5341)        # WL53 production configuration (protocol conventions, CFG)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


class Lam:
    """One scene at refinement s. coarse_mesh=True with bins=(64, 16) is literally the WL53 production C0 configuration."""

    def __init__(self, name, s=1, bins=(1, 1), coarse_mesh=False):
        self.name, self.s, self.bins = name, s, bins
        comps = CF.comps_at(name, s, coarse=coarse_mesh)
        self.ss = CP.SceneSet([CP.Comp(n, "lambert", m, V, rho=RHO) for n, m, V in comps], bins[0], bins[1])
        self.parent = np.arange(self.ss.Ptot) if (coarse_mesh or s == 1) else CF.parents(name, s)
        self.nc = int(self.parent.max()) + 1
        self.area = self.ss.port_area.double().cpu().numpy()
        self.sun = np.asarray(CF.light_of(name), float)
        self.C = self.esc = self.stats = self.Ep = None

    # -- building blocks
    def build(self, M, seed_C, K_ext, seed_E):
        t0 = time.time(); torch.cuda.synchronize()
        self.C, self.esc, self.stats = CP.cast_C(self.ss, M, seed_C)
        torch.cuda.synchronize(); self.t_C = time.time() - t0; t0 = time.time()
        Ep, _ = CP.sun_external(self.ss, self.sun, FX.E_SUN, K_ext, seed_E)
        torch.cuda.synchronize(); self.t_ext = time.time() - t0
        self.Ep = Ep.double().cpu().numpy(); self.M, self.K_ext = M, K_ext
        return self

    def port_C(self):
        """C reduced to patch level (sum over directional bins): (P, P) sparse; for Dout = Din = 1 this is C itself."""
        if self.bins == (1, 1):
            return self.C
        D, Dn = self.bins; idx = self.C._indices(); v = self.C._values()
        rows = idx[0] // Dn; cols = idx[1] // D
        return torch.sparse_coo_tensor(torch.stack([rows, cols]), v / D, (self.ss.Ptot, self.ss.Ptot)).coalesce()   # columns average over the D out-bins

    def solve(self, C=None, Ep=None, area=None, tol=1e-7, max_iter=3000):
        """Lambert radiosity on patch-level C (P x P); returns dict(L, phi, iters, rho)."""
        C = self.port_C() if C is None else C
        Ep = self.Ep if Ep is None else Ep; area = self.area if area is None else area
        a = torch.tensor(area, device=DEV, dtype=torch.float64); ep = torch.tensor(Ep, device=DEV, dtype=torch.float64)
        Cd = C.double(); phi0 = RHO * ep * a; phi = phi0.clone()
        for it in range(1, max_iter + 1):
            new = phi0 + RHO * torch.sparse.mm(Cd, phi[:, None]).squeeze(1)
            rel = float((new - phi).abs().sum() / new.abs().sum()); phi = new
            if rel < tol:
                break
        phin = torch.sparse.mm(Cd, phi[:, None]).squeeze(1)
        L = RHO / math.pi * (ep + phin / a)
        v = torch.rand(len(phi), device=DEV, dtype=torch.float64, generator=torch.Generator(device=DEV).manual_seed(0)); r = 0.0
        for _ in range(200):
            w = RHO * torch.sparse.mm(Cd, v[:, None]).squeeze(1); r = float(w.norm() / v.norm()); v = w / w.norm().clamp(min=1e-300)
        return dict(L=L.cpu().numpy(), phi=phi.cpu().numpy(), iters=it, converged=rel < tol, rel=rel, rho=r)

    def project(self, L):
        return np.bincount(self.parent, weights=self.area * L, minlength=self.nc) / np.bincount(self.parent, weights=self.area, minlength=self.nc)

    def galerkin(self):
        """Coarse operator from this (fine) C: returns (C_g sparse (nc, nc), A_c, Ep_c)."""
        C = self.port_C(); idx = C._indices(); v = C._values().double()
        par = torch.tensor(self.parent, device=DEV); a = torch.tensor(self.area, device=DEV, dtype=torch.float64)
        Ac = torch.zeros(self.nc, device=DEV, dtype=torch.float64).index_add_(0, par, a)
        rows, cols = par[idx[0]], par[idx[1]]
        vals = v * a[idx[1]] / Ac[cols]
        Cg = torch.sparse_coo_tensor(torch.stack([rows, cols]), vals, (self.nc, self.nc)).coalesce()
        epc = np.bincount(self.parent, weights=self.area * self.Ep, minlength=self.nc) / Ac.cpu().numpy()
        return Cg, Ac.cpu().numpy(), epc

    def closure(self):
        C = self.port_C(); col = torch.zeros(self.ss.Ptot, device=DEV, dtype=torch.float64).index_add_(0, C._indices()[1], C._values().double())
        esc = self.esc.double().view(self.ss.Ptot, -1).mean(1) if self.bins != (1, 1) else self.esc.double()
        tot = col + esc
        return dict(min=float(tot.min()), max=float(tot.max()), mean=float(tot.mean()), back_hits=int(self.stats["back_hits"]), rays=int(self.stats["rays"]))


def solve_coarse_from_fine(fine):
    """Galerkin-exact coarse solve from a fine C (conservative aggregation, uniform radiance within each coarse port)."""
    Cg, Ac, epc = fine.galerkin()
    helper = fine.solve(C=Cg, Ep=epc, area=Ac)
    return helper


def pixel_prediction(lam, L, geo):
    """Predicted radiance per pixel (centre ray) of a Blender ortho camera from patch radiances L (-1 -> nan)."""
    pid = CP.classify(lam.ss, CC.blender_ortho_rays(geo, ((0.5, 0.5),))).cpu().numpy()
    out = np.where(pid >= 0, L[np.clip(pid, 0, None)], np.nan)
    return out.reshape(geo["res"], geo["res"]), pid.reshape(geo["res"], geo["res"])
