"""WL46 directional K/C composition on the frozen r1 / k44 machinery (r1_solve unchanged; this subclasses its State).

Switches (protocol fnto_dir46, conditions R0 / R3 / R4 / R5 / R3x):
  nl      sender nonlocal part (glob + band + mid) read by a gather ray (p, k) hitting port q:
            'n'      production: evaluated at w_o = n_q (r1_solve.outgoing: Wm table + band_sender)          [R0, R4]
            'basis'  per-port 19-bin piecewise-constant basis, value at the bin of -d_pk in q's frame            [R3, R5]
            'exact'  exact query toward -d_pk per ray                                                           [R3x]
  loc_in  sender K_loc incident representation: 'E' production collapse (rho / pi E_tot + sum kappa / pi E_ind) or
          'dir' the learned BSDF on the 128 directional gathered samples of q (indirect part)
  loc_out direction of the 'dir' indirect K_loc evaluation: 'n' (port normal, R4) or 'ray' (-d_pk, R5)
In every mode the sun's diffuse term is rho / pi E_sun (isotropic) and the sun's GGX lobe is the production per-ray term
State.spec toward -d_pk (counted once; tests/test_dir46.py checks no double count).
A query direction with n_q . w <= 0 (port-normal / hit-side mismatch) is projected to the upper hemisphere
(w - (n.w - 1e-3) n, renormalised); the fraction of such rays is recorded (n_below).
nl = 'n', loc_in = 'E' reproduces r1_solve.State.step exactly up to fp16 storage (test).
"""

from __future__ import annotations

import math

import numpy as np
import torch

import dir46_core as D
import r1_common as C
import r1_solve as SV
import fnto_model as M


def upper(n, w):
    c = (n * w).sum(-1, keepdim=True)
    return torch.where(c > 1e-3, w, torch.nn.functional.normalize(w - (c - 1e-3) * n, dim=-1))


class DirState(SV.State):
    def __init__(self, T, H, Vfrac, w_sun, nl="n", loc_in="E", loc_out="ray", sun_strength=C.E_SUN, chunk=1 << 15):
        super().__init__(T, H, Vfrac, w_sun, sun_strength)
        self.nl, self.loc_in, self.loc_out = nl, loc_in, loc_out
        self.n_below = 0; self.rbin = None
        dev = T.dev
        if nl == "basis":                                                       # bin of -d in the hit port's frame
            self.rbin = torch.zeros(T.N, C.K_GATHER, dtype=torch.uint8, device=dev); nb = 0
            with torch.no_grad():
                for s in range(0, T.N, chunk):
                    sl = slice(s, s + chunk); q = self.Hc[sl]; w = -T.dirs[sl].float(); nq = T.nrm[q]
                    nb += int((((nq * w).sum(-1) <= 1e-3) & self.valid[sl]).sum())
                    w = upper(nq, w); td, bd = T.detf[0][q], T.detf[1][q]
                    loc = torch.stack([(w * td).sum(-1), (w * bd).sum(-1), (w * nq).sum(-1)], -1)
                    self.rbin[sl] = M.hemi_bin(loc).to(torch.uint8)
            self.n_below = nb
        self.cen = D.bin_centres_local()
        self.timing = {"outgoing": 0.0, "basis": 0.0, "rays": 0.0}

    @torch.no_grad()
    def basis_table(self):
        """O[q, b, 3]: nonlocal outgoing radiance of every port at its 19 bin centres (needs gamma / Ec / hB)."""
        T = self.T; O = torch.empty(T.N, len(self.cen), 3, device=T.dev)
        for s in range(0, T.N, 1 << 15):
            q = torch.arange(s, min(s + (1 << 15), T.N), device=T.dev); W = D.to_world(T.nrm[q], self.cen)
            for j in range(W.shape[1]):
                g, b, m = D.nonlocal_out(T, self, q, W[:, j].contiguous())
                O[q, j] = g + b + m
        return O

    @torch.no_grad()
    def prepare(self, sun_on=True):
        """Aggregates + production outgoing (sets gamma / Ec / hB) and the per-port tables the ray values need."""
        T = self.T; t0 = _t()
        Eind, h = self.aggregates()
        if not sun_on:
            h[torch.arange(T.N, device=T.dev), self.sunbin] -= self.Esun[:, None]
        I = self.outgoing(Eind, h, sun_on)
        Etot = Eind + (self.Esun[:, None] if sun_on else 0)
        self.loc_iso = T.rho / math.pi * Etot + T.kap.sum(1) / math.pi * Eind
        self.NL_n = I - self.loc_iso
        self.sund = T.rho / math.pi * (self.Esun[:, None] if sun_on else torch.zeros_like(Eind))
        self.timing["outgoing"] += _t() - t0; t0 = _t()
        self.O = self.basis_table() if self.nl == "basis" else None
        self.loc_n = None
        if self.loc_in == "dir" and self.loc_out == "n":                        # R4: one value per port (outgoing at the normal)
            q = torch.arange(T.N, device=T.dev); self.loc_n = self._loc_dir(q, T.nrm) + self.sund
        self.timing["basis"] += _t() - t0

    @torch.no_grad()
    def ray_values(self, q, d, bins=None):
        """Sender radiance of K ports q [n] read along gather directions d [n, 3] (d points from the receiver to q), WITHOUT the
        sun's GGX lobe (production: State.spec, added by the caller). bins: precomputed basis bins (else computed)."""
        T = self.T
        if self.nl == "n":
            nlv = self.NL_n[q]
        elif self.nl == "basis":
            if bins is None:
                nq = T.nrm[q]; w = upper(nq, -d); td, bd = T.detf[0][q], T.detf[1][q]
                bins = M.hemi_bin(torch.stack([(w * td).sum(-1), (w * bd).sum(-1), (w * nq).sum(-1)], -1))
            nlv = self.O[q, bins.long()]
        else:
            g, b, m = D.nonlocal_out(T, self, q, upper(T.nrm[q], -d)); nlv = g + b + m
        if self.loc_in == "E":
            lv = self.loc_iso[q]
        elif self.loc_n is not None:
            lv = self.loc_n[q]
        else:
            w = T.nrm[q] if self.loc_out == "n" else upper(T.nrm[q], -d)
            lv = self._loc_dir(q, w) + self.sund[q]
        return nlv + lv

    @torch.no_grad()
    def sun_lobe(self, q, w_out):
        """The sun's GGX lobe of K ports q toward w_out (= State.spec semantics for arbitrary rays)."""
        T = self.T; nq = T.nrm[q]; ws = self.w.expand(len(q), 3)
        g_ = torch.stack([M.ggx(nq, ws, w_out, T.alpha[q][:, l:l + 1])[:, 0] for l in range(2)], -1)
        return (g_[..., None] * T.kap[q]).sum(1) * self.Esun[q][:, None]

    @torch.no_grad()
    def k_rays(self, sun_on=True):
        """New ray values of every K-port gather from the current tables (after prepare). Returns (Ln, dn2, nn2)."""
        T = self.T; t0 = _t()
        Ln = torch.empty_like(self.L); dn2 = nn2 = 0.0
        ch = 1 << 13 if (self.nl == "exact" or (self.loc_in == "dir" and self.loc_out == "ray")) else 1 << 16
        for s in range(0, T.N, ch):
            sl = slice(s, s + ch); q = self.Hc[sl]; v = self.valid[sl]
            d = T.dirs[sl].float()
            val = self.ray_values(q.reshape(-1), d.reshape(-1, 3), None if self.rbin is None else self.rbin[sl].reshape(-1)).view(*q.shape, 3)
            val = val * v[..., None] + (self.spec[sl] if sun_on else 0)
            Ln[sl] = val.half()
            a = Ln[sl].float(); dn2 += float((a - self.L[sl].float()).square().sum()); nn2 += float(a.square().sum())
        self.timing["rays"] += _t() - t0
        return Ln, dn2, nn2

    @torch.no_grad()
    def step(self, sun_on=True):
        self.prepare(sun_on); Ln, dn2, nn2 = self.k_rays(sun_on); self.L = Ln
        return math.sqrt(dn2), math.sqrt(nn2)

    @torch.no_grad()
    def _loc_dir(self, q, w, chunk=1 << 13):
        """Indirect K_loc of ports q toward w from their directional gathered samples (diffuse + both GGX lobes)."""
        T = self.T; out = torch.empty(len(q), 3, device=T.dev)
        for s in range(0, len(q), chunk):
            qq = q[s:s + chunk]
            out[s:s + chunk] = M.DomainKernel.loc_response(T.nrm[qq], w[s:s + chunk], T.rho[qq], T.kap[qq], T.alpha[qq], T.dirs[qq].float(),
                                                           self.L[qq].float() * (math.pi / C.K_GATHER))
        return out


def _t():
    torch.cuda.synchronize()
    import time
    return time.time()


@torch.no_grad()
def spectral_radius(T, H, iters=30, seed=0, **mode):
    """Power iteration of the homogeneous composed map (no sun) for a DirState mode."""
    S = DirState(T, H, np.zeros(T.N), np.array([0.0, 0.0, 1.0]), **mode)
    g = torch.Generator(device=T.dev); g.manual_seed(seed)
    S.L = torch.rand(S.L.shape, generator=g, device=T.dev, dtype=torch.float16); S.L.mul_(S.valid[..., None])
    rates = []
    for _ in range(iters):
        n0 = float(S.L.float().norm()); S.step(sun_on=False); n1 = float(S.L.float().norm())
        if not np.isfinite(n1) or n1 == 0:
            rates.append(float("inf") if not np.isfinite(n1) else 0.0); break
        rates.append(n1 / n0); S.L.mul_(1e3 / n1)
    return {"rho_estimate": float(np.median(rates[-8:])) if rates else None, "rates": rates}
