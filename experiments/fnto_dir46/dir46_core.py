"""WL46 directional K/C boundary interface - core queries on the frozen r1 / k44 composition (r1_solve Tables / State).

Boundary transport quantities (units, conventions):
  L_in(p, k)   radiance [W m^-2 sr^-1] arriving at port p along gather k, i.e. travelling in direction -d_pk where
               d_pk = Tables.dirs[p, k] (unit, world frame, pointing AWAY from p's surface into its upper hemisphere);
               this is State.L. Gathers are cosine-distributed: E(p) = pi / K sum_k L_in(p, k).
  L_out(q, w)  radiance leaving port q in direction w (unit, world, n_q . w > 0). A gather ray (p, k) that hits q reads
               L_in(p, k) = L_out(q, -d_pk).
  Channels     linear RGB, the r1 references' channels.
Operator ownership is unchanged: L_out(q, .) is the frozen K of q's domain applied to q's incident field; C is the
relation table (which port each gather hits). Everything below is linear in the incident field for a fixed K and
geometry (no radiance enters a nonlinearity).

query_out(T, S, q, w_nl, w_loc, incoming):
  nonlocal part  glob (isotropic by K's design) + band + mid, evaluated with outgoing direction w_nl
                 (the production sender uses w_nl = n_q, see r1_solve.Tables.Wm / band_sender);
  local part     K_loc at q:
     incoming = "E"    production semantics: rho/pi (E_sun + E_ind) + sum_l kappa_l / pi E_ind (indirect glossy collapsed to
                        an isotropic value) + the sun's GGX lobe toward w_loc (r1_solve.State.spec, direction-resolved);
     incoming = "dir"  the full learned BSDF on the directional incident field: f(w_i, w_loc) over the sun (delta) and the
                        K gathered samples of q (L_in(q, k), weight pi / K), r1 fnto_model.loc_response.
  The production per-ray value is query_out(q, w_nl = n_q, w_loc = -d, incoming = "E") (identity test in tests/).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "fnto_lego_r1")); sys.path.insert(0, str(HERE.parent / "fnto_lego"))
import r1_common as C  # noqa: E402
import fnto_model as M  # noqa: E402

LUM = np.array([0.2126, 0.7152, 0.0722])


def dom_of(T, q):
    return (q >= T.nA).long()


@torch.no_grad()
def nonlocal_out(T, S, q, w_nl, chunk=1 << 13):
    """glob + band + mid outgoing radiance of ports q [n] in directions w_nl [n, 3]; needs S.gamma / S.Ec / S.hB (set by
    State.outgoing). Returns (glob, band, mid) each [n, 3]."""
    n = len(q); g = torch.zeros(n, 3, device=T.dev); b = torch.zeros_like(g); m = torch.zeros_like(g)
    d = dom_of(T, q)
    for o in (0, 1):
        idx = torch.nonzero(d == o)[:, 0]
        if not len(idx):
            continue
        Kk = T.Ks[o]; qo = q[idx]; loc_i = qo - (0 if o == 0 else T.nA)
        g[idx] = (T.p[o][loc_i] * S.gamma[o][None]).sum(1)
        for s in range(0, len(idx), chunk):
            ii = idx[s:s + chunk]; qq = q[ii]; w = w_nl[ii]
            nb = T.nbr[qq]; nbc = nb.clamp(min=0)
            W = Kk.band_w(T.hp[qq], T.hs[o][nbc], T.nrm[qq], T.bnrm[nbc], (T.bpos[nbc] - T.pos[qq][:, None]) / C.R_LOCAL, w)
            b[ii] = torch.einsum("pkbc,pkbc->pc", W * (T.nbrA[qq] * (nb >= 0))[..., None, None], S.hB[nbc])
            mc = T.mid[qq]; mcc = mc.clamp(min=0)
            Wm = Kk.mid_w(T.hp[qq], T.hc[o][mcc], T.nrm[qq], T.cnrm[mcc], (T.cpos[mcc] - T.pos[qq][:, None]) / C.R_MID, w)
            m[ii] = (Wm * (T.midA[qq] * (mc >= 0))[..., None] * S.Ec[mcc]).sum(1)
    return g, b, m


@torch.no_grad()
def local_out(T, S, q, w_loc, incoming="E", chunk=1 << 12):
    """K_loc outgoing radiance of ports q toward w_loc [n, 3]. incoming 'E' = production collapse, 'dir' = full BSDF."""
    n = len(q); out = torch.zeros(n, 3, device=T.dev)
    for s in range(0, n, chunk):
        qq = q[s:s + chunk]; w = w_loc[s:s + chunk]; nq = T.nrm[qq]
        rho, kap, alpha = T.rho[qq], T.kap[qq], T.alpha[qq]
        Esun = S.Esun[qq]                                                          # [m] cosine-weighted sun irradiance
        ws = S.w.expand(len(qq), 3)
        g_s = torch.stack([M.ggx(nq, ws, w, alpha[:, l:l + 1])[:, 0] for l in range(2)], -1)          # [m, 2]
        up_o = ((nq * w).sum(-1) > 0).float()[:, None]
        sun_spec = (g_s[..., None] * kap).sum(1) * Esun[:, None] * up_o
        if incoming == "E":
            Eind = S.L[qq].float().sum(1) * (math.pi / C.K_GATHER)
            out[s:s + chunk] = (rho / math.pi * (Eind + Esun[:, None]) + kap.sum(1) / math.pi * Eind) * up_o + sun_spec
        else:
            Li = S.L[qq].float() * (math.pi / C.K_GATHER)
            ind = M.DomainKernel.loc_response(nq, w, rho, kap, alpha, T.dirs[qq].float(), Li)
            out[s:s + chunk] = ind + rho / math.pi * Esun[:, None] * up_o + sun_spec
    return out


def query_out(T, S, q, w_nl, w_loc, incoming="E"):
    g, b, m = nonlocal_out(T, S, q, w_nl)
    return local_out(T, S, q, w_loc, incoming) + g + b + m


def cosine_dirs(n, k, seed=0):
    """k cosine-distributed unit directions around each normal n [m, 3] (torch), stratified, deterministic."""
    loc = torch.as_tensor(C.cosine_dirs_local(k), dtype=torch.float32, device=n.device)          # [k, 3]
    t, b = M.det_frame(n)
    return loc[None, :, 0:1] * t[:, None] + loc[None, :, 1:2] * b[:, None] + loc[None, :, 2:3] * n[:, None]


def bin_centres_local():
    """Centres of the 19 hemispherical bins of fnto_model.hemi_bin (local frame, z = normal)."""
    out = [(0.0, 0.0, 1.0)]
    for ring, th in ((6, math.radians(35)), (12, math.radians(70))):
        for j in range(ring):
            az = (j + 0.5) * 2 * math.pi / ring
            out.append((math.sin(th) * math.cos(az), math.sin(th) * math.sin(az), math.cos(th)))
    return np.asarray(out, np.float32)


def to_world(n, loc):
    """Local directions loc [B, 3] in each port's deterministic frame -> world [m, B, 3]."""
    t, b = M.det_frame(n)
    loc = torch.as_tensor(loc, device=n.device)
    return loc[None, :, 0:1] * t[:, None] + loc[None, :, 1:2] * b[:, None] + loc[None, :, 2:3] * n[:, None]
