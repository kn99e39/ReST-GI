"""WL50 composable K: matched Yellow in-support nonlocal arms (L0 = K44 representative aggregation, L1 = DASQ alpha 1.0)
inside the unchanged r1 / K_k44P composition (protocol fnto_comp50).

Decomposition (Yellow receiver ports = domain A ports with material 0; every other port is K44 production):
  out_Y(x, w) = K44 K_loc(x, w)                 [unchanged: rho / pi E_tot + kappa lobes on E_ind + the sun's GGX lobe]
              + K44 K_glob(x)                    [unchanged rank-64 global, p(x) . gamma]
              + N_arm(x)                         [REPLACES K44 band + mid for Yellow receivers; isotropic; linear in E_tot]
N_arm(x)_c = sum_e w_c(desc_e) * measure_e * area-mean E_tot,c over element e   (worklog-49 network, matched across arms)
Trained on the Cycles probe residual r = col - K_loc - K_glob at Yellow pixels (c50_targets), so K_loc / K_glob and the
arm cannot double count. No oracle or reference value is used at inference.
"""
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
for p in ("fnto_spatial49", "fnto_dir46", "fnto_lego_k44", "fnto_lego_r1", "fnto_lego"):
    sys.path.insert(0, str(HERE.parent / p))
import r1_common as C  # noqa: E402
import r1_solve as SV  # noqa: E402
import s49_data as S49  # noqa: E402
import dasq as DQ  # noqa: E402
import g3_learned as G3  # noqa: E402

PROD_STATE = SV.State
PROD_RENDER = SV.render_pixels
ARM = None                                            # the active ArmOp (None = production)
TIMING = {"arm_apply_s": 0.0, "arm_apply_calls": 0}


class Geo:
    """Shared Lego geometry for the arms (worklog-49 tables + DASQ octree)."""
    def __init__(self, dev):
        self.L = S49.Lego(dev); self.T = DQ.Octree(self.L.pos, self.L.area); DQ.set_normals(self.T, self.L.nrm); self.dev = dev


class ArmOp:
    def __init__(self, kind, net, geo, ports, chunk=4000):
        L, T, dev = geo.L, geo.T, geo.dev; self.kind, self.geo = kind, geo; t0 = time.time()
        self.ports = torch.as_tensor(np.asarray(ports), device=dev, dtype=torch.long); n = len(self.ports)
        Ws, Ss, Es, NB, MD = [], [], [], [], []
        with torch.no_grad():
            for i in range(0, n, chunk):
                xi = self.ports[i:i + chunk]; x = L.pos[xi]; nx = L.nrm[xi]
                if kind == "L1":
                    s_, e_, _ = T.elements(x, xi, 1.0, S49.R_IN); cen, nrm, meas = DQ.elem_geom(T, s_, e_); valid = (s_ >= 0).float()
                    Ss.append(s_.int()); Es.append(e_.int())
                else:
                    nbr, nbrA, mid, midA = L.rep_lists(xi)
                    cen = torch.cat([L.bpos[nbr.clamp(min=0)], L.cpos[mid.clamp(min=0)]], 1); nrm = torch.cat([L.bnrm[nbr.clamp(min=0)], L.cnrm[mid.clamp(min=0)]], 1)
                    meas = torch.cat([nbrA, midA], 1).clamp(min=0); valid = torch.cat([nbr >= 0, mid >= 0], 1).float(); NB.append(nbr); MD.append(mid)
                D = G3.descriptors(x, nx, cen, nrm, meas, valid)
                Ws.append(net.weights(D, meas, valid).float())
        K = max(w.shape[1] for w in Ws)
        pad = lambda a, v: torch.cat([a, torch.full((a.shape[0], K - a.shape[1]) + a.shape[2:], v, dtype=a.dtype, device=a.device)], 1)  # noqa: E731
        self.W = torch.cat([pad(w, 0.0) for w in Ws])                                            # [n, K, 3]
        if kind == "L1":
            self.S = torch.cat([pad(s, -1) for s in Ss]); self.E = torch.cat([pad(e, -1) for e in Es])
        else:
            self.nbr = torch.cat(NB); self.mid = torch.cat(MD)
        self.K = K; self.n_elements = int((self.W.abs().sum(-1) > 0).sum()); self.t_build_s = time.time() - t0

    @torch.no_grad()
    def apply(self, Etot, chunk=50000):
        """Etot [nA, 3] (domain-A ports) -> N_arm [n, 3] for self.ports."""
        t0 = time.time(); L, T = self.geo.L, self.geo.T; out = torch.empty(len(self.ports), 3, device=Etot.device)
        if self.kind == "L1":
            AE = (L.area[:, None].double() * Etot.double())[T.order]
            cs = torch.cat([torch.zeros(1, 3, device=Etot.device, dtype=torch.float64), torch.cumsum(AE, 0)])
            for i in range(0, len(self.ports), chunk):
                s = self.S[i:i + chunk].long(); e = self.E[i:i + chunk].long(); ok = s >= 0; s, e = s.clamp(min=0), e.clamp(min=0)
                ar = (T.csA[e] - T.csA[s]).clamp(min=1e-30)
                mean = ((cs[e] - cs[s]) / ar[..., None] * ok[..., None]).float()
                out[i:i + chunk] = (self.W[i:i + chunk] * mean).sum(1)
        else:
            Bm, Cm = L.agg_means(Etot)                                                               # [NB, 3], [NC, 3]
            for i in range(0, len(self.ports), chunk):
                nb, md = self.nbr[i:i + chunk], self.mid[i:i + chunk]
                f = torch.cat([Bm[nb.clamp(min=0)] * (nb >= 0)[..., None], Cm[md.clamp(min=0)] * (md >= 0)[..., None]], 1)
                out[i:i + chunk] = (self.W[i:i + chunk] * f).sum(1)
        torch.cuda.synchronize(); TIMING["arm_apply_s"] += time.time() - t0; TIMING["arm_apply_calls"] += 1
        return out


class ArmState(PROD_STATE):
    """r1_solve.State with the Yellow receivers' band + mid replaced by the active arm (ARM is None -> production)."""
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        T = self.T; self.keep = torch.ones(T.N, device=T.dev)
        if ARM is not None:
            self.keep[ARM.ports] = 0.0
        self.Narm = torch.zeros(T.N, 3, device=T.dev)

    def outgoing(self, Eind, h, sun_on=True):                                                    # = r1_solve.State.outgoing + keep mask + arm
        T = self.T; Etot = Eind + (self.Esun[:, None] if sun_on else 0)
        I = T.rho / math.pi * Etot + T.kap.sum(1) / math.pi * Eind
        gam = []
        for o in (0, 1):
            ii = T.dom[o]; g = (T.area[ii][:, None, None] * T.q[o] * Etot[ii][:, None, :]).sum(0); gam.append(g)
            I[ii] += (T.p[o] * g[None]).sum(1)
        Ec = torch.zeros(T.Nc, 3, device=T.dev).index_add_(0, T.f2c, Etot * T.area[:, None]) / T.carea[:, None].clamp(min=1e-12)
        hB = torch.zeros(T.Nb, C.N_BINS_H, 3, device=T.dev).index_add_(0, T.f2b, h * T.area[:, None, None]) / T.barea[:, None, None].clamp(min=1e-12)
        for s in range(0, T.N, 1 << 15):
            sl = slice(s, s + (1 << 15)); I[sl] += self.keep[sl, None] * (T.Wm[sl].float() * Ec[T.mid[sl].clamp(min=0)]).sum(1)
        I += self.keep[:, None] * T.band_sender(hB)
        if ARM is not None:
            nA = int(T.dom[0].numel()); n = ARM.apply(Etot[:nA]); I[ARM.ports] += n; self.Narm.zero_(); self.Narm[ARM.ports] = n
        self.gamma, self.Ec, self.hB = gam, Ec, hB
        return I


@torch.no_grad()
def arm_render_pixels(T, S, px, chunk=1 << 13):
    out, parts = PROD_RENDER(T, S, px, chunk)
    if ARM is not None:
        if getattr(S, "keep", None) is None:
            return out, parts
        yel = (px["dom"] == 0) & (S.keep[px["port"]] == 0)                                       # exactly the arm-owned receiver ports
        if yel.any():
            parts[yel, 3] = S.Narm[px["port"][yel]]; out[yel] = parts[yel].sum(1)
    return out, parts


def activate(arm):
    global ARM
    ARM = arm; SV.State = ArmState; SV.render_pixels = arm_render_pixels


def deactivate():
    global ARM
    ARM = None; SV.State = PROD_STATE; SV.render_pixels = PROD_RENDER


def yellow_ports():
    A = np.load(C.OUT1 / "anchors.npz"); nA = int(A["n_A"])
    return np.flatnonzero(A["mat"][:nA] == 0)


def load_net(path, dev):
    net = G3.Net().to(dev); net.load_state_dict(torch.load(path, map_location=dev)); net.eval(); return net
