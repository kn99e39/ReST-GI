"""WL54 Stage B (protocol source_receiver_separation): known-field transfer tests that isolate WHERE the coarse interface loses information.

    python c54_known_field.py <out json> <fixture,...> [s=8]
Setup: fine cells (s x s per port, exact Lambert K) with their C (unbiased, closure 1). A prescribed source field (flux per cell phi_j) is transferred once:
  exact       Phi_i = sum_j C_ij phi_j                          (fine source distribution)
  src-avg     Phi'_i = sum_j C_ij phibar_j,  phibar_j = A_j / A_J * sum_{j' in J} phi_j'    (uniform radiance per 1 cm port: the coarse interface's assumption)
and the coarse receiver flux is Phi_I = sum_{i in I} Phi_i (receiver averaging is a conservative SUM: lossless for flux by construction; checked below).
Fields: 'solution' = the converged fine outgoing flux of the scene; synthetic radiance fields on the sender group: uniform, linear ramp, exponential
edge decay (e-fold 5 mm from the sunward edge), step inside a port. The effect is split by source-receiver centroid distance (< h, h-2h, 2h-4h, > 4h) and
by bounce (Neumann terms): bounce 0 is the first-order (sun) emission, where sub-port visibility variation (lit strip) enters.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "phys_ref53"))
import c54_common as CC  # noqa: E402
import c54_fixtures as CF  # noqa: E402
import c54_lam as LM  # noqa: E402

DEV = "cuda"; H = 0.01
SENDER = {"gap4": ("slab_lo:top", "slab_hi:bot"), "gap8": ("slab_lo:top", "slab_hi:bot"), "gap30": ("slab_lo:top", "slab_hi:bot"), "floor_lam": ("floor", "dom:bot"),
          "lblock_lam": ("floor", "dom:bot"), "rot_lam": ("floor", "dom:bot")}


def cell_centroids(ss):
    V = ss.V; F = ss.F; tri_c = (V[F[:, 0]] + V[F[:, 1]] + V[F[:, 2]]) / 3
    num = torch.zeros(ss.Ptot, 3, device=DEV).index_add_(0, ss.tri_port, tri_c * ss.area[:, None]); den = torch.zeros(ss.Ptot, device=DEV).index_add_(0, ss.tri_port, ss.area)
    return num / den[:, None]


def run(name, s, kv):
    labels = CF.coarse_labels(name); f = LM.Lam(name, s).build(M=256 if s == 8 else 128, seed_C=5900, K_ext=128, seed_E=5901); r = f.solve()
    C = f.C; idx = C._indices(); val = C._values().double(); par = torch.tensor(f.parent, device=DEV); a = torch.tensor(f.area, device=DEV, dtype=torch.float64)
    Ac = torch.zeros(f.nc, device=DEV, dtype=torch.float64).index_add_(0, par, a); cen = cell_centroids(f.ss).double()
    d = (cen[idx[0]] - cen[idx[1]]).norm(dim=1); edges = torch.tensor([H, 2 * H, 4 * H], device=DEV, dtype=torch.float64); bins = torch.bucketize(d, edges)     # 0: <h, 1: h-2h, 2: 2h-4h, 3: >4h
    send_lab, recv_lab = SENDER[name]; lab = np.array(labels)
    send_cell = torch.tensor(lab[f.parent] == send_lab, device=DEV); recv_cell = torch.tensor(lab[f.parent] == recv_lab, device=DEV)
    x = cen[:, 0]; xmin = float(x[send_cell].min()); out = dict(fixture=name, s=s, sender=send_lab, receiver=recv_lab, fields={})

    def transfer(phi):
        """exact and source-averaged receiver flux per fine cell, per distance bin, aggregated to the receiver group."""
        mean_flux = torch.zeros(f.nc, device=DEV, dtype=torch.float64).index_add_(0, par, phi); phibar = mean_flux[par] * a / Ac[par]
        res = {}
        for tag, ph in (("exact", phi), ("avg", phibar)):
            contrib = val * ph[idx[1]]
            res[tag] = [float(contrib[(bins == b) & recv_cell[idx[0]]].sum()) for b in range(4)]
        res["flux_conservation"] = float(abs(phibar.sum() - phi.sum()) / phi.sum())
        return res

    def pack(res):
        ex, av = np.array(res["exact"]), np.array(res["avg"]); tot = ex.sum()
        return dict(total_exact=float(tot), total_avg=float(av.sum()), rel_effect=float((av.sum() - tot) / tot), by_distance_rel_to_total={k: float((av[b] - ex[b]) / tot) for b, k in enumerate(("lt_h", "h_2h", "2h_4h", "gt_4h"))},
                    by_distance_share_of_exact={k: float(ex[b] / tot) for b, k in enumerate(("lt_h", "h_2h", "2h_4h", "gt_4h"))}, flux_conservation=res["flux_conservation"])

    phi_sol = torch.tensor(r["phi"], device=DEV, dtype=torch.float64); mask = send_cell.double()
    out["fields"]["solution"] = pack(transfer(phi_sol * mask))
    L = {"uniform": torch.ones_like(x), "ramp": 1 + (x - xmin) / 0.1, "exp_edge_5mm": torch.exp(-(x - xmin) / 0.005), "step_in_port": (x > xmin + 0.015).double()}
    for k, Lx in L.items():
        out["fields"][k] = pack(transfer(a * Lx * mask))
    # receiver-side losslessness: coarse receiver flux = sum over cells (identity), checked on the solution field
    Phi_cell = torch.zeros(f.ss.Ptot, device=DEV, dtype=torch.float64).index_add_(0, idx[0], val * phi_sol[idx[1]]); Phi_port = torch.zeros(f.nc, device=DEV, dtype=torch.float64).index_add_(0, par, Phi_cell)
    out["receiver_sum_identity_err"] = float(abs(Phi_port.sum() - Phi_cell.sum()) / Phi_cell.sum())
    # bounce-resolved: fine vs Galerkin-coarse Neumann series, group-pooled radiance on the receiver group (coarse ports)
    Cg, Acg, epc = f.galerkin(); Cgd = Cg.double(); ep_f = torch.tensor(f.Ep, device=DEV, dtype=torch.float64); rho = LM.RHO
    phi0 = rho * ep_f * a; phif = phi0.clone(); psi0 = torch.zeros(f.nc, device=DEV, dtype=torch.float64).index_add_(0, par, phi0); psi = psi0.clone()
    Acg_t = torch.tensor(Acg, device=DEV, dtype=torch.float64); epc_t = torch.tensor(epc, device=DEV, dtype=torch.float64)
    rg = torch.tensor(lab == recv_lab, device=DEV); Ef = torch.zeros(f.ss.Ptot, device=DEV, dtype=torch.float64); Ec = torch.zeros(f.nc, device=DEV, dtype=torch.float64); rows = []
    for k in range(0, 9):
        Lf = rho / math.pi * (ep_f + Ef / a); Lfc = torch.zeros(f.nc, device=DEV, dtype=torch.float64).index_add_(0, par, a * Lf) / Ac
        Lc = rho / math.pi * (epc_t + Ec / Acg_t); sel = rg
        rows.append(dict(bounces=k, fine_group_L=float(Lfc[sel].sum()), coarse_group_L=float(Lc[sel].sum()), rel_diff=float((Lc[sel].sum() - Lfc[sel].sum()) / Lfc[sel].sum())))
        inc_f = torch.sparse.mm(C.double(), phif[:, None]).squeeze(1); Ef = Ef + inc_f; phif = rho * inc_f
        inc_c = torch.sparse.mm(Cgd, psi[:, None]).squeeze(1); Ec = Ec + inc_c; psi = rho * inc_c
    out["bounce_resolved"] = rows
    return out


def main():
    out = Path(sys.argv[1]); fixtures = sys.argv[2].split(","); kv = dict(a.split("=") for a in sys.argv[3:]); s = int(kv.get("s", 8)); res = {}
    for n in fixtures:
        res[n] = run(n, s, kv); r = res[n]
        print(n, "solution field: rel effect of source averaging on receiver flux", round(r["fields"]["solution"]["rel_effect"], 4), "by distance", {k: round(v, 4) for k, v in r["fields"]["solution"]["by_distance_rel_to_total"].items()},
              "| bounce-resolved coarse-vs-fine", [round(x["rel_diff"], 3) for x in r["bounce_resolved"][:5]], flush=True)
    CC.dump(res, out)


if __name__ == "__main__":
    main()
