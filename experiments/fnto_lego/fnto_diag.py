"""FNTO Batch 6 post-result attribution diagnostics (DESCRIPTIVE; they do not change the fnto_v1 verdict).

    python fnto_diag.py <kernels.pt> <out dir>

D1 per-component passivity: effective albedo pi I / E of every port under a uniform unit incident field, separately for
   K_loc (diffuse + glossy, albedo 1 lobes), K_glob, K_band, K_mid.
D2 growth on the real coupling graph: Jacobi iterations on the T0 relation table for one held-out T0 view's sun, with
   component subsets switched on; per-iteration growth ratio of the total incident energy (-> spectral radius of K C).
D3 identifiability: per-port number of Stage A training images that lit the port directly (E > 0), against the
   per-port albedo of D1 (are amplifying ports the rarely / never excited ones?), and against 'enclosed' ports (> 50 %
   of their gathers hit the inside of a closed solid).
D4 single-bounce composition (NOT the declared method; a truncation used only to look at the C channel): incident =
   sun + ONE gather of K[sun]; energy of the predicted remainder vs the reference remainder on held-out T0.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

import fnto_common as C
import fnto_eval as EV
import fnto_solve as SV

COMPS = ("loc", "glob", "band", "mid")


@torch.no_grad()
def outgoing(T, S, Eind, h, comps):
    Etot = Eind + S.Esun[:, None]
    I = torch.zeros(T.N, 3, device=T.dev)
    if "loc" in comps:
        I += T.rho / math.pi * Etot + T.kap.sum(1) / math.pi * Eind
    gam = []
    for o in (0, 1):
        ii = T.dom[o]
        g = (T.area[ii][:, None, None] * T.q[o] * Etot[ii][:, None, :]).sum(0); gam.append(g)
        if "glob" in comps:
            I[ii] += (T.p[o] * g[None]).sum(1)
    if "mid" in comps:
        Ec = torch.zeros(T.Nc, 3, device=T.dev).index_add_(0, T.f2c, Etot * T.area[:, None]) / T.carea[:, None]
        for s in range(0, T.N, 1 << 15):
            sl = slice(s, s + (1 << 15))
            I[sl] += (T.Wm[sl].float() * Ec[T.mid[sl].clamp(min=0)]).sum(1)
    if "band" in comps:
        I += T.band_sender(h)
    return I


@torch.no_grad()
def albedo_components(T):
    H = np.full((T.N, C.K_GATHER), -1, np.int64)
    S = SV.State(T, H, np.zeros(T.N, bool), np.array([0.0, 0.0, 1.0]))
    S.L = torch.ones(T.N, C.K_GATHER, 3, device=T.dev, dtype=torch.float16)
    Eind, h = S.incident_aggregates()
    return {c: (math.pi * outgoing(T, S, Eind, h, (c,)) / Eind.clamp(min=1e-9)).cpu().numpy() for c in COMPS}


@torch.no_grad()
def growth(T, H, V, w, comps, iters=12):
    S = SV.State(T, H, V, w)
    out = []
    prev = None
    for it in range(iters):
        Eind, h = S.incident_aggregates()
        I = outgoing(T, S, Eind, h, comps).half()
        Ln = torch.empty_like(S.L)
        for s in range(0, T.N, 1 << 16):
            sl = slice(s, s + (1 << 16))
            Ln[sl] = I[S.Hc[sl]] * S.valid[sl][..., None] + (S.spec[sl] if "loc" in comps else 0)
        S.L = Ln
        e = float(Ln.float().sum())
        if not np.isfinite(e):
            out.append(None); break
        out.append(e / prev if prev else None); prev = e if e > 0 else None
    return out


def main(kp, outd):
    outd = Path(outd); outd.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda")
    KA, KB = EV.load_kernels(kp, dev); T = SV.Tables(KA, KB, dev)
    A = np.load(C.OUT / "anchors.npz")
    rec = {}
    alb = albedo_components(T)
    tot = sum(alb.values())
    rec["D1_albedo"] = {c: {"q50_q90_q99": np.quantile(a, [0.5, 0.9, 0.99]).tolist(), "max": float(a.max()), "frac_gt_1": float((a > 1).mean())} for c, a in alb.items()}
    rec["D1_albedo"]["sum"] = {"q50_q90_q99": np.quantile(tot, [0.5, 0.9, 0.99]).tolist(), "frac_gt_1": float((tot > 1).mean())}
    rec["D1_albedo_by_material_frac_gt_1"] = {C.MATERIALS[int(m)]: {c: float((alb[c][A["mat"] == m] > 1).mean()) for c in COMPS} for m in np.unique(A["mat"])}
    # D2: growth on the T0 relation graph, sun of held-out T0 view 6
    D = np.load(C.OUT / "eval_px/heldout_T0.npz"); w = D["v6_light"]
    Vp = np.unpackbits(np.load(C.OUT / "eval_vis.npz")["heldout_T0_v6_T0"])[:T.N].astype(bool)
    H0 = np.load(C.OUT / "rel_T0.npz")["hit"]
    rec["D2_growth_T0_view6"] = {"+".join(cs): growth(T, H0, Vp, w, cs) for cs in
                                 (("loc",), ("loc", "band"), ("loc", "mid"), ("loc", "glob"), ("loc", "band", "mid"), COMPS)}
    # D3: Stage A excitation per port vs albedo
    pe = np.load(C.OUT / "stagea/prep_train/portE.npy", mmap_mode="r")
    cnt = np.zeros(pe.shape[1], np.int64)
    for s in range(0, pe.shape[0], 50):
        cnt += (np.asarray(pe[s:s + 50]) > 0).sum(0)
    enclosed = (H0 == -2).mean(1) > 0.5
    bins = [(0, 0), (1, 10), (11, 100), (101, 10 ** 9)]
    rec["D3_excitation"] = {"ports_never_lit": int((cnt == 0).sum()), "ports_enclosed": int(enclosed.sum()),
                            "albedo_sum_frac_gt_1_by_lit_images": {f"{a}-{b}": {"ports": int(((cnt >= a) & (cnt <= b)).sum()),
                                                                                  "frac_gt_1": float((tot[(cnt >= a) & (cnt <= b)] > 1).mean()) if ((cnt >= a) & (cnt <= b)).any() else None,
                                                                                  "albedo_q50": float(np.median(tot[(cnt >= a) & (cnt <= b)])) if ((cnt >= a) & (cnt <= b)).any() else None}
                                                                    for a, b in bins},
                            "albedo_sum_frac_gt_1_enclosed": float((tot[enclosed] > 1).mean()), "albedo_sum_frac_gt_1_not_enclosed": float((tot[~enclosed] > 1).mean())}
    np.save(outd / "port_excitation_count.npy", cnt.astype(np.int32)); np.save(outd / "port_albedo_components.npy", np.stack([alb[c] for c in COMPS], 1).astype(np.float16))
    # D4: single-bounce composition on held-out T0 (truncation, descriptive)
    first = dict(np.load(C.OUT / "firstevent/heldout_T0_first.npz"))
    pv = np.load(C.OUT / "eval_vis.npz"); rows = []
    for v in range(int(D["nv"])):
        light = D[f"v{v}_light"]; Vp = np.unpackbits(pv[f"heldout_T0_v{v}_T0"])[:T.N].astype(bool)
        S = SV.State(T, H0, Vp, light)
        S.gather_once()                                          # L = one free-flight gather of K[sun] (first bounce)
        Eind, h = S.incident_aggregates(); S.h, S.Eind = h, Eind; _, S.gamma = S.outgoing_iso(Eind, h)
        px = {k: torch.as_tensor(D[f"v{v}_{k}"].astype(np.float32 if k in ("pos", "nrm", "cam", "vis", "w4") else np.int64), device=dev)
              for k in ("pos", "nrm", "cam", "vis", "dom", "port", "port4", "w4")}
        p1 = SV.render_pixels(T, S, px)[0].cpu().numpy()
        S0 = SV.State(T, H0, Vp, light); E0, h0 = S0.incident_aggregates(); S0.h, S0.Eind = h0, E0; _, S0.gamma = S0.outgoing_iso(E0, h0)
        p0 = SV.render_pixels(T, S0, px)[0].cpu().numpy()
        tgt = D[f"v{v}_tgt"]; fr = first[f"v{v}_rgb"]
        rows.append({"v": v, "ref_first": float(fr.sum()), "ref_remainder": float((tgt - fr).sum()), "pred_first": float(p0.sum()),
                     "pred_one_bounce_increment": float((p1 - p0).sum()),
                     "display_psnr_one_bounce_vs_full_ref": float(10 * np.log10(1 / ((C.ldr3(p1) - C.ldr3(tgt)) ** 2).mean())),
                     "display_psnr_first_vs_first_ref": float(10 * np.log10(1 / ((C.ldr3(p0) - C.ldr3(fr)) ** 2).mean()))})
        del S, S0
    rec["D4_single_bounce_T0"] = {"views": rows, "sum_ref_first": sum(r["ref_first"] for r in rows), "sum_ref_remainder": sum(r["ref_remainder"] for r in rows),
                                  "sum_pred_first": sum(r["pred_first"] for r in rows), "sum_pred_one_bounce_increment": sum(r["pred_one_bounce_increment"] for r in rows),
                                  "mean_display_psnr_one_bounce_vs_full_ref": float(np.mean([r["display_psnr_one_bounce_vs_full_ref"] for r in rows])),
                                  "mean_display_psnr_first_vs_first_ref": float(np.mean([r["display_psnr_first_vs_first_ref"] for r in rows]))}
    (outd / "diag.json").write_text(json.dumps(rec, indent=1, default=float)); print(json.dumps(rec, indent=1, default=float)[:6000])


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
