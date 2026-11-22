"""WL104K-B Stage A: functional attribution and fitting audit of the four archived WL104K hard_1 P fits (12k states; the WL104K-A
extended states as a secondary copy). Diagnostic only; no state is changed or adopted.

A1 peak / shoulder / tail profiles at matched physical directions (fresh independent walks on a dense incidence x ring grid,
   plus the WL104K-A hard_1 scene lanes with their archived physical responses).
A2 component decomposition: gated component contributions q_j(mu_i) q_j(mu_o) C_j and ungated C_j on the same grid.
A3 decoded parameters (roughness, anisotropy, F0, amplitudes, attenuation knots, eta, gates) per start.
A4 local sensitivities of the mirror-peak height to parameter groups; Gauss-Newton response metric of the training data
   (surrogate of the L1 objective) and its spectrum; the peak change attainable per unit surrogate-objective increase.
A5 good-vs-bad parameter differences in the response-metric eigenbasis (compensation / correlated variation).
A6 counterfactuals: linear path between bad and good starts (actual training objective, peak, held-out el1 along the path),
   group swaps, integrated-gradient attribution of the peak-height difference (path dependent; non-identifiable by construction).
A7 numerical audit of all 9 x 4 terminal P states for the O1 choice: projected-gradient norms, gate-bound activity, Adam
   final step vs curvature, response-metric conditioning per material.

Usage: python stageA104kb.py <out_dir> [--walks N]
"""
import argparse
import time
from pathlib import Path

import numpy as np
import torch

import common104kb as KB

CM, IO, GF, CL, FT = KB.CM, KB.IO, KB.GF, KB.CL, KB.FT
LUMA = KB.LUMA
M_HARD = 2                       # test83_hard_1 in the 9-material order
GOOD, BAD = (2, 3), (0, 1)       # WL104K-A descriptive: starts 3-4 fit the peak, starts 1-2 overshoot (0-based r)
SEEDS = {"walk": 1041301, "pair0": 2 ** 31}          # new disjoint stream (WL104K-A used 1041210 at pair0 2^29)
INC_POLAR = np.arange(14.0, 42.01, 2.0)              # scene highlight 20-36 deg plus margin; two azimuths as the reference
INC_AZ = np.array([22.5, 112.5])
RADII = [0.25, 0.5, 0.75, 1, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10, 12, 16, 20, 30]
REGIONS = {"core_0_3": (0, 3), "shoulder_3_12": (3, 12), "tail_12_30": (12, 30.01)}


def group_matrix():
    """Columns: y groups (gf104i.raw_groups, orthonormal) then one group per component gate (15 knots each). Returns
    names, list of index arrays / basis blocks over the 178-vector x = [y (58), g (8*3*5)]."""
    out = []
    for k, B in KB.Y_GROUPS.items():
        blk = np.zeros((178, B.shape[1])); blk[:58] = B
        out.append((k, blk))
    for j, c in enumerate(CM.COMP):
        blk = np.zeros((178, 15)); blk[58 + 15 * j: 58 + 15 * (j + 1)] = np.eye(15)
        out.append((f"gate_{c}", blk))
    return out


def x_of(st, m):
    return np.concatenate([st["y"][m].reshape(-1), st["g"][m].reshape(-1)])


def split(x):
    return x[:58], x[58:].reshape(8, 3, 5)


def lum_t(F):
    return F @ KB.to_t(LUMA)


def jac_fwd(f, x0):
    """Forward-mode Jacobian [N_out, 178], one jvp per input coordinate (memory safe for large outputs)."""
    cols = []
    for i in range(x0.numel()):
        v = torch.zeros_like(x0); v[i] = 1.0
        cols.append(torch.func.jvp(f, (x0,), (v,))[1].detach())
    return torch.stack(cols, 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("out", type=Path)
    ap.add_argument("--walks", type=int, default=65536)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    t0 = time.monotonic()
    rows, raw, compiled, desc, ref, seen = CM.inputs()
    seen.update(KB.KA.authenticate_k())
    seen.update(KB.authenticate_ka())
    n = len(rows)
    assert rows[M_HARD]["tag"] == "test83_hard_1"
    stp, R, _ = KB.KA.states(KB.KA.K_ROOT / "fit_primary", n)
    ste, _, _ = KB.KA.states(KB.REPO / "results/ecro_wl84_104ka/primary/ext", n)
    fit = CM.load_set("fit_set", n)
    ev, peak = CM.eval_reference(ref, n)
    res = {"note": "Stage A diagnostic of archived states; no state changed or adopted; start r is 0-based (start r+1 in figures).",
           "good_starts": GOOD, "bad_starts": BAD, "seeds": SEEDS}

    # ---------------- A1/A2 dense matched grid with fresh physical walks (hard_1 only)
    inc = KB.sph(np.repeat(INC_POLAR, len(INC_AZ)), np.tile(INC_AZ, len(INC_POLAR)))
    wo, rad = KB.peak_grid(inc, RADII)
    wi = np.broadcast_to(inc[:, None], wo.shape).copy()
    ok = wo[..., 2] > 1e-6
    tw = time.monotonic()
    act = ok.ravel()
    a, b = KB.DA.ref_eval(desc, np.full(int(act.sum()), M_HARD), wi.reshape(-1, 3)[act], wo.reshape(-1, 3)[act], args.walks,
                          SEEDS["walk"], pair0=SEEDS["pair0"])
    A_ = np.zeros((act.size, 3)); B_ = np.zeros((act.size, 3)); A_[act], B_[act] = a, b
    ref_a, ref_b = (A_ @ LUMA).reshape(ok.shape), (B_ @ LUMA).reshape(ok.shape)
    refm = 0.5 * (ref_a + ref_b)
    res["grid"] = {"incidence_polar_deg": INC_POLAR.tolist(), "incidence_azimuth_deg": INC_AZ.tolist(), "radii_deg": [0] + RADII,
                   "walks": args.walks, "walk_seconds": time.monotonic() - tw, "pairs": int(act.sum())}
    rings = np.unique(np.round(rad, 4))

    def ring_mean(v):                                         # [P, K] -> [P, n_rings] azimuth mean over valid points
        out = np.full((v.shape[0], len(rings)), np.nan)
        for k, r_ in enumerate(rings):
            sel = np.isclose(rad, r_)
            vv = np.where(ok[:, sel], v[:, sel], np.nan)
            out[:, k] = np.nanmean(vv, 1)
        return out

    prof_ref = ring_mean(refm)
    prof_half = ring_mean(np.abs(ref_a - ref_b) / 2)
    res["grid"]["reference_profile"] = prof_ref.tolist()
    res["grid"]["reference_half_noise_relative_at_mirror"] = (prof_half[:, 0] / prof_ref[:, 0]).tolist()
    per_start = {}
    sh = (1, len(inc), wo.shape[1], 3)
    for tag, st in (("12k", stp), ("extended", ste)):
        rows_ = []
        for r in range(R):
            s1 = {k: v[M_HARD:M_HARD + 1] for k, v in st["P"][r].items()}
            Fp = CM.predict("P", s1, wi[None], wo[None])[0] @ LUMA
            comp = CM.comp_responses("P", s1, wi[None], wo[None])[0] @ LUMA                          # [P, K, 8] gated
            ung = CM.comp_responses("P", {**s1, "g": np.ones_like(s1["g"])}, wi[None], wo[None])[0] @ LUMA
            sa = {k: v[M_HARD:M_HARD + 1] for k, v in st["A"][r].items()}
            Fa = CM.predict("A", sa, wi[None], wo[None])[0] @ LUMA
            prof = ring_mean(Fp)
            rr = {"start": r, "peak_height_rel": (prof[:, 0] / prof_ref[:, 0] - 1).tolist(),
                  "A_peak_height_rel": (ring_mean(Fa)[:, 0] / prof_ref[:, 0] - 1).tolist(),
                  "profile_rel": (prof / prof_ref - 1).tolist(),
                  "A_profile_rel": (ring_mean(Fa) / prof_ref - 1).tolist(),
                  "component_at_mirror_gated": [ring_mean(comp[..., j])[:, 0].tolist() for j in range(8)],
                  "component_at_mirror_ungated": [ring_mean(ung[..., j])[:, 0].tolist() for j in range(8)],
                  "component_profile_gated": [ring_mean(comp[..., j]).tolist() for j in range(8)]}
            for nm, (lo, hi) in REGIONS.items():
                sel = (rad >= lo) & (rad < hi)
                w = np.where(ok[:, sel], 1.0, 0.0)
                rr[f"{nm}_rel_l1"] = (np.sum(w * np.abs(Fp[:, sel] - refm[:, sel]), 1) / np.sum(w * refm[:, sel], 1)).tolist()
                rr[f"{nm}_signed"] = (np.sum(w * (Fp[:, sel] - refm[:, sel]), 1) / np.sum(w * refm[:, sel], 1)).tolist()
            rows_.append(rr)
        per_start[tag] = rows_
    res["profiles"] = per_start
    print("A1-A2 done", round(time.monotonic() - t0, 1), flush=True)

    # ---------------- A3 decoded parameters
    par = []
    for r in range(R):
        s1 = stp["P"][r]
        p, g = CM.decode("P", {k: v[M_HARD:M_HARD + 1] for k, v in s1.items()})
        par.append({"start": r, **{k: v[0].cpu().numpy().tolist() for k, v in p.items()}, "gates": g[0].cpu().numpy().tolist()})
    res["parameters"] = par

    # ---------------- A4-A6 sensitivities, response metric, counterfactuals (hard_1, float64)
    fm = {k: KB.to_t(fit[k][M_HARD]) for k in ("wi", "wo", "mean", "w")}
    Pn, L = fit["w"].shape[1:]
    den = (fm["mean"] @ KB.to_t(LUMA) * fm["w"]).sum(1)                      # per-probe luminance energy normaliser
    inc_pk = KB.sph(np.array([20.0, 27.5, 35.0]), np.array([22.5, 22.5, 22.5]))
    mir = inc_pk * np.array([-1, -1, 1.0])
    wi_pk, wo_pk = KB.to_t(inc_pk), KB.to_t(mir)

    def resid(x):                                                            # GN residual vector of the training data
        y, g = x[:58], x[58:].reshape(8, 3, 5)
        F = KB.sdm_eval_t(y, g, fm["wi"].reshape(-1, 3), fm["wo"].reshape(-1, 3)).reshape(Pn, L, 3)
        r_ = (F - fm["mean"]) * (fm["w"] / den[:, None])[..., None]
        return r_.reshape(-1)

    def h(x):                                                                # mirror-peak luminance at 3 incidences
        y, g = x[:58], x[58:].reshape(8, 3, 5)
        return lum_t(KB.sdm_eval_t(y, g, wi_pk, wo_pk))

    def obj(x):
        return KB.objective_t(x[:58], x[58:].reshape(8, 3, 5), fm)

    groups = group_matrix()
    gB = np.concatenate([b_ for _, b_ in groups], 1)                         # 178 x 178 orthonormal
    sens = []
    for r in range(R):
        x0 = KB.to_t(x_of(stp["P"][r], M_HARD))
        J = jac_fwd(resid, x0)    # [N_res, 178]
        M = (J.T @ J).cpu().numpy()
        Jh = torch.autograd.functional.jacobian(h, x0).cpu().numpy()        # [3, 178]
        h0 = h(x0).detach().cpu().numpy()
        ev_, U = np.linalg.eigh(M)
        ev_ = np.clip(ev_, 0, None)
        gate = KB.to_t(x0.cpu().numpy())
        y0 = x0.clone().requires_grad_(True)
        o = obj(y0); o.backward()
        grad = y0.grad.cpu().numpy()
        # projected gradient (box only on the 120 gate coordinates)
        xg = x0.cpu().numpy()[58:]
        pg = grad.copy()
        at_lo, at_hi = xg <= 1e-7, xg >= 1 - 1e-7
        pg[58:][at_lo & (grad[58:] > 0)] = 0
        pg[58:][at_hi & (grad[58:] < 0)] = 0
        lam_max = ev_[-1]
        reg = 1e-10 * lam_max
        Minv = U @ np.diag(1 / (ev_ + reg)) @ U.T
        # peak change attainable at a surrogate-objective increase of eps (relative to the probe-mean metric), per incidence
        eps = 1e-4
        att = [float(np.sqrt(2 * eps * Jh[i] @ Minv @ Jh[i]) / h0[i]) for i in range(3)]
        sloppy = ev_ < 1e-6 * lam_max
        frac_sloppy = [float(((U.T @ Jh[i])[sloppy] ** 2).sum() / (Jh[i] ** 2).sum()) for i in range(3)]
        gsens = {nm: [float(np.linalg.norm(Jh[i] @ blk) / h0[i]) for i in range(3)] for nm, blk in groups}
        # actual (L1) objective cost of an ACTUAL +/-5 % peak change along the least-surrogate-curvature direction
        # u = M_reg^-1 grad h (M_reg = M + 1e-6 lambda_max I, unit norm); the step t is found by scan + bisection on the real response.
        Mreg_inv = U @ np.diag(1 / (ev_ + 1e-6 * lam_max)) @ U.T
        probe5 = []
        xc = x0.cpu().numpy()

        def at(t, u):
            xt = xc + t * u
            xt[58:] = np.clip(xt[58:], 0, 1)
            return KB.to_t(xt)

        with torch.no_grad():
            o0 = float(obj(x0))
            for i in range(3):
                u0 = Mreg_inv @ Jh[i]
                u0 = u0 / np.linalg.norm(u0)
                row = {}
                for sgn, nm in ((1, "plus"), (-1, "minus")):
                    u = sgn * u0
                    target = 0.05
                    ts = np.r_[0, np.geomspace(1e-4, 30, 60)]
                    hv = np.array([float(h(at(t, u))[i]) / h0[i] - 1 for t in ts])
                    hit = np.flatnonzero(np.abs(hv) >= target)
                    if len(hit) == 0:
                        row[nm] = {"reached": False, "max_abs_peak_change": float(np.abs(hv).max())}
                        continue
                    lo_, hi_ = ts[hit[0] - 1], ts[hit[0]]
                    for _ in range(40):
                        mid = 0.5 * (lo_ + hi_)
                        if abs(float(h(at(mid, u))[i]) / h0[i] - 1) >= target:
                            hi_ = mid
                        else:
                            lo_ = mid
                    xt = at(hi_, u)
                    row[nm] = {"reached": True, "step_norm": float(hi_), "objective_rel_change": float(obj(xt)) / o0 - 1,
                               "peak_rel_change": float(h(xt)[i]) / h0[i] - 1}
                probe5.append(row)
        sens.append({"start": r, "objective": float(o.detach()), "grad_norm": float(np.linalg.norm(grad)),
                     "projected_grad_norm": float(np.linalg.norm(pg)), "projected_grad_norm_y": float(np.linalg.norm(pg[:58])),
                     "projected_grad_norm_gates": float(np.linalg.norm(pg[58:])),
                     "gate_knots_at_lower": int(at_lo.sum()), "gate_knots_at_upper": int(at_hi.sum()),
                     "metric_eigenvalues": ev_.tolist(), "metric_condition_1e-12_floor": float(lam_max / max(ev_[0], 1e-12 * lam_max)),
                     "n_eig_below_1e-6": int(sloppy.sum()), "n_eig_below_1e-9": int((ev_ < 1e-9 * lam_max).sum()),
                     "peak_rel_change_at_surrogate_eps_1e-4": att, "peak_gradient_fraction_in_sloppy_subspace": frac_sloppy,
                     "peak_rel_sensitivity_by_group": gsens, "peak_height_mirror": h0.tolist(),
                     "five_percent_peak_probe": probe5})
        if r == 0:
            res["groups"] = [nm for nm, _ in groups]
        torch.cuda.empty_cache()
    res["sensitivity"] = sens
    print("A4 done", round(time.monotonic() - t0, 1), flush=True)

    # differences good vs bad, in the metric eigenbasis of the midpoint
    diffs = []
    for rb in BAD:
        for rg in GOOD:
            xb, xg_ = x_of(stp["P"][rb], M_HARD), x_of(stp["P"][rg], M_HARD)
            xm = KB.to_t(0.5 * (xb + xg_))
            J = jac_fwd(resid, xm)
            M = (J.T @ J).cpu().numpy()
            ev_, U = np.linalg.eigh(np.clip(M, -np.inf, np.inf))
            ev_ = np.clip(ev_, 0, None)
            d = xg_ - xb
            c = U.T @ d
            lam_max = ev_[-1]
            bands = {"sloppy_<1e-6": ev_ < 1e-6 * lam_max, "mid": (ev_ >= 1e-6 * lam_max) & (ev_ < 1e-2 * lam_max),
                     "stiff_>=1e-2": ev_ >= 1e-2 * lam_max}
            gnorm = {nm: float(np.linalg.norm(blk.T @ d)) for nm, blk in groups}
            # path
            ts = np.linspace(0, 1, 21)
            path = {"t": ts.tolist(), "objective": [], "peak_rel": [], "heldout_el1": []}
            for t in ts:
                x = KB.to_t((1 - t) * xb + t * xg_)
                with torch.no_grad():
                    path["objective"].append(float(obj(x)))
                    hv = h(x).cpu().numpy()
                path["peak_rel"].append(hv.tolist())
                y_, g_ = split((1 - t) * xb + t * xg_)
                st1 = {"y": y_[None], "g": g_[None]}
                pr = CM.predict("P", st1, ev["wi"][M_HARD:M_HARD + 1], ev["wo"][M_HARD:M_HARD + 1])[0]
                path["heldout_el1"].append(float(CM.wl82_terms(pr, ev["mean"][M_HARD], ev["w"][M_HARD])["el1"].mean()))
            # integrated gradients of the mirror peak (3 incidences) along the straight path, grouped
            K_ = 64
            ig = np.zeros((3, 178))
            for k in range(K_):
                t = (k + 0.5) / K_
                x = KB.to_t((1 - t) * xb + t * xg_)
                Jh = torch.autograd.functional.jacobian(h, x).cpu().numpy()
                ig += Jh * d[None] / K_
            hb = h(KB.to_t(xb)).detach().cpu().numpy(); hg = h(KB.to_t(xg_)).detach().cpu().numpy()
            ig_g = {nm: [float((ig[i] @ blk @ blk.T).sum() / hb[i]) for i in range(3)] for nm, blk in groups}
            # group swaps (bad state with one group taken from the good state, and the reverse)
            swaps = {}
            for nm, blk in groups:
                Pj = blk @ blk.T
                xs = xb + Pj @ d
                xr = xg_ - Pj @ d
                with torch.no_grad():
                    swaps[nm] = {"bad_with_good_group_peak_rel_change": (h(KB.to_t(xs)).cpu().numpy() / hb - 1).tolist(),
                                 "bad_with_good_group_objective_change": float(obj(KB.to_t(xs)) - obj(KB.to_t(xb))),
                                 "good_with_bad_group_peak_rel_change": (h(KB.to_t(xr)).cpu().numpy() / hg - 1).tolist(),
                                 "good_with_bad_group_objective_change": float(obj(KB.to_t(xr)) - obj(KB.to_t(xg_)))}
            diffs.append({"bad": rb, "good": rg, "distance_raw": float(np.linalg.norm(d)),
                          "response_metric_distance": float(np.sqrt(max(d @ M @ d, 0))),
                          "fraction_of_difference_in_band": {k: float((c[v] ** 2).sum() / (c ** 2).sum()) for k, v in bands.items()},
                          "difference_norm_by_group": gnorm, "path": path,
                          "peak_rel_difference_good_vs_bad": (hg / hb - 1).tolist(),
                          "integrated_gradient_by_group_rel": ig_g,
                          "integrated_gradient_completeness_residual_rel": ((ig.sum(1) - (hg - hb)) / hb).tolist(),
                          "group_swaps": swaps})
            torch.cuda.empty_cache()
    res["good_vs_bad"] = diffs
    print("A5-A6 done", round(time.monotonic() - t0, 1), flush=True)

    # ---------------- A7 numerical audit of every terminal P state (all materials)
    audit = []
    for m in range(n):
        fmm = {k: KB.to_t(fit[k][m]) for k in ("wi", "wo", "mean", "w")}
        for r in range(R):
            x0 = KB.to_t(x_of(stp["P"][r], m)).requires_grad_(True)
            o = KB.objective_t(x0[:58], x0[58:].reshape(8, 3, 5), fmm)
            o.backward()
            g_ = x0.grad.cpu().numpy(); xg = x0.detach().cpu().numpy()[58:]
            pg = g_.copy()
            pg[58:][(xg <= 1e-7) & (g_[58:] > 0)] = 0
            pg[58:][(xg >= 1 - 1e-7) & (g_[58:] < 0)] = 0
            audit.append({"material": KB.NAMES[rows[m]["tag"]], "start": r, "objective": float(o.detach()),
                          "projected_grad_norm": float(np.linalg.norm(pg)), "y_grad_norm": float(np.linalg.norm(pg[:58])),
                          "gate_grad_norm": float(np.linalg.norm(pg[58:])),
                          "gate_fraction_at_bounds": float(((xg <= 1e-7) | (xg >= 1 - 1e-7)).mean()),
                          "final_adam_step_bound": 3e-4})
    res["terminal_audit"] = audit

    # ---------------- scene lanes (WL104K-A hard_1, archived physical response): matched-direction component shares
    with np.load(KB.REPO / "results/ecro_wl84_104ka/primary/lanes/test83_hard_1_L1/lanes.npz") as z:
        Ln = {k: z[k] for k in ("wi", "wo", "lf", "ref_a", "ref_b")}
    rad_l = KB.mirror_radius(Ln["wi"].astype(np.float64), Ln["wo"].astype(np.float64))
    lane = {}
    for nm, (lo, hi) in {"0_3": (0, 3), "3_12": (3, 12), "12_24": (12, 24), "24_180": (24, 181)}.items():
        sel = (rad_l >= lo) & (rad_l < hi)
        lw = Ln["lf"].astype(np.float64)[sel]
        cref_a = (Ln["ref_a"][sel].astype(np.float64) * lw) @ LUMA
        cref_b = (Ln["ref_b"][sel].astype(np.float64) * lw) @ LUMA
        cref = 0.5 * (cref_a + cref_b)
        rows_ = []
        for r in range(R):
            s1 = {k: v[M_HARD:M_HARD + 1] for k, v in stp["P"][r].items()}
            comp = CM.comp_responses("P", s1, Ln["wi"][sel][None, None].astype(np.float64), Ln["wo"][sel][None, None].astype(np.float64))[0, 0]
            cc = (comp * lw[:, None, :]) @ LUMA                                                         # [L, 8]
            rows_.append({"start": r, "component_share_of_reference": (cc.sum(0) / cref.sum()).tolist(),
                          "total_signed_excess": float(cc.sum() / cref.sum() - 1)})
        lane[nm] = {"lanes": int(sel.sum()), "reference_share_of_all_lanes": None, "reference_sum": float(cref.sum()),
                    "reference_half_se_rel": float(np.sqrt((((cref_a - cref_b) / 2) ** 2).sum()) / cref.sum()), "starts": rows_}
    tot = sum(v["reference_sum"] for v in lane.values())
    for v in lane.values():
        v["reference_share_of_all_lanes"] = v["reference_sum"] / tot
    res["scene_lanes_by_mirror_offset"] = lane
    res.update({"commit": KB.commit(), "inputs_sha256": seen, "seconds": time.monotonic() - t0,
                "torch": torch.__version__, "gpu": torch.cuda.get_device_name()})
    KB.write(args.out / "stageA.json", res)
    print("stageA done", round(time.monotonic() - t0, 1), flush=True)


if __name__ == "__main__":
    main()
