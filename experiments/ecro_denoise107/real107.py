"""Held-out WL107 O0/O1/O2 measurement with independent reference and covariance."""
import argparse
from pathlib import Path
import time

import numpy as np
from scipy.ndimage import distance_transform_edt
import torch

from common107 import (LAMBDAS, LUMA, P, SPPS, VIEWS, coverage_masks, low_high, panel, read, rois, sha,
                       stencil, take_stencil, total, write)
from oracle107 import adaptive_intervals, adaptive_risk_bound, empirical_parts, min_adaptive_variance, solve_factor_torch
from reference107 import DECISION_ROIS, rel_rmse
from train105 import D_aux
import model105 as model


def luma(image):
    return np.einsum("...chw,c->...hw", image, LUMA, optimize=False)


def render_o0(net, comp, aux):
    with torch.no_grad():
        y, trans = net(torch.from_numpy(comp)[None].cuda(), torch.from_numpy(aux)[None].cuda())
    stack = np.stack([trans[key][0].cpu().numpy() for key in ("low", "high")])
    out = y[0].cpu().numpy()
    np.testing.assert_array_equal(stack.sum(0), out)
    return out, stack


def profile_masks(aux, masks):
    coverage = aux["cover"][6]
    silhouette = masks["silhouette"]
    bins = coverage_masks(coverage, silhouette)
    d = distance_transform_edt(~silhouette).astype(np.int16)
    d = np.where(coverage >= 0.5, d, -d)
    profile = {f"distance_{j}": (d == j) & masks["primary"] for j in range(-7, 8)}
    profile["distance_0"] = silhouette
    strata = {}
    for name, guide in (("mean_normal_z", aux["normal"][2]), ("mean_depth", aux["depth"])):
        cut = np.quantile(guide[silhouette], (1 / 3, 2 / 3))
        groups = np.searchsorted(cut, guide, side="right")
        strata.update({f"{name}_{k}": silhouette & (groups == k) for k in range(3)})
    return {**masks, **{f"coverage_{i}": m for i, m in enumerate(bins) if m.sum() >= 20},
            **{k: m for k, m in {**profile, **strata}.items() if m.sum() >= 20}}


def metric_rows(out, parts, target, masks):
    from analyze105 import grad_ratio

    lt = luma(target)
    lm = luma(out)
    rgb_var = out.var(axis=0, ddof=1)
    vl, vh, ch, vt = empirical_parts(luma(parts[:, 0]), luma(parts[:, 1]))
    assert np.allclose(vt, lm.var(axis=0, ddof=1), rtol=1e-4, atol=1e-10)
    d = lm - lt
    mean_hdr = out.mean(0)
    rows = {}
    for name, mask in masks.items():
        if not mask.any():
            continue
        lt_mean = float(lt[mask].mean())
        rows[name] = {
            "pixels": int(mask.sum()),
            "target_mean_luma": lt_mean,
            "relative_RGB_RMSE_draws": [rel_rmse(x, target, mask) for x in out],
            "signed_bias_draws": (d[:, mask].mean(axis=1) / lt_mean).tolist(),
            "luma_ensemble_spatial_RMSE": float(np.sqrt((d.mean(0)[mask] ** 2).mean()) / lt_mean),
            "linear_luma_variance": float(vt[mask].mean()),
            "linear_RGB_variance": float(rgb_var[:, mask].mean()),
            "signed_luma_covariance_terms": {k: float(v[mask].mean())
                                             for k, v in zip(("LOW", "HIGH", "2Cov", "TOTAL"), (vl, vh, ch, vt))},
            "squared_bias_RGB_ensemble": float(((mean_hdr - target)[:, mask] ** 2).mean()),
        }
        if name in ("silhouette", "material_boundary"):
            rows[name]["gradient_ratio_draws"] = [grad_ratio(o, target, mask) for o in out]
    return rows


def compose_taps(comp, indices):
    return take_stencil(low_high(comp), indices)


def fit_design(mu, draws, spp):
    """P x (3+3R) x 450 paired covariance factor, centered on simplex equality."""
    # mu[2, 3, P, 225]; draws[R, 2, 3, P, 225]
    m = mu.transpose(2, 1, 0, 3)
    x = draws.transpose(3, 0, 2, 1, 4).astype(np.float64)
    x -= x.mean(axis=1, keepdims=True)
    scaled = x * np.sqrt((32 / spp) / (3 * (draws.shape[0] - 1)))
    b = scaled[..., 112].sum(axis=-1).reshape(len(m), -1)
    bias = (m - m[..., 112:113]).reshape(len(m), 3, -1) / np.sqrt(3)
    var = (scaled - scaled[..., 112:113]).reshape(len(m), x.shape[1] * 3, -1)
    return bias, var, b


def panel_metrics(pred, target, coverage):
    pl = pred @ LUMA
    t = target @ LUMA
    norm = t.mean()
    b = (pl.mean(axis=0) - t) / norm
    bins = coverage_masks(coverage, np.ones_like(coverage, dtype=bool))
    profile = [float(b[m].mean()) if m.any() else None for m in bins]
    return {"pixels": len(b), "signed_bias": float(b.mean()), "coverage_profile": profile,
            "coverage_profile_rms_bias": float(np.sqrt(np.mean([x * x for x in profile if x is not None]))),
            "luma_variance": float(pl.var(axis=0, ddof=1).mean()),
            "RGB_rmse": float(np.sqrt(((pred - target) ** 2).mean() / (target ** 2).mean()))}


def oracle_panel(view, fit, target, draws, capture, out, matches, deadline):
    idx = panel(matches["aux"])
    taps = stencil(idx, 512, 512)
    coverage = matches["aux"]["cover"][6].ravel()[idx]
    all_cov = np.stack([np.load(capture / f"covariance_{i:02d}.npy") for i in range(64)])
    cv = np.stack((all_cov[:, :3] + all_cov[:, 3:6], all_cov[:, 6:9]), axis=1)
    fit_taps = compose_taps(fit, taps)
    ass_taps = compose_taps(target, taps)
    labels = ass_taps[..., 112].sum(axis=0).T
    all_rows = {}
    raw_data = {"panel_indices": idx, "target": labels, "coverage": coverage}
    for spp, nz in draws.items():
        rng_idx = str(spp)
        actual = compose_taps(nz, taps)
        bias, var, offset = fit_design(fit_taps, cv, spp)
        o0 = matches["R3"][spp].reshape(len(nz), 3, -1)[..., idx].transpose(0, 2, 1)
        raw = actual[..., 112].sum(axis=1).transpose(0, 2, 1)
        fit_centre = cv[..., 112].sum(axis=1).transpose(0, 2, 1)
        predicted_raw_var = float((fit_centre @ LUMA).var(axis=0, ddof=1).mean() * (32 / spp))
        rec = {"R0": panel_metrics(raw, labels, coverage), "O0": panel_metrics(o0, labels, coverage),
               "covariance_scale_control": {"fit_32_over_spp_raw_luma_var": predicted_raw_var},
               "rows": {}}
        rec["covariance_scale_control"]["heldout_over_fit_raw_variance"] = (
            rec["R0"]["luma_variance"] / predicted_raw_var)
        for lam in LAMBDAS:
            if time.monotonic() > deadline:
                raise TimeoutError("Frozen WL107 evaluation budget exhausted")
            weights, values, lbs, gaps = [], [], [], []
            for k in range(0, len(idx), 24):
                sl = slice(k, k + 24)
                f = np.concatenate((bias[sl], np.sqrt(lam) * var[sl]), axis=1)
                b = np.concatenate((np.zeros((len(f), 3)), np.sqrt(lam) * offset[sl]), axis=1)
                w, v, lb, gap = solve_factor_torch(f, b, "cuda")
                weights.append(w)
                values.extend(v)
                lbs.extend(lb)
                gaps.extend(gap)
            w = np.concatenate(weights, axis=0)
            pred = np.einsum("rgcpt,pgt->rgpc", actual, w, dtype=np.float64)
            clean = np.einsum("gcpt,pgt->gpc", ass_taps, w, dtype=np.float64).sum(0)
            row = panel_metrics(pred.sum(axis=1), labels, coverage)
            row["reference_mean"] = panel_metrics(np.repeat(clean[None], 2, axis=0), labels, coverage)
            row["covariance_luma"] = {k: float(x.mean()) for k, x in zip(
                ("LOW", "HIGH", "2Cov", "TOTAL"), empirical_parts(pred[:, 0] @ LUMA, pred[:, 1] @ LUMA))}
            row["variance_ratio_to_R0"] = row["luma_variance"] / rec["R0"]["luma_variance"]
            row["fitted_RGB_risk"] = float(np.mean(values))
            row["fitted_RGB_risk_lower_bound"] = float(np.mean(lbs))
            row["fitted_RGB_mean_gap_relative"] = float(np.mean(gaps) / np.mean(values))
            row["held_out_RGB_risk"] = float(((pred.sum(axis=1) - labels) ** 2).mean())
            row["panel_contract"] = bool(abs(row["reference_mean"]["signed_bias"]) <= 0.01
                                         and row["reference_mean"]["coverage_profile_rms_bias"] <= 0.01
                                         and row["variance_ratio_to_R0"] <= 0.50)
            rec["rows"][str(lam)] = row
            key = f"{rng_idx}_{lam:g}"
            raw_data["w_" + key] = w.astype(np.float32)
            raw_data["O1_" + key] = pred.astype(np.float32)
        all_rows[rng_idx] = rec
        print("O1 panel", view, spp, flush=True)
    np.savez_compressed(out / f"{view}_panel.npz", **raw_data)
    return all_rows


def adaptive_real(nz, target, silhouette):
    all_idx = np.flatnonzero(silhouette)
    lnz = luma(low_high(nz))
    tl = luma(target).ravel()[all_idx]
    taps = take_stencil(lnz, stencil(all_idx, *silhouette.shape)).transpose(0, 2, 1, 3)
    lo, hi = adaptive_intervals(taps)
    vmin = min_adaptive_variance(lo, hi)
    risk_lb = adaptive_risk_bound(lo, hi, tl).mean()
    raw_var = taps[..., 112].sum(axis=-1).var(axis=0, ddof=1).mean()
    return {"pixels": len(tl), "relative_luma_rmse_lb_on_estimated_target": float(np.sqrt(risk_lb / (tl ** 2).mean())),
            "min_luma_variance": float(vmin.mean()), "raw_luma_variance": float(raw_var),
            "min_variance_to_raw": float(vmin.mean() / raw_var),
            "fraction_nonzero_seed_interval_gap": float(np.mean(lo.max(axis=0) > hi.min(axis=0)))}


def intervals(a, b=None, kind="mean", seed=P["seeds"]["bootstrap"]):
    """Paired two-level intervals or a paired view cluster for variance ratios."""
    rng = np.random.default_rng(seed)
    n = len(a)
    v = rng.integers(n, size=(10000, n))
    if a.ndim == 2:
        r = rng.integers(a.shape[1], size=(10000, n, a.shape[1]))
        aa = a[v[..., None], r].mean((1, 2))
        bb = b[v[..., None], r].mean((1, 2)) if b is not None else None
    else:
        aa = a[v].mean(axis=-1)
        bb = b[v].mean(axis=-1) if b is not None else None
    if kind == "ratio":
        values, point = aa / bb, a.mean() / b.mean()
    elif kind == "bias_improvement":
        values = 1 - np.abs(aa) / np.maximum(np.abs(bb), 1e-12)
        point = 1 - np.abs(a.mean()) / max(np.abs(b.mean()), 1e-12)
    else:
        values, point = aa, a.mean()
    return {"point": float(point), "ci95": np.quantile(values, [0.025, 0.975]).tolist()}


def architecture_metrics(recs, trust):
    report = {}
    for spp in map(str, SPPS):
        a = {arm: [recs[view][spp][arm] for view in VIEWS] for arm in ("R0", "O0", "O2")}
        rows = report[spp] = {}
        for roi in sorted(set().union(*(row.keys() for row in a["O0"]))):
            comparable = [i for i in range(len(VIEWS)) if all(roi in series[i] for series in a.values())]
            if not comparable:
                continue
            r = {arm: np.array([series[i][roi]["relative_RGB_RMSE_draws"] for i in comparable]) for arm, series in a.items()}
            b = {arm: np.array([series[i][roi]["signed_bias_draws"] for i in comparable]) for arm, series in a.items()}
            v = {arm: np.array([series[i][roi]["linear_luma_variance"] for i in comparable]) for arm, series in a.items()}
            rows[roi] = {"rel_rmse": {k: intervals(x) for k, x in r.items()},
                         "signed_bias": {k: intervals(x) for k, x in b.items()},
                         "luma_variance": {k: intervals(x) for k, x in v.items()},
                         "O2_RMSE_ratio_O0": intervals(r["O2"], r["O0"], "ratio"),
                         "O2_abs_bias_reduction_over_O0": intervals(b["O2"], b["O0"], "bias_improvement"),
                         "O2_variance_ratio_R0": intervals(v["O2"], v["R0"], "ratio")}
            if roi in DECISION_ROIS:
                q = trust["pooled_reference_rgb_rms_se"][roi] / min(r["O0"].mean(), r["O2"].mean())
                rows[roi]["reference_noise_over_min_RMSE"] = float(q)
                rows[roi]["reference_pass"] = bool(q <= 0.10 and trust["max_view_stratum_signed_relative_se"] <= 0.0025)
            if roi == "silhouette":
                rows[roi]["gradient_ratio"] = {arm: intervals(np.array([series[i][roi]["gradient_ratio_draws"]
                                                                         for i in comparable]))
                                               for arm, series in a.items()}
        pr, si, mb = (rows[k] for k in DECISION_ROIS)
        rows["O2_gates"] = {"target": all(r["reference_pass"] for r in (pr, si, mb)),
                            "signed_bias": all(abs(r["signed_bias"]["O2"]["point"]) <= 0.01 for r in (pr, si, mb)),
                            "silhouette_variance": si["O2_variance_ratio_R0"]["ci95"][1] <= 0.50,
                            "primary_nonregression": pr["O2_RMSE_ratio_O0"]["point"] <= 1.02
                                                     and pr["O2_RMSE_ratio_O0"]["ci95"][1] <= 1.05,
                            "boundary_nonregression": max(r["O2_RMSE_ratio_O0"]["point"] for r in (si, mb)) <= 1.05,
                            "silhouette_bias_improvement": si["O2_abs_bias_reduction_over_O0"]["point"] >= 0.50
                                                           and si["O2_abs_bias_reduction_over_O0"]["ci95"][0] > 0,
                            "gradient_safety": 0.8 <= si["gradient_ratio"]["O2"]["point"] <= 1.2}
    return report


def main(baseline, capture, reference, diagnostic, replay, output, commit):
    deadline = time.monotonic() + 3600
    trust = read(reference / "reference107.json")
    assert trust["preliminary_pass"] and read(replay)["repeat_bitwise"]
    cman = read(capture / "capture_manifest.json")
    oman = read(diagnostic / "manifest.json") if diagnostic is not None else None
    assert cman["complete"]
    assert all(sha(capture / f) == r["sha256"] for f, r in cman["files"].items()
               if "/fit_" in f or "/covariance_" in f)
    if oman is not None:
        assert all(sha(diagnostic / f) == h for f, h in oman["files"].items())
    output.mkdir(parents=True, exist_ok=False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    model.FUSED = True
    net = model.Reconstructor("R3").cuda()
    net.load_state_dict(torch.load(baseline / "models/R3.pt", map_location="cuda"))
    net.eval()
    rec = {"analysis_commit": commit, "diagnostic_weights_sha256": oman["weights_sha256"] if oman is not None else None,
           "qualified_reference_manifest": sha(reference / "reference107.json"),
           "capture_manifest": sha(capture / "capture_manifest.json"), "views": {}}
    for view in VIEWS:
        folder = baseline / "data/eval" / view
        with np.load(folder / "aux.npz") as z:
            aux = {k: z[k] for k in z.files}
        target_file = reference / f"{view}_target.npy"
        assert sha(target_file) == trust["views"][view]["target_sha256"]
        target = np.load(target_file)
        y = total(target).astype(np.float64)
        fit = np.mean([np.load(capture / view / f"fit_{h:02d}.npy") for h in range(8)], axis=0, dtype=np.float64)
        masks = profile_masks(aux, rois(aux))
        rows = rec["views"][view] = {}
        archive = {}
        matches = {"aux": aux, "R3": {}}
        rawmaps = {"target": y.astype(np.float32), "silhouette": masks["silhouette"], "coverage": aux["cover"][6]}
        for spp in SPPS:
            aux_spp = D_aux(aux, spp)
            noisy = []
            arms = ("R0", "O0", "O2") if diagnostic is not None else ("R0", "O0")
            pred = {a: [] for a in arms}
            parts = {a: [] for a in arms}
            for r in range(8):
                with np.load(folder / f"noisy_spp{spp}_r{r}.npz") as z:
                    nz = z["comp"]
                noisy.append(nz)
                pred["R0"].append(total(nz))
                parts["R0"].append(low_high(nz))
                ro, rp = render_o0(net, nz, aux_spp)
                pred["O0"].append(ro)
                parts["O0"].append(rp)
                if diagnostic is not None:
                    with np.load(diagnostic / f"{view}_spp{spp}_r{r}.npz") as z:
                        pred["O2"].append(z["O2"])
                        parts["O2"].append(np.stack((z["low"], z["high"])))
            rows[str(spp)] = {arm: metric_rows(np.stack(pred[arm]).astype(np.float64), np.stack(parts[arm]), y, masks)
                              for arm in pred}
            archive[spp] = np.stack(noisy)
            matches["R3"][spp] = np.stack(pred["O0"])
            rows[str(spp)]["adaptive_bound"] = adaptive_real(archive[spp], y, masks["silhouette"])
            for arm in pred:
                rawmaps[f"{arm}_{spp}_mean"] = np.mean(pred[arm], axis=0)
                rawmaps[f"{arm}_{spp}_seed0"] = pred[arm][0]
            print("Real metrics", view, spp, flush=True)
        rows["O1"] = oracle_panel(view, fit, target, archive, capture / view, output, matches, deadline)
        np.savez_compressed(output / f"{view}_maps.npz", **rawmaps)
        write(output / "real107.json", rec)
    if diagnostic is not None:
        rec["pooled"] = architecture_metrics(rec["views"], trust)
    rec["complete"] = True
    write(output / "real107.json", rec)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("baseline", "capture", "reference", "replay", "output"):
        p.add_argument(name, type=Path)
    p.add_argument("commit")
    p.add_argument("--diagnostic", type=Path)
    a = p.parse_args()
    main(a.baseline, a.capture, a.reference, a.diagnostic, a.replay, a.output, a.commit)
