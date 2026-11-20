"""Paired factorial effects and frozen eligibility rules."""
import numpy as np

from common108 import ARMS, P, SEEDS, SPPS


def draws(nviews, ndraws):
    config = P["evaluation"]
    rng = np.random.default_rng(config["boot_seed"])
    n = config["boot_draws"]
    return (rng.integers(len(SEEDS), size=(n,len(SEEDS))), rng.integers(nviews,size=(n,nviews)),
            rng.integers(ndraws,size=(n,nviews,ndraws)))


def estimate(x, indices):
    """Data axis order: training seed, view, independent noisy frame."""
    iseed, iview, idraw = indices
    if x.shape[0] == 1:
        iseed = np.zeros_like(iseed)
    if x.ndim == 2:
        return x[iseed[:,:,None], iview[:,None,:]].mean((1,2))
    return x[iseed[:,:,None,None], iview[:,None,:,None], idraw[:,None,:,:]].mean((1,2,3))


def with_ci(point, sample):
    return {"point": float(point), "ci95": np.quantile(sample,(.025,.975)).tolist()}


def effect(new, old, indices, kind="improvement"):
    nxt, prev = estimate(new, indices), estimate(old, indices)
    if kind == "absolute_bias":
        nxt, prev = np.abs(nxt), np.abs(prev)
        point = 1 - abs(new.mean())/max(abs(old.mean()), 1e-15)
        return with_ci(point, 1 - nxt/np.maximum(prev, 1e-15))
    ratio = new.mean()/old.mean()
    if kind == "ratio":
        return with_ci(ratio,nxt/prev)
    return with_ci(1-ratio,1-nxt/prev)


def array(rows, arm, roi, key):
    seeds = ("baseline",) if arm in ("R0","R3") else tuple(str(s) for s in SEEDS)
    return np.array([[v[arm][seed]["rois"][roi][key] for v in rows] for seed in seeds],dtype=float)


CONTRASTS = {
    "A1_over_A0_objective": ("A1","A0"),
    "B0_over_A0_descriptor": ("B0","A0"),
    "B1_over_A1_descriptor": ("B1","A1"),
    "B1_over_B0_objective": ("B1","B0"),
    **{f"{a}_over_R3": (a,"R3") for a in ARMS}
}


def summarize(views, reference):
    aggregate = {}
    for spp in SPPS:
        rows = [v[str(spp)] for v in views.values()]
        roi_names = set.intersection(*(set(v["R3"]["baseline"]["rois"]) for v in rows))
        metrics = {}
        indices = draws(len(rows),P["evaluation"]["reps"])
        for roi in sorted(roi_names):
            outputs = ("R0","R3",*ARMS)
            errors = {a:array(rows,a,roi,"relative_RGB_RMSE_draws") for a in outputs}
            bias = {a:array(rows,a,roi,"signed_bias_draws") for a in outputs}
            var = {a:array(rows,a,roi,"linear_luma_variance") for a in outputs}
            bpower = {a:array(rows,a,roi,"seed_corrected_bias_power_RGB") for a in outputs}
            cell = {"quality": {}, "effects": {}}
            for a in outputs:
                c = {label:float(array(rows,a,roi,k).mean()) for label,k in
                     (("luma_variance", "linear_luma_variance"), ("RGB_variance", "linear_RGB_variance"))}
                c["component_luma_covariance"] = {
                    k:float(np.mean([[cell["rois"][roi]["signed_luma_covariance_terms"][k]
                                      for cell in v[a].values()] for v in rows]))
                    for k in ("LOW","HIGH","2Cov","TOTAL")}
                c["relative_rmse"] = with_ci(errors[a].mean(),estimate(errors[a], indices))
                c["signed_bias"] = with_ci(bias[a].mean(),estimate(bias[a], indices))
                c["mean_per_view_seed_absolute_bias"] = float(np.abs(bias[a].mean(2)).mean())
                c["seed_corrected_bias_power_RGB"] = float(bpower[a].mean())
                c["variance_over_R0"] = effect(var[a],var["R0"],indices,"ratio") if a != "R0" else None
                c["view_rmse"] = errors[a].mean((0,2)).tolist()
                c["view_absolute_signed_bias"] = np.abs(bias[a].mean(2)).mean(0).tolist()
                if roi in ("silhouette","material_boundary"):
                    c["gradient_ratio"] = float(array(rows,a,roi,"gradient_ratio_draws").mean())
                cell["quality"][a] = c
            for name,(a,b) in CONTRASTS.items():
                cell["effects"][name] = {"rmse_improvement":effect(errors[a],errors[b],indices),
                                          "rmse_ratio":effect(errors[a],errors[b],indices,"ratio"),
                                          "absolute_bias_improvement":effect(bias[a],bias[b],indices,"absolute_bias"),
                                          "variance_ratio":effect(var[a],var[b],indices,"ratio")}
            aw = (errors["A1"]-errors["B1"])-(errors["A0"]-errors["B0"])
            cell["descriptor_objective_interaction_RMSE"] = with_ci(aw.mean(),estimate(aw,indices))
            if roi in reference["pooled_reference_rgb_rms_se"]:
                cell["reference_noise_over_smallest_learned_rmse"] = float(
                    reference["pooled_reference_rgb_rms_se"][roi]/min(errors[a].mean() for a in ("R3",*ARMS)))
            metrics[roi] = cell
        aggregate[str(spp)] = metrics
    return aggregate


def diagnostic_gate(summary, contrast, nviews):
    first = summary["4"]["silhouette"]
    new, old = CONTRASTS[contrast]
    first_effect = first["effects"][contrast]["rmse_improvement"]
    per_view = sum(np.asarray(first["quality"][new]["view_rmse"]) < first["quality"][old]["view_rmse"])
    bands = ("silhouette","inside_1","inside_2","inside_3")
    improving_bands = sum(summary["4"][roi]["effects"][contrast]["rmse_improvement"]["point"] > 0
                          for roi in bands if roi in summary["4"])
    return {
        "substantial_at_4": first_effect["point"] >= .15 and first_effect["ci95"][0] > 0,
        "positive_at_16_and_32": all(summary[s]["silhouette"]["effects"][contrast]["rmse_improvement"]["point"] > 0
                                     for s in ("16","32")),
        "primary_le_105pct_at_all_spp": all(v["primary"]["effects"][contrast]["rmse_ratio"]["point"] <= 1.05
                                           for v in summary.values()),
        "variance_gate": all(v["silhouette"]["effects"][contrast]["variance_ratio"]["point"] <= 1.05
                             and v["silhouette"]["quality"][new]["variance_over_R0"]["point"] <= .5
                             for v in summary.values()),
        "view_and_shell_direction": bool(per_view > nviews/2 and improving_bands >= 2)
    }


def decide(summary, reference, nviews):
    decision = ("primary","silhouette","material_boundary")
    trust = reference["max_view_stratum_signed_relative_se"] <= .0025 and all(
        r[q]["reference_noise_over_smallest_learned_rmse"] <= .10 for r in summary.values() for q in decision)
    source = diagnostic_gate(summary,"A1_over_A0_objective",nviews)
    cross = summary["4"]["silhouette"]["effects"]["A1_over_R3"]
    source["R3_guard"] = all(q["primary"]["effects"]["A1_over_R3"]["rmse_ratio"]["point"] <= 1.05
                             for q in summary.values()) and cross["rmse_improvement"]["ci95"][0] > 0
    source["bias_reduction_guard"] = cross["absolute_bias_improvement"]["point"] >= .5 and (
        summary["4"]["silhouette"]["effects"]["A1_over_A0_objective"]["absolute_bias_improvement"]["point"] >= .5)
    incremental = diagnostic_gate(summary,"B1_over_A1_descriptor",nviews)
    production = {}
    for a in ARMS:
        production[a] = {}
        for spp,q in summary.items():
            primary,sil,material = (q[key] for key in decision)
            contrast = f"{a}_over_R3"
            primary_ratio = primary["effects"][contrast]["rmse_ratio"]
            bias_gain = sil["effects"][contrast]["absolute_bias_improvement"]
            production[a][spp] = {
                "target": trust,
                "signed_bias": all(abs(v["quality"][a]["signed_bias"]["point"]) <= .01
                                   for v in (primary,sil,material)),
                "gradient": .8 <= sil["quality"][a]["gradient_ratio"] <= 1.2,
                "variance": sil["quality"][a]["variance_over_R0"]["ci95"][1] <= .5,
                "primary_rmse": primary_ratio["point"] <= 1.02 and primary_ratio["ci95"][1] <= 1.05,
                "boundary_rmse": all(v["effects"][contrast]["rmse_ratio"]["point"] <= 1.05 for v in (sil,material)),
                "silhouette_bias_improvement": bias_gain["point"] >= .5 and bias_gain["ci95"][0] > 0,
            }
    return {"reference_qualified": trust, "original_input": source, "descriptor_incremental": incremental,
            "original_input_pass": trust and all(source.values()), "descriptor_incremental_pass": trust and all(incremental.values()),
            "production_comparison": production}
