"""Frozen decision rules for the matched factorial and historical comparator."""

import numpy as np

from common104d import P
from scenes104d import ROIS
from stats104b import jk_ci


def contrast(response, scenes, rows, candidate, control):
    rule = P["decision"]
    baseline, tested = response["arms"][control], response["arms"][candidate]
    band = np.asarray(baseline["band_shape"])
    delta = band - np.asarray(tested["band_shape"])
    half = np.asarray(baseline["band_shape_reference_halves"]) - np.asarray(tested["band_shape_reference_halves"])
    guarded = ("overall_albedo", "overall_el1", "grazing_shape")
    regressed = {}
    for key, limit in zip(guarded, ("regress_albedo", "regress_overall_el1", "regress_grazing_eshape")):
        regressed[key] = np.asarray(tested[key]) - np.asarray(baseline[key]) > rule[limit]
    scene_gains, scene_guards = [], []
    for row in rows:
        image = scenes[row["image"]]["arms"]
        cb, xb = (image[a]["stable_rms"]["objects"] for a in (control, candidate))
        scene_gains.append(jk_ci(cb["point"] - xb["point"], np.asarray(cb["loo"]) - np.asarray(xb["loo"])))
        shape_guard = xb["point"] - cb["point"] <= rule["regress_scene_rms"]
        absolute_signed_growth = max(
            abs(x["point"]) - abs(image[control]["signed"][roi][key]["point"])
            for roi in ROIS for key, x in image[candidate]["signed"].get(roi, {}).items()
        )
        scene_guards.append(shape_guard and absolute_signed_growth <= rule["regress_scene_signed_bias"])
    summary = {"response_gain": delta.tolist(), "response_halves_gain": half.tolist(), "scene_gains": scene_gains,
               "per_material_response_regressions": {k: v.tolist() for k, v in regressed.items()},
               "per_scene_guard_passed": scene_guards, "cohorts": {}}
    for name in ("development", "held_out"):
        indices = np.array([i for i, row in enumerate(rows) if row["cohort"] == name])
        n = len(indices)
        positive = (half[:, indices] > 0).all(axis=0)
        median_halves = np.median(half[:, indices], axis=1)
        scene_median = float(np.median([scene_gains[i]["point"] for i in indices]))
        scene_consistent = sum(scene_gains[i]["lo"] > 0 for i in indices)
        band_margin = max(rule["shape_gain_absolute"], rule["shape_gain_relative"] * np.median(band[indices]))
        summary["cohorts"][name] = {
            "median_band_gain": float(np.median(delta[indices])), "median_band_gain_by_half": median_halves.tolist(),
            "band_margin": float(band_margin), "positive_in_both_halves_count": int(positive.sum()),
            "median_scene_gain": scene_median, "scene_ci_positive_count": int(scene_consistent),
            "passed": bool(median_halves.min() >= band_margin and positive.sum() > n // 2
                           and scene_median >= rule["scene_gain_absolute"] and scene_consistent > n // 2),
        }
    summary["passed"] = (all(row["passed"] for row in summary["cohorts"].values())
                         and all(not v.any() for v in regressed.values()) and all(scene_guards))
    return summary


def classify(response, scenes, rows):
    pairs = ("CA", "BA", "DA", "DB", "DC")
    reports = {pair: contrast(response, scenes, rows, *pair) for pair in pairs}
    for arm in "BCD":
        reports[f"{arm}:ER_DF"] = contrast(response, scenes, rows, arm, "ER_DF")
    interaction = {}
    float_scene = lambda pair: np.array([row["point"] for row in reports[pair]["scene_gains"]])
    effect_response = np.array(reports["DA"]["response_gain"]) - np.array(reports["BA"]["response_gain"]) - np.array(reports["CA"]["response_gain"])
    effect_scene = float_scene("DA") - float_scene("BA") - float_scene("CA")
    for name in ("development", "held_out"):
        mask = [row["cohort"] == name for row in rows]
        interaction[name] = {"median_response_gain": float(np.median(effect_response[mask])),
                             "median_scene_gain": float(np.median(effect_scene[mask]))}
    accepted = {arm: reports[f"{arm}A"]["passed"] and reports[f"{arm}:ER_DF"]["passed"] for arm in "BCD"}
    if accepted["C"] and not accepted["B"]:
        label = "DATA_EFFECT_SUPPORTED"
    elif accepted["B"] and not accepted["C"]:
        label = "OBJECTIVE_EFFECT_SUPPORTED"
    elif accepted["C"] and accepted["B"]:
        def dominates(left, right):
            return all(reports[left]["cohorts"][name][key] > reports[right]["cohorts"][name][key]
                       for name in interaction for key in ("median_band_gain", "median_scene_gain"))
        label = ("DATA_EFFECT_SUPPORTED" if dominates("CA", "BA") else
                 "OBJECTIVE_EFFECT_SUPPORTED" if dominates("BA", "CA") else "INCONCLUSIVE")
    elif accepted["D"]:
        label = "INTERACTION_SUPPORTED" if (
            reports["DB"]["passed"] and reports["DC"]["passed"] and
            all(v > 0 for row in interaction.values() for v in row.values())
        ) else "INCONCLUSIVE"
    else:
        label = "NO_SUPPORTED_IMPROVEMENT"
    return {"verdict": label, "contrasts": reports, "factorial_interaction": interaction, "accepted_vs_A_and_historical": accepted}
