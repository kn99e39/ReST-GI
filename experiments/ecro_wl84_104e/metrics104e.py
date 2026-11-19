"""Frozen WL104D ROI/effect statistics and the pre-registered E2 decomposition."""

import numpy as np

from trace104e import H
from stats104a import ratio_jk
from stats104b import jk_ci, ms_loo

EVENTS = ("e1", "e2", "h", "total")
ROIS = ("objects", "highlight_core", "highlight_ring", "grazing")


def event_totals(a):
    return {**a, "total": a["e1"] + a["e2"] + a["h"]}


def stable_rms(residual, denom):
    ms, loo = ms_loo(residual, residual)
    stable = lambda x: np.sqrt(np.maximum(0, x) / denom)
    values = stable(loo)
    return {**jk_ci(float(stable(ms)), values), "loo": values.tolist()}


def event_metrics(buffer, ref, masks, gates, rms_regions=None, compact=False):
    ref = event_totals(ref)
    result = {}
    readiness = []
    events = ("e2", "total") if compact else EVENTS
    difference = {key: val - ref[key] for key, val in event_totals(buffer).items()}
    for name, region in masks.items():
        if not region.any():
            continue
        ref_sum = (ref["total"][:, region] @ H.LUMA).sum(1)
        energy = {key: ratio_jk((val[:, region] @ H.LUMA).sum(1), ref_sum) for key, val in difference.items()}
        spatial = {}
        if rms_regions is None or name in rms_regions:
            denom = (ref["total"][:, region].mean(axis=0)**2).mean()
            spatial = {
                key: stable_rms(difference[key][:, region].reshape(len(ref_sum), -1), denom)
                for key in events
            }
        if name in ROIS:
            for event, ci in energy.items():
                width = 0.5 * (ci["hi"] - ci["lo"])
                if width > gates["precision_energy_half_width"]:
                    readiness.append({"region": name, "event": event, "metric": "energy", "half_width": width})
        if name == "objects":
            for event in ("e2", "total"):
                if event not in spatial:
                    continue
                ci = spatial[event]
                width = 0.5 * (ci["hi"] - ci["lo"])
                if width > gates["precision_rms_half_width"]:
                    readiness.append({"region": name, "event": event, "metric": "rms", "half_width": width})
        result[name] = {"energy": energy, "rms": spatial}
    return result, readiness


def terms(hybrids, mask):
    x = {key: hybrids[key]["e2"][:, mask].reshape(len(hybrids[key]["e2"]), -1)
         for key in ("AA", "AC", "CA", "CC")}
    d = x["CC"] - x["AA"]
    first = x["CA"] - x["AA"]
    second = x["AC"] - x["AA"]
    interaction = x["CC"] - x["CA"] - x["AC"] + x["AA"]
    err = np.max(np.abs(first + second + interaction - d))
    if err > 1e-12:
        raise ValueError(f"Interaction reconstruction error {err}")
    den, den_loo = ms_loo(d, d)
    answer = {"delta_mean_square": jk_ci(den, den_loo), "identity_absolute": float(err)}
    answer["shares"] = {}
    if den <= 0 or np.any(den_loo <= 0):
        return answer
    for name, effect in zip(("S1", "N2", "interaction"), (first, second, interaction)):
        m, m_loo = ms_loo(effect, d)
        answer["shares"][name] = jk_ci(m / den, m_loo / den_loo)
    return answer


def rms_gain(base, tested):
    return jk_ci(base["point"] - tested["point"], np.asarray(base["loo"]) - np.asarray(tested["loo"]))


def make_decision(cases, config):
    case = cases[config["primary_case"]]
    projections = case["factorization"]["objects"]["shares"]
    supported = [key for key, value in projections.items() if value["lo"] > 0.5]
    attribution = supported[0] if len(supported) == 1 else "AMBIGUOUS"
    companion = next(value for key, value in cases.items() if key != config["primary_case"])
    results = {}
    for intervention in ("LC", "CL", "LL"):
        records = {}
        for event in ("e2", "total"):
            c = case["arms"]["CC"]["objects"]["rms"][event]
            a = case["arms"]["A"]["objects"]["rms"][event]
            t = case["arms"][intervention]["objects"]["rms"][event]
            gain = rms_gain(c, t)
            limit = max(0.005, 0.5 * (c["point"] - a["point"]))
            other_gain = rms_gain(companion["arms"]["CC"]["objects"]["rms"][event],
                                  companion["arms"][intervention]["objects"]["rms"][event])
            records[event] = {"gain": gain, "limit": limit, "companion_gain": other_gain,
                              "passes": bool(gain["point"] >= limit and gain["lo"] > 0 and other_gain["point"] >= -0.005)}
        results[intervention] = {"events": records, "supports_recovery": all(item["passes"] for item in records.values())}
    return {"attribution": attribution, "oracle_interventions": results,
            "primary_explicit_a_to_c": case["baseline_delta"]}
