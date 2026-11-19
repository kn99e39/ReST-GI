"""Independent response metrics and physical closure contracts for WL104D."""

import time

import numpy as np
import torch

from common104d import CD, CL, H, P, STATES, write
from data104d import radius_deg


def terms(predict, target, weights):
    # Same metric normalization as the WL82/WL83 evaluator.
    a = (predict * weights).sum(axis=1)
    b = (target * weights).sum(axis=1)
    total = b.sum(axis=1).clip(1e-20)
    error = np.abs(predict - target) * weights
    norm_p = predict / a[:, None].clip(1e-20)
    norm_t = target / b[:, None].clip(1e-20)
    return {
        "el1": error.sum(axis=(1, 2)) / total,
        "ealb": np.abs(a - b).sum(axis=1) / total,
        "eshape": (b * (weights * np.abs(norm_p - norm_t)).sum(axis=1)).sum(axis=1) / total,
        "albedo_ratio": (a @ H.LUMA) / (b @ H.LUMA).clip(1e-20),
    }


def metric_arrays(arrays, nm):
    return {key: np.asarray(value).reshape(nm, -1) for key, value in arrays.items()}


def record_response(data, states, out):
    p = P["response"]
    n = len(states["A"])
    angles = np.repeat(p["angles_deg"], len(p["azimuths_deg"]))
    band = (angles >= 55) & (angles <= 65)
    graze = angles >= 80
    wi, wo = data["wi"], data["wo"]
    weights = (data["use"] / data["q"])[..., None]
    a, b = data["a"], data["b"]
    ref = 0.5 * (a + b)
    noise = metric_arrays(terms(a, b, weights), n)
    precision = [{
        "band_shape": float(row[band].mean() * 0.5),
        "grazing_shape": float(row[graze].mean() * 0.5),
        "albedo": float(noise["ealb"][i].mean() * 0.5),
    } for i, row in enumerate(noise["eshape"])]
    passed = all(row["band_shape"] <= p["max_band_shape_noise"] and
                 row["grazing_shape"] <= p["max_grazing_shape_noise"] and row["albedo"] <= p["max_albedo_noise"]
                 for row in precision)
    result = {"angles_deg": p["angles_deg"], "azimuths_deg": p["azimuths_deg"], "precision": precision,
              "reference_precision_passed": passed, "physics_passed": False, "arms": {}}
    if not passed:
        write(out / "response.json", result)
        return result, {}

    rad = radius_deg(wi, wo)
    regions = {"core": rad < 3, "shoulder": (rad >= 3) & (rad < 12), "tail": rad >= 12}
    pk_a = np.max(data["peak_a"] @ H.LUMA, axis=1)
    pk_b = np.max(data["peak_b"] @ H.LUMA, axis=1)
    pk_ref = np.max(0.5 * ((data["peak_a"] + data["peak_b"]) @ H.LUMA), axis=1)
    peak_rad = radius_deg(data["peak_wi"], data["peak_wo"])
    out_states = {}
    for arm in STATES:
        pred = np.empty_like(ref)
        peak = np.empty(data["peak_a"].shape[:-1])
        tensor = torch.as_tensor(states[arm], device=H.dev, dtype=torch.float32)
        decoded = CL.decode("ER", "cert", tensor)
        compiled = CD.compiled_state("ER", decoded)
        physical = []
        nprobe = len(angles)
        for j in range(n):
            group = slice(j * nprobe, (j + 1) * nprobe)
            incident = wi[group].reshape(-1, 3)
            outgoing = wo[group].reshape(-1, 3)
            params = {key: value[j:j+1] for key, value in decoded.items()}
            predict = H.er_eval_torch(params, incident, outgoing)
            start = time.monotonic()
            render_eval = H.er_eval_dr(compiled[j], incident, outgoing)
            evaluate_seconds = time.monotonic() - start
            runtime_error = np.max(np.abs(render_eval - predict)) / max(np.max(np.abs(predict)), 1e-20)
            eligible = (incident[:, 2] > 0.05) & (outgoing[:, 2] > 0.05)
            precise = {key: value.double() for key, value in params.items()}
            forward = H.er_eval_torch(precise, incident[eligible], outgoing[eligible]) / outgoing[eligible, 2, None]
            reverse = H.er_eval_torch(precise, outgoing[eligible], incident[eligible]) / incident[eligible, 2, None]
            rec = np.max(np.abs(forward - reverse)) / np.max(np.abs(forward))
            mixture = compiled[j, :24].reshape(3, 8).astype(np.float64)
            bound = mixture.sum(axis=1)
            eta, knots = compiled[j, 68], compiled[j, 53:68]
            setting = P["physical"]
            passed = (np.isfinite(compiled[j]).all() and (mixture >= 0).all() and (bound <= 1 + setting["passivity_tolerance"]).all()
                      and 1 <= eta <= 2.5 and (knots >= 0).all() and (knots <= 1).all()
                      and runtime_error <= setting["torch_dr_peak_relative"] and rec <= setting["reciprocity_peak_relative"])
            physical.append({"passed": bool(passed), "reciprocity_relative": float(rec), "albedo_upper_bound": bound.tolist(),
                             "runtime_relative_error": float(runtime_error), "eval_lanes": len(incident),
                             "eval_wall_seconds_including_transfer": evaluate_seconds})
            pred[group] = render_eval.reshape(pred[group].shape)
            peak[group] = H.er_eval_dr(compiled[j], data["peak_wi"][group].reshape(-1, 3),
                                      data["peak_wo"][group].reshape(-1, 3)).reshape(-1, peak.shape[1], 3) @ H.LUMA
        out_states[arm] = compiled
        raw = metric_arrays(terms(pred, ref, weights), n)
        split = [metric_arrays(terms(pred, h, weights), n)["eshape"][:, band].mean(axis=1).tolist() for h in (a, b)]
        p_lum, r_lum = (pred @ H.LUMA) * weights[..., 0], (ref @ H.LUMA) * weights[..., 0]
        p_total, r_total = p_lum.sum(axis=1), r_lum.sum(axis=1)
        bands = {}
        for key, region in regions.items():
            pe, re = (p_lum * region).sum(axis=1), (r_lum * region).sum(axis=1)
            re_halves = [(h @ H.LUMA * weights[..., 0] * region).sum(axis=1) for h in (a, b)]
            bands[key] = {"signed_relative_energy": ((pe - re) / re.clip(1e-20)).reshape(n, -1).tolist(),
                          "shape_mass_difference": (pe / p_total.clip(1e-20) - re / r_total.clip(1e-20)).reshape(n, -1).tolist(),
                          "reference_half_relative": ((0.5 * np.abs(re_halves[0] - re_halves[1])) / re.clip(1e-20)).reshape(n, -1).tolist()}
        row = {
            "probes": {key: value.tolist() for key, value in raw.items()},
            "band_shape": raw["eshape"][:, band].mean(axis=1).tolist(),
            "band_shape_reference_halves": split,
            "grazing_shape": raw["eshape"][:, graze].mean(axis=1).tolist(),
            "overall_albedo": raw["ealb"].mean(axis=1).tolist(),
            "overall_el1": raw["el1"].mean(axis=1).tolist(),
            "peak_signed_relative": (peak.max(axis=1) / pk_ref.clip(1e-20) - 1).reshape(n, -1).tolist(),
            "peak_ref_half_relative": (0.5 * np.abs(pk_a - pk_b) / pk_ref.clip(1e-20)).reshape(n, -1).tolist(),
            "peak_offset_deg": np.take_along_axis(peak_rad, peak.argmax(axis=1)[:, None], axis=1).reshape(n, -1).tolist(),
            "reference_peak_offset_deg": np.take_along_axis(peak_rad, (0.5*((data["peak_a"]+data["peak_b"]) @ H.LUMA)).argmax(axis=1)[:, None], axis=1).reshape(n, -1).tolist(),
            "regions": bands, "physical": physical
        }
        result["arms"][arm] = row
        print("response", arm, "band eshape", np.round(row["band_shape"], 5).tolist(), flush=True)
    result["physics_passed"] = all(row["passed"] for arm in result["arms"].values() for row in arm["physical"])
    result["parameters_per_material"] = CL.n_params("ER")
    result["state_floats"] = compiled.shape[1]
    result["physics_passed"] &= (result["parameters_per_material"] == P["physical"]["parameter_count"] and
                                 result["state_floats"] == P["physical"]["state_floats"])
    np.savez(out / "compiled_states.npz", **out_states)
    write(out / "response.json", result)
    return result, out_states
