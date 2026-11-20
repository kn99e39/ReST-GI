"""Independent angular-response, physical and runtime measurements for WL104G."""

import argparse
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from fit104g import ANGULAR, H, HERE, P, PARENT, archived, compile_states, read, sha_ok, write, C
from qualify104g import mi, dr, vector
from common104d import P as D
from data104d import radius_deg
from response104d import metric_arrays, terms

ARMS = tuple(P["fit"]["arms"])
ANGLES = np.asarray(D["response"]["angles_deg"])
POLAR = np.repeat(ANGLES, len(D["response"]["azimuths_deg"]))
GRAZE = POLAR >= 80
BAND = (POLAR >= 55) & (POLAR <= 65)
CONTRASTS = (("R2", "R0"), ("R2", "R1"), ("R1", "R0"))


def runtime(state):
    n, repeats = 262144, 7
    rng = np.random.default_rng(10470601)
    u = [mi.Float(v) for v in rng.random((7, n), dtype=np.float32)]
    wi = mi.warp.square_to_cosine_hemisphere(mi.Point2f(u[0], u[1]))
    wo = mi.warp.square_to_cosine_hemisphere(mi.Point2f(u[2], u[3]))
    uv = mi.Point2f(u[5], u[6])
    dr.eval(wi, wo, *u)
    closure = H.CD.DrClosure83("ER", state)

    def samples():
        return closure.sample(wi, u[4], uv)

    arm = {"lanes": n, "repeats": repeats, "seconds_median_including_launch": {}}
    for op, callback in (("eval", lambda: (closure.eval(wi, wo),)),
                         ("pdf", lambda: (closure.pdf(wi, wo),)), ("sample", samples)):
        timings = []
        for i in range(repeats + 2):
            t = time.perf_counter()
            dr.eval(*callback())
            dr.sync_thread()
            if i >= 2:
                timings.append(time.perf_counter() - t)
        arm["seconds_median_including_launch"][op] = float(np.median(timings))
    wo, pdf, w, valid = samples()
    p = np.asarray(pdf, dtype=np.float64)
    use = np.asarray(valid, dtype=bool)
    evaluated, p_eval = closure.eval_pdf(wi, wo, valid)
    f = vector(evaluated)
    weight = vector(w)
    r = np.max(np.abs(p[:, None] * weight - f)) / f.max()
    rel = np.abs(p[use] - np.asarray(p_eval, dtype=np.float64)[use]) / p[use]
    parity = {"median_pdf_gap": float(np.median(rel)), "p999_pdf_gap": float(np.quantile(rel, .999)),
              "maximum_pdf_gap": float(rel.max()), "weight_peak_gap": float(r)}
    parity["passed"] = bool(np.isfinite(weight).all() and np.all(weight >= 0) and np.all(p[use] > 0)
                            and np.all(p[~use] == 0) and parity["median_pdf_gap"] <= 1e-5
                            and parity["p999_pdf_gap"] <= 1e-3 and r <= 2e-5)
    return {**arm, "sample_identity": parity}


def reference_precision(data, rows, w):
    ref = D["response"]
    noise = metric_arrays(terms(data["a"], data["b"], w), len(rows))
    values = [{"band_shape": float(sh[BAND].mean()*.5), "grazing_shape": float(sh[GRAZE].mean()*.5),
               "albedo": float(a.mean()*.5)} for sh, a in zip(noise["eshape"], noise["ealb"])]
    passed = all(v["band_shape"] <= ref["max_band_shape_noise"] and
                 v["grazing_shape"] <= ref["max_grazing_shape_noise"] and
                 v["albedo"] <= ref["max_albedo_noise"] for v in values)
    return values, passed


def physical(raw, compiled, wi, wo, rendered):
    config = D["physical"]
    device = torch.as_tensor(raw[None], device=H.dev, dtype=torch.float32)
    params = H.CL.decode("ER", "cert", device)
    predicted = H.er_eval_torch(params, wi, wo)
    parity = float(np.max(np.abs(rendered - predicted)) / np.max(np.abs(predicted)))
    good = (wi[:, 2] > .05) & (wo[:, 2] > .05)
    p64 = {k: v.double() for k, v in params.items()}
    a = H.er_eval_torch(p64, wi[good], wo[good]) / wo[good, 2, None]
    b = H.er_eval_torch(p64, wo[good], wi[good]) / wi[good, 2, None]
    rec = float(np.max(np.abs(a-b)) / np.max(np.abs(a)))
    mixing = compiled[:24].reshape(3, 8).astype(np.float64)
    bound = mixing.sum(axis=1)
    probs, attenuation, eta = compiled[45:53], compiled[53:68], compiled[68]
    passed = (np.isfinite(compiled).all() and np.isfinite(rendered).all() and np.all(rendered >= 0.)
              and np.all(mixing >= 0.) and max(bound) <= 1 + config["passivity_tolerance"]
              and 1 <= eta <= 2.5 and np.all(attenuation >= 0.) and np.all(attenuation <= 1.)
              and probs[0] > 0 and probs.min() >= 0 and abs(probs.sum()-1) < 1e-6
              and parity <= config["torch_dr_peak_relative"] and rec <= config["reciprocity_peak_relative"])
    return {"passed": bool(passed), "torch_runtime_peak_relative": parity, "reciprocity_relative": rec,
            "passivity_upper_bound_rgb": bound.tolist(), "lambert_proposal_probability": float(probs[0])}


def analyze(data, rows, arm, raw, compiled, w):
    a, b = data["a"], data["b"]
    ref = .5 * (a + b)
    wi, wo = data["wi"], data["wo"]
    pred = np.empty_like(ref)
    contracts = []
    for j, state in enumerate(compiled):
        sl = slice(j * len(POLAR), (j + 1) * len(POLAR))
        x, y = wi[sl].reshape(-1, 3), wo[sl].reshape(-1, 3)
        f = H.er_eval_dr(state, x, y)
        contracts.append(physical(raw[j], state, x, y, f))
        pred[sl] = f.reshape(pred[sl].shape)
    nm = len(rows)
    quality = metric_arrays(terms(pred, ref, w), nm)
    halves = [metric_arrays(terms(pred, h, w), nm)["el1"].mean(axis=1) for h in (a, b)]
    p_lum = (pred @ H.LUMA) * w[..., 0]
    r_lum = (ref @ H.LUMA) * w[..., 0]
    radius = radius_deg(wi, wo)
    regions = {}
    for name, mask in (("core", radius < 3), ("shoulder", (radius >= 3) & (radius < 12)), ("tail", radius >= 12)):
        ps, rs = (p_lum * mask).sum(axis=1), (r_lum * mask).sum(axis=1)
        regions[name] = {"signed_relative": ((ps - rs) / rs.clip(1e-20)).reshape(nm, -1).tolist(),
                         "normalized_mass_difference": (ps/p_lum.sum(axis=1).clip(1e-20) -
                                                        rs/r_lum.sum(axis=1).clip(1e-20)).reshape(nm, -1).tolist()}
    if arm == "R0":
        historical = read(PARENT / "amendment1/response.json")["arms"]["ER_DF"]["probes"]
        gap = max(float(np.max(np.abs(quality[k] - historical[k]))) for k in ("el1", "ealb", "eshape"))
        if gap > 1e-5:
            raise ValueError(f"Archived angular R0 did not replay ({gap})")
    else:
        gap = None
    return {"angular": {k: v.tolist() for k, v in quality.items()},
            "regions": regions, "el1_each_half": np.stack(halves).tolist(),
            "mean_el1": quality["el1"].mean(axis=1).tolist(),
            "mean_ealb": quality["ealb"].mean(axis=1).tolist(),
            "grazing_shape": quality["eshape"][:, GRAZE].mean(axis=1).tolist(),
            "polar_means_eshape": quality["eshape"].reshape(nm, len(ANGLES), -1).mean(axis=-1).tolist(),
            "physical": contracts, "historical_R0_max_metric_gap": gap}


def compare(arms, rows):
    comparisons = {}
    for candidate, baseline in CONTRASTS:
        u, v = arms[candidate], arms[baseline]
        gains = np.asarray(v["el1_each_half"]) - np.asarray(u["el1_each_half"])
        cohorts = {}
        for cohort in ("development", "held_out"):
            mask = np.array([row["cohort"] == cohort for row in rows])
            mean = gains[:, mask].mean(axis=1)
            counts = np.all(gains[:, mask] > 0, axis=0).sum()
            cohorts[cohort] = {"mean_el1_gain_each_half": mean.tolist(), "positive_both": int(counts),
                               "n": int(mask.sum()), "passed": bool(np.all(mean >= .005) and counts > mask.sum()/2)}
        reg = {k: (np.asarray(u[k]) - np.asarray(v[k])).tolist()
               for k in ("mean_el1", "mean_ealb", "grazing_shape")}
        no_regression = all(max(reg[k]) <= v for k, v in (("mean_el1", .01), ("mean_ealb", .008), ("grazing_shape", .02)))
        comparisons[f"{candidate}-{baseline}"] = {"cohorts": cohorts, "regressions": reg,
                                                   "passed": bool(no_regression and all(e["passed"] for e in cohorts.values()))}
    return comparisons


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fit", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    meta = read(args.fit / "fit.json")
    sha_ok(HERE / "protocol.json", meta["protocol_sha256"])
    sha_ok(args.fit / "fit_states.npz", meta["fit_states_sha256"])
    rows, _, historical, _, _, _ = archived()
    with np.load(ANGULAR) as z:
        data = {k: z[k] for k in ("a", "b", "wi", "wo", "use", "q")}
    w = (data["use"] / data["q"])[..., None]
    args.output.mkdir(parents=True, exist_ok=False)
    precision, ready = reference_precision(data, rows, w)
    check = {"precision": precision, "reference_precision_passed": ready}
    if not ready:
        write(args.output / "response.json", check)
        return
    started = time.monotonic()
    with np.load(args.fit / "fit_states.npz") as z:
        raw = {arm: z[arm] for arm in ARMS}
    compiled = {"R0": historical["ER_DF"], **{arm: compile_states(raw[arm]) for arm in ("R1", "R2")}}
    results = {}
    for arm in ARMS:
        results[arm] = analyze(data, rows, arm, raw[arm], compiled[arm], w)
        results[arm]["runtime_primary_material"] = runtime(compiled[arm][0])
        print("response", arm, np.round(results[arm]["mean_el1"], 4), flush=True)
    states_valid = all(z.shape == (len(rows), 69) for z in compiled.values()) and H.CL.n_params("ER") == 64
    passed = states_valid and all(x["passed"] for r in results.values() for x in r["physical"])
    passed &= all(r["runtime_primary_material"]["sample_identity"]["passed"] for r in results.values())
    np.savez(args.output / "compiled_states.npz", **compiled)
    write(args.output / "response.json", {
        **check, "physical_passed": bool(passed), "arms": results, "contrasts": compare(results, rows),
        "materials": rows, "theta_deg": POLAR.tolist(), "azimuth_deg": D["response"]["azimuths_deg"],
        "fit_states_sha256": meta["fit_states_sha256"], "reference_sha256": C.sha(ANGULAR),
        "compiled_sha256": C.sha(args.output / "compiled_states.npz"),
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "protocol_sha256": C.sha(HERE / "protocol.json"), "seconds": time.monotonic()-started})


if __name__ == "__main__":
    main()
