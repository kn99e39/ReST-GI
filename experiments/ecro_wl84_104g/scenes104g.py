"""WL104G native and fixed-proposal scenes, using WL104D/E/F transport."""

import argparse
import subprocess
import time
from pathlib import Path

import numpy as np

from fit104g import HERE, H, C, P, REPO, archived, read, sha_ok, write
from response104g import ARMS, CONTRASTS
from common104d import P as D
from qualify104g import dr, mi
from cross104f import Cross, l1_ratio, PARENT, PROV, load_case
from metrics104e import event_metrics, event_totals, rms_gain
from scenes104d import scene_masks
from stats104b import jk_ci, ms_loo

CFG = P["scene"]
EVENTS = ("e1", "e2", "h")
ROI_INTERACTION = {"sphere_lower_nz_max": 0., "floor_near": {"sphere_axis_m": .60, "cube_footprint_m": .25}}
IDENTITY_LIMIT = 2e-5


def native(tag, arm, state):
    key = f"wl104g_{tag}_{arm}"
    H.RR.REG[key] = ("ER", state)
    mat = {"type": "cmp83", "key": key}
    return H.RT.Ctx(mi.load_dict(C.scene_dict(mi, H.RT.T, mat, with_sensor=True)))


def render(ctx, image, spp, seed, reference=False):
    t = time.monotonic()
    events = H.HA.render(ctx, image, spp, seed, "cam", b=CFG["batch"], pass_seed=reference)
    if set(events) != set(EVENTS) or any(not np.isfinite(v).all() or (v < 0.).any() for v in events.values()):
        raise ValueError(f"Invalid native event buffers for {image}")
    return events, time.monotonic()-t


def stack(replicates):
    return {ev: np.stack([row[ev] for row in replicates]).astype(np.float64) for ev in replicates[0]}


def scene_metrics(values, ref, masks):
    return event_metrics(values, ref, masks, CFG)


def study_reference(row, mat, state, seed, directory):
    image, tag = row["image"], row["tag"]
    directory.mkdir(parents=True, exist_ok=False)
    old = C.CAM
    C.CAM = D["population"]["held_out_view"]
    try:
        H.R81.REG[f"lay_{tag}"] = mat
        ctx = {"reference": H.RT.Ctx(H.scene(tag, "LAY1024")), "R0": native(tag, "R0", state)}
        mask_events, mask_seconds = render(ctx["reference"], image, CFG["held_mask_spp"],
                                           seed + CFG["held_mask_offset"], reference=True)
        masks = scene_masks(ctx["reference"], event_totals(mask_events)["total"])
        masks.update({k: v for k, v in H.HA.masks(ctx["reference"], ROI_INTERACTION).items() if k not in masks})
        np.savez(directory / "region_masks.npz", **masks)
        np.savez(directory / "mask_reference.npz", **mask_events)
        totals = {"mask": mask_seconds, "reference": 0., "R0": 0.}
        fields = {"reference": [], "R0": []}
        for i in range(CFG["replicates"]):
            pack = {}
            for name, context in ctx.items():
                v, seconds = render(context, image, CFG["held_spp"], seed+i, name == "reference")
                fields[name].append(v)
                totals[name] += seconds
                pack.update({f"{name}_{ev}": evs.astype(np.float32) for ev, evs in v.items()})
            np.savez(directory / f"r{i:03d}.npz", **pack)
            print("reference preflight", image, i, flush=True)
        fields = {arm: stack(packets) for arm, packets in fields.items()}
        metrics, failed = scene_metrics(fields["R0"], fields["reference"], masks)
        ready = {"image": image, "precision_passed": not failed, "failures": failed, "R0": metrics,
                 "seconds": totals, "seed_base": seed, "spp": CFG["held_spp"], "camera": C.CAM}
        write(directory / "readiness.json", ready)
        return fields, masks, ready
    finally:
        C.CAM = old


def joint_gain(a, b):
    def norm(vals, name):
        return np.sqrt(sum(np.asarray(vals[k][name])**2 for k in ("e1", "e2")))
    return jk_ci(norm(a, "point") - norm(b, "point"), norm(a, "loo") - norm(b, "loo"))


def cancellation(fields, reference, masks):
    result = {}
    for arm, buffer in fields.items():
        data = {}
        for roi in ("objects", "cube", "sphere", "floor_near"):
            mask = masks[roi]
            ref = event_totals(reference)["total"][:, mask]
            den = (ref.mean(axis=0)**2).mean()
            if den <= 0:
                continue
            terms = {e: (buffer[e][:, mask] - reference[e][:, mask]).reshape(len(ref), -1) for e in EVENTS}
            cov = {}
            for i, x in enumerate(EVENTS):
                for y in EVENTS[i+1:]:
                    point, loo = ms_loo(terms[x], terms[y])
                    cov[f"{x}_{y}"] = jk_ci(2*point/den, 2*loo/den)
            joint, _ = ms_loo(sum(terms.values()), sum(terms.values()))
            marginal = sum(ms_loo(terms[e], terms[e])[0] for e in EVENTS)
            gap = abs(joint/den - marginal/den - sum(v["point"] for v in cov.values()))
            if gap > 1e-12:
                raise ValueError(f"Paired event mean-square identity failed for {arm}/{roi}: {gap}")
            data[roi] = {"twice_event_covariance_normalized": cov, "closure_absolute": float(gap)}
        result[arm] = data
    return result


def regressions(a, b):
    rms = {key: b["objects"]["rms"][key]["point"] - a["objects"]["rms"][key]["point"]
           for key in ("e1", "e2", "total")}
    bias = {region: {e: abs(v["point"]) - abs(a[region]["energy"][e]["point"]) for e, v in row["energy"].items()}
            for region, row in b.items() if region in a}
    passed = max(rms.values()) <= CFG["regression_rms"] and all(
        r <= CFG["regression_signed"] for row in bias.values() for r in row.values())
    return {"objects_rms_increase": rms, "absolute_signed_error_increase": bias, "passed": bool(passed)}


def compare(metrics, fixed):
    result = {}
    for tested, base in CONTRASTS:
        a, b = metrics[base], metrics[tested]
        per_roi = {
            roi: {event: rms_gain(row, b[roi]["rms"][event]) for event, row in a[roi]["rms"].items()}
            for roi in a if roi in b
        }
        fx = {roi: {event: rms_gain(row, fixed[tested][roi]["rms"][event])
                    for event, row in fixed[base][roi]["rms"].items() if event in ("e2", "total")}
              for roi in fixed[base] if roi in fixed[tested]}
        result[f"{tested}-{base}"] = {"native_gain": per_roi, "fixed_gain": fx,
                                       "joint_e1_e2_gain": joint_gain(a["objects"]["rms"], b["objects"]["rms"]),
                                       "regressions": regressions(a, b)}
    return result


def measure(row, mat, states, seed, out, prior=None):
    image, tag = row["image"], row["tag"]
    deployed = {arm: native(tag, arm, state) for arm, state in states.items()}
    config = CFG["dev_spp"] if prior is None else CFG["held_spp"]
    if prior is None:
        archive, masks, _ = load_case(PARENT / "parent", image, read(PROV / "provenance.json"))
        base, ref = archive["ER_DF"], archive["reference"]
        masks.update({k: v for k, v in H.HA.masks(deployed["R0"], ROI_INTERACTION).items() if k not in masks})
        replay, r0_seconds = render(deployed["R0"], image, config, seed)
        replay_gap = max(l1_ratio(replay[e], base[e][0]) for e in EVENTS)
        if replay_gap > IDENTITY_LIMIT:
            raise ValueError(f"Historical R0 replay changed in {image}: {replay_gap}")
        del archive
    else:
        inputs, masks, ready = prior
        base, ref = inputs["R0"], inputs["reference"]
        r0_seconds = ready["seconds"]["R0"]
        replay_gap = None
    precision_r0, failures_r0 = scene_metrics(base, ref, masks)
    if failures_r0:
        raise ValueError(f"Frozen baseline/reference scene no longer qualifies: {image} {failures_r0}")
    dest = out / image
    dest.mkdir(parents=True, exist_ok=False)
    np.savez(dest / "region_masks.npz", **masks)
    crosses = {arm: Cross(tag + "_wl104g_" + arm, mat, states[arm], states["R0"]) for arm in ("R1", "R2")}
    light = H.RT.point_lights(row["light"])
    focused = {}
    for arm, ctx in crosses.items():
        one = ctx.evaluate_cross(seed, "C", light, 4, 4)
        two = ctx.evaluate_cross(seed, "C", light, 4, 4)
        within = max(l1_ratio(one[k], two[k]) for k in one)
        if within > IDENTITY_LIMIT:
            raise ValueError(f"Cross replay not deterministic for {image}/{arm}")
        focused[arm] = within
    native_fields = {"R1": [], "R2": []}
    fixed_fields = {arm: [] for arm in ARMS}
    times = {"R0": r0_seconds, "R1": 0., "R2": 0., "fixed_R1": 0., "fixed_R2": 0.}
    max_cross_gap: float = 0.
    for i in range(CFG["replicates"]):
        packet = {f"{arm}_{ev}": e[i].astype(np.float32) for arm, row2 in (("reference", ref), ("R0", base))
                  for ev, e in row2.items()}
        for arm, ctx in crosses.items():
            value, seconds = render(deployed[arm], image, config, seed+i)
            times[arm] += seconds
            native_fields[arm].append(value)
            t = time.monotonic()
            cross = ctx.evaluate_cross(seed+i, "C", light, config, CFG["batch"])
            times[f"fixed_{arm}"] += time.monotonic()-t
            gap = l1_ratio(cross["CC"], base["e2"][i])
            max_cross_gap = max(max_cross_gap, gap)
            if gap > IDENTITY_LIMIT:
                raise ValueError(f"Fixed R0 response failed native E2 identity: {image}/{arm}/{i} {gap}")
            fixed_fields[arm].append({"e1": value["e1"], "e2": cross["AC"], "h": base["h"][i]})
            packet.update({f"{arm}_{ev}": a.astype(np.float32) for ev, a in value.items()})
            packet[f"fixed_{arm}_e2"] = cross["AC"].astype(np.float32)
        fixed_fields["R0"].append({ev: base[ev][i] for ev in EVENTS})
        np.savez(dest / f"r{i:03d}.npz", **packet)
        print("scene", image, i, flush=True)
    nat = {"R0": base, **{arm: stack(v) for arm, v in native_fields.items()}}
    fixed = {arm: stack(v) for arm, v in fixed_fields.items()}
    native_metrics, fixed_metrics, failed = {"R0": precision_r0}, {"R0": precision_r0}, []
    for arm in ("R1", "R2"):
        for field, output, kind in ((nat, native_metrics, "native"), (fixed, fixed_metrics, "fixed")):
            output[arm], fail = scene_metrics(field[arm], ref, masks)
            failed.extend({"arm": arm, "kind": kind, **r} for r in fail)
    means = {f"{arm}_{event}": values.mean(axis=0).astype(np.float32)
             for arm, fields in {**nat, "reference": ref, **{f"fixed_{a}": fixed[a] for a in ("R1", "R2")}}.items()
             for event, values in event_totals(fields).items()}
    np.savez(dest / "means.npz", **means)
    evidence = {"scene": image, "spp": config, "seed_base": seed, "replicates": CFG["replicates"],
                "camera": C.CAM, "roi_counts": {k: int(v.sum()) for k, v in masks.items()},
                "seconds": times, "R0_native_replay_gap": replay_gap, "R0_render_time_scope":
                "one paired archival replay" if prior is None else "all paired held-out R0 renders",
                "focused_fixed_replay_gap": focused, "max_R0_cross_native_E2_relative_gap": max_cross_gap,
                "precision_passed": not failed, "precision_failures": failed, "native": native_metrics, "fixed": fixed_metrics,
                "contrasts": compare(native_metrics, fixed_metrics), "cancellation": cancellation(nat, ref, masks)}
    write(dest / "metrics.json", evidence)
    return evidence


def judge(cases, held, response, fit):
    valid = bool(held) and all(v["precision_passed"] for v in cases.values())
    primary = CFG["primary"]
    contrasts = {}
    for candidate, baseline in CONTRASTS:
        key = f"{candidate}-{baseline}"
        quality = {scene: evidence["contrasts"][key] for scene, evidence in cases.items()}
        scene_pass = {}
        for scene, margin in ((primary, CFG["primary_gain"]), (held, CFG["held_out_gain"])):
            if scene:
                gain = quality[scene]["native_gain"]["objects"]["total"]
                joint = quality[scene]["joint_e1_e2_gain"]
                scene_pass[scene] = bool(gain["point"] >= margin and gain["lo"] > 0 and joint["lo"] > 0)
        contrast = {"angular": response["contrasts"][key]["passed"], "scene_pass": scene_pass,
                    "all_scene_regressions_passed": all(x["regressions"]["passed"] for x in quality.values()),
                    "opt_converged": all(fit["arms"][arm]["converged"] for arm in (candidate, baseline) if arm != "R0")}
        contrast["passed"] = bool(valid and contrast["angular"] and all(scene_pass.values()) and
                                  contrast["all_scene_regressions_passed"])
        contrasts[key] = contrast
    recovered = any(contrasts[key]["passed"] for key in ("R2-R0", "R1-R0"))
    label = ("EVIDENCE_UNQUALIFIED" if not valid else "FIDELITY_IMPROVED_WITHIN_CERTIFIED_ER" if recovered else
             "NO_DEMONSTRATED_FIDELITY_IMPROVEMENT")
    return {"outcome": label, "evidence_qualified": bool(valid), "contrasts": contrasts, "held_out": held}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fit", type=Path)
    parser.add_argument("response", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    fit = read(args.fit / "fit.json")
    response = read(args.response / "response.json")
    if not response.get("reference_precision_passed") or not response.get("physical_passed"):
        raise ValueError("Valid physical and reference response qualification is required")
    sha_ok(HERE / "protocol.json", response["protocol_sha256"])
    sha_ok(args.fit / "fit_states.npz", response["fit_states_sha256"])
    sha_ok(args.response / "compiled_states.npz", response["compiled_sha256"])
    rows, descriptors, _, _, _, _ = archived()
    indices = {r["image"]: i for i, r in enumerate(rows)}
    with np.load(args.response / "compiled_states.npz") as z:
        states = {a: z[a] for a in ARMS}
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    readiness = {}
    held = None
    for name in ("held_out", "held_out_fallback"):
        image, seed = next(iter(CFG[name].items()))
        i = indices[image]
        prior = study_reference(rows[i], descriptors[i], states["R0"][i], seed, args.output / "references" / image)
        readiness[image] = prior[2]["precision_passed"]
        if readiness[image]:
            held = image
            break
    if held is None:
        write(args.output / "decision.json", {"outcome": "EVIDENCE_UNQUALIFIED", "reference_readiness": readiness})
        return
    cases = {}
    for image, seed in {**CFG["development"], held: seed}.items():
        i = indices[image]
        C.CAM = D["population"]["held_out_view"] if image == held else C.CAM
        cases[image] = measure(rows[i], descriptors[i], {a: z[i] for a, z in states.items()}, seed,
                               args.output, prior if image == held else None)
    result = judge(cases, held, response, fit)
    result.update({"commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                   "response_sha256": C.sha(args.response / "response.json"), "seconds": time.monotonic()-started,
                   "reference_readiness": readiness,
                   "allocator": [line.strip() for line in dr.whos_str().splitlines() if "used (peak:" in line]})
    write(args.output / "decision.json", result)
    print(result["outcome"], flush=True)


if __name__ == "__main__":
    main()
