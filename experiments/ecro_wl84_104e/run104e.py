"""Check archived baselines, replay the intervention, and quantify precision and attribution."""

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

from trace104e import C, H, HERE, PT, R81, RT, Counterfactual
from metrics104e import event_metrics, make_decision, rms_gain, terms

CONFIG = HERE / "protocol.json"
P = json.loads(CONFIG.read_text(encoding="utf-8"))
PROV = C.REPO / "evaluation/104D/amendment1"
LAYOUT = ("A", "C", "ER_DF", "reference")
EVENTS = ("e1", "e2", "h")
HYBRIDS = tuple(P["operators"]["intervention_arms"])
ANGLE_EDGES = [0, 50, 55, 65, 70, 80, 90.001]
MIRROR_EDGES = [0, 3, 12, 180.001]


class StopExperiment(Exception):
    """An evidence gate stopped the frozen measurement."""


def require(valid, reason):
    if not valid:
        raise StopExperiment(reason)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, data):
    C.write_json(path, data)


def sha_ok(path, digest):
    require(C.sha(path) == digest, f"FROZEN_HASH_MISMATCH: {path}")


def l1_ratio(observed, expected):
    return float(np.abs(observed.astype(np.float64) - expected).sum() / max(np.abs(expected).sum(), 1e-20))


def load_case(parent, case, prov):
    folder = parent / "scenes" / case
    files = [folder / f"r{i:03d}.npz" for i in range(P["budget"]["replicates"])]
    for path in [*files, folder / "metrics.json", folder / "region_masks.npz"]:
        sha_ok(path, prov["output_sha256"][f"scenes/{case}/{path.name}"])
    metrics = read(folder / "metrics.json")
    b = P["budget"]
    require(metrics["precision_passed"] and metrics["spp_per_replicate"] == b["spp_each"]
            and metrics["seed_base"] == b["seeds"][case] and metrics["replicates"] == b["replicates"],
            f"FROZEN_CASE_CONTRACT: {case}")
    with np.load(folder / "region_masks.npz") as z:
        masks = dict(z)
    stacks = {arm: {ev: [] for ev in EVENTS} for arm in LAYOUT}
    for path in files:
        with np.load(path) as archive:
            for arm, fields in stacks.items():
                for event in EVENTS:
                    fields[event].append(archive[f"{arm}_{event}"])
    fields = {arm: {ev: np.stack(seq).astype(np.float64) for ev, seq in items.items()}
              for arm, items in stacks.items()}
    replay = {}
    for arm in ("A", "C"):
        ref = metrics["arms"][arm]
        trial, _ = event_metrics(fields[arm], fields["reference"], {"objects": masks["objects"]}, P["gates"])
        observed = trial["objects"]
        rms, signed = observed["rms"]["total"]["point"], observed["energy"]["e2"]["point"]
        difference = max(abs(rms - ref["stable_rms"]["objects"]["point"]), abs(signed - ref["signed"]["objects"]["e2"]["point"]))
        require(difference < P["gates"]["archived_metric_abs"], f"BASELINE_METRIC_NOT_REPRODUCED: {case}/{arm}")
        replay[arm] = {"total_rgb_rms": rms, "signed_e2": signed, "max_saved_discrepancy": difference}
    return fields, masks, replay


def bound_weights(diag, sampler):
    valid = diag["first_valid"]
    pdf = diag["first_pdf"][valid]
    require(pdf.size > 0 and np.isfinite(pdf).all() and np.all(pdf > 0), "INVALID_PARENT_PDF")
    for key in ("throughput", "e2_a", "e2_c"):
        v = diag[key]
        require(np.isfinite(v).all() and np.all(v >= 0), f"INVALID_PATH_VALUE: {key}")
    error = float(diag["sample_weight_error"].max() / diag["sample_eval_peak"].max())
    require(error <= P["gates"]["sample_weight_relative_peak"], f"NATIVE_EL1_WEIGHT_PDF: {sampler} {error}")
    return {"maximum_eval_weight_residual": error, "minimum_valid_pdf": float(pdf.min()), "valid_lanes": int(valid.sum())}


def native_context(context, sampler):
    mat = (
        {"type": "lay104b", "key": f"wl104e_{context}_L", "max_depth": 1024} if sampler == "L"
        else {"type": "cmp83", "key": f"wl104e_{context}_{sampler}"}
    )
    return RT.Ctx(H.mi.load_dict(C.scene_dict(H.mi, RT.T, mat, with_sensor=True)))


def tests(ctx, tag, name, seed, light):
    spp = P["budget"]["focused_identity_spp_each"]
    evidence = {}
    for sampler, terminal in (("A", "AA"), ("C", "CC"), ("L", "LL")):
        native = H.HA.render(native_context(tag, sampler), name, spp, seed, "cam", pass_seed=sampler == "L")
        custom = ctx.evaluate(seed, sampler, light, spp, spp)[terminal]
        gap = l1_ratio(custom, native["e2"])
        require(gap <= P["gates"]["identity_relative_l1"], f"OPERATOR_IDENTITY: {name}/{sampler} {gap}")
        evidence[sampler] = {"e2_native_relative_l1": gap}
        if sampler in ("A", "C"):
            diag = ctx.packet(seed, 0, spp, sampler, light, True)
            evidence[sampler]["weight_pdf"] = bound_weights(diag, sampler)
    a = ctx.evaluate(seed, "C", light, spp, spp)
    b = ctx.evaluate(seed, "C", light, spp, spp)
    repeat = max(l1_ratio(a[k], b[k]) for k in a)
    require(repeat <= P["gates"]["identity_relative_l1"], f"PAIRED_DETERMINISM: {name} {repeat}")
    evidence["repeat_relative_l1"] = repeat
    toy = {k: {"e2": np.full((4, 1, 1, 3), v, dtype=np.float64)}
           for k, v in (("AA", 0), ("CA", 3), ("AC", 2), ("CC", 6))}
    split = terms(toy, np.ones((1, 1), bool))
    require(np.allclose([x["point"] for x in split["shares"].values()], [0.5, 1/3, 1/6], rtol=0, atol=1e-12),
            "SYNTHETIC_STAGE_RECONSTRUCTION")
    evidence["synthetic_factorization_exact"] = True
    return evidence


def diagnostic_summary(data, ctx, masks, spp):
    visible = data["second_visible"]
    first_sid, second_sid = data["first_sid"], data["second_sid"]
    energy = data["e2_c"].astype(np.float64)
    delta = (data["e2_c"] - data["e2_a"]).astype(np.float64)
    out = {}
    selected = ("sphere", "cube")
    for roi in ("objects", *selected):
        region = np.tile(masks[roi].ravel(), spp)
        active = region & visible
        def histogram(field, mask, edges):
            return {
                "c_energy": (np.histogram(data[field][mask], edges, weights=energy[mask])[0] / spp).tolist(),
                "signed_n2_c_minus_a": (np.histogram(data[field][mask], edges, weights=delta[mask])[0] / spp).tolist(),
                "absolute_n2_change": (np.histogram(data[field][mask], edges, weights=np.abs(delta[mask]))[0] / spp).tolist(),
            }
        pair = {}
        for i, name in enumerate(ctx.names):
            for j, dest in enumerate(ctx.names):
                use = active & (first_sid == i) & (second_sid == j)
                if not use.any():
                    continue
                pair[f"{name}->{dest}"] = {"c_energy": float(energy[use].sum() / spp),
                                           "signed_n2_change": float(delta[use].sum() / spp),
                                           "active_pairs": int(use.sum())}
        er_ids = [ctx.sphere_index, ctx.cube_index]
        out[roi] = {"routes": pair}
        for label, sid in (("first", first_sid), ("second", second_sid)):
            accepted = active & np.isin(sid, er_ids)
            out[roi][f"{label}_theta"] = histogram(f"{label}_theta", accepted, ANGLE_EDGES)
            out[roi][f"{label}_mirror"] = histogram(f"{label}_mirror", accepted, MIRROR_EDGES)
    return out


def readout(ctx, name, light, fields, masks, out):
    b, g = P["budget"], P["gates"]
    spp, batch, base = b["spp_each"], b["launch_spp"], b["seeds"][name]
    all_hybrids = {arm: {ev: [] for ev in EVENTS} for arm in HYBRIDS}
    times = {tag: 0.0 for tag in "ACL"}
    records = []
    for r in range(b["replicates"]):
        packet, tail = {}, {}
        for sampler in "ACL":
            start = time.monotonic()
            part = ctx.evaluate(base + r, sampler, light, spp, batch)
            times[sampler] += time.monotonic() - start
            if "tail" in part:
                tail[sampler] = part.pop("tail")
            packet.update(part)
        old = {"AA": fields["A"]["e2"][r], "CC": fields["C"]["e2"][r], "LL": fields["reference"]["e2"][r]}
        identity = {arm: l1_ratio(packet[arm], expected) for arm, expected in old.items()}
        np.savez(out / f"r{r:03d}.npz", **packet, tail_a=tail["A"], tail_l=tail["L"])
        write(out / f"identity{r:03d}.json", identity)
        require(max(identity.values()) <= g["identity_relative_l1"], f"PRIMARY_IDENTITY: {name}/rep{r} {identity}")
        diag = {}
        for sampler in "AC":
            lanes = ctx.packet(base + r, 0, b["diagnostic_spp_each"], sampler, light, True)
            diag[sampler] = {"pdf": bound_weights(lanes, sampler),
                             "summary": diagnostic_summary(lanes, ctx, masks, b["diagnostic_spp_each"])}
        write(out / f"diagnostic{r:03d}.json", diag)
        records.append(identity)
        for arm in HYBRIDS:
            all_hybrids[arm]["e1"].append(fields["C"]["e1"][r])
            all_hybrids[arm]["e2"].append(packet[arm])
            all_hybrids[arm]["h"].append(tail.get(arm[0], fields["C"]["h"][r]))
        print(name, "rep", r, "max identity", max(identity.values()), flush=True)
    hybrids = {arm: {ev: np.stack(x).astype(np.float64) for ev, x in value.items()}
               for arm, value in all_hybrids.items()}
    arms, ref = {**hybrids, **{a: fields[a] for a in LAYOUT[:-1]}}, fields["reference"]
    results, fails = {}, []
    for arm, value in arms.items():
        metrics, violations = event_metrics(value, ref, masks, g, compact=True)
        results[arm] = metrics
        fails.extend({"arm": arm, **item} for item in violations)
    np.savez(out / "means.npz",
             **{f"{arm}_{ev}": val.mean(0).astype(np.float32)
                for arm, source in {**arms, "reference": ref}.items() for ev, val in source.items()})
    np.savez(out / "masks.npz", **masks)
    base_delta = {ev: rms_gain(results["C"]["objects"]["rms"][ev], results["A"]["objects"]["rms"][ev])
                  for ev in ("e2", "total")}
    result = {"precision_passed": not fails, "precision_failures": fails, "rms_convention": "ref TOTAL normalization",
              "arms": results, "factorization": {key: terms(hybrids, masks[key]) for key in ("objects", "cube", "sphere")},
              "baseline_delta": base_delta, "times_seconds_by_prefix": times, "native_identity": records}
    mid = len(ref["e2"]) // 2
    result["reference_half_noise_rms"] = {}
    for mask in ("objects", "cube", "sphere"):
        region = masks[mask]
        total = sum(ref[event][:, region] for event in EVENTS)
        denominator = np.sqrt((total.mean(0)**2).mean())
        result["reference_half_noise_rms"][mask] = {}
        for event, field in (("e2", ref["e2"][:, region]), ("total", total)):
            delta = 0.5 * (field[:mid].mean(0) - field[mid:].mean(0))
            result["reference_half_noise_rms"][mask][event] = float(np.sqrt((delta**2).mean()) / denominator)
    write(out / "metrics.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parent", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    prov = read(PROV / "provenance.json")
    b = P["budget"]
    cases = {}
    status = {"commit": args.commit, "phase": "frozen_input_verification", "smoke": args.smoke, "admission": "RUNNING"}
    start = time.monotonic()
    try:
        require(PT.D == 6 and C.CAM["origin"] == (0.0, -3.0, 1.2), "FROZEN_TRACER_CAMERA")
        sha_ok(PROV / "compiled_states.npz", prov["output_sha256"]["compiled_states.npz"])
        sha_ok(PROV / "materials.json", prov["output_sha256"]["materials.json"])
        sha_ok(C.W83 / "data/materials83.npz", prov["parent_source_sha256"]["data/materials83.npz"])
        for label, mesh in (("sphere", C.SPHERE_PLY), ("quad", C.QUAD_PLY)):
            sha_ok(mesh, C.P["scene"][f"{label}_ply_sha256"])
        rows = read(PROV / "materials.json")
        with np.load(PROV / "compiled_states.npz") as f:
            cert = {k: f[k] for k in "AC"}
        with np.load(C.W83 / "data/materials83.npz") as f:
            desc = [f[row["set"]][row["index"]] for row in rows]
        for name in P["cases"]:
            status["phase"] = name
            write(args.output / "status.json", status)
            folder = args.output / name
            folder.mkdir()
            i, row = next((i, row) for i, row in enumerate(rows) if row["image"] == name)
            fields, masks, baseline = load_case(args.parent, name, prov)
            write(folder / "baseline.json", baseline)
            context = Counterfactual(row["tag"], desc[i], cert["A"][i], cert["C"][i])
            light = RT.point_lights(row["light"])
            audit = tests(context, row["tag"], name, b["seeds"][name], light)
            write(folder / "focused_tests.json", audit)
            print(name, "ARCHIVE AND FOCUSED TESTS PASSED", flush=True)
            if args.smoke:
                continue
            cases[name] = readout(context, name, light, fields, masks, folder)
            write(args.output / "cases.json", cases)
            require(cases[name]["precision_passed"], f"SCENE_PRECISION_UNQUALIFIED: {name}")
            del fields, masks, context
        if not args.smoke:
            write(args.output / "decision.json", make_decision(cases, P))
        status["admission"] = "FOCUSED_TESTS_PASSED" if args.smoke else "QUALIFIED"
    except StopExperiment as exc:
        status.update({"admission": "STOPPED", "reason": str(exc)})
        print(status, flush=True)
    except Exception as exc:
        status.update({"admission": "EXECUTION_ERROR", "reason": str(exc)})
        raise
    finally:
        status["elapsed_seconds"] = time.monotonic() - start
        write(args.output / "status.json", status)
        executed = {Path(m.__file__).resolve() for m in tuple(sys.modules.values()) if getattr(m, "__file__", None)}
        hashes = {str(p.relative_to(C.REPO)): C.sha(p) for p in sorted(executed) if p.is_relative_to(C.EXP)}
        write(args.output / "provenance.json", {
            "commit": args.commit, "protocol_sha256": C.sha(CONFIG), "executed_code_sha256": hashes,
            "original_provenance_sha256": C.sha(PROV / "provenance.json"),
            "parent_used_output_sha256": {k: v for k, v in prov["output_sha256"].items()
                                            if any(k.startswith(f"scenes/{case}/") for case in P["cases"])
                                            or k in ("compiled_states.npz", "materials.json")},
            "parent_descriptor_sha256": prov["parent_source_sha256"]["data/materials83.npz"],
            "output_sha256": {str(p.relative_to(args.output)): C.sha(p) for p in sorted(args.output.rglob("*")) if p.is_file()},
            "runtime": {"system": platform.platform(), "python": platform.python_version(), "numpy": np.__version__,
                        "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "mitsuba": H.mi.__version__,
                        "drjit": H.dr.__version__, "variant": H.mi.variant(),
                        "driver": subprocess.check_output(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True).strip()},
        })
        print("WL104E", status, flush=True)
    return 0 if status["admission"] in ("QUALIFIED", "FOCUSED_TESTS_PASSED") else 2


if __name__ == "__main__":
    raise SystemExit(main())
