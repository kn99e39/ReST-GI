"""One frozen reference increase followed by conditional WL104D evaluation."""

import argparse
import json
import platform
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common104d import ARMS, DA, H, P, REPO, S, historical_states, write
from decision104d import classify
from response104d import metric_arrays, record_response, terms
from scenes104d import render_case

CONFIG = HERE / "protocol.json"
A1 = json.loads(CONFIG.read_text(encoding="utf-8"))
SPEC = A1["sampling"]


def require(condition, detail):
    if not condition:
        raise ValueError(detail)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def verify_hash(path, expected):
    found = H.C.sha(path)
    require(found == expected, f"SHA-256 mismatch for {path}: {found} != {expected}")
    return found


def verify_parent(primary, source):
    frozen = primary / "provenance.json"
    verify_hash(frozen, A1["parent_provenance_sha256"])
    provenance = read(frozen)
    require(provenance["commit"] == A1["parent_commit"], "Wrong parent execution commit")
    require(provenance["protocol"] == P, "Parent evaluation protocol values differ")
    verify_hash(HERE.parent / "protocol.json", A1["parent_protocol_sha256"])
    for path, expected in provenance["code_and_configuration_sha256"].items():
        verify_hash(REPO / path, expected)
    for path, expected in provenance["output_sha256"].items():
        verify_hash(primary / path, expected)
    for path, expected in provenance["source_sha256"].items():
        verify_hash(source / path, expected)
    for path, expected in provenance["scene_inputs_sha256"].items():
        verify_hash(REPO / path, expected)
    require(provenance["output_sha256"]["independent_response.npz"] == A1["parent_reference_sha256"],
            "Parent reference identity differs")
    require(provenance["output_sha256"]["fit_states.npz"] == A1["parent_fit_states_sha256"],
            "Parent fit archive identity differs")
    status = read(primary / "status.json")
    require(status["admission"] == "REFERENCE_UNQUALIFIED" and status["verdict"] == "INCONCLUSIVE",
            "Unexpected parent stop or verdict")
    rows = read(primary / "materials.json")
    descriptors = np.load(primary / "descriptors.npy")
    with np.load(primary / "independent_response.npz") as archive:
        data = dict(archive)
    with np.load(primary / "fit_states.npz") as archive:
        fits = dict(archive)
    fit = read(primary / "fit.json")
    require(fit["converged"] and fits["all_terminal"].shape == (4, 2, len(rows), 64)
            and fits["all_initial"].shape == (4 * 2 * len(rows), 64), "Wrong fitted state layout or qualification")
    require(np.isfinite(fits["all_terminal"]).all() and np.isfinite(fits["all_initial"]).all(), "Non-finite parent fit")
    for i, arm in enumerate(ARMS):
        selected = np.asarray(fit["selected_start"][arm])
        require(np.array_equal(fits[arm], fits["all_terminal"][i, selected, np.arange(len(rows))]),
                f"Selected state mismatch for {arm}")
    return provenance, rows, descriptors, data, {arm: fits[arm] for arm in ARMS}


def sampling_plan(data, rows):
    config = P["response"]
    target = [i for i, row in enumerate(rows) if row["tag"] == SPEC["material"]]
    require(len(target) == 1, "The authorized material does not resolve uniquely")
    n_az = len(config["azimuths_deg"])
    n_probes = len(config["angles_deg"]) * n_az
    groups = [target[0] * n_probes + config["angles_deg"].index(t) * n_az + config["azimuths_deg"].index(p)
              for t in SPEC["polar_deg"] for p in SPEC["azimuth_deg"]]
    require(groups == SPEC["group_indices"] and len(set(groups)) == 6, "Unauthorized angular allocation")
    group_count, lanes = data["use"].shape
    require(group_count == len(rows) * n_probes and lanes == SPEC["lanes_per_group"] == 1024,
            "Unexpected frozen directional-lane layout")
    require(np.all(data["mid"][groups] == target[0]), "Frozen material index does not match the target")
    n_old, n_new = config["walks"], SPEC["additional_walks_per_lane"]
    require(n_old == SPEC["parent_walks_per_lane"] == 4096 and n_new == 12288, "Unauthorized walk counts")
    require(n_old + n_new == SPEC["effective_walks_per_active_lane"] == 16384
            and (n_old + n_new) // 2 == SPEC["walks_per_combined_half"] == 8192, "Combined half size differs")
    slots = len(groups) * lanes
    require(slots * n_new == SPEC["nominal_additional_walks"] == 75497472, "Incorrect walk budget")
    seq = SPEC["sequence"]
    require(seq["initstate"] not in S.values() and seq["round"] == 0 and seq["max_depth"] == 1024,
            "Seed collision or reference configuration differs")
    active = data["use"].astype(bool)
    old_ordinal = np.cumsum(active.ravel(), dtype=np.int64).reshape(active.shape) - 1
    old_start = np.where(active[groups], old_ordinal[groups] * n_old, -1)
    new_group = np.arange(slots, dtype=np.int64).reshape(len(groups), lanes)
    new_start = (seq["pair0"] + new_group) * n_new
    last = int(new_start[-1, -1] + n_new)
    old_limit = int(active.sum()) * n_old
    peak_limit = data["peak_wo"].shape[0] * data["peak_wo"].shape[1] * config["peak_walks"]
    require(new_start[0, 0] >= max(old_limit, peak_limit) and last < 2**63,
            "Additional stream sequences overlap a parent sequence range")
    require(np.all(np.diff(new_start.ravel()) == n_new), "Per-lane new stream intervals overlap")
    for key in ("a", "b"):
        require(np.all(data[key][groups][~active[groups]] == 0), "Unexpected parent value at a rejected lane")
    ids = {"groups": np.asarray(groups), "use": active[groups], "parent_a_start": old_start,
           "parent_b_start": np.where(old_start < 0, -1, old_start + n_old // 2),
           "additional_a_start": new_start, "additional_b_start": new_start + n_new // 2}
    plan = {
        "material": SPEC["material"], "groups": groups, "lanes_per_group": lanes,
        "active_by_group": active[groups].sum(axis=1).tolist(),
        "original_initstate": S["evaluation_walks"],
        "original_initseq_range_half_open": [0, old_limit],
        "additional_initstate": seq["initstate"], "additional_pair0": seq["pair0"],
        "additional_initseq_range_half_open": [int(new_start[0, 0]), last],
        "parent_walks_per_active_half": n_old // 2, "additional_walks_per_half": n_new // 2,
        "combined_walks_per_active_half": (n_old + n_new) // 2,
        "additional_walks_total": slots * n_new, "additional_walks_used": int(active[groups].sum()) * n_new,
        "additional_walks_zero_weight": int((~active[groups]).sum()) * n_new,
    }
    return ids, plan


def combine_halves(parent, additional, use, n_old=2048, n_new=6144):
    require(parent.shape == additional.shape and parent.shape[:-1] == use.shape, "Half layout differs")
    out = parent.copy()
    out[use] = (n_old * parent[use] + n_new * additional[use]) / (n_old + n_new)
    return out


def precision(data, rows, groups):
    config = P["response"]
    angles = np.repeat(config["angles_deg"], len(config["azimuths_deg"]))
    in_band = (angles >= 55) & (angles <= 65)
    graze = angles >= 80
    weigh = (data["use"] / data["q"])[..., None]
    metrics = terms(data["a"], data["b"], weigh)
    noise = metric_arrays(metrics, len(rows))
    by_material = [
        {"band_shape": float(row[in_band].mean() * 0.5), "grazing_shape": float(row[graze].mean() * 0.5),
         "albedo": float(noise["ealb"][i].mean() * 0.5)}
        for i, row in enumerate(noise["eshape"])
    ]
    passed = all(row["band_shape"] <= config["max_band_shape_noise"] and
                 row["grazing_shape"] <= config["max_grazing_shape_noise"] and row["albedo"] <= config["max_albedo_noise"]
                 for row in by_material)
    return by_material, (metrics["eshape"][groups] * 0.5).tolist(), passed


def focused_tests(data, rows, ids, plan, original_metrics):
    require(plan["additional_walks_total"] == 75497472 and plan["combined_walks_per_active_half"] == 8192,
            "Sample-count contract failed")
    require(plan["additional_walks_used"] + plan["additional_walks_zero_weight"] == 75497472,
            "Mask accounting failed")
    require(np.all(ids["additional_b_start"] - ids["additional_a_start"] == 6144), "New half overlap")
    require(np.all((ids["parent_b_start"] - ids["parent_a_start"])[ids["use"]] == 2048), "Original half overlap")
    require(np.all(ids["additional_a_start"][ids["use"]] > ids["parent_b_start"][ids["use"]] + 2047),
            "Original/new stream overlap")
    a = np.array([[2.0], [0.0]])
    b = np.array([[5.0], [15.0]])
    require(np.array_equal(combine_halves(a, b, np.array([True, False])), [[4.25], [0.0]]),
            "Count-weighted half reconstruction failed")
    prior, _, passed = precision(data, rows, ids["groups"])
    require(prior == original_metrics["precision"] and not passed and not original_metrics["arms"],
            "Original frozen precision calculation did not reproduce its stop")
    print("PASS parent hashes, exact interval accounting, independent halves, and frozen gate replay", flush=True)


def ref_stage(out, data, rows, descriptors, ids, plan):
    groups, use = ids["groups"], ids["use"]
    seq = SPEC["sequence"]
    wi, wo, mid = (data[key][groups] for key in ("wi", "wo", "mid"))
    start = time.monotonic()
    a, b = DA.ref_eval(descriptors, mid.ravel(), wi.reshape(-1, 3), wo.reshape(-1, 3),
                       SPEC["additional_walks_per_lane"], seq["initstate"], seq["round"], seq["max_depth"], seq["pair0"])
    elapsed = time.monotonic() - start
    a, b = a.reshape(wi.shape), b.reshape(wi.shape)
    require(np.isfinite(a).all() and np.isfinite(b).all(), "Non-finite added reference walks")
    np.savez(out / "additional_halves.npz", a=a, b=b, **ids)
    amend = data.copy()
    for key, fresh in (("a", a), ("b", b)):
        half = data[key].copy()
        half[groups] = combine_halves(half[groups], fresh, use)
        amend[key] = half
    untouched = np.ones(data["a"].shape[:2], dtype=bool)
    untouched[groups] = ~use
    for key in data:
        before, after = data[key], amend[key]
        if key in ("a", "b"):
            require(np.array_equal(before[untouched], after[untouched]), f"Unmodified reference lanes changed: {key}")
        else:
            require(np.array_equal(before, after), f"Immutable reference array changed: {key}")
    before, old_probes, old_pass = precision(data, rows, groups)
    after, new_probes, new_pass = precision(amend, rows, groups)
    np.savez(out / "amended_reference.npz", **amend)
    row_index = next(i for i, row in enumerate(rows) if row["tag"] == SPEC["material"])
    write(out / "reference_precision.json", {
        "reference_seconds": elapsed, "plan": plan, "target": SPEC["material"],
        "original_passed": old_pass, "amended_passed": new_pass,
        "original_by_material": before, "amended_by_material": after,
        "original_target": before[row_index], "amended_target": after[row_index],
        "per_group": [{"theta_deg": t, "phi_deg": p, "original_eshape_half_noise": old_probes[i],
                       "amended_eshape_half_noise": new_probes[i]}
                      for i, (t, p) in enumerate((t, p) for t in SPEC["polar_deg"] for p in SPEC["azimuth_deg"])],
        "added_arrays_sha256": H.C.sha(out / "additional_halves.npz"),
        "amended_arrays_sha256": H.C.sha(out / "amended_reference.npz"),
        "inactive_and_untouched_lanes_bit_equal": True,
    })
    print("REFERENCE", before[row_index], "->", after[row_index], "passed", new_pass, flush=True)
    return amend, new_pass


def summarize_effects(decision, rows):
    contrasts = decision["contrasts"]
    result = {}
    for label, transform in (
        ("data", lambda ca, ba, da, db, dc: 0.5 * (ca + db)),
        ("objective", lambda ca, ba, da, db, dc: 0.5 * (ba + dc)),
        ("interaction", lambda ca, ba, da, db, dc: da - ba - ca),
    ):
        response = transform(*(np.asarray(contrasts[p]["response_gain"]) for p in ("CA", "BA", "DA", "DB", "DC")))
        scene = transform(*(np.asarray([r["point"] for r in contrasts[p]["scene_gains"]])
                            for p in ("CA", "BA", "DA", "DB", "DC")))
        result[label] = {
            "per_material_response_eshape_gain": response.tolist(), "per_scene_total_rms_gain": scene.tolist(),
            "cohorts": {name: {
                "median_response": float(np.median(response[[r["cohort"] == name for r in rows]])),
                "median_scene": float(np.median(scene[[r["cohort"] == name for r in rows]])),
            } for name in ("development", "held_out")},
        }
    return result


def provenance(out, primary, commit, parent):
    paths = {Path(m.__file__).resolve() for m in tuple(sys.modules.values()) if getattr(m, "__file__", None)}
    source = {str(path.relative_to(REPO)): H.C.sha(path) for path in sorted(paths)
              if path.is_relative_to(REPO / "experiments")}
    source[str(CONFIG.relative_to(REPO))] = H.C.sha(CONFIG)
    write(out / "provenance.json", {
        "commit": commit, "protocol": A1, "code_and_configuration_sha256": source,
        "parent_primary": str(primary), "parent_provenance_sha256": H.C.sha(primary / "provenance.json"),
        "parent_output_sha256": parent["output_sha256"], "parent_source_sha256": parent["source_sha256"],
        "environment": {
            "host": socket.gethostname(), "platform": platform.platform(), "python": platform.python_version(),
            "numpy": np.__version__, "torch": torch.__version__, "cuda": torch.version.cuda,
            "mitsuba": H.mi.__version__, "drjit": H.dr.__version__, "variant": H.mi.variant(),
            "gpu": subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                                            "--format=csv,noheader"], text=True).strip(),
        },
        "output_sha256": {str(path.relative_to(out)): H.C.sha(path) for path in sorted(out.rglob("*"))
                          if path.is_file() and path.name != "provenance.json"},
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("primary", type=Path)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--commit", required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    status = {"commit": args.commit, "admission": "RUNNING", "stage": "verify_parent",
              "amendment_protocol_sha256": H.C.sha(CONFIG)}
    parent = None
    start = time.monotonic()

    def mark(stage):
        status["stage"] = stage
        write(out / "status.json", status)
        print("stage", stage, flush=True)

    try:
        mark("verify_parent")
        parent, rows, desc, old, states = verify_parent(args.primary, args.source)
        ids, plan = sampling_plan(old, rows)
        focused_tests(old, rows, ids, plan, read(args.primary / "response.json"))
        write(out / "materials.json", rows)
        write(out / "sampling_plan.json", plan)
        mark("reference_increase")
        ref, qualified = ref_stage(out, old, rows, desc, ids, plan)
        if not qualified:
            status.update({"admission": "REFERENCE_UNQUALIFIED", "verdict": "INCONCLUSIVE"})
            return
        mark("independent_response")
        states.update(historical_states(args.source, rows))
        response, compiled = record_response(ref, states, out)
        require(response["reference_precision_passed"], "The parent evaluator disagrees with precision replay")
        require(response["precision"] == read(out / "reference_precision.json")["amended_by_material"],
                "The parent evaluator produced different precision records")
        if not response["physics_passed"]:
            status.update({"admission": "PHYSICAL_CONTRACT_FAILED", "verdict": "INCONCLUSIVE"})
            return
        mark("scene_evaluation")
        require(H.RT.P.D == 6, "Frozen tracer event count differs")
        del old, ref
        torch.cuda.empty_cache()
        scenes = {}
        for i, row in enumerate(rows):
            scenes[row["image"]] = render_case(row, desc[i], {name: state[i] for name, state in compiled.items()}, i, out)
            write(out / "scenes.json", scenes)
        if not all(row["precision_passed"] for row in scenes.values()):
            status.update({"admission": "SCENE_PRECISION_UNQUALIFIED", "verdict": "INCONCLUSIVE"})
            return
        mark("decision")
        decision = classify(response, scenes, rows)
        write(out / "decision.json", decision)
        write(out / "factorial_effects.json", summarize_effects(decision, rows))
        status.update({"admission": "QUALIFIED", "verdict": decision["verdict"]})
    except Exception as exc:
        status.update({"admission": "EXECUTION_ERROR", "verdict": "INCONCLUSIVE", "error": str(exc)})
        raise
    finally:
        status["seconds"] = time.monotonic() - start
        write(out / "status.json", status)
        if parent is not None:
            provenance(out, args.primary, args.commit, parent)
        print("WL104D Amendment 1", status, flush=True)


if __name__ == "__main__":
    main()
