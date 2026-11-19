"""Replay the frozen baseline, then check angular support without fitting."""

import argparse
import json
import platform
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
PROTOCOL = HERE / "protocol" / "wl104c.json"
P = json.loads(PROTOCOL.read_text(encoding="utf-8"))
EVENTS = ("e1", "e2", "h", "total")


def angular_support(path, band, radius, grazing):
    with np.load(path) as data:
        n_mat, n_wi, n_wo = (int(data[key]) for key in ("n_mat", "n_wi", "n_wo"))
        wi = data["wi"].astype(np.float64).reshape(n_mat, n_wi, n_wo, 3)
        wo = data["wo"].astype(np.float64).reshape(wi.shape)
        use = data["use"].reshape(wi.shape[:-1]).astype(bool)
        pdf = data["q"].reshape(use.shape)
    if not np.array_equal(wi, np.broadcast_to(wi[:, :, :1], wi.shape)):
        raise ValueError("Expected one incident direction per WL83 fit group")
    if not (np.isfinite(pdf[use]).all() and (pdf[use] > 0).all()):
        raise ValueError("Invalid proposal density in a used fit lane")
    theta_i = np.degrees(np.arccos(np.clip(wi[..., 2], -1, 1)))
    theta_o = np.degrees(np.arccos(np.clip(wo[..., 2], -1, 1)))
    in_i = (theta_i >= band[0]) & (theta_i <= band[1])
    in_o = (theta_o >= band[0]) & (theta_o <= band[1])
    same_band = use & in_i & in_o
    near_mirror = np.sum(wi * wo * [-1, -1, 1], axis=-1) >= np.cos(np.radians(radius))
    theta_group = theta_i[:, :, 0]
    grazing_groups = ((theta_group >= grazing) & use.any(axis=-1)).sum(axis=1)
    return {
        "materials": n_mat,
        "incident_groups_per_material": n_wi,
        "fit_lanes": n_mat * n_wi * n_wo,
        "used_fit_lanes": int(use.sum()),
        "incident_groups_in_band": int(in_i[:, :, 0].sum()),
        "used_lanes_with_outgoing_in_band": int((use & in_o).sum()),
        "used_lanes_with_both_in_band": int(same_band.sum()),
        "used_near_mirror_lanes_with_both_in_band": int((same_band & near_mirror).sum()),
        "nearest_incidence_below_band_deg": float(theta_group[theta_group < band[0]].max()),
        "nearest_incidence_above_band_deg": float(theta_group[theta_group > band[1]].min()),
        "grazing_groups_per_material": grazing_groups.tolist(),
    }


def response_replay(H, mat, set_name, index):
    import torch

    with np.load(H.IN63B / "data" / f"eval83_{set_name}.npz") as data:
        _, n_th, n_phi, n_wo = data["shape"]
        n_groups = n_th * n_phi
        selection = slice(index * n_groups * n_wo, (index + 1) * n_groups * n_wo)
        if not np.all(data["m"][selection] == index):
            raise ValueError("Unexpected WL83 evaluation lane order")
        wi, wo, ref, q, use = (data[k][selection] for k in ("wi", "wo", "f", "q", "use"))
    fitted = mat["ER_DF"]
    predict = H.er_eval_torch(fitted["p32"], wi, wo)
    runtime = H.er_eval_dr(fitted["state"], wi, wo)
    fw = H.er_eval_torch(fitted["p"], wi, wo) / wo[:, 2, None].clip(1e-12)
    bw = H.er_eval_torch(fitted["p"], wo, wi) / wi[:, 2, None].clip(1e-12)
    eligible = (wi[:, 2] > 0.05) & (wo[:, 2] > 0.05)

    def tensor(array):
        return torch.as_tensor(array, device=H.dev, dtype=torch.float64)

    metrics, _, _ = H.E82.group_metrics(
        tensor(predict).reshape(n_groups, n_wo, 3),
        tensor(ref).reshape(n_groups, n_wo, 3),
        tensor(use.astype(np.float64) / q).reshape(n_groups, n_wo, 1),
        tensor(np.ones((n_groups, 3))),
    )
    with np.load(H.IN63B / "eval" / f"lanes83_{set_name}.npz") as lanes:
        differences = {
            key: float(np.max(np.abs(metrics[key] - lanes[f"DF_ER__{key}"][index].reshape(-1))))
            for key in ("el1", "ealb", "eshape")
        }
    lam = fitted["state"][:24].reshape(3, 8).astype(np.float64)
    albedo_bound = lam.sum(axis=1)
    runtime_error = float(np.max(np.abs(predict - runtime)) / np.max(np.abs(predict)))
    reciprocity = float(np.max(np.abs(fw - bw)[eligible]) / np.max(np.abs(fw[eligible])))
    state_ok = np.isfinite(fitted["state"]).all() and (lam >= 0).all() and (albedo_bound <= 1.000001).all()
    return {
        "lane_differences_max": differences,
        "lane_means": {key: float(metrics[key].mean()) for key in differences},
        "runtime_error": runtime_error,
        "reciprocity_error": reciprocity,
        "certified_albedo_upper_bound_rgb": albedo_bound.tolist(),
        "compiled_state_sha256": fitted["sha256"],
        "passed": bool(state_ok and max(differences.values()) <= 1e-4 and runtime_error <= 1e-5 and reciprocity <= 1e-4),
    }


def scene_replay(H, image, image_index, archive, out):
    tag, _, _, _ = H.C.image_spec(image)
    ctx_ref = H.RT.Ctx(H.scene(tag, "LAY1024"))
    ctx_df = H.RT.Ctx(H.scene(tag, "ER_DF"))
    sid, _, _, names = H.HA.gbuffer(ctx_ref)
    objects = np.isin(sid, [names.index("sphere"), names.index("cube")])
    replicate = P["baseline"]["replicate"]
    seed = 1049000 + 10000 * image_index + replicate
    archive_path = archive / image / f"rep/r{replicate:03d}.npz"
    start = time.monotonic()
    replay = {
        "LAY1024": H.HA.render(ctx_ref, image, 4096, seed, "cam", pass_seed=True),
        "ER_DF": H.HA.render(ctx_df, image, 4096, seed, "cam"),
    }
    wall = time.monotonic() - start
    path = out / f"{image}_replay.npz"
    np.savez(path, **{f"{arm}_{event}": a for arm, values in replay.items() for event, a in values.items()})
    with np.load(archive_path) as raw:
        actual = {
            arm: {key: raw[f"{arm}_{key}"].astype(np.float64) for key in EVENTS[:3]}
            for arm in replay
        }
    for bundle in (replay, actual):
        for values in bundle.values():
            values["total"] = sum(values[key] for key in EVENTS[:3])
    ref_energy = float(H.lum(actual["LAY1024"]["total"][objects]).sum())
    comparisons = {}
    for arm, values in replay.items():
        comparisons[arm] = {}
        for key in EVENTS:
            archived = actual[arm][key][objects]
            difference = values[key][objects] - archived
            comparisons[arm][key] = {
                "relative_l1": float(np.abs(difference).sum() / np.abs(archived).sum()),
                "normalized_signed_difference": float(H.lum(difference).sum() / ref_energy),
            }
    passed = all(
        row["relative_l1"] <= P["baseline"]["scene_rgb_relative_l1_max"]
        and abs(row["normalized_signed_difference"]) <= P["baseline"]["scene_energy_difference_max"]
        for arm in comparisons.values() for row in arm.values()
    )
    return {
        "seconds": wall, "seed": seed, "spp": 4096, "objects_pixels": int(objects.sum()),
        "archived_replicate_sha256": H.C.sha(archive_path), "replay_sha256": H.C.sha(path),
        "comparisons": comparisons, "passed": passed,
    }


def input_hashes(H, sets):
    wl104 = H.C.P
    inputs = {"WL104B_metrics": H.C.REPO / "evaluation/104B/metrics.json"}
    for set_name in sets:
        path = H.IN63B / "data" / f"fit82_{set_name}.npz"
        inputs[path.name] = path
    relative_paths = ["data/materials83.npz"]
    for set_name in ("test83_iid", "test83_hard"):
        relative_paths.append(f"fit/fit_{set_name}_ER_cert_metric_12000_generic.npz")
        for folder, name in (("data", f"eval83_{set_name}"), ("eval", f"lanes83_{set_name}")):
            path = H.IN63B / folder / f"{name}.npz"
            inputs[path.name] = path
    for relative in relative_paths:
        path = H.C.W83 / relative
        if H.C.sha(path) != wl104["inputs_sha256"][f"rest_compiled83/results/{relative}"]:
            raise ValueError(f"WL104 input SHA mismatch for {relative}")
        inputs[path.name] = path
    for label, path in (("sphere", H.C.SPHERE_PLY), ("quad", H.C.QUAD_PLY)):
        if H.C.sha(path) != wl104[f"{label}_ply_sha256"]:
            raise ValueError(f"WL104 mesh SHA mismatch for {label}")
        inputs[path.name] = path
    return {key: H.C.sha(path) for key, path in inputs.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path, help="WL104B evidence directory")
    parser.add_argument("output", type=Path)
    parser.add_argument("--commit", required=True, help="Git commit used to build the server code archive")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(HERE.parent / "ecro_wl84_104b"))
    import h104b as H
    import torch

    start = time.monotonic()
    coverage = P["coverage"]
    result = {
        "base_commit": P["base_commit"],
        "experiment_commit": args.commit,
        "protocol_sha256": H.C.sha(PROTOCOL),
        "environment": {
            "host": socket.gethostname(), "platform": platform.platform(), "python": platform.python_version(),
            "numpy": np.__version__, "torch": torch.__version__, "cuda": torch.version.cuda,
            "mitsuba": H.mi.__version__, "drjit": H.dr.__version__, "variant": H.mi.variant(),
            "gpu_and_driver": subprocess.check_output(
                ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"], text=True
            ).strip(),
        },
        "input_sha256": input_hashes(H, coverage["sets"]),
        "response": {}, "scenes": {},
    }
    report = args.output / "preflight.json"
    for index, image in enumerate(H.C.IMAGES):
        tag, _, set_name, k = H.C.image_spec(image)
        mat = H.material(tag)
        response = response_replay(H, mat, set_name, k)
        response["compiled_state_matches_wl84"] = mat["ER_DF"]["sha256"] == H.DF_STATE_SHA[tag]
        result["response"][image] = response
        print(image, "response", response, flush=True)
        if response["passed"]:
            H.register(tag, mat)
            result["scenes"][image] = scene_replay(H, image, index, args.archive, args.output)
            print(image, "scene", result["scenes"][image], flush=True)
        H.C.write_json(report, result)
        if not response["passed"] or not result["scenes"][image]["passed"]:
            result["admission"] = "BASELINE_REPRODUCTION_FAILED"
            break
    else:
        result["coverage"] = {
            set_name: angular_support(
                H.IN63B / "data" / f"fit82_{set_name}.npz", coverage["required_incident_band_deg"],
                coverage["near_mirror_radius_deg"], coverage["grazing_min_deg"],
            )
            for set_name in coverage["sets"]
        }
        no_support = any(row["used_near_mirror_lanes_with_both_in_band"] == 0 for row in result["coverage"].values())
        result["admission"] = "DATA_UNQUALIFIED" if no_support else "STRUCTURAL_CHECK_PASSED"
    result["architecture_verdict"] = "INCONCLUSIVE"
    result["seconds"] = time.monotonic() - start
    result["executed_local_code_sha256"] = {
        str(Path(filename).relative_to(REPO)): H.C.sha(filename)
        for module in tuple(sys.modules.values())
        if (filename := getattr(module, "__file__", None)) and Path(filename).is_relative_to(HERE.parent)
    }
    H.C.write_json(report, result)
    print("WL104C", result["admission"], result["architecture_verdict"], flush=True)


if __name__ == "__main__":
    main()
