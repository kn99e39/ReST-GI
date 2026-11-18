"""Run WL101A using the original optimized WL101 renderer and archived arrays."""
import argparse
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np

from common import (EVENTS, P, PROTOCOL, RES, components, fingerprint, image,
                    rel_l1, sample_seed, sha, write_json)


def gpu_query(query, fields):
    return subprocess.check_output(
        ["nvidia-smi", f"--query-{query}={fields}", "--format=csv,noheader,nounits"],
        text=True,
        timeout=15,
    ).strip()


def foreign_gpu():
    return [int(pid) for pid in gpu_query("compute-apps", "pid").splitlines()
            if pid.strip() and int(pid) != os.getpid()]


def wait_for_gpu():
    while foreign_gpu():
        print("Waiting for foreign GPU process to finish", flush=True)
        time.sleep(10)


def validate_inputs(wl101, ref_dir):
    stack = P["stack"]
    hashes = {
        "experiments/ecro_dyn101/rt101.py": stack["server_renderer_sha256"],
        "experiments/ecro_dyn101/seq101.py": stack["server_workload_sha256"],
        "experiments/ecro_dyn101/protocol/dyn101.json": stack["server_wl101_protocol_sha256"],
        "experiments/ecro_material100/protocol.json": stack["server_wl100_protocol_sha256"],
        "results/ecro_material100/configs.json": P["configs_sha256"],
        "results/ecro_material100/sss/model.pt": P["checkpoint_sha256"],
    }
    assert (wl101 / "DEPLOYED_COMMIT").read_text().strip() == stack["wl101_deployed_commit"]
    for name, digest in hashes.items():
        assert sha(wl101 / name) == digest, name
    reference = P["references"]
    for name, digest in reference["sha256"].items():
        assert sha(ref_dir / name) == digest, name
    import json

    for half, seed in enumerate(reference["seed_by_half"]):
        meta = json.loads((ref_dir / f"REF_0_{half}.json").read_text())
        assert meta["geom"] == P["geometry"] and meta["light"] == P["light"]
        assert meta["seed"] == seed and meta["spp"] == reference["spp_per_half"]
        for name, digest in meta["frozen_sources"].items():
            assert sha(wl101 / "experiments" / name) == digest, name
            hashes["experiments/" + name] = digest

    for arm, versions in (
        ("A", (stack["reference_mitsuba"], stack["reference_drjit"])),
        ("B", (stack["mitsuba"], stack["drjit"])),
    ):
        meta = json.loads((ref_dir / f"PRED_{arm}_0.json").read_text())
        assert (meta["versions"]["mitsuba"], meta["versions"]["drjit"]) == versions
        assert meta["checkpoint_sha256"] == P["checkpoint_sha256"]
        assert (meta["state"], meta["event"]) == (0, 4)
        assert meta["seed"] == reference["historical_prediction_seed"]
        assert meta["spp"] == reference["spp_per_half"]
    return hashes


def compare(a, b):
    c_a, c_b = components(a), components(b)
    return {key: rel_l1(c_a[..., i], c_b[..., i])
            for i, key in enumerate(("full", "low", "high", "e1", "e2"))}


def main(wl101, out, commit):
    assert len(commit) == 40 and all(char in "0123456789abcdef" for char in commit)
    out.mkdir(parents=True, exist_ok=False)
    ref_dir = wl101 / P["references"]["path_in_wl101"]
    source_hashes = validate_inputs(wl101, ref_dir)
    photos_a = image(ref_dir / "PRED_A_0.npz")
    photos_b = image(ref_dir / "PRED_B_0.npz")
    evidence = {"code_commit": commit, "historical_A_B": compare(photos_b, photos_a)}
    threshold = P["gates"]["historical_and_replay_image_rel_l1"]
    assert max(evidence["historical_A_B"].values()) <= threshold, evidence

    gpu = [field.strip() for field in gpu_query("gpu", "name,driver_version").split(",")]
    stack = P["stack"]
    assert gpu == [stack["gpu"], stack["driver"]], gpu
    wait_for_gpu()
    sys.path.insert(0, str(wl101 / "experiments" / "ecro_dyn101"))
    os.environ["FREEZE"] = "1"
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    sys.dont_write_bytecode = True
    import rt101 as runtime
    import seq101 as seq

    assert (runtime.mi.__version__, runtime.dr.__version__, runtime.torch.__version__, np.__version__) == (
        stack["mitsuba"], stack["drjit"], stack["torch"], stack["numpy"]
    )
    assert runtime.FREEZE and runtime.mi.variant() == stack["variant"]
    cam = dict(runtime.C100.P["datasets"]["sss"]["camera"], res=RES)
    assert cam == P["camera"]
    case, start, events = seq.workload(P["workload"])
    assert case == "sss" and seq.WARM == 4
    assert events[seq.WARM] == (P["geometry"], P["light"], 0)
    model = runtime.load_model(case)
    rt = runtime.RT(start[0], case, model, start[1])
    for geometry, light, state_id in events[:seq.WARM + 1]:
        moved, old_geometry = rt.set_geom(geometry)
        rt.update_state(old_geometry, moved)
        rt.set_light(light)
        source = rt.source()
        table, _ = rt.respond(source)
    assert state_id == 0
    second_source = rt.source()
    second_table, _ = rt.respond(second_source)
    assert runtime.torch.equal(source, second_source)
    assert runtime.torch.equal(table, second_table)
    table_before = table.detach().cpu().numpy().copy()
    checks = {}
    for arm in ("A", "B"):
        with np.load(ref_dir / f"PRED_{arm}_0.npz") as historical:
            checks[arm] = rel_l1(table_before, historical["B"])
    evidence["table_rel_l1"] = checks
    assert max(checks.values()) <= P["gates"]["table_rel_l1"], checks
    table_dr = runtime.t2d(table.view(-1))

    def render(spp, seed):
        wait_for_gpu()
        runtime.sync()
        start_time = time.perf_counter()
        accum = runtime.render_hq(rt, table_dr, cam, RES, spp, seed, b=min(32, spp))
        runtime.sync()
        seconds = time.perf_counter() - start_time
        foreign = foreign_gpu()
        normalized = accum.reshape(3, 3, RES, RES).transpose(2, 3, 1, 0) / spp
        assert np.isfinite(normalized).all() and normalized.min() >= 0
        e1, e2, high = np.moveaxis(normalized.astype(np.float32), -1, 0)
        full_native = normalized.sum(axis=-1).astype(np.float32)
        recomposed = (e1.astype(np.float64) + e2 + high).astype(np.float32)
        recomp_error = rel_l1(recomposed, full_native)
        assert recomp_error <= P["gates"]["repeat_rel_l1"]
        return {"e1": e1, "e2": e2, "hl": high, "full": full_native}, {
            "seed": seed, "spp": spp, "seconds": seconds, "foreign_pids_after": foreign,
            "recomposition_relative_l1": recomp_error
        }

    metadata = {
        "environment": {**stack, "python": platform.python_version(), "os": platform.platform(),
                        "observed_gpu": gpu, "device": runtime.torch.cuda.get_device_name()},
        "benchmark_scope": P["sweep"]["timing"],
        "scene_sha256": fingerprint({key: P[key] for key in ("geometry", "light", "camera", "sphere")}),
        "protocol_sha256": sha(PROTOCOL), "configs_sha256": P["configs_sha256"],
        "checkpoint_sha256": P["checkpoint_sha256"], "original_files_sha256": source_hashes,
        "reference_root": str(ref_dir), "archived_arrays_sha256": P["references"]["sha256"],
        "response_sha256": __import__("hashlib").sha256(table_before.tobytes()).hexdigest(),
        "frames": [], "matched": [], "gates": evidence,
    }
    np.savez(out / "state.npz", table=table_before, source=source.cpu().numpy())
    write_json(out / "run_progress.json", metadata)

    def write_frame(name, result):
        np.savez(out / name, **result)
        return {"path": name, "sha256": sha(out / name)}

    def as_events(result):
        return np.stack([result[key] for key in EVENTS], axis=-1)

    reference = P["references"]
    high_spp = reference["spp_per_half"]
    result, stamp = render(high_spp, reference["historical_prediction_seed"])
    evidence["fresh_B_replay"] = compare(as_events(result), photos_b)
    assert max(evidence["fresh_B_replay"].values()) <= threshold, evidence
    metadata["historical_replay"] = {**stamp, **write_frame("replay_B_4096.npz", result)}
    print("Archived A/B and fresh B full and component checks passed", flush=True)

    for half, seed in enumerate(reference["seed_by_half"]):
        result, stamp = render(high_spp, seed)
        comparison = compare(as_events(result), image(ref_dir / f"REF_0_{half}.npz", "h"))
        checks = {key: comparison[key] for key in ("e1", "e2", "low")}
        assert max(checks.values()) <= P["gates"]["paired_low_event_rel_l1"], checks
        metadata["matched"].append({**stamp, **write_frame(f"paired_{half}_4096.npz", result),
                                    "half": half, "paired_low_rel_l1": checks})
        print(f"Matched reference half {half}: {checks}", flush=True)

    sweep = P["sweep"]
    seeds = [sample_seed(spp, rep) for spp in sweep["spp"] for rep in range(sweep["replicates"])]
    assert len(seeds) == len(set(seeds)) and set(seeds).isdisjoint(reference["seed_by_half"])
    for spp in sweep["spp"]:
        warm_seed = 10169000 + spp
        assert warm_seed not in seeds
        render(spp, warm_seed)
        for rep in range(sweep["replicates"]):
            result, stamp = render(spp, sample_seed(spp, rep))
            name = f"spp{spp:03d}_seed{stamp['seed']}.npz"
            metadata["frames"].append({**stamp, **write_frame(name, result), "replicate": rep})
            if spp == 32 and rep == 0:
                repeated, _ = render(spp, stamp["seed"])
                evidence["repeat_32"] = compare(as_events(repeated), as_events(result))
                assert max(evidence["repeat_32"].values()) <= P["gates"]["repeat_rel_l1"], evidence
        write_json(out / "run_progress.json", metadata)
        print(f"Completed {spp} spp × {sweep['replicates']}", flush=True)
    assert np.array_equal(table.detach().cpu().numpy(), table_before)
    evidence["table_bit_identical"] = True
    write_json(out / "run_manifest.json", metadata)
    print("WL101A render complete", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wl101", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("commit")
    args = parser.parse_args()
    main(args.wl101.resolve(), args.out.resolve(), args.commit)
