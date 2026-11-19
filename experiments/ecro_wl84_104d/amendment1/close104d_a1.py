"""Record an owner-mandated scene precision stop without evaluating unfinished cases."""

import argparse
import datetime as dt
import re
import subprocess
import zipfile
from pathlib import Path

from run104d_a1 import CONFIG, H, P, provenance, read, require, verify_hash, verify_parent, write

ROIS = ("objects", "highlight_core", "highlight_ring", "grazing")


def precision_failures(scenes):
    limits = P["scene"]
    failures = []
    for image, scene in scenes.items():
        first = len(failures)
        for arm, raw in scene["arms"].items():
            t = raw["stable_rms"]["objects"]
            width = 0.5 * (t["hi"] - t["lo"])
            if width > limits["precision_rms_half_width"]:
                failures.append({"image": image, "arm": arm, "roi": "objects", "event": "total",
                                 "metric": "total_rms", "half_width": width, "limit": limits["precision_rms_half_width"]})
            for roi in ROIS:
                for e, v in raw["signed"].get(roi, {}).items():
                    width = 0.5 * (v["hi"] - v["lo"])
                    if width > limits["precision_energy_half_width"]:
                        failures.append({"image": image, "arm": arm, "roi": roi, "event": e,
                                         "metric": "signed_energy", "half_width": width,
                                         "limit": limits["precision_energy_half_width"]})
        require(scene["precision_passed"] == (len(failures) == first), f"Stored gate differs for {image}")
    return failures


def verify_replicates(out, scenes, rows):
    expected = {row["image"] for row in rows}
    target_count = P["scene"]["replicates"]
    counts = {}
    for directory in sorted((out / "scenes").iterdir()):
        require(directory.name in expected, f"Unknown case {directory.name}")
        valid, truncated, indices = [], [], []
        for path in sorted(directory.glob("r[0-9][0-9][0-9].npz")):
            require(re.fullmatch(r"r\d{3}\.npz", path.name), f"Unknown replicate name {path}")
            indices.append(int(path.stem[1:]))
            try:
                with zipfile.ZipFile(path) as archive:
                    if archive.testzip() is None and len(archive.namelist()) == 21:
                        valid.append(path.name)
                    else:
                        truncated.append(path.name)
            except zipfile.BadZipFile:
                truncated.append(path.name)
        require(indices == list(range(len(indices))) and len(indices) <= target_count, "Noncontiguous replicate files")
        completed = directory.name in scenes
        if completed:
            require(len(valid) == target_count and not truncated, f"Completed case corrupted: {directory.name}")
            require(read(directory / "metrics.json") == scenes[directory.name], "Completed case metrics differ")
        counts[directory.name] = {"completed": completed, "valid_replicates": valid, "incomplete_files": truncated}
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("primary", type=Path)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("execution_log", type=Path)
    parser.add_argument("--pid", type=int, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    process = subprocess.run(["ps", "-p", str(args.pid), "-o", "stat="], capture_output=True, text=True)
    require(not process.stdout.strip() or process.stdout.lstrip().startswith("Z"), "Renderer is still running")
    original_status = read(out / "status.json")
    require(original_status["admission"] == "RUNNING" and original_status["stage"] == "scene_evaluation",
            "Unexpected status for owner-mandated stop")
    require(original_status["amendment_protocol_sha256"] == H.C.sha(CONFIG), "Amendment protocol hash differs")
    require(not (out / "decision.json").exists(), "An end-to-end decision was already computed")
    primary, rows, _, _, _ = verify_parent(args.primary, args.source)
    precision = read(out / "reference_precision.json")
    verify_hash(out / "additional_halves.npz", precision["added_arrays_sha256"])
    verify_hash(out / "amended_reference.npz", precision["amended_arrays_sha256"])
    response = read(out / "response.json")
    require(precision["amended_passed"] and response["physics_passed"], "Invalid preceding stages")
    scenes = read(out / "scenes.json")
    failures = precision_failures(scenes)
    require(bool(failures), "The frozen scene precision gate has not failed")
    buffer_status = verify_replicates(out, scenes, rows)
    with (out / "status_before_stop.json").open("xb") as dest:
        dest.write((out / "status.json").read_bytes())
    with (out / "execution.log").open("xb") as dest:
        dest.write(args.execution_log.read_bytes())
    closeout = {
        **original_status, "admission": "SCENE_PRECISION_UNQUALIFIED", "verdict": "INCONCLUSIVE",
        "termination": "Owner stop condition applied to an already completed scene precision failure.",
        "terminated_render_pid": args.pid,
        "recorded_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "frozen_precision_failures": failures, "case_buffers": buffer_status,
        "unreached_cases": [row["image"] for row in rows if row["image"] not in buffer_status],
        "full_scene_count": len(scenes),
        "completed_render_seconds_sum_by_arm": {
            arm: sum(scene["seconds_by_arm"][arm] for scene in scenes.values())
            for arm in next(iter(scenes.values()))["seconds_by_arm"]
        },
        "closeout_script_sha256": H.C.sha(Path(__file__)),
    }
    write(out / "status.json", closeout)
    provenance(out, args.primary, original_status["commit"], primary)
    print("Preserved owner stop", {key: closeout[key] for key in ("admission", "verdict", "full_scene_count")}, flush=True)
    print("Frozen precision failure count", len(failures), flush=True)


if __name__ == "__main__":
    main()
