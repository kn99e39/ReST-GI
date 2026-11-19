"""WL104D gated run. Paths are resolved relative to this experiment's archive."""

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

from common104d import H, HERE, P, REPO, StopRun, historical_states, load_original, population, write
from data104d import coverage, response_reference
from decision104d import classify
from fit104d import fit
from response104d import record_response
from scenes104d import render_case
from tests104d import run_tests


def baseline(out, commit):
    archived = REPO / "evaluation/104C/preflight.json"
    result_dir = out / "baseline_replay"
    ref_inputs = REPO / "results/ecro_wl84_104b/evidence"
    subprocess.run([sys.executable, "-B", str(HERE.parent / "ecro_wl84_104c/preflight104c.py"),
                    str(ref_inputs), str(result_dir), "--commit", commit], check=True)
    fresh = json.loads((result_dir / "preflight.json").read_text())
    frozen = json.loads(archived.read_text())
    ok = (fresh["admission"] == "DATA_UNQUALIFIED" and fresh["coverage"] == frozen["coverage"]
          and fresh["input_sha256"] == frozen["input_sha256"]
          and all(row["compiled_state_matches_wl84"] and row["passed"] for row in fresh["response"].values())
          and all(row["passed"] for row in fresh["scenes"].values()))
    if not ok:
        raise StopRun("baseline", "BASELINE_REPRODUCTION_FAILED")


def manifest(out, source, rows, commit):
    executed = {Path(m.__file__).resolve() for m in tuple(sys.modules.values()) if getattr(m, "__file__", None)}
    paths = sorted(p for p in executed if p.is_relative_to(HERE.parent))
    configs = [HERE / "protocol.json", HERE.parent / "fnto_compiled82/protocol/compiled82.json",
               HERE.parent / "fnto_compiled81/protocol/compiled81.json",
               HERE.parent / "ecro_wl84_104/protocol/wl104.json",
               HERE.parent / "ecro_wl84_104b/protocol/wl104b.json"]
    files = {"data/materials83.npz"}
    for name in {row["set"] for row in rows}:
        files.add(f"data/fit82_{name}.npz")
        for suffix in ("_ER_cert_metric_12000_generic", "_L_ER_s8301"):
            files.add(f"fit/fit_{name}{suffix}.npz")
    refs = [H.C.SPHERE_PLY, H.C.QUAD_PLY]
    for row in rows:
        if row["cohort"] == "development":
            refs.extend(H.C.ROOT / "evidence/refs" / row["image"] / f"LAY65_h{i}.npz" for i in (0, 1))
    out_files = sorted(path for path in out.rglob("*") if path.is_file() and path.name != "provenance.json")
    write(out / "provenance.json", {
        "commit": commit, "base_commit": P["base_commit"],
        "code_and_configuration_sha256": {str(path.relative_to(REPO)): H.C.sha(path) for path in (*paths, *configs)},
        "source_sha256": {name: H.C.sha(source / name) for name in sorted(files)},
        "scene_inputs_sha256": {str(path.relative_to(REPO)): H.C.sha(path) for path in refs},
        "output_sha256": {str(path.relative_to(out)): H.C.sha(path) for path in out_files},
        "environment": {
            "host": socket.gethostname(), "platform": platform.platform(), "python": platform.python_version(),
            "numpy": np.__version__, "torch": torch.__version__, "cuda": torch.version.cuda,
            "mitsuba": H.mi.__version__, "drjit": H.dr.__version__, "variant": H.mi.variant(),
            "gpu": subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                                            "--format=csv,noheader"], text=True).strip(),
        },
        "protocol": P,
    })


def run(out, source, commit, rows, descriptors, status):
    def mark(stage):
        status["latest_stage"] = stage
        write(out / "status.json", status)
        print("stage", stage, flush=True)

    mark("focused_tests")
    run_tests()
    mark("baseline")
    baseline(out, commit)
    mark("training_support")
    original = load_original(source, rows)
    np.savez(out / "train_original.npz", **original)
    covered, support = coverage(descriptors, original, out)
    if not support["passed"]:
        raise StopRun("training_support", "DATA_UNQUALIFIED")
    mark("matched_fit")
    states, summary = fit(original, covered, out)
    if not summary["converged"]:
        raise StopRun("matched_fit", "OPTIMIZATION_UNQUALIFIED")
    states.update(historical_states(source, rows))
    mark("independent_response")
    reference = response_reference(descriptors, out)
    response, compiled = record_response(reference, states, out)
    if not response["reference_precision_passed"]:
        raise StopRun("independent_response", "REFERENCE_UNQUALIFIED")
    if not response["physics_passed"]:
        raise StopRun("independent_response", "PHYSICAL_CONTRACT_FAILED")
    mark("scene_evaluation")
    if H.RT.P.D != 6:
        raise StopRun("scene_evaluation", "TRACER_CONTRACT_FAILED")
    del reference, original, covered
    torch.cuda.empty_cache()
    scenes = {}
    for i, row in enumerate(rows):
        scenes[row["image"]] = render_case(row, descriptors[i], {key: data[i] for key, data in compiled.items()}, i, out)
        write(out / "scenes.json", scenes)
    if not all(row["precision_passed"] for row in scenes.values()):
        raise StopRun("scene_evaluation", "SCENE_PRECISION_UNQUALIFIED")
    mark("decision")
    result = classify(response, scenes, rows)
    write(out / "decision.json", result)
    return result["verdict"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Frozen WL83 result directory")
    parser.add_argument("output", type=Path)
    parser.add_argument("--commit", required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    rows, descriptors = population(args.source)
    write(out / "materials.json", rows)
    np.save(out / "descriptors.npy", descriptors)
    start = time.monotonic()
    status = {"commit": args.commit, "protocol_sha256": H.C.sha(HERE / "protocol.json"), "admission": "RUNNING"}
    try:
        status["verdict"] = run(out, args.source, args.commit, rows, descriptors, status)
        status["admission"] = "QUALIFIED"
    except StopRun as exc:
        status.update({"admission": exc.status, "latest_stage": exc.stage, "verdict": "INCONCLUSIVE"})
    except Exception as exc:
        status.update({"admission": "EXECUTION_ERROR", "error": str(exc)})
        write(out / "status.json", status)
        raise
    status["seconds"] = time.monotonic() - start
    write(out / "status.json", status)
    manifest(out, args.source, rows, args.commit)
    print("WL104D", status, flush=True)


if __name__ == "__main__":
    main()
