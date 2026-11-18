"""Independent 4096-spp raw ECRO renders using the unchanged WL101 runtime."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent/"ecro_sss101a"))
from common import P as A, fingerprint, rel_l1, sha, write_json
from run import gpu_query, foreign_gpu, validate_inputs, wait_for_gpu

PROTOCOL = HERE/"protocol/contact.json"
P = json.loads(PROTOCOL.read_text())


def main(root, output, commit):
    assert len(commit) == 40
    ref = root/A["references"]["path_in_wl101"]
    original_sources = validate_inputs(root, ref)
    assert [part.strip() for part in gpu_query("gpu", "name,driver_version").split(",")] == [
        "NVIDIA GeForce RTX 3080 Ti", "595.71.05"]
    seeds = P["seeds"]
    assert len(seeds) == len(set(seeds)) == 8
    assert not set(seeds) & (set(A["references"]["seed_by_half"]) | {101500}
                              | {10160000 + 100*spp + r for spp in A["sweep"]["spp"] for r in range(16)})
    wait_for_gpu()
    sys.path.insert(0, str(root/"experiments/ecro_dyn101"))
    os.environ["FREEZE"] = "1"
    sys.dont_write_bytecode = True
    import rt101 as runtime
    import seq101 as seq

    assert (runtime.mi.__version__, runtime.dr.__version__, runtime.torch.__version__) == (
        "3.8.0", "1.3.1", "2.3.1+cu121")
    assert runtime.FREEZE and runtime.mi.variant() == "cuda_ad_rgb"
    camera = dict(runtime.C100.P["datasets"]["sss"]["camera"], res=512)
    assert camera == A["camera"]
    workload, start, events = seq.workload(A["workload"])
    assert workload == "sss" and events[seq.WARM] == (A["geometry"], A["light"], 0)
    scene_hash = fingerprint({k:A[k] for k in ("geometry","light","camera","sphere")})
    assert scene_hash == P["scene_sha256"]
    rt = runtime.RT(start[0], "sss", runtime.load_model("sss"), start[1])
    active_optix = next(Path(line.split()[-1]) for line in Path("/proc/self/maps").read_text().splitlines()
                        if "/libnvoptix.so." in line)
    assert active_optix.name == "libnvoptix.so.595.71.05"
    for geom, light, _ in events[:seq.WARM+1]:
        moved, old = rt.set_geom(geom)
        rt.update_state(old, moved)
        rt.set_light(light)
        table, _ = rt.respond(rt.source())
    saved_table = table.detach().cpu().numpy().copy()
    digest = hashlib.sha256(saved_table.tobytes()).hexdigest()
    assert digest == P["table_sha256"]
    with np.load(ref/"PRED_B_0.npz") as historical:
        assert np.array_equal(historical["B"], saved_table)
    resident = runtime.t2d(table.view(-1))
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "commit": commit, "protocol_sha256": sha(PROTOCOL), "scene_sha256": scene_hash,
        "checkpoint_sha256": A["checkpoint_sha256"], "configs_sha256": A["configs_sha256"],
        "table_sha256": digest, "table_bit_identical_to_archive": True,
        "archived_inputs_sha256": A["references"]["sha256"], "source_sha256": original_sources,
        "geometry": A["geometry"], "light": A["light"], "camera": camera,
        "render_path": "rt101.render_hq -> original RT._frame_prog -> original pt100.Ctx.trace (ecro)",
        "postprocessing": "None; pack original per-event sums and compose linear HDR.",
        "environment": {"gpu": runtime.torch.cuda.get_device_name(), "driver": "595.71.05",
                        "optix": str(active_optix), "optix_sha256": sha(active_optix), "os": platform.platform(),
                        "python": platform.python_version(), "mitsuba": runtime.mi.__version__,
                        "drjit": runtime.dr.__version__, "torch": runtime.torch.__version__, "numpy":np.__version__},
        "frames": [],
    }
    spp, batch = P["spp"], P["launch_spp"]
    print("Identity validated. Rendering original ECRO at 4096 spp.", flush=True)
    for seed in seeds:
        wait_for_gpu()
        runtime.sync()
        start_time = time.perf_counter()
        accumulator = runtime.render_hq(rt, resident, camera, 512, spp, seed, b=batch)
        runtime.sync()
        elapsed = time.perf_counter()-start_time
        raw = (accumulator.reshape(3,3,512,512).transpose(2,3,1,0)/spp)
        assert np.isfinite(raw).all() and raw.min() >= 0
        e1, e2, hl = np.moveaxis(raw.astype(np.float32), -1, 0)
        low = (e1.astype(np.float64)+e2).astype(np.float32)
        full = np.sum(raw, axis=-1).astype(np.float32)
        relative_error = rel_l1((low.astype(np.float64)+hl).astype(np.float32), full)
        assert relative_error <= 1e-7
        path = output/f"raw_4096_seed{seed}.npz"
        np.savez_compressed(path, e1=e1, e2=e2, low=low, hl=hl, full=full)
        manifest["frames"].append({"file":path.name, "sha256":sha(path), "bytes":path.stat().st_size,
                                   "seed":seed, "spp":spp, "launch_spp":batch, "seconds":elapsed,
                                   "recomposition_rel_l1":relative_error,
                                   "foreign_gpu_pids_after":foreign_gpu()})
        write_json(output/"run_progress.json", manifest)
        print(f"{seed}: {elapsed:.3f}s", flush=True)
    assert np.array_equal(saved_table, table.detach().cpu().numpy())
    manifest["table_preserved_after_rendering"] = True
    write_json(output/"run_manifest.json", manifest)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wl101", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("commit")
    a = parser.parse_args()
    main(a.wl101.resolve(), a.out.resolve(), a.commit)
