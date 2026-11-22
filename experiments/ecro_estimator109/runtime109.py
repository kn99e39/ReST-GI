"""WL109 admission of the frozen WL101 ECRO renderer on LabServer69 (reuses the WL108 admission unchanged) and frozen inputs.

frozen_runtime (experiments/ecro_denoise108/renderer108.py) validates the WL101 renderer sources, checkpoint, configuration and
scene, rebuilds the frozen SSS state through the original lifecycle, checks the learned HIGH table hash and replays the archived
4096-spp V0 ECRO predictions within the WL105 tolerance. Nothing in it is modified here.
"""
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np

from common109 import P, HERE, array_sha, read, sha

HOME = Path(os.path.expanduser("~"))
BASE = HOME / P["environment"]["remote_root"]
INPUTS = BASE / P["inputs"]["frozen_inputs"]
WL101 = INPUTS / "wl101"
D105 = INPUTS / "results/ecro_denoise105/data"
REF107 = INPUTS / "results/ecro_denoise107/reference"
CAPTURE107 = BASE / P["inputs"]["wl107_capture"]


def foreign_gpu_apps():
    out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, timeout=30).stdout.split()
    return [int(p) for p in out if p.strip() and int(p) != os.getpid()]


def gpu_snapshot():
    q = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,clocks.sm,temperature.gpu",
                        "--format=csv,noheader"], capture_output=True, text=True, timeout=30).stdout.strip()
    return {"query": q, "foreign_compute_pids": foreign_gpu_apps(), "t": time.time()}


def wait_free_gpu(log=print, poll=30):
    while foreign_gpu_apps():
        log("waiting for foreign GPU process", foreign_gpu_apps())
        time.sleep(poll)


def admit():
    """-> R, rt, Bd (frozen learned table as a Dr.Jit array), audit, r (render105).
    GPU policy: renderer outputs are deterministic functions of the seeds, so data stages do not need an exclusive GPU;
    the admission's exclusive-GPU wait is therefore replaced by a logged snapshot, and every timing block waits for an
    exclusive GPU itself (exclusive_block)."""
    argv = sys.argv[:]
    import render105 as r          # noqa: E402  (render105 rewrites sys.argv at import)
    sys.argv = argv
    from renderer108 import frozen_runtime
    snap = gpu_snapshot()
    r.RG.F101.wait_for_gpu = lambda *a, **k: None
    R, rt, Bd, audit = frozen_runtime(r, WL101, radiance=True)
    audit["admission_gpu_snapshot"] = snap
    table = np.asarray(Bd, dtype=np.float32)
    audit["table_bytes_sha256"] = array_sha(table)
    audit["table_shape"] = [int(rt.N), int(R.PT.NB), 3]
    audit["admission_source_sha256"] = {"renderer108.py": sha(HERE.parent / "ecro_denoise108/renderer108.py"),
                                        "render105.py": sha(HERE.parent / "ecro_denoise105/render105.py")}
    return R, rt, Bd, audit, r


def table_bytes_sha(Bd):
    return array_sha(np.asarray(Bd, dtype=np.float32))


def cameras():
    m = read(D105 / "run_manifest.json")
    views = m["views"]
    out = {}
    for name in P["views"]["eval"]:
        out[name] = dict(views[f"eval/{name}"]["camera"])
    for name in P["views"]["dev"]:
        out[name] = dict(views[f"train/{name}"]["camera"])
    return out


def aux(name):
    split = "eval" if name in P["views"]["eval"] else "train"
    path = D105 / split / name / "aux.npz"
    raw = read(INPUTS / "evaluation/105/raw_manifest105.json")["raw_files"]
    assert sha(path) == raw[f"data/{split}/{name}/aux.npz"], ("aux hash", name)
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def archived_noisy(name, spp, r):
    path = D105 / "eval" / name / f"noisy_spp{spp}_r{r}.npz"
    raw = read(INPUTS / "evaluation/105/raw_manifest105.json")["raw_files"]
    assert sha(path) == raw[f"data/eval/{name}/noisy_spp{spp}_r{r}.npz"], ("noisy hash", name)
    with np.load(path) as z:
        return z["comp"].astype(np.float32), int(z["seed"])


def reference(name):
    """WL107 qualified frozen-ECRO assessment reference (48 x 8192 spp) -> (mean components [9,H,W] float64,
    per-pixel RGB variance of the mean [3,H,W] float64, record). Blocks: 46 WL107 assessment + 2 WL105 halves."""
    manifest = REF107 / "reference107.json"
    assert sha(manifest) == P["inputs"]["wl107_reference_manifest_sha256"]
    rec = read(manifest)["views"][name]
    target = REF107 / f"{name}_target.npy"
    assert sha(target) == rec["target_sha256"]
    cap = CAPTURE107 / "capture_manifest.json"
    assert sha(cap) == P["inputs"]["wl107_capture_manifest_sha256"]
    files = read(cap)["files"]
    raw = read(INPUTS / "evaluation/105/raw_manifest105.json")["raw_files"]
    blocks = []
    for h in range(2):
        p = D105 / "eval" / name / f"target_half{h}.npz"
        assert sha(p) == raw[f"data/eval/{name}/target_half{h}.npz"]
        with np.load(p) as z:
            blocks.append(z["comp"].astype(np.float64))
    for h in range(46):
        key = f"{name}/assessment_{h:02d}.npy"
        p = CAPTURE107 / key
        assert sha(p) == files[key]["sha256"], key
        blocks.append(np.load(p, allow_pickle=False).astype(np.float64))
    comp = np.stack(blocks)                                  # [48, 9, H, W]
    mean = comp.mean(0)
    rgb = comp[:, 0:3] + comp[:, 3:6] + comp[:, 6:9]
    var_mean = rgb.var(0, ddof=1) / comp.shape[0]
    stored = np.load(target).astype(np.float64)
    assert np.allclose(stored, mean, rtol=2e-6, atol=1e-7), float(np.abs(stored - mean).max())
    return mean, var_mean, {"blocks": int(comp.shape[0]), "target_sha256": rec["target_sha256"],
                            "stored_vs_recomputed_max_abs": float(np.abs(stored - mean).max())}


def exclusive_block(fn, log=print, tries=6, require=True):
    """Run a timing block on an exclusive GPU: wait until no foreign compute process exists, run, and redo the block if a
    foreign process appeared meanwhile. Returns (result, record). require=False (code-path smoke only): run once, record."""
    rec = []
    if not require:
        before = gpu_snapshot(); out = fn(); after = gpu_snapshot()
        return out, [{"before": before, "after": after, "clean": not before["foreign_compute_pids"] and not after["foreign_compute_pids"],
                      "smoke_not_required": True}]
    for k in range(tries):
        wait_free_gpu(log)
        before = gpu_snapshot()
        out = fn()
        after = gpu_snapshot()
        clean = not before["foreign_compute_pids"] and not after["foreign_compute_pids"]
        rec.append({"before": before, "after": after, "clean": clean})
        if clean:
            return out, rec
        log("timing block contaminated by a foreign GPU process; repeating")
    raise RuntimeError("no uncontended timing block after %d tries" % tries)


def versions(R):
    import torch
    return {"mitsuba": R.mi.__version__, "drjit": R.dr.__version__, "torch": torch.__version__, "numpy": np.__version__,
            "python": platform.python_version(), "gpu": torch.cuda.get_device_name(0), "variant": R.mi.variant(),
            "freeze": R.FREEZE, "host": platform.node()}
