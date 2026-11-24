"""WL109B Amendment 2 (owner decision 2026-11-24): LabServer70 admission for the frozen WL101 renderer.

Copy of experiments/ecro_denoise108/renderer108.py::frozen_runtime with ONLY its environment pins changed. The frozen WL108
admission pins LabServer69 (RTX 3090, driver 550.54.15, libnvoptix.so.550.54.15, host label). On LabServer70 (RTX 3090 at
GPU 0 next to an RTX 3080 Ti, driver 525.60.13):
  - the GPU row is selected by the UUID in CUDA_VISIBLE_DEVICES (the frozen single-row parse fails with two GPUs); the GPU
    name must still be "NVIDIA GeForce RTX 3090";
  - the driver must equal LabServer70's declared driver 525.60.13, and the loaded OptiX library must match that driver;
  - the host label and the OptiX sha256 are recorded as found.
Every substantive check is unchanged and executed: baseline / prior manifests, renderer source hashes, Mitsuba / Dr.Jit /
Torch / NumPy / Python versions, workload / camera / scene fingerprint, learned HIGH table hash, and the archived 4096-spp V0
replay within the WL105 tolerance. The frozen file itself is not edited; common109b installs this function in place of
renderer108.frozen_runtime only when WL109B_ADMISSION=lab70.
"""
import os
from pathlib import Path
import platform
import socket
import subprocess
import sys

import numpy as np

from common108 import read

LAB70_GPU = "NVIDIA GeForce RTX 3090"
LAB70_DRIVER = "525.60.13"


def _gpu_row():
    uuid = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    out = subprocess.check_output(["nvidia-smi", "--query-gpu=uuid,name,driver_version", "--format=csv,noheader,nounits"],
                                  text=True, timeout=15).strip().splitlines()
    rows = [[x.strip() for x in r.split(",")] for r in out]
    sel = [r for r in rows if r[0] == uuid]
    assert len(sel) == 1, ("CUDA_VISIBLE_DEVICES must name exactly one GPU UUID", uuid, rows)
    return sel[0][1], sel[0][2], uuid


def frozen_runtime(r, root, *, radiance):
    """Validate legacy source bytes; initialize the same scene on the pinned LabServer70 RTX 3090."""
    f = r.RG.F101
    reference = root / f.A["references"]["path_in_wl101"]
    baseline = root.parent / "results/ecro_sss101e/recovery2"
    prior = root.parent / "results/ecro_sss101a/qualified63"
    assert f.sha(baseline / "run_manifest.json") == f.P["baseline_manifest_sha256"]
    anchor = read(baseline / "run_manifest.json")
    assert f.sha(prior / "run_manifest.json") == anchor["prior_manifest_sha256"]
    sources = f.validate_inputs(root, reference)
    assert sources == anchor["renderer_sources_sha256"]
    f.wait_for_gpu()
    gpu, driver, uuid = _gpu_row()                                         # Amendment 2: was a single-row parse
    assert (gpu, driver) == (LAB70_GPU, LAB70_DRIVER), (gpu, driver)        # Amendment 2: was ("...3090", "550.54.15")
    sys.path.insert(0, str(root/"experiments/ecro_dyn101"))
    os.environ["FREEZE"] = "1"
    import rt101 as R
    import seq101
    assert (R.mi.__version__, R.dr.__version__, R.torch.__version__, np.__version__, platform.python_version()) == (
        "3.8.0", "1.3.1", "2.3.1+cu121", "1.26.4", "3.10.21")
    case, initial, events = seq101.workload(f.A["workload"])
    camera = dict(R.C100.P["datasets"][case]["camera"],res=512)
    assert camera == f.A["camera"] and events[seq101.WARM] == (f.A["geometry"], f.A["light"], 0)
    rt = R.RT(initial[0], case, R.load_model(case), initial[1])
    for geo, light, _ in events[:seq101.WARM+1]:
        moved, old = rt.set_geom(geo)
        rt.update_state(old, moved)
        rt.set_light(light)
        if radiance:
            learned, _ = rt.respond(rt.source())
    assert f.fingerprint({k:f.A[k] for k in ("geometry","camera","sphere","light")}) == anchor["scene_sha256"]
    library = next(Path(row.split()[-1]) for row in Path("/proc/self/maps").read_text().splitlines()
                   if "/libnvoptix.so." in row)
    assert library.name == f"libnvoptix.so.{LAB70_DRIVER}"                 # Amendment 2: was libnvoptix.so.550.54.15
    audit = {k:anchor[k] for k in ("scene_sha256","checkpoint_sha256","cfg_sha256",
                                  "learned_table_sha256","input_sha256","renderer_sources_sha256")}
    audit["environment"] = {"host": f"LabServer70 ({socket.gethostname()})", "gpu": gpu, "gpu_uuid": uuid, "driver": driver,
                            "optix_sha256": f.sha(library), "mitsuba": R.mi.__version__, "drjit": R.dr.__version__,
                            "torch": R.torch.__version__, "python": platform.python_version(), "numpy": np.__version__,
                            "admission": "WL109B Amendment 2 (experiments/ecro_estimator109b/admission70.py)"}
    if not radiance:
        return R, rt, None, audit

    learned = learned.cpu().numpy().copy()
    assert f.array_hash(learned) == anchor["learned_table_sha256"], "historical HIGH table replay failed on LabServer70"
    worker = R.PT.fl(learned)
    history = read(prior/"run_manifest.json")
    errors = []
    for row in history["matched"]:
        raw = prior / row["path"]
        assert f.sha(raw) == row["sha256"]
        component = r.frame_components(R,rt,worker,r.CAM.V0,4096,row["seed"])
        with np.load(raw) as z:
            expected = np.concatenate([z[k].reshape(512,512,3).transpose(2,0,1) for k in ("e1","e2","hl")])
        for i in range(0,9,3):
            errors.append(float(np.abs(component[i:i+3]-expected[i:i+3]).sum()/np.abs(expected[i:i+3]).sum()))
    assert max(errors) <= r.P["gates"]["replay_relative_l1"], errors
    audit["archive_replay_relative_l1"] = errors
    return R, rt, worker, audit
