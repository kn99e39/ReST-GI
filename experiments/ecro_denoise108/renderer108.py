"""WL108 experiment-local LabServer69 admission for the frozen WL101 renderer."""
import os
from pathlib import Path
import platform
import sys

import numpy as np

from common108 import read


def frozen_runtime(r, root, *, radiance):
    """Validate legacy source bytes; initialize the same scene on the requested GPU."""
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
    gpu, driver = (x.strip() for x in f.gpu_query("gpu","name,driver_version").split(","))
    assert (gpu, driver) == ("NVIDIA GeForce RTX 3090", "550.54.15"), (gpu,driver)
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
    assert library.name == "libnvoptix.so.550.54.15"
    audit = {k:anchor[k] for k in ("scene_sha256","checkpoint_sha256","cfg_sha256",
                                  "learned_table_sha256","input_sha256","renderer_sources_sha256")}
    audit["environment"] = {"host":"LabServer69","gpu":gpu,"driver":driver,"optix_sha256":f.sha(library),
                            "mitsuba":R.mi.__version__,"drjit":R.dr.__version__,"torch":R.torch.__version__,
                            "python":platform.python_version(),"numpy":np.__version__}
    if not radiance:
        return R, rt, None, audit

    learned = learned.cpu().numpy().copy()
    assert f.array_hash(learned) == anchor["learned_table_sha256"], "LabServer69 historical HIGH table replay failed"
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
