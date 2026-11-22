"""WL109 Amendment 2: Stage 3 timing, one evaluation view per process, then a CPU-only merge into the capture manifest.

python timing109.py RUN_DIR DEV_JSON VIEW      timing of one view -> RUN_DIR/timing_parts/VIEW.json  (GPU; run under guard109)
python timing109.py RUN_DIR DEV_JSON --merge   all six parts -> RUN_DIR/manifest.json (timing, table_sha_after, complete)

The per-view timing block is run109.py's Stage 3 block unchanged in content: the same slots, seeds (seed_timing with the
view's EVAL_VIEWS index), warm-up, rounds, frozen randomized order, end-to-end composition, first-call and watermark, and
the same exclusive_block contamination rule. The only differences are execution ones: each view runs in a fresh process
after its own admission, which a guard kills and restarts if a foreign GPU process appears, and the record is merged later.
"""
import argparse
import os
from pathlib import Path
import shutil
import time

import numpy as np

from common109 import P, EVAL_VIEWS, GRID, NOMINAL, apply_smoke, read, seed_timing, sha, write, HERE
import runtime109 as RT


def view_timing(run, dev_json, view):
    from gpu109 import Engine, timing_schedule
    man = read(run / "manifest.json")
    smoke = man["smoke"]
    if smoke:
        apply_smoke()
    assert not man["complete"], "manifest already complete"
    assert man["protocol_sha256"] == sha(HERE / "protocol109.json") and man["dev_json_sha256"] == sha(dev_json)
    M = int(man["selected_M"])
    assert M == int(read(dev_json)["selected_M"])
    cfg = P["final"]
    vi = EVAL_VIEWS.index(view)
    parts = run / "timing_parts"
    parts.mkdir(exist_ok=True)
    logf = open(run / "timing.log", "a")

    def log(*a):
        s = " ".join(str(x) for x in a)
        print(time.strftime("%H:%M:%S"), view, s, flush=True); logf.write(time.strftime("%H:%M:%S ") + view + " " + s + "\n"); logf.flush()
    R, rt, Bd, audit, r105 = RT.admit()
    eng = Engine(R, rt, Bd)
    cams = RT.cameras()
    table_sha = audit["table_bytes_sha256"]
    slots = [("R0", s, 0) for s in GRID] + [("C1", s, M) for s in NOMINAL]

    def block():
        rec = {"first_call_ms": {}, "frame_ms": {}, "e2e_ms": {}, "watermark_bytes": {}}
        eng._frozen.clear()                          # first call = recording + compile, measured per block
        for si, (kind, spp, m) in enumerate(slots):
            key = f"{kind}_{spp}"
            eng.sync()
            t0 = time.perf_counter()
            eng.run_frozen(kind, view, cams[view], spp, seed_timing(vi, si, 0, smoke), m)
            eng.sync()
            rec["first_call_ms"][key] = 1e3 * (time.perf_counter() - t0)
            for w in range(cfg["timing_warmup"]):
                eng.run_frozen(kind, view, cams[view], spp, seed_timing(vi, si, 1 + w, smoke), m)
            rec["watermark_bytes"][key] = eng.watermark(kind, view, cams[view], spp, seed_timing(vi, si, 9, smoke), m)
            rec["frame_ms"][key] = []
            if spp in NOMINAL:
                rec["e2e_ms"][key] = []
        for rr, (kind, spp, m) in timing_schedule(slots, cfg["timing_rounds"], P["seeds"]["timing_order"] + 10 + vi):
            si = slots.index((kind, spp, m))
            key = f"{kind}_{spp}"
            rec["frame_ms"][key].append(eng.timed(kind, view, cams[view], spp, seed_timing(vi, si, 10 + rr, smoke), m))
            if spp in NOMINAL:
                rec["e2e_ms"][key].append(eng.timed(kind, view, cams[view], spp, seed_timing(vi, si, 50 + rr, smoke), m, compose=True))
        return rec
    log("timing start")
    rec, excl = RT.exclusive_block(block, log, require=not smoke)
    rec["exclusivity"] = excl
    rec["table_sha_after"] = RT.table_bytes_sha(Bd)
    assert rec["table_sha_after"] == table_sha
    rec["admission"] = audit
    rec["versions"] = RT.versions(R)
    rec["code_sha256"] = {p.name: sha(p) for p in sorted(HERE.glob("*.py"))}
    rec["pid"] = os.getpid()
    tmp = parts / f".{view}.json.tmp"
    write(tmp, rec)
    os.replace(tmp, parts / f"{view}.json")
    log("timing", {k: round(float(np.median(v)), 3) for k, v in rec["frame_ms"].items()})


def merge(run, dev_json):
    man = read(run / "manifest.json")
    assert not man["complete"], "manifest already complete"
    assert man["protocol_sha256"] == sha(HERE / "protocol109.json") and man["dev_json_sha256"] == sha(dev_json)
    backup = run / "manifest_capture.json"
    if not backup.exists():
        shutil.copy2(run / "manifest.json", backup)
    table_sha = man["admission"][0]["table_bytes_sha256"]
    timing, parts = {}, {}
    for view in EVAL_VIEWS:
        p = run / "timing_parts" / f"{view}.json"
        rec = read(p)
        parts[view] = sha(p)
        assert rec["table_sha_after"] == table_sha == rec["admission"]["table_bytes_sha256"], view
        man["admission"].append(rec.pop("admission"))
        man["versions"].append(rec.pop("versions"))
        timing[view] = rec
    man["timing"] = timing
    man["table_sha_after"] = table_sha
    man["amendment2"] = {"capture_manifest_sha256": sha(backup), "timing_part_sha256": parts,
                         "code_sha256": {p.name: sha(p) for p in sorted(HERE.glob("*.py"))},
                         "note": "Stage 3 timing run per view in fresh guarded processes (WL109 Amendment 2)"}
    man["complete"] = True
    write(run / "manifest.json", man)
    print("MERGE COMPLETE", {v: round(float(np.median(timing[v]["frame_ms"]["R0_4"])), 3) for v in EVAL_VIEWS}, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("dev_json")
    ap.add_argument("view", nargs="?")
    ap.add_argument("--merge", action="store_true")
    a = ap.parse_args()
    run, dev = Path(a.run).resolve(), Path(a.dev_json).resolve()
    if a.merge:
        merge(run, dev)
    else:
        assert a.view in EVAL_VIEWS, a.view
        view_timing(run, dev, a.view)
