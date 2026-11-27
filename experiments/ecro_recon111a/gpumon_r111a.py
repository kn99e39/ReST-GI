"""WL111A host GPU-activity monitor (owner request 2026-11-27; imports no GPU library, holds no device memory).

Every POLL seconds it lists compute processes on one GPU (nvidia-smi) and classifies them: own = host PIDs of running
containers named recon111a_* (docker top); foreign = every other PID, counted only if seen in two consecutive polls and still
present in /proc (as guard_r111 execution fix 1). It atomically rewrites STATE_JSON with
  {"t": now, "uuid": ..., "started": monitor start, "last_foreign": last time a foreign PID was present, "foreign_now": [...]}
The monitor's start time counts as foreign activity (history before it is unknown), so the first fast start needs FREE s
of observation. guard_r111a reads this file to skip its own idle wait when the GPU has already been free long enough.

python gpumon_r111a.py UUID STATE_JSON [--poll 2]
"""
import argparse
import json
import os
import subprocess
import time


def compute_pids(uuid):
    r = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid", "--format=csv,noheader,nounits"],
                       capture_output=True, text=True, timeout=60)
    if r.returncode != 0:
        return None
    out = set()
    for line in r.stdout.splitlines():
        f = [x.strip() for x in line.split(",")]
        if len(f) >= 2 and f[0].isdigit() and f[1] == uuid:
            out.add(int(f[0]))
    return out


def own_pids():
    r = subprocess.run(["docker", "ps", "--format", "{{.Names}}"], capture_output=True, text=True, timeout=30)
    pids = set()
    for name in r.stdout.split():
        if name.startswith("recon111a_"):
            t = subprocess.run(["docker", "top", name, "-eo", "pid"], capture_output=True, text=True, timeout=30)
            pids |= {int(p) for p in t.stdout.split() if p.strip().isdigit()}
    return pids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("uuid")
    ap.add_argument("state")
    ap.add_argument("--poll", type=float, default=2)
    a = ap.parse_args()
    started = last_foreign = time.time()
    prev = set()
    while True:
        try:
            gp = compute_pids(a.uuid)
        except (OSError, subprocess.TimeoutExpired):
            gp = None
        now = time.time()
        if gp is None:
            foreign = {"nvidia-smi failed"}
            last_foreign = now
        else:
            cand = {p for p in gp - own_pids() if os.path.exists(f"/proc/{p}")}
            foreign = cand & prev
            prev = cand
            if foreign:
                last_foreign = now
        tmp = a.state + ".tmp"
        with open(tmp, "w") as f:
            json.dump({"t": now, "uuid": a.uuid, "started": started, "last_foreign": last_foreign,
                       "foreign_now": sorted(str(x) for x in foreign)}, f)
        os.replace(tmp, a.state)
        time.sleep(a.poll)


if __name__ == "__main__":
    main()
