"""WL111A copy of guard_r111.py with ONE addition (owner request 2026-11-27): --idle-file STATE_JSON (written by
gpumon_r111a.py). Before its own idle wait, the guard reads the monitor state; if the monitor is alive (state updated within
10 s), watches the same GPU, no foreign process is present now, the last foreign activity is at least --free seconds old,
and no tmux session exists, the start condition is already met and the guard starts at once (event "idle_fast"). This
covers a step that directly follows this batch's own previous step, and a GPU that was already idle before the guard
started. Otherwise the original 120 s wait applies unchanged. Everything else is guard_r111.py.

WL111 execution fix 1 (2026-11-24): copy of the WL109B guard (experiments/ecro_estimator109b/guard109c.py, sha256
f7d6072b...) with ONE change: during a run, a compute PID counts as foreign only if it is seen in two consecutive polls
and still exists in /proc. Reason: WL111 runs several processes one after another in one container; at a stage
transition the exiting process can still be listed by nvidia-smi after it has left the container's process table, and
the unchanged guard preempted the chain on its own exiting process (first evidence launch, before any evidence).
Foreign work is still detected within two polls. The start condition (120 s without any compute process) is unchanged.
The original docstring follows.

WL109B Amendment 3: container-aware copy of the WL109 GPU guard (experiments/ecro_estimator109/guard109.py, unchanged
otherwise). Used on LabServer70, where the frozen renderer runs inside a Docker container (ctr109b.sh).

Changes from guard109.py:
  --container NAME  the child runs `docker run --name NAME ...`; GPU processes of that container (host PIDs from
                    `docker top NAME`) count as the guard's own. A stale container of that name is removed before every
                    attempt; on preemption or timeout the container is killed (`docker kill NAME`) before the child's
                    process group, which releases all of its device memory.
  --gpu-uuid U      (Amendment 5 addendum) only compute processes on GPU U count as foreign, for the start condition and
                    for preemption; default: every GPU on the host, as in guard109.py. Data stages on a multi-GPU host use it;
                    timing-bearing stages keep the whole-host rule (another GPU's work can still perturb measured times).
  --max-load L      the start condition additionally requires the host 1-minute load average <= L throughout the
                    `--free` window (protects timing stages from CPU contention and leaves CPU to other users); the load
                    average is recorded at every start / exit / preemption.
The original guard docstring follows.

WL109 Amendment 2: GPU guard for every WL109 stage that touches the GPU. This process imports no GPU library.

python guard109.py --name NAME --log GUARD.jsonl [--free 120] [--poll 2] [--attempt-timeout S] [--deadline S]
                   [--before 'shell snippet'] -- CMD ARGS...

Start condition, checked in the guard process, which holds no device memory:
    no tmux session exists for this user (TransGI and other foreign chains run in tmux; WL109 does not), and
    nvidia-smi has listed no compute process for `--free` consecutive seconds.
While the child runs, nvidia-smi is polled every `--poll` seconds. When any compute process outside the child's process
tree appears, the child's whole process group is terminated at once (SIGTERM, then SIGKILL after 20 s), which releases
all WL109 device memory. The guard then waits for the start condition again and restarts the command from the beginning.
WL109 stage outputs depend only on the seeds, so a restarted command reproduces the same result.
`--before` runs (bash) before every attempt, e.g. to move aside a partial output directory.
Exit code: the child's exit code; 124 when one attempt exceeds --attempt-timeout; 125 when --deadline passes.
"""
import argparse
import json
import os
import signal
import subprocess
import sys
import time


GPU_UUID = ""     # Amendment 5 addendum: when set (--gpu-uuid), only compute processes on that GPU count


def gpu_pids():
    """Compute-process PIDs on the GPU (all GPUs, or only GPU_UUID), or None when nvidia-smi fails (treated as busy)."""
    try:
        r = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if r.returncode != 0:
        return None
    out = set()
    for line in r.stdout.splitlines():
        f = [x.strip() for x in line.split(",")]
        if len(f) >= 2 and f[0].isdigit() and (not GPU_UUID or f[1] == GPU_UUID):
            out.add(int(f[0]))
    return out


def tmux_sessions():
    try:
        r = subprocess.run(["tmux", "ls", "-F", "#{session_name}"], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return []
    return [s for s in r.stdout.split() if s] if r.returncode == 0 else []


def container_pids(name):
    if not name:
        return set()
    try:
        r = subprocess.run(["docker", "top", name, "-eo", "pid"], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return set()
    return {int(p) for p in r.stdout.split() if p.strip().isdigit()} if r.returncode == 0 else set()


def load1():
    with open("/proc/loadavg") as f:
        return float(f.read().split()[0])


def docker_kill(name):
    if name:
        subprocess.run(["docker", "kill", name], capture_output=True, text=True, timeout=60)
        subprocess.run(["docker", "rm", "-f", name], capture_output=True, text=True, timeout=60)


def descendants(root):
    r = subprocess.run(["ps", "-A", "-o", "pid=,ppid="], capture_output=True, text=True, timeout=30)
    kids = {}
    for line in r.stdout.splitlines():
        f = line.split()
        if len(f) == 2:
            kids.setdefault(int(f[1]), []).append(int(f[0]))
    out, st = {root}, [root]
    while st:
        for c in kids.get(st.pop(), []):
            if c not in out:
                out.add(c)
                st.append(c)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--log", required=True)
    ap.add_argument("--free", type=float, default=120)
    ap.add_argument("--poll", type=float, default=2)
    ap.add_argument("--idle-poll", type=float, default=10)
    ap.add_argument("--attempt-timeout", type=float, default=0)
    ap.add_argument("--deadline", type=float, default=72 * 3600)
    ap.add_argument("--before", default="")
    ap.add_argument("--container", default="")
    ap.add_argument("--max-load", type=float, default=0)
    ap.add_argument("--gpu-uuid", default="", help="count only compute processes on this GPU (default: every GPU)")
    ap.add_argument("--idle-file", default="", help="gpumon_r111a state; skip the idle wait when it already proves --free s idle")
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    cmd = a.cmd[1:] if a.cmd[:1] == ["--"] else a.cmd
    global GPU_UUID
    GPU_UUID = a.gpu_uuid
    assert cmd, "no command"
    t_start = time.time()

    def event(kind, **kw):
        rec = {"t": time.time(), "iso": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "name": a.name, "event": kind, **kw}
        with open(a.log, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(rec["iso"], a.name, kind, {k: v for k, v in kw.items()}, flush=True)

    def fast_idle():
        if not a.idle_file or not a.gpu_uuid:
            return False
        try:
            with open(a.idle_file) as f:
                s = json.load(f)
        except (OSError, ValueError):
            return False
        now = time.time()
        ok = (s.get("uuid") == a.gpu_uuid and now - s["t"] <= 10 and not s["foreign_now"] and now - s["last_foreign"] >= a.free
              and not tmux_sessions() and not (a.max_load and load1() > a.max_load))
        if ok:
            event("idle_fast", idle_s=round(now - s["last_foreign"], 1), load1=load1())
        return ok

    def wait_idle():
        if fast_idle():
            return
        free, last_reason, last_report = 0.0, None, 0.0
        while free < a.free:
            if time.time() - t_start > a.deadline:
                event("deadline")
                sys.exit(125)
            tm, gp = tmux_sessions(), gpu_pids()
            reason = {"tmux": tm} if tm else ({"gpu": "nvidia-smi failed"} if gp is None else ({"gpu": sorted(gp)} if gp else None))
            if reason is None and a.max_load and load1() > a.max_load:
                reason = {"load1_above": a.max_load}
            if reason:
                free = 0.0
                if reason != last_reason or time.time() - last_report > 1800:
                    event("waiting", **reason)
                    last_reason, last_report = reason, time.time()
            else:
                free += a.idle_poll
            time.sleep(a.idle_poll)
        event("idle", free_s=a.free, load1=load1())

    attempt = 0
    while True:
        wait_idle()
        attempt += 1
        if a.before:
            subprocess.run(["bash", "-c", a.before], check=True)
        docker_kill(a.container)
        child = subprocess.Popen(cmd, start_new_session=True)
        event("start", attempt=attempt, pid=child.pid, cmd=cmd, container=a.container, load1=load1())
        t0, preempt, timed_out = time.time(), None, False
        last_foreign = set()
        while child.poll() is None:
            time.sleep(a.poll)
            if a.attempt_timeout and time.time() - t0 > a.attempt_timeout:
                timed_out = True
                break
            gp = gpu_pids()
            if gp is None or child.poll() is not None:
                continue
            foreign = {p for p in gp - descendants(child.pid) - container_pids(a.container) if os.path.exists(f"/proc/{p}")}
            if foreign and foreign & last_foreign:
                preempt = sorted(foreign & last_foreign)
                break
            last_foreign = foreign
        if preempt is not None or timed_out:
            docker_kill(a.container)
            try:
                os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                child.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
            if timed_out:
                event("attempt_timeout", attempt=attempt, elapsed_s=time.time() - t0)
                sys.exit(124)
            event("preempted", attempt=attempt, foreign_pids=preempt, elapsed_s=time.time() - t0, child_rc=child.returncode, load1=load1())
            continue
        event("exit", attempt=attempt, rc=child.returncode, elapsed_s=time.time() - t0, load1=load1())
        sys.exit(child.returncode)


if __name__ == "__main__":
    main()
