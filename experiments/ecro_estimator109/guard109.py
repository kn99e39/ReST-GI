"""WL109 Amendment 2: GPU guard for every WL109 stage that touches the GPU. This process imports no GPU library.

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


def gpu_pids():
    """Compute-process PIDs on the GPU, or None when nvidia-smi fails (treated as busy)."""
    try:
        r = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if r.returncode != 0:
        return None
    return {int(p) for p in r.stdout.split() if p.strip().isdigit()}


def tmux_sessions():
    try:
        r = subprocess.run(["tmux", "ls", "-F", "#{session_name}"], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return []
    return [s for s in r.stdout.split() if s] if r.returncode == 0 else []


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
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    cmd = a.cmd[1:] if a.cmd[:1] == ["--"] else a.cmd
    assert cmd, "no command"
    t_start = time.time()

    def event(kind, **kw):
        rec = {"t": time.time(), "iso": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "name": a.name, "event": kind, **kw}
        with open(a.log, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(rec["iso"], a.name, kind, {k: v for k, v in kw.items()}, flush=True)

    def wait_idle():
        free, last_reason, last_report = 0.0, None, 0.0
        while free < a.free:
            if time.time() - t_start > a.deadline:
                event("deadline")
                sys.exit(125)
            tm, gp = tmux_sessions(), gpu_pids()
            reason = {"tmux": tm} if tm else ({"gpu": "nvidia-smi failed"} if gp is None else ({"gpu": sorted(gp)} if gp else None))
            if reason:
                free = 0.0
                if reason != last_reason or time.time() - last_report > 1800:
                    event("waiting", **reason)
                    last_reason, last_report = reason, time.time()
            else:
                free += a.idle_poll
            time.sleep(a.idle_poll)
        event("idle", free_s=a.free)

    attempt = 0
    while True:
        wait_idle()
        attempt += 1
        if a.before:
            subprocess.run(["bash", "-c", a.before], check=True)
        child = subprocess.Popen(cmd, start_new_session=True)
        event("start", attempt=attempt, pid=child.pid, cmd=cmd)
        t0, preempt, timed_out = time.time(), None, False
        while child.poll() is None:
            time.sleep(a.poll)
            if a.attempt_timeout and time.time() - t0 > a.attempt_timeout:
                timed_out = True
                break
            gp = gpu_pids()
            if gp is None or child.poll() is not None:
                continue
            foreign = gp - descendants(child.pid)
            if foreign:
                preempt = sorted(foreign)
                break
        if preempt is not None or timed_out:
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
            event("preempted", attempt=attempt, foreign_pids=preempt, elapsed_s=time.time() - t0, child_rc=child.returncode)
            continue
        event("exit", attempt=attempt, rc=child.returncode, elapsed_s=time.time() - t0)
        sys.exit(child.returncode)


if __name__ == "__main__":
    main()
