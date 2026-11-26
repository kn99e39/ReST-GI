#!/usr/bin/env bash
# WL109B Amendment 8 chain on LabServer70.  usage: run109b_amend8.sh COMMIT
# Amendment 8 (owner decision 2026-11-26 19:12): no GPU guard. The remaining GPU stages (timing, then dynamic if
# STATIC_PASS) start immediately in the same container (ctr109b.sh, RTX 3090 only) without an idle window, load gate or
# preemption. Exclusivity is no longer enforced; it is recorded instead: both GPUs' compute processes and utilization every
# 10 s (evidence/gpu_util_lab70.csv, evidence/gpu_apps_lab70.csv; nvidia-smi queries only), so units that overlapped foreign
# 3090 work can be identified afterwards. Resumes after the completed grid stage. Everything else as run109b_amend7.sh.
set -eu -o pipefail
cd "$HOME/rest_sss101a"
commit=$1
src=experiments/ecro_estimator109b
out="$PWD/results/ecro_estimator109b/evidence"
py="$HOME/NeuralShaderDemo/external/8dna26/.venv/bin/python"
guardpy="$PWD/experiments/ecro_estimator109/guard109.py"
GPU3090=GPU-ab3bc073-096c-6d0b-a304-13cef177a843
mkdir -p "$out/tmp" "$out/mpl"
export DYN101_MI_DIR="$HOME/rest_dyn101/external/mi_3.8.0" DYN101_MI_VERSION=3.8.0
export LD_LIBRARY_PATH="$PWD/results/ecro_denoise108/runtime/lib:${LD_LIBRARY_PATH:-}"
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 WL109B_CPUS=4
export MPLCONFIGDIR="$out/mpl" TMPDIR="$out/tmp" PYTHONDONTWRITEBYTECODE=1
IMAGE=$(docker image inspect python:3.10-bookworm --format '{{.Id}}')
[[ $IMAGE == sha256:6fd6ebbd05fc4cf5dc3a789c47e42f4d787cb9c90fe5d29835875a934680271d ]]
ctr="$PWD/experiments/ecro_estimator109b/ctr109b.sh"
guardc="$PWD/experiments/ecro_estimator109b/guard109c.py"
# guard NAME CAP MAXLOAD [--before ...] -- python-args...  (python runs inside container wl109b_NAME)
# Amendment 8: no guard. guard NAME CAP MAXLOAD [--before ...] -- python-args...  runs the command directly in container
# wl109b_NAME (CAP and MAXLOAD are ignored); start / exit are appended to the guard log for continuity.
guard() { local name=$1; shift 3; if [[ $1 == --before ]]; then bash -c "$2"; shift 2; fi; shift
  docker rm -f "wl109b_$name" >/dev/null 2>&1 || true
  printf '{"iso": "%s", "name": "%s", "event": "start_unguarded", "load1": %s}\n' "$(date +%Y-%m-%dT%H:%M:%S%z)" "$name" "$(cut -d' ' -f1 /proc/loadavg)" >> "$out/guard109b_lab70.jsonl"
  local rc=0; "$ctr" "wl109b_$name" "$py" "$@" || rc=$?
  printf '{"iso": "%s", "name": "%s", "event": "exit_unguarded", "rc": %d, "load1": %s}\n' "$(date +%Y-%m-%dT%H:%M:%S%z)" "$name" "$rc" "$(cut -d' ' -f1 /proc/loadavg)" >> "$out/guard109b_lab70.jsonl"
  return $rc; }
cpu() { CUDA_VISIBLE_DEVICES= nice -n 10 taskset -c 12-15 "$py" "$@"; }
stage=start
state() { printf '%s %s\n' "$(date -Iseconds)" "$*" | tee -a "$out/chain.state"; }
trap 'rc=$?; if (( rc )); then state "STOP stage=$stage exit=$rc (LabServer70)"; fi' EXIT
state "AMENDMENT 8 (no guard) START on $(hostname) (LabServer70) commit=$commit driver=$(cat /proc/driver/nvidia/version | head -1 | grep -o '550[.0-9]*') image=${IMAGE:7:12}"
cd "$src"
jget() { "$py" -c 'import json,sys; d=json.load(open(sys.argv[1]));
for k in sys.argv[2].split("."): d=d[k]
print(d)' "$1" "$2"; }
R=$(jget "$out/calib/budget109b.json" decision.R)
[[ $(jget "$out/calib/budget109b.json" decision.outcome) == PROCEED && $R == 512 ]]

bv=$(jget "$out/metrics/bias109b.json" bias_verdict)
[[ $bv == EQUIVALENT_ALL ]] || { echo "bias verdict $bv" >&2; exit 3; }
[[ $(jget "$out/xcheck/xcheck109b.json" outcome) == PASS ]]
state "resume after bias: BIAS verdict=$bv, XCHECK PASS"

# host GPU-occupancy record for Amendment 7 (nvidia-smi queries only; no CUDA context): both GPUs every 10 s while the chain lives
( chain=$$; while kill -0 "$chain" 2>/dev/null; do
    nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used --format=csv,noheader >> "$out/gpu_util_lab70.csv" 2>/dev/null || true
    { date +%s; nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader; } | paste -sd'|' >> "$out/gpu_apps_lab70.csv" 2>/dev/null || true
    sleep 10; done ) &

stage=timing; state "RUNNING $stage"
guard timing 10800 "${WL109B_TIMING_MAXLOAD:-2}" -- capture109b.py "$out/cap" timing --R "$R" > "$out/timing_stage.log" 2>&1
stage=static; state "RUNNING $stage"
cpu analyze109b.py "$out/cap" "$out/calib/budget109b.json" "$out/metrics" --phase static > "$out/static.log" 2>&1
cpu decide109b.py "$out/metrics/metrics109b.json" "$out/metrics/static109b.json" >> "$out/static.log" 2>&1
sv=$(jget "$out/metrics/static109b.json" final_static_verdict)
state "STATIC verdict=$sv"
if [[ $sv != STATIC_PASS ]]; then stage=done; state "COMPLETE static=$sv"; exit 0; fi

stage=dynamic; state "RUNNING $stage"
guard dynamic 28800 "${WL109B_TIMING_MAXLOAD:-2}" --before "if [ -d '$out/dyn' ]; then mv '$out/dyn' '$out/dyn_preempted_'\$(date +%s); fi" -- \
  dynamic109b.py "$out/dyn" "$out/metrics/static109b.json" "$out/calib/budget109b.json" > "$out/dyn_stage.log" 2>&1
dv=$(jget "$out/dyn/dyn109b.json" verdict)
stage=done; state "COMPLETE verdict=$dv"
