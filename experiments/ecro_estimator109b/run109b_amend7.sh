#!/usr/bin/env bash
# WL109B Amendment 7 chain on LabServer70.  usage: run109b_amend7.sh COMMIT
# Amendment 7 (owner decision 2026-11-26): every guarded stage, including the timing-bearing stages (timing, dynamic), counts
# only compute processes on this host's RTX 3090 (guard109c.py --gpu-uuid) as foreign, for the start condition and for
# preemption. Work on the RTX 3080 Ti no longer blocks or preempts WL109B. The load gate of the timing-bearing stages
# (1-minute load average <= 2 at admission) is unchanged. Resumes after the completed bias stage (bias109b.json =
# EQUIVALENT_ALL, recomputed on 2026-11-26 by the Amendment 5 chain): grid (complete units are skipped) -> timing -> static
# -> dynamic. Everything else is identical to run109b_amend5.sh (Amendments 1 + 3 + 5 + 6).
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
guard() { local name=$1 cap=$2 ml=$3; shift 3; local pre=(); if [[ $1 == --before ]]; then pre=(--before "$2"); shift 2; fi; shift
  "$py" "$guardc" --name "$name" --log "$out/guard109b_lab70.jsonl" --free 120 --poll 2 --attempt-timeout "$cap" \
    --container "wl109b_$name" --gpu-uuid "$GPU3090" --max-load "$ml" ${pre[@]+"${pre[@]}"} -- "$ctr" "wl109b_$name" "$py" "$@"; }
cpu() { CUDA_VISIBLE_DEVICES= nice -n 10 taskset -c 12-15 "$py" "$@"; }
stage=start
state() { printf '%s %s\n' "$(date -Iseconds)" "$*" | tee -a "$out/chain.state"; }
trap 'rc=$?; if (( rc )); then state "STOP stage=$stage exit=$rc (LabServer70)"; fi' EXIT
state "AMENDMENT 7 (3090-only guard) START on $(hostname) (LabServer70) commit=$commit driver=$(cat /proc/driver/nvidia/version | head -1 | grep -o '550[.0-9]*') image=${IMAGE:7:12}"
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
    sleep 10; done ) &

stage=grid; state "RUNNING $stage"
guard grid 10800 0 -- capture109b.py "$out/cap" grid > "$out/grid_stage.log" 2>&1
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
