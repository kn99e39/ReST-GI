#!/usr/bin/env bash
# WL109B Amendment 1 + 3 + 4 GPU chain on LabServer70.  usage: run109b_amend4_gpu.sh COMMIT PHASE
# Amendment 4 (owner decision 2026-11-24): LabServer70's CPU is held by other work, so every CPU stage (tests, bias and static
# analyses, decisions) runs on Main WSL (run109b_main.sh) and this chain runs only the GPU stages, in three phases:
#   gpu1: xcheck + paired             gpu2: grid + timing (needs metrics/bias109b.json EQUIVALENT_ALL from Main)
#   gpu3: dynamic (needs metrics/static109b.json STATIC_PASS from Main)
# Amendment 3 (owner decisions 2026-11-24): driver 525.60.13 -> 550.54.15 (same as LabServer69, done by the owner without a
# reboot); every GPU command runs inside a Docker container (ctr109b.sh, glibc 2.36 userland, frozen files bind-mounted) under
# the container-aware guard guard109c.py; the frozen WL108 admission is used unchanged (Amendment 2 is not used).
# Load gate: GPU data stages start only while the host 1-minute load average <= 8 (leave CPU to other users); timing stages
# (timing, dynamic) only while <= 2 (no CPU contention in the measured frame times).
# Resumes the frozen WL109B protocol at Stage D. Stages A-C (Stage A reproduction, sealed qualification, budget R = 512) and
# 16 of 18 paired units completed on LabServer69 and are reused only if the cross-host check passes; otherwise every paired
# unit is recaptured here. GPU commands run in a container under guard109c.py, pinned to this host's RTX 3090; the guard
# watches every GPU and tmux session on the host. CPU stages run on the host with CUDA_VISIBLE_DEVICES=.
set -eu -o pipefail
cd "$HOME/rest_sss101a"
commit=$1; phase=$2
src=experiments/ecro_estimator109b
out="$PWD/results/ecro_estimator109b/evidence"
py="$HOME/NeuralShaderDemo/external/8dna26/.venv/bin/python"
guardpy="$PWD/experiments/ecro_estimator109/guard109.py"
GPU3090=GPU-ab3bc073-096c-6d0b-a304-13cef177a843
mkdir -p "$out/tmp" "$out/mpl"
export DYN101_MI_DIR="$HOME/rest_dyn101/external/mi_3.8.0" DYN101_MI_VERSION=3.8.0
export LD_LIBRARY_PATH="$PWD/results/ecro_denoise108/runtime/lib:${LD_LIBRARY_PATH:-}"
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export MPLCONFIGDIR="$out/mpl" TMPDIR="$out/tmp" PYTHONDONTWRITEBYTECODE=1
IMAGE=$(docker image inspect python:3.10-bookworm --format '{{.Id}}')
[[ $IMAGE == sha256:6fd6ebbd05fc4cf5dc3a789c47e42f4d787cb9c90fe5d29835875a934680271d ]]
ctr="$PWD/experiments/ecro_estimator109b/ctr109b.sh"
guardc="$PWD/experiments/ecro_estimator109b/guard109c.py"
# guard NAME CAP MAXLOAD [--before ...] -- python-args...  (python runs inside container wl109b_NAME)
guard() { local name=$1 cap=$2 ml=$3; shift 3; local pre=(); if [[ $1 == --before ]]; then pre=(--before "$2"); shift 2; fi; shift
  "$py" "$guardc" --name "$name" --log "$out/guard109b_lab70.jsonl" --free 120 --poll 2 --attempt-timeout "$cap" \
    --container "wl109b_$name" --max-load "$ml" ${pre[@]+"${pre[@]}"} -- "$ctr" "wl109b_$name" "$py" "$@"; }
cpu() { CUDA_VISIBLE_DEVICES= nice -n 10 "$py" "$@"; }
stage=start
state() { printf '%s %s\n' "$(date -Iseconds)" "$*" | tee -a "$out/chain.state"; }
trap 'rc=$?; if (( rc )); then state "STOP stage=$stage exit=$rc (LabServer70)"; fi' EXIT
state "AMENDMENT 1+3+4 $phase START on $(hostname) (LabServer70) commit=$commit driver=$(cat /proc/driver/nvidia/version | head -1 | grep -o '550[.0-9]*') image=${IMAGE:7:12}"
cd "$src"
jget() { "$py" -c 'import json,sys; d=json.load(open(sys.argv[1]));
for k in sys.argv[2].split("."): d=d[k]
print(d)' "$1" "$2"; }
R=$(jget "$out/calib/budget109b.json" decision.R)
[[ $(jget "$out/calib/budget109b.json" decision.outcome) == PROCEED && $R == 512 ]]

R=$(jget "$out/calib/budget109b.json" decision.R)
[[ $(jget "$out/calib/budget109b.json" decision.outcome) == PROCEED && $R == 512 ]]

if [[ $phase == gpu1 ]]; then
  stage=xcheck; state "RUNNING $stage"
  if [[ ! -f $out/xcheck/xcheck109b.json ]]; then
    mkdir -p "$out/xcheck"
    guard xcheck 7200 8 -- xcheck109b.py "$out/cap" "$out/xcheck/xcheck109b.json" > "$out/xcheck/xcheck.log" 2>&1
  fi
  xo=$(jget "$out/xcheck/xcheck109b.json" outcome)
  state "XCHECK outcome=$xo"
  if [[ $xo != PASS && -d $out/cap/paired ]]; then
    mv "$out/cap/paired" "$out/cap/paired_lab69_not_reproduced"
    state "paired units from LabServer69 moved aside; every paired unit is recaptured on LabServer70"
  fi
  stage=paired; state "RUNNING $stage R=$R (LabServer70)"
  guard paired 28800 8 -- capture109b.py "$out/cap" paired --R "$R" > "$out/paired_stage_lab70.log" 2>&1
  stage=done; state "GPU1 COMPLETE (paired ready; bias analysis on Main)"
elif [[ $phase == gpu2 ]]; then
  [[ $(jget "$out/metrics/bias109b.json" bias_verdict) == EQUIVALENT_ALL ]]
  stage=grid; state "RUNNING $stage"
  guard grid 10800 8 -- capture109b.py "$out/cap" grid > "$out/grid_stage.log" 2>&1
  stage=timing; state "RUNNING $stage"
  guard timing 10800 2 -- capture109b.py "$out/cap" timing --R "$R" > "$out/timing_stage.log" 2>&1
  stage=done; state "GPU2 COMPLETE (grid / timing ready; static analysis on Main)"
elif [[ $phase == gpu3 ]]; then
  [[ $(jget "$out/metrics/static109b.json" final_static_verdict) == STATIC_PASS ]]
  stage=dynamic; state "RUNNING $stage"
  guard dynamic 28800 2 --before "if [ -d '$out/dyn' ]; then mv '$out/dyn' '$out/dyn_preempted_'\$(date +%s); fi" -- \
    dynamic109b.py "$out/dyn" "$out/metrics/static109b.json" "$out/calib/budget109b.json" > "$out/dyn_stage.log" 2>&1
  dv=$(jget "$out/dyn/dyn109b.json" verdict)
  stage=done; state "COMPLETE verdict=$dv (dynamic, LabServer70)"
else
  echo "unknown phase $phase" >&2; exit 2
fi
